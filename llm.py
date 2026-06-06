import json
import os
import time

from dotenv import load_dotenv

from cache import cache_key, get_json, set_json
from observability import llm_errors, log, timed

load_dotenv()

LLM_BACKEND = os.getenv("LLM_BACKEND", "api")


def _client():
    from openai import OpenAI
    return OpenAI(
        base_url=os.getenv("LLM_API_BASE_URL", "https://api.groq.com/openai/v1"),
        api_key=os.getenv("LLM_API_KEY", ""),
    )


def _model() -> str:
    return os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


# ---------- Guardrail ----------

_GUARDRAIL_SYS = """Ты классификатор запросов для fashion-магазина.
Определи, относится ли запрос покупателя к поиску одежды, обуви, аксессуаров, модных брендов, стилей или подарков из мира fashion.
К fashion относятся: одежда, обувь, сумки, украшения, часы, парфюм, бренды (Nike, Adidas, Zara и т.п.), описания («подарок жене», «что-то для зала», «к костюму»).
НЕ относятся: погода, политика, новости, код, математика, оскорбления, другие товары (еда, техника, мебель).
Верни строго JSON: {"is_fashion": true/false, "reason": "короткое пояснение по-русски"}"""


def is_fashion_query(query: str) -> tuple[bool, str]:
    """True если запрос про fashion. Результат кешируется в Redis на 24ч."""
    q = query.strip()
    if not q:
        return False, "Пустой запрос"
    key = cache_key("guardrail", q)
    cached = get_json("guardrail", key)
    if cached is not None:
        return bool(cached[0]), str(cached[1])
    with timed("llm_guardrail"):
        try:
            resp = _client().chat.completions.create(
                model=_model(),
                messages=[
                    {"role": "system", "content": _GUARDRAIL_SYS},
                    {"role": "user", "content": q},
                ],
                response_format={"type": "json_object"},
                max_tokens=80,
                temperature=0.0,
            )
            data = json.loads(resp.choices[0].message.content)
            result = (bool(data.get("is_fashion", True)), str(data.get("reason", "")))
            set_json("guardrail", key, list(result), ttl_seconds=86400)
            return result
        except Exception as e:
            llm_errors.labels(type="guardrail").inc()
            log.warning("guardrail_failed", error=str(e))
            # При ошибке guardrail разрешаем — fail open, чтобы не ломать UX
            return True, "guardrail недоступен, пропускаем"


# ---------- Query reformulation ----------

_REFORMULATE_SYS = """Ты помогаешь искать товары в fashion-каталоге.
Получив запрос пользователя, сгенерируй {n} разные короткие поисковые фразы НА АНГЛИЙСКОМ (каталог на английском).
Учитывай синонимы, связанные категории, стили, бренды.
Отвечай строго JSON: {{"queries": ["фраза1", "фраза2", "фраза3"]}}"""


def reformulate_query(query: str, n: int = 3) -> list[str]:
    if not query.strip():
        return [query]
    key = cache_key("reformulate", query, n)
    cached = get_json("reformulate", key)
    if cached is not None:
        return list(cached)
    with timed("llm_reformulate"):
        try:
            resp = _client().chat.completions.create(
                model=_model(),
                messages=[
                    {"role": "system", "content": _REFORMULATE_SYS.format(n=n)},
                    {"role": "user", "content": query},
                ],
                response_format={"type": "json_object"},
                max_tokens=200,
                temperature=0.3,
            )
            data = json.loads(resp.choices[0].message.content)
            queries = [q.strip() for q in data.get("queries", []) if q and q.strip()]
            result = [query] + queries[:n]
            set_json("reformulate", key, result, ttl_seconds=86400)
            return result
        except Exception as e:
            llm_errors.labels(type="reformulate").inc()
            log.warning("llm_reformulate_failed", error=str(e))
            return [query]


# ---------- Structured recommendation ----------

_RECOMMEND_SYS = """Ты консультант fashion-магазина. Отвечай по-русски, дружелюбно, конкретно.
На вход получаешь запрос пользователя и список найденных товаров.
Верни СТРОГО JSON вида:
{
  "pick": "название товара №1 из списка как есть",
  "reason": "1-2 коротких предложения почему этот вариант лучший",
  "alternatives": "1 предложение — упомяни что в списке есть и другие варианты",
  "suggestions": ["уточняющий вопрос 1", "уточняющий вопрос 2", "уточняющий вопрос 3"]
}
Suggestions — короткие вопросы-уточнения, которые покупатель может задать дальше (бренд, цвет, размер, стиль)."""


_EMPTY_RECOMMENDATION = {
    "pick": None,
    "reason": "По запросу ничего не нашлось. Попробуй убрать фильтры или переформулировать.",
    "alternatives": None,
    "suggestions": [
        "Убрать фильтры?",
        "Расширить категорию?",
        "Искать по картинке?",
    ],
}


def _format_products(query: str, products: list[dict]) -> str:
    lines = [f"Запрос покупателя: {query}", "", "Найденные товары:"]
    for i, p in enumerate(products, 1):
        lines.append(f"{i}. {p['title']} ({p['category']}, {p['color']}, {p['gender']})")
    return "\n".join(lines)


def _generate_api(query: str, products: list[dict], retries: int = 2) -> dict:
    from openai import RateLimitError
    client = _client()
    with timed("llm_generate"):
        for attempt in range(retries):
            try:
                resp = client.chat.completions.create(
                    model=_model(),
                    messages=[
                        {"role": "system", "content": _RECOMMEND_SYS},
                        {"role": "user", "content": _format_products(query, products)},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=800,
                    temperature=0.5,
                )
                content = resp.choices[0].message.content or "{}"
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    # Verbose модели иногда обрезаются на max_tokens посреди JSON
                    log.warning("llm_generate_json_truncated", content_tail=content[-100:])
                    data = {"pick": products[0]["title"] if products else None,
                            "reason": "LLM-ответ был обрезан, показываем top-1 как есть",
                            "alternatives": None, "suggestions": []}
                return {
                    "pick": data.get("pick"),
                    "reason": data.get("reason"),
                    "alternatives": data.get("alternatives"),
                    "suggestions": list(data.get("suggestions", []))[:3],
                }
            except RateLimitError:
                llm_errors.labels(type="rate_limit").inc()
                if attempt < retries - 1:
                    time.sleep(3)
                else:
                    log.error("llm_generate_rate_limited")
                    raise
            except Exception as e:
                llm_errors.labels(type="generate").inc()
                log.error("llm_generate_failed", error=str(e))
                raise


def _generate_local(query: str, products: list[dict]) -> dict:
    from llama_cpp import Llama
    llm = Llama(
        model_path=os.getenv("LOCAL_MODEL_PATH", ""),
        n_ctx=2048,
        n_threads=os.cpu_count(),
        n_gpu_layers=0,
        verbose=False,
        use_mmap=True,
    )
    prompt = (
        "<|im_start|>system\n" + _RECOMMEND_SYS + "<|im_end|>\n"
        "<|im_start|>user\n" + _format_products(query, products) + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    out = llm(prompt, max_tokens=500, temperature=0.5, stop=["<|im_end|>", "</s>"])
    text = out["choices"][0]["text"].strip()
    try:
        return json.loads(text)
    except Exception:
        return {"pick": None, "reason": text, "alternatives": None, "suggestions": []}


def generate(query: str, products: list[dict]) -> dict:
    if not products:
        return _EMPTY_RECOMMENDATION
    if LLM_BACKEND == "local":
        return _generate_local(query, products)
    return _generate_api(query, products)
