"""Conversational режим на LangGraph: multi-turn диалог с persisted сессией в Redis.

Граф (3 ноды + условный routing):

    [analyze]  →  intent="search"     →  [search_and_respond]  →  END
                  intent="off_topic"  →  [decline]             →  END

State хранится в Redis под ключом chat:session:<id> с TTL CHAT_SESSION_TTL_SECONDS.
Если Redis недоступен — каждый turn идёт как новая сессия (fail-open).
"""
import json
import uuid
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from cache import cache_key, get_json, set_json
from config import settings
from llm import _client, _model, generate
from observability import log, timed
from search import multi_query_search

MAX_HISTORY_MESSAGES = 10


class ChatState(TypedDict, total=False):
    session_id: str
    messages: list[dict]
    intent: str
    search_query: str
    filters: dict
    products: list[dict]
    response: dict


_ANALYZE_SYS = """Ты ассистент fashion-магазина. Получаешь историю диалога с покупателем.
Твоя задача — посмотреть на ПОСЛЕДНЕЕ сообщение в контексте предыдущих и вернуть JSON.

Определи intent:
- "search" — покупатель ищет товар (новый запрос ИЛИ уточнение к предыдущему поиску)
- "off_topic" — запрос не про одежду/обувь/аксессуары (погода, рецепты, политика и т.п.)

Если intent="search", построй запрос на АНГЛИЙСКОМ (каталог англоязычный), учитывая контекст:
- если это уточнение ("а поярче?", "теперь синие") — собери полный запрос из истории
- если новый поиск — просто переведи

Также извлеки фильтры если упоминаются: color (на английском: blue/red/black/...), gender (Men/Women/Unisex), category.

JSON формат строго:
{"intent": "search" | "off_topic", "query": "english query", "color": null или "Blue", "gender": null или "Men", "category": null или "Shirts"}
"""


def _build_history_text(messages: list[dict]) -> str:
    """Превращает messages в формат для LLM-промпта."""
    lines = []
    for m in messages:
        role = "Покупатель" if m["role"] == "user" else "Ассистент"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def analyze_node(state: ChatState) -> dict:
    """LLM анализирует историю + последнее сообщение, возвращает intent + query + filters."""
    history_text = _build_history_text(state["messages"])
    with timed("chat_analyze"):
        try:
            resp = _client().chat.completions.create(
                model=_model(),
                messages=[
                    {"role": "system", "content": _ANALYZE_SYS},
                    {"role": "user", "content": history_text},
                ],
                response_format={"type": "json_object"},
                max_tokens=200,
                temperature=0.1,
            )
            data = json.loads(resp.choices[0].message.content)
        except Exception as e:
            log.warning("chat_analyze_failed", error=str(e))
            data = {"intent": "search", "query": state["messages"][-1]["content"]}

    intent = data.get("intent", "search")
    query = data.get("query", "") or state["messages"][-1]["content"]
    filters = {
        "color": data.get("color"),
        "gender": data.get("gender"),
        "category": data.get("category"),
    }
    return {"intent": intent, "search_query": query, "filters": filters}


def search_and_respond_node(state: ChatState) -> dict:
    """Запускает hybrid search + LLM рекомендацию."""
    with timed("chat_search"):
        products = multi_query_search(
            queries=[state["search_query"]],
            top_k=5,
            filters=state.get("filters") or {},
            rerank=True,
            hybrid=True,
        )
    response = generate(state["search_query"], products)
    return {"products": products, "response": response}


def decline_node(state: ChatState) -> dict:
    """Вежливый отказ для off-topic."""
    return {
        "products": [],
        "response": {
            "pick": None,
            "reason": "Я помогаю только с поиском одежды и аксессуаров — расскажи, что тебе нужно из fashion?",
            "alternatives": None,
            "suggestions": [
                "Найди мне синюю рубашку",
                "Что подойдёт к чёрному пальто?",
                "Кроссовки для бега",
            ],
        },
    }


def _route_after_analyze(state: ChatState) -> str:
    return "search_and_respond" if state.get("intent") == "search" else "decline"


def _build_graph():
    g = StateGraph(ChatState)
    g.add_node("analyze", analyze_node)
    g.add_node("search_and_respond", search_and_respond_node)
    g.add_node("decline", decline_node)
    g.add_edge(START, "analyze")
    g.add_conditional_edges("analyze", _route_after_analyze, {
        "search_and_respond": "search_and_respond",
        "decline": "decline",
    })
    g.add_edge("search_and_respond", END)
    g.add_edge("decline", END)
    return g.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph()
    return _compiled_graph


# ---------- Session persistence ----------

def _session_key(session_id: str) -> str:
    return cache_key("chat_session", session_id)


def load_session(session_id: str) -> list[dict]:
    """Загружает messages из Redis. Пустой список если сессия новая или Redis недоступен."""
    raw = get_json("chat_session", _session_key(session_id))
    if raw is None:
        return []
    return list(raw.get("messages", []))[-MAX_HISTORY_MESSAGES:]


def save_session(session_id: str, messages: list[dict]) -> None:
    """Сохраняет messages в Redis с TTL."""
    set_json("chat_session", _session_key(session_id),
             {"messages": messages[-MAX_HISTORY_MESSAGES:]},
             ttl_seconds=settings.chat_session_ttl_seconds)


def run_chat_turn(session_id: str | None, user_message: str) -> dict:
    """Главная точка входа. Возвращает {session_id, intent, response, products}."""
    sid = session_id or uuid.uuid4().hex[:12]
    history = load_session(sid)
    history.append({"role": "user", "content": user_message})

    state: ChatState = {"session_id": sid, "messages": history}
    result = get_graph().invoke(state)

    assistant_summary = result.get("response", {}).get("reason") or "..."
    history.append({"role": "assistant", "content": assistant_summary})
    save_session(sid, history)

    return {
        "session_id": sid,
        "intent": result.get("intent", "search"),
        "response": result.get("response"),
        "products": result.get("products", []),
    }


def reset_graph_for_tests() -> None:
    global _compiled_graph
    _compiled_graph = None
