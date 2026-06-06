import asyncio
import io
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from prometheus_client import make_asgi_app
from pydantic import BaseModel

from cache import check_rate_limit
from chat import run_chat_turn
from llm import generate, is_fashion_query, reformulate_query
from observability import (
    clear_context,
    configure_logging,
    guardrail_rejections,
    log,
    new_request_id,
    rate_limited,
    search_requests,
    timed,
)
from search import client as qdrant_client
from search import multi_query_search, search, search_by_image

configure_logging()

RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
RATE_LIMITED_PATHS = {"/search", "/search/image", "/chat"}


_filters_cache: dict = {"colors": [], "genders": [], "categories": []}


def _load_filters_cache():
    points, _ = qdrant_client.scroll(collection_name="fashion", limit=10000, with_payload=True)
    colors, genders, categories = set(), set(), set()
    for p in points:
        if p.payload.get("color"):
            colors.add(p.payload["color"])
        if p.payload.get("gender"):
            genders.add(p.payload["gender"])
        if p.payload.get("category"):
            categories.add(p.payload["category"])
    _filters_cache["colors"] = sorted(colors)
    _filters_cache["genders"] = sorted(genders)
    _filters_cache["categories"] = sorted(categories)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_filters_cache()
    yield


app = FastAPI(title="Multimodal Fashion RAG", lifespan=lifespan)
if os.path.isdir("images"):
    app.mount("/images", StaticFiles(directory="images"), name="images")
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    rid = new_request_id()
    t0 = time.perf_counter()
    status = "ok"

    # Rate limit для тяжёлых LLM-эндпоинтов (fail-open при отсутствии Redis)
    if request.url.path in RATE_LIMITED_PATHS:
        client_id = request.client.host if request.client else "unknown"
        allowed, retry_after = check_rate_limit(
            namespace="search", identifier=client_id,
            limit=RATE_LIMIT_PER_MINUTE, window_seconds=60,
        )
        if not allowed:
            rate_limited.labels(endpoint=request.url.path).inc()
            log.info("rate_limited", path=request.url.path, client=client_id, retry_after=retry_after)
            dur_ms = round((time.perf_counter() - t0) * 1000, 1)
            search_requests.labels(endpoint=request.url.path, status="rate_limited").inc()
            log.info("request", path=request.url.path, method=request.method, duration_ms=dur_ms, status="rate_limited")
            clear_context()
            return JSONResponse(
                status_code=429,
                content={"detail": f"Слишком много запросов. Попробуй через {retry_after} сек."},
                headers={"X-Request-ID": rid, "Retry-After": str(retry_after)},
            )

    try:
        response = await call_next(request)
        if response.status_code >= 500:
            status = "error"
        response.headers["X-Request-ID"] = rid
        return response
    except Exception:
        status = "error"
        raise
    finally:
        dur_ms = round((time.perf_counter() - t0) * 1000, 1)
        log.info(
            "request",
            path=request.url.path,
            method=request.method,
            duration_ms=dur_ms,
            status=status,
        )
        if request.url.path in ("/search", "/search/image"):
            search_requests.labels(endpoint=request.url.path, status=status).inc()
        clear_context()


@app.get("/")
def index():
    return FileResponse("static/index.html")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    color: str | None = None
    gender: str | None = None
    category: str | None = None
    with_llm: bool = True
    rerank: bool = True
    reformulate: bool = True
    guardrail: bool = True


class ProductResult(BaseModel):
    score: float
    title: str | None
    category: str | None
    gender: str | None
    color: str | None
    image_url: str | None


class Recommendation(BaseModel):
    pick: str | None = None
    reason: str | None = None
    alternatives: str | None = None
    suggestions: list[str] = []


class SearchResponse(BaseModel):
    query: str
    subqueries: list[str]
    recommendation: Recommendation | None
    products: list[ProductResult]


def _clean(v: str | None) -> str | None:
    if not v or v.strip().lower() == "string":
        return None
    return v


def _run(func, *args, **kwargs):
    return asyncio.to_thread(func, *args, **kwargs)


async def _resolve_queries(query: str, use_reformulation: bool) -> list[str]:
    if not use_reformulation:
        return [query]
    return await _run(reformulate_query, query, 3)


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(req: SearchRequest):
    with timed("total"):
        if req.guardrail:
            ok, reason = await _run(is_fashion_query, req.query)
            if not ok:
                guardrail_rejections.labels(reason="off_topic").inc()
                log.info("guardrail_rejected", query=req.query, reason=reason)
                return SearchResponse(
                    query=req.query,
                    subqueries=[],
                    recommendation=Recommendation(
                        pick=None,
                        reason=f"Я помогаю только с поиском одежды и аксессуаров. {reason}".strip(),
                        alternatives=None,
                        suggestions=[
                            "Найди мне синюю рубашку",
                            "Что подарить на день рождения подруге?",
                            "Кроссовки Nike для бега",
                        ],
                    ),
                    products=[],
                )

        filters = {
            "color": _clean(req.color),
            "gender": _clean(req.gender),
            "category": _clean(req.category),
        }

        subqueries = await _resolve_queries(req.query, req.reformulate)
        if len(subqueries) > 1:
            products = await _run(multi_query_search, subqueries, req.top_k, filters, req.rerank, True)
        else:
            products = await _run(search, req.query, req.top_k, filters, req.rerank, True)

        recommendation = await _maybe_generate(req.query, products, req.with_llm)
        return SearchResponse(
            query=req.query,
            subqueries=subqueries,
            recommendation=recommendation,
            products=_build_results(products),
        )


async def _maybe_generate(query: str, products: list[dict], enabled: bool) -> Recommendation | None:
    if not enabled:
        return None
    try:
        data = await _run(generate, query, products)
        return Recommendation(**data)
    except Exception as e:
        return Recommendation(reason=f"Ошибка LLM: {e}")


def _build_results(products: list[dict]) -> list[ProductResult]:
    results = []
    for p in products:
        image_url = None
        if p.get("image_path"):
            filename = p["image_path"].replace("./images/", "")
            image_url = f"http://localhost:8000/images/{filename}"
        results.append(ProductResult(
            score=p["score"],
            title=p.get("title"),
            category=p.get("category"),
            gender=p.get("gender"),
            color=p.get("color"),
            image_url=image_url,
        ))
    return results


@app.post("/search/image", response_model=SearchResponse)
async def search_image_endpoint(
    top_k: int = 5,
    color: str | None = None,
    gender: str | None = None,
    category: str | None = None,
    with_llm: bool = True,
    file: UploadFile = File(...),
):
    with timed("total"):
        image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
        filters = {"color": _clean(color), "gender": _clean(gender), "category": _clean(category)}
        products = await _run(search_by_image, image, top_k, filters)
        recommendation = await _maybe_generate("похожие товары по загруженному изображению", products, with_llm)
        return SearchResponse(
            query=f"image: {file.filename}",
            subqueries=[],
            recommendation=recommendation,
            products=_build_results(products),
        )


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    intent: str
    response: Recommendation
    products: list[ProductResult]


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    with timed("total"):
        result = await _run(run_chat_turn, req.session_id, req.message)
        return ChatResponse(
            session_id=result["session_id"],
            intent=result["intent"],
            response=Recommendation(**(result["response"] or {})),
            products=_build_results(result["products"]),
        )


@app.get("/filters")
def get_filters():
    return _filters_cache


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
