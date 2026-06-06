# Multimodal Fashion RAG

Мультимодальный поиск по fashion-каталогу: русские запросы → англоязычные товары, поиск по тексту и по картинке, LLM-консультант с follow-up подсказками. Observability через Prometheus + Grafana, тесты и CI.

[![CI](https://github.com/EthernalSolitude/multimodal-fashion-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/EthernalSolitude/multimodal-fashion-rag/actions/workflows/ci.yml) ![Coverage](https://img.shields.io/badge/coverage-82%25-brightgreen) ![Python](https://img.shields.io/badge/python-3.12-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-async-009688) ![Qdrant](https://img.shields.io/badge/Qdrant-hybrid-red) ![Redis](https://img.shields.io/badge/Redis-cache%20%2B%20ratelimit-DC382D) ![LangGraph](https://img.shields.io/badge/LangGraph-chat-FF6B35) ![Docker](https://img.shields.io/badge/docker--compose-ready-2496ED)

---

## Ключевые особенности

- **Гибридный поиск** (dense CLIP + sparse BM42 + cross-encoder rerank) — **P@5 = 0.934 [0.904, 0.960]** на 100-query benchmark с bootstrap 95% CI, **+43% над dense-only** baseline (статистически значимо, non-overlapping CI)
- **LLM переформулирует запрос** в несколько английских фраз, поиск идёт по каждой с дедупликацией, потом cross-encoder ранжирует общий список — помогает на нечётких запросах типа «что-нибудь для зала»
- **Guardrail** — LLM проверяет что запрос про одежду, отказывает на off-topic («расскажи про погоду»), кэширует результат
- **Observability** — структурированные JSON-логи со сквозным `request_id`, Prometheus-метрики по каждой стадии пайплайна, готовый Grafana-дашборд
- **Redis shared cache** для повторяющихся LLM-запросов (guardrail и переформулировка), fail-open паттерн — сервис работает и без Redis
- **Rate limiting** через Redis (fixed-window, по IP) для тяжёлых LLM-эндпоинтов; защита от злоупотреблений с возвратом 429 + `Retry-After`
- **Conversational mode на LangGraph** — мультитёрный диалог с persisted-сессией в Redis, граф из 3 нод с условным роутингом (`analyze → search_and_respond / decline`)
- **90 автоматических тестов** (~10 сек, без GPU и БД, **coverage 82%**), **CI/CD на GitHub Actions** — линтер, тесты и автоматическая публикация Docker-образа в [GitHub Container Registry](https://github.com/EthernalSolitude/multimodal-fashion-rag/pkgs/container/multimodal-fashion-rag) на каждый push в main
- **Один `docker compose up --build`** поднимает всё: API, Qdrant, Redis, Prometheus, Grafana — или `docker pull ghcr.io/ethernalsolitude/multimodal-fashion-rag:latest`

---

## Демо

```
Запрос:  "синяя рубашка для мужчин"
   │
   ├─ LLM reformulate → ["blue shirt men", "navy mens top", "men casual blue shirts"]
   │
   ├─ Поиск по каждому из 3 подзапросов + оригиналу, дедуп по id:
   │    ├─ Dense (CLIP-multilingual) → top-20
   │    └─ Sparse (BM42) → top-20
   │       ↓ Reciprocal Rank Fusion
   │    Union кандидатов ≈ 50 товаров
   │
   ├─ Cross-encoder rerank (mmarco-mMiniLMv2 multilingual) → top-5
   │
   └─ LLM recommendation: pick + reason + 3 follow-up suggestions
```

---

## Как работает поиск (по шагам)

1. **Guardrail** — LLM проверяет: запрос вообще про одежду? Если нет («расскажи про погоду») — отказ с вежливым объяснением, поиск не запускается. Результат кэшируется, чтобы не дёргать API на повторах.
2. **Reformulate** — LLM превращает русский запрос в 3 короткие английские фразы-синонимы (каталог на английском). Пример: «синяя рубашка для мужчин» → `["blue shirt men", "navy mens top", "men casual blue shirts"]`.
3. **Hybrid search** — по каждой фразе последовательно ищем в Qdrant двумя способами и объединяем результаты по id:
   - **Dense** — CLIP-эмбеддинг, ищет по семантическому смыслу
   - **Sparse (BM42)** — ищет по редким токенам и точным совпадениям (бренды, артикулы)
   - **RRF (Reciprocal Rank Fusion)** — Qdrant сам объединяет два рейтинга в один. Это делается нативно через `FusionQuery`, не руками.
4. **Rerank** — собираем ~50 кандидатов со всех подзапросов, cross-encoder (мультиязычный) пересчитывает релевантность точнее чем векторный поиск, оставляем топ-5.
5. **LLM recommendation** — модель получает топ-5 и возвращает JSON: лучший товар, почему он подходит, альтернативы, 3 follow-up вопроса для уточнения.

```mermaid
flowchart LR
    U[Запрос] --> API[FastAPI]
    API --> G[Guardrail<br/>fashion или нет?]
    G -->|fashion| R[Reformulate<br/>RU → 3× EN]
    R --> H[Hybrid Search]
    H --> D[Dense<br/>CLIP]
    H --> S[Sparse<br/>BM42]
    D --> Q[(Qdrant<br/>RRF)]
    S --> Q
    Q -->|top-20| X[Cross-encoder<br/>rerank]
    X -->|top-5| L[LLM<br/>рекомендация]
    L --> API
    API -.метрики.-> P[Prometheus]
    P --> GR[Grafana]
    API -.логи.-> ST[structlog]
```

**Про поиск по картинке:** текстовая CLIP-модель и image-CLIP работают в общем векторном пространстве, поэтому картинку и текст можно искать в одном индексе без отдельной базы для изображений.

---

## Conversational режим (LangGraph)

Эндпоинт `POST /chat` принимает сообщения с привязкой к `session_id` и хранит историю диалога — можно вести многоходовой разговор, уточняя предыдущий запрос. Внутри простой граф из трёх узлов: `analyze` смотрит на всю историю и через LLM решает что хочет пользователь, в зависимости от intent граф идёт либо в `search_and_respond` (ищет по каталогу и отвечает), либо в `decline` (вежливый отказ на off-topic).

```mermaid
graph TD;
    __start__([__start__]):::first
    analyze(analyze)
    search_and_respond(search_and_respond)
    decline(decline)
    __end__([__end__]):::last
    __start__ --> analyze;
    analyze -.-> decline;
    analyze -.-> search_and_respond;
    decline --> __end__;
    search_and_respond --> __end__;
    classDef first fill-opacity:0
    classDef last fill:#bfb6fc
```

Эта диаграмма рисуется автоматически: `compiled.get_graph().draw_mermaid()` отдаёт mermaid-источник, эндпоинт `GET /chat/graph` пробрасывает его наружу, веб-UI рендерит в браузере. Поэтому картинка всегда в синке с кодом — поддерживать руками ничего не нужно.

История живёт в Redis под ключом `chat:session:{id}` с TTL 1 час. На каждый ход грузим последние 10 сообщений, прогоняем через граф, сохраняем обратно. Без Redis сервис тоже работает — просто без памяти между ходами (fail-open).

Зачем тут вообще LangGraph, а не пара `if`'ов в функции — потому что это удобная база под более сложные сценарии. Например, добавить self-correction (нода оценивает релевантность результатов и при низкой возвращается на reformulate) — это новый узел и одно условное ребро, без переписывания основного pipeline'а. Сейчас граф минимальный, но структура уже разводит orchestration и бизнес-логику.

---

## Метрики

Eval-харнесс с **bootstrap 95% CI** (1000 ресэмплов) на фиксированном test-set из **100 категориальных запросов** — `{Color} {Category} for {Gender}`. Ground truth: товары, совпадающие по всем трём фасетам. Test-set заморожен в `eval_data/test_queries.json` для воспроизводимости между прогонами.

| Конфиг                       | P@5                       | R@10                      | NDCG@10                   | MRR@10                    |
|------------------------------|---------------------------|---------------------------|---------------------------|---------------------------|
| BM25 sparse only             | 0.894 [0.856, 0.926]      | 0.935 [0.907, 0.960]      | 0.920 [0.891, 0.945]      | 0.940 [0.910, 0.970]      |
| Dense CLIP only              | 0.654 [0.592, 0.720]      | 0.646 [0.590, 0.703]      | 0.661 [0.606, 0.716]      | 0.813 [0.749, 0.870]      |
| Hybrid RRF                   | 0.806 [0.764, 0.844]      | 0.862 [0.826, 0.895]      | 0.848 [0.813, 0.880]      | 0.946 [0.913, 0.977]      |
| **Hybrid + cross-encoder rerank** | **0.934 [0.904, 0.960]** | **0.961 [0.938, 0.979]** | **0.955 [0.933, 0.973]** | **0.978 [0.953, 0.995]** |

**Что означают цифры:**

- **Полный пайплайн лучший на всех 4 метриках**: P@5 = 0.934 [0.904, 0.960]. Non-overlapping CI с dense-only baseline (0.654 [0.592, 0.720]) подтверждает статистическую значимость улучшения (+43% P@5).
- **Cross-encoder rerank — главный лифт качества**: +12pp P@5 над голым hybrid'ом, +8pp NDCG@10.
- **Любопытный нюанс**: на прямых categorical-запросах **BM25 alone сильнее голого hybrid'а** (P@5 0.894 vs 0.806) — sparse идеально матчит keywords, dense на таких прямых запросах добавляет шума. Это инсайт, который скрыт в single-metric eval'е.
- Полный отчёт: [`eval_data/results.md`](eval_data/results.md). Сырые данные: [`eval_data/results.json`](eval_data/results.json).

Запуск eval:
```bash
python eval_full.py                # использует фиксированный test-set, иначе создаёт
python eval_full.py --regenerate   # пересоздать test-set
```

---

## Стек

| Слой               | Технология                                        |
|--------------------|---------------------------------------------------|
| Web framework      | FastAPI (async) + Pydantic                        |
| Vector DB          | Qdrant (named vectors: dense + sparse, RRF native)|
| Dense embeddings   | `sentence-transformers/clip-ViT-B-32-multilingual`|
| Image embeddings   | `clip-ViT-B-32` (общее CLIP embedding-space)      |
| Sparse embeddings  | BM42 (`Qdrant/bm42-all-minilm-l6-v2-attentions`)  |
| Reranker           | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`      |
| LLM                | OpenAI-compatible (Cerebras / Groq / OpenAI / Ollama) |
| LLM cache          | Redis (shared cache для guardrail + reformulate, fail-open) |
| Rate limiting      | Redis fixed-window per IP (fail-open), 429 + Retry-After |
| Conversational     | LangGraph (3-node graph, conditional routing) + Redis session state |
| Observability      | Prometheus + Grafana + structlog (JSON)           |
| Tests              | pytest + pytest-mock + pytest-cov (90 тестов, coverage 82%) |
| Eval               | Frozen test-set (100 queries) + bootstrap 95% CI + 4 baselines |
| CI/CD              | GitHub Actions (ruff + pytest + Docker → GHCR)    |
| Orchestration      | Docker Compose                                    |

---

## Quickstart (Docker Compose)

Требования: Docker Desktop, ~15 GB свободного места (модели + датасет).

```bash
# 1. Подготовка
cp .env.example .env            # впиши LLM_API_KEY
python download_models.py       # скачает CLIP + reranker + BM42 в ./models/

# 2. Данные (первичная индексация — один раз)
docker compose up -d qdrant
python build_index.py           # скачает Fashion dataset и проиндексирует в Qdrant

# 3. Полный запуск
docker compose up --build -d    # первый build ~5 мин (CPU-only torch)
```

Сервисы после старта:

| Что              | URL                               |
|------------------|-----------------------------------|
| API + UI         | http://localhost:8000/            |
| API docs         | http://localhost:8000/docs        |
| Prometheus       | http://localhost:9090/            |
| Grafana          | http://localhost:3000/ (admin/admin) |
| Qdrant dashboard | http://localhost:6333/dashboard   |

---

## Dev-режим (без пересборки образа)

Compose монтирует `.:/app` и запускает `uvicorn --reload`. Правки любого `.py` подхватываются мгновенно без пересборки. Чтобы итерироваться ещё быстрее:

```bash
docker compose up -d qdrant redis prometheus grafana
python api.py                   # API нативно → Prometheus скрейпит через host.docker.internal:8000
```

---

## API

### POST `/search`
```json
{
  "query": "синяя рубашка для мужчин",
  "top_k": 5,
  "color": "Blue",
  "gender": "Men",
  "rerank": true,
  "reformulate": true,
  "guardrail": true,
  "with_llm": true
}
```

Ответ:
```json
{
  "query": "синяя рубашка для мужчин",
  "subqueries": ["синяя рубашка для мужчин", "blue shirt men", "navy mens top"],
  "recommendation": {
    "pick": "Flying Machine Men Check Blue Shirts",
    "reason": "...",
    "alternatives": "...",
    "suggestions": ["Бренд: Flying Machine или Spykar?", "..."]
  },
  "products": [ {"score": 0.87, "title": "...", "image_url": "..."} ]
}
```

### POST `/search/image`
`multipart/form-data` с полем `file` (JPEG/PNG) — поиск по содержимому картинки через CLIP.

### GET `/metrics`
Prometheus text format. Scraped каждые 10 сек.

### GET `/health`, GET `/filters`
Health-check + список доступных значений для фильтров (кеш at startup).

---

## Observability

**Structured logs** (каждая стадия пишется с `request_id`):
```json
{"event":"stage_complete","stage":"hybrid_rrf","duration_ms":67.3,"request_id":"a3f2...","timestamp":"..."}
{"event":"stage_complete","stage":"rerank","duration_ms":125.4,"request_id":"a3f2..."}
{"event":"stage_complete","stage":"llm_generate","duration_ms":890.1,"request_id":"a3f2..."}
{"event":"request","path":"/search","duration_ms":1087.9,"status":"ok","request_id":"a3f2..."}
```

**Prometheus metrics:**

| Метрика                              | Тип        | Labels                    |
|--------------------------------------|------------|---------------------------|
| `search_duration_seconds`            | histogram  | `stage` (hybrid_rrf/rerank/llm_guardrail/llm_reformulate/llm_generate/chat_analyze/chat_search/total) |
| `search_requests_total`              | counter    | `endpoint`, `status`      |
| `llm_errors_total`                   | counter    | `type` (guardrail/reformulate/generate/rate_limit) |
| `guardrail_rejections_total`         | counter    | `reason`                  |
| `cache_hits_total` / `cache_misses_total` | counter | `namespace` (guardrail/reformulate/chat_session) |
| `rate_limited_requests_total`        | counter    | `endpoint`                |

**Grafana dashboard** (`monitoring/grafana/dashboards/fashion-rag.json`) подгружается автоматически через provisioning — 5 панелей: latency p50/p95/p99 по стадиям, request rate, error rate, LLM errors, latency heatmap.

---

## Тестирование и CI/CD

```bash
pip install -r requirements-dev.txt
pytest --cov=.                  # 90 тестов, ~10 сек, coverage 82%
ruff check .                    # линтер
```

**Что покрыто:** HTTP-эндпоинты через FastAPI TestClient (health, filters, search, chat, /chat/graph, rate-limit), LangGraph-ноды (analyze/search_and_respond/decline, routing, end-to-end через мокнутый LLM, session persistence), guardrail-логика (пропуск/отказ/fail-open), reformulate, Redis-кеш (детерминистичность ключей, fail-open, round-trip), rate-limit (под/над порогом, изоляция по identifier), eval-метрики (P@5/R@10/NDCG@10/MRR@10 на известных входах + bootstrap CI), валидация pydantic-моделей, observability-хелперы.

Тесты не поднимают настоящие модели, Qdrant и Redis — заменяют их заглушками. Поэтому весь прогон ~10 сек и не требует GPU, интернета или запущенной инфры. Живую интеграцию проверяет `docker compose up`.

**GitHub Actions** (`.github/workflows/ci.yml`) на каждом push/PR автоматически:
1. Поднимает чистую Linux-машину
2. Ставит зависимости
3. Прогоняет линтер (`ruff`)
4. Прогоняет тесты с измерением coverage (порог 60%)
5. На push в main — собирает Docker-образ и пушит в GitHub Container Registry с тегами `latest` и `<git-sha>`

Готовый образ можно запустить без клонирования:
```bash
docker pull ghcr.io/ethernalsolitude/multimodal-fashion-rag:latest
```

Результат виден галочкой ✅ или крестиком ❌ рядом с коммитом.

---

## Структура проекта

```
.
├── api.py                      FastAPI endpoints (search/chat/image/metrics/health) + middleware
├── search.py                   Hybrid search + cross-encoder rerank + multi-query
├── llm.py                      LLM guardrail, reformulate, recommendation (с Redis-кешем)
├── chat.py                     LangGraph (analyze/search_and_respond/decline) + Redis сессии
├── cache.py                    Redis client wrapper: JSON get/set + rate-limit helper, fail-open
├── observability.py            structlog + Prometheus counters/histograms
├── build_index.py              Индексация датасета в Qdrant (dense + sparse)
├── download_models.py          Прекачать модели в ./models/
├── eval_full.py                Production eval-harness (frozen test-set, bootstrap CI, 4 baselines)
├── eval_metrics.py             Чистые метрики: P@5/R@10/NDCG@10/MRR@10 + bootstrap_ci
├── eval.py / eval_diverse.py   Legacy: старый categorical eval + LLM-judged eval
│
├── eval_data/                  Замороженный test-set + результаты eval'а (commited)
├── static/index.html           Web UI: вкладки Search/Chat + Mermaid-рендер LangGraph
├── tests/                      pytest + conftest со стабами тяжёлых зависимостей
├── monitoring/
│   ├── prometheus.yml          scrape config
│   └── grafana/
│       ├── dashboards/         JSON дашборды (автоподгрузка)
│       └── provisioning/       datasources + dashboard provider
│
├── docker-compose.yml          qdrant + redis + app + prometheus + grafana
├── Dockerfile                  CPU-only torch (GPU-вариант рядом закомментирован)
├── pyproject.toml              pytest + ruff + coverage config
├── requirements.txt
└── requirements-dev.txt
```

---

## Архитектурные решения

**Почему Qdrant, а не FAISS / pgvector?** Qdrant из коробки умеет hybrid-поиск: держит dense и sparse векторы в одной коллекции и сам объединяет рейтинги через RRF. В других решениях пришлось бы мержить результаты двух отдельных индексов руками.

**Почему CLIP-multilingual?** Текстовая модель (включая русский) и картиночная CLIP живут в общем векторном пространстве — поэтому и текст, и картинку можно искать в одном индексе, без отдельной базы для изображений.

**Почему BM42, а не классический BM25?** BM42 использует веса трансформера вместо IDF и лучше работает на коротких текстах (название товара), не требует токенизации и подготовки корпуса.

**Почему мультиязычный reranker?** Стандартный `ms-marco-MiniLM` умеет только английский — на русском запросе выдавал нулевые score. Замена на `mmarco-mMiniLMv2` (14 языков) решила проблему без переизобретения пайплайна.

**Почему guardrail не блокирует при ошибке?** Если LLM-классификатор временно недоступен, вежливее пропустить запрос, чем показать пользователю «сервис не работает». Все сбои при этом считаются в метрике `llm_errors_total{type="guardrail"}` — видно в Grafana когда система деградирует.

**Почему LangGraph для чата, а не if/else?** Сейчас граф минимальный — analyze + две ветки. Но структура уже отделяет orchestration от бизнес-логики, и добавить self-correction loop (нода оценивает результаты, при низкой релевантности уходит на reformulate) — это новый узел и одно ребро, без рефакторинга всего пайплайна. Плюс LangGraph сам себе рисует диаграмму через `draw_mermaid()` — UI её показывает, README встраивает, поддерживать руками не нужно.

**Почему Redis fail-open?** И кеш, и rate-limit, и chat-сессии могут работать без Redis (просто без кеша / без лимитов / без памяти между ходами). Выбор сознательный: лучше пропустить лишние LLM-запросы во время outage'а Redis'а, чем уронить весь сервис. Все деградации видны в Prometheus — `cache_misses_total` спайкнет, можно алертить.

**Почему в тестах заглушки, а не живые зависимости?** Прогон за 10 сек можно запускать на каждое сохранение файла. Поднимать Qdrant, Redis и скачивать 2 GB моделей ради unit-теста — медленно и ненадёжно в CI. Настоящая интеграция проверяется одним `docker compose up`.

---

## Ограничения / что можно улучшить

- **Test-set синтетический** — ground truth задан категориальными триплетами (color × gender × category), а не реальными кликами пользователей. На fuzzy-запросах вроде «что-нибудь для зала» картина будет другой — отдельная eval с LLM-judge'ем в `eval_diverse.py` показывает большую вариативность.
- **Нет персонализации и A/B** — для реального прода добавился бы userId, контекстный re-ранк, feature flags для экспериментов с конфигами baseline.
- **Auth/quotas отсутствуют** — rate-limit по IP это база, но для production нужны API-keys или JWT + per-tenant квоты.
- **Conversational mode зависит от качества LLM** — на `gpt-oss-120b` контекстные уточнения иногда промахиваются. На более сильной модели (Llama 3.3 70B / Claude Haiku) расчёт intent'а из истории заметно стабильнее.

---

## Лицензия

MIT
