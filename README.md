# 🚛 Freight Ops Copilot: AI for a Port Klang 3PL

[![Live demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://supply-chain-ai-copilot-rul.streamlit.app/)
[![CI](https://github.com/kl2412018279-ux/supply-chain-ai-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/kl2412018279-ux/supply-chain-ai-copilot/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Cost](https://img.shields.io/badge/running%20cost-%240-brightgreen)

**🚀 [Live demo](https://supply-chain-ai-copilot-rul.streamlit.app/)**: try the copilot in your browser. The free host sleeps when idle, so the first load can take about 30 seconds.

An AI copilot for a **Malaysian third-party logistics (3PL) provider at Port Klang**: the kind of company that handles customs clearance, container haulage, warehousing and final delivery of **chemicals, palm oil and plastics** for international freight forwarders.

> **Case study, not a real company.** The copilot is built for *Muara Logistics Sdn Bhd*, a **fictional** company modelled on typical Port Klang ground-handling 3PLs. All volumes are synthetic, and all rates are clearly labelled examples, see [Rate card](#rate-card-rm). This is an independent portfolio project, not affiliated with any company.

| Quote builder | Volume forecast |
|---|---|
| ![Quote page: itemised RM cost breakdown](docs/images/quote.png) | ![Forecast page: weekly TEU actual vs forecast](docs/images/forecast.png) |
| **Route & carrier planner** | **Delay-risk prediction** |
| ![Route planner: options ranked by price, transit, CO2e and reliability](docs/images/planner.png) | ![Delay risk: probabilities and risk drivers](docs/images/delay-risk.png) |
| **Document intake (B/L extraction)** | **Haulier scorecard (dark mode)** |
| ![Document intake: extracted fields and validation](docs/images/intake.png) | ![Haulier scorecard with ESG evidence](docs/images/hauliers-dark.png) |

## What it does

| Question an ops team asks | What the copilot does |
|---|---|
| *"Quote 2 × 40ft palm oil from Westport to Pasir Gudang"* | Builds an **itemised quote in RM** on the **real truck road distance** (OpenRouteService, with the road drawn on the map and turn-by-turn directions): diesel, truck maintenance, driver, tolls, Westports handling, customs (K1/K2), cargo surcharges, warehousing and margin. Shows **CO2e computed from the same diesel litres**. |
| *"Compare truck vs rail for chemicals to Penang"* | Prices every transport mode side by side and reports the **RM cost per tonne of CO2e avoided**. |
| *"How many palm oil containers next month?"* | **LightGBM TEU forecast** per commodity, with Chinese New Year and Hari Raya dips and the Sep–Nov palm-oil harvest peak. |
| *"What's our projected haulage spend for chemicals?"* | Forecast TEU × rate card → **projected cost, revenue, diesel and CO2e in RM**. |
| *"Which carrier is best for chemicals to Penang? Cheapest please"* | **Route & carrier planner**: ranks every mode × carrier on price, transit days, CO2e and *historical* on-time rate with your priority weights, flags Pareto-dominated options and explains the trade-off (*"1 day slower, 42% cheaper, 71% less CO2e"*). |
| *"Will a chemicals shipment to Kuching in December be delayed?"* | **Delay-risk classifier** (Low / Medium / High) with probabilities, the drivers behind the prediction (monsoon, **live rainfall at the destination from Open-Meteo**, congestion, customs inspection, carrier…) and expected **demurrage exposure in RM**. |
| *(upload a Bill of Lading PDF)* | **Document intake**: extracts B/L and invoice fields into a Pydantic record, validates ISO 6346 container check digits, HS codes and the 25 t weight limit, turns the document into a haulage quote, saves it to the database after review. |
| *"Show the haulier scorecard"* / *"How does Cempaka score on ESG?"* | **Haulier scorecard**: on-time, damage and claims from SQL over the delivery log, plus ESG criteria evidenced by quoting each haulier's own policy document; composite grade A–D. |
| *"What documents do I need to clear a chemical import?"* | **Hybrid RAG** over the operations manual (customs forms, dangerous goods, palm oil, demurrage, warehousing), with page citations. |
| *"Ignore previous instructions…"* | 🛡️ Blocked by input guardrails before any tokens are spent. |

It runs **fully offline with no API key**. Road routing and live weather are optional: without an `ORS_API_KEY` quotes use the rate card's distance table, and without network access the delay model uses typical rainfall for the month. Without an LLM configured, a deterministic rule-based agent calls the same tools. It also falls back to that agent automatically if the LLM provider fails.

---

## Architecture

```mermaid
flowchart LR
    U[Ops / sales user] --> UI[Streamlit dashboard]
    F[Forwarder systems] --> API[FastAPI]
    N8N[n8n: Google Drive trigger] -->|POST /documents/ingest| API
    UI --> COP
    API --> COP

    subgraph COP[Copilot pipeline]
        direction TB
        G1[Input guardrails<br/>injection / off-topic / length] --> C{Semantic cache<br/>Redis + entity fingerprint}
        C -- hit --> OUT
        C -- miss --> AG[LangGraph agent<br/>Groq gpt-oss / Ollama / OpenAI / Claude<br/>fallback: rule-based agent]
        AG --> OUT[Output guardrails<br/>redaction + Pydantic schema]
    end

    AG -->|tool| Q[quote_container_job<br/>compare_transport_modes]
    AG -->|tool| FC[forecast_container_volume<br/>estimate_haulage_spend]
    AG -->|tool| V[get_volume_summary<br/>parameterised SQL]
    AG -->|tool| D[search_operations_documents]
    AG -->|tool| R[recommend_route_and_carrier]
    AG -->|tool| DR[predict_delay_risk]
    AG -->|tool| HS[get_haulier_scorecard]

    Q --> RC[(rates.json<br/>RM rate card + carriers)]
    R --> RC
    R --> HS
    DR --> RF[Random-forest delay model]
    HS --> DL[(Delivery log<br/>SQL)]
    API -->|POST /documents/extract| X[B/L & invoice extraction<br/>LLM or rules + validation]
    X --> PG
    FC --> LGB[LightGBM TEU model]
    D --> HR[Hybrid retriever<br/>pgvector + full-text → RRF → FlashRank]
    HR --> PG[(PostgreSQL + pgvector<br/>volumes + embeddings)]
    V --> PG
    AG -. traces .-> LS[LangSmith]
    LGB -. trained with .-> MLF[MLflow]
```

### Request flow

1. **Input guardrails** block prompt injection, jailbreaks, SQL keywords, secret fishing and off-topic questions in under 1 ms.
2. **Semantic cache.** A similar question is answered from Redis in milliseconds. A hit also requires the *entity fingerprint* to match (numbers, destinations, commodities, transport modes), so *"quote 40ft to Penang"* never returns the cached answer for *"quote 40ft to Ipoh"*.
3. **LangGraph agent.** An explicit `agent → tools → agent` state machine. The LLM picks tools, and the number of steps is capped.
4. **Output guardrails.** API keys, e-mails, DSNs and card numbers are redacted, and the answer is validated against a Pydantic schema. Tool calls and cited sources are returned as JSON.

---

## Rate card (RM)

Every quote is built bottom-up from [`data/reference/rates.json`](data/reference/rates.json). A planner can update the weekly diesel price or contract rates there without touching code. Each rate records where it came from:

| Rate | Example value | Source |
|---|---|---|
| Diesel (Peninsular) | RM2.99 / L | MOF weekly APM retail price, Feb 2026 (latest found) |
| Port handling 20ft / 40ft | RM375 / RM562.50 | Westports container tariff effective 1 Jan 2026 |
| Max gross weight on trailer | 25,000 kg | Association of Malaysian Hauliers limit at Port Klang (May 2025) |
| CO2 per litre diesel | 2.68 kg | Standard tank-to-wheel combustion factor |
| Fuel use, truck costs, driver, tolls, customs fees, surcharges, warehousing, rail/sea | see file | **Assumptions**: replace with real contract rates |

How a truck quote is calculated:
- **Fuel** = round-trip km (loaded out, empty back) × litres per km × diesel price.
- **Truck and driver** = km × maintenance rate + hours × (fixed cost + driver wage).
- **Fixed charges**: tolls, port handling, customs declaration and inspection.
- **Cargo surcharges**: 25% dangerous-goods surcharge for chemicals, palm-oil tank fee.
- **Warehousing** (DG storage at 1.8× the normal rate).
- **15% margin** on top of cost.
- **CO2e** = diesel litres × 2.68.

Sanity check: haulage alone from Westport to Shah Alam comes out at about RM557 cost (RM640 with margin). That is in line with the last published regulated rate of RM639 (RM533 haulage + tolls + fuel adjustment, MIDA, 2011).

---

## Production features

| # | Feature | Implementation |
|---|---|---|
| 1 | **Hybrid search + reranking** | pgvector HNSW and PostgreSQL full-text search (BM25 in memory), fused with Reciprocal Rank Fusion and reranked with a FlashRank cross-encoder. Exact codes like *K8* or *UN* numbers are still matched. `src/rag/` |
| 2 | **Observability & MLOps** | LangSmith traces every node, LLM call and tool call. `/metrics` reports p50/p95 latency, cache hit rate and guardrail blocks. MLflow logs every grid-search trial with MAE, RMSE, R² and MAPE. |
| 3 | **AI safety & guardrails** | Pydantic schemas for every tool argument and response, injection and off-topic filters, output redaction, no text-to-SQL. `src/guardrails/` |
| 4 | **Semantic caching** | Redis-backed (in-memory fallback), with TTL, entity fingerprinting, invalidation on document changes, and no caching of fallback answers. `src/cache/` |
| 5 | **Resilience** | If the LLM provider fails (retired model, outage, rate limit), the rule-based agent answers with the same tools. Every external dependency (Postgres, Redis, LLM, embedding model) is optional. |
| 6 | **Testing & CI/CD** | 192 pytest tests: cost engine, routing, delay model, extraction, scorecard SQL, forecasting, RAG, agents (including a scripted-LLM LangGraph test), API, headless tests of every dashboard page, and pgvector/Redis integration. GitHub Actions runs ruff → pytest (Python 3.10 and 3.12) → Docker build. |

---

## Results

**Container-volume forecast** (synthetic data, 56-day time-based holdout, `python -m src.forecasting.train --tune`):

| Model | MAE (TEU/day) | RMSE | R² |
|---|---|---|---|
| Seasonal naive (same weekday last week) | 4.43 | 5.52 | 0.417 |
| **LightGBM** (lags, rolling stats, calendar, holidays) | **3.23** | **4.24** | **0.656** |

That is **27% lower MAE** than the baseline. R² is moderate because daily counts of around 10 TEU are inherently noisy.

**Delay-risk classifier** (5,142 synthetic shipments; time-based split, tested on the latest 20%; `python -m src.delays.model`):

| Model | Accuracy | Macro F1 | High-risk recall |
|---|---|---|---|
| Always the majority class | 0.627 | 0.257 | 0.000 |
| Majority class per transport mode | 0.726 | 0.454 | 0.000 |
| **Random forest** (congestion, monsoon, rainfall, holiday, inspection, carrier…) | **0.878** | **0.788** | **0.810** |

Both baselines miss every high-risk shipment, which is the class that costs money. The model catches about 4 in 5 of them.

**Where the inputs come from.** The history is synthetic (a real deployment would train on the company's own TMS records). At prediction time, rainfall on the ETA date comes from [Open-Meteo](https://open-meteo.com) (free, no key): the forecast for the next 15 days, observed archive data for past dates, and the same date last year for dates further ahead. If the lookup fails, the model uses typical rainfall for that month and region, the same climatology the synthetic history was generated from. Berth occupancy defaults to a seasonal estimate and can be set by hand. Open-Meteo's free tier is for non-commercial use.

**Retrieval** (`python -m src.rag.evaluate`: 20 labelled questions over the operations manual, searched across the whole library, including the haulier policies):

| Method | hit@1 | hit@3 | MRR@3 |
|---|---|---|---|
| Dense only | 0.60 | 0.85 | 0.72 |
| Keyword only (BM25) | 0.70 | 0.95 | 0.82 |
| Hybrid (RRF) | 0.70 | 0.85 | 0.78 |
| **Hybrid + rerank** | **0.70** | **1.00** | **0.84** |

These numbers use the offline backends (hashing embedder, lexical reranker), which is what CI runs. Measure the real bge-small + FlashRank stack with `EMBEDDING_BACKEND=sentence-transformers RERANKER_BACKEND=flashrank make eval`.

---

## Quick start

### Option A: local Python (about 2 minutes)

```bash
git clone https://github.com/kl2412018279-ux/supply-chain-ai-copilot.git
cd supply-chain-ai-copilot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # optionally add a free GROQ_API_KEY
python -m src.forecasting.train --tune      # train the TEU model
streamlit run app.py                        # http://localhost:8501
```

### Option B: full stack with Docker

```bash
cp .env.example .env            # add GROQ_API_KEY (free at https://console.groq.com)
docker compose up -d --build
```

| Service | URL |
|---|---|
| Streamlit dashboard | http://localhost:8501 |
| FastAPI (Swagger UI) | http://localhost:8000/docs |
| n8n automation | http://localhost:5678 |
| MLflow UI (`docker compose --profile mlops up -d`) | http://localhost:5000 |

On first start the API trains the forecaster, loads the volumes into Postgres and indexes the documents in `data/raw/`.

### Road routing (optional, free)

Truck quotes use the rate card's approximate distance table by default. For the real road, sign up at [openrouteservice.org](https://openrouteservice.org/dev/#/signup) (free tier: 2,000 route requests a day) and set `ORS_API_KEY`. Quotes then use the actual road distance and drive time (heavy-goods-vehicle profile), the Route view draws the road and lists turn-by-turn directions, and the copilot names the main roads. Routes are cached per origin and destination, and any API error falls back to the distance table.

### LLM providers (all have $0 options)

| `LLM_PROVIDER` | Key | Default model |
|---|---|---|
| `groq` (recommended, free tier) | `GROQ_API_KEY` | `openai/gpt-oss-120b` |
| `ollama` (fully local) | none, run `ollama pull llama3.1` | `llama3.1` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` |
| `none` | none | rule-based offline agent |

Override the model with `LLM_MODEL`. Providers retire models over time (Groq shut down `llama-3.3-70b-versatile` in August 2026). The copilot falls back to the rule-based agent if the LLM call fails.

---

## REST API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/chat` | Copilot answer + tool calls + sources (guardrails and cache applied) |
| `POST` | `/quote` | Itemised RM quote + CO2e for a container job |
| `POST` | `/quote/compare` | Every transport mode for a destination, with RM per tonne CO2e avoided |
| `POST` | `/routing/recommend` | `{job, priorities}` → every mode × carrier option ranked, with explanation |
| `POST` | `/delays/predict` | Delay risk, drivers and demurrage exposure for one shipment |
| `GET` | `/delays/upcoming` | Upcoming arrivals scored and sorted by risk |
| `GET` | `/hauliers/scorecard`, `/hauliers/{id}/trend` | Ranked scorecard with ESG evidence; monthly on-time trend |
| `POST` | `/documents/extract` | Bill of Lading / invoice PDF → validated fields + suggested quote |
| `POST` / `GET` | `/documents/records` | Save a reviewed extraction / list saved documents |
| `GET` | `/destinations`, `/rates` | Destinations served and the full rate card |
| `POST` | `/forecast` | `{commodity, horizon_days}` → daily TEU forecast |
| `GET` | `/commodities` | Commodities known to the model |
| `GET` | `/volumes/summary` | `?commodity=&start_date=&end_date=` historical TEU |
| `POST` | `/documents/ingest` | Upload a PDF/TXT/MD (used by n8n) |
| `POST` | `/documents/search` | Hybrid search with optional rerank |
| `GET` / `DELETE` | `/documents`, `/documents/{source}` | List / remove documents |
| `POST` | `/emissions` | Generic tonne-km CO2e for any shipment |
| `GET` | `/health`, `/metrics` | Liveness, latency percentiles, cache hit rate |

```bash
curl -s localhost:8000/quote -H 'content-type: application/json' \
  -d '{"destination": "Penang", "container_size": "40", "cargo_type": "chemicals"}' | jq
```

**n8n:** import `n8n/drive_to_copilot_ingest.json`, connect Google Drive and set the folder. Every SOP or manual dropped into that folder is indexed automatically.

---

## Data

- **`data/raw/container_volumes.csv`**: synthetic daily TEU per commodity (chemicals, palm oil, plastics, general), January 2024 to September 2026. It includes weekday patterns, holiday dips, the palm-oil harvest peak, year-end peaks and growth. Regenerate it with `python -m src.forecasting.data`. Real exports work too: any CSV with a date, a commodity/series column and a TEU/volume column.
- **`data/raw/operations_manual.pdf`**: a **fictional** 10-section operations manual covering customs (K1/K2/K8, uCustoms), dangerous goods, palm oil (heating, MSPO, EUDR), plastics, haulage, demurrage/detention, warehousing, pricing and ESG. It is generated from `scripts/sample_docs/operations_manual.md`. Its rules are simplified examples, so verify real requirements with JKDM and the port operators.
- **`data/raw/shipment_history.csv`**: 5,142 synthetic shipments with delay outcomes. Delays are driven by congestion, monsoon, daily rainfall at the destination, holiday weeks, customs inspection, dangerous goods, weekend arrival, distance and carrier reliability. Regenerate it with `python -m src.delays.model --regenerate`.
- **`data/raw/haulier_deliveries.csv`** and **`data/raw/haulier_esg_*.md`**: synthetic delivery logs and **fictional** policy documents for six fictional carriers. The policies range from strong to weak, so the scorecard has something real to separate.
- **`data/samples/*.pdf`**: fictional Bills of Lading and a commercial invoice (`scripts/make_sample_shipping_docs.py`). `bl_chemicals_errors.pdf` deliberately contains a bad container check digit and an overweight container.
- **`data/reference/rates.json`**: the rate card described above, plus the fictional carriers, transit assumptions and demurrage rates.

## Development

```bash
pip install -r requirements-ci.txt   # lightweight: no torch / LLM SDKs
make lint                             # ruff check + format check
make test                             # all unit tests, no services needed
docker compose up -d postgres redis && make test-all   # + pgvector/Redis integration tests
```

## Deploying the demo for free

- **Streamlit Community Cloud.** Use main file `app.py` with Python 3.11 or 3.12. `packages.txt` installs LightGBM's OpenMP runtime. Add these secrets:

  ```toml
  GROQ_API_KEY = "gsk_..."
  ORS_API_KEY = "..."             # optional: real truck routes (free at openrouteservice.org/dev/#/signup)
  EMBEDDING_BACKEND = "hashing"   # keeps memory under the free 1 GB limit
  RERANKER_BACKEND = "lexical"
  ```
- **Hugging Face Spaces.** Use a Docker Space with the command `streamlit run app.py --server.port 7860 --server.address 0.0.0.0`.
- **Supabase** (free Postgres with pgvector). Set `DATABASE_URL`.

## Project structure

```
supply-chain-ai-copilot/
├── app.py                       # Streamlit entry: sidebar + top navigation
├── views/                       # pages: copilot, quote, planner, forecast, delays, intake, hauliers, documents
├── .streamlit/config.toml       # light/dark theme
├── data/
│   ├── raw/                     # container_volumes.csv, operations_manual.pdf
│   └── reference/rates.json     # RM rate card (example values + sources)
├── src/
│   ├── costing/                 # rate card loader, quote engine, mode comparison, route & carrier planner, OpenRouteService road routing
│   ├── delays/                  # shipment history generator, delay-risk classifier + explanations, Open-Meteo rainfall
│   ├── extraction/              # B/L & invoice extraction, ISO 6346 validation, document store
│   ├── hauliers/                # delivery logs, SQL scorecard, ESG evidence
│   ├── forecasting/             # TEU data/features, train (MLflow), predict, volume repo, CO2 calc
│   ├── rag/                     # ingestion, embeddings, pgvector store, reranker, retriever, eval
│   ├── agent/                   # tools, LangGraph workflow + rule-based agent, copilot pipeline
│   ├── guardrails/              # input checks, redaction, response schemas
│   ├── cache/                   # semantic cache (Redis / memory)
│   ├── api/main.py              # FastAPI
│   ├── config.py, observability.py
├── notebooks/                   # 01 volume exploration, 02 training + RM projection
├── docker/, docker-compose.yml, Dockerfile, n8n/, scripts/
└── tests/                       # pytest suite (192 tests)
```

## Roadmap

**Next ideas:**
- **Free-time countdown**: combine extracted B/Ls with the delay model to alert on containers about to incur demurrage.
- **Fleet maintenance tracker**: cost per km per prime mover, plus service-due alerts.
- **Warehouse occupancy forecast** for the general and DG stores.
- **n8n automation**: a forwarder's rate-request e-mail produces a draft quote; a scanned B/L in Google Drive triggers extraction (scanned images need OCR or a vision model first).

**Known limitations:**
- Rates are examples; distances and tolls are approximate.
- The forecast is a point estimate. Quantile regression would add prediction intervals for capacity buffers.
- The semantic cache scans entries linearly; use a RediSearch vector index at scale.
- Guardrails are deterministic rules. Add NeMo Guardrails or an LLM judge to check that answers are grounded in their sources.
