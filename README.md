# Financial Market Intelligence System

> AI-powered financial news sentiment analysis and stock movement prediction platform.

---

## Architecture

```
financial-market-intelligence-system/
├── backend/                   # FastAPI + ML pipeline (Python / uv)
│   ├── src/market_intelligence/
│   │   ├── api/               # FastAPI routers (predict, news, history, models)
│   │   ├── data/              # Yahoo Finance + AlphaVantage ingestors
│   │   ├── db/                # SQLAlchemy models + Alembic migrations
│   │   ├── ml/                # Trainer, explainability (SHAP), feature engineering
│   │   ├── nlp/               # FinBERT + VADER sentiment pipeline
│   │   └── scheduler.py       # APScheduler: hourly refresh + auto-retrain
│   ├── tests/                 # pytest: unit, integration, system
│   ├── alembic/               # Database migrations
│   └── Dockerfile
├── frontend/                  # React 18 + TypeScript + Vite
│   ├── src/
│   │   ├── api/               # Axios API client
│   │   └── components/        # Sidebar, ModelSelector, StockChart, ShapExplainer, NewsFeed
│   ├── nginx.conf             # Production nginx config
│   └── Dockerfile
├── docker-compose.yml         # One-command full-stack deployment
└── README.md
```

## Features

- **Multi-Model Prediction** — XGBoost, LightGBM, Random Forest, Logistic Regression with selectable champion model
- **Explainability** — SHAP feature impact charts per prediction
- **NLP Pipeline** — FinBERT (primary) + VADER (fallback) sentiment scoring
- **Hourly Auto-Refresh** — APScheduler fetches new prices and news every hour
- **Auto-Retrain** — Models automatically retrain when 30+ new price rows are added
- **Full Audit Trail** — All predictions and API calls logged to PostgreSQL

---

## Quick Start (Docker)

```bash
git clone https://github.com/YOUR_USERNAME/financial-market-intelligence-system
cd financial-market-intelligence-system

# Add your API keys
cp .env.example .env
# Fill in ALPHAVANTAGE_API_KEY and HUGGINGFACE_TOKEN in .env

docker compose up --build
```

- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8080
- **Swagger Docs:** http://localhost:8080/docs

---

## Local Development

### Prerequisites
- Python 3.10+ with `uv` (`pip install uv`)
- Node.js 18+
- Docker + Docker Compose

### Backend

```bash
cd backend
uv sync
docker compose up -d postgres   # start PostgreSQL only
uv run alembic upgrade head     # apply DB migrations
PYTHONPATH=src uv run python src/market_intelligence/ml/trainer.py   # train models
PYTHONPATH=src uv run fastapi dev src/market_intelligence/api/main.py --port 8080
```

### Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

### Tests

```bash
cd backend
PYTHONPATH=src uv run pytest -v
```

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string | ✅ |
| `ALPHAVANTAGE_API_KEY` | AlphaVantage news API key | ✅ |
| `HUGGINGFACE_TOKEN` | HuggingFace token for FinBERT | ✅ |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | System health + available models |
| `GET` | `/api/v1/models` | All trained models with metrics |
| `GET` | `/api/v1/history/{symbol}` | Historical OHLCV prices |
| `GET` | `/api/v1/news/{symbol}` | Scored news articles |
| `POST` | `/api/v1/predict` | AI prediction with SHAP explanation |

---

## License

MIT
