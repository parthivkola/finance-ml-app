"""
FastAPI application entry point — satisfies FR-API-001 to FR-API-008.
Run with: PYTHONPATH=src uv run fastapi dev src/market_intelligence/api/main.py
"""
from __future__ import annotations


from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from market_intelligence.api import predict, routes
from market_intelligence.api.schemas import HealthResponse
from market_intelligence.ml.trainer import MODELS_DIR
from market_intelligence.scheduler import create_scheduler

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the background scheduler for data refresh
    scheduler = create_scheduler()
    scheduler.start()
    yield
    # Shutdown: Stop the scheduler gracefully
    scheduler.shutdown()

app = FastAPI(
    title="Financial Market Intelligence System API",
    description="AI-powered financial news sentiment analysis & stock movement prediction.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — allow React frontend and any API consumer (FR-API-001)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8080",
        "http://finance-market-intelligence.app",
        "https://finance-market-intelligence.app",
        "http://www.finance-market-intelligence.app",
        "https://www.finance-market-intelligence.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routers
app.include_router(predict.router, prefix="/api/v1")
app.include_router(routes.router, prefix="/api/v1")


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """FR-API-004 — health probe for Docker / load balancer."""
    available = [p.stem.rsplit("_", 1)[0] for p in MODELS_DIR.glob("*_v*.joblib")]
    return HealthResponse(status="healthy", models_available=sorted(set(available)))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """FR-API-007 — structured JSON errors for all unhandled exceptions."""
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "path": str(request.url)},
    )
