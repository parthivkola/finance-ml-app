from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, UniqueConstraint
from sqlalchemy.sql import func
from market_intelligence.db.session import Base


class StockPrice(Base):
    __tablename__ = "stock_prices"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    date = Column(DateTime, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)

    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_stock_symbol_date"),
    )


class NewsArticle(Base):
    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    url_hash = Column(String, unique=True, index=True)
    url = Column(String, nullable=True)
    published_date = Column(DateTime)
    title = Column(String)
    summary = Column(String, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    sentiment_label = Column(String, nullable=True)
    sentiment_model = Column(String, nullable=True)  # "FinBERT" or "VADER"


class Prediction(Base):
    """Stores every prediction made by the API — satisfies FR-DB-001."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    predicted_at = Column(DateTime, server_default=func.now(), index=True)
    model_name = Column(String)
    prediction = Column(String)          # "UP", "DOWN", "NEUTRAL"
    confidence = Column(Float)
    shap_explanation = Column(JSON, nullable=True)   # top-N features as JSON


class ModelRegistry(Base):
    """Tracks every trained model version — satisfies FR-DB-003."""

    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)   # "xgboost", "lightgbm", etc.
    version = Column(String)                  # "v1", "v2", ...
    trained_at = Column(DateTime, server_default=func.now())
    accuracy = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    train_accuracy = Column(Float, nullable=True)
    overfit_status = Column(String, nullable=True)
    artifact_path = Column(String)            # path to .joblib file
    total_price_rows = Column(Integer, nullable=True)  # snapshot of row count at train time


class APILog(Base):
    """Logs every API request — satisfies FR-DB-004."""

    __tablename__ = "api_logs"

    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String)
    method = Column(String)
    symbol = Column(String, nullable=True)
    status_code = Column(Integer)
    response_time_ms = Column(Float, nullable=True)
    requested_at = Column(DateTime, server_default=func.now(), index=True)
