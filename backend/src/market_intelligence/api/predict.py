"""
POST /api/v1/predict  — satisfies FR-API-001, FR-XAI-001, FR-DB-001.

Fetches fresh prices, pulls latest sentiment from DB, builds features,
runs the selected model, generates SHAP explanations, and persists results.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from market_intelligence.api.schemas import FeatureImpact, PredictRequest, PredictResponse, TechnicalIndicators
from market_intelligence.data.yahoo_finance import fetch_prices
from market_intelligence.db.models import APILog, Prediction
from market_intelligence.db.session import SessionLocal, engine
from market_intelligence.ml.explainability import explain
from market_intelligence.ml.trainer import FEATURE_COLS, MODELS_DIR, build_features

router = APIRouter()

VALID_MODELS = tuple(f"{m}_{h}d" for m in ("xgboost", "lightgbm", "random_forest", "logistic_regression") for h in (1, 3, 5)) + tuple(f"auto_{h}d" for h in (1, 3, 5))


def _load_model(model_name: str):
    candidates = sorted(MODELS_DIR.glob(f"{model_name}_v*.joblib"), reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"Model '{model_name}' not found in {MODELS_DIR}. Run trainer.py first."
        )
    return joblib.load(candidates[0])


def _get_avg_sentiment(symbol: str) -> tuple[float, bool]:
    """
    Pulls the average sentiment score for a symbol from the DB (last 30 days).
    Returns (score, has_data). If no scored articles exist → (0.0, False).
    """
    try:
        result = pd.read_sql(
            """
            SELECT AVG(sentiment_score) AS avg_score
            FROM news_articles
            WHERE symbol = %(symbol)s
              AND sentiment_score IS NOT NULL
              AND published_date >= NOW() - INTERVAL '30 days'
            """,
            engine,
            params={"symbol": symbol},
        )
        val = result["avg_score"].iloc[0]
        if val is None or pd.isna(val):
            return 0.0, False
        return float(val), True
    except Exception:
        return 0.0, False


def _get_feature_row(symbol: str, horizon: int = 1) -> tuple[pd.DataFrame, bool]:
    """
    Fetch prices + attach real sentiment from DB + merge macro market context.
    Returns (feature_row, has_sentiment_data).
    """
    prices = fetch_prices(symbol, days=400)
    if prices.empty:
        raise HTTPException(status_code=400, detail=f"Invalid stock symbol or no data found for '{symbol}'. Please enter a valid symbol (e.g. AAPL).")

    prices.columns = [c.lower() for c in prices.columns]
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)

    # Fetch daily sentiment up to today
    try:
        sentiment_history = pd.read_sql(
            """
            SELECT DATE(published_date) AS date, AVG(sentiment_score) AS sentiment_score
            FROM news_articles
            WHERE symbol = %(symbol)s AND sentiment_score IS NOT NULL
            GROUP BY DATE(published_date)
            ORDER BY date
            """,
            engine,
            params={"symbol": symbol},
        )
        sentiment_history["date"] = pd.to_datetime(sentiment_history["date"])
        prices["date"] = pd.to_datetime(prices["date"])
        prices = prices.merge(sentiment_history, on="date", how="left")
        
        # Check if we have *any* sentiment data in the last 30 days
        has_sentiment = _get_avg_sentiment(symbol)[1]
    except Exception:
        prices["sentiment_score"] = 0.0
        has_sentiment = False

    # Fetch and merge market context
    try:
        from market_intelligence.ml.trainer import fetch_market_context
        macro_df = fetch_market_context()
        prices = prices.merge(macro_df, on="date", how="left")
    except Exception as e:
        print(f"⚠️  Failed to attach market context in predict.py: {e}")

    featured = build_features(prices, horizon=horizon)
    if featured.empty:
        raise ValueError(f"Not enough price data to build features for {symbol}")

    available = [c for c in FEATURE_COLS if c in featured.columns]
    return featured[available].iloc[[-1]], has_sentiment


@router.post("/predict", response_model=PredictResponse, tags=["Predictions"])
async def predict(req: PredictRequest, request: Request):
    """
    Predict next-day movement for a stock symbol.
    model_name: 'xgboost' | 'lightgbm' | 'random_forest' | 'logistic_regression'
    """
    t0 = time.time()
    symbol = req.symbol.upper().strip().replace(".", "-")
    model_name = req.model_name or "xgboost_1d"

    if model_name not in VALID_MODELS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid model_name. Choose from: {VALID_MODELS}",
        )

    # Extract horizon from model_name (e.g. "xgboost_3d" -> 3)
    try:
        horizon = int(model_name.split("_")[-1].replace("d", ""))
    except Exception:
        horizon = 1

    db = SessionLocal()
    try:
        # 1. Build feature row (with real sentiment from DB)
        try:
            feature_row, has_sentiment = _get_feature_row(symbol, horizon)
            if not has_sentiment:
                # Try fetching live news and scoring it immediately
                from market_intelligence.data.rss_news import fetch_rss_news
                from market_intelligence.data.storage import save_news
                from market_intelligence.nlp.process_news import process_unscored_news
                news_df = fetch_rss_news(symbol)
                if not news_df.empty:
                    save_news(news_df, symbol)
                    process_unscored_news()
                    # Retry getting sentiment
                    new_score, has_sentiment = _get_avg_sentiment(symbol)
                    if has_sentiment:
                        feature_row["sentiment_score"] = new_score

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Feature build failed: {e}")

        # 2. Load model
        try:
            if model_name.startswith("auto_"):
                best_conf = -1.0
                best_model = None
                best_actual_name = None
                best_feature_input = None
                
                # Test all 4 underlying models for this horizon
                # First, query DB to find which ones are overfit so we can exclude them
                overfit_models = set()
                try:
                    from market_intelligence.db.models import ModelRegistry
                    registry_records = db.query(ModelRegistry).filter(
                        ModelRegistry.model_name.like(f"%_{horizon}d")
                    ).all()
                    for r in registry_records:
                        if r.overfit_status and "OVERFIT" in r.overfit_status:
                            overfit_models.add(r.model_name)
                except Exception as db_err:
                    print(f"Failed to query overfit status for ensemble: {db_err}")

                import numpy as np
                ensemble_probs = []
                model_objects = []

                for base_name in ["xgboost", "lightgbm", "random_forest", "logistic_regression"]:
                    test_name = f"{base_name}_{horizon}d"
                    if test_name in overfit_models:
                        continue  # Skip overfit models
                        
                    try:
                        m = _load_model(test_name)
                    except FileNotFoundError:
                        continue
                        
                    if test_name.startswith("logistic_regression"):
                        scaler_path = MODELS_DIR / f"scaler_{horizon}d.joblib"
                        if scaler_path.exists():
                            scaler = joblib.load(scaler_path)
                            f_input = scaler.transform(feature_row)
                        else:
                            f_input = feature_row.values
                    else:
                        f_input = feature_row
                        
                    probs = m.predict_proba(f_input)[0]
                    ensemble_probs.append(probs)
                    model_objects.append({
                        "name": test_name,
                        "model": m,
                        "feature_input": f_input,
                        "probs": probs
                    })
                        
                if not model_objects:
                    # Fallback: if all were overfit, just load whatever is available
                    for base_name in ["xgboost", "lightgbm", "random_forest", "logistic_regression"]:
                        test_name = f"{base_name}_{horizon}d"
                        try:
                            m = _load_model(test_name)
                        except FileNotFoundError:
                            continue
                            
                        if test_name.startswith("logistic_regression"):
                            scaler_path = MODELS_DIR / f"scaler_{horizon}d.joblib"
                            if scaler_path.exists():
                                scaler = joblib.load(scaler_path)
                                f_input = scaler.transform(feature_row)
                            else:
                                f_input = feature_row.values
                        else:
                            f_input = feature_row
                            
                        probs = m.predict_proba(f_input)[0]
                        ensemble_probs.append(probs)
                        model_objects.append({
                            "name": test_name,
                            "model": m,
                            "feature_input": f_input,
                            "probs": probs
                        })
                            
                    if not model_objects:
                        raise FileNotFoundError(f"No models found for horizon {horizon}d to run ensemble.")
                    
                # Calculate Soft Voting Average
                avg_probs = np.mean(ensemble_probs, axis=0)
                winning_class = 1 if avg_probs[1] > 0.5 else 0
                
                # Pick the model with highest confidence in the winning direction for SHAP proxy
                best_proxy_model = max(model_objects, key=lambda x: x["probs"][winning_class])
                
                model = best_proxy_model["model"]
                model_name = best_proxy_model["name"]
                feature_row_input = best_proxy_model["feature_input"]
                probas = avg_probs
                
            else:
                model = _load_model(model_name)
                # Scale if logistic regression
                if model_name.startswith("logistic_regression"):
                    scaler_path = MODELS_DIR / f"scaler_{horizon}d.joblib"
                    if scaler_path.exists():
                        scaler = joblib.load(scaler_path)
                        feature_row_input = scaler.transform(feature_row)
                    else:
                        feature_row_input = feature_row.values
                else:
                    feature_row_input = feature_row
                    
                probas = model.predict_proba(feature_row_input)[0]
                
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

        confidence = float(max(probas))
        
        if confidence < 0.60:
            pred_label = "NEUTRAL"
        else:
            pred_label = "UP" if probas[1] > 0.5 else "DOWN"

        # 3. Calculate SHAP explainability
        try:
            available_cols = [c for c in FEATURE_COLS if c in feature_row.columns]
            shap_impacts = explain(feature_row[available_cols], model_name)
        except Exception:
            shap_impacts = []

        # 4. Persist prediction
        pred_record = Prediction(
            symbol=symbol,
            predicted_at=datetime.now(timezone.utc),
            model_name=model_name,
            prediction=pred_label,
            confidence=confidence,
            shap_explanation=shap_impacts,
        )
        db.add(pred_record)
        db.commit()

        # 5. Log API call
        elapsed_ms = (time.time() - t0) * 1000
        log = APILog(
            endpoint="/api/v1/predict",
            method="POST",
            symbol=symbol,
            status_code=200,
            response_time_ms=elapsed_ms,
        )
        db.add(log)
        db.commit()

        # 6. Build disclaimer — warn if no sentiment data was available
        if not has_sentiment:
            disclaimer = (
                f"⚠️ No recent news found for {symbol}. Prediction is based on "
                "technical indicators only (sentiment = 0.0). Accuracy may be lower. "
                "Not financial advice."
            )
        else:
            disclaimer = (
                "Prediction based on technical indicators + NLP sentiment. "
                "Not financial advice."
            )

        indicators = TechnicalIndicators(
            sma_20=feature_row["SMA_20"].iloc[0] if "SMA_20" in feature_row.columns else None,
            sma_50=feature_row["SMA_50"].iloc[0] if "SMA_50" in feature_row.columns else None,
            rsi=feature_row["RSI"].iloc[0] if "RSI" in feature_row.columns else None,
            macd=feature_row["MACD"].iloc[0] if "MACD" in feature_row.columns else None,
        )

        return PredictResponse(
            symbol=symbol,
            model_name=model_name,
            prediction=pred_label,
            confidence=round(confidence, 4),
            explanation=[FeatureImpact(**i) for i in shap_impacts],
            disclaimer=disclaimer,
            indicators=indicators,
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
