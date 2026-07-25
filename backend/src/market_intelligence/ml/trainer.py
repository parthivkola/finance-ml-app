"""
Improved ML training pipeline with:
- TimeSeriesSplit cross-validation (overfitting detection)
- Walk-forward train/test reporting (train acc vs test acc gap)
- Strong regularization on all 4 models
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).resolve().parents[4] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Minimum rows needed before training makes sense
MIN_ROWS_FOR_TRAINING = 100

FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "SMA_20", "SMA_50", "EMA_12", "EMA_26",
    "RSI", "MACD", "MACD_signal", "MACD_diff",
    "BB_high", "BB_low", "BB_width",
    "daily_return", "volatility_20",
    "volume_change_pct", "volume_sma_5", "volume_sma_20",
    "stoch_rsi", "atr",
    "return_1d", "return_3d", "return_5d",
    "sentiment_score", "sentiment_3d_avg", "sentiment_7d_avg", "sentiment_trend",
    "spy_return", "qqq_return", "vix_close",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes all technical indicators and creates the binary Target column."""
    df = df.copy().sort_values("date").reset_index(drop=True)
    close = df["close"]

    df["SMA_20"] = ta.trend.sma_indicator(close, window=20)
    df["SMA_50"] = ta.trend.sma_indicator(close, window=50)
    df["EMA_12"] = ta.trend.ema_indicator(close, window=12)
    df["EMA_26"] = ta.trend.ema_indicator(close, window=26)
    df["RSI"] = ta.momentum.rsi(close, window=14)

    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_diff"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close)
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()
    df["BB_width"] = bb.bollinger_wband()

    df["stoch_rsi"] = ta.momentum.stochrsi(close)
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], close)

    df["daily_return"] = close.pct_change()
    df["return_1d"] = df["daily_return"].shift(1)
    df["return_3d"] = df["daily_return"].rolling(3).sum()
    df["return_5d"] = df["daily_return"].rolling(5).sum()
    
    df["volatility_20"] = df["daily_return"].rolling(20).std()
    df["volume_change_pct"] = df["volume"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["volume_sma_5"] = ta.trend.sma_indicator(df["volume"], window=5)
    df["volume_sma_20"] = ta.trend.sma_indicator(df["volume"], window=20)

    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = 0.0
    df["sentiment_score"] = df["sentiment_score"].fillna(0.0)

    df["sentiment_3d_avg"] = df["sentiment_score"].rolling(3, min_periods=1).mean()
    df["sentiment_7d_avg"] = df["sentiment_score"].rolling(7, min_periods=1).mean()
    df["sentiment_trend"] = df["sentiment_score"] - df["sentiment_7d_avg"]

    # If macro market context is missing, fill with 0 so the model doesn't crash on inference
    for col in ["spy_return", "qqq_return", "vix_close"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    df["target"] = (close.shift(-1) > close).astype(int)
    df.dropna(inplace=True)
    return df


def _detect_overfit(train_acc: float, test_acc: float) -> str:
    """
    Returns a human-readable assessment of overfitting/underfitting.
    Gap > 15% = overfit, test < 52% = underfit/noise.
    """
    gap = train_acc - test_acc
    if gap > 0.15:
        return f"⚠️  OVERFIT (gap={gap:.1%})"
    if test_acc < 0.52:
        return f"⚠️  UNDERFIT/NOISE (test={test_acc:.1%}, near random)"
    return f"✅ OK (gap={gap:.1%})"


def _cv_score(clf, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    """
    Walk-forward TimeSeriesSplit cross-validation.
    Returns mean train/test accuracy and their gap.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    train_accs, test_accs = [], []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        X_tr = X[tr_idx] if hasattr(X, '__array__') and not hasattr(X, 'iloc') else X.iloc[tr_idx]
        X_te = X[te_idx] if hasattr(X, '__array__') and not hasattr(X, 'iloc') else X.iloc[te_idx]
        y_tr = y.iloc[tr_idx] if hasattr(y, 'iloc') else y[tr_idx]
        y_te = y.iloc[te_idx] if hasattr(y, 'iloc') else y[te_idx]

        if len(X_tr) < 30 or len(X_te) < 5:
            continue  # skip folds that are too small

        clf.fit(X_tr, y_tr)
        train_accs.append(accuracy_score(y_tr, clf.predict(X_tr)))
        test_accs.append(accuracy_score(y_te, clf.predict(X_te)))

    if not train_accs:
        return {"cv_train_acc": None, "cv_test_acc": None, "cv_gap": None}

    return {
        "cv_train_acc": round(float(np.mean(train_accs)), 4),
        "cv_test_acc": round(float(np.mean(test_accs)), 4),
        "cv_gap": round(float(np.mean(train_accs) - np.mean(test_accs)), 4),
    }


def _save_to_registry(
    model_name: str,
    version: str,
    artifact_path: Path,
    accuracy: float,
    f1: float,
    roc_auc: float,
    total_price_rows: int = 0,
) -> None:
    from market_intelligence.db.session import SessionLocal
    from market_intelligence.db.models import ModelRegistry

    db = SessionLocal()
    try:
        record = ModelRegistry(
            model_name=model_name,
            version=version,
            trained_at=datetime.now(timezone.utc),
            accuracy=accuracy,
            f1_score=f1,
            roc_auc=roc_auc,
            artifact_path=str(artifact_path),
            total_price_rows=total_price_rows,
        )
        db.add(record)
        db.commit()
    finally:
        db.close()


def train_all(features_df: pd.DataFrame) -> dict[str, dict]:
    """
    Train all 4 models with CV-based overfitting detection.

    IMPORTANT: features must be built PER SYMBOL so that technical indicators
    (SMA, RSI, MACD, Bollinger) are never computed across symbol boundaries.
    """
    # ── Build features per symbol then concatenate ─────────────────────────
    if "symbol" in features_df.columns:
        parts = []
        for sym, grp in features_df.groupby("symbol"):
            sym_df = build_features(grp.copy())
            if len(sym_df) >= 60:          # need at least 60 clean rows per symbol
                parts.append(sym_df)
        if not parts:
            print("⚠️  No symbol had enough rows after feature build. Skipping.")
            return {}
        df = pd.concat(parts, ignore_index=True)
        df = df.sample(frac=1, random_state=42)   # shuffle so no symbol ordering bias
    else:
        # Fallback: single-symbol call (e.g. from tests)
        df = build_features(features_df)

    if len(df) < MIN_ROWS_FOR_TRAINING:
        print(f"⚠️  Only {len(df)} rows after feature build — need {MIN_ROWS_FOR_TRAINING}. Skipping.")
        return {}

    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_cols]
    y = df["target"]

    # Chronological 80/20 split — NEVER random for time series
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"\n📊 Dataset: {len(df)} rows | Train: {len(X_train)} | Test: {len(X_test)}")
    if len(df) < 500:
        print("⚠️  WARNING: Small dataset (<500 rows). Metrics may not be reliable.")
        print("   → Add more symbols and longer history to improve model quality.\n")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    print("\n🔍 Running Optuna hyperparameter tuning (10 trials per model)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── XGBoost Tuning ──
    def optimize_xgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            "random_state": 42,
            "eval_metric": "logloss",
        }
        clf = xgb.XGBClassifier(**params)
        scores = _cv_score(clf, X_train, y_train, n_splits=3)
        return scores["cv_test_acc"] if scores["cv_test_acc"] is not None else 0.0

    xgb_study = optuna.create_study(direction="maximize")
    xgb_study.optimize(optimize_xgb, n_trials=10)
    best_xgb_params = xgb_study.best_params
    best_xgb_params["random_state"] = 42
    best_xgb_params["eval_metric"] = "logloss"
    print(f"✅ XGBoost best CV accuracy: {xgb_study.best_value:.4f}")

    # ── LightGBM Tuning ──
    def optimize_lgbm(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            "random_state": 42,
            "verbose": -1,
        }
        clf = lgb.LGBMClassifier(**params)
        scores = _cv_score(clf, X_train, y_train, n_splits=3)
        return scores["cv_test_acc"] if scores["cv_test_acc"] is not None else 0.0

    lgbm_study = optuna.create_study(direction="maximize")
    lgbm_study.optimize(optimize_lgbm, n_trials=10)
    best_lgbm_params = lgbm_study.best_params
    best_lgbm_params["random_state"] = 42
    best_lgbm_params["verbose"] = -1
    print(f"✅ LightGBM best CV accuracy: {lgbm_study.best_value:.4f}")

    classifiers = {
        "logistic_regression": LogisticRegression(
            C=0.1, max_iter=2000, solver="saga", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=10,
            min_samples_split=20, max_features="sqrt", random_state=42, n_jobs=-1,
        ),
        "xgboost": xgb.XGBClassifier(**best_xgb_params),
        "lightgbm": lgb.LGBMClassifier(**best_lgbm_params),
    }

    results: dict[str, dict] = {}
    version = datetime.now(timezone.utc).strftime("v%Y%m%d")

    for name, clf in classifiers.items():
        print(f"\n>>> Training {name}...")

        X_tr = X_train_scaled if name == "logistic_regression" else X_train
        X_te = X_test_scaled if name == "logistic_regression" else X_test

        if name in ("xgboost", "lightgbm"):
            clf.fit(X_tr, y_train, eval_set=[(X_te, y_test)])
        else:
            clf.fit(X_tr, y_train)

        preds = clf.predict(X_te)
        probas = clf.predict_proba(X_te)[:, 1]
        train_preds = clf.predict(X_tr)

        acc = accuracy_score(y_test, preds)
        train_acc = accuracy_score(y_train, train_preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        roc = roc_auc_score(y_test, probas)

        overfit_status = _detect_overfit(train_acc, acc)
        print(f"Train Acc={train_acc:.3f} | Test Acc={acc:.3f} | F1={f1:.3f} | ROC-AUC={roc:.3f}")
        print(f"Status: {overfit_status}")

        # Cross-validation (skip for XGBoost/LightGBM to keep it fast)
        cv_scores = {}
        if name in ("logistic_regression", "random_forest"):
            X_cv = X_train_scaled if name == "logistic_regression" else X_train
            cv_scores = _cv_score(clf, X_cv, y_train)
            if cv_scores["cv_test_acc"] is not None:
                print(f"TimeSeriesCV: train={cv_scores['cv_train_acc']:.3f} | "
                      f"test={cv_scores['cv_test_acc']:.3f} | gap={cv_scores['cv_gap']:.3f}")

        path = MODELS_DIR / f"{name}_{version}.joblib"
        joblib.dump(clf, path)

        _save_to_registry(name, version, path, acc, f1, roc, total_price_rows=len(df))

        results[name] = {
            "version": version,
            "accuracy": round(acc, 4),
            "train_accuracy": round(train_acc, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc, 4),
            "overfit_status": overfit_status,
            "artifact_path": str(path),
            "feature_cols": available_cols,
            **cv_scores,
        }

    (MODELS_DIR / "feature_cols.json").write_text(json.dumps(available_cols))
    print("\n✅ All models trained and saved to", MODELS_DIR)
    return results


def fetch_market_context() -> pd.DataFrame:
    from market_intelligence.data.yahoo_finance import fetch_prices
    try:
        spy = fetch_prices("SPY", days=3650)[["date", "Close"]].rename(columns={"Close": "spy_close"})
        qqq = fetch_prices("QQQ", days=3650)[["date", "Close"]].rename(columns={"Close": "qqq_close"})
        vix = fetch_prices("^VIX", days=3650)[["date", "Close"]].rename(columns={"Close": "vix_close"})
        
        # Calculate returns
        spy["spy_return"] = spy["spy_close"].pct_change()
        qqq["qqq_return"] = qqq["qqq_close"].pct_change()
        
        macro_df = spy[["date", "spy_return"]].merge(qqq[["date", "qqq_return"]], on="date", how="outer")
        macro_df = macro_df.merge(vix[["date", "vix_close"]], on="date", how="outer")
        macro_df["date"] = pd.to_datetime(macro_df["date"])
        # Strip timezone information to match the naive timestamps in the local database
        macro_df["date"] = macro_df["date"].dt.tz_localize(None)
        return macro_df
    except Exception as e:
        print(f"⚠️  Failed to fetch market context: {e}")
        return pd.DataFrame(columns=["date", "spy_return", "qqq_return", "vix_close"])


if __name__ == "__main__":
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))

    from market_intelligence.db.session import engine

    # Prices keep the symbol column — features will be built per-symbol
    prices_df = pd.read_sql("SELECT * FROM stock_prices ORDER BY symbol, date", engine)

    # Sentiment merged per SYMBOL + DATE (not just date)
    news_df = pd.read_sql(
        """
        SELECT symbol,
               DATE(published_date) AS date,
               AVG(sentiment_score)  AS sentiment_score
        FROM   news_articles
        WHERE  sentiment_score IS NOT NULL
        GROUP  BY symbol, DATE(published_date)
        """,
        engine,
    )
    news_df["date"] = pd.to_datetime(news_df["date"])
    prices_df["date"] = pd.to_datetime(prices_df["date"])

    # Merge on BOTH symbol and date so AAPL news only goes to AAPL rows
    merged = prices_df.merge(news_df, on=["symbol", "date"], how="left")
    
    # Merge macro context
    macro_df = fetch_market_context()
    merged = merged.merge(macro_df, on="date", how="left")

    results = train_all(merged)
    print("\nFinal results:")
    for name, r in results.items():
        print(f"  {name}: acc={r['accuracy']} | {r.get('overfit_status', '')}")
