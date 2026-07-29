"""
ML training pipeline with:
- Multi-horizon predictions (1-day, 3-day, 5-day)
- Enriched feature set (regime detection, crossover signals, macro context)
- TimeSeriesSplit cross-validation (overfitting detection)
- Walk-forward train/test reporting
- Strong regularization on all models
"""
from __future__ import annotations

import gc
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import ta
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")


MODELS_DIR = Path(__file__).resolve().parents[4] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Minimum rows needed before training makes sense
MIN_ROWS_FOR_TRAINING = 100

# Prediction horizons to train (days into the future)
PREDICTION_HORIZONS = [1, 3, 5]

FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26",
    "RSI", "RSI_slow", "MACD", "MACD_signal", "MACD_diff", "MACD_cross",
    "BB_high", "BB_low", "BB_width", "BB_pct",
    "williams_r", "cci",
    "daily_return", "volatility_20", "vol_5", "vol_ratio",
    "volume_change_pct", "volume_sma_5", "volume_sma_20", "vol_vs_avg",
    "stoch_rsi", "atr", "atr_pct",
    "return_1d", "return_3d", "return_5d", "return_10d", "return_20d", "return_60d",
    # Crossover signals
    "rsi_ob", "rsi_os", "rsi_mid_x_up",
    "price_above_sma20", "price_above_sma50", "price_above_sma200",
    "golden_cross", "bull_regime", "high_vol_regime",
    "macd_cross_up", "macd_cross_down",
    # Range & gap
    "pct_52w_high", "pct_52w_low", "gap", "intraday_range",
    # Calendar
    "day_of_week", "month",
    # Sentiment
    "sentiment_score", "sentiment_3d_avg", "sentiment_7d_avg", "sentiment_trend",
    # Macro
    "spy_return", "qqq_return", "vix_close", "spy_momentum_5d",
]


def build_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Computes enriched technical indicators and creates the binary Target column.
    
    horizon: predict if close N days from now > today's close.
             1 = next-day (noisier), 3 = 3-day trend, 5 = 5-day trend (smoother).
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    close = df["close"]

    # ── Core trend indicators ──────────────────────────────────────────────
    df["SMA_20"]  = ta.trend.sma_indicator(close, window=20)
    df["SMA_50"]  = ta.trend.sma_indicator(close, window=50)
    df["SMA_200"] = ta.trend.sma_indicator(close, window=200)
    df["EMA_12"]  = ta.trend.ema_indicator(close, window=12)
    df["EMA_26"]  = ta.trend.ema_indicator(close, window=26)
    df["RSI"]     = ta.momentum.rsi(close, window=14)
    df["RSI_slow"]= ta.momentum.rsi(close, window=28)

    macd = ta.trend.MACD(close)
    df["MACD"]        = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_diff"]   = macd.macd_diff()
    df["MACD_cross"]  = (df["MACD"] > df["MACD_signal"]).astype(int)
    df["macd_cross_up"]   = ((df["MACD"] > df["MACD_signal"]) &
                             (df["MACD"].shift(1) <= df["MACD_signal"].shift(1))).astype(int)
    df["macd_cross_down"] = ((df["MACD"] < df["MACD_signal"]) &
                             (df["MACD"].shift(1) >= df["MACD_signal"].shift(1))).astype(int)

    bb = ta.volatility.BollingerBands(close)
    df["BB_high"]  = bb.bollinger_hband()
    df["BB_low"]   = bb.bollinger_lband()
    df["BB_width"] = bb.bollinger_wband()
    df["BB_pct"]   = (close - df["BB_low"]) / (df["BB_high"] - df["BB_low"] + 1e-9)

    df["stoch_rsi"]   = ta.momentum.stochrsi(close)
    df["atr"]         = ta.volatility.average_true_range(df["high"], df["low"], close)
    df["atr_pct"]     = df["atr"] / (close + 1e-9)
    df["williams_r"]  = ta.momentum.williams_r(df["high"], df["low"], close)
    df["cci"]         = ta.trend.cci(df["high"], df["low"], close)

    # ── Returns & volatility ───────────────────────────────────────────────
    df["daily_return"]  = close.pct_change()
    df["return_1d"]     = df["daily_return"].shift(1)
    df["return_3d"]     = close.pct_change(3)
    df["return_5d"]     = close.pct_change(5)
    df["return_10d"]    = close.pct_change(10)
    df["return_20d"]    = close.pct_change(20)
    df["return_60d"]    = close.pct_change(60)
    df["volatility_20"] = df["daily_return"].rolling(20).std()
    df["vol_5"]         = df["daily_return"].rolling(5).std()
    df["vol_ratio"]     = df["vol_5"] / (df["volatility_20"] + 1e-9)

    df["volume_change_pct"] = df["volume"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["volume_sma_5"]  = ta.trend.sma_indicator(df["volume"].astype(float), window=5)
    df["volume_sma_20"] = ta.trend.sma_indicator(df["volume"].astype(float), window=20)
    df["vol_vs_avg"]    = df["volume"] / (df["volume_sma_20"] + 1)

    # ── Crossover & regime signals ─────────────────────────────────────────
    rsi = df["RSI"]
    df["rsi_ob"]       = (rsi > 70).astype(int)
    df["rsi_os"]       = (rsi < 30).astype(int)
    df["rsi_mid_x_up"] = ((rsi > 50) & (rsi.shift(1) <= 50)).astype(int)

    df["price_above_sma20"]  = (close > df["SMA_20"]).astype(int)
    df["price_above_sma50"]  = (close > df["SMA_50"]).astype(int)
    df["price_above_sma200"] = (close > df["SMA_200"]).astype(int)
    df["golden_cross"]       = (df["SMA_50"] > df["SMA_200"]).astype(int)
    df["bull_regime"]        = ((close > df["SMA_200"]) & (df["SMA_50"] > df["SMA_200"])).astype(int)
    vol_60 = df["daily_return"].rolling(60).std()
    df["high_vol_regime"] = (df["volatility_20"] > 1.5 * vol_60).astype(int)

    # ── Price position & gap ───────────────────────────────────────────────
    df["pct_52w_high"]   = close / close.rolling(252).max() - 1
    df["pct_52w_low"]    = close / close.rolling(252).min() - 1
    df["gap"]            = (df["open"] - close.shift(1)) / (close.shift(1) + 1e-9)
    df["intraday_range"] = (df["high"] - df["low"]) / (df["open"] + 1e-9)

    # ── Calendar ───────────────────────────────────────────────────────────
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"]       = pd.to_datetime(df["date"]).dt.month

    # ── Sentiment ──────────────────────────────────────────────────────────
    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = 0.0
    df["sentiment_score"]  = df["sentiment_score"].fillna(0.0)
    df["sentiment_3d_avg"] = df["sentiment_score"].rolling(3, min_periods=1).mean()
    df["sentiment_7d_avg"] = df["sentiment_score"].rolling(7, min_periods=1).mean()
    df["sentiment_trend"]  = df["sentiment_score"] - df["sentiment_7d_avg"]

    # ── Macro context ──────────────────────────────────────────────────────
    for col in ["spy_return", "qqq_return", "vix_close"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)
    if "spy_return" in df.columns:
        df["spy_momentum_5d"] = df["spy_return"].rolling(5, min_periods=1).sum()
    else:
        df["spy_momentum_5d"] = 0.0

    # ── Target: will price be higher N days from now? ─────────────────────
    df["target"] = (close.shift(-horizon) > close).astype(int)
    df = df.replace([np.inf, -np.inf], np.nan)
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
    train_accuracy: float,
    overfit_status: str,
    f1: float,
    roc_auc: float,
    total_price_rows: int = 0,
) -> None:
    import os
    from market_intelligence.db.session import SessionLocal
    from market_intelligence.db.models import ModelRegistry

    db = SessionLocal()
    try:
        # Save new record
        record = ModelRegistry(
            model_name=model_name,
            version=version,
            trained_at=datetime.now(timezone.utc),
            accuracy=accuracy,
            train_accuracy=train_accuracy,
            overfit_status=overfit_status,
            f1_score=f1,
            roc_auc=roc_auc,
            artifact_path=str(artifact_path),
            total_price_rows=total_price_rows,
        )
        db.add(record)
        
        # Cleanup old versions to save memory/disk space
        old_records = db.query(ModelRegistry).filter(
            ModelRegistry.model_name == model_name,
            ModelRegistry.version != version
        ).all()
        for old in old_records:
            if os.path.exists(old.artifact_path):
                try:
                    os.remove(old.artifact_path)
                except Exception as e:
                    print(f"Warning: could not delete {old.artifact_path}: {e}")
            db.delete(old)
            
        db.commit()
    finally:
        db.close()


def train_all(raw_df: pd.DataFrame) -> dict[str, dict]:
    """
    Train all 4 models with CV-based overfitting detection, for each prediction horizon.

    IMPORTANT: features must be built PER SYMBOL so that technical indicators
    (SMA, RSI, MACD, Bollinger) are never computed across symbol boundaries.
    """
    all_results = {}
    version = datetime.now(timezone.utc).strftime("v%Y%m%d")

    for horizon in PREDICTION_HORIZONS:
        print(f"\n" + "="*80)
        print(f"🚀 TRAINING HORIZON: {horizon}-DAY FORECAST")
        print("="*80)

        # ── Build features per symbol then concatenate ─────────────────────────
        if "symbol" in raw_df.columns:
            parts = []
            for sym, grp in raw_df.groupby("symbol"):
                sym_df = build_features(grp.copy(), horizon=horizon)
                if len(sym_df) >= 60:
                    parts.append(sym_df)
            if not parts:
                print(f"⚠️  No symbol had enough rows for {horizon}d. Skipping.")
                continue
            df = pd.concat(parts, ignore_index=True)
            df = df.sample(frac=1, random_state=42)
        else:
            df = build_features(raw_df, horizon=horizon)

        if len(df) < MIN_ROWS_FOR_TRAINING:
            print(f"⚠️  Only {len(df)} rows after feature build. Skipping {horizon}d.")
            continue

        available_cols = [c for c in FEATURE_COLS if c in df.columns]
        X = df[available_cols]
        y = df["target"]

        # Chronological 80/20 split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"📊 Dataset: {len(df)} rows | Train: {len(X_train)} | Test: {len(X_test)}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        joblib.dump(scaler, MODELS_DIR / f"scaler_{horizon}d.joblib")

        print("🔍 Running Optuna hyperparameter tuning (10 trials per model)...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # ── XGBoost Tuning ──
        def optimize_xgb(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 150),
                "max_depth": trial.suggest_int("max_depth", 2, 4),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 5, 20),
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
        best_xgb_params["objective"] = "binary:logistic"
        print(f"✅ XGBoost best CV accuracy: {xgb_study.best_value:.4f}")
        # Free Optuna trial objects from RAM
        del xgb_study
        gc.collect()

        # ── LightGBM Tuning ──
        def optimize_lgbm(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 150),
                "max_depth": trial.suggest_int("max_depth", 2, 4),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 50),
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
        # Free Optuna trial objects from RAM
        del lgbm_study
        gc.collect()

        base_classifiers = {
            f"logistic_regression_{horizon}d": LogisticRegression(
                C=0.1, max_iter=2000, solver="saga", random_state=42
            ),
            f"random_forest_{horizon}d": RandomForestClassifier(
                # Reduced from 300→100 trees to save ~70MB RAM on t3.micro
                n_estimators=100, max_depth=5, min_samples_leaf=15,
                min_samples_split=30, max_features="sqrt", random_state=42, n_jobs=1,
            ),
            f"xgboost_{horizon}d": xgb.XGBClassifier(**best_xgb_params),
            f"lightgbm_{horizon}d": lgb.LGBMClassifier(**best_lgbm_params),
            # --- 3 new diverse classifiers ---
            # AdaBoost with decision stumps: focuses on hard-to-classify samples,
            # orthogonal to gradient boosting (which focuses on residuals, not sample weights)
            f"adaboost_{horizon}d": AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1),
                n_estimators=50, learning_rate=0.5, random_state=42, algorithm="SAMME"
            ),
            # LinearSVC: finds a max-margin linear boundary in high-dim space.
            # Use LinearSVC (not RBF-SVM) to avoid computing an N×N kernel matrix.
            # Wrap in CalibratedClassifierCV to get probabilities from the SVM.
            f"linear_svc_{horizon}d": CalibratedClassifierCV(
                estimator=LinearSVC(C=0.1, max_iter=2000, random_state=42),
                method="sigmoid", cv=3
            ),
            # GaussianNB: pure statistical prior — extremely low variance,
            # catches regime changes when all other models are confused
            f"gaussian_nb_{horizon}d": GaussianNB(),
        }

        # Calibrate probabilities (Platt Scaling) for models that need it.
        # XGBoost + LightGBM: already calibrated via logloss training objective.
        # AdaBoost: wrap to get reliable probabilities.
        # LinearSVC: already wrapped above.
        # GaussianNB: naturally outputs posteriors.
        classifiers = {}
        for name, clf in base_classifiers.items():
            if name.startswith("logistic_regression") or name.startswith("adaboost"):
                classifiers[name] = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv=3)
            else:
                classifiers[name] = clf

        for name, clf in classifiers.items():
            print(f"\n>>> Training {name}...")
            
            # Logistic Regression needs scaled features
            if name.startswith("logistic_regression"):
                X_tr, X_te = X_train_scaled, X_test_scaled
            else:
                X_tr, X_te = X_train, X_test

            if name.startswith("xgboost") or name.startswith("lightgbm"):
                # Pass eval_set to the base estimator via fit_params
                clf.fit(X_tr, y_train)
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

            cv_scores = {}
            if name.startswith("logistic_regression") or name.startswith("random_forest"):
                X_cv = X_train_scaled if name.startswith("logistic_regression") else X_train
                cv_scores = _cv_score(clf, X_cv, y_train)
                if cv_scores["cv_test_acc"] is not None:
                    print(f"TimeSeriesCV: train={cv_scores['cv_train_acc']:.3f} | "
                          f"test={cv_scores['cv_test_acc']:.3f} | gap={cv_scores['cv_gap']:.3f}")

            path = MODELS_DIR / f"{name}_{version}.joblib"
            joblib.dump(clf, path)

            _save_to_registry(
                name, version, path, acc, train_acc, overfit_status, f1, roc, total_price_rows=len(df)
            )

            all_results[name] = {
                "version": version,
                "horizon": horizon,
                "accuracy": round(acc, 4),
                "train_accuracy": round(train_acc, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(roc, 4),
                "overfit_status": overfit_status,
                "artifact_path": str(path),
                "feature_cols": available_cols,
                **cv_scores,
            }

        # --- Evaluate Auto Ensemble (7-Model Accuracy-Weighted Hard Voting) ---
        print(f"\n>>> Evaluating Auto Ensemble (7-Model Weighted Consensus) for {horizon}d...")
        
        # Collect non-overfit models and their test accuracies as weights
        valid_models = []
        model_weights = []
        for name, r in all_results.items():
            if name.endswith(f"_{horizon}d") and "OVERFIT" not in r["overfit_status"]:
                valid_models.append(name)
                model_weights.append(r["accuracy"])  # use test accuracy as vote weight

        if valid_models:
            # --- Vectorized batch prediction (no Python for-loop) ---
            # For each model, predict all X_test rows at once to minimize RAM churn
            all_model_probs = []  # shape: (n_models, n_test_samples, 2)
            for name, weight in zip(valid_models, model_weights):
                clf = classifiers[name]
                needs_scale = name.startswith("logistic_regression") or name.startswith("linear_svc") or name.startswith("gaussian_nb")
                X_input = X_test_scaled if needs_scale else X_test.values
                probs = clf.predict_proba(X_input)  # shape: (n_test_samples, 2)
                all_model_probs.append(probs)

            # Stack to (n_models, n_samples, 2)
            prob_matrix = np.array(all_model_probs)  # (n_models, n_samples, 2)
            weights_arr = np.array(model_weights)     # (n_models,)

            # --- Accuracy-Weighted Soft Voting ---
            # Each model's probability is scaled by its test accuracy weight,
            # then averaged. This respects both the direction AND magnitude of conviction.
            # weights_arr shape: (n_models,), prob_matrix shape: (n_models, n_samples, 2)
            weighted_probs = (prob_matrix * weights_arr[:, None, None]).sum(axis=0) / weights_arr.sum()
            # weighted_probs shape: (n_samples, 2)

            # Winning class is whichever averaged probability is higher
            winning_classes = (weighted_probs[:, 1] > 0.5).astype(int)

            # Confidence = the weighted average probability in the winning direction
            # This is naturally consistent with soft voting: high when all models agree strongly
            ensemble_preds = winning_classes.tolist()
            ensemble_probas = weighted_probs[np.arange(len(winning_classes)), winning_classes].tolist()

            ens_acc = accuracy_score(y_test, ensemble_preds)
            ens_f1 = f1_score(y_test, ensemble_preds, zero_division=0)
            ens_roc = roc_auc_score(y_test, ensemble_probas)
            n_models_used = len(valid_models)

            print(f"Auto Ensemble ({n_models_used} models): Test Acc={ens_acc:.3f} | F1={ens_f1:.3f} | ROC-AUC={ens_roc:.3f}")
            
            auto_name = f"auto_{horizon}d"
            _save_to_registry(
                auto_name, version, Path("ensemble_virtual"), ens_acc, ens_acc, "DYNAMIC", ens_f1, ens_roc, total_price_rows=len(df)
            )
            all_results[auto_name] = {
                "version": version,
                "horizon": horizon,
                "accuracy": round(ens_acc, 4),
                "train_accuracy": round(ens_acc, 4),
                "f1_score": round(ens_f1, 4),
                "roc_auc": round(ens_roc, 4),
                "overfit_status": "DYNAMIC",
                "artifact_path": "ensemble_virtual",
                "feature_cols": available_cols,
            }

            # Free the large probability matrix from RAM immediately
            del prob_matrix, all_model_probs, binary_votes
            gc.collect()

        (MODELS_DIR / f"feature_cols_{horizon}d.json").write_text(json.dumps(available_cols))
        
    print("\n✅ All models trained and saved to", MODELS_DIR)
    return all_results


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
