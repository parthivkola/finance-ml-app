"""
SHAP Explainability — satisfies FR-XAI-001 to FR-XAI-005.

Uses TreeExplainer for XGBoost / LightGBM / RandomForest
and LinearExplainer for LogisticRegression.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import shap

MODELS_DIR = Path(__file__).resolve().parents[4] / "models"


def _load_latest(model_name: str):
    """Loads the most recently saved artifact for a given model name."""
    candidates = sorted(MODELS_DIR.glob(f"{model_name}_v*.joblib"), reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No trained artifact found for '{model_name}' in {MODELS_DIR}. "
            "Run trainer.py first."
        )
    return joblib.load(candidates[0])


def explain(
    feature_row: pd.DataFrame,
    model_name: str = "xgboost_1d",
    top_n: int = 10,
) -> list[dict]:
    """
    Returns the top-N SHAP feature impacts for a single prediction row.

    Args:
        feature_row: Single-row DataFrame with correct feature columns.
        model_name: e.g. 'xgboost_1d', 'lightgbm_3d', 'random_forest_5d',
                    'logistic_regression_1d'.
        top_n: Number of top features to return.

    Returns:
        List of {"feature": str, "impact": float} sorted by |impact| desc.
    """
    import numpy as np
    model = _load_latest(model_name)

    # Extract horizon for scaler path (e.g. "xgboost_3d" -> 3)
    try:
        horizon = int(model_name.split("_")[-1].replace("d", ""))
    except Exception:
        horizon = 1

    is_tree = model_name.startswith(("xgboost", "lightgbm", "random_forest"))
    is_rf = model_name.startswith("random_forest")

    if is_tree:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(feature_row)

        if is_rf:
            # sklearn RF returns list of [neg_class_array, pos_class_array]
            # each array is shape (n_samples, n_features)
            if isinstance(shap_vals, list) and len(shap_vals) >= 2:
                vals = np.array(shap_vals[1][0])  # positive class, first sample
            elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                vals = shap_vals[0, :, 1]  # shape (n_samples, n_features, n_classes)
            else:
                vals = np.array(shap_vals).flatten()
        else:
            # XGBoost / LightGBM: returns 2D array (n_samples, n_features)
            if isinstance(shap_vals, list):
                vals = shap_vals[1][0]
            else:
                vals = shap_vals[0] if shap_vals.ndim == 1 else shap_vals[0]
    else:
        # Logistic Regression → LinearExplainer with proper background
        scaler_path = MODELS_DIR / f"scaler_{horizon}d.joblib"
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        data = scaler.transform(feature_row) if scaler else feature_row.values
        # Use the feature row as background so we get relative impacts, not zeros
        background = np.zeros_like(data)  # zero baseline = neutral baseline
        explainer = shap.LinearExplainer(
            model, masker=shap.maskers.Independent(background)
        )
        shap_vals = explainer.shap_values(data)
        vals = np.array(shap_vals[0]) if isinstance(shap_vals, list) else np.array(shap_vals).flatten()

    vals = np.array(vals).flatten()
    feature_names = feature_row.columns.tolist()

    # Guard against length mismatch
    min_len = min(len(feature_names), len(vals))
    impacts = sorted(
        zip(feature_names[:min_len], vals[:min_len]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    return [
        {"feature": name, "impact": round(float(val), 6)}
        for name, val in impacts[:top_n]
        if val != 0.0  # filter out zero-impact features for clarity
    ]
