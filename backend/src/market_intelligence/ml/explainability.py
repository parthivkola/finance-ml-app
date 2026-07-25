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
    model_name: str = "xgboost",
    top_n: int = 5,
) -> list[dict]:
    """
    Returns the top-N SHAP feature impacts for a single prediction row.

    Args:
        feature_row: Single-row DataFrame with correct feature columns.
        model_name: One of 'xgboost', 'lightgbm', 'random_forest',
                    'logistic_regression'.
        top_n: Number of top features to return.

    Returns:
        List of {"feature": str, "impact": float} sorted by |impact| desc.
    """
    model = _load_latest(model_name)

    # Choose explainer based on model family (FR-XAI-004, FR-XAI-005)
    tree_models = ("xgboost", "lightgbm", "random_forest")
    if model_name in tree_models:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(feature_row)

        # For binary classifiers shap_values may be a list [neg_class, pos_class]
        if isinstance(shap_vals, list):
            vals = shap_vals[1][0]  # positive class
        else:
            # XGBoost returns a 2D array directly
            vals = shap_vals[0] if shap_vals.ndim == 1 else shap_vals[0]
    else:
        # Logistic Regression → LinearExplainer (FR-XAI-005)
        scaler_path = MODELS_DIR / "scaler.joblib"
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        data = scaler.transform(feature_row) if scaler else feature_row.values
        explainer = shap.LinearExplainer(
            model, masker=shap.maskers.Independent(data)
        )
        shap_vals = explainer.shap_values(data)
        vals = shap_vals[0]

    feature_names = feature_row.columns.tolist()
    impacts = sorted(
        zip(feature_names, vals),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    # Distinguish positive vs negative influences (FR-XAI-003)
    return [
        {"feature": name, "impact": round(float(val), 6)}
        for name, val in impacts[:top_n]
    ]
