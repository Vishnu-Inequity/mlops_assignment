"""
Explainability Module: Global Feature Importance, SHAP Summary, and Local Instance Explanations.
Computes TreeSHAP for tree ensembles and coefficient attribution for linear models.
Generates artifact plots logged to MLflow and provides local explanation APIs.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb

from src.data.loader import load_config
from src.features.pipeline import transform_data, get_transformed_feature_names
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def compute_global_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Computes global feature importance:
    - LogisticRegression: absolute standardized coefficient magnitude.
    - XGBoost / Tree models: gain / weight or mean absolute TreeSHAP.
    """
    if hasattr(model, "coef_"):
        # Linear model coefficients
        coefs = np.abs(model.coef_[0])
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": coefs,
            "metric": "absolute_coefficient"
        })
    elif isinstance(model, xgb.Booster):
        score_dict = model.get_score(importance_type="gain")
        scores = []
        for i, f in enumerate(feature_names):
            # XGBoost default feature keys are 'f0', 'f1', etc.
            f_key = f"f{i}"
            scores.append(score_dict.get(f_key, 0.0))
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": scores,
            "metric": "gain"
        })
    else:
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": np.zeros(len(feature_names)),
            "metric": "uniform"
        })

    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return importance_df

def generate_shap_summary(model: xgb.Booster, X_sample: np.ndarray, feature_names: list, output_path: str = "reports/plots/shap_summary.png", top_n: int = 20) -> dict:
    """
    Computes exact TreeSHAP values for XGBoost on evaluation sample.
    Generates and saves a publication-quality global SHAP summary plot.
    """
    dmat = xgb.DMatrix(X_sample)
    # pred_contribs=True returns (N, num_features + 1), where last column is bias
    contribs = model.predict(dmat, pred_contribs=True)
    shap_values = contribs[:, :-1]
    base_value = float(np.mean(contribs[:, -1]))

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]

    top_features = [feature_names[i] for i in top_indices]
    top_importance = mean_abs_shap[top_indices]

    # Plot horizontal bar chart of mean |SHAP value|
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    y_pos = np.arange(len(top_features))[::-1]
    bars = ax.barh(y_pos, top_importance, color="#3B82F6", edgecolor="#1D4ED8", alpha=0.85, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_xlabel("Mean |SHAP Value| (Average Impact on Log-Odds Risk)", fontsize=10)
    ax.set_title("Global Feature Importance (TreeSHAP Summary)", fontsize=12, pad=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)

    return {
        "plot_path": str(out_file),
        "base_value": base_value,
        "top_features": [
            {"feature": f, "mean_abs_shap": round(float(imp), 4)}
            for f, imp in zip(top_features, top_importance)
        ]
    }

def explain_instance(model, X_single: np.ndarray, feature_names: list, top_k: int = 5) -> dict:
    """
    Computes local feature attribution for a single patient record.
    Returns the top K contributing features, their impact magnitude, and direction.
    """
    if len(X_single.shape) == 1:
        X_single = X_single.reshape(1, -1)

    if isinstance(model, xgb.Booster):
        dmat = xgb.DMatrix(X_single)
        contribs = model.predict(dmat, pred_contribs=True)[0]
        shap_vals = contribs[:-1]
        base_value = float(contribs[-1])
    elif hasattr(model, "coef_"):
        # Linear approximation: w_i * x_i
        shap_vals = (model.coef_[0] * X_single[0])
        base_value = float(model.intercept_[0]) if hasattr(model, "intercept_") else 0.0
    else:
        shap_vals = np.zeros(len(feature_names))
        base_value = 0.0

    # Sort by absolute magnitude of contribution
    abs_indices = np.argsort(np.abs(shap_vals))[::-1][:top_k]

    top_explanations = []
    for idx in abs_indices:
        val = float(shap_vals[idx])
        top_explanations.append({
            "feature": feature_names[idx],
            "shap_value": round(val, 4),
            "direction": "increases_risk" if val > 0 else "decreases_risk",
            "feature_value_scaled": round(float(X_single[0, idx]), 4)
        })

    return {
        "base_value": round(base_value, 4),
        "top_features": top_explanations
    }

def run_explainability_pipeline(config_path: str = "configs/config.yaml"):
    """
    Executes Phase 3 explainability workflow:
    1. Loads best model, preprocessor, and validation data.
    2. Computes global importance and SHAP summary plot.
    3. Demonstrates local explanation on sample instances.
    4. Logs artifacts to MLflow.
    """
    config = load_config(config_path)
    best_artifact = joblib.load("models/best_model.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")
    val_df = pd.read_parquet(config["data"]["val_data_path"])

    feature_names = get_transformed_feature_names(preprocessor, config)
    X_val, _ = transform_data(val_df, preprocessor, config)

    # 1. Global Feature Importance
    model = best_artifact["model"]
    model_name = best_artifact["model_name"]
    logger.info(f"Computing global feature importance for {model_name}...")
    importance_df = compute_global_feature_importance(model, feature_names)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    importance_df.head(25).to_json(reports_dir / "global_feature_importance.json", orient="records", indent=2)

    # 2. SHAP Global Summary Plot
    logger.info("Computing TreeSHAP summary on validation cohort...")
    # Use 1000 representative validation samples for SHAP summary
    sample_indices = np.random.RandomState(42).choice(len(X_val), size=min(1000, len(X_val)), replace=False)
    X_sample = X_val[sample_indices]

    shap_results = generate_shap_summary(
        model, X_sample, feature_names,
        output_path="reports/plots/shap_summary.png",
        top_n=20
    )

    # 3. Local Explanation Demonstration
    sample_patient = X_val[0:1]
    sample_explanation = explain_instance(model, sample_patient, feature_names, top_k=5)
    logger.info(f"Sample local explanation -> Base Value: {sample_explanation['base_value']}")
    for feat in sample_explanation["top_features"]:
        logger.info(f"  * {feat['feature']}: {feat['shap_value']:+.4f} ({feat['direction']})")

    # Save local sample to reports
    with open(reports_dir / "local_explanation_sample.json", "w", encoding="utf-8") as f:
        json.dump(sample_explanation, f, indent=2)

    # 4. Log to MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    run_id = best_artifact.get("run_id")
    try:
        if run_id:
            logger.info(f"Logging explainability artifacts to MLflow run {run_id}...")
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(shap_results["plot_path"], artifact_path="explainability")
                mlflow.log_artifact(str(reports_dir / "global_feature_importance.json"), artifact_path="explainability")
                mlflow.log_artifact(str(reports_dir / "local_explanation_sample.json"), artifact_path="explainability")
        else:
            with mlflow.start_run(run_name="TreeSHAP_Explainability"):
                mlflow.log_artifact(shap_results["plot_path"], artifact_path="explainability")
                mlflow.log_artifact(str(reports_dir / "global_feature_importance.json"), artifact_path="explainability")
                mlflow.log_artifact(str(reports_dir / "local_explanation_sample.json"), artifact_path="explainability")
    except Exception as e:
        logger.warning(f"Could not log to existing run {run_id} ({e}); starting a new MLflow run for explainability.")
        with mlflow.start_run(run_name="TreeSHAP_Explainability"):
            mlflow.log_artifact(shap_results["plot_path"], artifact_path="explainability")
            mlflow.log_artifact(str(reports_dir / "global_feature_importance.json"), artifact_path="explainability")
            mlflow.log_artifact(str(reports_dir / "local_explanation_sample.json"), artifact_path="explainability")

    logger.info("Phase 3 Explainability completed successfully.")
    return shap_results, sample_explanation

if __name__ == "__main__":
    run_explainability_pipeline()
