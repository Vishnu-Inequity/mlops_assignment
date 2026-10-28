"""
MLflow Model Registry and Versioning Module.
Registers selected models into the MLflow Model Registry with versioning,
production/staging aliases, descriptions, metrics tags, and loads models by alias for serving.
"""
from pathlib import Path
import json
import joblib
import mlflow
from mlflow.tracking import MlflowClient

from src.data.loader import load_config
from src.features.pipeline import NumericalImputer, CategoricalImputer
from src.evaluation.metrics import SigmoidCalibrator
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def register_best_model(config_path: str = "configs/config.yaml") -> dict:
    """
    Registers the champion model in the MLflow Model Registry.
    Applies aliases ('production', 'champion', 'staging') and detailed metadata tags.
    """
    config = load_config(config_path)
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    client = MlflowClient()

    best_model_path = Path("models/best_model.joblib")
    if not best_model_path.exists():
        raise FileNotFoundError("Best model file not found at models/best_model.joblib. Run training first.")

    best_artifact = joblib.load(best_model_path)
    model_name = config["mlflow"].get("registered_model_name", "HospitalReadmissionClassifier")
    run_id = best_artifact.get("run_id")
    test_metrics = best_artifact.get("metrics", {})

    # 1. Create or get registered model
    try:
        client.create_registered_model(
            name=model_name,
            description="30-day hospital readmission risk screening classifier. "
                        "ACADEMIC DECISION-SUPPORT PROTOTYPE ONLY - Not for clinical diagnostic use."
        )
        logger.info(f"Created registered model: {model_name}")
    except Exception:
        logger.info(f"Registered model {model_name} already exists.")

    # 2. Locate source artifact
    source_artifact_path = Path(f"models/{best_artifact['model_name']}.joblib").resolve()
    if not source_artifact_path.exists():
        source_artifact_path = best_model_path.resolve()

    # 3. Create model version
    description_text = (
        f"Algorithm: {best_artifact['model_name']} | "
        f"Screening Recall: {test_metrics.get('recall', 'N/A')} | "
        f"Precision: {test_metrics.get('precision', 'N/A')} | "
        f"ROC-AUC: {test_metrics.get('roc_auc', 'N/A')} | "
        f"Brier Score: {test_metrics.get('brier_score', 'N/A')} | "
        f"ECE: {test_metrics.get('ece', 'N/A')} | "
        f"Decision Threshold: {best_artifact.get('threshold', 0.10)} | "
        f"Data: UCI Diabetes 130-US (anti-leakage grouped split)"
    )

    model_version = client.create_model_version(
        name=model_name,
        source=str(source_artifact_path),
        run_id=run_id,
        description=description_text
    )
    v_num = model_version.version
    logger.info(f"Registered {model_name} version {v_num}")

    # 4. Set aliases and tags
    client.set_registered_model_alias(name=model_name, alias="production", version=v_num)
    client.set_registered_model_alias(name=model_name, alias="champion", version=v_num)
    client.set_registered_model_alias(name=model_name, alias="staging", version=v_num)

    tags = {
        "model_type": best_artifact["model_name"],
        "data_version": "v1.0_anti_leakage_grouped",
        "recall_test": str(test_metrics.get("recall", "")),
        "precision_test": str(test_metrics.get("precision", "")),
        "roc_auc_test": str(test_metrics.get("roc_auc", "")),
        "brier_score_test": str(test_metrics.get("brier_score", "")),
        "decision_threshold": str(best_artifact.get("threshold", 0.10)),
        "disclaimer": "Academic decision-support prototype only. Not for clinical diagnostic use."
    }

    for k, v in tags.items():
        client.set_model_version_tag(name=model_name, version=v_num, key=k, value=v)

    logger.info(f"Assigned aliases ['production', 'champion', 'staging'] to version {v_num}")

    registry_info = {
        "model_name": model_name,
        "version": v_num,
        "aliases": ["production", "champion", "staging"],
        "run_id": run_id,
        "description": description_text,
        "tags": tags
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "model_registry_info.json", "w", encoding="utf-8") as f:
        json.dump(registry_info, f, indent=2)

    return registry_info

def load_model_by_alias(model_name: str = "HospitalReadmissionClassifier", alias: str = "production", config_path: str = "configs/config.yaml") -> dict:
    """
    Loads a model by alias from the MLflow Model Registry for the serving layer.
    Falls back gracefully to local artifact if MLflow registry storage is unreachable.
    Returns model, calibrator, decision threshold, preprocessor, and version metadata.
    """
    version_val = "1"
    tags_val = {"disclaimer": "Academic decision-support prototype only. Not for clinical diagnostic use."}
    desc_val = "Academic decision-support prototype"
    model_source_path = Path("models/best_model.joblib")

    try:
        config = load_config(config_path)
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        client = MlflowClient()
        mv = client.get_model_version_by_alias(name=model_name, alias=alias)
        logger.info(f"Retrieved model version {mv.version} for alias '@{alias}' (run_id: {mv.run_id})")
        if Path(mv.source).exists():
            model_source_path = Path(mv.source)
        version_val = mv.version
        tags_val = mv.tags
        desc_val = mv.description
    except Exception as e:
        logger.warning(f"MLflow Model Registry retrieval failed ({e}); falling back to local artifact: {model_source_path}")

    artifact = joblib.load(model_source_path)

    # Load preprocessor
    preprocessor_path = Path("models/preprocessor.joblib")
    if not preprocessor_path.exists():
        raise FileNotFoundError("Preprocessor artifact not found at models/preprocessor.joblib")
    preprocessor = joblib.load(preprocessor_path)

    return {
        "model_name": model_name,
        "version": str(version_val),
        "alias": alias,
        "model": artifact.get("model"),
        "calibrator": artifact.get("calibrator"),
        "threshold": artifact.get("threshold", 0.10),
        "preprocessor": preprocessor,
        "tags": tags_val,
        "description": desc_val
    }

if __name__ == "__main__":
    reg_info = register_best_model()
    loaded = load_model_by_alias(alias="production")
    logger.info(f"Verification successful: loaded model '{loaded['model_name']}' version {loaded['version']} by alias '@{loaded['alias']}'")
