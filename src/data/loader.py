"""
Data loading, cleaning, anti-leakage filtering, and patient-grouped splitting.
"""
import sys
import os
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml
import numpy as np
import pandas as pd
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Loads configuration YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def categorize_icd9(code: str) -> str:
    """
    Categorizes ICD-9 codes into standard clinical diagnostic groupings.
    Based on standard literature (e.g. Strack et al., 2014).
    """
    if pd.isna(code) or code == "?" or str(code).strip() == "":
        return "Missing"
    
    code_str = str(code).strip()
    
    # Handle codes starting with V or E
    if code_str.startswith("V") or code_str.startswith("E"):
        return "Other"
    
    try:
        val = float(code_str)
    except ValueError:
        return "Other"
    
    # Diabetes specific (250.xx)
    if 250.0 <= val < 251.0 or code_str.startswith("250"):
        return "Diabetes"
    elif 390 <= val <= 459 or val == 785:
        return "Circulatory"
    elif 460 <= val <= 519 or val == 786:
        return "Respiratory"
    elif 520 <= val <= 579 or val == 787:
        return "Digestive"
    elif 580 <= val <= 629 or val == 788:
        return "Genitourinary"
    elif 140 <= val <= 239:
        return "Neoplasms"
    elif 710 <= val <= 739:
        return "Musculoskeletal"
    elif 800 <= val <= 999:
        return "Injury"
    else:
        return "Other"

def prepare_dataset(config: dict) -> pd.DataFrame:
    """
    Reads raw CSV, applies diagnosis categorization, defines 30-day binary target,
    and removes deceased/hospice encounters to prevent target leakage and survival bias.
    """
    raw_path = config["data"]["raw_path"]
    if not Path(raw_path).exists():
        nested_path = Path("data/raw/input/dataset_diabetes/diabetic_data.csv")
        if nested_path.exists():
            raw_path = str(nested_path)
        else:
            logger.info("Raw data not found locally. Downloading UCI diabetes dataset...")
            from src.ingestion.ingest import download_and_extract_data
            url = os.getenv("DATA_SOURCE_URL", "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip")
            download_and_extract_data(url, "data/raw/input")
            if nested_path.exists() and not Path(config["data"]["raw_path"]).exists():
                import shutil
                shutil.copy(nested_path, config["data"]["raw_path"])

    logger.info(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)

    # Standardize '?' to NaN
    df = df.replace("?", np.nan)

    # 1. Target Definition (CMS HRRP standard: 30-day readmission)
    raw_target_col = config["data"]["raw_target_col"]
    positive_label = config["data"]["positive_label"]
    target_col = config["data"]["target_col"]
    
    # Binary: 1 if <30, 0 if >30 or NO
    df[target_col] = (df[raw_target_col] == positive_label).astype(int)
    logger.info(f"Target '{target_col}' created. Positive rate: {df[target_col].mean():.4f}")

    # 2. Prevent Survival Bias & Leakage: Exclude deceased and hospice encounters
    leakage_ids = config["data"].get("leakage_discharge_disposition_ids", [11, 13, 14, 19, 20, 21])
    initial_len = len(df)
    df = df[~df["discharge_disposition_id"].isin(leakage_ids)].copy()
    excluded_count = initial_len - len(df)
    logger.info(f"Excluded {excluded_count} encounters with expired/hospice disposition to prevent leakage.")

    # 3. Categorize ICD-9 diagnoses
    for diag_col in ["diag_1", "diag_2", "diag_3"]:
        if diag_col in df.columns:
            cat_col = f"{diag_col}_category"
            df[cat_col] = df[diag_col].apply(categorize_icd9)

    # 4. Cast numerical columns to numeric
    num_cols = config["features"]["numerical_cols"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Clean demographics
    if "gender" in df.columns:
        df["gender"] = df["gender"].replace("Unknown/Invalid", np.nan)

    return df

def patient_grouped_split(df: pd.DataFrame, config: dict):
    """
    Performs a patient-grouped train/validation/test split.
    Guarantees no patient_nbr overlap across splits, preventing patient identity leakage.
    """
    patient_id_col = config["data"]["patient_id_col"]
    split_cfg = config["data"]["split"]
    train_ratio = split_cfg["train_ratio"]
    val_ratio = split_cfg["val_ratio"]
    test_ratio = split_cfg["test_ratio"]
    random_seed = config["project"]["random_seed"]

    unique_patients = df[patient_id_col].drop_duplicates().values
    np.random.seed(random_seed)
    shuffled_patients = np.random.permutation(unique_patients)

    n_total = len(shuffled_patients)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_patients = set(shuffled_patients[:n_train])
    val_patients = set(shuffled_patients[n_train:n_train + n_val])
    test_patients = set(shuffled_patients[n_train + n_val:])

    train_df = df[df[patient_id_col].isin(train_patients)].copy().reset_index(drop=True)
    val_df = df[df[patient_id_col].isin(val_patients)].copy().reset_index(drop=True)
    test_df = df[df[patient_id_col].isin(test_patients)].copy().reset_index(drop=True)

    # Sanity check: Assert zero overlap
    train_set = set(train_df[patient_id_col])
    val_set = set(val_df[patient_id_col])
    test_set = set(test_df[patient_id_col])

    assert len(train_set.intersection(val_set)) == 0, "Patient leakage between train and val!"
    assert len(train_set.intersection(test_set)) == 0, "Patient leakage between train and test!"
    assert len(val_set.intersection(test_set)) == 0, "Patient leakage between val and test!"

    logger.info(
        f"Grouped split complete. "
        f"Train encounters: {len(train_df)} ({len(train_patients)} patients), "
        f"Val encounters: {len(val_df)} ({len(val_patients)} patients), "
        f"Test encounters: {len(test_df)} ({len(test_patients)} patients)"
    )

    return train_df, val_df, test_df

def run_data_pipeline(config_path: str = "configs/config.yaml"):
    """Orchestrates data preparation, grouped splitting, and parquet storage."""
    config = load_config(config_path)
    df = prepare_dataset(config)
    train_df, val_df, test_df = patient_grouped_split(df, config)

    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(config["data"]["train_data_path"])
    val_path = Path(config["data"]["val_data_path"])
    test_path = Path(config["data"]["test_data_path"])

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    logger.info(f"Saved splits to {train_path}, {val_path}, {test_path}")
    return train_df, val_df, test_df

if __name__ == "__main__":
    run_data_pipeline()
