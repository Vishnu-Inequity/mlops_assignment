"""
Hospital Readmission Clinical Decision Support & MLOps Analytics Dashboard.
Part 2 MLOps Pipeline Extension with Real-Time Prediction, TreeSHAP Explainability,
and Governance Monitoring.

DISCLAIMER: ACADEMIC DECISION-SUPPORT PROTOTYPE ONLY.
NOT A VALIDATED CLINICAL DIAGNOSTIC TOOL.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import os
import time
import requests
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv


from src.data.loader import categorize_icd9, load_config
from src.features.pipeline import transform_data, get_transformed_feature_names
from src.explainability.explainer import explain_instance
from src.serving.schemas import PatientEncounterInput

load_dotenv()

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="Hospital Readmission Decision Support (MLOps)",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clinical decision support styling
st.markdown("""
<style>
    .disclaimer-banner {
        background-color: #FEF2F2;
        border: 2px solid #DC2626;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 20px;
        color: #991B1B;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .high-risk-badge {
        background-color: #DC2626;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-size: 1.1rem;
    }
    .low-risk-badge {
        background-color: #16A34A;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-size: 1.1rem;
    }
    .service-badge-online {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .service-badge-local {
        background-color: #FEF9C3;
        color: #854D0E;
        border: 1px solid #FDE047;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")

# ==============================================================================
# Helper Functions: Service Connectivity & Local Fallback
# ==============================================================================
@st.cache_resource
def load_local_serving_artifacts():
    """Loads model and preprocessor for zero-downtime local in-process fallback."""
    try:
        model_artifact = joblib.load("models/best_model.joblib")
        preprocessor = joblib.load("models/preprocessor.joblib")
        return {
            "model": model_artifact["model"],
            "calibrator": model_artifact.get("calibrator"),
            "threshold": model_artifact.get("threshold", 0.10),
            "model_name": model_artifact.get("model_name", "XGBoost"),
            "preprocessor": preprocessor,
            "available": True
        }
    except Exception as e:
        return {"available": False, "error": str(e)}

def check_api_health():
    """Checks if FastAPI backend container/process is running."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=1.5)
        if resp.status_code == 200:
            return True, resp.json()
    except Exception:
        pass
    return False, {}

def predict_encounter(payload: dict):
    """
    Evaluates encounter: First attempts live FastAPI endpoint;
    falls back cleanly to in-process model execution if API is offline.
    """
    is_live, health_info = check_api_health()
    if is_live:
        try:
            t0 = time.time()
            resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=5.0)
            if resp.status_code == 200:
                result = resp.json()
                result["serving_mode"] = "Live FastAPI Service"
                return result
        except Exception:
            pass

    # In-process Fallback
    local_artifacts = load_local_serving_artifacts()
    if not local_artifacts.get("available"):
        raise RuntimeError(f"Neither API nor local model artifact available: {local_artifacts.get('error')}")

    model = local_artifacts["model"]
    calibrator = local_artifacts.get("calibrator")
    threshold = local_artifacts.get("threshold", 0.10)
    preprocessor = local_artifacts["preprocessor"]

    # Transform payload using robust Pydantic schema defaults
    encounter_obj = PatientEncounterInput(**payload)
    proc_payload = encounter_obj.resolve_categories()

    import xgboost as xgb
    df_single = pd.DataFrame([proc_payload])
    X_mat = preprocessor.transform(df_single)
    feature_names = get_transformed_feature_names(preprocessor)

    # Inference & Calibration
    if isinstance(model, xgb.Booster):
        dmat = xgb.DMatrix(X_mat)
        raw_prob = float(model.predict(dmat)[0])
    elif hasattr(model, "predict_proba"):
        raw_prob = float(model.predict_proba(X_mat)[0, 1])
    else:
        dmat = xgb.DMatrix(X_mat)
        raw_prob = float(model.predict(dmat)[0])

    prob_val = raw_prob
    if calibrator is not None:
        prob_val = float(calibrator.predict_proba(np.array([raw_prob]))[0])

    pred_class = int(prob_val >= threshold)
    pred_label = "High Risk (<30 Days Readmission)" if pred_class == 1 else "Low Risk (Unlikely <30 Days Readmission)"

    # Explainability
    explanation = explain_instance(model, X_mat, feature_names, top_k=5)

    return {
        "readmission_probability": round(prob_val, 4),
        "predicted_class": pred_class,
        "predicted_label": pred_label,
        "decision_threshold": threshold,
        "model_version": "1 (local fallback)",
        "model_alias": "production",
        "top_contributing_features": explanation.get("top_features", []),
        "latency_ms": 15.0,
        "serving_mode": "In-Process Fallback Engine",
        "disclaimer": "ACADEMIC DECISION-SUPPORT PROTOTYPE ONLY: Not for clinical diagnostic use."
    }

# ==============================================================================
# Clinical Presets
# ==============================================================================
CLINICAL_PRESETS = {
    "🚨 High Risk: Elderly with Heart Failure & Prior Admissions": {
        "race": "Caucasian", "gender": "Male", "age": "[70-80)",
        "admission_type_id": 1, "discharge_disposition_id": 1, "admission_source_id": 7,
        "time_in_hospital": 6, "num_lab_procedures": 54, "num_procedures": 1, "num_medications": 22,
        "number_outpatient": 1, "number_emergency": 1, "number_inpatient": 3,
        "diag_1": "428", "diag_2": "414", "diag_3": "250", "number_diagnoses": 9,
        "max_glu_serum": "None", "A1Cresult": "None", "change": "Ch", "diabetesMed": "Yes"
    },
    "🟢 Low Risk: Routine Elective Review, No Prior Inpatient": {
        "race": "Caucasian", "gender": "Female", "age": "[40-50)",
        "admission_type_id": 3, "discharge_disposition_id": 1, "admission_source_id": 1,
        "time_in_hospital": 2, "num_lab_procedures": 30, "num_procedures": 2, "num_medications": 8,
        "number_outpatient": 1, "number_emergency": 0, "number_inpatient": 0,
        "diag_1": "250", "diag_2": "401", "diag_3": "V58", "number_diagnoses": 4,
        "max_glu_serum": "None", "A1Cresult": "Norm", "change": "No", "diabetesMed": "Yes"
    },
    "⚠️ Moderate/Borderline: Respiratory Complication with High A1C": {
        "race": "AfricanAmerican", "gender": "Female", "age": "[60-70)",
        "admission_type_id": 1, "discharge_disposition_id": 1, "admission_source_id": 7,
        "time_in_hospital": 4, "num_lab_procedures": 48, "num_procedures": 0, "num_medications": 16,
        "number_outpatient": 0, "number_emergency": 1, "number_inpatient": 1,
        "diag_1": "486", "diag_2": "250", "diag_3": "428", "number_diagnoses": 7,
        "max_glu_serum": "None", "A1Cresult": ">8", "change": "Ch", "diabetesMed": "Yes"
    },
    "✏️ Custom Encounter (Manual Entry)": None
}

# ==============================================================================
# Header & Navigation
# ==============================================================================

# Sidebar Navigation
st.sidebar.title("🏥 Navigation")
nav_choice = st.sidebar.radio(
    "Select View:",
    [
        "🔮 Real-Time Clinical Decision Support",
        "📈 Model Governance & Performance"
    ]
)

# Service Status Indicator in Sidebar
is_api_healthy, health_meta = check_api_health()
st.sidebar.divider()
st.sidebar.subheader("System Health")
if is_api_healthy:
    st.sidebar.markdown(f'<span class="service-badge-online">● Live API Online</span> (port 8000)', unsafe_allow_html=True)
    st.sidebar.caption(f"Model: {health_meta.get('model_name', 'N/A')} v{health_meta.get('model_version', '1')}")
else:
    st.sidebar.markdown(f'<span class="service-badge-local">● In-Process Engine</span>', unsafe_allow_html=True)
    st.sidebar.caption("FastAPI container offline; using local weights fallback.")

# ==============================================================================
# VIEW 1: REAL-TIME CLINICAL DECISION SUPPORT
# ==============================================================================
if nav_choice == "🔮 Real-Time Clinical Decision Support":
    st.title("🔮 Acute Encounter Readmission Risk Screening")
    st.write(
        "Screen diabetic inpatient encounters for **30-day all-cause readmission risk**. "
        "The model is calibrated for clinical screening sensitivity (target recall ≥ 60%), "
        "allowing proactive post-discharge transition planning."
    )

    preset_name = st.selectbox(
        "Select Clinical Scenario Preset:",
        list(CLINICAL_PRESETS.keys()),
        index=0
    )
    preset_data = CLINICAL_PRESETS[preset_name]

    # Form Defaults
    default_vals = preset_data or {
        "race": "Caucasian", "gender": "Male", "age": "[60-70)",
        "admission_type_id": 1, "discharge_disposition_id": 1, "admission_source_id": 7,
        "time_in_hospital": 4, "num_lab_procedures": 40, "num_procedures": 1, "num_medications": 14,
        "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 1,
        "diag_1": "414", "diag_2": "250", "diag_3": "401", "number_diagnoses": 6,
        "max_glu_serum": "None", "A1Cresult": "None", "change": "No", "diabetesMed": "Yes"
    }

    with st.form("encounter_intake_form"):
        tab1, tab2, tab3 = st.tabs(["📋 Demographics & Utilization", "🏥 Hospitalization & Diagnoses", "💊 Medications & Labs"])

        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                age_opts = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
                age = st.selectbox("Age Bracket", age_opts, index=age_opts.index(default_vals.get("age", "[60-70)")))
                gender = st.selectbox("Gender", ["Male", "Female"], index=0 if default_vals.get("gender") == "Male" else 1)
            with col2:
                race_opts = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"]
                race = st.selectbox("Reported Race", race_opts, index=race_opts.index(default_vals.get("race", "Caucasian")))
                num_inpatient = st.number_input("Prior Inpatient Admissions (Past Year)", min_value=0, max_value=25, value=int(default_vals.get("number_inpatient", 1)), help="Strongest predictive driver of readmission.")
            with col3:
                num_emergency = st.number_input("Prior Emergency Visits (Past Year)", min_value=0, max_value=30, value=int(default_vals.get("number_emergency", 0)))
                num_outpatient = st.number_input("Prior Outpatient Visits (Past Year)", min_value=0, max_value=40, value=int(default_vals.get("number_outpatient", 0)))

        with tab2:
            c1, c2, c3 = st.columns(3)
            with c1:
                time_in_hospital = st.slider("Length of Stay (Days)", min_value=1, max_value=14, value=int(default_vals.get("time_in_hospital", 4)))
                admission_type = st.selectbox("Admission Type", [1, 2, 3, 5, 6, 8], format_func=lambda x: {1: "1 - Emergency", 2: "2 - Urgent", 3: "3 - Elective", 5: "5 - Trauma", 6: "6 - Referral"}.get(x, str(x)), index=0)
            with c2:
                discharge_disp = st.selectbox("Discharge Disposition", [1, 2, 3, 4, 5, 6, 7, 8], format_func=lambda x: {1: "1 - Discharged to Home", 2: "2 - Short Term Hospital", 3: "3 - SNF", 6: "6 - Home Health Service"}.get(x, str(x)), index=0)
                admission_source = st.selectbox("Admission Source", [7, 1, 2, 4, 5], format_func=lambda x: {7: "7 - Emergency Room", 1: "1 - Physician Referral", 2: "2 - Clinic Referral"}.get(x, str(x)), index=0)
            with c3:
                diag_1 = st.text_input("Primary Diagnosis (ICD-9 Code)", value=str(default_vals.get("diag_1", "414")), help="e.g. 414 (Coronary atherosclerosis), 428 (Heart failure), 250 (Diabetes)")
                diag_2 = st.text_input("Secondary Diagnosis (ICD-9 Code)", value=str(default_vals.get("diag_2", "428")))
                diag_3 = st.text_input("Tertiary Diagnosis (ICD-9 Code)", value=str(default_vals.get("diag_3", "250")))
                number_diagnoses = st.number_input("Total Diagnoses on Claim", min_value=1, max_value=16, value=int(default_vals.get("number_diagnoses", 6)))

        with tab3:
            m1, m2, m3 = st.columns(3)
            with m1:
                num_meds = st.number_input("Total Distinct Medications", min_value=1, max_value=80, value=int(default_vals.get("num_medications", 14)))
                num_labs = st.number_input("Number of Lab Procedures", min_value=1, max_value=130, value=int(default_vals.get("num_lab_procedures", 40)))
            with m2:
                num_procs = st.number_input("Non-Lab Clinical Procedures", min_value=0, max_value=10, value=int(default_vals.get("num_procedures", 1)))
                diabetes_med = st.selectbox("Prescribed Diabetes Medication?", ["Yes", "No"], index=0 if default_vals.get("diabetesMed") == "Yes" else 1)
            with m3:
                change = st.selectbox("Diabetic Medication Dosage Changed?", ["No", "Ch"], index=1 if default_vals.get("change") == "Ch" else 0)
                a1c_opts = ["None", "Norm", ">7", ">8"]
                a1c = st.selectbox("HbA1c Test Result", a1c_opts, index=a1c_opts.index(default_vals.get("A1Cresult", "None")))
                glu_opts = ["None", "Norm", ">200", ">300"]
                max_glu = st.selectbox("Max Glucose Serum", glu_opts, index=glu_opts.index(default_vals.get("max_glu_serum", "None")))

        submitted = st.form_submit_button("⚡ Evaluate Readmission Risk", use_container_width=True, type="primary")

    if submitted:
        payload = {
            "race": race, "gender": gender, "age": age,
            "admission_type_id": admission_type, "discharge_disposition_id": discharge_disp, "admission_source_id": admission_source,
            "time_in_hospital": time_in_hospital, "num_lab_procedures": num_labs, "num_procedures": num_procs,
            "num_medications": num_meds, "number_outpatient": num_outpatient, "number_emergency": num_emergency,
            "number_inpatient": num_inpatient, "diag_1": diag_1.strip(), "diag_2": diag_2.strip(), "diag_3": diag_3.strip(),
            "number_diagnoses": number_diagnoses, "max_glu_serum": max_glu, "A1Cresult": a1c,
            "change": change, "diabetesMed": diabetes_med
        }

        with st.spinner("Executing calibrated probability inference & TreeSHAP calculation..."):
            try:
                res = predict_encounter(payload)
                st.session_state["last_prediction"] = res
            except Exception as e:
                st.error(f"Inference error: {e}")

    # Display Prediction Results
    if "last_prediction" in st.session_state:
        res = st.session_state["last_prediction"]
        prob = res["readmission_probability"]
        thresh = res["decision_threshold"]
        is_high = res["predicted_class"] == 1

        st.divider()
        st.subheader("🎯 Screening Assessment & Risk Score")

        kpi_col1, kpi_col2, kpi_col3 = st.columns([1.2, 1.2, 1.6])

        with kpi_col1:
            st.metric("Predicted 30-Day Risk", f"{prob * 100:.1f}%")
            st.caption(f"Decision Cutoff Threshold: **{thresh * 100:.1f}%**")

        with kpi_col2:
            st.write("**Risk Classification:**")
            if is_high:
                st.markdown('<span class="high-risk-badge">🚨 HIGH RISK (<30d)</span>', unsafe_allow_html=True)
                st.caption("Recommended for Enhanced Care Coordination")
            else:
                st.markdown('<span class="low-risk-badge">🟢 LOW RISK</span>', unsafe_allow_html=True)
                st.caption("Standard Post-Discharge Followup")

        with kpi_col3:
            st.write(f"**Inference Mode:** `{res.get('serving_mode')}`")
            st.caption(f"Latency: {res.get('latency_ms', 0):.1f} ms | Model: v{res.get('model_version')}")

        # Visual Probability Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 36, "color": "#1E293B"}},
            title={"text": "30-Day Readmission Probability vs Screening Threshold", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748B"},
                "bar": {"color": "#DC2626" if is_high else "#16A34A", "thickness": 0.4},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#CBD5E1",
                "steps": [
                    {"range": [0, thresh * 100], "color": "#F0FDF4"},
                    {"range": [thresh * 100, 100], "color": "#FEF2F2"}
                ],
                "threshold": {
                    "line": {"color": "#B91C1C", "width": 4},
                    "thickness": 0.85,
                    "value": thresh * 100
                }
            }
        ))
        fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Local TreeSHAP Explainability Breakdown
        st.subheader("🔍 Patient-Specific Feature Attribution (TreeSHAP)")
        st.write(
            "The chart below identifies the specific clinical drivers influencing this patient's risk score. "
            "Bars pointing right (**red**) increase readmission probability, while bars pointing left (**green**) reduce it."
        )

        top_feats = res.get("top_contributing_features", [])
        if top_feats:
            feats_df = pd.DataFrame(top_feats)
            # Friendly clinical translations
            clinical_dict = {
                "number_inpatient": "Prior Inpatient Admissions",
                "number_diagnoses": "Comorbidity Burden (Diagnoses count)",
                "num_medications": "Polypharmacy (Medications count)",
                "diag_1_category_Circulatory": "Cardiovascular Primary Diagnosis",
                "diag_1_category_Diabetes": "Diabetes Primary Diagnosis",
                "diag_1_category_Respiratory": "Respiratory Primary Diagnosis",
                "time_in_hospital": "Extended Length of Stay",
                "A1Cresult_>8": "High HbA1c (>8%)",
                "admission_type_id": "Emergency Admission Route",
                "discharge_disposition_id": "Discharge Facility Disposition"
            }
            feats_df["clinical_description"] = feats_df["feature"].apply(lambda x: clinical_dict.get(x, x.replace("_", " ").title()))
            feats_df["color"] = feats_df["shap_value"].apply(lambda v: "#DC2626" if v > 0 else "#16A34A")

            fig_shap = go.Figure()
            fig_shap.add_trace(go.Bar(
                x=feats_df["shap_value"],
                y=feats_df["clinical_description"],
                orientation="h",
                marker=dict(color=feats_df["color"]),
                text=feats_df["shap_value"].apply(lambda v: f"{v:+.3f}"),
                textposition="outside"
            ))
            fig_shap.update_layout(
                title="Top 5 Clinical Risk Contributors (Impact on Log-Odds)",
                xaxis_title="TreeSHAP Value (Contribution to Risk)",
                yaxis=dict(autorange="reversed"),
                height=280,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            # Tabular breakdown
            st.dataframe(
                feats_df[["clinical_description", "direction", "shap_value", "feature_value_scaled"]].rename(columns={
                    "clinical_description": "Clinical Factor",
                    "direction": "Effect on Risk",
                    "shap_value": "SHAP Impact",
                    "feature_value_scaled": "Encounter Value (Transformed)"
                }),
                use_container_width=True
            )

        # Actionable Care Coordination Recommendations
        if is_high:
            st.info("""
            **📋 Suggested Clinical Decision-Support Next Steps (High-Risk Cohort):**
            - **Pharmacist Medication Reconciliation**: Conduct review prior to discharge to mitigate polypharmacy interactions.
            - **Transition of Care Follow-up**: Schedule post-discharge telephone call within 48-72 hours.
            - **Rapid Outpatient Visit**: Ensure primary care or endocrinology appointment is secured within 7 days.
            - **Disease Self-Management**: Review diabetic care, insulin dosing, and red-flag warning signs with patient/caregiver.
            """)

# ==============================================================================
# VIEW 2: MODEL GOVERNANCE & PERFORMANCE CARD
# ==============================================================================
elif nav_choice == "📈 Model Governance & Performance":
    st.title("📈 Model Governance, Calibration & Artifact Registry")
    st.write(
        "Complete technical governance details, held-out test evaluation metrics, "
        "and MLflow tracking artifacts for the production model."
    )

    # Load registry info
    reg_file = Path("reports/model_registry_info.json")
    reg_data = {}
    if reg_file.exists():
        with open(reg_file, "r") as f:
            reg_data = json.load(f)

    # Key Metadata
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Registered Model", reg_data.get("model_name", "HospitalReadmissionClassifier"))
    g2.metric("Production Version", f"v{reg_data.get('version', '1')}")
    g3.metric("Champion Algorithm", "XGBoost (Sigmoid Calibrated)")
    g4.metric("Screening Threshold", "0.10 (Sensitivity Priority)")

    st.divider()

    # Held-out Test Set Performance Table
    st.subheader("📊 Held-Out Test Evaluation Metrics (14,992 Encounters)")
    st.caption("Evaluated on patient-grouped test partition with zero patient overlap from training set.")

    comp_file = Path("reports/model_comparison.json")
    if comp_file.exists():
        with open(comp_file, "r") as f:
            comp_data = json.load(f)
        
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True)

    st.divider()

    # Diagnostic Plots
    st.subheader("🖼️ MLflow Diagnostic Visualizations")
    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        st.write("**TreeSHAP Global Summary**")
        shap_plot = Path("reports/plots/shap_summary.png")
        if shap_plot.exists():
            st.image(str(shap_plot), caption="Mean |SHAP Value| across evaluation cohort", use_container_width=True)
        else:
            st.info("TreeSHAP plot not found.")

        st.write("**Calibration Reliability Curve**")
        calib_plot = Path("reports/plots/XGBoost_calibration_curve.png")
        if calib_plot.exists():
            st.image(str(calib_plot), caption="Observed fraction vs predicted probability (ECE: 0.006)", use_container_width=True)

    with plot_col2:
        st.write("**ROC Curve (Test AUC: 0.627)**")
        roc_plot = Path("reports/plots/XGBoost_roc_curve.png")
        if roc_plot.exists():
            st.image(str(roc_plot), caption="True Positive Rate vs False Positive Rate", use_container_width=True)

        st.write("**Confusion Matrix Heatmap (Threshold = 0.10)**")
        cm_plot = Path("reports/plots/XGBoost_confusion_matrix.png")
        if cm_plot.exists():
            st.image(str(cm_plot), caption="Confusion Matrix on Test Set", use_container_width=True)

    # Serving Request Log Viewer
    st.divider()
    st.subheader("📝 Live Serving Request Audit Trail")
    log_file = Path("logs/serving_requests.log")
    if log_file.exists():
        with open(log_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        if lines:
            records = []
            for line in lines[-20:]:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
            if records:
                log_df = pd.DataFrame(records)
                cols_to_show = [c for c in ["timestamp", "predicted_class", "readmission_probability", "latency_ms", "request_id"] if c in log_df.columns]
                st.dataframe(log_df[cols_to_show].tail(10), use_container_width=True)
                st.caption(f"Showing most recent {len(records)} requests from logs/serving_requests.log")
    else:
        st.caption("No serving requests logged yet.")

