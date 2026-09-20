"""
No-Show Appointment Predictor — Interactive Dashboard
=======================================================
Input appointment details, get a predicted no-show probability 
from a Random Forest model, and leverage SHAP explanations 
to understand WHY the prediction was made.

Run locally:
    pip install streamlit shap joblib pandas scikit-learn matplotlib
    streamlit run dashboard.py

Requires the artifacts/ directory next to this file:
(rf_model.pkl, columns.json)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="No-Show Predictor", page_icon="🏥", layout="wide")

# ---------- Load Model & Artifacts (Cached) ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/rf_model.pkl")
    with open("artifacts/columns.json") as f:
        columns = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, columns, explainer

model, MODEL_COLUMNS, explainer = load_artifacts()


def age_group(age: int) -> str:
    """Classifies age into demographic buckets matching feature engineering logic."""
    if age <= 12:
        return "child"
    elif age <= 24:
        return "young_adult"
    elif age <= 59:
        return "adult"
    else:
        return "senior"


def build_feature_vector(inputs: dict) -> pd.DataFrame:
    """
    Transforms raw user inputs into the exact feature vector format 
    (one-hot encoded, aligned column order) expected by the trained model.
    """
    row = {col: 0 for col in MODEL_COLUMNS}

    row["Age"] = inputs["age"]
    row["wait_days"] = inputs["wait_days"]
    row["Scholarship"] = int(inputs["scholarship"])
    row["Hipertension"] = int(inputs["hipertension"])
    row["Diabetes"] = int(inputs["diabetes"])
    row["Alcoholism"] = int(inputs["alcoholism"])
    row["Handcap"] = inputs["handcap"]
    row["SMS_received"] = int(inputs["sms_received"])
    row["has_chronic_condition"] = int(
        inputs["hipertension"] or inputs["diabetes"] or inputs["alcoholism"] or inputs["handcap"] > 0
    )

    if inputs["gender"] == "Male":
        row["Gender_M"] = 1  # (Female maps to baseline / dropped column)

    ag = age_group(inputs["age"])
    if ag == "child":
        row["age_group_child"] = 1
    elif ag == "senior":
        row["age_group_senior"] = 1
    elif ag == "young_adult":
        row["age_group_young_adult"] = 1
    # 'adult' serves as the reference baseline

    dow_col = f"appointment_dow_{inputs['appointment_dow']}"
    if dow_col in row:
        row[dow_col] = 1
    # 'Friday' serves as the reference baseline

    return pd.DataFrame([row], columns=MODEL_COLUMNS)


# ---------------------- UI ----------------------
st.title("🏥 Medical Appointment No-Show Predictor")
st.caption(
    "An Explainable AI (XAI) demo tool: predicts patient no-show probability "
    "and explains the key factors driving the model's decision."
)

with st.sidebar:
    st.header("Appointment Features")

    age = st.slider("Patient Age", 0, 115, 35)
    wait_days = st.slider(
        "Waiting Days (Scheduling → Appointment)", 0, 180, 5,
        help="0 = Same-day appointment. Top predictive feature for the model."
    )
    gender = st.selectbox("Gender", ["Female", "Male"])
    appointment_dow = st.selectbox(
        "Appointment Day of Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    )
    sms_received = st.checkbox("SMS Reminder Received", value=False)

    st.subheader("Medical History & Social Profile")
    scholarship = st.checkbox("Bolsa Família Welfare Scholarship", value=False)
    hipertension = st.checkbox("Hypertension", value=False)
    diabetes = st.checkbox("Diabetes", value=False)
    alcoholism = st.checkbox("Alcoholism", value=False)
    handcap = st.slider("Disabilities Count (Handicap)", 0, 4, 0)

inputs = dict(
    age=age, wait_days=wait_days, gender=gender, appointment_dow=appointment_dow,
    sms_received=sms_received, scholarship=scholarship, hipertension=hipertension,
    diabetes=diabetes, alcoholism=alcoholism, handcap=handcap,
)

X_input = build_feature_vector(inputs)

# ---------------------- Prediction ----------------------
proba = model.predict_proba(X_input)[0, 1]

col1, col2 = st.columns([1, 2])

with col1:
    st.metric("No-Show Probability", f"{proba*100:.1f}%")
    if proba >= 0.35:
        st.error("🔴 High Risk — Recommended Action: Send extra reminder or follow-up call")
    elif proba >= 0.20:
        st.warning("🟡 Moderate Risk — Monitor appointment status")
    else:
        st.success("🟢 Low Risk — Expected to attend")

with col2:
    st.write("**Submitted Inputs Summary:**")
    st.json({
        "Age": age, "Waiting Days": wait_days, "Gender": gender,
        "Day of Week": appointment_dow, "SMS Sent": sms_received,
        "Welfare Scholarship": scholarship, 
        "Has Chronic Condition": hipertension or diabetes or alcoholism or handcap > 0
    })

st.divider()

# ---------------------- SHAP Explanation ----------------------
st.subheader("Why did the model make this prediction?")

shap_values = explainer.shap_values(X_input)

if isinstance(shap_values, list):
    sv = shap_values[1][0]
    base_value = explainer.expected_value[1]
elif np.ndim(shap_values) == 3:
    sv = shap_values[0, :, 1]
    base_value = explainer.expected_value[1]
else:
    sv = shap_values[0]
    base_value = explainer.expected_value

explanation = shap.Explanation(
    values=sv,
    base_values=base_value,
    data=X_input.iloc[0],
    feature_names=X_input.columns.tolist(),
)

fig, ax = plt.subplots(figsize=(9, 5))
shap.plots.waterfall(explanation, show=False, max_display=10)
st.pyplot(fig)

st.caption(
    "The plot starts at the average baseline probability across the dataset (base value) "
    "and illustrates how each feature pushes the probability higher (red, towards No-Show) "
    "or lower (blue, towards Show) for THIS specific patient profile."
)

st.divider()
st.caption(
    "⚠️ Portfolio / Demo project trained on public data (Kaggle: Medical Appointment No Shows). "
    "Not intended for real clinical deployment without further validation."
)