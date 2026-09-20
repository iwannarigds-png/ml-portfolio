"""
No-Show Appointment Predictor — Interactive Dashboard
=======================================================
Εισάγεις χαρακτηριστικά ενός ραντεβού, το μοντέλο (Random Forest) δίνει
πιθανότητα no-show, και το SHAP εξηγεί ΓΙΑΤΙ έδωσε αυτή την πρόβλεψη.

Εκτέλεση τοπικά:
    pip install streamlit shap joblib pandas scikit-learn matplotlib
    streamlit run dashboard.py

Χρειάζεται τον φάκελο artifacts/ δίπλα σε αυτό το αρχείο
(rf_model.pkl, columns.json) — έρχονται μαζί στο ίδιο πακέτο.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="No-Show Predictor", page_icon="🏥", layout="wide")

# ---------- Φόρτωση μοντέλου & artifacts (μία φορά, cached) ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/rf_model.pkl")
    with open("artifacts/columns.json") as f:
        columns = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, columns, explainer

model, MODEL_COLUMNS, explainer = load_artifacts()


def age_group(age: int) -> str:
    """Ίδια λογική με το feature engineering στο notebook 02."""
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
    Μετατρέπει τις ωμές εισόδους του χρήστη στο ακριβές format
    (one-hot encoded, ίδια σειρά στηλών) που περιμένει το μοντέλο.
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

    if inputs["gender"] == "Άνδρας":
        row["Gender_M"] = 1  # (False στη βάση σημαίνει "Gender_F" — drop_first)

    ag = age_group(inputs["age"])
    if ag == "child":
        row["age_group_child"] = 1
    elif ag == "senior":
        row["age_group_senior"] = 1
    elif ag == "young_adult":
        row["age_group_young_adult"] = 1
    # 'adult' = baseline, καμία στήλη δεν ενεργοποιείται

    dow_col = f"appointment_dow_{inputs['appointment_dow']}"
    if dow_col in row:
        row[dow_col] = 1
    # 'Friday' = baseline (dropped στο encoding)

    return pd.DataFrame([row], columns=MODEL_COLUMNS)


# ---------------------- UI ----------------------
st.title("🏥 Πρόβλεψη No-Show Ραντεβού")
st.caption(
    "Demo εργαλείο explainable AI: προβλέπει την πιθανότητα να μην εμφανιστεί "
    "ένας ασθενής, και εξηγεί ποιοι παράγοντες οδήγησαν στην πρόβλεψη."
)

with st.sidebar:
    st.header("Χαρακτηριστικά Ραντεβού")

    age = st.slider("Ηλικία ασθενή", 0, 115, 35)
    wait_days = st.slider(
        "Μέρες αναμονής (scheduling → appointment)", 0, 180, 5,
        help="0 = ραντεβού αυθημερόν. Το πιο σημαντικό feature του μοντέλου."
    )
    gender = st.selectbox("Φύλο", ["Γυναίκα", "Άνδρας"])
    appointment_dow = st.selectbox(
        "Ημέρα ραντεβού",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    )
    sms_received = st.checkbox("Έλαβε SMS reminder", value=False)

    st.subheader("Ιατρικό ιστορικό / κοινωνικά")
    scholarship = st.checkbox("Scholarship (κοινωνικό πρόγραμμα)", value=False)
    hipertension = st.checkbox("Υπέρταση", value=False)
    diabetes = st.checkbox("Διαβήτης", value=False)
    alcoholism = st.checkbox("Αλκοολισμός", value=False)
    handcap = st.slider("Handcap (αριθμός αναπηριών)", 0, 4, 0)

inputs = dict(
    age=age, wait_days=wait_days, gender=gender, appointment_dow=appointment_dow,
    sms_received=sms_received, scholarship=scholarship, hipertension=hipertension,
    diabetes=diabetes, alcoholism=alcoholism, handcap=handcap,
)

X_input = build_feature_vector(inputs)

# ---------------------- Πρόβλεψη ----------------------
proba = model.predict_proba(X_input)[0, 1]

col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Πιθανότητα No-Show", f"{proba*100:.1f}%")
    if proba >= 0.35:
        st.error("🔴 Υψηλό ρίσκο — προτεινόμενη ενέργεια: extra reminder / follow-up call")
    elif proba >= 0.20:
        st.warning("🟡 Μέτριο ρίσκο")
    else:
        st.success("🟢 Χαμηλό ρίσκο")

with col2:
    st.write("**Εισαχθέντα χαρακτηριστικά:**")
    st.json({
        "Ηλικία": age, "Μέρες αναμονής": wait_days, "Φύλο": gender,
        "Ημέρα ραντεβού": appointment_dow, "SMS": sms_received,
        "Scholarship": scholarship, "Χρόνιες παθήσεις": hipertension or diabetes or alcoholism or handcap > 0
    })

st.divider()

# ---------------------- SHAP Explanation ----------------------
st.subheader("Γιατί το μοντέλο έδωσε αυτή την πρόβλεψη;")

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
    "Το γράφημα ξεκινά από τη μέση πρόβλεψη σε όλο το dataset (base value) και δείχνει "
    "πώς κάθε χαρακτηριστικό 'σπρώχνει' την πρόβλεψη πάνω (κόκκινο, προς no-show) ή "
    "κάτω (μπλε, προς εμφάνιση) για ΑΥΤΟΝ τον συγκεκριμένο ασθενή."
)

st.divider()
st.caption(
    "⚠️ Demo/portfolio project πάνω σε δημόσιο dataset (Kaggle: Medical Appointment No Shows). "
    "Δεν προορίζεται για πραγματική κλινική χρήση χωρίς περαιτέρω validation."
)
