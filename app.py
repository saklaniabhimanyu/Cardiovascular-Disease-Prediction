"""
Cardiovascular Disease Risk Calculator — Streamlit deployment app.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="CVD Risk Calculator", page_icon="🫀", layout="centered")

BUNDLE_PATH = Path("model")/"cvd_model_bundle.joblib"


@st.cache_resource
def load_bundle(path: str):
    return joblib.load(path)


def compute_bp_stage(ap_hi: float, ap_lo: float) -> int:
    """Mirrors the bp_stage logic in Section 3 of the notebook — staging is
    driven by whichever of systolic/diastolic pressure is more severe."""
    if ap_hi >= 140 or ap_lo >= 90:
        return 3  # Stage 2 HTN
    if ap_hi >= 130 or ap_lo >= 80:
        return 2  # Stage 1 HTN
    if ap_hi >= 120:
        return 1  # Elevated
    return 0  # Normal


BP_STAGE_LABELS = {0: "Normal", 1: "Elevated", 2: "Stage 1 Hypertension", 3: "Stage 2 Hypertension"}


def build_feature_row(inputs: dict, all_raw_features: list) -> pd.DataFrame:
    """Recreates every engineered feature exactly as in the notebook, then
    returns a single-row DataFrame with columns in the training column order."""
    age, gender, height, weight = inputs["age"], inputs["gender"], inputs["height"], inputs["weight"]
    ap_hi, ap_lo = inputs["ap_hi"], inputs["ap_lo"]
    cholesterol, gluc = inputs["cholesterol"], inputs["gluc"]
    smoke, alco, active = inputs["smoke"], inputs["alco"], inputs["active"]

    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = ap_hi - ap_lo
    mean_arterial_pressure = ap_lo + (pulse_pressure / 3)
    bp_stage = compute_bp_stage(ap_hi, ap_lo)

    row = {
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "bmi": bmi,
        "pulse_pressure": pulse_pressure,
        "map": mean_arterial_pressure,
        "bp_stage": bp_stage,
    }

    df = pd.DataFrame([row])
    # keep only / order exactly as the training feature matrix
    return df[all_raw_features], bp_stage, bmi


def main():
    st.title("🫀 Cardiovascular Disease Risk Calculator")
    st.caption(
        "Predicts probability of cardiovascular disease from routine clinical "
        "and lifestyle inputs. This is a portfolio/educational model — "
        "**not a medical device** and should not be used for real diagnosis."
    )

    try:
        bundle = load_bundle(BUNDLE_PATH)
    except FileNotFoundError:
        st.error(
            f"Couldn't find `{BUNDLE_PATH}`. Run Section 13 of the notebook "
            "first to generate it, and place it in the same folder as this app."
        )
        st.stop()

    model = bundle["model"]
    scaler = bundle.get("scaler", None)
    uses_scaled = bundle["uses_scaled"]
    selected_features = bundle["selected_features"]
    all_raw_features = bundle["all_raw_features"]
    threshold = bundle["threshold"]
    model_name = bundle["model_name"]

    with st.sidebar:
        st.subheader("Model info")
        st.write(f"**Model:** {model_name}")
        st.write(f"**Decision threshold:** {threshold:.3f}")
        st.write("**Test-set performance:**")
        for k, v in bundle["metrics_test"].items():
            st.write(f"- {k}: {v:.3f}")

    st.subheader("Patient information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
        gender = st.selectbox("Gender", options=[("Female", 1), ("Male", 2)], format_func=lambda x: x[0])[1]
        height = st.number_input("Height (cm)", min_value=120, max_value=250, value=165)
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0, step=0.5)
        active = st.selectbox("Physically active?", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]

    with col2:
        ap_hi = st.number_input("Systolic BP (ap_hi, mmHg)", min_value=60, max_value=280, value=120)
        ap_lo = st.number_input("Diastolic BP (ap_lo, mmHg)", min_value=40, max_value=180, value=80)
        cholesterol_value = st.number_input(
                                "Total Cholesterol (mg/dL)",
                                min_value=80,
                                max_value=500,
                                value=180
                            )

        if cholesterol_value < 200:
            cholesterol = 1
        elif cholesterol_value < 240:
            cholesterol = 2
        else:
            cholesterol = 3
        st.caption(f"Cholesterol Category: {['Normal', 'Above Normal', 'Well Above Normal'][cholesterol-1]}")
        glucose_value = st.number_input(
                                        "Fasting Blood Glucose (mg/dL)",
                                        min_value=40,
                                        max_value=400,
                                        value=90
                                    )

        if glucose_value < 100:
            gluc = 1
        elif glucose_value < 126:
            gluc = 2
        else:
            gluc = 3
        st.caption(f"Glucose Category: {['Normal', 'Above Normal', 'Well Above Normal'][gluc-1]}")

    col3, col4 = st.columns(2)
    with col3:
        smoke = st.selectbox("Smoker?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    with col4:
        alco = st.selectbox("Drinks alcohol?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

    if ap_lo >= ap_hi:
        st.warning("Diastolic (ap_lo) should be lower than systolic (ap_hi). Please check your inputs.")

    _, center, _ = st.columns([1,2,1])

    with center:
        predict = st.button("Predict CVD Risk", use_container_width=True)

    if predict:
        inputs = dict(
            age=age, gender=gender, height=height, weight=weight,
            ap_hi=ap_hi, ap_lo=ap_lo, cholesterol=cholesterol, gluc=gluc,
            smoke=smoke, alco=alco, active=active,
        )
        full_row, bp_stage, bmi = build_feature_row(inputs, all_raw_features)
        model_row = full_row[selected_features]

        if uses_scaled and scaler is not None:
            model_input = scaler.transform(model_row)
        else:
            model_input = model_row

        proba = model.predict_proba(model_input)[0, 1]
        pred = int(proba >= threshold)

        st.divider()
        st.subheader("Result")

        c1, c2, c3 = st.columns(3)
        c1.metric("BMI", f"{bmi:.1f}")
        c2.metric("Pulse pressure", f"{ap_hi - ap_lo} mmHg")
        c3.metric("BP stage", BP_STAGE_LABELS[bp_stage])

        risk = proba * 100
        if risk < 30:
            st.success("🟢 Low predicted risk")
        elif risk < 60:
            st.warning("🟡 Moderate predicted risk")
        else:
            st.error("🔴 High predicted risk")
        proba = float(model.predict_proba(model_input)[0, 1])

        st.metric("Predicted CVD Probability", f"{proba * 100:.1f}%")
        st.progress(proba)
        if pred == 1:
            st.error(
                f"Model predicts **higher risk of cardiovascular disease** "
                f"(probability {proba*100:.1f}% ≥ threshold {threshold*100:.1f}%)."
            )
        else:
            st.success(
                f"Model predicts **lower risk of cardiovascular disease** "
                f"(probability {proba*100:.1f}% < threshold {threshold*100:.1f}%)."
            )

        st.caption(
            "Reminder: this is a statistical model trained on a single public "
            "dataset. It does not account for family history, medications, or "
            "many other clinically relevant factors, and should not replace "
            "professional medical advice."
        )
        st.subheader("Generated Features")

        summary = pd.DataFrame({
            "Feature": [
                "BMI",
                "Pulse Pressure",
                "Mean Arterial Pressure",
                "Blood Pressure Stage"
            ],
            "Value": [
                f"{bmi:.2f}",
                f"{ap_hi - ap_lo} mmHg",
                f"{ap_lo + (ap_hi - ap_lo) / 3:.2f} mmHg",
                BP_STAGE_LABELS[bp_stage]
            ]
        })
        st.table(summary)


if __name__ == "__main__":
    main()