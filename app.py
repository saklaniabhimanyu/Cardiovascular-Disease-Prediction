import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="CVD Risk Calculator", page_icon="🫀", layout="centered", initial_sidebar_state="collapsed")

BUNDLE_PATH = Path("model")/"cvd_model_bundle.joblib"

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 0.6rem;
            padding-bottom: 2rem;
            max-width: 720px;
        }
        header[data-testid="stHeader"] { height: 1.5rem; }
        h1 { font-size: 2.05rem !important; margin-top: 0 !important; margin-bottom: 0.2rem !important; }
        h2 { font-size: 1.2rem !important; }
        h3, .stSubheader { font-size: 1rem !important; }
        p, .stCaption, [data-testid="stCaptionContainer"] { font-size: 0.85rem !important; }

        div[data-testid="stNumberInput"] input,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] {
            min-height: 2.3rem !important;
            font-size: 0.9rem !important;
        }
        div[data-testid="stNumberInput"] label,
        div[data-testid="stSelectbox"] label {
            font-size: 0.82rem !important;
            margin-bottom: 0.1rem !important;
        }
        div[data-testid="stNumberInput"], div[data-testid="stSelectbox"] {
            margin-bottom: 0 !important;
        }
        div[data-testid="stCaptionContainer"] { margin-bottom: 0 !important; }
        div[data-testid="column"] { padding: 0 0.35rem !important; }
        div[data-testid="stVerticalBlockBorderWrapper"] { gap: 0.3rem !important; }
        .stButton button { padding: 0.45rem 0.9rem !important; font-size: 0.95rem !important; }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 0.7rem 0.9rem;
        }
        div[data-testid="stMetricLabel"] > div {
            font-size: 0.82rem !important;
            color: #9aa0a6 !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.35rem !important;
            font-weight: 600 !important;
        }

        div[data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
        hr { margin: 1.1rem 0 !important; }

        .result-card {
            border-radius: 14px;
            padding: 1.2rem 1.3rem;
            text-align: center;
            border: 1px solid;
        }
        .result-card .rc-label {
            font-size: 0.9rem;
            opacity: 0.85;
            margin-bottom: 0.15rem;
        }
        .result-card .rc-value {
            font-size: 1.6rem;
            font-weight: 700;
        }

        .ps-item .ps-label {
            font-size: 0.76rem;
            color: #9aa0a6;
            margin-bottom: 0.1rem;
        }
        .ps-item .ps-value {
            font-size: 1.1rem;
            font-weight: 600;
        }

        .cat-badge-wrap {
            padding-top: 1.85rem;
        }
        .cat-badge {
            display: inline-block;
            font-size: 0.75rem;
            color: #9aa0a6;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 6px;
            padding: 0.35rem 0.6rem;
            white-space: nowrap;
        }

        .footer-note {
            font-size: 0.78rem;
            color: #9aa0a6;
            text-align: center;
            margin-top: 2rem;
        }
        .top-note {
            font-size: 0.8rem;
            color: #9aa0a6;
            margin-bottom: 0.6rem;
            line-height: 1.35;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_bundle(path: str):
    return joblib.load(path)


def compute_bp_stage(ap_hi: float, ap_lo: float) -> int:
    if ap_hi >= 140 or ap_lo >= 90:
        return 3
    if ap_hi >= 130 or ap_lo >= 80:
        return 2
    if ap_hi >= 120:
        return 1
    return 0


BP_STAGE_LABELS = {0: "Normal", 1: "Elevated", 2: "Stage 1 Hypertension", 3: "Stage 2 Hypertension"}
BP_STAGE_COLORS = {0: "#22c55e", 1: "#eab308", 2: "#f97316", 3: "#ef4444"}


def build_feature_row(inputs: dict, all_raw_features: list) -> pd.DataFrame:
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
    return df[all_raw_features], bp_stage, bmi


def scroll_to(element_id: str):
    components.html(
        f"""
        <script>
            var el = window.parent.document.getElementById("{element_id}");
            if (el) {{
                el.scrollIntoView({{behavior: "smooth", block: "start"}});
            }}
        </script>
        """,
        height=0,
    )


def render_ps_item(label: str, value: str):
    st.markdown(
        f"""
        <div class="ps-item">
            <div class="ps-label">{label}</div>
            <div class="ps-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.title("🫀 Cardiovascular Disease Risk Calculator")
    st.markdown(
        """
        <div class="top-note">
        Predicts probability of cardiovascular disease from routine clinical and lifestyle inputs.
        This is a portfolio/educational model — not a medical device and should not be used for real diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
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

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        age = st.number_input("Age (yrs)", min_value=1, max_value=120, value=50)
    with r1c2:
        gender = st.selectbox("Gender", options=[("Female", 1), ("Male", 2)], format_func=lambda x: x[0])[1]
    with r1c3:
        height = st.number_input("Height (cm)", min_value=120, max_value=250, value=165)

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0, step=0.5)
    with r2c2:
        ap_hi = st.number_input("Systolic BP", min_value=60, max_value=280, value=120)
    with r2c3:
        ap_lo = st.number_input("Diastolic BP", min_value=40, max_value=180, value=80)

    r3c1, r3c2, r3c3, r3c4 = st.columns([2, 1, 2, 1])
    with r3c1:
        cholesterol_value = st.number_input("Cholesterol (mg/dL)", min_value=80, max_value=500, value=180)
        if cholesterol_value < 200:
            cholesterol = 1
        elif cholesterol_value < 240:
            cholesterol = 2
        else:
            cholesterol = 3
    with r3c2:
        st.markdown(
            f"<div class='cat-badge-wrap'><span class='cat-badge'>{['Normal', 'Above Normal', 'Well Above Normal'][cholesterol-1]}</span></div>",
            unsafe_allow_html=True,
        )
    with r3c3:
        glucose_value = st.number_input("Glucose (mg/dL)", min_value=40, max_value=400, value=90)
        if glucose_value < 100:
            gluc = 1
        elif glucose_value < 126:
            gluc = 2
        else:
            gluc = 3
    with r3c4:
        st.markdown(
            f"<div class='cat-badge-wrap'><span class='cat-badge'>{['Normal', 'Above Normal', 'Well Above Normal'][gluc-1]}</span></div>",
            unsafe_allow_html=True,
        )

    r4c1, r4c2, r4c3 = st.columns(3)
    with r4c1:
        active = st.selectbox("Active?", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
    with r4c2:
        smoke = st.selectbox("Smoker?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    with r4c3:
        alco = st.selectbox("Alcohol?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

    if ap_lo >= ap_hi:
        st.warning("Diastolic (ap_lo) should be lower than systolic (ap_hi). Please check your inputs.")

    _, center, _ = st.columns([1, 2, 1])

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
        risk = proba * 100

        risk_label = "High Risk" if pred == 1 else "Low Risk"
        risk_color = "#ef4444" if pred == 1 else "#22c55e"
        bp_color = BP_STAGE_COLORS[bp_stage]
        bp_label = BP_STAGE_LABELS[bp_stage]

        st.markdown('<div id="result_anchor"></div>', unsafe_allow_html=True)
        st.divider()
        st.header("Result")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""
                <div class="result-card" style="background:{risk_color}1A; border-color:{risk_color}66;">
                    <div class="rc-label">🟢 Predicted Risk</div>
                    <div class="rc-value" style="color:{risk_color};">{risk_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="result-card" style="background:{bp_color}1A; border-color:{bp_color}66;">
                    <div class="rc-label">🩺 Blood Pressure Stage</div>
                    <div class="rc-value" style="color:{bp_color};">{bp_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("##### Predicted CVD Probability")
        st.metric("Predicted CVD Probability", f"{risk:.1f}%", label_visibility="collapsed")
        st.progress(float(proba))

        if pred == 1:
            st.error(
                f"Model predicts **higher risk of cardiovascular disease** "
                f"(probability {risk:.1f}% ≥ threshold {threshold*100:.1f}%)."
            )
        else:
            st.success(
                f"Model predicts **lower risk of cardiovascular disease** "
                f"(probability {risk:.1f}% < threshold {threshold*100:.1f}%)."
            )

        st.markdown("###### 📋 Patient Summary")
        chol_label = {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[cholesterol]
        gluc_label = {1: "Normal", 2: "Prediabetes", 3: "Diabetes"}[gluc]
        map_value = ap_lo + (ap_hi - ap_lo) / 3

        c1, c2, c3 = st.columns(3)
        with c1:
            render_ps_item("Age", f"{age} yrs")
        with c2:
            render_ps_item("Gender", "Male" if gender == 2 else "Female")
        with c3:
            render_ps_item("BP Stage", bp_label)

        c1, c2, c3 = st.columns(3)
        with c1:
            render_ps_item("Pulse Pressure", f"{ap_hi - ap_lo} mmHg")
        with c2:
            render_ps_item("Cholesterol", f"{cholesterol_value} mg/dL")
        with c3:
            render_ps_item("Chol. Category", chol_label)

        c1, c2, c3 = st.columns(3)
        with c1:
            render_ps_item("Mean Arterial Pressure", f"{map_value:.2f} mmHg")
        with c2:
            render_ps_item("Glucose", f"{glucose_value} mg/dL")
        with c3:
            render_ps_item("Gluc. Category", gluc_label)

        c1, c2, c3 = st.columns(3)
        with c1:
            render_ps_item("Physical Activity", "Active" if active else "Inactive")
        with c2:
            render_ps_item("Smoking", "Yes" if smoke else "No")
        with c3:
            render_ps_item("Alcohol", "Yes" if alco else "No")

        st.markdown(
            """
            <div class="footer-note">
            This is a statistical model trained on a single public dataset. It does not account for family history,
            medications, or many other clinically relevant factors, and should not replace professional medical advice.
            </div>
            """,
            unsafe_allow_html=True,
        )

        scroll_to("result_anchor")


if __name__ == "__main__":
    main()