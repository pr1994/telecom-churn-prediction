import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📡", layout="centered")

with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.markdown("""
<style>
.stApp {
    background-color: #0b1020;
    background-image:
        radial-gradient(circle at 50% 50%, rgba(56,189,248,0.10) 0%, rgba(56,189,248,0) 45%),
        repeating-radial-gradient(circle at 50% 50%, rgba(56,189,248,0.05) 0px, rgba(56,189,248,0.05) 1px, transparent 1px, transparent 80px),
        repeating-conic-gradient(rgba(56,189,248,0.04) 0deg 2deg, transparent 2deg 90deg);
    background-position: center;
}

.block-container {
    padding-top: 2rem;
    max-width: 760px;
}

.hero {
    text-align: center;
    padding: 1.5rem 0 1rem 0;
}

.hero h1 {
    font-size: 2rem;
    font-weight: 600;
    color: #e7f6ff;
    letter-spacing: 0.5px;
    margin-bottom: 0.25rem;
}

.hero p {
    color: #7fa8c9;
    font-size: 0.95rem;
}

.section-card {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
    backdrop-filter: blur(6px);
}

.section-title {
    color: #5fd4ff;
    font-size: 0.8rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1rem;
    font-weight: 600;
}

div[data-testid="stSlider"] label, div[data-testid="stSelectbox"] label {
    color: #cfe8f7 !important;
    font-size: 0.85rem !important;
}

.stButton button {
    background: linear-gradient(90deg, #0ea5e9, #38bdf8);
    color: #06121f;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 0;
    width: 100%;
    letter-spacing: 0.5px;
}

.stButton button:hover {
    background: linear-gradient(90deg, #38bdf8, #7dd3fc);
    color: #06121f;
}

.result-card {
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    margin-top: 1rem;
    border: 1px solid;
}

.result-high {
    background: rgba(239, 68, 68, 0.12);
    border-color: rgba(239, 68, 68, 0.4);
}

.result-low {
    background: rgba(34, 197, 94, 0.12);
    border-color: rgba(34, 197, 94, 0.4);
}

.result-label {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
}

.result-high .result-label { color: #fca5a5; }
.result-low .result-label { color: #86efac; }

.result-prob {
    font-size: 2.2rem;
    font-weight: 700;
    color: #f1f5f9;
}

footer, #MainMenu, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>📡 Telecom churn radar</h1>
    <p>Predict customer churn risk in real time using a trained Random Forest model</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Account profile</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    account_length = st.slider("Account length (days)", 0, 250, 100)
    area_code = st.selectbox("Area code", [408, 415, 510])
with c2:
    international_plan = st.selectbox("International plan", ["No", "Yes"])
    voice_mail_plan = st.selectbox("Voice mail plan", ["No", "Yes"])
number_vmail_messages = st.slider("Number of voicemail messages", 0, 50, 0)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Usage patterns</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    total_day_minutes = st.slider("Total day minutes", 0.0, 350.0, 180.0)
    total_day_calls = st.slider("Total day calls", 0, 165, 100)
    total_eve_minutes = st.slider("Total evening minutes", 0.0, 350.0, 200.0)
    total_eve_calls = st.slider("Total evening calls", 0, 170, 100)
with c2:
    total_night_minutes = st.slider("Total night minutes", 0.0, 400.0, 200.0)
    total_night_calls = st.slider("Total night calls", 0, 175, 100)
    total_intl_minutes = st.slider("Total international minutes", 0.0, 20.0, 10.0)
    total_intl_calls = st.slider("Total international calls", 0, 20, 4)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Support history</div>', unsafe_allow_html=True)
customer_service_calls = st.slider("Customer service calls", 0, 9, 1)
st.markdown('</div>', unsafe_allow_html=True)

if st.button("🔍 Predict churn risk"):
    intl_plan_encoded = 1 if international_plan == "Yes" else 0
    vmail_plan_encoded = 1 if voice_mail_plan == "Yes" else 0

    input_data = np.array([[
        account_length, area_code, intl_plan_encoded, vmail_plan_encoded,
        number_vmail_messages, total_day_minutes, total_day_calls,
        total_eve_minutes, total_eve_calls, total_night_minutes,
        total_night_calls, total_intl_minutes, total_intl_calls,
        customer_service_calls
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.markdown(f"""
        <div class="result-card result-high">
            <div class="result-label">🔴 High risk — customer likely to churn</div>
            <div class="result-prob">{probability*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card result-low">
            <div class="result-label">🟢 Low risk — customer likely to stay</div>
            <div class="result-prob">{probability*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.progress(float(probability))