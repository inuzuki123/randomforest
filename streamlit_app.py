import streamlit as st
import pandas as pd
import joblib

# MODEL INLADEN
model = joblib.load("rf_model.pkl")
reversedmodel = joblib.load("reversed_rf_model.pkl")

st.title("📊 Trading Model - Setup Scorer")

st.write("Vul je trade in en krijg direct Expected Value + TAKE/SKIP")

# -----------------------------
# INPUTS (jouw features)
# -----------------------------

col1, col2, col3 = st.columns(3)

with col1:
    trend_15m = st.selectbox("15m trend", [-1, 1])
    trend_1h = st.selectbox("1h trend", [-1, 1])
    trend_4h = st.selectbox("4h trend", [-1, 1])

with col2:
    dtrend = st.selectbox("Daily trend", [-1, 1])
    move = st.selectbox("Move", [-1, 1])
    liquidity = st.selectbox("Liquidity", [-1, 1])

with col3:
    fibzone = st.selectbox("Fibzone", [-1, 1])
    doubletop = st.selectbox("Double Top", [-1, 1])
    trending = st.selectbox("Trending", [-1, 1])

# -----------------------------
# INPUT DATAFRAME
# -----------------------------

input_data = pd.DataFrame([{
    "15mtrend": trend_15m,
    "1htrend": trend_1h,
    "4htrend": trend_4h,
    "dtrend": dtrend,
    "move": move,
    "liquidity": liquidity,
    "fibzone": fibzone,
    "doubletop": doubletop,
    "trending": trending
}])

# -----------------------------
# PREDICT
# -----------------------------

if st.button("🚀 Predict Trade"):

    prediction = model.predict(input_data)[0]
    reversedPrediction = reversedmodel.predict(input_data)[0]

    st.subheader("📈 Expected Values")
    st.write(f"Long model: {round(prediction, 4)}")
    st.write(f"Short model: {round(reversedPrediction, 4)}")

    st.write("---")

    # beslissing
    if prediction > reversedPrediction and prediction > 0.0855:
        st.success("✅ TAKE NORMAL")
    elif reversedPrediction > prediction and reversedPrediction > 0.211:
        st.success("🔻 TAKE REVERSED")
    else:
        st.error("❌ SKIP TRADE")

# -----------------------------
# EXTRA INFO
# -----------------------------

st.write("---")
st.write("Model output = expected profit per trade (not win/loss probability)")