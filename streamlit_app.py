import streamlit as st
import pandas as pd
import numpy as np
import math, os
from datetime import datetime

st.set_page_config(page_title="NYC Taxi Predictor", page_icon="🚕", layout="wide")

@st.cache_resource
def load_models():
    models_dir = "models"
    required = ["LR_coefficients.npy", "LR_intercept.npy",
                "scaler_mean.npy", "scaler_std.npy"]
    if not all(os.path.exists(f"{models_dir}/{f}") for f in required):
        return {}
    coef      = np.load(f"{models_dir}/LR_coefficients.npy")
    intercept = np.load(f"{models_dir}/LR_intercept.npy")[0]
    mean      = np.load(f"{models_dir}/scaler_mean.npy")
    std       = np.load(f"{models_dir}/scaler_std.npy")
    def predict_fn(feature_vals):
        scaled   = (np.array(feature_vals) - mean) / std
        log_pred = np.dot(scaled, coef) + intercept
        return float(np.expm1(log_pred))
    return {"Linear Regression": predict_fn}

def calc_features(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
                  dt_str, vendor_id, pax):
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    hour  = dt.hour
    dow   = dt.isoweekday() % 7 + 1
    month = dt.month
    day   = dt.day
    is_weekend  = 1 if dow in (1, 7) else 0
    is_rush     = 1 if (hour in range(7,10) or hour in range(16,20)) and not is_weekend else 0
    R = 6371.0
    lat1, lat2 = math.radians(pickup_lat),  math.radians(dropoff_lat)
    lon1, lon2 = math.radians(pickup_lon),  math.radians(dropoff_lon)
    dlat = lat2-lat1; dlon = lon2-lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    haversine = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    x = math.sin(dlon)*math.cos(lat2)
    y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
    bearing   = (math.degrees(math.atan2(x, y)) + 360) % 360
    manhattan = abs(dropoff_lat-pickup_lat) + abs(dropoff_lon-pickup_lon)
    mid_lat   = (pickup_lat+dropoff_lat)/2
    mid_lon   = (pickup_lon+dropoff_lon)/2
    return [hour, dow, month, day, is_weekend, is_rush,
            pax, vendor_id, 0,
            haversine, manhattan, bearing,
            pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
            mid_lat, mid_lon], haversine

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🚕 NYC Taxi Trip Duration Predictor")
st.caption("Big Data Project · PySpark MLlib · 1.46M rows")

models = load_models()

if not models:
    st.error("No saved models found. Run the training notebook first.")
    st.stop()

tab1, tab2 = st.tabs(["🔮 Single Prediction", "📁 Batch Upload (BONUS)"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📍 Pickup")
        pickup_lat = st.number_input("Pickup Latitude",  value=40.7614, format="%.4f")
        pickup_lon = st.number_input("Pickup Longitude", value=-73.9776, format="%.4f")
        vendor_id  = st.selectbox("Vendor", [1, 2])
        pax        = st.slider("Passengers", 1, 6, 2)
    with col2:
        st.subheader("🏁 Dropoff & Time")
        dropoff_lat = st.number_input("Dropoff Latitude",  value=40.6501, format="%.4f")
        dropoff_lon = st.number_input("Dropoff Longitude", value=-73.9496, format="%.4f")
        dt_str      = st.text_input("Pickup DateTime", value="2016-03-14 08:30:00")
        model_name  = st.selectbox("Model", list(models.keys()))

    if st.button("🚀 Predict"):
        feats, hav = calc_features(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
                                   dt_str, vendor_id, pax)
        pred_sec = max(60, models[model_name](feats))
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Duration", f"{pred_sec/60:.1f} min")
        c2.metric("In Seconds", f"{pred_sec:.0f} s")
        c3.metric("Distance", f"{hav:.2f} km")
        st.success(f"Model: {model_name}")

with tab2:
    st.info("Upload a CSV with the same columns as train.csv")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        pdf = pd.read_csv(uploaded)
        st.write(f"Loaded {len(pdf):,} rows")
        st.dataframe(pdf.head())
        model_b = st.selectbox("Model for batch", list(models.keys()), key="batch")
        if st.button("Run Batch Prediction"):
            pdf["pickup_datetime"] = pd.to_datetime(pdf["pickup_datetime"])
            preds = []
            for _, row in pdf.iterrows():
                feats, _ = calc_features(
                    row["pickup_latitude"], row["pickup_longitude"],
                    row["dropoff_latitude"], row["dropoff_longitude"],
                    str(row["pickup_datetime"]), row["vendor_id"], row["passenger_count"]
                )
                preds.append(max(60, models[model_b](feats)))
            pdf["predicted_sec"] = preds
            pdf["predicted_min"] = (pdf["predicted_sec"] / 60).round(2)
            st.dataframe(pdf[["id","predicted_sec","predicted_min"]].head(20))
            st.download_button("⬇️ Download CSV",
                               pdf.to_csv(index=False), "predictions.csv")