#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import tensorflow as tf
import joblib

# -----------------------
# 1️⃣ Setup
# -----------------------
st.set_page_config(page_title="RSM Contour App", layout="wide")
st.title("🎛️ Response Surface Modeling (RSM) — Synthetic ANN Predictions")

BASE_DIR = r"C:\Users\gantrav01\RD_predictability_11925"

TRAIN_X_PATH = os.path.join(BASE_DIR, "h_vs_eta_training.xlsx")
TRAIN_Y_PATH = os.path.join(BASE_DIR, "H_vs_eta_Target.xlsx")
SYNTHETIC_PATH = os.path.join(BASE_DIR, "synthetic_eta_98.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "h_vs_eta_best_model.keras")
X_SCALER_PATH = os.path.join(BASE_DIR, "x_eta_scaler.pkl")
Y_SCALER_PATH = os.path.join(BASE_DIR, "y_eta_scaler.pkl")

# -----------------------
# 2️⃣ Load Data & Model
# -----------------------
try:
    X_train = pd.read_excel(TRAIN_X_PATH)
    y_train = pd.read_excel(TRAIN_Y_PATH)
    df = pd.read_excel(SYNTHETIC_PATH)
    st.success("✅ Synthetic dataset loaded successfully.")
except Exception as e:
    st.error(f"❌ Could not load files: {e}")
    st.stop()

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    x_scaler = joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)
    st.info("✅ Model and scalers loaded successfully.")
except Exception as e:
    st.error(f"❌ Could not load model/scalers: {e}")
    st.stop()

# -----------------------
# 3️⃣ Sidebar Controls
# -----------------------
st.sidebar.header("⚙️ RSM Visualization Controls")

input_cols = list(X_train.columns)
output_cols = list(y_train.columns)

feature_x = st.sidebar.selectbox("Select Feature X", input_cols)
feature_y = st.sidebar.selectbox("Select Feature Y", input_cols)
target_col = st.sidebar.selectbox("Select Target Output (Z)", output_cols)

grid_resolution = st.sidebar.slider("Grid Resolution", 30, 100, 60)

# -----------------------
# 4️⃣ Feature Alignment
# -----------------------
if hasattr(x_scaler, "feature_names_in_"):
    scaler_features = list(x_scaler.feature_names_in_)
else:
    scaler_features = input_cols

missing_in_df = [c for c in scaler_features if c not in df.columns]
X_mean = df.mean(numeric_only=True)

# Add missing columns if any
for col in missing_in_df:
    if col == "H1":
        df[col] = 100.0
    else:
        df[col] = X_mean.get(col, 0.0)

df = df.reindex(columns=scaler_features, fill_value=0.0)

# -----------------------
# 5️⃣ Generate Synthetic Grid (Keep X,Y as is)
# -----------------------
f1, f2 = feature_x, feature_y

f1_range = np.linspace(df[f1].min(), df[f1].max(), grid_resolution)
f2_range = np.linspace(df[f2].min(), df[f2].max(), grid_resolution)
F1, F2 = np.meshgrid(f1_range, f2_range)

# Build grid keeping X,Y variable — others constant
grid = pd.DataFrame({f1: F1.ravel(), f2: F2.ravel()})
for col in scaler_features:
    if col not in [f1, f2]:
        grid[col] = 100.0 if col == "H1" else X_mean.get(col, 0.0)

# Align column order exactly to training scaler
grid = grid.reindex(columns=scaler_features, fill_value=0.0)

# -----------------------
# 6️⃣ Predict with ANN Model
# -----------------------
grid_scaled = x_scaler.transform(grid.astype(np.float32))
preds_scaled = model.predict(grid_scaled, verbose=0)
preds = y_scaler.inverse_transform(preds_scaled)[:, y_train.columns.get_loc(target_col)]
preds = preds.reshape(F1.shape)

# -----------------------
# 7️⃣ Plot Contour
# -----------------------
st.subheader(f"📈 Response Surface — {target_col} vs {f1} & {f2}")

fig = go.Figure()

# Contour plot (RSM)
fig.add_trace(go.Contour(
    z=preds,
    x=f1_range,
    y=f2_range,
    colorscale="RdYlGn_r",
    ncontours=30,
    colorbar=dict(title=dict(text=f"{target_col}", font=dict(size=12))),
    hovertemplate=(
        f"<b>{f1}</b>: %{{x:.3f}}<br>"
        f"<b>{f2}</b>: %{{y:.3f}}<br>"
        f"<b>Predicted {target_col}</b>: %{{z:.3f}}<extra></extra>"
    )
))

# Overlay the original synthetic data points
fig.add_trace(go.Scatter(
    x=df[f1],
    y=df[f2],
    mode="markers",
    name="Synthetic Data Points",
    marker=dict(size=6, color="blue", line=dict(width=0.5, color="white")),
    hovertext=[
        f"{f1}: {xv:.3f}<br>{f2}: {yv:.3f}" for xv, yv in zip(df[f1], df[f2])
    ],
    hoverinfo="text"
))

fig.update_layout(
    title=f"{target_col} Contour Plot (Predicted via ANN Model)",
    xaxis_title=f1,
    yaxis_title=f2,
    width=900,
    height=600,
    template="simple_white",
    font=dict(size=12)
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------
# 8️⃣ Sample Predicted Values
# -----------------------
st.markdown(f"### 🔍 Sample Predicted Values for `{target_col}` (first 10 rows)")
sample_df = grid[[f1, f2]].copy().head(10)
sample_df[f"Predicted_{target_col}"] = preds.ravel()[:10]
st.dataframe(sample_df, use_container_width=True)

st.info("""
✅ **Interpretation:**
- Red → Higher predicted output  
- Green →
