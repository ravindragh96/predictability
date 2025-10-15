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
st.set_page_config(page_title="RSM Comparison Dashboard", layout="wide")
st.title("🎛️ Response Surface Modeling (RSM) — Real vs Synthetic Comparison (Optimized)")

BASE_DIR = r"C:\Users\gantrav01\RD_predictability_11925"

TRAIN_X_PATH = os.path.join(BASE_DIR, "H_vs_Tau_training.xlsx")
TRAIN_Y_PATH = os.path.join(BASE_DIR, "H_vs_Tau_target.xlsx")
SYNTH_PATH   = os.path.join(BASE_DIR, "synthetic_tau_98.xlsx")
TEST_PATH    = os.path.join(BASE_DIR, "Copy of T33_100_Samples_for_testing.xlsx")
MODEL_PATH   = os.path.join(BASE_DIR, "checkpoints", "h_vs_tau_best_model.keras")
X_SCALER_PATH = os.path.join(BASE_DIR, "x_eta_scaler.pkl")
Y_SCALER_PATH = os.path.join(BASE_DIR, "y_eta_scaler.pkl")

# -----------------------
# 2️⃣ Caching for Speed
# -----------------------
@st.cache_resource
def load_model_and_scalers():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    x_scaler = joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)
    return model, x_scaler, y_scaler

@st.cache_data
def load_datasets():
    X_train = pd.read_excel(TRAIN_X_PATH)
    y_train = pd.read_excel(TRAIN_Y_PATH)
    test_df = pd.read_excel(TEST_PATH)
    synth_df = pd.read_excel(SYNTH_PATH)
    return X_train, y_train, test_df, synth_df

model, x_scaler, y_scaler = load_model_and_scalers()
X_train, y_train, test_df, synth_df = load_datasets()

# -----------------------
# 3️⃣ Prepare Data
# -----------------------
feature_cols = X_train.columns.tolist()
target_cols = y_train.columns.tolist()
target_output = "T1"

X_test = test_df[feature_cols]
y_actual_df = test_df[target_cols]
X_synth = synth_df[feature_cols]
y_synth_df = synth_df[target_cols]

if "H1" in X_test.columns:
    X_test["H1"] = 100.0
if "H1" in X_synth.columns:
    X_synth["H1"] = 100.0

output_index = y_train.columns.get_loc(target_output)

# -----------------------
# 4️⃣ Sidebar Controls
# -----------------------
st.sidebar.header("⚙️ RSM Controls")
feature_x = st.sidebar.selectbox("Select Feature X", feature_cols)
feature_y = st.sidebar.selectbox("Select Feature Y", [c for c in feature_cols if c != feature_x])

x_min, x_max = float(X_test[feature_x].min()), float(X_test[feature_x].max())
y_min, y_max = float(X_test[feature_y].min()), float(X_test[feature_y].max())

x_range = st.sidebar.slider(f"{feature_x} Range", min_value=x_min, max_value=x_max, value=(x_min, x_max))
y_range = st.sidebar.slider(f"{feature_y} Range", min_value=y_min, max_value=y_max, value=(y_min, y_max))

show_free_synth = st.sidebar.checkbox("Show Synthetic (Free Features)", value=True)

# -----------------------
# 5️⃣ Helper Functions
# -----------------------
def predict_df(df):
    scaled = x_scaler.transform(df.astype(np.float32))
    preds = model.predict(scaled, verbose=0)
    return y_scaler.inverse_transform(preds)

# -----------------------
# 6️⃣ Compute Predictions
# -----------------------
X_mean = X_test.mean(numeric_only=True)

# --- Constant Feature Grid ---
f1_range = np.linspace(x_range[0], x_range[1], 40)
f2_range = np.linspace(y_range[0], y_range[1], 40)
F1, F2 = np.meshgrid(f1_range, f2_range)

grid_const = pd.DataFrame({feature_x: F1.ravel(), feature_y: F2.ravel()})
for col in feature_cols:
    if col not in [feature_x, feature_y]:
        grid_const[col] = 100.0 if col == "H1" else X_mean[col]
grid_const = grid_const[feature_cols]

pred_const = predict_df(grid_const)[:, output_index].reshape(F1.shape)
real_preds = predict_df(X_test)[:, output_index]
actual_vals = y_actual_df[target_output].values

# --- Free Synthetic Prediction (cached) ---
if show_free_synth:
    pred_free = predict_df(X_synth)[:, output_index]
else:
    pred_free = None

# -----------------------
# 7️⃣ Filtering for Sliders
# -----------------------
mask = (
    (X_test[feature_x] >= x_range[0]) & (X_test[feature_x] <= x_range[1]) &
    (X_test[feature_y] >= y_range[0]) & (X_test[feature_y] <= y_range[1])
)
filtered_X = X_test[mask]
filtered_actual = actual_vals[mask]
filtered_pred = real_preds[mask]

# -----------------------
# 8️⃣ Error Metrics
# -----------------------
eps = 1e-8
percent_errors = np.abs((filtered_actual - filtered_pred) / (filtered_actual + eps)) * 100
global_mape = np.mean(np.abs((actual_vals - real_preds) / (actual_vals + eps)) * 100)
local_mape = np.mean(percent_errors)

# -----------------------
# 9️⃣ Shared Scale
# -----------------------
zmin = np.min([np.min(pred_const), np.min(real_preds)])
zmax = np.max([np.max(pred_const), np.max(real_preds)])

# -----------------------
# 🔟 RSM Plots
# -----------------------
fig_real = go.Figure(data=go.Contour(
    x=X_test[feature_x], y=X_test[feature_y], z=real_preds,
    colorscale="RdYlGn_r", zmin=zmin, zmax=zmax,
    colorbar=dict(title=f"{target_output}"), contours=dict(showlabels=True)
))
fig_real.add_trace(go.Scatter(
    x=filtered_X[feature_x], y=filtered_X[feature_y],
    mode="markers", marker=dict(size=7, color="black", line=dict(width=1, color="white")),
    text=[f"{target_output}: {val:.3f}" for val in filtered_pred],
    name="Filtered Points"
))
fig_real.update_layout(title="🟩 Real Data RSM", xaxis_title=feature_x, yaxis_title=feature_y, template="plotly_white")

fig_synth_const = go.Figure(data=go.Contour(
    x=f1_range, y=f2_range, z=pred_const,
    colorscale="RdYlGn_r", zmin=zmin, zmax=zmax,
    colorbar=dict(title=f"{target_output}"), contours=dict(showlabels=True)
))
fig_synth_const.update_layout(title="🟨 Synthetic RSM (Constant Features)", xaxis_title=feature_x, yaxis_title=feature_y, template="plotly_white")

# Free synthetic plot (optional)
if show_free_synth:
    fig_synth_free = go.Figure(data=go.Contour(
        x=X_synth[feature_x], y=X_synth[feature_y], z=pred_free,
        colorscale="RdYlGn_r", zmin=zmin, zmax=zmax,
        colorbar=dict(title=f"{target_output}"), contours=dict(showlabels=True)
    ))
    fig_synth_free.update_layout(title="🟦 Synthetic RSM (All Features Free)", xaxis_title=feature_x, yaxis_title=feature_y, template="plotly_white")

# -----------------------
# 11️⃣ Display Layout
# -----------------------
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_real, use_container_width=True)
with col2:
    st.plotly_chart(fig_synth_const, use_container_width=True)

if show_free_synth:
    st.plotly_chart(fig_synth_free, use_container_width=True)

# -----------------------
# 12️⃣ Donut Charts
# -----------------------
col_d1, col_d2 = st.columns(2)
with col_d1:
    fig_mape = go.Figure(data=[go.Pie(labels=['MAPE', 'Accuracy'], values=[global_mape, 100 - global_mape], hole=0.6)])
    fig_mape.update_layout(title_text=f"🌍 Global MAPE: {global_mape:.2f}%", showlegend=False)
    st.plotly_chart(fig_mape, use_container_width=True)

with col_d2:
    fig_local = go.Figure(data=[go.Pie(labels=['Local MAPE', 'Accuracy'], values=[local_mape, 100 - local_mape], hole=0.6)])
    fig_local.update_layout(title_text=f"📍 Local MAPE: {local_mape:.2f}%", showlegend=False)
    st.plotly_chart(fig_local, use_container_width=True)

# -----------------------
# 13️⃣ Data Table
# -----------------------
st.subheader("📋 Matched Data Points (Filtered by X–Y Range)")
compare_df = pd.DataFrame({
    feature_x: filtered_X[feature_x].values,
    feature_y: filtered_X[feature_y].values,
    f"Actual_{target_output}": filtered_actual,
    f"Predicted_{target_output}": filtered_pred,
    "Error_%": percent_errors
})
st.dataframe(compare_df, use_container_width=True)
