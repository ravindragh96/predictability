#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import tensorflow as tf
import joblib
from sklearn.metrics import mean_absolute_percentage_error

# -----------------------
# 1️⃣ Setup & Paths
# -----------------------
st.set_page_config(page_title="RSM 2D Contour App", layout="wide")
st.title("🎛️ Response Surface Modeling (RSM) — 2D Contour Visualization")

BASE_DIR = r"C:\Users\gantrav01\RD_predictability_11925"

TRAIN_X_PATH = os.path.join(BASE_DIR, "H_vs_Tau_training.xlsx")
TRAIN_Y_PATH = os.path.join(BASE_DIR, "H_vs_Tau_target.xlsx")
TEST_PATH = os.path.join(BASE_DIR, "Copy of T33_100_Samples_for_testing.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "h_vs_tau_best_model.keras")
X_SCALER_PATH = os.path.join(BASE_DIR, "x_eta_scaler.pkl")
Y_SCALER_PATH = os.path.join(BASE_DIR, "y_eta_scaler.pkl")

# -----------------------
# 2️⃣ Load Data
# -----------------------
def clean_cols(df):
    df = df.copy()
    # Replace non-alphanumeric with underscores
    df.columns = df.columns.str.strip().str.replace('[^A-Za-z0-9]+', '_', regex=True)
    # Remove trailing underscores
    df.columns = df.columns.str.rstrip('_')
    return df

X_train = clean_cols(pd.read_excel(TRAIN_X_PATH))
y_train = clean_cols(pd.read_excel(TRAIN_Y_PATH))
t33_df = clean_cols(pd.read_excel(TEST_PATH))

feature_cols = [c for c in X_train.columns if c in t33_df.columns]
X_test = t33_df[feature_cols]

target_cols = [c for c in y_train.columns if c in t33_df.columns]
y_actual_df = t33_df[target_cols] if target_cols else pd.DataFrame()

# Force H1 constant = 100
if "h1" in X_test.columns:
    X_test["h1"] = 100.0

# -----------------------
# 3️⃣ Load Model & Scalers
# -----------------------
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

try:
    x_scaler = joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)
    st.info("✅ Scalers loaded successfully.")
except Exception as e:
    st.error(f"❌ Error loading scalers: {e}")
    st.stop()

# -----------------------
# 4️⃣ Sidebar Controls
# -----------------------
st.sidebar.header("⚙️ Select RSM Inputs")

feature_x = st.sidebar.selectbox("Select Feature X (horizontal axis)", feature_cols, index=0)
feature_y = st.sidebar.selectbox("Select Feature Y (vertical axis)", feature_cols, index=1)
target_option = st.sidebar.selectbox("Select Target Output", list(y_train.columns))

if feature_x == feature_y:
    st.warning("Please select two **distinct** features for X and Y.")
    st.stop()

if not target_option:
    st.warning("Please select a target output.")
    st.stop()

# -----------------------
# 5️⃣ Prepare Test Data
# -----------------------
output_to_plot = target_option
output_index = y_train.columns.get_loc(output_to_plot)

if hasattr(x_scaler, "feature_names_in_"):
    all_features = list(x_scaler.feature_names_in_)
else:
    all_features = list(X_train.columns)

# Align & fill missing columns in X_test
X_mean = X_test.mean(numeric_only=True)
for col in all_features:
    if col not in X_test.columns:
        X_test[col] = 100.0 if col.lower() == "h1" else X_mean.mean()

X_test = X_test.reindex(columns=all_features, fill_value=0.0)

# -----------------------
# Validate selected features exist in columns
# -----------------------
f1, f2 = feature_x, feature_y

if f1 not in X_test.columns or f2 not in X_test.columns:
    st.error(f"Selected features '{f1}' or '{f2}' not found in test data columns.")
    st.write("Columns available:", list(X_test.columns))
    st.stop()

# -----------------------
# Scale test data and predict
# -----------------------
X_test_scaled = x_scaler.transform(X_test.astype(np.float32))
y_pred_scaled = model.predict(X_test_scaled, verbose=0)
y_pred = y_scaler.inverse_transform(y_pred_scaled)

# -----------------------
# 6️⃣ Compute MAPE
# -----------------------
if output_to_plot in y_actual_df.columns:
    y_actual = y_actual_df[output_to_plot].values
    eps = 1e-8
    mape_val = np.mean(np.abs((y_actual - y_pred[:, output_index]) / (y_actual + eps))) * 100
    st.success(f"✅ Verified MAPE for {output_to_plot}: {mape_val:.2f}% (Expected ≈ 3.33%)")
else:
    y_actual = np.zeros_like(y_pred[:, output_index])
    st.warning(f"⚠️ No actual values for {output_to_plot} found — skipping MAPE check.")

# -----------------------
# 7️⃣ Build Contour Grid (safe)
# -----------------------
f1_range = np.linspace(X_test[f1].min(), X_test[f1].max(), 60)
f2_range = np.linspace(X_test[f2].min(), X_test[f2].max(), 60)
F1, F2 = np.meshgrid(f1_range, f2_range)

grid = pd.DataFrame({f1: F1.ravel(), f2: F2.ravel()})
for colname in all_features:
    if colname not in [f1, f2]:
        grid[colname] = 100.0 if colname.lower() == "h1" else X_mean.get(colname, 0.0)

grid = grid.reindex(columns=all_features, fill_value=0.0)
grid_scaled = x_scaler.transform(grid.astype(np.float32))

preds_scaled = model.predict(grid_scaled, verbose=0)
preds = y_scaler.inverse_transform(preds_scaled)[:, output_index]
preds = preds.reshape(F1.shape)

# -----------------------
# 8️⃣ Plotly Contour Plot
# -----------------------
fig = go.Figure(data=go.Contour(
    z=preds,
    x=f1_range,
    y=f2_range,
    colorscale="Viridis",
    colorbar=dict(title=f"{output_to_plot} (Actual Scale)", titleside="right"),
    contours=dict(showlabels=True, labelfont=dict(size=12, color="white")),
    hovertemplate=(
        f"<b>{f1}</b>: %{{x:.3f}}<br>"
        f"<b>{f2}</b>: %{{y:.3f}}<br>"
        f"<b>Predicted {output_to_plot}</b>: %{{z:.3f}}<extra></extra>"
    ),
))

# Overlay actual points
if output_to_plot in y_actual_df.columns:
    fig.add_trace(go.Scatter(
        x=X_test[f1],
        y=X_test[f2],
        mode="markers",
        marker=dict(size=6, color="red", line=dict(width=1, color="black")),
        name=f"Actual {output_to_plot}",
        text=[
            f"<b>{f1}:</b>{X_test.at[i,f1]:.3f}<br>"
            f"<b>{f2}:</b>{X_test.at[i,f2]:.3f}<br>"
            f"<b>Actual {output_to_plot}:</b>{y_actual_df[output_to_plot].iloc[i]:.3f}"
            for i in range(len(X_test))
        ],
        hoverinfo="text"
    ))

fig.update_layout(
    title=f"RSM Contour: {f1} vs {f2} (H1 fixed at 100) — Output: {output_to_plot}",
    xaxis_title=f1,
    yaxis_title=f2,
    width=850,
    height=650,
    template="plotly_white",
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------
# 9️⃣ Display Sample Table
# -----------------------
st.markdown(f"### 🔍 Sample Predictions for `{output_to_plot}` (first 10 rows)")
compare_df = pd.DataFrame({
    f1: X_test[f1].values[:10],
    f2: X_test[f2].values[:10],
    f"Pred_{output_to_plot}": y_pred[:10, output_index],
})
if np.any(y_actual):
    compare_df[f"Actual_{output_to_plot}"] = y_actual[:10]
st.dataframe(compare_df)
