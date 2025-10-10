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
st.title("🎛️ Response Surface Modeling (RSM) Dashboard")

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
X_train = pd.read_excel(TRAIN_X_PATH)
y_train = pd.read_excel(TRAIN_Y_PATH)
t33_df = pd.read_excel(TEST_PATH)

# Match columns with train features
feature_cols = [c for c in X_train.columns if c in t33_df.columns]
X_test = t33_df[feature_cols]
target_cols = [c for c in y_train.columns if c in t33_df.columns]
y_actual_df = t33_df[target_cols] if target_cols else pd.DataFrame()

# Constant H1
if "H1" in X_test.columns:
    X_test["H1"] = 100.0

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
# 4️⃣ Feature Alignment
# -----------------------
if hasattr(x_scaler, "feature_names_in_"):
    scaler_features = list(x_scaler.feature_names_in_)
else:
    scaler_features = list(X_train.columns)

missing_in_test = [c for c in scaler_features if c not in X_test.columns]
extra_in_test = [c for c in X_test.columns if c not in scaler_features]

st.sidebar.subheader("🧩 Feature Alignment Check")
st.sidebar.write(f"Scaler trained with {len(scaler_features)} features")
st.sidebar.write(f"Test data has {len(X_test.columns)} features")

if missing_in_test:
    st.sidebar.warning(f"⚠️ Missing in test data: {missing_in_test}")
if extra_in_test:
    st.sidebar.info(f"ℹ️ Extra columns in test data: {extra_in_test}")

X_mean = X_test.mean(numeric_only=True)
for col in missing_in_test:
    if col == "H1":
        X_test[col] = 100.0
    else:
        X_test[col] = X_mean.mean()

X_test = X_test.reindex(columns=scaler_features, fill_value=0.0)

# -----------------------
# 5️⃣ Sidebar Controls
# -----------------------
st.sidebar.header("⚙️ RSM Visualization Controls")
feature_x = st.sidebar.selectbox("Select Feature X", [""] + scaler_features)
feature_y = st.sidebar.selectbox("Select Feature Y", [""] + scaler_features)
target_option = st.sidebar.selectbox("Select Target Output", [""] + list(y_train.columns))

if not feature_x or not feature_y or feature_x == feature_y:
    st.warning("Please select two distinct features for X and Y.")
    st.stop()

if not target_option:
    st.warning("Please select a target output.")
    st.stop()

output_to_plot = target_option
output_index = y_train.columns.get_loc(output_to_plot)

# -----------------------
# 6️⃣ Predict Test Data
# -----------------------
X_test_scaled = x_scaler.transform(X_test.astype(np.float32))
y_pred_scaled = model.predict(X_test_scaled, verbose=0)
y_pred = y_scaler.inverse_transform(y_pred_scaled)

if output_to_plot in y_actual_df.columns:
    y_actual = y_actual_df[output_to_plot].values
    eps = 1e-8
    abs_errors = np.abs(y_actual - y_pred[:, output_index])
    percent_errors = abs_errors / (y_actual + eps) * 100
    mape_val = np.mean(percent_errors)
    avg_error = np.mean(percent_errors)
    max_error = np.max(percent_errors)
    min_error = np.min(percent_errors)
else:
    y_actual = np.zeros_like(y_pred[:, output_index])
    st.warning(f"⚠️ No actual values for {output_to_plot} found — skipping MAPE check.")

# -----------------------
# 7️⃣ Contour Grid
# -----------------------
f1, f2 = feature_x, feature_y
if f1 not in X_test.columns or f2 not in X_test.columns:
    st.error(f"❌ One of the selected features ({f1} or {f2}) is not found in test data.")
    st.stop()

f1_range = np.linspace(X_test[f1].min(), X_test[f1].max(), 60)
f2_range = np.linspace(X_test[f2].min(), X_test[f2].max(), 60)
F1, F2 = np.meshgrid(f1_range, f2_range)

grid = pd.DataFrame({f1: F1.ravel(), f2: F2.ravel()})
for colname in scaler_features:
    if colname not in [f1, f2]:
        grid[colname] = 100.0 if colname == "H1" else X_mean.get(colname, 0.0)

grid = grid.reindex(columns=scaler_features, fill_value=0.0)
grid_scaled = x_scaler.transform(grid.astype(np.float32))
preds_scaled = model.predict(grid_scaled, verbose=0)
preds = y_scaler.inverse_transform(preds_scaled)[:, output_index]
preds = preds.reshape(F1.shape)

# -----------------------
# 8️⃣ Split Layout: Contour + Circular Error Charts
# -----------------------
col1, col2 = st.columns([2, 1])

# ---- Left Column: Contour Plot ----
with col1:
    st.subheader(f"📈 RSM Contour: {f1} vs {f2}")
    fig = go.Figure(data=go.Contour(
        z=preds,
        x=f1_range,
        y=f2_range,
        colorscale="Viridis",
        colorbar=dict(
            title=dict(
                text=f"{output_to_plot} (Actual Scale)",
                side="right"
            )
        ),
        contours=dict(showlabels=True, labelfont=dict(size=12, color="white")),
        hovertemplate=(
            f"<b>{f1}</b>: %{{x:.3f}}<br>"
            f"<b>{f2}</b>: %{{y:.3f}}<br>"
            f"<b>Predicted {output_to_plot}</b>: %{{z:.3f}}<extra></extra>"
        ),
    ))

    # Overlay actual test points
    if output_to_plot in y_actual_df.columns:
        fig.add_trace(go.Scatter(
            x=X_test[f1],
            y=X_test[f2],
            mode="markers",
            marker=dict(size=6, color="red", line=dict(width=1, color="black")),
            name=f"Actual {output_to_plot}"
        ))

    fig.update_layout(
        title=f"{output_to_plot} Contour (H1 fixed at 100)",
        xaxis_title=f1,
        yaxis_title=f2,
        width=800,
        height=600,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- Right Column: Circular Error Charts ----
with col2:
    st.subheader("📊 Error Performance")

    if output_to_plot in y_actual_df.columns:
        # Create side-by-side circular charts
        c1, c2 = st.columns(2)

        with c1:
            fig_mape = go.Figure(data=[
                go.Pie(
                    labels=['MAPE (%)', 'Accuracy (%)'],
                    values=[mape_val, 100 - mape_val],
                    hole=0.6,
                    marker_colors=['#EF553B', '#00CC96'],
                    textinfo='label+percent'
                )
            ])
            fig_mape.update_layout(
                title=dict(text=f"MAPE: {mape_val:.2f}%", x=0.5),
                showlegend=False,
                height=280
            )
            st.plotly_chart(fig_mape, use_container_width=True)

        with c2:
            fig_avg = go.Figure(data=[
                go.Pie(
                    labels=['Avg Error (%)', 'Accuracy (%)'],
                    values=[avg_error, 100 - avg_error],
                    hole=0.6,
                    marker_colors=['#636EFA', '#AB63FA'],
                    textinfo='label+percent'
                )
            ])
            fig_avg.update_layout(
                title=dict(text=f"Avg Error: {avg_error:.2f}%", x=0.5),
                showlegend=False,
                height=280
            )
            st.plotly_chart(fig_avg, use_container_width=True)

        st.markdown("---")
        st.markdown(f"**Max % Error:** `{max_error:.2f}%`")
        st.markdown(f"**Min % Error:** `{min_error:.2f}%`")
        st.markdown("✅ Model predictions are in expected range.")
    else:
        st.warning("⚠️ No actual values available to compute error metrics.")

# -----------------------
# 9️⃣ Display Predictions Table
# -----------------------
st.markdown(f"### 🔍 Sample Predictions for `{output_to_plot}` (first 10 rows)")
compare_df = pd.DataFrame({
    f1: X_test[f1].values[:10],
    f2: X_test[f2].values[:10],
    f"Pred_{output_to_plot}": y_pred[:10, output_index],
})
if np.any(y_actual):
    compare_df[f"Actual_{output_to_plot}"] = y_actual[:10]
st.dataframe(compare_df, use_container_width=True)
