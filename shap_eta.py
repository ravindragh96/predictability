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
st.title("🎛️ Response Surface Modeling (RSM) — Full Dashboard with Error Visualization")

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

feature_cols = [c for c in X_train.columns if c in t33_df.columns]
X_test = t33_df[feature_cols]
target_cols = [c for c in y_train.columns if c in t33_df.columns]
y_actual_df = t33_df[target_cols] if target_cols else pd.DataFrame()

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
# 4️⃣ Align features
# -----------------------
if hasattr(x_scaler, "feature_names_in_"):
    scaler_features = list(x_scaler.feature_names_in_)
else:
    scaler_features = list(X_train.columns)

X_mean = X_test.mean(numeric_only=True)
missing_in_test = [c for c in scaler_features if c not in X_test.columns]
for col in missing_in_test:
    X_test[col] = X_mean.mean() if col != "H1" else 100.0

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
    global_mape = np.mean(percent_errors)
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
# 8️⃣ Local MAPE
# -----------------------
f1_min, f1_max = X_test[f1].min(), X_test[f1].max()
f2_min, f2_max = X_test[f2].min(), X_test[f2].max()

f1_margin = (f1_max - f1_min) * 0.10
f2_margin = (f2_max - f2_min) * 0.10

mask = (
    (X_test[f1] >= (f1_min + f1_margin)) & (X_test[f1] <= (f1_max - f1_margin)) &
    (X_test[f2] >= (f2_min + f2_margin)) & (X_test[f2] <= (f2_max - f2_margin))
)

local_errors = percent_errors[mask]
local_mape = np.mean(local_errors) if len(local_errors) > 0 else np.nan

# -----------------------
# 9️⃣ Contour + Donuts
# -----------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📈 RSM Contour: {f1} vs {f2}")

    fig = go.Figure(data=go.Contour(
        z=preds,
        x=f1_range,
        y=f2_range,
        colorscale="Viridis",
        colorbar=dict(title=f"{output_to_plot} (Actual Scale)"),
        contours=dict(showlabels=True, labelfont=dict(size=12, color="white")),
        hovertemplate=(
            f"<b>{f1}</b>: %{{x:.3f}}<br>"
            f"<b>{f2}</b>: %{{y:.3f}}<br>"
            f"<b>Predicted {output_to_plot}</b>: %{{z:.3f}}<extra></extra>"
        )
    ))

    hover_texts = [
        f"<b>{f1}:</b> {X_test.at[i, f1]:.3f}<br>"
        f"<b>{f2}:</b> {X_test.at[i, f2]:.3f}<br>"
        f"<b>Actual {output_to_plot}:</b> {y_actual[i]:.3f}<br>"
        f"<b>Predicted {output_to_plot}:</b> {y_pred[i, output_index]:.3f}<br>"
        f"<b>Error %:</b> {percent_errors[i]:.2f}%"
        for i in range(len(X_test))
    ]

    fig.add_trace(go.Scatter(
        x=X_test[f1],
        y=X_test[f2],
        mode="markers",
        marker=dict(size=6, color="red", line=dict(width=1, color="black")),
        name="Actual Points",
        text=hover_texts,
        hoverinfo="text"
    ))

    fig.update_layout(
        title=f"{output_to_plot} Contour (H1 fixed at 100)",
        xaxis_title=f1,
        yaxis_title=f2,
        width=850,
        height=600,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- Right Column: Donut Charts ----
with col2:
    st.subheader("📊 Error Performance Summary")

    c1, c2, c3 = st.columns(3)
    with c1:
        fig_mape = go.Figure(data=[
            go.Pie(labels=['MAPE (%)', 'Accuracy (%)'],
                   values=[global_mape, 100 - global_mape],
                   hole=0.6, marker_colors=['#EF553B', '#00CC96'],
                   textinfo='label+percent')
        ])
        fig_mape.update_layout(title=dict(text=f"Global MAPE: {global_mape:.2f}%", x=0.5),
                               showlegend=False, height=250)
        st.plotly_chart(fig_mape, use_container_width=True)

    with c2:
        fig_avg = go.Figure(data=[
            go.Pie(labels=['Avg Error (%)', 'Accuracy (%)'],
                   values=[avg_error, 100 - avg_error],
                   hole=0.6, marker_colors=['#636EFA', '#AB63FA'],
                   textinfo='label+percent')
        ])
        fig_avg.update_layout(title=dict(text=f"Avg Error: {avg_error:.2f}%", x=0.5),
                              showlegend=False, height=250)
        st.plotly_chart(fig_avg, use_container_width=True)

    with c3:
        fig_local = go.Figure(data=[
            go.Pie(labels=['Local MAPE (%)', 'Accuracy (%)'],
                   values=[local_mape if not np.isnan(local_mape) else 0,
                           100 - local_mape if not np.isnan(local_mape) else 100],
                   hole=0.6, marker_colors=['#FFA15A', '#19D3F3'],
                   textinfo='label+percent')
        ])
        fig_local.update_layout(title=dict(text=f"Local MAPE: {local_mape:.2f}%", x=0.5),
                                showlegend=False, height=250)
        st.plotly_chart(fig_local, use_container_width=True)

# -----------------------
# 🔟 Sample Predictions
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

# -----------------------
# 11️⃣ Actual vs Predicted Line Graph + Error Zones
# -----------------------
if output_to_plot in y_actual_df.columns:
    st.markdown(f"### 📊 Actual vs Predicted — `{output_to_plot}` with Error%")

    plot_df = pd.DataFrame({
        "Index": np.arange(1, len(y_actual) + 1),
        f"Actual_{output_to_plot}": y_actual,
        f"Predicted_{output_to_plot}": y_pred[:, output_index],
        "Error_%": percent_errors
    })

    fig_line = go.Figure()

    # Shaded region for high-error areas (>5%)
    high_error_mask = plot_df["Error_%"] > 5
    if high_error_mask.any():
        fig_line.add_vrect(
            x0=plot_df["Index"][high_error_mask].min(),
            x1=plot_df["Index"][high_error_mask].max(),
            fillcolor="rgba(255,0,0,0.1)",
            layer="below",
            line_width=0,
            annotation_text="High Error Zone",
            annotation_position="top left"
        )

    # Actual vs Predicted lines
    fig_line.add_trace(go.Scatter(x=plot_df["Index"], y=plot_df[f"Actual_{output_to_plot}"],
                                  mode="lines+markers", name="Actual",
                                  line=dict(color="green", width=2)))
    fig_line.add_trace(go.Scatter(x=plot_df["Index"], y=plot_df[f"Predicted_{output_to_plot}"],
                                  mode="lines+markers", name="Predicted",
                                  line=dict(color="blue", width=2, dash="dash")))

    # Error line (secondary axis)
    fig_line.add_trace(go.Scatter(x=plot_df["Index"], y=plot_df["Error_%"],
                                  mode="lines+markers", name="Error (%)",
                                  line=dict(color="red", width=2, dash="dot"),
                                  yaxis="y2"))

    fig_line.update_layout(
        title=f"Actual vs Predicted {output_to_plot} with Error %",
        xaxis_title="Test Sample Index",
        yaxis=dict(title=f"{output_to_plot} (Actual & Predicted)"),
        yaxis2=dict(title="Error %", overlaying="y", side="right"),
        legend=dict(x=0, y=1.1, orientation="h"),
        height=500,
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")
    st.markdown(f"**📉 Average Error %:** `{avg_error:.2f}`")
    st.markdown(f"**📈 Max Error %:** `{max_error:.2f}`")
    st.markdown(f"**📊 Min Error %:** `{min_error:.2f}`")
