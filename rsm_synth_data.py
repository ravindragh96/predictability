#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

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
    global_mape = avg_error = max_error = min_error = np.nan
    percent_errors = np.zeros_like(y_pred[:, output_index])

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
# 9️⃣ Classic RSM Contour + Donuts
# -----------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📈 Classic RSM Contour: {f1} vs {f2}")

    fig, ax = plt.subplots(figsize=(7, 5))
    cp = ax.contourf(f1_range, f2_range, preds, levels=20, cmap='RdYlGn_r')
    cbar = plt.colorbar(cp, ax=ax)
    cbar.set_label(f"{output_to_plot}", rotation=270, labelpad=15)
    ax.scatter(X_test[f1], X_test[f2], color='blue', s=25, edgecolors='black', label='Sample Points')
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_title(f"{output_to_plot} — RSM Contour (H1 fixed at 100)")
    ax.legend()
    st.pyplot(fig)

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
# 🔟 Sample Predictions Table
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
# 11️⃣ Scatter Comparison
# -----------------------
if output_to_plot in y_actual_df.columns:
    st.markdown(f"### 📊 {output_to_plot} — Actual vs Predicted Distribution")

    plot_df = pd.DataFrame({
        f1: X_test[f1].values,
        f2: X_test[f2].values,
        f"Actual_{output_to_plot}": y_actual,
        f"Predicted_{output_to_plot}": y_pred[:, output_index],
    })
    plot_df["Error_%"] = np.abs(plot_df[f"Actual_{output_to_plot}"] - plot_df[f"Predicted_{output_to_plot}"]) / (
        np.abs(plot_df[f"Actual_{output_to_plot}"]) + 1e-8
    ) * 100

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=plot_df[f1], y=plot_df[f2], mode="markers",
        marker=dict(size=10, color=plot_df[f"Actual_{output_to_plot}"], colorscale="Blues",
                    colorbar=dict(title=f"Actual {output_to_plot}"), symbol="circle", line=dict(width=1, color="black")),
        name="Actual Values"
    ))
    fig_scatter.add_trace(go.Scatter(
        x=plot_df[f1], y=plot_df[f2], mode="markers",
        marker=dict(size=10, color=plot_df[f"Predicted_{output_to_plot}"], colorscale="Oranges",
                    colorbar=dict(title=f"Predicted {output_to_plot}"), symbol="diamond", line=dict(width=1, color="black")),
        name="Predicted Values"
    ))

    fig_scatter.update_layout(
        title=f"{output_to_plot} — Actual vs Predicted Scatter",
        xaxis_title=f1, yaxis_title=f2, template="plotly_white", height=700
    )

    overall_avg_error = np.mean(plot_df["Error_%"])
    fig_scatter.add_annotation(
        x=0.5, y=1.05, xref="paper", yref="paper",
        text=f"<b>Overall Avg Error: {overall_avg_error:.2f}%</b>",
        showarrow=False, bgcolor="white", bordercolor="black", borderwidth=1
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.markdown(f"**📉 Average Error %:** `{overall_avg_error:.2f}`")
    st.markdown(f"**📈 Max Error %:** `{plot_df['Error_%'].max():.2f}`")
    st.markdown(f"**📊 Min Error %:** `{plot_df['Error_%'].min():.2f}`")

else:
    st.warning("⚠️ No actual values available for comparison.")
