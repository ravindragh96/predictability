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
st.set_page_config(page_title="Response Surface Modeling (RSM)", layout="wide")
st.title("🎛️ Response Surface Modeling — Synthetic vs Real (Constant & Free Features)")

BASE_DIR = r"C:\Users\gantrav01\RD_predictability_11925"

TRAIN_X_PATH = os.path.join(BASE_DIR, "H_vs_Tau_training.xlsx")
TRAIN_Y_PATH = os.path.join(BASE_DIR, "H_vs_Tau_target.xlsx")
REAL_PATH = os.path.join(BASE_DIR, "Copy of T33_100_Samples_for_testing.xlsx")
SYNTH_PATH = os.path.join(BASE_DIR, "synthetic_tau_98.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "h_vs_tau_best_model.keras")
X_SCALER_PATH = os.path.join(BASE_DIR, "x_eta_scaler.pkl")
Y_SCALER_PATH = os.path.join(BASE_DIR, "y_eta_scaler.pkl")

# -----------------------
# 2️⃣ Load Data & Model
# -----------------------
X_train = pd.read_excel(TRAIN_X_PATH)
y_train = pd.read_excel(TRAIN_Y_PATH)
real_df = pd.read_excel(REAL_PATH)
synth_df = pd.read_excel(SYNTH_PATH)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
x_scaler = joblib.load(X_SCALER_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)

# -----------------------
# 3️⃣ Sidebar Controls
# -----------------------
feature_cols = list(X_train.columns)
target_cols = list(y_train.columns)

st.sidebar.header("⚙️ Controls")
feature_x = st.sidebar.selectbox("Feature X", feature_cols)
feature_y = st.sidebar.selectbox("Feature Y", [c for c in feature_cols if c != feature_x])
target_option = st.sidebar.selectbox("Target Output", target_cols)

x_min, x_max = synth_df[feature_x].min(), synth_df[feature_x].max()
y_min, y_max = synth_df[feature_y].min(), synth_df[feature_y].max()

x_range = st.sidebar.slider(f"{feature_x} Range", float(x_min), float(x_max), (float(x_min), float(x_max)))
y_range = st.sidebar.slider(f"{feature_y} Range", float(y_min), float(y_max), (float(y_min), float(y_max)))

# -----------------------
# 4️⃣ Filter Data
# -----------------------
synth_filtered = synth_df[
    (synth_df[feature_x] >= x_range[0]) & (synth_df[feature_x] <= x_range[1]) &
    (synth_df[feature_y] >= y_range[0]) & (synth_df[feature_y] <= y_range[1])
].reset_index(drop=True)

# -----------------------
# 5️⃣ ANN Prediction Function
# -----------------------
def predict_ann(df):
    df_aligned = df[X_train.columns].copy()
    scaled = x_scaler.transform(df_aligned.astype(np.float32))
    preds = model.predict(scaled, verbose=0)
    preds = y_scaler.inverse_transform(preds)
    return preds[:, y_train.columns.get_loc(target_option)]

# -----------------------
# 6️⃣ Predict Synthetic Data (Constant vs Free)
# -----------------------
X_mean = X_train.mean(numeric_only=True)
X_std = X_train.std(numeric_only=True)

grid_const = synth_filtered.copy()
for c in X_train.columns:
    if c not in [feature_x, feature_y]:
        grid_const[c] = X_mean[c]

synth_const_pred = predict_ann(grid_const)
synth_free_pred = predict_ann(synth_filtered)

# -----------------------
# 7️⃣ Random Real Points + Predictions
# -----------------------
filtered_real = real_df[
    (real_df[feature_x] >= x_range[0]) & (real_df[feature_x] <= x_range[1]) &
    (real_df[feature_y] >= y_range[0]) & (real_df[feature_y] <= y_range[1])
]
sampled_real = filtered_real.sample(n=min(5, len(filtered_real)), random_state=42) if len(filtered_real) > 0 else pd.DataFrame()

if not sampled_real.empty:
    sampled_real_pred_const = predict_ann(sampled_real.assign(**{c: X_mean[c] for c in X_train.columns if c not in [feature_x, feature_y]}))
    sampled_real_pred_free = predict_ann(sampled_real)
    sampled_real["Pred_Const"] = sampled_real_pred_const
    sampled_real["Pred_Free"] = sampled_real_pred_free
    sampled_real["Error_Const_%"] = np.abs(sampled_real[target_option] - sampled_real_pred_const) / (np.abs(sampled_real[target_option]) + 1e-8) * 100
    sampled_real["Error_Free_%"] = np.abs(sampled_real[target_option] - sampled_real_pred_free) / (np.abs(sampled_real[target_option]) + 1e-8) * 100

# -----------------------
# 8️⃣ Scatter Plots
# -----------------------
st.markdown("## 🎨 Scatter Plots — Synthetic vs Real Comparison")

zmin = min(synth_const_pred.min(), synth_free_pred.min(), real_df[target_option].min())
zmax = max(synth_const_pred.max(), synth_free_pred.max(), real_df[target_option].max())

col1, col2, col3 = st.columns(3)

def scatter_plot(title, x, y, color, zmin, zmax, hover_label, real_points=None, custom=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=6, color=color, colorscale="RdYlGn_r", cmin=zmin, cmax=zmax,
                    colorbar=dict(title=hover_label)),
        hovertemplate=(
            f"<b>{feature_x}</b>: %{{x:.2f}}<br>"
            f"<b>{feature_y}</b>: %{{y:.2f}}<br>"
            f"<b>{hover_label}</b>: %{{marker.color:.2f}}<extra></extra>"
        )
    ))
    if real_points is not None:
        fig.add_trace(go.Scatter(
            x=real_points[feature_x], y=real_points[feature_y],
            mode="markers",
            marker=dict(size=12, color="blue", symbol="star", line=dict(width=1.5, color="black")),
            customdata=custom,
            hovertemplate=(
                f"<b>{feature_x}</b>: %{{x:.2f}}<br>"
                f"<b>{feature_y}</b>: %{{y:.2f}}<br>"
                f"<b>Actual {target_option}</b>: %{{customdata[0]:.2f}}<br>"
                f"<b>Predicted</b>: %{{customdata[1]:.2f}}<br>"
                f"<b>Error %</b>: %{{customdata[2]:.2f}}<extra></extra>"
            )
        ))
    fig.update_layout(title=title, template="plotly_white")
    return fig

with col1:
    st.subheader("🟡 Synthetic (Constant Features)")
    st.plotly_chart(scatter_plot("Constant Mean", synth_filtered[feature_x], synth_filtered[feature_y],
                                 synth_const_pred, zmin, zmax, f"Pred {target_option}",
                                 sampled_real if not sampled_real.empty else None,
                                 sampled_real[[target_option, "Pred_Const", "Error_Const_%"]].values if not sampled_real.empty else None),
                    use_container_width=True)

with col2:
    st.subheader("🔵 Synthetic (Free Features)")
    st.plotly_chart(scatter_plot("Free Features", synth_filtered[feature_x], synth_filtered[feature_y],
                                 synth_free_pred, zmin, zmax, f"Pred {target_option}",
                                 sampled_real if not sampled_real.empty else None,
                                 sampled_real[[target_option, "Pred_Free", "Error_Free_%"]].values if not sampled_real.empty else None),
                    use_container_width=True)

with col3:
    st.subheader("🟢 Real Data (Actual)")
    st.plotly_chart(scatter_plot("Actual Data", real_df[feature_x], real_df[feature_y],
                                 real_df[target_option], zmin, zmax, f"Actual {target_option}",
                                 sampled_real if not sampled_real.empty else None,
                                 sampled_real[[target_option, target_option, "Error_Const_%"]].values if not sampled_real.empty else None),
                    use_container_width=True)

# -----------------------
# 9️⃣ RSM Contour Plots
# -----------------------
st.markdown("## 🌀 Response Surface (RSM)")

f1_range = np.linspace(x_range[0], x_range[1], 60)
f2_range = np.linspace(y_range[0], y_range[1], 60)
F1, F2 = np.meshgrid(f1_range, f2_range)

grid_surface = pd.DataFrame({feature_x: F1.ravel(), feature_y: F2.ravel()})
for c in X_train.columns:
    if c not in [feature_x, feature_y]:
        grid_surface[c] = X_mean[c]

pred_const = predict_ann(grid_surface).reshape(F1.shape)
pred_free = predict_ann(synth_filtered.assign(**{feature_x: np.tile(f1_range, len(f2_range)), feature_y: np.repeat(f2_range, len(f1_range))})).reshape(F1.shape)

colA, colB = st.columns(2)

with colA:
    st.subheader("RSM — Constant Features")
    fig_rsm1 = go.Figure(data=go.Contour(z=pred_const, x=f1_range, y=f2_range, colorscale="RdYlGn_r", ncontours=25,
                                         colorbar=dict(title=target_option)))
    fig_rsm1.add_trace(go.Scatter(
        x=synth_filtered[feature_x], y=synth_filtered[feature_y],
        mode="markers", marker=dict(size=6, color="black"), name="Synthetic Points"))
    st.plotly_chart(fig_rsm1, use_container_width=True)

with colB:
    st.subheader("RSM — Free Features")
    fig_rsm2 = go.Figure(data=go.Contour(z=pred_free, x=f1_range, y=f2_range, colorscale="RdYlGn_r", ncontours=25,
                                         colorbar=dict(title=target_option)))
    fig_rsm2.add_trace(go.Scatter(
        x=synth_filtered[feature_x], y=synth_filtered[feature_y],
        mode="markers", marker=dict(size=6, color="black"), name="Synthetic Points"))
    st.plotly_chart(fig_rsm2, use_container_width=True)

# -----------------------
# 🔟 Error Metrics + DataFrame Side by Side
# -----------------------
st.markdown("## 📊 Error Metrics & Matched Results")

if not sampled_real.empty:
    mape_const = np.mean(sampled_real["Error_Const_%"])
    mape_free = np.mean(sampled_real["Error_Free_%"])

    colE1, colE2 = st.columns([1, 2])

    with colE1:
        fig_mape1 = go.Figure(data=[go.Pie(labels=['MAPE (Const)', 'Accuracy'], values=[mape_const, 100 - mape_const],
                                           hole=0.6, marker_colors=['#EF553B', '#00CC96'])])
        fig_mape1.update_layout(title_text=f"Global MAPE (Const): {mape_const:.2f}%", showlegend=False, height=300)
        fig_mape2 = go.Figure(data=[go.Pie(labels=['MAPE (Free)', 'Accuracy'], values=[mape_free, 100 - mape_free],
                                           hole=0.6, marker_colors=['#FFA15A', '#19D3F3'])])
        fig_mape2.update_layout(title






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
st.set_page_config(page_title="Response Surface Modeling (RSM)", layout="wide")
st.title("🎛️ Response Surface Modeling — Synthetic Constant vs Free Feature Comparison")

BASE_DIR = r"C:\Users\gantrav01\RD_predictability_11925"

TRAIN_X_PATH = os.path.join(BASE_DIR, "H_vs_Tau_training.xlsx")
TRAIN_Y_PATH = os.path.join(BASE_DIR, "H_vs_Tau_target.xlsx")
REAL_PATH = os.path.join(BASE_DIR, "Copy of T33_100_Samples_for_testing.xlsx")
SYNTH_PATH = os.path.join(BASE_DIR, "synthetic_tau_98.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "h_vs_tau_best_model.keras")
X_SCALER_PATH = os.path.join(BASE_DIR, "x_eta_scaler.pkl")
Y_SCALER_PATH = os.path.join(BASE_DIR, "y_eta_scaler.pkl")

# -----------------------
# 2️⃣ Load Data & Model
# -----------------------
X_train = pd.read_excel(TRAIN_X_PATH)
y_train = pd.read_excel(TRAIN_Y_PATH)
real_df = pd.read_excel(REAL_PATH)
synth_df = pd.read_excel(SYNTH_PATH)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
x_scaler = joblib.load(X_SCALER_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)

# -----------------------
# 3️⃣ Sidebar Controls
# -----------------------
feature_cols = list(X_train.columns)
target_cols = list(y_train.columns)

st.sidebar.header("⚙️ Controls")
feature_x = st.sidebar.selectbox("Feature X", feature_cols)
feature_y = st.sidebar.selectbox("Feature Y", [c for c in feature_cols if c != feature_x])
target_option = st.sidebar.selectbox("Target Output", target_cols)

x_min, x_max = synth_df[feature_x].min(), synth_df[feature_x].max()
y_min, y_max = synth_df[feature_y].min(), synth_df[feature_y].max()
x_range = st.sidebar.slider(f"{feature_x} Range", float(x_min), float(x_max), (float(x_min), float(x_max)))
y_range = st.sidebar.slider(f"{feature_y} Range", float(y_min), float(y_max), (float(y_min), float(y_max)))

# -----------------------
# 4️⃣ Filter Synthetic Data
# -----------------------
synth_filtered = synth_df[
    (synth_df[feature_x] >= x_range[0]) & (synth_df[feature_x] <= x_range[1]) &
    (synth_df[feature_y] >= y_range[0]) & (synth_df[feature_y] <= y_range[1])
].reset_index(drop=True)

# -----------------------
# 5️⃣ ANN Prediction Function
# -----------------------
def predict_ann(df):
    df_aligned = df[X_train.columns].copy()
    scaled = x_scaler.transform(df_aligned.astype(np.float32))
    preds = model.predict(scaled, verbose=0)
    preds = y_scaler.inverse_transform(preds)
    return preds[:, y_train.columns.get_loc(target_option)]

# -----------------------
# 6️⃣ Synthetic Predictions
# -----------------------
X_mean = X_train.mean(numeric_only=True)
X_std = X_train.std(numeric_only=True)

grid_const = synth_filtered.copy()
for c in X_train.columns:
    if c not in [feature_x, feature_y]:
        grid_const[c] = X_mean[c]

synth_const_pred = predict_ann(grid_const)
synth_free_pred = predict_ann(synth_filtered)

# -----------------------
# 7️⃣ Random Real Points + Predictions
# -----------------------
filtered_real = real_df[
    (real_df[feature_x] >= x_range[0]) & (real_df[feature_x] <= x_range[1]) &
    (real_df[feature_y] >= y_range[0]) & (real_df[feature_y] <= y_range[1])
]

sampled_real = filtered_real.sample(n=min(5, len(filtered_real)), random_state=42) if len(filtered_real) > 0 else pd.DataFrame()

if not sampled_real.empty:
    sampled_real_pred_const = predict_ann(sampled_real.assign(**{c: X_mean[c] for c in X_train.columns if c not in [feature_x, feature_y]}))
    sampled_real_pred_free = predict_ann(sampled_real)
    sampled_real["Pred_Const"] = sampled_real_pred_const
    sampled_real["Pred_Free"] = sampled_real_pred_free
    sampled_real["Error_Const_%"] = np.abs(sampled_real[target_option] - sampled_real_pred_const) / (np.abs(sampled_real[target_option]) + 1e-8) * 100
    sampled_real["Error_Free_%"] = np.abs(sampled_real[target_option] - sampled_real_pred_free) / (np.abs(sampled_real[target_option]) + 1e-8) * 100

# -----------------------
# 8️⃣ Scatter Plots — with Hover Cards
# -----------------------
st.markdown("## 🎨 Scatter Plots — Synthetic vs Real Comparison")

zmin = min(synth_const_pred.min(), synth_free_pred.min(), real_df[target_option].min())
zmax = max(synth_const_pred.max(), synth_free_pred.max(), real_df[target_option].max())

col1, col2, col3 = st.columns(3)

def hover_info(x, y, val, name):
    return f"<b>{feature_x}</b>: {x:.3f}<br><b>{feature_y}</b>: {y:.3f}<br><b>{name}</b>: {val:.3f}"

# Constant
with col1:
    st.subheader("🟡 Synthetic (Mean Constant)")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=synth_filtered[feature_x],
        y=synth_filtered[feature_y],
        mode="markers",
        marker=dict(size=6, color=synth_const_pred, colorscale="RdYlGn_r", cmin=zmin, cmax=zmax,
                    colorbar=dict(title=f"{target_option}")),
        hovertemplate=(
            f"<b>{feature_x}</b>: %{{x:.2f}}<br>"
            f"<b>{feature_y}</b>: %{{y:.2f}}<br>"
            f"<b>Predicted {target_option}</b>: %{{marker.color:.2f}}<extra></extra>"
        )
    ))
    if not sampled_real.empty:
        fig1.add_trace(go.Scatter(
            x=sampled_real[feature_x], y=sampled_real[feature_y],
            mode="markers",
            marker=dict(size=12, color="blue", symbol="star", line=dict(width=1.5, color="black")),
            customdata=sampled_real[[target_option, "Pred_Const", "Error_Const_%"]].values,
            hovertemplate=(
                f"<b>{feature_x}</b>: %{{x:.2f}}<br>"
                f"<b>{feature_y}</b>: %{{y:.2f}}<br>"
                f"<b>Actual {target_option}</b>: %{{customdata[0]:.2f}}<br>"
                f"<b>Pred (Const)</b>: %{{customdata[1]:.2f}}<br>"
                f"<b>Error %</b>: %{{customdata[2]:.2f}}<extra></extra>"
            )
        ))
    st.plotly_chart(fig1, use_container_width=True)

# Free
with col2:
    st.subheader("🔵 Synthetic (Free Features)")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=synth_filtered[feature_x], y=synth_filtered[feature_y],
        mode="markers",
        marker=dict(size=6, color=synth_free_pred, colorscale="RdYlGn_r", cmin=zmin, cmax=zmax,
                    colorbar=dict(title=f"{target_option}")),
        hovertemplate=(
            f"<b>{feature_x}</b>: %{{x:.2f}}<br>"
            f"<b>{feature_y}</b>: %{{y:.2f}}<br>"
            f"<b>Predicted {target_option}</b>: %{{marker.color:.2f}}<extra></extra>"
        )
    ))
    if not sampled_real.empty:
        fig2.add_trace(go.Scatter(
            x=sampled_real[feature_x], y=sampled_real[feature_y],
            mode="markers",
            marker=dict(size=12, color="blue", symbol="star", line=dict(width=1.5, color="black")),
            customdata=sampled_real[[target_option, "Pred_Free", "Error_Free_%"]].values,
            hovertemplate=(
                f"<b>{feature_x}</b>: %{{x:.2f}}<br>"
                f"<b>{feature_y}</b>: %{{y:.2f}}<br>"
                f"<b>Actual {target_option}</b>: %{{customdata[0]:.2f}}<br>"
                f"<b>Pred (Free)</b>: %{{customdata[1]:.2f}}<br>"
                f"<b>Error %</b>: %{{customdata[2]:.2f}}<extra></extra>"
            )
        ))
    st.plotly_chart(fig2, use_container_width=True)

# Real
with col3:
    st.subheader("🟢 Real Data (Actual)")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=real_df[feature_x], y=real_df[feature_y],
        mode="markers",
        marker=dict(size=6, color=real_df[target_option], colorscale="RdYlGn_r", cmin=zmin, cmax=zmax,
                    colorbar=dict(title=f"{target_option}")),
        hovertemplate=(
            f"<b>{feature_x}</b>: %{{x:.2f}}<br>"
            f"<b>{feature_y}</b>: %{{y:.2f}}<br>"
            f"<b>Actual {target_option}</b>: %{{marker.color:.2f}}<extra></extra>"
        )
    ))
    st.plotly_chart(fig3, use_container_width=True)

# -----------------------
# 9️⃣ RSM Contour Plots
# -----------------------
st.markdown("## 🌀 Response Surface (RSM)")

f1_range = np.linspace(x_range[0], x_range[1], 60)
f2_range = np.linspace(y_range[0], y_range[1], 60)
F1, F2 = np.meshgrid(f1_range, f2_range)

grid_surface = pd.DataFrame({feature_x: F1.ravel(), feature_y: F2.ravel()})
for c in X_train.columns:
    if c not in [feature_x, feature_y]:
        grid_surface[c] = X_mean[c]

pred_const = predict_ann(grid_surface).reshape(F1.shape)
pred_free = predict_ann(grid_surface.assign(**{c: synth_df[c].mean() for c in synth_df.columns if c not in [feature_x, feature_y]})).reshape(F1.shape)

colA, colB = st.columns(2)

with colA:
    st.subheader("RSM — Constant Features")
    fig_rsm1 = go.Figure(data=go.Contour(z=pred_const, x=f1_range, y=f2_range,
                                         colorscale="RdYlGn_r", ncontours=25,
                                         colorbar=dict(title=target_option)))
    st.plotly_chart(fig_rsm1, use_container_width=True)

with colB:
    st.subheader("RSM — Free Features")
    fig_rsm2 = go.Figure(data=go.Contour(z=pred_free, x=f1_range, y=f2_range,
                                         colorscale="RdYlGn_r", ncontours=25,
                                         colorbar=dict(title=target_option)))
    st.plotly_chart(fig_rsm2, use_container_width=True)

# -----------------------
# 🔟 Donut Charts + DF
# -----------------------
st.markdown("## 📊 Error Metrics")

if not sampled_real.empty:
    mape_const = np.mean(sampled_real["Error_Const_%"])
    mape_free = np.mean(sampled_real["Error_Free_%"])

    colD1, colD2 = st.columns(2)
    with colD1:
        fig_mape1 = go.Figure(data=[go.Pie(labels=['MAPE (%)', 'Accuracy (%)'],
                                           values=[mape_const, 100 - mape_const],
                                           hole=0.6, marker_colors=['#EF553B', '#00CC96'])])
        fig_mape1.update_layout(title_text=f"Global MAPE (Const): {mape_const:.2f}%", showlegend=False)
        st.plotly_chart(fig_mape1, use_container_width=True)

    with colD2:
        fig_mape2 = go.Figure(data=[go.Pie(labels=['MAPE (%)', 'Accuracy (%)'],
                                           values=[mape_free, 100 - mape_free],
                                           hole=0.6, marker_colors=['#FFA15A', '#19D3F3'])])
        fig_mape2.update_layout(title_text=f"Global MAPE (Free): {mape_free:.2f}%", showlegend=False)
        st.plotly_chart(fig_mape2, use_container_width=True)

    st.subheader("🔍 Comparison DataFrame")
    df_summary = sampled_real[[feature_x, feature_y, target_option, "Pred_Const", "Pred_Free", "Error_Const_%", "Error_Free_%"]]
    df_summary.rename(columns={
        target_option: f"Actual_{target_option}",
        "Pred_Const": f"Pred_{target_option}_Const",
        "Pred_Free": f"Pred_{target_option}_Free",
        "Error_Const_%": "Error_Const(%)",
        "Error_Free_%": "Error_Free(%)"
    }, inplace=True)
    st.dataframe(df_summary.style.format(precision=2), use_container_width=True)








#============================================================updated==================================
#!/usr/bin/env python
#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import tensorflow as tf
import joblib

# -----------------------
# 1️⃣ Setup
# -----------------------
st.set_page_config(page_title="Response Surface Modeling (RSM)", layout="wide")
st.title("🎛️ Response Surface Modeling (RSM) — Constant vs Free Synthetic Comparison")

BASE_DIR = r"C:\Users\gantrav01\RD_predictability_11925"

TRAIN_X_PATH = os.path.join(BASE_DIR, "H_vs_Tau_training.xlsx")
TRAIN_Y_PATH = os.path.join(BASE_DIR, "H_vs_Tau_target.xlsx")
REAL_PATH = os.path.join(BASE_DIR, "Copy of T33_100_Samples_for_testing.xlsx")
SYNTH_PATH = os.path.join(BASE_DIR, "synthetic_tau_98.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "h_vs_tau_best_model.keras")
X_SCALER_PATH = os.path.join(BASE_DIR, "x_eta_scaler.pkl")
Y_SCALER_PATH = os.path.join(BASE_DIR, "y_eta_scaler.pkl")

# -----------------------
# 2️⃣ Load Data & Model
# -----------------------
X_train = pd.read_excel(TRAIN_X_PATH)
y_train = pd.read_excel(TRAIN_Y_PATH)
real_df = pd.read_excel(REAL_PATH)
synth_df = pd.read_excel(SYNTH_PATH)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
x_scaler = joblib.load(X_SCALER_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)

# -----------------------
# 3️⃣ Sidebar Controls
# -----------------------
feature_cols = list(X_train.columns)
target_cols = list(y_train.columns)

st.sidebar.header("⚙️ Controls")
feature_x = st.sidebar.selectbox("Feature X", feature_cols)
feature_y = st.sidebar.selectbox("Feature Y", [c for c in feature_cols if c != feature_x])
target_option = st.sidebar.selectbox("Target Output", target_cols)

# -----------------------
# 4️⃣ Range Selection
# -----------------------
x_min, x_max = synth_df[feature_x].min(), synth_df[feature_x].max()
y_min, y_max = synth_df[feature_y].min(), synth_df[feature_y].max()

x_range = st.sidebar.slider(f"{feature_x} Range", float(x_min), float(x_max), (float(x_min), float(x_max)))
y_range = st.sidebar.slider(f"{feature_y} Range", float(y_min), float(y_max), (float(y_min), float(y_max)))

# -----------------------
# 5️⃣ Filter Data
# -----------------------
synth_filtered = synth_df[
    (synth_df[feature_x] >= x_range[0]) & (synth_df[feature_x] <= x_range[1]) &
    (synth_df[feature_y] >= y_range[0]) & (synth_df[feature_y] <= y_range[1])
].reset_index(drop=True)

# -----------------------
# 6️⃣ ANN Prediction Function
# -----------------------
def predict_ann(df):
    df_aligned = df[X_train.columns].copy()
    scaled = x_scaler.transform(df_aligned.astype(np.float32))
    preds = model.predict(scaled, verbose=0)
    preds = y_scaler.inverse_transform(preds)
    return preds[:, y_train.columns.get_loc(target_option)]

# Mean & std for variation
X_mean = X_train.mean(numeric_only=True)
X_std = X_train.std(numeric_only=True)

# Constant mean ± std setup
grid_const = synth_filtered.copy()
for c in X_train.columns:
    if c not in [feature_x, feature_y]:
        grid_const[c] = np.random.choice(
            [X_mean[c] - 0.5 * X_std[c], X_mean[c] + 0.5 * X_std[c]]
        )
synth_const_pred = predict_ann(grid_const)

# Free setup
synth_free_pred = predict_ann(synth_filtered)

# -----------------------
# 7️⃣ Random Real Points + Predictions
# -----------------------
filtered_real = real_df[
    (real_df[feature_x] >= x_range[0]) & (real_df[feature_x] <= x_range[1]) &
    (real_df[feature_y] >= y_range[0]) & (real_df[feature_y] <= y_range[1])
]
sampled_real = filtered_real.sample(n=min(5, len(filtered_real)), random_state=42) if len(filtered_real) > 0 else pd.DataFrame()

if not sampled_real.empty:
    sampled_real_pred_const = predict_ann(sampled_real.assign(**{c: X_mean[c] for c in X_train.columns if c not in [feature_x, feature_y]}))
    sampled_real_pred_free = predict_ann(sampled_real)
    sampled_real["Pred_Const"] = sampled_real_pred_const
    sampled_real["Pred_Free"] = sampled_real_pred_free
    sampled_real["Error_Const_%"] = np.abs(sampled_real[target_option] - sampled_real_pred_const) / (np.abs(sampled_real[target_option]) + 1e-8) * 100
    sampled_real["Error_Free_%"] = np.abs(sampled_real[target_option] - sampled_real_pred_free) / (np.abs(sampled_real[target_option]) + 1e-8) * 100

# -----------------------
# 8️⃣ Scatter Plots with Hover Cards
# -----------------------
st.markdown("## 🎨 Scatter Plots — Synthetic vs Real Comparison")

zmin = min(synth_const_pred.min(), synth_free_pred.min(), real_df[target_option].min())
zmax = max(synth_const_pred.max(), synth_free_pred.max(), real_df[target_option].max())

col1, col2, col3 = st.columns(3)

# Scatter 1: Synthetic Constant
with col1:
    st.subheader("🟡 Synthetic (Mean ± Std)")
    fig1 = go.Figure(data=go.Scatter(
        x=synth_filtered[feature_x],
        y=synth_filtered[feature_y],
        mode="markers",
        marker=dict(size=6, color=synth_const_pred, colorscale="RdYlGn_r", cmin=zmin, cmax=zmax,
                    colorbar=dict(title=f"{target_option}")),
        hovertemplate=(
            f"<b>{feature_x}</b>: %{{x:.3f}}<br>"
            f"<b>{feature_y}</b>: %{{y:.3f}}<br>"
            f"<b>Predicted {target_option} (Const)</b>: %{{marker.color:.3f}}<extra></extra>"
        )
    ))

    if not sampled_real.empty:
        fig1.add_trace(go.Scatter(
            x=sampled_real[feature_x],
            y=sampled_real[feature_y],
            mode="markers",
            marker=dict(size=11, color="blue", symbol="star", line=dict(width=1.5, color="black")),
            hovertemplate=(
                f"<b>{feature_x}</b>: %{{x:.2f}}<br>"
                f"<b>{feature_y}</b>: %{{y:.2f}}<br>"
                f"<b>Actual {target_option}</b>: %{{customdata[0]:.2f}}<br>"
                f"<b>Pred (Const)</b>: %{{customdata[1]:.2f}}<br>"
                f"<b>Error %</b>: %{{customdata[2]:.2f}}<extra></extra>"
            ),
            customdata=sampled_real[[target_option, "Pred_Const", "Error_Const_%"]].values,
            name="Highlighted Real Points"
        ))

    fig1.update_layout(title=f"{target_option} (Mean ± Std)", template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

# Scatter 2: Synthetic Free
with col2:
    st.subheader("🔵 Synthetic (Free Features)")
    fig2 = go.Figure(data=go.Scatter(
        x=synth_filtered[feature_x],
        y=synth_filtered[feature_y],
        mode="markers",
        marker=dict(size=6, color=synth_free_pred, colorscale="RdYlGn_r", cmin=zmin, cmax=zmax,
                    colorbar=dict(title=f"{target_option}")),
        hovertemplate=(
            f"<b>{feature_x}</b>: %{{x:.3f}}<br>"
            f"<b>{feature_y}</b>: %{{y:.3f}}<br>"
            f"<b>Predicted {target_option} (Free)</b>: %{{marker.color:.3f}}<extra></extra>"
        )
    ))

    if not sampled_real.empty:
        fig2.add_trace(go.Scatter(
            x=sampled_real[feature_x],
            y=sampled_real[feature_y],
            mode="markers",
            marker=dict(size=11, color="blue", symbol="star", line=dict(width=1.5, color="black")),
            hovertemplate=(
                f"<b>{feature_x}</b>: %{{x:.2f}}<br>"
                f"<b>{feature_y}</b>: %{{y:.2f}}<br>"
                f"<b>Actual {target_option}</b>: %{{customdata[0]:.2f}}<br>"
                f"<b>Pred (Free)</b>: %{{customdata[1]:.2f}}<br>"
                f"<b>Error %</b>: %{{customdata[2]:.2f}}<extra></extra>"
            ),
            customdata=sampled_real[[target_option, "Pred_Free", "Error_Free_%"]].values,
            name="Highlighted Real Points"
        ))

    fig2.update_layout(title=f"{target_option} (Free Features)", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

# Scatter 3: Real Data
with col3:
    st.subheader("🟢 Real Data (Actual)")
    fig3 = go.Figure(data=go.Scatter(
        x=real_df[feature_x],
        y=real_df[feature_y],
        mode="markers",
        marker=dict(size=6, color=real_df[target_option], colorscale="RdYlGn_r", cmin=zmin, cmax=zmax,
                    colorbar=dict(title=f"{target_option}")),
        hovertemplate=(
            f"<b>{feature_x}</b>: %{{x:.3f}}<br>"
            f"<b>{feature_y}</b>: %{{y:.3f}}<br>"
            f"<b>Actual {target_option}</b>: %{{marker.color:.3f}}<extra></extra>"
        )
    ))
    fig3.update_layout(title=f"Actual {target_option}", template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

# -----------------------
# 9️⃣ DataFrame Summary
# -----------------------
st.markdown("## 🧾 Comparison DataFrame")

if not sampled_real.empty:
    df_summary = sampled_real[[feature_x, feature_y, target_option, "Pred_Const", "Pred_Free", "Error_Const_%", "Error_Free_%"]]
    df_summary.rename(columns={
        target_option: f"Actual_{target_option}",
        "Pred_Const": f"Pred_{target_option}_Const",
        "Pred_Free": f"Pred_{target_option}_Free",
        "Error_Const_%": "Error_Const(%)",
        "Error_Free_%": "Error_Free(%)"
    }, inplace=True)
    st.dataframe(df_summary.style.format(precision=2), use_container_width=True)
else:
    st.warning("⚠️ No random points found within this range.")

st.info(f"""
**X:** {feature_x} | **Y:** {feature_y} | **Target:** {target_option}  
Hover over any point to see: X, Y, and T1 (Predicted / Actual).  
Blue stars = Real samples with predicted values and % error.
""")






#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.spatial import cKDTree
import tensorflow as tf
import joblib

# -----------------------
# 1️⃣ Setup
# -----------------------
st.set_page_config(page_title="Response Surface Modeling (RSM)", layout="wide")
st.title("🎛️ Response Surface Modeling (RSM) — Real vs Synthetic (Constant & Free) Comparison")

BASE_DIR = r"C:\Users\gantrav01\RD_predictability_11925"

TRAIN_X_PATH = os.path.join(BASE_DIR, "H_vs_Tau_training.xlsx")
TRAIN_Y_PATH = os.path.join(BASE_DIR, "H_vs_Tau_target.xlsx")
REAL_PATH = os.path.join(BASE_DIR, "Copy of T33_100_Samples_for_testing.xlsx")
SYNTH_PATH = os.path.join(BASE_DIR, "synthetic_tau_98.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "h_vs_tau_best_model.keras")
X_SCALER_PATH = os.path.join(BASE_DIR, "x_eta_scaler.pkl")
Y_SCALER_PATH = os.path.join(BASE_DIR, "y_eta_scaler.pkl")

# -----------------------
# 2️⃣ Load Data & Model
# -----------------------
X_train = pd.read_excel(TRAIN_X_PATH)
y_train = pd.read_excel(TRAIN_Y_PATH)
real_df = pd.read_excel(REAL_PATH)
synth_df = pd.read_excel(SYNTH_PATH)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
x_scaler = joblib.load(X_SCALER_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)

# -----------------------
# 3️⃣ Sidebar Controls
# -----------------------
feature_cols = list(X_train.columns)
target_cols = list(y_train.columns)

st.sidebar.header("⚙️ Controls")
feature_x = st.sidebar.selectbox("Feature X", feature_cols)
feature_y = st.sidebar.selectbox("Feature Y", [c for c in feature_cols if c != feature_x])
target_option = st.sidebar.selectbox("Target Output", target_cols)

# -----------------------
# 4️⃣ Range & Matching Setup
# -----------------------
x_min, x_max = synth_df[feature_x].min(), synth_df[feature_x].max()
y_min, y_max = synth_df[feature_y].min(), synth_df[feature_y].max()

x_range = st.sidebar.slider(f"{feature_x} Range", float(x_min), float(x_max), (float(x_min), float(x_max)))
y_range = st.sidebar.slider(f"{feature_y} Range", float(y_min), float(y_max), (float(y_min), float(y_max)))

threshold_percent = st.sidebar.slider("Matching tolerance (% of feature range)", 0.5, 10.0, 2.0)
threshold = ((x_max - x_min) + (y_max - y_min)) / 2 * (threshold_percent / 100)

# -----------------------
# 5️⃣ Filter Data
# -----------------------
synth_filtered = synth_df[
    (synth_df[feature_x] >= x_range[0]) & (synth_df[feature_x] <= x_range[1]) &
    (synth_df[feature_y] >= y_range[0]) & (synth_df[feature_y] <= y_range[1])
].reset_index(drop=True)

# Nearest neighbors to match synthetic to real
tree = cKDTree(real_df[[feature_x, feature_y]].values)
distances, indices = tree.query(synth_filtered[[feature_x, feature_y]].values, k=1)
valid_mask = distances < threshold
matched_real = real_df.iloc[indices[valid_mask]].reset_index(drop=True)
matched_synth = synth_filtered.loc[valid_mask].reset_index(drop=True)

# -----------------------
# 6️⃣ Prediction Function
# -----------------------
def predict_ann(df):
    df_aligned = df[X_train.columns].copy()
    scaled = x_scaler.transform(df_aligned.astype(np.float32))
    preds = model.predict(scaled, verbose=0)
    preds = y_scaler.inverse_transform(preds)
    return preds[:, y_train.columns.get_loc(target_option)]

# Synthetic predictions
X_mean = X_train.mean(numeric_only=True)

# Constant mean setup
grid = synth_filtered.copy()
for c in X_train.columns:
    if c not in [feature_x, feature_y]:
        grid[c] = X_mean[c]
synth_const_pred = predict_ann(grid)

# Free synthetic setup
synth_free_pred = predict_ann(synth_filtered)

# -----------------------
# 7️⃣ Match Values & Errors
# -----------------------
y_real = matched_real[target_option].values
y_synth = matched_synth[target_option].values
abs_error = np.abs(y_real - y_synth)
percent_error = abs_error / (np.abs(y_real) + 1e-8) * 100
mape = np.mean(percent_error)
local_mape = np.mean(percent_error)

# -----------------------
# 8️⃣ RSM Surface + Donuts Layout
# -----------------------
st.markdown("## 🌀 Response Surface & Error Metrics")

col_rsm, col_donuts = st.columns([2.5, 1])

with col_rsm:
    f1_range = np.linspace(x_range[0], x_range[1], 80)
    f2_range = np.linspace(y_range[0], y_range[1], 80)
    F1, F2 = np.meshgrid(f1_range, f2_range)

    grid_surface = pd.DataFrame({feature_x: F1.ravel(), feature_y: F2.ravel()})
    for c in X_train.columns:
        if c not in [feature_x, feature_y]:
            grid_surface[c] = X_mean[c]
    grid_pred = predict_ann(grid_surface).reshape(F1.shape)

    # Pick random real points
    filtered_real = real_df[
        (real_df[feature_x] >= x_range[0]) & (real_df[feature_x] <= x_range[1]) &
        (real_df[feature_y] >= y_range[0]) & (real_df[feature_y] <= y_range[1])
    ]
    sampled_real = filtered_real.sample(n=min(5, len(filtered_real)), random_state=42) if len(filtered_real) > 0 else pd.DataFrame()

    if not sampled_real.empty:
        # Predicted under both conditions
        sampled_real_pred_const = predict_ann(sampled_real.assign(**{c: X_mean[c] for c in X_train.columns if c not in [feature_x, feature_y]}))
        sampled_real_pred_free = predict_ann(sampled_real)

        sampled_real["Predicted_Synth_Constant"] = sampled_real_pred_const
        sampled_real["Predicted_Synth_Free"] = sampled_real_pred_free
        sampled_real["Abs_Error_Const"] = np.abs(sampled_real[target_option] - sampled_real_pred_const)
        sampled_real["Abs_Error_Free"] = np.abs(sampled_real[target_option] - sampled_real_pred_free)

    # ---- RSM Plot ----
    fig_rsm = go.Figure(data=go.Contour(
        z=grid_pred, x=f1_range, y=f2_range,
        colorscale="RdYlGn_r", ncontours=25,
        colorbar=dict(title=f"{target_option}"),
        contours=dict(showlabels=True, labelfont=dict(size=12, color="black"))
    ))

    fig_rsm.add_trace(go.Scatter(
        x=synth_filtered[feature_x],
        y=synth_filtered[feature_y],
        mode="markers",
        marker=dict(size=6, color="white", line=dict(width=1, color="black")),
        name="Synthetic Points"
    ))

    if not sampled_real.empty:
        fig_rsm.add_trace(go.Scatter(
            x=sampled_real[feature_x],
            y=sampled_real[feature_y],
            mode="markers",
            marker=dict(size=11, color="blue", symbol="star", line=dict(width=2, color="black")),
            name="Highlighted Real Points",
            text=[
                f"<b>{feature_x}</b>: {row[feature_x]:.2f}<br>"
                f"<b>{feature_y}</b>: {row[feature_y]:.2f}<br>"
                f"<b>Actual {target_option}</b>: {row[target_option]:.2f}<br>"
                f"<b>Predicted (Mean-based)</b>: {row['Predicted_Synth_Constant']:.2f}<br>"
                f"<b>Predicted (Free)</b>: {row['Predicted_Synth_Free']:.2f}<br>"
                f"<b>Error (Mean-based)</b>: {row['Abs_Error_Const']:.2f}<br>"
                f"<b>Error (Free)</b>: {row['Abs_Error_Free']:.2f}"
                for _, row in sampled_real.iterrows()
            ],
            hoverinfo="text"
        ))

    fig_rsm.update_layout(
        title=f"RSM Surface — {target_option} vs {feature_x}, {feature_y}",
        xaxis_title=feature_x,
        yaxis_title=feature_y,
        template="plotly_white",
        height=600
    )
    st.plotly_chart(fig_rsm, use_container_width=True)

with col_donuts:
    st.subheader("📊 Error Summary")
    donut_cols = st.columns(2)
    with donut_cols[0]:
        fig_mape = go.Figure(data=[go.Pie(
            labels=['MAPE (%)', 'Accuracy (%)'],
            values=[mape, 100 - mape], hole=0.6,
            marker_colors=['#EF553B', '#00CC96']
        )])
        fig_mape.update_layout(title_text=f"Global MAPE: {mape:.2f}%", showlegend=False, height=250)
        st.plotly_chart(fig_mape, use_container_width=True)
    with donut_cols[1]:
        fig_local = go.Figure(data=[go.Pie(
            labels=['Local Error (%)', 'Accuracy (%)'],
            values=[local_mape, 100 - local_mape], hole=0.6,
            marker_colors=['#FFA15A', '#19D3F3']
        )])
        fig_local.update_layout(title_text=f"Local Error: {local_mape:.2f}%", showlegend=False, height=250)
        st.plotly_chart(fig_local, use_container_width=True)

# -----------------------
# 9️⃣ Comparison Table
# -----------------------
st.subheader("🔍 Matched Data Points (Real vs Synthetic)")
comparison_df = pd.DataFrame({
    feature_x: matched_synth[feature_x],
    feature_y: matched_synth[feature_y],
    f"Synthetic_{target_option}": y_synth,
    f"Actual_{target_option}": y_real,
    "Abs_Error": abs_error,
    "Percent_Error": percent_error
}).sort_values("Percent_Error")

st.dataframe(comparison_df, use_container_width=True, height=300)

# -----------------------
# 🔟 Info Summary
# -----------------------
st.info(f"""
**Target:** `{target_option}` | **X:** `{feature_x}` | **Y:** `{feature_y}`  
**Threshold:** ±{threshold_percent:.1f}% | **Matches Found:** {len(matched_synth)}  
**Global MAPE:** {mape:.2f}% | **Local Error:** {local_mape:.2f}%  
All other features are fixed at their mean values for the contour surface.  
⭐ Highlighted blue points show actual T1 and predicted T1 (both constant & free).  
""")










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
