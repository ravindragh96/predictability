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
st.title("🌀 Response Surface Modeling (RSM) using Synthetic Data (ANN Predictions)")

BASE_DIR = r"C:\Users\gantrav01\RD_predictability_11925"

SYNTHETIC_PATH = os.path.join(BASE_DIR, "synthetic_eta_predictions.xlsx")  # generated synthetic data
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "h_vs_eta_best_model.keras")
X_SCALER_PATH = os.path.join(BASE_DIR, "x_eta_scaler.pkl")
Y_SCALER_PATH = os.path.join(BASE_DIR, "y_eta_scaler.pkl")

# -----------------------
# 2️⃣ Load Data and Model
# -----------------------
try:
    df = pd.read_excel(SYNTHETIC_PATH)
    st.success(f"✅ Synthetic dataset loaded successfully — shape: {df.shape}")
except Exception as e:
    st.error(f"❌ Could not load synthetic data: {e}")
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

input_cols = [c for c in df.columns if not c.startswith('e')]
output_cols = [c for c in df.columns if c.startswith('e')]

feature_x = st.sidebar.selectbox("Select Feature X", input_cols)
feature_y = st.sidebar.selectbox("Select Feature Y", input_cols)
target_col = st.sidebar.selectbox("Select Target Output (Z)", output_cols)

grid_resolution = st.sidebar.slider("Grid Resolution", 30, 100, 60)

# -----------------------
# 4️⃣ Prepare Data
# -----------------------
X_mean = df[input_cols].mean(numeric_only=True)

# Build grid for selected features
x_range = np.linspace(df[feature_x].min(), df[feature_x].max(), grid_resolution)
y_range = np.linspace(df[feature_y].min(), df[feature_y].max(), grid_resolution)
X1, X2 = np.meshgrid(x_range, y_range)

grid = pd.DataFrame({feature_x: X1.ravel(), feature_y: X2.ravel()})

# Fix remaining features to their mean values
for col in input_cols:
    if col not in [feature_x, feature_y]:
        grid[col] = X_mean[col]

# -----------------------
# 5️⃣ Scale & Predict using Model
# -----------------------
grid_scaled = x_scaler.transform(grid.astype(np.float32))
preds_scaled = model.predict(grid_scaled, verbose=0)
preds_all = y_scaler.inverse_transform(preds_scaled)

# Find index of the selected output
output_index = output_cols.index(target_col)
preds = preds_all[:, output_index].reshape(X1.shape)

# -----------------------
# 6️⃣ Plot Contour (RSM)
# -----------------------
st.subheader(f"📈 RSM Contour Plot — {target_col} vs {feature_x} & {feature_y}")

fig = go.Figure()

# Contour Plot
fig.add_trace(go.Contour(
    z=preds,
    x=x_range,
    y=y_range,
    colorscale="RdYlGn_r",
    ncontours=30,
    colorbar=dict(
        title=dict(
            text=f"{target_col}",
            font=dict(size=12)
        ),
        bgcolor="rgba(255,255,255,0.8)"
    ),
    hovertemplate=(
        f"<b>{feature_x}</b>: %{{x:.3f}}<br>"
        f"<b>{feature_y}</b>: %{{y:.3f}}<br>"
        f"<b>Predicted {target_col}</b>: %{{z:.3f}}<extra></extra>"
    )
))

# Overlay Synthetic Points (optional)
fig.add_trace(go.Scatter(
    x=df[feature_x],
    y=df[feature_y],
    mode="markers",
    name="Synthetic Data Points",
    marker=dict(
        size=5,
        color="blue",
        symbol="circle",
        line=dict(width=0.5, color="white")
    ),
    hovertext=[
        f"{feature_x}: {xv:.3f}<br>{feature_y}: {yv:.3f}<br>{target_col}: {zv:.3f}"
        for xv, yv, zv in zip(df[feature_x], df[feature_y], df[target_col])
    ],
    hoverinfo="text"
))

fig.update_layout(
    title=f"Response Surface for {target_col}",
    xaxis_title=feature_x,
    yaxis_title=feature_y,
    width=850,
    height=600,
    template="simple_white",
    font=dict(size=12)
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------
# 7️⃣ Sample Predictions Table
# -----------------------
st.markdown(f"### 🔍 Sample Predicted Values for `{target_col}`")
sample_df = grid[[feature_x, feature_y]].copy()
sample_df[f"Predicted_{target_col}"] = preds.ravel()[:10]
st.dataframe(sample_df.head(10), use_container_width=True)

st.info("""
✅ **Interpretation:**  
- The color gradient shows how the ANN-predicted output varies across the X–Y feature space.  
- Red → high predicted output, Green → low predicted output.  
- Hover over the surface to see predicted values.
""")
