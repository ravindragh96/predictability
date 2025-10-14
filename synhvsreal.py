import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------
# 1️⃣ Setup
# -----------------------
st.set_page_config(page_title="Synthetic vs Real Validation", layout="wide")
st.title("🤖 Synthetic Data Validation Dashboard — ANN Model RSM")

BASE_DIR = r"C:\Users\gantrav01\RD_predictability_11925"

REAL_TEST_PATH = os.path.join(BASE_DIR, "Copy of T33_100_Samples_for_testing.xlsx")
SYNTH_PATH = os.path.join(BASE_DIR, "synthetic_tau_98.xlsx")

# -----------------------
# 2️⃣ Load Data
# -----------------------
try:
    real_df = pd.read_excel(REAL_TEST_PATH)
    synth_df = pd.read_excel(SYNTH_PATH)
    st.success("✅ Data loaded successfully.")
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

# -----------------------
# 3️⃣ Sidebar Controls
# -----------------------
st.sidebar.header("⚙️ Validation Controls")

feature_cols = [c for c in real_df.columns if c in synth_df.columns and not c.startswith("t")]
target_cols = [c for c in real_df.columns if c.startswith("t") or c.startswith("e")]

feature_x = st.sidebar.selectbox("Select Feature X", feature_cols)
feature_y = st.sidebar.selectbox("Select Feature Y", feature_cols)
target_option = st.sidebar.selectbox("Select Target Output (Y)", target_cols)

if not feature_x or not feature_y or feature_x == feature_y:
    st.warning("Please select two distinct input features.")
    st.stop()

if not target_option:
    st.warning("Please select a target output.")
    st.stop()

# -----------------------
# 4️⃣ Build KDTree for Nearest Neighbor Match
# -----------------------
key_features = [feature_x, feature_y]
tree = cKDTree(synth_df[key_features].values)

# Find nearest synthetic point for each real test point
distances, indices = tree.query(real_df[key_features].values, k=1)

# Apply distance threshold (e.g., keep top 10% closest points)
threshold = np.percentile(distances, 10)
close_mask = distances < threshold

matched_real = real_df.loc[close_mask].reset_index(drop=True)
matched_synth = synth_df.iloc[indices[close_mask]].reset_index(drop=True)

# -----------------------
# 5️⃣ Get Real vs Synthetic Y
# -----------------------
y_real = matched_real[target_option].values
y_synth = matched_synth[target_option].values

# -----------------------
# 6️⃣ Compute Validation Metrics
# -----------------------
eps = 1e-8
mae = mean_absolute_error(y_real, y_synth)
rmse = np.sqrt(mean_squared_error(y_real, y_synth))
mape = np.mean(np.abs((y_real - y_synth) / (y_real + eps))) * 100

st.markdown(f"""
### 📊 Validation Summary for `{target_option}`
| Metric | Value |
|--------|--------|
| **Matched Samples** | `{len(matched_real)}` |
| **MAE** | `{mae:.4f}` |
| **RMSE** | `{rmse:.4f}` |
| **MAPE** | `{mape:.2f}%` |
""")

# -----------------------
# 7️⃣ Visualization — Scatter: Real vs Synthetic
# -----------------------
st.subheader(f"📈 Real vs Synthetic Predictions — {target_option}")

compare_df = pd.DataFrame({
    "Real_Y": y_real,
    "Synthetic_Y": y_synth,
    "Distance": distances[close_mask]
})

fig = px.scatter(
    compare_df,
    x="Real_Y", y="Synthetic_Y",
    color="Distance",
    color_continuous_scale="Viridis",
    title=f"{target_option}: Real vs Synthetic Predictions (Nearest Matches)",
    labels={"Real_Y": "Actual (Test Data)", "Synthetic_Y": "Predicted (Synthetic)"}
)

# Add diagonal line (perfect match)
fig.add_shape(
    type="line",
    x0=compare_df["Real_Y"].min(), y0=compare_df["Real_Y"].min(),
    x1=compare_df["Real_Y"].max(), y1=compare_df["Real_Y"].max(),
    line=dict(color="red", dash="dash"),
    name="Perfect Fit"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------
# 8️⃣ Validation Surface Comparison
# -----------------------
st.subheader(f"🌍 Feature Space Comparison — {feature_x} vs {feature_y}")

real_plot = matched_real[[feature_x, feature_y, target_option]].copy()
real_plot["Type"] = "Real Test"

synth_plot = matched_synth[[feature_x, feature_y, target_option]].copy()
synth_plot["Type"] = "Synthetic"

merged_plot = pd.concat([real_plot, synth_plot], axis=0)

fig2 = px.scatter(
    merged_plot,
    x=feature_x,
    y=feature_y,
    color=target_option,
    facet_col="Type",
    color_continuous_scale="Turbo",
    title=f"{target_option}: Real vs Synthetic Feature Distribution",
)
st.plotly_chart(fig2, use_container_width=True)

# -----------------------
# 9️⃣ Optional: Residual Plot
# -----------------------
st.subheader("📉 Residual Distribution (Error vs Real Y)")

compare_df["Residual"] = y_real - y_synth

fig3 = px.scatter(
    compare_df,
    x="Real_Y",
    y="Residual",
    color="Distance",
    color_continuous_scale="RdBu",
    title="Residual Plot (Real - Synthetic)",
    labels={"Real_Y": "Actual Y", "Residual": "Error"}
)
fig3.add_hline(y=0, line_dash="dash", line_color="black")
st.plotly_chart(fig3, use_container_width=True)

# -----------------------
# ✅ Done
# -----------------------
st.success("✅ Validation completed successfully. Review the plots above for alignment between synthetic and real predictions.")
