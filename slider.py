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
st.set_page_config(page_title="RSM Validation App", layout="wide")
st.title("🎛️ Response Surface Modeling (RSM) — Synthetic Data Validation")

BASE_DIR = r"C:\Users\gantrav01\RD_predictability_11925"

TRAIN_X_PATH = os.path.join(BASE_DIR, "H_vs_Tau_training.xlsx")
TRAIN_Y_PATH = os.path.join(BASE_DIR, "H_vs_Tau_target.xlsx")
REAL_PATH = os.path.join(BASE_DIR, "Copy of T33_100_Samples_for_testing.xlsx")
SYNTH_PATH = os.path.join(BASE_DIR, "synthetic_tau_98.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "h_vs_tau_best_model.keras")
X_SCALER_PATH = os.path.join(BASE_DIR, "x_eta_scaler.pkl")
Y_SCALER_PATH = os.path.join(BASE_DIR, "y_eta_scaler.pkl")

# -----------------------
# 2️⃣ Load Data
# -----------------------
X_train = pd.read_excel(TRAIN_X_PATH)
y_train = pd.read_excel(TRAIN_Y_PATH)
real_df = pd.read_excel(REAL_PATH)
synth_df = pd.read_excel(SYNTH_PATH)

st.sidebar.markdown("### 📂 Data Overview")
st.sidebar.write(f"Real Data Samples: {len(real_df)}")
st.sidebar.write(f"Synthetic Data Samples: {len(synth_df)}")

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
scaler_features = list(X_train.columns)
target_features = list(y_train.columns)

st.sidebar.header("⚙️ RSM Visualization Controls")
feature_x = st.sidebar.selectbox("Select Feature X", [""] + scaler_features)
feature_y = st.sidebar.selectbox("Select Feature Y", [""] + scaler_features)
target_option = st.sidebar.selectbox("Select Target Output", [""] + target_features)

if not feature_x or not feature_y or feature_x == feature_y:
    st.warning("Please select two distinct features for X and Y.")
    st.stop()
if not target_option:
    st.warning("Please select a target output.")
    st.stop()

# -----------------------
# 5️⃣ Dynamic Range Filter (for Synthetic Data)
# -----------------------
x_min, x_max = float(synth_df[feature_x].min()), float(synth_df[feature_x].max())
y_min, y_max = float(synth_df[feature_y].min()), float(synth_df[feature_y].max())

st.sidebar.markdown("### 🎚️ Synthetic Range Filter")
x_range = st.sidebar.slider(f"{feature_x} Range", min_value=x_min, max_value=x_max, value=(x_min, x_max))
y_range = st.sidebar.slider(f"{feature_y} Range", min_value=y_min, max_value=y_max, value=(y_min, y_max))

synth_filtered = synth_df[
    (synth_df[feature_x] >= x_range[0]) & (synth_df[feature_x] <= x_range[1]) &
    (synth_df[feature_y] >= y_range[0]) & (synth_df[feature_y] <= y_range[1])
].reset_index(drop=True)

st.sidebar.write(f"🧩 Filtered Synthetic Samples: {len(synth_filtered)}")

if len(synth_filtered) == 0:
    st.warning("⚠️ No synthetic samples found in this range.")
    st.stop()

# -----------------------
# 6️⃣ Match Synthetic ↔ Real Points (using KDTree)
# -----------------------
key_features = [feature_x, feature_y]
tree = cKDTree(real_df[key_features].values)
distances, indices = tree.query(synth_filtered[key_features].values, k=1)

matched_real = real_df.iloc[indices].reset_index(drop=True)
matched_synth = synth_filtered.copy()

# -----------------------
# 7️⃣ Find Target Column Safely (Case-Insensitive Match)
# -----------------------
target_col_real = target_option
possible_synth_cols = [c for c in matched_synth.columns if c.lower() == target_option.lower()]

if possible_synth_cols:
    target_col_synth = possible_synth_cols[0]
else:
    st.error(f"⚠️ Target column '{target_option}' not found in synthetic data. "
             f"Available synthetic columns: {list(matched_synth.columns)}")
    st.stop()

st.write(f"✅ Using Real Target: {target_col_real}")
st.write(f"✅ Using Synthetic Target: {target_col_synth}")

# -----------------------
# 8️⃣ Compute Errors Between Real and Synthetic Targets
# -----------------------
y_real = matched_real[target_col_real].values
y_synth = matched_synth[target_col_synth].values

eps = 1e-8
mae = np.mean(np.abs(y_real - y_synth))
rmse = np.sqrt(np.mean((y_real - y_synth) ** 2))
mape = np.mean(np.abs((y_real - y_synth) / (y_real + eps))) * 100

st.markdown(f"""
### 📊 Validation Summary for `{target_option}`
| Metric | Value |
|--------|--------|
| **Filtered Synthetic Samples** | `{len(synth_filtered)}` |
| **MAE** | `{mae:.4f}` |
| **RMSE** | `{rmse:.4f}` |
| **MAPE** | `{mape:.2f}%` |
""")

# -----------------------
# 9️⃣ Display Matched Data
# -----------------------
comparison_df = pd.DataFrame({
    f"{feature_x}_synthetic": matched_synth[feature_x],
    f"{feature_y}_synthetic": matched_synth[feature_y],
    f"Synthetic_{target_option}": y_synth,
    f"Real_{target_option}": y_real,
    "Distance": distances,
})
comparison_df["Abs_Error"] = np.abs(comparison_df[f"Real_{target_option}"] - comparison_df[f"Synthetic_{target_option}"])
comparison_df["Percent_Error"] = (
    comparison_df["Abs_Error"] / (np.abs(comparison_df[f"Real_{target_option}"]) + 1e-8) * 100
)

st.markdown(f"### 🧾 Synthetic–Real Matches Within {feature_x}: {x_range}, {feature_y}: {y_range}")
st.dataframe(comparison_df.head(25), use_container_width=True)

# -----------------------
# 🔟 Plot Contour Map (Optional)
# -----------------------
fig = go.Figure(data=go.Scatter(
    x=comparison_df[f"{feature_x}_synthetic"],
    y=comparison_df[f"{feature_y}_synthetic"],
    mode="markers",
    marker=dict(
        size=8,
        color=comparison_df[f"Synthetic_{target_option}"],
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title=f"{target_option} (Predicted)"),
        line=dict(width=1, color="black")
    ),
    text=[
        f"<b>{feature_x}</b>: {row[f'{feature_x}_synthetic']:.3f}<br>"
        f"<b>{feature_y}</b>: {row[f'{feature_y}_synthetic']:.3f}<br>"
        f"<b>Synth {target_option}</b>: {row[f'Synthetic_{target_option}']:.3f}<br>"
        f"<b>Real {target_option}</b>: {row[f'Real_{target_option}']:.3f}<br>"
        f"<b>Error %:</b> {row['Percent_Error']:.2f}%"
        for _, row in comparison_df.iterrows()
    ],
    hoverinfo="text",
    name="Synthetic Points"
))

fig.update_layout(
    title=f"🧭 RSM Prediction Contour — {target_option} vs ({feature_x}, {feature_y})",
    xaxis_title=feature_x,
    yaxis_title=feature_y,
    template="plotly_white",
    height=700,
    hovermode="closest"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------
# 🔚 Summary Box
# -----------------------
st.info(f"""
**Selected Range:**
- {feature_x}: {x_range[0]:.2f} → {x_range[1]:.2f}
- {feature_y}: {y_range[0]:.2f} → {y_range[1]:.2f}

**Matches Found:** {len(comparison_df)}  
**MAPE:** {mape:.2f}%  
**RMSE:** {rmse:.4f}
""")







# -----------------------
# 4️⃣ Dynamic Range Filter (for Synthetic Data Only)
# -----------------------
x_min, x_max = float(synth_df[feature_x].min()), float(synth_df[feature_x].max())
y_min, y_max = float(synth_df[feature_y].min()), float(synth_df[feature_y].max())

st.sidebar.markdown("### 🎚️ Synthetic Range Filter")
x_range = st.sidebar.slider(f"{feature_x} Range", min_value=x_min, max_value=x_max, value=(x_min, x_max))
y_range = st.sidebar.slider(f"{feature_y} Range", min_value=y_min, max_value=y_max, value=(y_min, y_max))

synth_filtered = synth_df[
    (synth_df[feature_x] >= x_range[0]) & (synth_df[feature_x] <= x_range[1]) &
    (synth_df[feature_y] >= y_range[0]) & (synth_df[feature_y] <= y_range[1])
].reset_index(drop=True)

st.sidebar.write(f"🧩 Filtered Synthetic Samples: {len(synth_filtered)}")

if len(synth_filtered) == 0:
    st.warning("⚠️ No synthetic samples found in this range.")
    st.stop()

# -----------------------
# 5️⃣ Find Closest Real Data for Each Synthetic Point
# -----------------------
key_features = [feature_x, feature_y]
tree = cKDTree(real_df[key_features].values)
distances, indices = tree.query(synth_filtered[key_features].values, k=1)

matched_real = real_df.iloc[indices].reset_index(drop=True)
matched_synth = synth_filtered.copy()

# -----------------------
# 6️⃣ Compute Errors Between Real and Synthetic Targets
# -----------------------
y_real = matched_real[target_option].values
y_synth = matched_synth[target_option].values

eps = 1e-8
mae = np.mean(np.abs(y_real - y_synth))
rmse = np.sqrt(np.mean((y_real - y_synth) ** 2))
mape = np.mean(np.abs((y_real - y_synth) / (y_real + eps))) * 100

st.markdown(f"""
### 📊 Validation Summary for `{target_option}`
| Metric | Value |
|--------|--------|
| **Filtered Synthetic Samples** | `{len(synth_filtered)}` |
| **MAE** | `{mae:.4f}` |
| **RMSE** | `{rmse:.4f}` |
| **MAPE** | `{mape:.2f}%` |
""")

# -----------------------
# 7️⃣ Display Matched Data
# -----------------------
comparison_df = pd.DataFrame({
    f"{feature_x}_synthetic": matched_synth[feature_x],
    f"{feature_y}_synthetic": matched_synth[feature_y],
    f"Synthetic_{target_option}": y_synth,
    f"Real_{target_option}": y_real,
    "Distance": distances,
})
comparison_df["Abs_Error"] = np.abs(comparison_df[f"Real_{target_option}"] - comparison_df[f"Synthetic_{target_option}"])
comparison_df["Percent_Error"] = (
    comparison_df["Abs_Error"] / (np.abs(comparison_df[f"Real_{target_option}"]) + 1e-8) * 100
)

st.markdown(f"### 🧾 Synthetic–Real Matches Within {feature_x}: {x_range}, {feature_y}: {y_range}")
st.dataframe(comparison_df.head(25), use_container_width=True)









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
st.title("🤖 Synthetic Data Validation — ANN Model with Range Filtering")

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

feature_cols = [c for c in real_df.columns if c in synth_df.columns and not c.startswith(("t", "e"))]
target_cols = [c for c in real_df.columns if c.startswith(("t", "e"))]

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
# 4️⃣ Range Slider for Feature Filters
# -----------------------
x_min, x_max = float(real_df[feature_x].min()), float(real_df[feature_x].max())
y_min, y_max = float(real_df[feature_y].min()), float(real_df[feature_y].max())

st.sidebar.markdown("### 🔧 Filter Data Range")
x_range = st.sidebar.slider(f"{feature_x} Range", min_value=x_min, max_value=x_max, value=(x_min, x_max))
y_range = st.sidebar.slider(f"{feature_y} Range", min_value=y_min, max_value=y_max, value=(y_min, y_max))

# Filter both real and synthetic data dynamically
real_filtered = real_df[
    (real_df[feature_x] >= x_range[0]) & (real_df[feature_x] <= x_range[1]) &
    (real_df[feature_y] >= y_range[0]) & (real_df[feature_y] <= y_range[1])
]

synth_filtered = synth_df[
    (synth_df[feature_x] >= x_range[0]) & (synth_df[feature_x] <= x_range[1]) &
    (synth_df[feature_y] >= y_range[0]) & (synth_df[feature_y] <= y_range[1])
]

st.sidebar.write(f"🧩 Filtered Real Samples: {len(real_filtered)}")
st.sidebar.write(f"🔹 Filtered Synthetic Samples: {len(synth_filtered)}")

if len(real_filtered) == 0 or len(synth_filtered) == 0:
    st.warning("⚠️ No overlapping samples found within selected range.")
    st.stop()

# -----------------------
# 5️⃣ KDTree Matching — Nearest Neighbor
# -----------------------
key_features = [feature_x, feature_y]
tree = cKDTree(synth_filtered[key_features].values)
distances, indices = tree.query(real_filtered[key_features].values, k=1)

# Threshold for closeness
threshold = np.percentile(distances, 10)
close_mask = distances < threshold

matched_real = real_filtered.loc[close_mask].reset_index(drop=True)
matched_synth = synth_filtered.iloc[indices[close_mask]].reset_index(drop=True)

st.sidebar.write(f"🎯 Matches Found: {len(matched_real)}")

# -----------------------
# 6️⃣ Compare Y values (Real vs Synthetic)
# -----------------------
if len(matched_real) == 0:
    st.warning("⚠️ No matches found within the selected range.")
    st.stop()

y_real = matched_real[target_option].values
y_synth = matched_synth[target_option].values

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
# 7️⃣ Matched DataFrame (Real vs Synthetic)
# -----------------------
comparison_df = pd.DataFrame({
    f"{feature_x}_real": matched_real[feature_x],
    f"{feature_y}_real": matched_real[feature_y],
    f"Real_{target_option}": y_real,
    f"Synth_{target_option}": y_synth,
    "Distance": distances[close_mask]
})

comparison_df["Abs_Error"] = np.abs(comparison_df[f"Real_{target_option}"] - comparison_df[f"Synth_{target_option}"])
comparison_df["Percent_Error"] = (
    comparison_df["Abs_Error"] / (np.abs(comparison_df[f"Real_{target_option}"]) + 1e-8) * 100
)

st.markdown("### 🧾 Matched Real vs Synthetic Data Points")
st.dataframe(comparison_df.head(25), use_container_width=True)

# -----------------------
# 8️⃣ Visualization — Scatter Plot: Real vs Synthetic
# -----------------------
st.subheader(f"📈 Real vs Synthetic Predictions — {target_option}")

fig = px.scatter(
    comparison_df,
    x=f"Real_{target_option}",
    y=f"Synth_{target_option}",
    color="Distance",
    color_continuous_scale="Viridis",
    title=f"{target_option}: Real vs Synthetic Predictions (Filtered Range)",
)
fig.add_shape(
    type="line",
    x0=comparison_df[f"Real_{target_option}"].min(),
    y0=comparison_df[f"Real_{target_option}"].min(),
    x1=comparison_df[f"Real_{target_option}"].max(),
    y1=comparison_df[f"Real_{target_option}"].max(),
    line=dict(color="red", dash="dash"),
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------
# 9️⃣ Feature Distribution (within selected range)
# -----------------------
st.subheader(f"🌍 Feature Space Comparison — {feature_x} vs {feature_y}")

real_plot = matched_real[[feature_x, feature_y, target_option]].copy()
real_plot["Type"] = "Real"
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
    title=f"{target_option}: Real vs Synthetic Feature Distribution (Filtered Range)",
)
st.plotly_chart(fig2, use_container_width=True)

# -----------------------
# ✅ Completion
# -----------------------
st.success("✅ Validation completed. Use sliders on the sidebar to explore specific feature ranges dynamically!")
