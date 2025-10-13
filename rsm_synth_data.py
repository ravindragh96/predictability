# rsm_clean_app.py

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
st.title("🎛️ Response Surface Modeling (RSM) Visualization")

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

# -----------------------
# 7️⃣ Generate Contour Grid
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
# 8️⃣ Plot Contour
# -----------------------
st.subheader(f"📈 RSM Contour Plot — {output_to_plot} vs {f1} & {f2}")

fig = go.Figure()

# Smooth contour like example image
fig.add_trace(go.Contour(
    z=preds,
    x=f1_range,
    y=f2_range,
    colorscale="RdYlGn_r",  # red-high to green-low
    ncontours=20,
    colorbar=dict(title=f"{output_to_plot}", titleside="right"),
    contours=dict(showlabels=True, labelfont=dict(size=10, color="black")),
    hovertemplate=(
        f"<b>{f1}</b>: %{{x:.3f}}<br>"
        f"<b>{f2}</b>: %{{y:.3f}}<br>"
        f"<b>{output_to_plot}</b>: %{{z:.3f}}<extra></extra>"
    )
))

# Overlay data points
fig.add_trace(go.Scatter(
    x=X_test[f1],
    y=X_test[f2],
    mode="markers",
    name="Data Points",
    marker=dict(
        size=7,
        color="blue",
        symbol="circle",
        line=dict(width=1, color="white")
    ),
    hovertext=[
        f"{f1}: {xv:.2f}<br>{f2}: {yv:.2f}"
        for xv, yv in zip(X_test[f1], X_test[f2])
    ],
    hoverinfo="text"
))

# Layout
fig.update_layout(
    width=850,
    height=600,
    template="simple_white",
    title=f"{output_to_plot} Contour (Fixed H1 = 100)",
    xaxis_title=f1,
    yaxis_title=f2,
    font=dict(size=12)
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------
# 9️⃣ Sample Predictions
# -----------------------
st.markdown(f"### 🔍 Sample Predicted {output_to_plot} Values")
compare_df = pd.DataFrame({
    f1: X_test[f1].values[:10],
    f2: X_test[f2].values[:10],
    f"Pred_{output_to_plot}": y_pred[:10, output_index],
})
st.dataframe(compare_df, use_container_width=True)

st.info("""
✅ **Interpretation**  
- Red zones = high predicted output  
- Green zones = low predicted output  
- Blue dots = actual test data points  
- This is the Response Surface built using your ANN model’s predictions
""")
