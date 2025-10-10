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
# 1️⃣  SETUP & LOAD FILES
# -----------------------
st.set_page_config(page_title="RSM Interactive Plotly App", layout="wide")
st.title("🎛️ Response Surface Modeling (RSM) — Test Data Visualization")

BASE_DIR = r"C:\Users\gantrav01\RD_predictability_11925"

# Paths
TRAIN_X_PATH = os.path.join(BASE_DIR, "H_vs_Tau_training.xlsx")     # for feature headers only
TRAIN_Y_PATH = os.path.join(BASE_DIR, "H_vs_Tau_target.xlsx")       # for target names
TEST_PATH    = os.path.join(BASE_DIR, "Copy of T33_100_Samples_for_testing.xlsx")
MODEL_PATH   = os.path.join(BASE_DIR, "checkpoints", "h_vs_tau_best_model.keras")
X_SCALER_PATH = os.path.join(BASE_DIR, "x_eta_scaler.pkl")
Y_SCALER_PATH = os.path.join(BASE_DIR, "y_eta_scaler.pkl")

# Load
X_train = pd.read_excel(TRAIN_X_PATH)
y_train = pd.read_excel(TRAIN_Y_PATH)
t33_df = pd.read_excel(TEST_PATH)

# Clean column names
def clean_cols(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace('[^A-Za-z0-9]+', ' ', regex=True)
    return df

X_train = clean_cols(X_train)
y_train = clean_cols(y_train)
t33_df = clean_cols(t33_df)

# Extract only shared features
feature_cols = [c for c in X_train.columns if c in t33_df.columns]
X_test = t33_df[feature_cols]

# Extract actual targets if available
target_cols = [c for c in y_train.columns if c in t33_df.columns]
y_actual_df = t33_df[target_cols] if target_cols else pd.DataFrame()

# Load model and scalers
model = tf.keras.models.load_model(MODEL_PATH)
x_scaler = joblib.load(X_SCALER_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)

# -----------------------
# 2️⃣  SIDEBAR CONTROLS
# -----------------------
st.sidebar.header("⚙️ Controls")
feature_x = st.sidebar.selectbox("Select Feature X", [""] + feature_cols)
feature_y = st.sidebar.selectbox("Select Feature Y", [""] + feature_cols)
target_option = st.sidebar.selectbox("Select Target Output", [""] + list(y_train.columns))
show_3d = st.sidebar.checkbox("Show 3D Surface", value=False)

if "h1" in X_test.columns:
    h1_value = st.sidebar.number_input("Constant h1 value", value=float(X_test["h1"].mean()))
else:
    h1_value = None

# -----------------------
# 3️⃣  VALIDATE INPUTS
# -----------------------
if not feature_x or not feature_y or feature_x == feature_y:
    st.warning("Please select two distinct features for visualization.")
    st.stop()

if not target_option:
    st.warning("Please select a target output.")
    st.stop()

# -----------------------
# 4️⃣  PREPARE GRID DATA
# -----------------------
f1, f2 = feature_x, feature_y
f1_range = np.linspace(X_test[f1].min(), X_test[f1].max(), 40)
f2_range = np.linspace(X_test[f2].min(), X_test[f2].max(), 40)
xx, yy = np.meshgrid(f1_range, f2_range)

grid = pd.DataFrame({f1: xx.ravel(), f2: yy.ravel()})
for col in X_test.columns:
    if col not in [f1, f2]:
        grid[col] = X_test[col].mean()

if h1_value is not None and "h1" in X_test.columns:
    grid["h1"] = h1_value

# Align feature order with scaler
scaler_features = list(x_scaler.feature_names_in_)
grid = grid.reindex(columns=scaler_features, fill_value=X_test.mean().iloc[0])
X_test = X_test.reindex(columns=scaler_features, fill_value=X_test.mean().iloc[0])

# -----------------------
# 5️⃣  SCALE, PREDICT, INVERSE SCALE
# -----------------------
X_test_scaled = x_scaler.transform(X_test)
grid_scaled = x_scaler.transform(grid)

y_pred_scaled = model.predict(X_test_scaled)
y_pred_grid_scaled = model.predict(grid_scaled)

y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_pred_grid = y_scaler.inverse_transform(y_pred_grid_scaled)

output_index = y_train.columns.get_loc(target_option)

# Reshape for contour
pred_surface = y_pred_grid[:, output_index].reshape(xx.shape)

# -----------------------
# 6️⃣  MAPE VERIFICATION
# -----------------------
if target_option in y_actual_df.columns:
    y_actual = y_actual_df[target_option].values
    mape_value = mean_absolute_percentage_error(y_actual, y_pred[:, output_index]) * 100
    st.success(f"✅ Verified MAPE for {target_option}: {mape_value:.2f}% (Expected ≈ 3.33%)")
else:
    y_actual = np.zeros_like(y_pred[:, output_index])
    st.warning(f"⚠️ No actual values for {target_option} found in test file — skipping MAPE check.")

# -----------------------
# 7️⃣  2D PLOTLY CONTOUR
# -----------------------
fig = go.Figure(data=go.Contour(
    z=pred_surface,
    x=f1_range,
    y=f2_range,
    colorscale="Viridis",
    contours=dict(showlabels=True, labelfont=dict(size=12, color="white")),
    colorbar=dict(title=f"{target_option} (Actual Scale)"),
    hovertemplate=(
        f"<b>{f1}</b>: %{{x:.3f}}<br>"
        f"<b>{f2}</b>: %{{y:.3f}}<br>"
        f"<b>Predicted {target_option}</b>: %{{z:.3f}}<extra></extra>"
    ),
))

# Overlay actual test points
if target_option in y_actual_df.columns:
    fig.add_trace(go.Scatter(
        x=X_test[f1],
        y=X_test[f2],
        mode='markers',
        marker=dict(size=6, color='red', line=dict(width=1, color='black')),
        name=f'Actual {target_option}',
        text=[
            f"{f1}: {X_test.at[i, f1]:.3f}<br>"
            f"{f2}: {X_test.at[i, f2]:.3f}<br>"
            f"Actual {target_option}: {y_actual_df[target_option].iloc[i]:.3f}"
            for i in range(len(X_test))
        ],
        hoverinfo='text'
    ))

fig.update_layout(
    title=f"Contour Plot — {f1} vs {f2} (Predicted {target_option})",
    xaxis_title=f1,
    yaxis_title=f2,
    width=850,
    height=700,
    template="plotly_white",
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------
# 8️⃣  OPTIONAL 3D SURFACE
# -----------------------
if show_3d:
    st.subheader(f"🌐 3D Surface View — {f1} vs {f2} vs {target_option}")
    fig3d = go.Figure(data=[go.Surface(
        z=pred_surface,
        x=f1_range,
        y=f2_range,
        colorscale="Viridis",
        contours={"z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen", "project_z": True}},
        colorbar=dict(title=f"{target_option}")
    )])
    fig3d.update_layout(
        title=f"3D Surface of Predicted {target_option} (H1={h1_value if h1_value else 'N/A'})",
        scene=dict(
            xaxis_title=f1,
            yaxis_title=f2,
            zaxis_title=f"Predicted {target_option}"
        ),
        width=900,
        height=750
    )
    st.plotly_chart(fig3d, use_container_width=True)

# -----------------------
# 9️⃣  SAMPLE TABLE
# -----------------------
st.markdown(f"### 🔍 Sample Actual vs Predicted {target_option} (first 15 rows)")
compare_df = pd.DataFrame({
    f1: X_test[f1].values[:15],
    f2: X_test[f2].values[:15],
    f"Pred_{target_option}": y_pred[:15, output_index],
})
if np.any(y_actual):
    compare_df[f"Actual_{target_option}"] = y_actual[:15]
st.dataframe(compare_df)






#================================above is recent ===================================================================
##!/usr/bin/env python
# coding: utf-8

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import tensorflow as tf

# -----------------------
# Config / Paths
# -----------------------
BASE_DIR = r"C:\Users\gantrav01\RD_predictability_11925"
X_TRAIN_PATH = os.path.join(BASE_DIR, "H_vs_Tau_training.xlsx")         # used only for column names
Y_TRAIN_PATH = os.path.join(BASE_DIR, "H_vs_Tau_target.xlsx")          # used only for target names
TEST_PATH    = os.path.join(BASE_DIR, "Copy of T33_100_Samples_for_testing.xlsx")  # your test-only file
MODEL_PATH   = os.path.join(BASE_DIR, "checkpoints", "h_vs_tau_best_model.keras")
X_SCALER_PATH = os.path.join(BASE_DIR, "x_eta_scaler.pkl")  # change if different
Y_SCALER_PATH = os.path.join(BASE_DIR, "y_eta_scaler.pkl")  # change if different

# -----------------------
# Load files (X_train only for column names)
# -----------------------
st.set_page_config(page_title="RSM (Test-only) Visualizer", layout="wide")
st.title("RSM Visualizer — operate on TEST file only")

try:
    X_train = pd.read_excel(X_TRAIN_PATH)
    y_train = pd.read_excel(Y_TRAIN_PATH)
except Exception as e:
    st.error(f"Could not read train header files: {e}")
    st.stop()

try:
    t33_df = pd.read_excel(TEST_PATH)
except Exception as e:
    st.error(f"Could not read test file (t33): {e}")
    st.stop()

# Clean column names (consistent cleaning)
def clean_cols(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace('[^A-Za-z0-9]+', ' ', regex=True)
    return df

X_train = clean_cols(X_train)   # only for names
y_train = clean_cols(y_train)
t33_df   = clean_cols(t33_df)   # actual test data to be used for everything

# Feature list: only those columns that exist both in X_train header and in test file
feature_cols = [c for c in X_train.columns if c in t33_df.columns]
if not feature_cols:
    st.error("No shared feature columns found between X_train header and your test file. "
             "Ensure the training header file matches test file column names.")
    st.stop()

# Target columns available in test file (intersection of y_train headers and test file)
available_targets = [c for c in y_train.columns if c in t33_df.columns]

# -----------------------
# Load model & scalers
# -----------------------
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model at {MODEL_PATH}: {e}")
    st.stop()

try:
    x_scaler = joblib.load(X_SCALER_PATH)
except Exception as e:
    st.error(f"Could not load x scaler at {X_SCALER_PATH}: {e}")
    st.stop()

try:
    y_scaler = joblib.load(Y_SCALER_PATH)
except Exception as e:
    st.error(f"Could not load y scaler at {Y_SCALER_PATH}: {e}")
    st.stop()

# -----------------------
# UI controls
# -----------------------
st.sidebar.header("Controls (test-only)")
st.sidebar.write("Note: X_train is used only for column names. All predictions use the test file (t33).")

# Feature selectors - NO defaults (user must intentionally select)
select_opts = ["-- select --"] + feature_cols
feature_x = st.sidebar.selectbox("Feature X (horizontal axis)", select_opts, index=0)
feature_y = st.sidebar.selectbox("Feature Y (vertical axis)", select_opts, index=0)

# Target selector - no default
target_opts = ["-- select --"] + (available_targets if available_targets else list(y_train.columns))
selected_target = st.sidebar.selectbox("Target to visualize (choose from test file if available)", target_opts, index=0)

# Optional: constant for h1 if present in test file
h1_const = None
if "h1" in t33_df.columns:
    h1_const = st.sidebar.number_input("Constant value for h1 (if used)", value=float(t33_df["h1"].mean()))

st.sidebar.markdown("---")
st.sidebar.write(f"Test samples in file: {len(t33_df)}")

# -----------------------
# Helper: get scaler feature order
# -----------------------
if hasattr(x_scaler, "feature_names_in_"):
    scaler_features = list(x_scaler.feature_names_in_)
else:
    # fallback to header feature_cols (order from X_train)
    scaler_features = feature_cols

# For safety, ensure scaler_features is a list of strings
scaler_features = [str(f) for f in scaler_features]

# -----------------------
# User must select valid X, Y, and Target
# -----------------------
if feature_x == "-- select --" or feature_y == "-- select --":
    st.warning("Please select both Feature X and Feature Y (they must be distinct).")
    st.stop()

if feature_x == feature_y:
    st.warning("Feature X and Feature Y must be different.")
    st.stop()

if selected_target == "-- select --":
    st.warning("Please select a target to visualize.")
    st.stop()

# -----------------------
# Build grid using test-only data
# -----------------------
# Use ranges from test file (t33_df) for the chosen variables
x_min, x_max = float(t33_df[feature_x].min()), float(t33_df[feature_x].max())
y_min, y_max = float(t33_df[feature_y].min()), float(t33_df[feature_y].max())

n_grid = 40
xx_vals = np.linspace(x_min, x_max, n_grid)
yy_vals = np.linspace(y_min, y_max, n_grid)
xx, yy = np.meshgrid(xx_vals, yy_vals)

grid = pd.DataFrame({feature_x: xx.ravel(), feature_y: yy.ravel()})

# Fill other features with their mean computed from the test file (perform on test alone)
for col in scaler_features:
    if col not in [feature_x, feature_y]:
        if col in t33_df.columns:
            grid[col] = float(t33_df[col].mean())
        else:
            # feature expected by scaler but not present in test file — fill with 0
            grid[col] = 0.0

# If user provided h1 const and 'h1' is among scaler_features, set it
if h1_const is not None and "h1" in scaler_features:
    grid["h1"] = float(h1_const)

# Reindex grid to scaler_features order (this ensures column names/order match scaler)
grid = grid.reindex(columns=scaler_features, fill_value=0.0)

# Prepare test data for model input (align with scaler_features)
X_test_for_model = t33_df.reindex(columns=scaler_features, fill_value=0.0)

# -----------------------
# Check shapes match model input
# -----------------------
expected_input_dim = None
try:
    # for Keras Functional/Sequential: model.input_shape is like (None, n_features)
    expected_input_shape = model.input_shape
    if isinstance(expected_input_shape, (list, tuple)) and len(expected_input_shape) >= 2:
        expected_input_dim = expected_input_shape[1]
except Exception:
    expected_input_dim = None

if expected_input_dim is not None:
    if expected_input_dim != grid.shape[1]:
        st.error(f"Model expected input dim = {expected_input_dim}, but prepared grid has {grid.shape[1]} features.\n"
                 "This usually means the model was trained with a different set/order of features. "
                 "Check the scaler & model training pipeline.")
        st.stop()

# -----------------------
# Scale, predict, inverse-transform
# -----------------------
try:
    grid_scaled = x_scaler.transform(grid)        # pandas -> scaler will check names if present
    X_test_scaled = x_scaler.transform(X_test_for_model)
except Exception as e:
    st.error(f"Error while scaling inputs (mismatch with scaler feature names/order): {e}")
    st.stop()

# Predictions (scaled)
y_grid_scaled = model.predict(grid_scaled)
y_test_scaled = model.predict(X_test_scaled)

# Inverse-transform predictions to original target scale
try:
    y_grid = y_scaler.inverse_transform(y_grid_scaled)
    y_test_pred = y_scaler.inverse_transform(y_test_scaled)
except Exception as e:
    st.error(f"Error while inverse-transforming predictions with y_scaler: {e}")
    st.stop()

# -----------------------
# Map target index
# -----------------------
if selected_target in list(y_train.columns):
    output_index = list(y_train.columns).index(selected_target)
else:
    st.error("Selected target not found in y_train headers. Check target names.")
    st.stop()

# Get 2D z for contour (target slice)
z_vals = y_grid[:, output_index].reshape(xx.shape)

# -----------------------
# Compute error metrics using test file actuals (if present)
# -----------------------
eps = 1e-8
if selected_target in t33_df.columns:
    y_actual_t = t33_df[selected_target].values
    y_pred_t = y_test_pred[:, output_index]
    # T1 (selected target) MAPE
    t_mape = np.mean(np.abs((y_actual_t - y_pred_t) / (np.abs(y_actual_t) + eps))) * 100
else:
    t_mape = np.nan

# Overall MAPE across all targets that are present in test file (intersection)
targets_in_test = [c for c in y_train.columns if c in t33_df.columns]
if targets_in_test:
    # Build actuals matrix and predicted matrix for the same column order
    actuals_mat = t33_df[targets_in_test].values
    # determine indices of these targets in y_train order
    idxs = [list(y_train.columns).index(c) for c in targets_in_test]
    preds_mat = y_test_pred[:, idxs]
    overall_mape = np.mean(np.abs((actuals_mat - preds_mat) / (np.abs(actuals_mat) + eps))) * 100
else:
    overall_mape = np.nan

# -----------------------
# Display results
# -----------------------
col_l, col_r = st.columns([1, 1])

with col_l:
    st.subheader(f"Response Surface — Predicted `{selected_target}` (test-only)")
    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contourf(xx, yy, z_vals, cmap="viridis", levels=20)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(f"Predicted {selected_target} (original scale)")
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_title(f"Predicted {selected_target} (from test-only grid)")
    st.pyplot(fig)

with col_r:
    st.subheader("Prediction metrics (test-only)")

    if not np.isnan(t_mape):
        st.metric(label=f"{selected_target} Mean Absolute % Error (MAPE)", value=f"{t_mape:.2f}%")
    else:
        st.write(f"MAPE for `{selected_target}`: N/A (actuals not found in test file)")

    if not np.isnan(overall_mape):
        st.metric(label="Overall Avg MAPE (all targets present in test)", value=f"{overall_mape:.2f}%")
    else:
        st.write("Overall Avg MAPE: N/A (no target columns present in test file)")

    # Show a small table of actual vs predicted for the first 15 test rows
    st.markdown("### Sample: Actual vs Predicted (first 15 rows from test file)")
    sample_df = pd.DataFrame({
        feature_x: t33_df[feature_x].values[:15],
        feature_y: t33_df[feature_y].values[:15],
        f"Pred_{selected_target}": y_test_pred[:15, output_index]
    })
    if selected_target in t33_df.columns:
        sample_df[f"Actual_{selected_target}"] = t33_df[selected_target].values[:15]
    st.dataframe(sample_df)

st.success("Done — predictions performed only on the test file (t33).")
#===========================================================================================
#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# -------------------------
# 1. Load and Prepare Data
# -------------------------
os.chdir(r'C:\Users\gantrav01\RD_predictability_11925')

df = pd.read_excel(r'T53T73_PT.xlsx')

eta_col_list = ['Mu', 'Vitesse_Rotation:', 'Variation_Travail:', 'Aur_Prh_AngMetalBA:',
       'Angle_Metal_BA_I:', 'EtaFctPhiDiamReel:', 'PolytropicEfficiencyCurveRef:',
       'Epaisseur_3_Moyeu:', 'Position_Radiale_Entree_Alimentation_I:',
       'Perte_Pression_Totale_Relative:', 'Position_Radiale_Entree_Alimentation_E:',
       'Pourcentage_Partie_Droite_E:', 'Ralentissement_Roue:',
       'Position_Axiale_Entree_Diffuseur_I:', 'Position_Radiale_Bord_Attaque_I:',
       'RoueAngFluideEntree:', 'RoueCentreGravite:', 'Debit_Masse:',
       'Hauteur_Labyrinthe_F:', 'RptVitMeridBA:', 'Jeu_Axial_Ouie_F:',
       'Aur_Prs_Beta_LaPourcentage_A_Beta_Constant_E:',
       'Position_Axiale_Bord_Attaque_E:', 'Aur_Prh_Beta_LaPoint2_X_I:',
       'Position_Radiale_Bord_Attaque_E:', 'b5_width_cdr_in_new_val:',
       'DebitVolumeEntreeSection:', 'Aur_Prh_Beta_LaPoint2_Y_I:',
       'RoueSectCol:', 'Angle_Fluide_Absolu_Sortie_Roue:',
       'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12']

X_eta = df[eta_col_list]
y_eta = df[[f"e{i}" for i in range(1, 13)]].drop('e2', axis=1)

# -------------------------
# 2. Split and Scale Data
# -------------------------
xtrain, xtest, ytrain, ytest = train_test_split(X_eta, y_eta, test_size=0.2, random_state=42)

x_scaler = StandardScaler()
xtrain_s = x_scaler.fit_transform(xtrain)
xtest_s  = x_scaler.transform(xtest)

y_scaler = StandardScaler()
ytrain_s = y_scaler.fit_transform(ytrain)
ytest_s  = y_scaler.transform(ytest)

# -------------------------
# 3. Build Lightweight ANN for SHAP
# -------------------------
def build_light_ann(input_dim, output_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    out = layers.Dense(output_dim, activation='linear')(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_light_ann(xtrain_s.shape[1], ytrain_s.shape[1])
model.fit(xtrain_s, ytrain_s, epochs=30, batch_size=64, verbose=0, validation_split=0.1)

print("\n✅ Lightweight ANN trained for SHAP analysis")

# -------------------------
# 4. SHAP DeepExplainer
# -------------------------
# Use a smaller sample to speed up computation
X_background = xtrain_s[:100]
X_eval = xtest_s[:200]

explainer = shap.DeepExplainer(model, X_background)
shap_values = explainer.shap_values(X_eval)

# shap_values is a list (one per output target)
# We'll aggregate SHAP importance across all outputs
shap_array = np.mean(np.abs(np.stack(shap_values, axis=-1)), axis=-1)
mean_abs_shap = np.mean(shap_array, axis=0)

feat_imp_df = pd.DataFrame({
    'Feature': X_eta.columns,
    'SHAP Importance': mean_abs_shap
}).sort_values('SHAP Importance', ascending=False)

# -------------------------
# 5. Select Important Features
# -------------------------
h_features = [f'h{i}' for i in range(3, 13)]  # keep all h3–h12
top_non_h_features = feat_imp_df[~feat_imp_df['Feature'].isin(h_features)].head(10)['Feature'].tolist()

selected_features = h_features + top_non_h_features

print("\nTop 10 important non-h features based on SHAP:")
print(top_non_h_features)

print("\nFinal selected features for main training:")
print(selected_features)

# -------------------------
# 6. Visualize & Save
# -------------------------
shap.summary_plot(shap_array, features=X_eval, feature_names=X_eta.columns, show=False)
plt.tight_layout()
plt.savefig("SHAP_Summary_ANN_eta.png", dpi=300)
plt.show()

feat_imp_df.to_excel("SHAP_Feature_Importance_ANN_eta.xlsx", index=False)
joblib.dump(selected_features, "selected_features_eta.pkl")

print("\n✅ SHAP feature importance completed and saved successfully!")
