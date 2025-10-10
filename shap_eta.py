import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import shap
import plotly.graph_objects as go
import itertools
import plotly.io as pio
# -----------------------
# Load your data and model
# -----------------------
# Replace with your paths
# X_train, X_test, y_test, model = ...
# Assume model is your trained ANN
os.chdir(r'C:\Users\gantrav01\RD_predictability_11925')

X_train = pd.read_excel(r'H_vs_Tau_training.xlsx')
y_train = pd.read_excel(r'H_vs_Tau_target.xlsx')
t33_df = pd.read_excel(r'Copy of T33_100_Samples_for_testing.xlsx')

# Clean column names
X_train.columns = X_train.columns.str.strip().str.replace('[^A-Za-z0-9]+', ' ', regex=True)
t33_df.columns = t33_df.columns.str.strip().str.replace('[^A-Za-z0-9]+', ' ', regex=True)

X_test = t33_df[X_train.columns]
y_actual_df = t33_df[[col for col in t33_df.columns if col in y_train.columns]]  # extract actual targets if available

# Convert to numpy
X_train_np = X_train.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)

# Load model
chk_path = r'checkpoints/h_vs_tau_best_model.keras'
model = tf.keras.models.load_model(chk_path)
st.set_page_config(page_title="RSM Visualization App", layout="wide")

st.title("🎛️ Response Surface Modeling (RSM) Interactive App")

# -----------------------
# Sidebar controls
# -----------------------
st.sidebar.header("Select Features for RSM")

all_features = X_test.columns.tolist()
feature_x = st.sidebar.selectbox("Select Feature X", all_features, index=0)
feature_y = st.sidebar.selectbox("Select Feature Y", all_features, index=1)

# Keep h1 constant (you can define it explicitly)
h1_value = 100
st.sidebar.write(f"Constant h1 = {h1_value}")

# -----------------------
# Prepare grid for RSM
# -----------------------
x_range = np.linspace(X_test[feature_x].min(), X_test[feature_x].max(), 40)
y_range = np.linspace(X_test[feature_y].min(), X_test[feature_y].max(), 40)

xx, yy = np.meshgrid(x_range, y_range)
grid = pd.DataFrame({feature_x: xx.ravel(), feature_y: yy.ravel()})

# Keep all other features constant at their mean
for col in X_test.columns:
    if col not in [feature_x, feature_y]:
        grid[col] = X_test[col].mean()

# Add constant h1 column if applicable
if "h1" in X_test.columns:
    grid["h1"] = h1_value

# -----------------------
# Model Predictions
# -----------------------
y_pred_grid = model.predict(grid)
y_pred_grid = np.array(y_pred_grid).flatten()
zz = y_pred_grid.reshape(xx.shape)

# -----------------------
# Layout
# -----------------------
col1, col2 = st.columns(2)

# ---- Left: 2D RSM Plot ----
with col1:
    st.subheader("📈 RSM 2D Visualization")
    fig, ax = plt.subplots(figsize=(6,5))
    contour = ax.contourf(xx, yy, zz, cmap="viridis")
    cbar = plt.colorbar(contour)
    cbar.set_label("Predicted T1")
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_title("Response Surface of Predicted T1")
    st.pyplot(fig)

# ---- Right: Actual vs Predicted ----
with col2:
    st.subheader("📊 Actual vs Predicted T1 (Test Data)")
    y_pred_test = model.predict(X_test)
    df_results = pd.DataFrame({
        "Actual_T1": y_test.flatten(),
        "Predicted_T1": y_pred_test.flatten()
    })
    # Optional: add chosen features for reference
    df_results[feature_x] = X_test[feature_x].values
    df_results[feature_y] = X_test[feature_y].values
    st.dataframe(df_results[[feature_x, feature_y, "Actual_T1", "Predicted_T1"]].head(15))



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
