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
