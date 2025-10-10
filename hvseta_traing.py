#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard


# In[2]:


os.chdir(r'C:\Users\gantrav01\RD_predictability_11925')


# In[3]:


pd.set_option("display.max_columns",None)


# In[4]:


df = pd.read_excel(r'T53T73_PT.xlsx')


# In[5]:


eta_col_list = ['Mu', 'Vitesse_Rotation:', 'Variation_Travail:', 'Aur_Prh_AngMetalBA:',
       'Angle_Metal_BA_I:', 'EtaFctPhiDiamReel:',
       'PolytropicEfficiencyCurveRef:', 'Epaisseur_3_Moyeu:',
       'Position_Radiale_Entree_Alimentation_I:',
       'Perte_Pression_Totale_Relative:',
       'Position_Radiale_Entree_Alimentation_E:',
       'Pourcentage_Partie_Droite_E:', 'Ralentissement_Roue:',
       'Position_Axiale_Entree_Diffuseur_I:',
       'Position_Radiale_Bord_Attaque_I:', 'RoueAngFluideEntree:',
       'RoueCentreGravite:', 'Debit_Masse:', 'Hauteur_Labyrinthe_F:',
       'RptVitMeridBA:', 'Jeu_Axial_Ouie_F:',
       'Aur_Prs_Beta_LaPourcentage_A_Beta_Constant_E:',
       'Position_Axiale_Bord_Attaque_E:', 'Aur_Prh_Beta_LaPoint2_X_I:',
       'Position_Radiale_Bord_Attaque_E:', 'b5_width_cdr_in_new_val:',
       'DebitVolumeEntreeSection:', 'Aur_Prh_Beta_LaPoint2_Y_I:',
       'RoueSectCol:', 'Angle_Fluide_Absolu_Sortie_Roue:','h3', 'h4', 'h5',
       'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12'] #have to be added the h2 here 


# In[6]:


X_eta = df[eta_col_list]


# In[7]:


X_eta


# In[8]:


X_eta.to_excel(r'h_vs_eta_training.xlsx',index=False)


# In[9]:


y_eta = df[[f"e{i}" for i in range(1, 13)]]


# In[10]:


y_eta = y_eta.drop('e2', axis = 1)


# In[11]:


y_eta.to_excel(r'H_vs_eta_Target.xlsx', index = False)


# In[12]:


# reproducibility (best effort)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# In[13]:


# Train/Val/Test split (keep yours)
xtrain, xtest, ytrain, ytest = train_test_split(X_eta, y_eta, test_size=0.2, random_state=SEED)
xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.25, random_state=SEED)
print("Shapes:", xtrain.shape, xval.shape, xtest.shape, ytrain.shape, yval.shape, ytest.shape)


# In[14]:


# ---------- Scaling ----------
x_scaler = StandardScaler()
xtrain_s = x_scaler.fit_transform(xtrain)
xval_s   = x_scaler.transform(xval)
xtest_s  = x_scaler.transform(xtest)

y_scaler = StandardScaler()
ytrain_s = y_scaler.fit_transform(ytrain)
yval_s   = y_scaler.transform(yval)
ytest_s  = y_scaler.transform(ytest)


# In[15]:


import joblib
joblib.dump(x_scaler, "x_eta_scaler.pkl")
joblib.dump(y_scaler, "y_eta_scaler.pkl")


# In[16]:


# Model builder with L2, Dropout, BatchNorm and optional kernel_constraint
def build_model(input_dim, output_dim,
                l2_reg=1e-4,
                dropout_rate=0.25,
                use_batchnorm=True):
    reg = regularizers.l2(l2_reg)
    inp = layers.Input(shape=(input_dim,))
    x = inp

    x = layers.Dense(1024, activation='relu', kernel_regularizer=reg)(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(512, activation='relu', kernel_regularizer=reg)(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(256, activation='relu', kernel_regularizer=reg)(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate * 0.8)(x)

    x = layers.Dense(128, activation='relu', kernel_regularizer=reg)(x)
    if use_batchnorm: x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)

    out = layers.Dense(output_dim, activation='linear')(x)
    model = models.Model(inputs=inp, outputs=out)
    return model


# In[17]:


def get_optimizer(lr=1e-3, weight_decay=1e-5):
    try:
        # TF newer API
        opt = tf.keras.optimizers.experimental.AdamW(learning_rate=lr, weight_decay=weight_decay)
        print("Using AdamW optimizer.")
    except Exception:
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        print("Using Adam optimizer (no AdamW available).")
    return opt


# In[18]:


# Learning rate schedule (InverseTimeDecay or ReduceLROnPlateau)
# Example: InverseTimeDecay for smoother decay
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=1e-1,
    staircase=False
)


# In[19]:


# Compile model (Huber loss + metrics including MAPE)
input_dim = xtrain_s.shape[1]
output_dim = ytrain_s.shape[1]
model = build_model(input_dim=input_dim, output_dim=output_dim, l2_reg=1e-4, dropout_rate=0.3)
optimizer = get_optimizer(lr=1e-3, weight_decay=1e-5)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.Huber(),            # robust to outliers
    metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae'),
             tf.keras.metrics.MeanAbsolutePercentageError(name='mape')]
)
model.summary()


# In[20]:


# Callbacks (important: include ModelCheckpoint)
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
chk_path = os.path.join(checkpoint_dir, "h_vs_eta_best_model.keras")

cb_checkpoint = ModelCheckpoint(
    filepath=chk_path,
    save_best_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1
)


# In[21]:


# early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-7, verbose=1)

# optional: tensorboard logging
tb_cb = TensorBoard(log_dir="logs", histogram_freq=0)

cb_list = [cb_checkpoint, reduce_lr, tb_cb]


# In[22]:


history = model.fit(
    xtrain_s, ytrain_s,
    validation_data=(xval_s, yval_s),
    epochs=700,
    batch_size=64,
    callbacks=cb_list,
    verbose=2
)


# In[23]:


try:
    best_model = tf.keras.models.load_model(chk_path)
    print("Loaded best model from checkpoint.")
except Exception as e:
    print("Could not load from checkpoint, using current model. Error:", e)


# In[24]:


# Evaluate on test set (scaled -> then inverse)
loss, mae, mape = model.evaluate(xtest_s, ytest_s, verbose=1)
print(f"Test Loss: {loss:.6f} | Test MAE: {mae:.6f} | Test MAPE: {mape:.6f}")


# In[25]:


# Predict and inverse-transform
y_pred_s = model.predict(xtest_s)
y_pred = y_scaler.inverse_transform(y_pred_s)
y_test_orig = y_scaler.inverse_transform(ytest_s)


# In[26]:


cols = y_eta.columns.tolist()
df_res = pd.DataFrame()
eps = 1e-8
for i, col in enumerate(cols):
    df_res[f"{col}_actual"] = y_test_orig[:, i]
    df_res[f"{col}_pred"]   = y_pred[:, i]
    df_res[f"{col}_error%"] = np.abs((df_res[f"{col}_actual"] - df_res[f"{col}_pred"]) /
                                     (df_res[f"{col}_actual"] + eps)) * 100

# Column-wise average % error (MAPE per target)
error_cols = [c for c in df_res.columns if c.endswith("_error%")]
column_avg_error = df_res[error_cols].mean()
overall_avg_error = df_res[error_cols].values.mean()

print("\nAverage Error % per target (MAPE per output):")
print(column_avg_error)
print("\nOverall average error % (across all outputs & rows): {:.6f}%".format(overall_avg_error))


# In[27]:


# Plot loss curve
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




