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


t33_df = pd.read_excel(r"Copy of T33_100_Samples_for_testing.xlsx")
X_train_df = pd.read_excel(r'H_vs_Tau_training.xlsx')
y_train_df =  pd.read_excel(r'H_vs_Tau_target.xlsx')



# In[5]:


X_test_data = t33_df[X_train_df.columns]
X_test_data


# In[6]:


y_test_data = t33_df[y_train_df.columns]
y_test_data


# In[17]:


#Generate the synthetic daat within min-max range of each feature
n_samples = 98
synthetic_data = pd.DataFrame()


# In[18]:


for col in X_test_data.columns:
    min_val = X_test_data[col].min()
    max_val = X_test_data[col].max()
    synthetic_data[col] = np.random.uniform(min_val, max_val, n_samples)


# In[19]:


synthetic_data


# In[20]:


#load the model
chk_path = r'checkpoints/h_vs_tau_best_model.keras'
try:
    trained_model = tf.keras.models.load_model(chk_path)
    print("Loaded best model from checkpoint.")
except Exception as e:
    print("Could not load from checkpoint, using current model. Error:", e)


# In[21]:


#scale the synthetic data
import joblib
x_scaler = joblib.load('x_scaler.pkl')
y_scaler = joblib.load("y_scaler.pkl")


# In[22]:


synth_scaled = x_scaler.transform(synthetic_data)


# In[23]:


#predict the y using tarined model on synth data
y_synth_pred = trained_model.predict(synth_scaled, verbose=0)


# In[24]:


y_synth_inv = y_scaler.inverse_transform(y_synth_pred)


# In[25]:


#combine the X,y 
output_cols = [f't{i}' for i in range(1,13) if i != 2]
syn_full = synthetic_data.copy()
for i, col in enumerate(output_cols):
    syn_full[col] = y_synth_inv[:, i]


# In[27]:


syn_full.to_excel("synthetic_tau_98.xlsx", index = False)


# In[ ]:




