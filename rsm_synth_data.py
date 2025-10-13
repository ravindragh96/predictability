# rsm_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# ----------------------------
# 1️⃣ Page setup
# ----------------------------
st.set_page_config(page_title="RSM Visualization", layout="wide")
st.title("🌀 Response Surface Modeling (RSM) Visualization")
st.markdown("""
Select any two input features for the X–Y axes and one output feature to visualize as the response surface.
This plot helps you explore how the ANN-predicted outputs vary across combinations of your inputs.
""")

# ----------------------------
# 2️⃣ Load synthetic dataset
# ----------------------------
uploaded_file = st.file_uploader("📂 Upload your synthetic dataset (Excel or CSV)", type=["xlsx", "csv"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
        
    st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
    st.write(df.head())

    # ----------------------------
    # 3️⃣ Sidebar - column selectors
    # ----------------------------
    st.sidebar.header("🔧 Choose Features")
    input_cols = [c for c in df.columns if not c.startswith('e')]  # assuming inputs are not prefixed with 'e'
    output_cols = [c for c in df.columns if c.startswith('e')]     # outputs: e1, e3–e12

    x_col = st.sidebar.selectbox("Select X-axis feature", input_cols, index=0)
    y_col = st.sidebar.selectbox("Select Y-axis feature", input_cols, index=1)
    z_col = st.sidebar.selectbox("Select output feature (Z)", output_cols, index=0)

    n_grid = st.sidebar.slider("Grid resolution", 20, 100, 50)

    # ----------------------------
    # 4️⃣ Prepare data for contour
    # ----------------------------
    X = df[x_col].values
    Y = df[y_col].values
    Z = df[z_col].values

    # Create smooth grid
    xi = np.linspace(X.min(), X.max(), n_grid)
    yi = np.linspace(Y.min(), Y.max(), n_grid)
    XI, YI = np.meshgrid(xi, yi)

    # Interpolate Z values on grid
    ZI = griddata((X, Y), Z, (XI, YI), method='cubic')

    # ----------------------------
    # 5️⃣ Plot contour like example image
    # ----------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(XI, YI, ZI, levels=20, cmap='RdYlGn_r')  # red-high, green-low like example
    plt.colorbar(contour, ax=ax, label=z_col)
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.set_title(f"Response Surface: {z_col} vs {x_col} & {y_col}", fontsize=14)
    st.pyplot(fig)

    # ----------------------------
    # 6️⃣ Optional Data download
    # ----------------------------
    st.sidebar.markdown("### 💾 Download Filtered Data")
    filtered_df = df[[x_col, y_col, z_col]]
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download X–Y–Z Data (CSV)",
        data=csv,
        file_name=f"RSM_{x_col}_{y_col}_{z_col}.csv",
        mime='text/csv'
    )

else:
    st.info("👆 Upload your synthetic dataset (the one with ANN predictions) to begin.")
