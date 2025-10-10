ValueError: Invalid property specified for object of type plotly.graph_objs.contour.ColorBar: 'titleside' Did you mean "title"? Valid properties: bgcolor Sets the color of padded area. bordercolor Sets the axis line color. borderwidth Sets the width (in px) or the border enclosing this color bar. dtick Sets the step in-between ticks on this axis. Use with `tick0`. Must be a positive number, or special strings available to "log" and "date" axes. If the axis `type` is "log", then ticks are set every 10^(n*dtick) where n is the tick number. For example, to set a tick mark at 1, 10, 100, 1000, ... set dtick to 1. To set tick marks at 1, 100, 10000, ... set dtick to 2. To set tick marks at 1, 5, 25, 125, 625, 3125, ... set dtick to log_10(5), or 0.69897000433. "log" has several special values; "L<f>", where `f` is a positive number, gives ticks linearly spaced in value (but not position). For example `tick0` = 0.1, `dtick` = "L0.5" will put ticks at 0.1, 0.6, 1.1, 1.6 etc. To show powers of 10 plus small digits between, use "D1" (all digits) or "D2" (only 2 and 5). `tick0` is ignored for "D1" and "D2". If the axis `type` is "date", then you must convert the time to milliseconds. For example, to set the interval between ticks to one day, set `dtick` to 86400000.0. "date" also has special values "M<n>" gives ticks spaced by a number of months. `n` must be a positive integer. To set ticks on the 15th of every third month, set `tick0` to "2000-01-15" and `dtick` to "M3". To set ticks every 4 years, set `dtick` to "M48" exponentformat Determines a formatting rule for the tick exponents. For example, consider the number 1,000,000,000. If "none", it appears as 1,000,000,000. If "e", 1e+9. If "E", 1E+9. If "power", 1x10^9 (with 9 in a super script). If "SI", 1G. If "B", 1B. labelalias Replacement text for specific tick or hover labels. For example using {US: 'USA', CA: 'Canada'} changes US to USA and CA to Canada. The labels we would have shown must match the keys exactly, after adding any tickprefix or ticksuffix. For negative numbers the minus sign symbol used (U+2212) is wider than the regular ascii dash. That means you need to use −1 instead of -1. labelalias can be used with any axis type, and both keys (if needed) and values (if desired) can include html-like tags or MathJax. len Sets the length of the color bar This measure excludes the padding of both ends. That is, the color bar length is this length minus the padding on both ends. lenmode Determines whether this color bar's length (i.e. the measure in the color variation direction) is set in units of plot "fraction" or in *pixels. Use `len` to set the value. minexponent Hide SI prefix for 10^n if |n| is below this number. This only has an effect when `tickformat` is "SI" or "B". nticks Specifies the maximum number of ticks for the particular axis. The actual number of ticks will be chosen automatically to be less than or equal to `nticks`. Has an effect only if `tickmode` is set to "auto". orientation Sets the orientation of the colorbar. outlinecolor Sets the axis line color. outlinewidth Sets the width (in px) of the axis line. separatethousands If "true", even 4-digit integers are separated showexponent If "all", all exponents are shown besides their significands. If "first", only the exponent of the first tick is shown. If "last", only the exponent of the last tick is shown. If "none", no exponents appear. showticklabels Determines whether or not the tick labels are drawn. showtickprefix If "all", all tick labels are displayed with a prefix. If "first", only the first tick is displayed with a prefix. If "last", only the last tick is displayed with a suffix. If "none", tick prefixes are hidden. showticksuffix Same as `showtickprefix` but for tick suffixes. thickness Sets the thickness of the color bar This measure excludes the size of the padding, ticks and labels. thicknessmode Determines whether this color bar's thickness (i.e. the measure in the constant color direction) is set in units of plot "fraction" or in "pixels". Use `thickness` to set the value. tick0 Sets the placement of the first tick on this axis. Use with `dtick`. If the axis `type` is "log", then you must take the log of your starting tick (e.g. to set the starting tick to 100, set the `tick0` to 2) except when `dtick`=*L<f>* (see `dtick` for more info). If the axis `type` is "date", it should be a date string, like date data. If the axis `type` is "category", it should be a number, using the scale where each category is assigned a serial number from zero in the order it appears. tickangle Sets the angle of the tick labels with respect to the horizontal. For example, a `tickangle` of -90 draws the tick labels vertically. tickcolor Sets the tick color. tickfont Sets the color bar's tick label font tickformat Sets the tick label formatting rule using d3 formatting mini-languages which are very similar to those in Python. For numbers, see: https://github.com/d3/d3-format/tree/v1.4.5#d3-format. And for dates see: https://github.com/d3/d3-time- format/tree/v2.2.3#locale_format. We add two items to d3's date formatter: "%h" for half of the year as a decimal number as well as "%{n}f" for fractional seconds with n digits. For example, *2016-10-13 09:15:23.456* with tickformat "%H~%M~%S.%2f" would display "09~15~23.46" tickformatstops A tuple of :class:`plotly.graph_objects.contour.colorba r.Tickformatstop` instances or dicts with compatible properties tickformatstopdefaults When used in a template (as layout.template.data.contou r.colorbar.tickformatstopdefaults), sets the default property values to use for elements of contour.colorbar.tickformatstops ticklabeloverflow Determines how we handle tick labels that would overflow either the graph div or the domain of the axis. The default value for inside tick labels is *hide past domain*. In other cases the default is *hide past div*. ticklabelposition Determines where tick labels are drawn relative to the ticks. Left and right options are used when `orientation` is "h", top and bottom when `orientation` is "v". ticklabelstep Sets the spacing between tick labels as compared to the spacing between ticks. A value of 1 (default) means each tick gets a label. A value of 2 means shows every 2nd label. A larger value n means only every nth tick is labeled. `tick0` determines which labels are shown. Not implemented for axes with `type` "log" or "multicategory", or when `tickmode` is "array". ticklen Sets the tick length (in px). tickmode Sets the tick mode for this axis. If "auto", the number of ticks is set via `nticks`. If "linear", the placement of the ticks is determined by a starting position `tick0` and a tick step `dtick` ("linear" is the default value if `tick0` and `dtick` are provided). If "array", the placement of the ticks is set via `tickvals` and the tick text is `ticktext`. ("array" is the default value if `tickvals` is provided). tickprefix Sets a tick label prefix. ticks Determines whether ticks are drawn or not. If "", this axis' ticks are not drawn. If "outside" ("inside"), this axis' are drawn outside (inside) the axis lines. ticksuffix Sets a tick label suffix. ticktext Sets the text displayed at the ticks position via `tickvals`. Only has an effect if `tickmode` is set to "array". Used with `tickvals`. ticktextsrc Sets the source reference on Chart Studio Cloud for `ticktext`. tickvals Sets the values at which ticks on this axis appear. Only has an effect if `tickmode` is set to "array". Used with `ticktext`. tickvalssrc Sets the source reference on Chart Studio Cloud for `tickvals`. tickwidth Sets the tick width (in px). title :class:`plotly.graph_objects.contour.colorbar.Title` instance or dict with compatible properties x Sets the x position with respect to `xref` of the color bar (in plot fraction). When `xref` is "paper", defaults to 1.02 when `orientation` is "v" and 0.5 when `orientation` is "h". When `xref` is "container", defaults to 1 when `orientation` is "v" and 0.5 when `orientation` is "h". Must be between 0 and 1 if `xref` is "container" and between "-2" and 3 if `xref` is "paper". xanchor Sets this color bar's horizontal position anchor. This anchor binds the `x` position to the "left", "center" or "right" of the color bar. Defaults to "left" when `orientation` is "v" and "center" when `orientation` is "h". xpad Sets the amount of padding (in px) along the x direction. xref Sets the container `x` refers to. "container" spans the entire `width` of the plot. "paper" refers to the width of the plotting area only. y Sets the y position with respect to `yref` of the color bar (in plot fraction). When `yref` is "paper", defaults to 0.5 when `orientation` is "v" and 1.02 when `orientation` is "h". When `yref` is "container", defaults to 0.5 when `orientation` is "v" and 1 when `orientation` is "h". Must be between 0 and 1 if `yref` is "container" and between "-2" and 3 if `yref` is "paper". yanchor Sets this color bar's vertical position anchor This anchor binds the `y` position to the "top", "middle" or "bottom" of the color bar. Defaults to "middle" when `orientation` is "v" and "bottom" when `orientation` is "h". ypad Sets the amount of padding (in px) along the y direction. yref Sets the container `y` refers to. "container" spans the entire `height` of the plot. "paper" refers to the height of the plotting area only. Did you mean "title"? Bad property path: titleside ^^^^^^^^^
Traceback:
File "C:\Users\gantrav01\RD_predictability_11925\rsm_tau_fin.py", line 155, in <module>
    fig = go.Figure(data=go.Contour(
                         ^^^^^^^^^^^
File "C:\Users\gantrav01\AppData\Local\anaconda3\Lib\site-packages\plotly\graph_objs\_contour.py", line 2544, in __init__
    self._set_property("colorbar", arg, colorbar)
File "C:\Users\gantrav01\AppData\Local\anaconda3\Lib\site-packages\plotly\basedatatypes.py", line 4403, in _set_property
    _set_property_provided_value(self, name, arg, provided)
File "C:\Users\gantrav01\AppData\Local\anaconda3\Lib\site-packages\plotly\basedatatypes.py", line 398, in _set_property_provided_value
    obj[name] = val
    ~~~^^^^^^
File "C:\Users\gantrav01\AppData\Local\anaconda3\Lib\site-packages\plotly\basedatatypes.py", line 4924, in __setitem__
    self._set_compound_prop(prop, value)
File "C:\Users\gantrav01\AppData\Local\anaconda3\Lib\site-packages\plotly\basedatatypes.py", line 5335, in _set_compound_prop
    val = validator.validate_coerce(val, skip_invalid=self._skip_invalid)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\gantrav01\AppData\Local\anaconda3\Lib\site-packages\_plotly_utils\basevalidators.py", line 2425, in validate_coerce
    v = self.data_class(v, skip_invalid=skip_invalid, _validate=_validate)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\gantrav01\AppData\Local\anaconda3\Lib\site-packages\plotly\graph_objs\contour\_colorbar.py", line 1721, in __init__
    self._process_kwargs(**dict(arg, **kwargs))
File "C:\Users\gantrav01\AppData\Local\anaconda3\Lib\site-packages\plotly\basedatatypes.py", line 4451, in _process_kwargs
    raise err

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
# 1️⃣ Setup
# -----------------------
st.set_page_config(page_title="RSM Contour App (Original Names)", layout="wide")
st.title("🎛️ Response Surface Modeling (RSM) — Original Column Names Preserved")

BASE_DIR = r"C:\Users\gantrav01\RD_predictability_11925"

TRAIN_X_PATH = os.path.join(BASE_DIR, "H_vs_Tau_training.xlsx")
TRAIN_Y_PATH = os.path.join(BASE_DIR, "H_vs_Tau_target.xlsx")
TEST_PATH = os.path.join(BASE_DIR, "Copy of T33_100_Samples_for_testing.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "h_vs_tau_best_model.keras")
X_SCALER_PATH = os.path.join(BASE_DIR, "x_eta_scaler.pkl")
Y_SCALER_PATH = os.path.join(BASE_DIR, "y_eta_scaler.pkl")

# -----------------------
# 2️⃣ Load Data (NO COLUMN CLEANING)
# -----------------------
X_train = pd.read_excel(TRAIN_X_PATH)
y_train = pd.read_excel(TRAIN_Y_PATH)
t33_df = pd.read_excel(TEST_PATH)

# Extract matching columns
feature_cols = [c for c in X_train.columns if c in t33_df.columns]
X_test = t33_df[feature_cols]
target_cols = [c for c in y_train.columns if c in t33_df.columns]
y_actual_df = t33_df[target_cols] if target_cols else pd.DataFrame()

# Force H1 constant = 100
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
# 4️⃣ Verify Feature Alignment
# -----------------------
if hasattr(x_scaler, "feature_names_in_"):
    scaler_features = list(x_scaler.feature_names_in_)
else:
    scaler_features = list(X_train.columns)

missing_in_test = [c for c in scaler_features if c not in X_test.columns]
extra_in_test = [c for c in X_test.columns if c not in scaler_features]

st.sidebar.subheader("🧩 Feature Alignment Check")
st.sidebar.write(f"Scaler trained with {len(scaler_features)} features")
st.sidebar.write(f"Test data has {len(X_test.columns)} features")

if missing_in_test:
    st.sidebar.warning(f"⚠️ Missing in test data: {missing_in_test}")
if extra_in_test:
    st.sidebar.info(f"ℹ️ Extra columns in test data: {extra_in_test}")

# Fill missing columns safely
X_mean = X_test.mean(numeric_only=True)
for col in missing_in_test:
    if col == "H1":
        X_test[col] = 100.0
    else:
        X_test[col] = X_mean.mean()

# Reindex columns to match scaler order
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

if output_to_plot in y_actual_df.columns:
    y_actual = y_actual_df[output_to_plot].values
    eps = 1e-8
    mape_val = np.mean(np.abs((y_actual - y_pred[:, output_index]) / (y_actual + eps))) * 100
    st.success(f"✅ Verified MAPE for {output_to_plot}: {mape_val:.2f}% (Expected ≈ 3.33%)")
else:
    y_actual = np.zeros_like(y_pred[:, output_index])
    st.warning(f"⚠️ No actual values for {output_to_plot} found — skipping MAPE check.")

# -----------------------
# 7️⃣ Contour Grid (Preserve Original Names)
# -----------------------
f1, f2 = feature_x, feature_y
if f1 not in X_test.columns or f2 not in X_test.columns:
    st.error(f"❌ One of the selected features ({f1} or {f2}) is not found in test data.")
    st.stop()

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
# 8️⃣ Plotly Contour Plot
# -----------------------
fig = go.Figure(data=go.Contour(
    z=preds,
    x=f1_range,
    y=f2_range,
    colorscale="Viridis",
    colorbar=dict(title=f"{output_to_plot} (Actual Scale)", titleside="right"),
    contours=dict(showlabels=True, labelfont=dict(size=12, color="white")),
    hovertemplate=(
        f"<b>{f1}</b>: %{{x:.3f}}<br>"
        f"<b>{f2}</b>: %{{y:.3f}}<br>"
        f"<b>Predicted {output_to_plot}</b>: %{{z:.3f}}<extra></extra>"
    ),
))

# Overlay actual points
if output_to_plot in y_actual_df.columns:
    fig.add_trace(go.Scatter(
        x=X_test[f1],
        y=X_test[f2],
        mode="markers",
        marker=dict(size=6, color="red", line=dict(width=1, color="black")),
        name=f"Actual {output_to_plot}"
    ))

fig.update_layout(
    title=f"RSM Contour: {f1} vs {f2} (H1 fixed at 100) — Output: {output_to_plot}",
    xaxis_title=f1,
    yaxis_title=f2,
    width=850,
    height=650,
    template="plotly_white",
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------
# 9️⃣ Display Predictions Table
# -----------------------
st.markdown(f"### 🔍 Sample Predictions for `{output_to_plot}` (first 10 rows)")
compare_df = pd.DataFrame({
    f1: X_test[f1].values[:10],
    f2: X_test[f2].values[:10],
    f"Pred_{output_to_plot}": y_pred[:10, output_index],
})
if np.any(y_actual):
    compare_df[f"Actual_{output_to_plot}"] = y_actual[:10]
st.dataframe(compare_df)
