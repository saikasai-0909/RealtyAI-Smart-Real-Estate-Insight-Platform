# realty_flow_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 10})
import datetime
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="🏡 Realty AI App", page_icon="🏠", layout="wide")

# ---------------- DARK THEME CSS (Option B) ----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    :root{
      --bg:#0b1220;
      --card:#0f1724;
      --muted:#9aa8c3;
      --accent1:#0ea5e9;
      --accent2:#7c3aed;
      --glass: rgba(255,255,255,0.03);
      --success: #16a34a;
    }
    html, body, [class*="css"]  {
      background: linear-gradient(180deg, var(--bg) 0%, #071022 100%) !important;
      color: #e6eef8 !important;
      font-family: 'Inter', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    .reportview-container .main .block-container{
      padding-top: 1rem;
      padding-left: 2rem;
      padding-right: 2rem;
      max-width: 1400px;
    }
    .step-badge {
      background: linear-gradient(90deg,var(--accent1),var(--accent2));
      color: #051029;
      padding: 6px 12px;
      border-radius: 10px;
      font-weight: 700;
      display:inline-block;
      box-shadow: 0 6px 18px rgba(124,58,237,0.18);
      font-size: 13px;
    }
    .card {
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      border: 1px solid rgba(255,255,255,0.04);
      padding: 14px;
      border-radius: 12px;
      box-shadow: 0 8px 30px rgba(2,6,23,0.6);
    }
    .small-muted { color: var(--muted); font-size: 13px; }
    .big-pred {
      font-size:28px;
      font-weight:800;
      color: white;
      margin:0;
    }
    .btn-primary {
      background: linear-gradient(90deg,var(--accent1),var(--accent2));
      color: white !important;
      padding: 10px 18px;
      border-radius: 10px;
      font-weight: 700;
      border: none;
      box-shadow: 0 6px 18px rgba(14,165,233,0.12);
    }
    .btn-ghost {
      background: transparent;
      color: var(--muted);
      border: 1px solid rgba(255,255,255,0.04);
      padding: 8px 14px;
      border-radius: 10px;
    }
    .metric-box {
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      padding: 14px; border-radius: 10px; text-align:center;
      border: 1px solid rgba(255,255,255,0.03);
    }
    .muted-small { color: #9aa8c3; font-size:12px; }
    .kpi { font-size:20px; font-weight:700; color: white; margin-top:6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- SESSION INIT ----------------
if "step" not in st.session_state:
    st.session_state.step = 1

# session keys
for k in [
    "sat_image_bytes", "sat_mask_bytes", "sat_classification",
    "reg_inputs_df", "reg_pred_df", "reg_pred_enc_df", "reg_manual_inputs",
    "zillow_inputs_df", "zillow_pred_df", "zillow_forecast_df",
    "zillow_selected_region", "zillow_forecast_years", "z_manual",
    "sat_uploaded_flag", "sat_pred_done"
]:
    if k not in st.session_state:
        st.session_state[k] = None

# ---------------- MODEL PATHS (change these to your actual paths) ----------------
UNET_MODEL_PATH = r"C:\Users\HP\Desktop\Reality_AI\spacenet\unet_best_model.keras"
LGB_MODEL_PATH = r"C:\Users\HP\Desktop\Reality_AI\housing\lightgbm_housing_model.pkl"
ZILLOW_MODEL_PATH = r"C:\Users\HP\Desktop\Reality_AI\zillow\zillow_lstm_final.keras"
ZILLOW_FEATURE_SCALER = r"C:\Users\HP\Desktop\Reality_AI\zillow\feature_scaler.save"
ZILLOW_TARGET_SCALER = r"C:\Users\HP\Desktop\Reality_AI\zillow\target_scaler.save"

# ---------------- REGRESSION FEATURE LIST (as provided) ----------------
REGRESSION_FEATURES = [
 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape',
 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu',
 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold',
 'SaleType', 'SaleCondition'
]

# ---------------- MODEL LOADERS (cached) ----------------
@st.cache_resource
def load_unet():
    try:
        return tf.keras.models.load_model(UNET_MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"Error loading U-Net model: {e}")
        return None

@st.cache_resource
def load_lgb():
    try:
        return joblib.load(LGB_MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading LightGBM model: {e}")
        return None

@st.cache_resource
def load_zillow_model_and_scalers():
    try:
        model = load_model(ZILLOW_MODEL_PATH)
        feat_scaler = joblib.load(ZILLOW_FEATURE_SCALER)
        targ_scaler = joblib.load(ZILLOW_TARGET_SCALER)
        return model, feat_scaler, targ_scaler
    except Exception as e:
        st.error(f"Error loading Zillow LSTM or scalers: {e}")
        return None, None, None

# ---------------- HEADER ----------------
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:14px;">
      <div style="font-size:26px;font-weight:800;color:white">🏡 Realty AI</div>
      <div class="small-muted">— Satellite • Price Prediction • Zillow Forecast</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("---")

# ---------------- NAV / STEP UI ----------------
col1, col2, col3, col4 = st.columns([1,1,1,6])
with col1:
    st.markdown(f"<div class='step-badge'>Step 1</div>", unsafe_allow_html=True)
    st.write("Satellite")
with col2:
    st.markdown(f"<div class='step-badge'>Step 2</div>", unsafe_allow_html=True)
    st.write("Price Prediction")
with col3:
    st.markdown(f"<div class='step-badge'>Step 3</div>", unsafe_allow_html=True)
    st.write("Zillow Forecast")
with col4:
    st.write("")

st.write("---")

# ---------------- HELPERS ----------------
def go_next():
    st.session_state.step = min(4, st.session_state.step + 1)

def go_back():
    st.session_state.step = max(1, st.session_state.step - 1)

def pil_to_bytes(img, fmt="PNG"):
    b = BytesIO()
    img.save(b, format=fmt)
    b.seek(0)
    return b.getvalue()

def bytes_to_pil(b):
    return Image.open(BytesIO(b)).convert("RGB")

# keep compute_feature_contributions if wanted later (not used in minimal UI)
def compute_feature_contributions(lgb_model, input_df, prediction):
    # simplified fallback (not used in UI)
    row = input_df.iloc[0].copy()
    feats = list(row.index)
    vals = np.array([float(row[f]) if pd.notna(row[f]) else 0.0 for f in feats])
    absvals = np.abs(vals) + 1e-9
    if absvals.sum() == 0:
        contribs = [prediction / len(feats)] * len(feats)
    else:
        weights = absvals / absvals.sum()
        contribs = weights * prediction
    return list(zip(feats, contribs.tolist()))

def build_forecast_for_region(z_model, feat_scaler, targ_scaler, df_region, features, forecast_years):
    TIMESTEPS = 12
    df_region = df_region.sort_values(['year','month']).reset_index(drop=True)
    r_feat = df_region[features].fillna(0)
    scaled = feat_scaler.transform(r_feat)
    preds = [np.nan]*len(df_region)
    for i in range(TIMESTEPS, len(df_region)):
        seq = scaled[i-TIMESTEPS:i].reshape(1, TIMESTEPS, -1)
        p_scaled = z_model.predict(seq, verbose=0)
        p = targ_scaler.inverse_transform(p_scaled)[0][0]
        preds[i] = p
    df_region['Predicted_ZHVI'] = preds

    # avg growth for zhvi features
    zhvi_feats = [f for f in features if f.startswith('zhvi')]
    avg_growth = {}
    for feat in zhvi_feats:
        vals = df_region[feat].dropna()
        avg_growth[feat] = (vals.iloc[-1] - vals.iloc[0]) / max(len(vals)-1, 1) if len(vals) > 1 else 0.0

    last_seq = scaled[-TIMESTEPS:].copy()
    future_seq = last_seq.copy()
    curr_year = int(df_region['year'].iloc[-1])
    curr_month = int(df_region['month'].iloc[-1])
    months = forecast_years * 12
    future_preds = []
    years_list = []
    months_list = []
    for _ in range(months):
        seq_in = future_seq[-TIMESTEPS:].reshape(1, TIMESTEPS, -1)
        p_scaled = z_model.predict(seq_in, verbose=0)
        p = targ_scaler.inverse_transform(p_scaled)[0][0]
        future_preds.append(max(p,0))
        # update next row
        next_row = future_seq[-1].copy()
        for feat in zhvi_feats:
            next_row[features.index(feat)] += avg_growth.get(feat, 0)
        curr_month += 1
        if curr_month > 12:
            curr_month = 1
            curr_year += 1
        next_row[features.index('year')] = curr_year
        next_row[features.index('month')] = curr_month
        next_row_scaled = feat_scaler.transform(next_row.reshape(1, -1))[0]
        future_seq = np.vstack([future_seq, next_row_scaled])
        years_list.append(curr_year); months_list.append(curr_month)
    future_df = pd.DataFrame({
        'regionname': df_region['regionname'].iloc[0],
        'year': years_list,
        'month': months_list,
        'Predicted_ZHVI': future_preds
    })
    return df_region, future_df

def generate_pdf_bytes():
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    from io import BytesIO

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # -------- TITLE --------
    story.append(Paragraph("<b>Realty AI — Full Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    # -------- CLASSIFICATION --------
    sat_class = st.session_state.sat_classification or "N/A"
    story.append(Paragraph(f"<b>Satellite Classification:</b> {sat_class}", styles["BodyText"]))
    story.append(Spacer(1, 6))

    # -------- ESTIMATED VALUE --------
    try:
        if st.session_state.reg_pred_df is not None:
            est_val = st.session_state.reg_pred_df["Predicted_SalePrice"].iloc[0]
            est_val_fmt = f"₹ {est_val:,.2f}"
        else:
            est_val_fmt = "N/A"
    except:
        est_val_fmt = "N/A"

    story.append(Paragraph(f"<b>Estimated Property Value:</b> {est_val_fmt}", styles["BodyText"]))
    story.append(Spacer(1, 12))

    # -------- UPLOADED IMAGE --------
    if "uploaded_image" in st.session_state and st.session_state.uploaded_image is not None:
        img = st.session_state.uploaded_image
        img_buf = BytesIO()
        img.save(img_buf, format="PNG")
        img_buf.seek(0)
        story.append(Paragraph("<b>Uploaded Image:</b>", styles["Heading3"]))
        story.append(Image(img_buf, width=4*inch, height=4*inch))
        story.append(Spacer(1, 12))

    # -------- PREDICTED MASK --------
    if "predicted_mask" in st.session_state and st.session_state.predicted_mask is not None:
        mask = st.session_state.predicted_mask
        mask_buf = BytesIO()
        mask.save(mask_buf, format="PNG")
        mask_buf.seek(0)
        story.append(Paragraph("<b>Predicted Segmentation Mask:</b>", styles["Heading3"]))
        story.append(Image(mask_buf, width=4*inch, height=4*inch))
        story.append(Spacer(1, 12))

    # -------- FORECAST SUMMARY --------
    if st.session_state.zillow_forecast_df is not None and not st.session_state.zillow_forecast_df.empty:
        region = st.session_state.zillow_forecast_df["regionname"].unique()[0]
        years = st.session_state.zillow_forecast_years or "N/A"

        story.append(Paragraph(f"<b>Forecast Region:</b> {region}", styles["BodyText"]))
        story.append(Paragraph(f"<b>Forecast Years:</b> {years}", styles["BodyText"]))
        story.append(Spacer(1, 8))

        # -------- CURRENT VALUE --------
        curr_val = None
        if st.session_state.zillow_pred_df is not None:
            try:
                curr_val = st.session_state.zillow_pred_df["Predicted_ZHVI"].iloc[-1]
            except:
                curr_val = None

        if curr_val is None:  
            try:
                df = st.session_state.zillow_inputs_df
                t = df[df["regionname"] == region].sort_values(["year", "month"])
                curr_val = t["zhvi_middletier"].iloc[-1]
            except:
                curr_val = None

        curr_fmt = f"₹ {curr_val:,.2f}" if curr_val else "N/A"
        story.append(Paragraph(f"<b>Current Property Value:</b> {curr_fmt}", styles["BodyText"]))
        story.append(Spacer(1, 6))

        # -------- FORECASTED VALUE --------
        fut_df = st.session_state.zillow_forecast_df
        fut_val = fut_df[fut_df["regionname"] == region]["Predicted_ZHVI"].iloc[-1]
        fut_fmt = f"₹ {fut_val:,.2f}"

        story.append(Paragraph(f"<b>Forecasted Future Value ({years} years):</b> {fut_fmt}", styles["BodyText"]))
        story.append(Spacer(1, 6))

    # -------- BUILD PDF --------
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# ---------------- STEP 1: SATELLITE ----------------
if st.session_state.step == 1:
    st.header("Step 1 — Satellite Image Classification 🛰️")
    st.markdown("<div class='small-muted'>Upload an image, then click <b>Run Segmentation</b> to see results.</div>", unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns([1, 2])
    with c1:
        img_file = st.file_uploader("Upload satellite image (jpg/png)", type=["jpg", "jpeg", "png"], key="sat_upload")
        if img_file:
            try:
                # save uploaded bytes but DO NOT preview yet
                image = Image.open(img_file).convert("RGB")
                st.session_state.sat_image_bytes = pil_to_bytes(image)
                st.session_state.sat_uploaded_flag = True
                st.success("Image uploaded — click *Run Segmentation* to process.")
            except Exception as e:
                st.error(f"Error reading image: {e}")
        else:
            st.info("Please upload an image to enable segmentation.")
        st.write("")
        run_seg = st.button("Run Segmentation", key="run_seg", help="Run the U-Net segmentation on uploaded image", )
        if run_seg:
            if not st.session_state.sat_uploaded_flag:
                st.error("Upload an image first.")
            else:
                with st.spinner("Running segmentation..."):
                    unet = load_unet()
                    if unet is None:
                        st.error("U-Net model not available. Check model path.")
                    else:
                        # run segmentation now
                        img = bytes_to_pil(st.session_state.sat_image_bytes)
                        resized = img.resize((256, 256))
                        arr = np.array(resized) / 255.0
                        arr = np.expand_dims(arr, axis=0)
                        try:
                            pred_mask = unet.predict(arr)[0]
                        except Exception as e:
                            st.error(f"Segmentation prediction failed: {e}")
                            pred_mask = None
                        if pred_mask is not None:
                            if pred_mask.ndim == 3:
                                pred_mask = pred_mask[:, :, 0]
                            mask = (pred_mask > 0.5).astype(np.uint8) * 255
                            mask_img = Image.fromarray(mask).resize(img.size)

                            # colored overlay
                            mnp = np.array(mask_img)
                            colored_np = np.zeros((mnp.shape[0], mnp.shape[1], 3), dtype=np.uint8)
                            colored_np[mnp == 255] = [0, 200, 120]   # green areas
                            colored_np[mnp != 255] = [20, 30, 40]    # background tint
                            colored = Image.fromarray(colored_np)
                            overlay = Image.blend(img.convert("RGBA"), colored.convert("RGBA"), alpha=0.35)

                            # store
                            st.session_state.sat_mask_bytes = pil_to_bytes(overlay)

                            # classification
                            mask_np = np.array(mask)
                            frac = np.count_nonzero(mask_np == 255) / mask_np.size
                            classification = "Residential" if frac > 0.5 else "Commercial"
                            st.session_state.sat_classification = classification
                            st.session_state.sat_pred_done = True
                            st.success("Segmentation complete.")
    with c2:
        # results area (only show after prediction)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if st.session_state.sat_pred_done:
            try:
                img = bytes_to_pil(st.session_state.sat_image_bytes)
                overlay = bytes_to_pil(st.session_state.sat_mask_bytes)
                colA, colB = st.columns(2)
                with colA:
                    st.image(img, caption="Original Image", use_container_width=True)
                with colB:
                    st.image(overlay, caption="Predicted Mask Overlay", use_container_width=True)
                st.markdown(f"<div style='margin-top:6px'><span class='muted-small'>Classification:</span> <span style='font-weight:800;color:var(--accent1)'> {st.session_state.sat_classification} </span></div>", unsafe_allow_html=True)
            except Exception:
                st.error("Failed to load image/mask preview.")
        else:
            st.markdown("<div style='padding:18px'><div class='muted-small'>Results will appear here after segmentation.</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("---")
    nav_l, nav_r = st.columns([1, 1])
    with nav_l:
        st.button("← Back", on_click=go_back)
    with nav_r:
        st.button("Next →", on_click=go_next)

# ---------------- STEP 2: REGRESSION PRICE PREDICTION (simplified) ----------------
elif st.session_state.step == 2:
    st.header("Step 2 — Housing Price Prediction 🏡")
    st.markdown("<div class='small-muted'>Upload a CSV or enter quick manual inputs. Only the estimated property value will be shown.</div>", unsafe_allow_html=True)

    st.write(f"**Detected area (from Step 1):** {st.session_state.sat_classification or 'N/A'}")
    left, right = st.columns([1, 1.1])
    with left:
        st.subheader("Upload CSV (batch)")
        reg_csv = st.file_uploader("Upload CSV with inputs", type=["csv"], key="reg_csv_step2")
        if reg_csv:
            try:
                df = pd.read_csv(reg_csv)
                st.success(f"Loaded {reg_csv.name} ({df.shape[0]} rows)")
                st.session_state.reg_inputs_df = df
                st.dataframe(df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    with right:
        st.subheader("Quick Manual Input")
        manual = {}
        manual["OverallQual"] = st.slider("Overall Quality (1-10)", 1, 10, 6, key="m_overallqual2")
        manual["GrLivArea"] = st.number_input("Living area (sqft)", min_value=100, max_value=10000, value=1500, step=50, key="m_grliv2")
        manual["GarageCars"] = st.number_input("Garage (car capacity)", min_value=0, max_value=4, value=1, step=1, key="m_garagecars2")
        manual["YearBuilt"] = st.number_input("Year Built", min_value=1800, max_value=2025, value=2005, key="m_yearbuilt2")
        manual["BedroomAbvGr"] = st.number_input("Bedrooms", min_value=0, max_value=10, value=3, key="m_beds2")
        manual["FullBath"] = st.number_input("Full Baths", min_value=0, max_value=5, value=2, key="m_fullbath2")
        # NOTE: Neighborhood input removed per your request
        st.session_state.reg_manual_inputs = manual

    st.write("---")
    run_col, out_col = st.columns([1, 2])
    with run_col:
        if st.button("Estimate Property Value", key="estimate_val"):
            lgb = load_lgb()
            if lgb is None:
                st.error("LightGBM model not available.")
            else:
                model_feature_names = getattr(lgb, "feature_name_", None)
                try:
                    if st.session_state.reg_inputs_df is not None:
                        df_in = st.session_state.reg_inputs_df.copy()
                        cat_cols = df_in.select_dtypes(include='object').columns.tolist()
                        df_enc = pd.get_dummies(df_in, columns=cat_cols, drop_first=True)
                        if model_feature_names is not None:
                            df_enc = df_enc.reindex(columns=model_feature_names, fill_value=0)
                        else:
                            df_enc = df_enc.reindex(columns=REGRESSION_FEATURES, fill_value=0)
                        preds = lgb.predict(df_enc)
                        df_display = df_in.copy()
                        df_display['Predicted_SalePrice'] = preds
                        st.session_state.reg_pred_df = df_display
                        st.success("Batch predictions generated.")
                        st.dataframe(df_display.head(), use_container_width=True)
                    else:
                        sample = st.session_state.reg_manual_inputs
                        sample_df = pd.DataFrame([sample])
                        sample_enc = pd.get_dummies(sample_df, drop_first=True)
                        if model_feature_names is not None:
                            sample_enc = sample_enc.reindex(columns=model_feature_names, fill_value=0)
                        else:
                            sample_enc = sample_enc.reindex(columns=REGRESSION_FEATURES, fill_value=0)
                        pred = lgb.predict(sample_enc)[0]
                        sample_display = sample_df.copy()
                        sample_display['Predicted_SalePrice'] = pred
                        st.session_state.reg_pred_df = sample_display
                        st.success(f"Estimated Property Value: ₹ {pred:,.2f}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    with out_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if st.session_state.reg_pred_df is not None:
            try:
                # show single-sample or first of batch
                pred_val = st.session_state.reg_pred_df['Predicted_SalePrice'].iloc[0]
                st.markdown("<div style='display:flex;gap:12px;align-items:center'>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-box'><div class='muted-small'>Estimated Value</div><div class='kpi'>₹ {pred_val:,.2f}</div></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception:
                st.info("Prediction available but could not render value.")
        else:
            st.markdown("<div class='muted-small'>Estimated property value will appear here after running estimation.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("---")
    nav_l, nav_r = st.columns([1, 1])
    with nav_l:
        st.button("← Back", on_click=go_back)
    with nav_r:
        st.button("Next →", on_click=go_next)

# ---------------- STEP 3: ZILLOW FORECASTING ----------------
elif st.session_state.step == 3:
    st.header("Step 3 — Zillow Forecasting 📈")
    st.markdown("<div class='small-muted'>Upload a Zillow CSV (preferred) or pick a manual region. Select forecast years and run.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.1])
    with c1:
        st.subheader("Upload Zillow CSV (optional)")
        zfile = st.file_uploader("Zillow CSV (columns: regionname, year, month, zhvi_...)", type=["csv"], key="z_csv_step3")
        if zfile:
            try:
                zdf = pd.read_csv(zfile)
                st.success(f"Loaded {zfile.name} ({zdf.shape[0]} rows)")
                st.session_state.zillow_inputs_df = zdf
                st.dataframe(zdf.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    with c2:
        st.subheader("Region & Horizon")
        if st.session_state.zillow_inputs_df is not None:
            regions = sorted(st.session_state.zillow_inputs_df['regionname'].unique().tolist())
            sel_region = st.selectbox("Choose region to forecast (from CSV)", regions, key="sel_region_csv")
            st.session_state.zillow_selected_region = sel_region
        else:
            sel_region = st.text_input("Region name (manual)", value="San Francisco", key="man_region_step3")
            st.session_state.zillow_selected_region = sel_region
        yrs = st.number_input("Forecast years (1-10)", min_value=1, max_value=10, value=5, key="man_years_step3")
        st.session_state.zillow_forecast_years = int(yrs)

    st.write("---")
    run_col, out_col = st.columns([1, 2])
    with run_col:
        if st.button("Generate Forecast", key="gen_forecast"):
            z_model, feat_scaler, targ_scaler = load_zillow_model_and_scalers()
            if z_model is None or feat_scaler is None or targ_scaler is None:
                # If models missing, fallback to heuristic
                st.warning("Zillow model or scalers not available — creating heuristic forecast (4% annual).")
                use_model = False
            else:
                use_model = True

            features = [
                'zhvi_middletier', 'zhvi_singlefamilyresidence', 'zhvi_toptier',
                'zhvipersqft_allhomes', 'zhvi_bottomtier', 'zhvi_4bedroom',
                'zhvi_3bedroom', 'zhvi_5bedroomormore', 'zhvi_2bedroom', 'zhvi_condocoop',
                'zri_allhomes', 'zri_allhomesplusmultifamily', 'zri_singlefamilyresidencerental',
                'pricetorentratio_allhomes', 'zripersqft_allhomes', 'year',
                'month', 'pctofhomesdecreasinginvalues_allhomes', 'pctofhomesincreasinginvalues_allhomes',
                'inventoryraw_allhomes', 'inventoryseasonallyadjusted_allhomes'
            ]

            if use_model and st.session_state.zillow_inputs_df is not None:
                df = st.session_state.zillow_inputs_df.copy()
                missing = [f for f in features if f not in df.columns]
                if missing:
                    st.error(f"CSV missing required features for model forecasting: {missing}")
                    use_model = False
                else:
                    sel = st.session_state.zillow_selected_region
                    df_region = df[df['regionname'] == sel].reset_index(drop=True)
                    if df_region.shape[0] < 13:
                        st.warning("Not enough historical rows for LSTM (need >12). Falling back to heuristic.")
                        use_model = False
                    else:
                        with st.spinner("Running LSTM forecast..."):
                            hist_df, fut_df = build_forecast_for_region(z_model, feat_scaler, targ_scaler, df_region, features, st.session_state.zillow_forecast_years)
                            st.session_state.zillow_pred_df = hist_df
                            st.session_state.zillow_forecast_df = fut_df
                            st.success("Forecast generated using model.")
            if not use_model:
                # manual heuristic forecast (4% annual)
                region = st.session_state.zillow_selected_region or "ManualRegion"
                years = st.session_state.zillow_forecast_years or 5
                st.session_state.zillow_selected_region = region
                months = years * 12
                today = datetime.date.today()
                cy = today.year; cm = today.month
                months_list = []; years_list = []; values = []
                # choose base from recent historical if available
                base = 100000.0
                if st.session_state.zillow_inputs_df is not None:
                    tmp = st.session_state.zillow_inputs_df
                    tmp_reg = tmp[tmp['regionname'] == region]
                    if not tmp_reg.empty and 'zhvi_middletier' in tmp_reg.columns:
                        base = tmp_reg.sort_values(['year','month'])['zhvi_middletier'].iloc[-1]
                for i in range(months):
                    cm += 1
                    if cm > 12:
                        cm = 1; cy += 1
                    years_list.append(cy); months_list.append(cm)
                    base = base * (1 + 0.04/12)
                    values.append(base)
                futdf = pd.DataFrame({
                    'regionname': region,
                    'year': years_list, 'month': months_list, 'Predicted_ZHVI': values
                })
                st.session_state.zillow_forecast_df = futdf
                st.success("Heuristic forecast created.")

    with out_col:
        st.subheader("Forecast Output")
        # Display current & future & growth + graph
        sel_region = st.session_state.zillow_selected_region
        curr_val = None
        if st.session_state.zillow_pred_df is not None and not st.session_state.zillow_pred_df.empty:
            # historical predicted series exists
            if sel_region in st.session_state.zillow_pred_df['regionname'].unique():
                hist = st.session_state.zillow_pred_df[st.session_state.zillow_pred_df['regionname'] == sel_region].dropna(subset=['Predicted_ZHVI'])
                if not hist.empty:
                    curr_val = hist['Predicted_ZHVI'].iloc[-1]
                    st.markdown("<div style='display:flex;gap:12px'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-box'><div class='muted-small'>Current Value</div><div class='kpi'> {curr_val:,.2f} </div></div>", unsafe_allow_html=True)
        # fallback current from raw inputs
        if curr_val is None and st.session_state.zillow_inputs_df is not None:
            tmp = st.session_state.zillow_inputs_df
            if 'regionname' in tmp.columns and 'zhvi_middletier' in tmp.columns:
                t = tmp[tmp['regionname'] == sel_region].sort_values(['year','month'])
                if not t.empty:
                    curr_val = t['zhvi_middletier'].iloc[-1]
                    st.markdown(f"<div class='metric-box'><div class='muted-small'>Current Value</div><div class='kpi'> {curr_val:,.2f} </div></div>", unsafe_allow_html=True)
        if curr_val is None:
            st.markdown("<div class='metric-box'><div class='muted-small'>Current Value</div><div class='kpi'> N/A </div></div>", unsafe_allow_html=True)

        # future value
        fut_val = None
        if st.session_state.zillow_forecast_df is not None and not st.session_state.zillow_forecast_df.empty:
            fut_df = st.session_state.zillow_forecast_df
            if sel_region in fut_df['regionname'].unique():
                fut_region = fut_df[fut_df['regionname'] == sel_region]
                if not fut_region.empty:
                    fut_val = fut_region['Predicted_ZHVI'].iloc[-1]
                    st.markdown(f"<div style='display:inline-block;margin-left:12px' class='metric-box'><div class='muted-small'>Forecast ({st.session_state.zillow_forecast_years}y)</div><div class='kpi'> {fut_val:,.2f} </div></div>", unsafe_allow_html=True)

        # growth percent & CAGR
        if (curr_val is not None) and (fut_val is not None) and curr_val > 0:
            pct = (fut_val - curr_val) / curr_val * 100
            years = st.session_state.zillow_forecast_years or 1
            cagr = (fut_val / curr_val) ** (1.0 / years) - 1
            st.markdown("<div style='margin-top:12px'>", unsafe_allow_html=True)
            st.markdown(f"<div class='muted-small'>Growth over horizon: <b style='color:var(--accent1)'>{pct:.2f}%</b> — CAGR: <b style='color:var(--accent2)'>{cagr*100:.2f}% / yr</b></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='muted-small' style='margin-top:12px'>Growth: N/A</div>", unsafe_allow_html=True)

    st.write("---")
    nav_l, nav_r = st.columns([1,1])
    with nav_l:
        st.button("← Back", on_click=go_back)
    with nav_r:
        st.button("Next →", on_click=go_next)

# ---------------- STEP 4: SUMMARY (PDF only) ----------------
elif st.session_state.step == 4:
    st.header("Final Summary & Report 🧾")
    st.markdown("<div class='small-muted'>Download a PDF summary of the current session (images, estimate, forecast).</div>", unsafe_allow_html=True)

    st.write("**Satellite classification:**", st.session_state.sat_classification or "N/A")
    if st.session_state.reg_pred_df is not None:
        try:
            st.write("**Estimated Property Value:**", f"₹ {st.session_state.reg_pred_df['Predicted_SalePrice'].iloc[0]:,.2f}")
        except Exception:
            st.write("**Estimated Property Value:** N/A")
    else:
        st.write("**Estimated Property Value:** N/A")
    if st.session_state.zillow_forecast_df is not None and not st.session_state.zillow_forecast_df.empty:
        region = st.session_state.zillow_forecast_df['regionname'].unique()[0]
        st.write("**Forecast Region:**", region)
        st.write("**Forecast Years:**", st.session_state.zillow_forecast_years or "N/A")

    st.write("---")
    if st.button("Generate PDF Report"):
        with st.spinner("Composing PDF..."):
            try:
                pdf_bytes = generate_pdf_bytes()
                st.success("PDF report ready.")
                st.download_button("📥 Download Final PDF Report", data=pdf_bytes, file_name="realty_full_report.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"Failed to generate PDF: {e}")

    st.write("---")
    st.button("← Back", on_click=lambda: setattr(st.session_state, "step", 3))
    st.markdown("Flow complete. Refresh the page to start over or clear session state from the menu.")
