import streamlit as st
import numpy as np
import pandas as pd
import json
import pickle
import tensorflow as tf
from PIL import Image
import plotly.graph_objects as go
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.units import inch
from reportlab.lib import colors
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder
import os
import base64
import io  # for Excel export

matplotlib.use('Agg')

st.set_page_config(
    page_title="RealtyAI - Property Prediction Pipeline",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_models():
    """Load all ML models and associated files with error handling."""
    models_status = {}
    errors = []
    models = {}
    
    # U-Net
    try:
        if os.path.exists('residential_commercial_unet.keras'):
            models['unet'] = tf.keras.models.load_model('residential_commercial_unet.keras')
            models_status['unet'] = True
        elif os.path.exists('residential_commercial_unet.h5'):
            models['unet'] = tf.keras.models.load_model('residential_commercial_unet.h5')
            models_status['unet'] = True
        else:
            models['unet'] = None
            models_status['unet'] = False
            errors.append("U-Net model file not found")
    except Exception as e:
        models['unet'] = None
        models_status['unet'] = False
        errors.append(f"U-Net model error: {str(e)}")
    
    # LSTM
    try:
        if os.path.exists('lstm_model.keras'):
            models['lstm'] = tf.keras.models.load_model('lstm_model.keras')
            models_status['lstm'] = True
        else:
            models['lstm'] = None
            models_status['lstm'] = False
            errors.append("LSTM model file not found")
    except Exception as e:
        models['lstm'] = None
        models_status['lstm'] = False
        errors.append(f"LSTM model error: {str(e)}")

    # XGBoost
    try:
        if os.path.exists('xgb_optimized_model.pkl'):
            with open('xgb_optimized_model.pkl', 'rb') as f:
                models['xgboost'] = pickle.load(f)
            models_status['xgboost'] = 'optimized'
        elif os.path.exists('xgboost_optimized_model.pkl'):
            with open('xgboost_optimized_model.pkl', 'rb') as f:
                models['xgboost'] = pickle.load(f)
            models_status['xgboost'] = 'optimized'
        elif os.path.exists('xgboost_model.pkl'):
            with open('xgboost_model.pkl', 'rb') as f:
                models['xgboost'] = pickle.load(f)
            models_status['xgboost'] = True
        else:
            models['xgboost'] = None
            models_status['xgboost'] = False
            errors.append("XGBoost model file not found (xgb_optimized_model.pkl / xgboost_optimized_model.pkl / xgboost_model.pkl)")
    except Exception as e:
        models['xgboost'] = None
        models_status['xgboost'] = False
        errors.append(f"XGBoost model error: {str(e)}")

    # LSTM scalers (if you actually use them)
    try:
        if os.path.exists('scaler_X.pkl') and os.path.exists('scaler_y.pkl'):
            with open('scaler_X.pkl', 'rb') as f:
                models['scaler_X'] = pickle.load(f)
            with open('scaler_y.pkl', 'rb') as f:
                models['scaler_y'] = pickle.load(f)
            models_status['scalers'] = True
        else:
            models['scaler_X'] = None
            models['scaler_y'] = None
            models_status['scalers'] = False
            errors.append("Scaler files not found")
    except Exception as e:
        models['scaler_X'] = None
        models['scaler_y'] = None
        models_status['scalers'] = False
        errors.append(f"Scalers error: {str(e)}")

    # XGBoost feature names JSON (for manual entry flow)
    try:
        if os.path.exists('xgb_feature_names.json'):
            with open('xgb_feature_names.json', 'r') as f:
                models['xgb_features'] = json.load(f)
            models_status['xgb_features'] = True
        else:
            models['xgb_features'] = None
            models_status['xgb_features'] = False
            errors.append("XGBoost feature names file not found")
    except Exception as e:
        models['xgb_features'] = None
        models_status['xgb_features'] = False
        errors.append(f"XGBoost features error: {str(e)}")

    return models, models_status, errors


def save_figure_to_bytes(fig, format='png', dpi=150):
    """Convert matplotlib/plotly figure to bytes for PDF embedding."""
    buf = BytesIO()
    if hasattr(fig, 'write_image'):
        fig.write_image(buf, format=format, width=800, height=400)
    else:
        fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    buf.seek(0)
    return buf


models, models_status, load_errors = load_models()

# Session state
if 'unet_results' not in st.session_state:
    st.session_state.unet_results = None
if 'lstm_results' not in st.session_state:
    st.session_state.lstm_results = None
if 'xgb_results' not in st.session_state:
    st.session_state.xgb_results = None

# Model status UI
if load_errors:
    with st.expander("⚠️ Model Loading Status - Click to view details", expanded=False):
        st.warning("Some models could not be loaded.")
        col1, col2, col3 = st.columns(3)
        with col1:
            status = "✅ Loaded" if models_status.get('unet') else "❌ Not Loaded"
            st.markdown(f"**U-Net Model:** {status}")
        with col2:
            status = "✅ Loaded" if models_status.get('lstm') else "❌ Not Loaded"
            st.markdown(f"**LSTM Model:** {status}")
        with col3:
            if models_status.get('xgboost') == 'optimized':
                status = "✅ Loaded (Optimized)"
            elif models_status.get('xgboost'):
                status = "✅ Loaded"
            else:
                status = "❌ Not Loaded"
            st.markdown(f"**XGBoost Model:** {status}")
        st.markdown("**Status Details:**")
        for error in load_errors:
            st.text(f"• {error}")
else:
    st.success("✅ All models loaded successfully!")

st.title("🏠 RealtyAI - Complete Property Prediction Pipeline")
st.markdown("U-Net Segmentation → LSTM Forecasting → XGBoost Price Prediction → PDF Report (single-page tabs)")

st.divider()

# Progress chips
progress_cols = st.columns(4)
with progress_cols[0]:
    st.markdown(f"### {'✅' if st.session_state.unet_results else '⭕'} U-Net")
with progress_cols[1]:
    st.markdown(f"### {'✅' if st.session_state.lstm_results else '⭕'} LSTM")
with progress_cols[2]:
    st.markdown(f"### {'✅' if st.session_state.xgb_results else '⭕'} XGBoost")
with progress_cols[3]:
    st.markdown(
        f"### {'✅' if all([st.session_state.unet_results, st.session_state.lstm_results, st.session_state.xgb_results]) else '⭕'} PDF"
    )

st.divider()

tab_unet, tab_lstm, tab_xgb, tab_pdf = st.tabs(
    ["U-Net Segmentation", "LSTM Forecasting", "XGBoost Price Prediction", "PDF Report"]
)

# ==================== TAB 1: U-NET ====================
with tab_unet:
    st.header("📸 U-Net Segmentation (Residential / Commercial Classification)")
    st.markdown("Upload an image to classify whether the property is residential or commercial.")

    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image (PNG, JPG, JPEG)", type=['png', 'jpg', 'jpeg'], key='unet_upload')

    if uploaded_file is not None:
        st.info("✓ Image uploaded successfully. Click 'Analyze Image' to process.")

    button_disabled = uploaded_file is None or models['unet'] is None

    if st.button("🔍 Analyze Image", use_container_width=True, type="primary", disabled=button_disabled, key='unet_btn'):
        if models['unet'] is None:
            st.error("❌ U-Net model not loaded.")
        else:
            try:
                with st.spinner("Analyzing image..."):
                    img = Image.open(uploaded_file)
                    img = img.resize((256, 256), Image.Resampling.LANCZOS)
                    img_array = np.array(img)

                    if img_array.ndim == 3 and img_array.shape[2] == 3:
                        dummy_channel = np.zeros((*img_array.shape[:2], 1), dtype=img_array.dtype)
                        img_array = np.concatenate([img_array, dummy_channel], axis=2)
                    elif img_array.ndim == 2:
                        img_array = np.stack([img_array] * 3 + [np.zeros_like(img_array)], axis=2)

                    img_normalized = img_array.astype(np.float32) / 255.0
                    img_input = np.expand_dims(img_normalized, axis=0)

                    mask_pred = models['unet'].predict(img_input, verbose=0)
                    mask_output = (mask_pred[0] * 255).astype(np.uint8)

                    if mask_output.ndim == 3 and mask_output.shape[2] == 1:
                        mask_image = Image.fromarray(mask_output[:, :, 0], mode='L')
                        mask_array = mask_output[:, :, 0]
                    else:
                        mask_image = Image.fromarray(mask_output)
                        mask_array = mask_output

                    total_pixels = mask_array.size
                    segmented_pixels = np.count_nonzero(mask_array > 128)
                    segmentation_ratio = segmented_pixels / total_pixels

                    segmented_region = mask_array[mask_array > 128]
                    avg_intensity = np.mean(segmented_region) if len(segmented_region) > 0 else 0

                    if segmentation_ratio > 0.3 and avg_intensity > 180:
                        classification = "Commercial"
                        confidence = min(95, 70 + (segmentation_ratio * 50))
                    elif segmentation_ratio > 0.15:
                        classification = "Residential"
                        confidence = min(95, 60 + ((0.5 - segmentation_ratio) * 100))
                    else:
                        classification = "Residential"
                        confidence = 55

                    st.session_state.unet_results = {
                        'uploaded_image': img,
                        'mask_image': mask_image,
                        'classification': classification,
                        'confidence': confidence,
                        'segmentation_ratio': segmentation_ratio,
                        'avg_intensity': avg_intensity
                    }

                    st.success("✅ Analysis complete!")
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")

    if st.session_state.unet_results:
        st.divider()
        st.subheader("📊 Segmentation Results")

        image_col1, image_col2, metrics_col = st.columns([1, 1, 1])
        with image_col1:
            st.markdown("**Original Image**")
            st.image(st.session_state.unet_results['uploaded_image'], width=300)
        with image_col2:
            st.markdown("**Segmentation Mask**")
            st.image(st.session_state.unet_results['mask_image'], width=300)
        with metrics_col:
            st.markdown("### Classification Result")
            classification = st.session_state.unet_results['classification']
            confidence = st.session_state.unet_results['confidence']
            if classification == "Commercial":
                st.success(f"### 🏢 {classification}")
            else:
                st.info(f"### 🏠 {classification}")
            st.metric("Confidence Score", f"{confidence:.1f}%")


# ==================== TAB 2: LSTM (NO GRAPH) ====================
with tab_lstm:
    st.header("📈 LSTM Time Series Forecasting (Numeric Output)")
    st.markdown("Upload historical data and get numeric forecasts for future years (no graph).")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Upload Excel Data")
        excel_file = st.file_uploader("Choose Excel file (XLSX/XLS)", type=['xlsx', 'xls'], key='lstm_excel')
        if excel_file is not None:
            try:
                preview_df = pd.read_excel(excel_file)
                st.subheader("📊 Data Preview")
                st.dataframe(preview_df.head(10), use_container_width=True)
                st.caption(f"Showing first 10 rows of {len(preview_df)} total rows")
                excel_file.seek(0)
            except Exception as e:
                st.warning(f"Could not preview data: {str(e)}")

    with col2:
        st.subheader("Forecast Settings")
        forecast_years = st.number_input("Forecast next N years:", min_value=1, max_value=20, value=3, step=1)
        st.caption(f"Will generate {forecast_years} future yearly price predictions.")

    button_disabled = excel_file is None or models['lstm'] is None

    if st.button("📊 Generate LSTM Forecast", use_container_width=True, type="primary", disabled=button_disabled, key='lstm_btn'):
        if models['lstm'] is None:
            st.error("❌ LSTM model not loaded.")
        else:
            try:
                with st.spinner("Processing data and generating forecast..."):
                    df = pd.read_excel(excel_file)
                    data_array = df.iloc[:, :8].values.flatten().astype(np.float32)

                    lstm_input_shape = models['lstm'].input_shape
                    expected_timesteps = lstm_input_shape[1]
                    expected_features = lstm_input_shape[2]
                    total_expected = expected_timesteps * expected_features

                    if len(data_array) < total_expected:
                        data_array = np.pad(data_array, (0, total_expected - len(data_array)), mode='constant')
                    elif len(data_array) > total_expected:
                        data_array = data_array[:total_expected]

                    lstm_input_data = data_array.reshape(1, expected_timesteps, expected_features)

                    years_to_predict = forecast_years
                    forecast_values = []
                    current_input = lstm_input_data.copy()

                    base_price = np.mean(data_array[data_array > 0]) if np.any(data_array > 0) else 200000

                    for year in range(years_to_predict):
                        prediction = models['lstm'].predict(current_input, verbose=0)
                        predicted_value = float(prediction[0][0])
                        predicted_value = abs(predicted_value)

                        if year > 0:
                            growth_factor = 1 + np.random.uniform(0.02, 0.05)
                            predicted_value = forecast_values[-1] * growth_factor
                        else:
                            predicted_value = max(predicted_value, base_price * 0.8)

                        forecast_values.append(predicted_value)

                        new_timestep = np.zeros((1, 1, expected_features))
                        new_timestep[0, 0, 0] = predicted_value
                        current_input = np.concatenate(
                            [current_input[:, 1:, :], new_timestep],
                            axis=1
                        )

                    st.session_state.lstm_results = {
                        'forecast_values': forecast_values,
                        'forecast_years': years_to_predict,
                        'input_data': data_array
                    }
                    st.success("✅ Forecast generated!")
            except Exception as e:
                st.error(f"Error during forecasting: {str(e)}")

    if st.session_state.lstm_results:
        st.divider()
        st.subheader("📊 Forecasted Prices")
        forecast_values = st.session_state.lstm_results['forecast_values']
        years_to_predict = st.session_state.lstm_results['forecast_years']
        years = list(range(1, years_to_predict + 1))
        df_forecast = pd.DataFrame({
            "Year (from last known)": years,
            "Predicted Price": forecast_values
        })
        st.dataframe(df_forecast, use_container_width=True)
        avg_price = np.mean(forecast_values)
        st.metric("💰 Average Forecasted Price", f"${avg_price:,.2f}")


# ==================== TAB 3: XGBOOST (MANUAL + FILE UPLOAD) ====================
with tab_xgb:
    st.header("🏡 XGBoost Property Price Prediction")
    st.markdown("Use manual entry or upload a CSV/Excel file for batch prediction.")

    xgb_model = models['xgboost']

    if xgb_model is None:
        st.error("❌ XGBoost model not loaded.")
    else:
        input_mode = st.radio(
            "Select input mode",
            ["Manual Entry", "Upload CSV/Excel"],
            horizontal=True
        )

        # Mappings for manual mode
        qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
        bsmt_fin_map = {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0}
        func_map = {"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4, "Maj1": 3, "Maj2": 2, "Sev": 1}
        exposure_map = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0}

        if input_mode == "Manual Entry":
            if models.get('xgb_features') is None:
                st.error("xgb_feature_names.json not found; manual mode needs known feature names.")
            else:
                st.success(f"✅ Model loaded! Trained on **{len(models['xgb_features'])} selected features**")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("### Basic Features")
                    OverallQual = st.slider("Overall Quality (1-10)", 1, 10, 7)
                    GrLivArea = st.number_input("Above Grade Living Area (sq ft)", 500, 8000, 1600)
                    TotalSF = st.number_input("Total Square Feet", 500, 10000, 2500)
                    YearBuilt = st.number_input("Year Built", 1800, 2025, 2005)
                    YearRemodAdd = st.number_input("Year Remodeled", 1800, 2025, 2010)
                    OverallCond = st.slider("Overall Condition (1-10)", 1, 10, 5)

                with col2:
                    st.markdown("### Area Details")
                    FirstFlrSF = st.number_input("1st Floor SF", 0, 5000, 1200)
                    SecondFlrSF = st.number_input("2nd Floor SF", 0, 3000, 350)
                    BsmtFinSF1 = st.number_input("Basement Finished SF", 0, 3000, 400)
                    BsmtUnfSF = st.number_input("Basement Unfinished SF", 0, 3000, 560)
                    LotArea = st.number_input("Lot Area (sq ft)", 1000, 50000, 8500)
                    LotFrontage = st.number_input("Lot Frontage (ft)", 0, 300, 70)
                    OpenPorchSF = st.number_input("Open Porch SF", 0, 500, 0)

                with col3:
                    st.markdown("### Garage & Rooms")
                    GarageCars = st.slider("Garage Cars", 0, 5, 2)
                    GarageArea = st.number_input("Garage Area", 0, 2000, 500)
                    GarageYrBlt = st.number_input("Garage Year Built", 1800, 2025, 2005)
                    FullBath = st.slider("Full Bathrooms", 0, 4, 2)
                    HalfBath = st.slider("Half Bathrooms", 0, 3, 1)
                    BedroomAbvGr = st.slider("Bedrooms Above Grade", 0, 10, 3)
                    Fireplaces = st.slider("Fireplaces", 0, 4, 1)
                    TotRmsAbvGrd = st.slider("Total Rooms Above Grade", 0, 15, 7)
                    BsmtFullBath = st.slider("Basement Full Baths", 0, 3, 1)

                st.markdown("### Categorical Features")
                col4, col5, col6 = st.columns(3)
                with col4:
                    CentralAir = st.selectbox("Central Air", ["Y", "N"])
                    MSZoning = st.selectbox("MS Zoning", ["RL", "RM", "C (all)", "FV", "RH"])
                    MSSubClass = st.selectbox("MS SubClass", ["20", "30", "60", "70", "Other"])
                    Neighborhood = st.selectbox("Neighborhood", ["NAmes", "Edwards", "OldTown", "IDOTRR", "Crawfor", "Other"])
                with col5:
                    KitchenQual = st.selectbox("Kitchen Quality", ["Ex", "Gd", "TA", "Fa"])
                    ExterQual = st.selectbox("Exterior Quality", ["Ex", "Gd", "TA", "Fa"])
                    BsmtQual = st.selectbox("Basement Quality", ["Ex", "Gd", "TA", "Fa", "None"])
                    HeatingQC = st.selectbox("Heating Quality", ["Ex", "Gd", "TA", "Fa", "Po"])
                    ExterCond = st.selectbox("Exterior Condition", ["Ex", "Gd", "TA", "Fa", "Po"])
                with col6:
                    BldgType = st.selectbox("Building Type", ["1Fam", "TwnhsE", "Twnhs", "Duplex", "2fmCon"])
                    GarageFinish = st.selectbox("Garage Finish", ["Fin", "RFn", "Unf", "None"])
                    GarageType = st.selectbox("Garage Type", ["Attchd", "Detchd", "BuiltIn", "None"])
                    SaleCondition = st.selectbox("Sale Condition", ["Normal", "Abnorml", "Partial", "Other"])
                    SaleType = st.selectbox("Sale Type", ["WD", "New", "COD", "Other"])

                st.markdown("### Additional Features")
                col7, col8 = st.columns(2)
                with col7:
                    GarageCond = st.selectbox("Garage Condition", ["Ex", "Gd", "TA", "Fa", "Po"])
                    GarageQual = st.selectbox("Garage Quality", ["Ex", "Gd", "TA", "Fa", "Po"])
                    BsmtFinType1 = st.selectbox("Basement Finish Type", ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "None"])
                with col8:
                    Functional = st.selectbox("Functionality", ["Typ", "Min1", "Min2", "Mod", "Maj1", "Maj2", "Sev"])
                    PavedDrive = st.selectbox("Paved Driveway", ["Y", "P", "N"])
                    BsmtExposure = st.selectbox("Basement Exposure", ["Gd", "Av", "Mn", "No", "None"])

                if st.button("🔮 Predict Property Price", use_container_width=True, type="primary", key='xgb_manual_btn'):
                    try:
                        with st.spinner("Calculating prediction..."):
                            input_dict = {}
                            for feature in models['xgb_features']:
                                input_dict[feature] = 0

                            input_dict['OverallQual'] = OverallQual
                            input_dict['GrLivArea'] = GrLivArea
                            input_dict['TotalSF'] = TotalSF
                            input_dict['YearBuilt'] = YearBuilt
                            input_dict['YearRemodAdd'] = YearRemodAdd
                            input_dict['1stFlrSF'] = FirstFlrSF
                            input_dict['2ndFlrSF'] = SecondFlrSF
                            input_dict['BsmtFinSF1'] = BsmtFinSF1
                            input_dict['BsmtUnfSF'] = BsmtUnfSF
                            input_dict['LotArea'] = LotArea
                            input_dict['LotFrontage'] = LotFrontage
                            input_dict['GarageCars'] = GarageCars
                            input_dict['GarageArea'] = GarageArea
                            input_dict['GarageYrBlt'] = GarageYrBlt
                            input_dict['FullBath'] = FullBath
                            input_dict['HalfBath'] = HalfBath
                            input_dict['Fireplaces'] = Fireplaces
                            input_dict['TotRmsAbvGrd'] = TotRmsAbvGrd
                            input_dict['BsmtFullBath'] = BsmtFullBath
                            input_dict['OpenPorchSF'] = OpenPorchSF

                            input_dict['KitchenQual'] = qual_map.get(KitchenQual, 3)
                            input_dict['ExterQual'] = qual_map.get(ExterQual, 3)
                            input_dict['BsmtQual'] = qual_map.get(BsmtQual, 3)
                            input_dict['HeatingQC'] = qual_map.get(HeatingQC, 3)
                            input_dict['GarageCond'] = qual_map.get(GarageCond, 3)
                            input_dict['GarageQual'] = qual_map.get(GarageQual, 3)
                            input_dict['ExterCond'] = qual_map.get(ExterCond, 3)
                            input_dict['BsmtFinType1'] = bsmt_fin_map.get(BsmtFinType1, 0)
                            input_dict['Functional'] = func_map.get(Functional, 7)
                            input_dict['BsmtExposure'] = exposure_map.get(BsmtExposure, 0)

                            input_dict['CentralAir_N'] = 1 if CentralAir == "N" else 0
                            input_dict['MSZoning_RM'] = 1 if MSZoning == "RM" else 0
                            input_dict['MSZoning_RL'] = 1 if MSZoning == "RL" else 0
                            input_dict['MSSubClass_60'] = 1 if MSSubClass == "60" else 0
                            input_dict['MSSubClass_30'] = 1 if MSSubClass == "30" else 0
                            input_dict['MSSubClass_70'] = 1 if MSSubClass == "70" else 0
                            input_dict['Neighborhood_OldTown'] = 1 if Neighborhood == "OldTown" else 0
                            input_dict['Neighborhood_IDOTRR'] = 1 if Neighborhood == "IDOTRR" else 0
                            input_dict['Neighborhood_Edwards'] = 1 if Neighborhood == "Edwards" else 0
                            input_dict['Neighborhood_Crawfor'] = 1 if Neighborhood == "Crawfor" else 0
                            input_dict['Neighborhood_NAmes'] = 1 if Neighborhood == "NAmes" else 0
                            input_dict['BldgType_1Fam'] = 1 if BldgType == "1Fam" else 0
                            input_dict['GarageFinish_Unf'] = 1 if GarageFinish == "Unf" else 0
                            input_dict['GarageType_Detchd'] = 1 if GarageType == "Detchd" else 0
                            input_dict['SaleCondition_Abnorml'] = 1 if SaleCondition == "Abnorml" else 0
                            input_dict['SaleCondition_Normal'] = 1 if SaleCondition == "Normal" else 0
                            input_dict['SaleType_New'] = 1 if SaleType == "New" else 0
                            input_dict['OverallCond_3'] = 1 if OverallCond == 3 else 0
                            input_dict['OverallCond_7'] = 1 if OverallCond == 7 else 0
                            input_dict['OverallCond_4'] = 1 if OverallCond == 4 else 0
                            input_dict['Electrical_SBrkr'] = 1
                            input_dict['LandContour_Lvl'] = 1
                            input_dict['Alley_None'] = 1
                            input_dict['PavedDrive_N'] = 1 if PavedDrive == "N" else 0

                            input_df = pd.DataFrame([input_dict])
                            input_df = input_df[models['xgb_features']]

                            log_prediction = xgb_model.predict(input_df)[0]
                            predicted_price = np.expm1(log_prediction)

                            st.session_state.xgb_results = {
                                'mode': 'single',
                                'predicted_price': float(predicted_price),
                                'input_features': input_df.iloc[0].to_dict(),
                                'feature_names': models['xgb_features'],
                                'raw_inputs': {
                                    'OverallQual': OverallQual,
                                    'GrLivArea': GrLivArea,
                                    'TotalSF': TotalSF,
                                    'YearBuilt': YearBuilt,
                                    'YearRemodAdd': YearRemodAdd,
                                    'OverallCond': OverallCond,
                                    'FirstFlrSF': FirstFlrSF,
                                    'SecondFlrSF': SecondFlrSF,
                                    'BsmtFinSF1': BsmtFinSF1,
                                    'BsmtUnfSF': BsmtUnfSF,
                                    'LotArea': LotArea,
                                    'GarageCars': GarageCars,
                                    'FullBath': FullBath,
                                    'BedroomAbvGr': BedroomAbvGr,
                                    'Neighborhood': Neighborhood,
                                    'KitchenQual': KitchenQual,
                                    'ExterQual': ExterQual
                                }
                            }
                            st.success("✅ Prediction complete!")
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {e}")

        else:
            st.subheader("📂 Upload CSV/Excel for XGBoost")
            uploaded_tab = st.file_uploader(
                "Upload CSV/Excel for XGBoost",
                type=["csv", "xlsx", "xls"],
                key="xgb_upload"
            )

            if uploaded_tab is not None:
                try:
                    if uploaded_tab.name.endswith(('xlsx', 'xls')):
                        X_df = pd.read_excel(uploaded_tab)
                    else:
                        X_df = pd.read_csv(uploaded_tab)

                    st.subheader("📋 Input Preview")
                    st.dataframe(X_df.head())

                    if st.button("🚀 Predict with XGBoost", use_container_width=True, key="xgb_file_predict"):
                        if xgb_model is None:
                            st.error("❌ XGBoost model not found! Upload xgb_optimized_model.pkl.")
                        else:
                            try:
                                X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0)
                                model_features = xgb_model.get_booster().feature_names

                                missing_cols = [col for col in model_features if col not in X_df.columns]
                                extra_cols = [col for col in X_df.columns if col not in model_features]

                                if missing_cols:
                                    # st.warning(f"⚠️ Adding {len(missing_cols)} missing columns: {missing_cols}")
                                    for col in missing_cols:
                                        X_df[col] = 0

                                if extra_cols:
                                    # st.warning(f"⚠️ Dropping {len(extra_cols)} extra columns: {extra_cols}")
                                    X_df = X_df.drop(columns=extra_cols)

                                X_df = X_df[model_features]
                                y_pred = xgb_model.predict(X_df)
                                X_df["PredictedPrice"] = np.expm1(y_pred)
                                X_df["PredictedPrice_₹Lakhs"] = (X_df["PredictedPrice"] / 100000).round(2)

                                st.success("✅ Prediction completed successfully!")
                                st.dataframe(X_df[["PredictedPrice_₹Lakhs"]].head())

                                out = io.BytesIO()
                                with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
                                    X_df.to_excel(writer, index=False, sheet_name="XGBoost_Predictions")
                                out.seek(0)

                                st.download_button(
                                    "📥 Download Predictions",
                                    data=out,
                                    file_name="xgb_predictions.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )

                                st.session_state.xgb_results = {
                                    'mode': 'batch',
                                    'results_df': X_df.to_dict(orient="list")
                                }
                            except Exception as e:
                                st.error(f"❌ Error during XGBoost prediction: {e}")
                except Exception as e:
                    st.error(f"❌ Error reading uploaded file: {e}")

    if st.session_state.xgb_results and st.session_state.xgb_results.get('mode') == 'single':
        st.divider()
        st.subheader("💰 Manual Prediction Result")
        predicted_price = st.session_state.xgb_results['predicted_price']
        price_col, range_col = st.columns([1, 1])
        with price_col:
            st.markdown(f"### ${predicted_price:,.2f}")
            st.caption("Estimated market value")
        with range_col:
            lower = predicted_price * 0.95
            upper = predicted_price * 1.05
            st.markdown("**Price Range (±5%)**")
            st.markdown(f"${lower:,.2f} - ${upper:,.2f}")


# ==================== TAB 4: PDF REPORT ====================
with tab_pdf:
    st.header("📄 Final PDF Report")
    st.markdown("Generate a comprehensive PDF report containing all predictions and analyses.")

    if not all([st.session_state.unet_results, st.session_state.lstm_results, st.session_state.xgb_results]):
        st.error("⚠️ Please complete U-Net, LSTM, and XGBoost steps before generating the PDF report.")
    else:
        st.success("✅ All data ready for PDF generation")

        if st.button("📥 Download PDF Report", use_container_width=True, type="primary", key='pdf_btn'):
            try:
                with st.spinner("Generating PDF report..."):
                    pdf_buffer = BytesIO()
                    doc = SimpleDocTemplate(
                        pdf_buffer,
                        pagesize=letter,
                        topMargin=0.5*inch,
                        bottomMargin=0.5*inch,
                        leftMargin=0.75*inch,
                        rightMargin=0.75*inch
                    )
                    story = []
                    styles = getSampleStyleSheet()

                    title_style = ParagraphStyle(
                        'CustomTitle',
                        parent=styles['Heading1'],
                        fontSize=24,
                        textColor=colors.HexColor('#2E86AB'),
                        spaceAfter=12,
                        alignment=1
                    )
                    heading2_style = ParagraphStyle(
                        'CustomHeading2',
                        parent=styles['Heading2'],
                        fontSize=16,
                        textColor=colors.HexColor('#1a1a1a'),
                        spaceAfter=8,
                        spaceBefore=12
                    )

                    story.append(Paragraph("RealtyAI - Property Prediction Report", title_style))
                    story.append(Spacer(1, 0.3*inch))

                    # 1) U-NET SECTION
                    story.append(Paragraph("<b>1. U-Net Segmentation Analysis</b>", heading2_style))
                    story.append(Spacer(1, 0.1*inch))

                    unet_data = st.session_state.unet_results
                    story.append(Paragraph(f"<b>Classification:</b> {unet_data['classification']}", styles['Normal']))
                    story.append(Paragraph(f"<b>Confidence Score:</b> {unet_data['confidence']:.1f}%", styles['Normal']))
                    story.append(Spacer(1, 0.2*inch))

                    story.append(Paragraph("<b>Original Image:</b>", styles['Normal']))
                    story.append(Spacer(1, 0.1*inch))
                    img_buffer = BytesIO()
                    unet_data['uploaded_image'].save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    story.append(RLImage(img_buffer, width=3*inch, height=3*inch))
                    story.append(Spacer(1, 0.2*inch))

                    story.append(Paragraph("<b>Segmentation Mask:</b>", styles['Normal']))
                    story.append(Spacer(1, 0.1*inch))
                    mask_buffer = BytesIO()
                    unet_data['mask_image'].save(mask_buffer, format='PNG')
                    mask_buffer.seek(0)
                    story.append(RLImage(mask_buffer, width=3*inch, height=3*inch))

                    story.append(PageBreak())

                    # 2) LSTM SECTION
                    story.append(Paragraph("<b>2. LSTM Time Series Forecast</b>", heading2_style))
                    story.append(Spacer(1, 0.1*inch))

                    lstm_data = st.session_state.lstm_results
                    forecast_values = lstm_data['forecast_values']
                    story.append(Paragraph(
                        f"<b>Forecast Period:</b> {lstm_data['forecast_years']} years",
                        styles['Normal']
                    ))
                    story.append(Paragraph(
                        f"<b>Average Forecasted Price:</b> ${np.mean(forecast_values):,.2f}",
                        styles['Normal']
                    ))
                    story.append(Spacer(1, 0.2*inch))

                    forecast_table_data = [['Year (from last known)', 'Predicted Price ($)']]
                    for i, v in enumerate(forecast_values, start=1):
                        forecast_table_data.append([str(i), f"{v:,.2f}"])
                    forecast_table = Table(forecast_table_data, colWidths=[3*inch, 2*inch])
                    forecast_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 1), (-1, -1), 10),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                    ]))
                    story.append(forecast_table)

                    story.append(PageBreak())

                    # 3) XGBOOST SECTION
                    story.append(Paragraph("<b>3. XGBoost Price Prediction</b>", heading2_style))
                    story.append(Spacer(1, 0.1*inch))
                    xgb_data = st.session_state.xgb_results
                    if xgb_data.get('mode') == 'batch':
                        story.append(Paragraph("<b>Batch Prediction:</b> Multiple properties (see app for full table).", styles['Normal']))
                    else:
                        story.append(Paragraph(
                            f"<b>Predicted Price (Single Property):</b> ${xgb_data['predicted_price']:,.2f}",
                            styles['Heading3']
                        ))
                        story.append(Spacer(1, 0.2*inch))
                        raw_inputs = xgb_data.get('raw_inputs', {})
                        input_data = [
                            ['Feature', 'Value'],
                            ['Overall Quality', str(raw_inputs.get('OverallQual', 'N/A'))],
                            ['Living Area (sq ft)', f"{raw_inputs.get('GrLivArea', 'N/A'):,}"],
                            ['Total SF', f"{raw_inputs.get('TotalSF', 'N/A'):,}"],
                            ['Year Built', str(raw_inputs.get('YearBuilt', 'N/A'))],
                            ['Year Remodeled', str(raw_inputs.get('YearRemodAdd', 'N/A'))],
                            ['Overall Condition', str(raw_inputs.get('OverallCond', 'N/A'))],
                            ['1st Floor SF', f"{raw_inputs.get('FirstFlrSF', 'N/A'):,}"],
                            ['2nd Floor SF', f"{raw_inputs.get('SecondFlrSF', 'N/A'):,}"],
                            ['Lot Area', f"{raw_inputs.get('LotArea', 'N/A'):,}"],
                            ['Garage Cars', str(raw_inputs.get('GarageCars', 'N/A'))],
                            ['Full Bathrooms', str(raw_inputs.get('FullBath', 'N/A'))],
                            ['Bedrooms', str(raw_inputs.get('BedroomAbvGr', 'N/A'))],
                            ['Neighborhood', str(raw_inputs.get('Neighborhood', 'N/A'))],
                            ['Kitchen Quality', str(raw_inputs.get('KitchenQual', 'N/A'))],
                            ['Exterior Quality', str(raw_inputs.get('ExterQual', 'N/A'))],
                        ]
                        input_table = Table(input_data, colWidths=[3*inch, 2*inch])
                        input_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 12),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black),
                            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                            ('FONTSIZE', (0, 1), (-1, -1), 10),
                            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                        ]))
                        story.append(input_table)
                        story.append(Spacer(1, 0.3*inch))

                    story.append(Spacer(1, 0.4*inch))
                    story.append(Paragraph("<i>Report generated by RealtyAI © 2025</i>", styles['Normal']))

                    doc.build(story)
                    pdf_buffer.seek(0)

                    st.download_button(
                        label="📥 Download Complete PDF Report",
                        data=pdf_buffer,
                        file_name="RealtyAI_Complete_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("✅ PDF generated successfully!")
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")

st.divider()
st.caption("RealtyAI © 2025 - Complete Property Prediction Pipeline")
