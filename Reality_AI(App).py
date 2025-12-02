import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import cv2
from PIL import Image
from io import BytesIO
from difflib import get_close_matches
import matplotlib.pyplot as plt
from prophet import Prophet
from pathlib import Path
import logging


#Configuration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Reality AI: Real Estate Intelligence",
    layout="wide",
    page_icon="🏠"
)


#  DASHBOARD ENTRY SCREEN

if "entered" not in st.session_state:
    st.session_state.entered = False

if not st.session_state.entered:
    st.markdown("""
        <div style='text-align:center; margin-top:150px;'>
            <h1 style='color:white; font-size:64px; font-weight:800;'>Reality AI</h1>
            <p style='color:#bbbbbb; font-size:22px;'>A Comprehensive Real Estate Intelligence Platform</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    _, mid, _ = st.columns(3)
    with mid:
        if st.markdown(
            """
            <style>
            div.stButton > button:first-child {
                background: linear-gradient(90deg, #4facfe, #00f2fe);
                color: black;
                border-radius: 10px;
                font-weight: 700;
                border: none;
                padding: 0.75rem 1.2rem;
                height: 3.5rem;
                font-size: 1.2rem;
                box-shadow: 0 4px 15px rgba(0, 242, 254, 0.4);
            }
            div.stButton > button:first-child:hover {
                background: linear-gradient(90deg, #4facfe, #4be4c2);
                box-shadow: 0 4px 20px rgba(79, 172, 254, 0.6);
            }
            </style>
            """, unsafe_allow_html=True
        ):
            pass

        if st.button("🚀 ENTER APPLICATION", use_container_width=True):
            st.session_state.entered = True
            st.rerun()
    st.stop()


st.markdown("""
<style>
/* Main Background and Text */
body, .stApp {
    background-color: #1a1a1a !important;
    color: #e6e6e6 !important;
    font-family: 'Inter', sans-serif;
}

/* Center section padding */
.main {
    padding: 2rem;
}

/* Titles */
.app-title {
    font-size: 3.5rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 0.5rem;
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.app-subtitle {
    text-align: center;
    font-size: 1.2rem;
    color: #aaaaaa;
    margin-bottom: 2.5rem;
}

/* Simple Card */
.simple-card {
    background: #242424;
    padding: 2rem;
    border-radius: 14px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
    margin-top: 1.5rem;
    border: 1px solid #333;
}

/* Buttons */
.stButton>button {
    background: #333;
    color: #eee;
    border-radius: 10px;
    font-weight: 600;
    border: none;
    padding: 0.5rem 1rem;
    height: 2.5rem;
    transition: 0.2s;
}

.stButton>button:hover {
    background: #444;
    color: #4facfe;
}

/* Navigation Step Buttons */
.step-nav {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}

/* Navigation button styling */
.stButton[key^="nav_"] button {
    background: #333;
    color: #bbb;
    border: 2px solid transparent;
}

.stButton[key^="nav_"] button:hover {
    background: #444;
    border-color: #00f2fe;
}

/* Metrics */
div[data-testid="stMetricValue"] {
    font-size: 2.2rem;
    font-weight: 800;
    color: #00f2fe;
}

div[data-testid="stMetricLabel"] {
    font-size: 0.9rem;
    color: #aaaaaa;
}

/* DataFrame */
.stDataFrame, .dataframe {
    background-color: #2c2c2c !important;
    border-radius: 8px;
}

.stDataFrame th, .stDataFrame td {
    color: #ffffff !important;
}

/* Inputs */
.stTextInput>div>div>input,
.stNumberInput>div>div>input,
.stSelectbox>div>div>div,
.stFileUploader>section {
    background-color: #2d2d2d;
    color: #fff;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 0.5rem;
}

/* Alerts */
.stAlert {
    background-color: #2a2a2a !important;
    border-left: 5px solid #4facfe !important;
    border-radius: 8px;
    color: #eee;
}

.stAlert p {
    color: #eee !important;
}

/* HR styling */
hr {
    border: none;
    height: 1px;
    background: #3a3a3a;
    margin: 2rem 0;
}

/* Image styling */
img {
    border-radius: 12px;
    box-shadow: 0 3px 15px rgba(0,0,0,0.5);
    border: 1px solid #333;
}

/* Badge */
.badge {
    padding: 0.5rem 1.2rem;
    font-weight: 700;
    border-radius: 20px;
    display: inline-block;
    color: white;
    margin-bottom: 1rem;
    font-size: 1rem;
    box-shadow: 0 2px 5px rgba(0,0,0,0.3);
}

.badge-commercial {
    background: #EB952C;
}

.badge-residential {
    background: #0A637B;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: nowrap;
    background: #333;
    border-radius: 8px 8px 0 0;
    padding: 0 15px;
    font-weight: 600;
    color: #bbb;
    transition: background 0.2s;
}

.stTabs [aria-selected="true"] {
    background: #4facfe;
    color: black;
    border-bottom: 3px solid #00f2fe;
}
</style>
""", unsafe_allow_html=True)


#  State Initialization

if "step" not in st.session_state:
    st.session_state.step = "satellite"
if "sat_classification" not in st.session_state:
    st.session_state.sat_classification = None
if "mask_image" not in st.session_state:
    st.session_state.mask_image = None
if "overlay_image" not in st.session_state:
    st.session_state.overlay_image = None
if "forecast_plot" not in st.session_state:
    st.session_state.forecast_plot = None
if "predicted_df" not in st.session_state:
    st.session_state.predicted_df = None
if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = None
if "forecast_region" not in st.session_state:
    st.session_state.forecast_region = None
if "user_region" not in st.session_state:
    st.session_state.user_region = None


# 🛠️ Utility Functions

def safe_load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return df, None
    except Exception as e:
        return None, str(e)

def create_download_button_excel(data, filename, label):
    buffer = BytesIO()
    data.to_excel(buffer, index=False)
    buffer.seek(0)
    st.download_button(
        label=label,
        data=buffer,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ✅ PDF REPORT GENERATOR

def create_final_report_pdf():
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    pdf.setFont("Helvetica-Bold", 22)
    pdf.setFillColor(colors.HexColor('#4facfe'))
    pdf.drawString(50, y, "Reality AI - Final Report")
    pdf.setFillColor(colors.black)
    y -= 40

    if st.session_state.sat_classification:
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(50, y, "Satellite Image Analysis")
        y -= 25
        pdf.setFont("Helvetica", 12)
        pdf.drawString(50, y, f"Classification: {st.session_state.sat_classification}")
        y -= 20

        if st.session_state.mask_image:
            try:
                pdf.drawString(50, y, "Mask Image:")
                y -= 15
                pdf.drawImage(ImageReader(st.session_state.mask_image), 50, y - 120, width=200, height=120)
                y -= 140
            except Exception:
                pass

        if st.session_state.overlay_image:
            try:
                pdf.drawString(50, y, "Overlay Image:")
                y -= 15
                pdf.drawImage(ImageReader(st.session_state.overlay_image), 50, y - 120, width=200, height=120)
                y -= 150
            except Exception:
                pass
        pdf.line(40, y, width - 40, y)
        y -= 30

    if st.session_state.predicted_df is not None:
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(50, y, "Property Price Predictions")
        y -= 30
        try:
            df = st.session_state.predicted_df.copy()
            max_rows = 20
            display_df = df if len(df) <= max_rows else df.head(max_rows)
            if "Predicted_Price" in display_df.columns:
                display_df["Predicted_Price"] = display_df["Predicted_Price"].apply(lambda x: f"${x:,.0f}")
            table_data = [list(display_df.columns)] + display_df.values.tolist()
            
            num_cols = len(display_df.columns)
            col_width = (width - 100) / num_cols if num_cols > 0 else 100
            table = Table(table_data, colWidths=[col_width]*num_cols)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4facfe')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.black),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
            ]))
            table_height = table.wrapOn(pdf, width - 100, height)[1]
            table.drawOn(pdf, 50, y - table_height)
            y -= (table_height + 20)
        except Exception:
            pdf.drawString(50, y, "Table rendering error.")
            y -= 20
        pdf.line(40, y, width - 40, y)
        y -= 30

    if st.session_state.forecast_df is not None:
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(50, y, "Market Forecast")
        y -= 30
        pdf.setFont("Helvetica", 12)
        pdf.drawString(50, y, f"Region: {st.session_state.forecast_region}")
        y -= 20
        try:
            last_yhat = st.session_state.forecast_df["yhat"].iloc[-1]
            pdf.drawString(50, y, f"Latest Forecasted Value: ${last_yhat:,.2f}")
            y -= 20
        except Exception:
            pass
        if st.session_state.forecast_plot:
            try:
                pdf.drawImage(ImageReader(st.session_state.forecast_plot), 50, y - 220, width=350, height=220)
                y -= 240
            except Exception:
                pass

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer


#  Satellite Model Loading & Helpers

@tf.keras.utils.register_keras_serializable()
def iou_metric(y_true, y_pred):
    K = tf.keras.backend
    y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
    inter = tf.reduce_sum(tf.abs(y_true * y_pred), axis=[1,2,3])
    union = tf.reduce_sum(tf.abs(y_true) + tf.abs(y_pred), axis=[1,2,3]) - inter
    return tf.reduce_mean((inter + K.epsilon()) / (union + K.epsilon()), axis=0)

@st.cache_resource
def load_satellite_model():
    model_path = Path(r"C:\Users\mouni\Downloads\Satillite_Images\processed_data\unet_model\unet_final_model.keras")
    if not model_path.exists():
        return None 
    try:
        model = tf.keras.models.load_model(str(model_path), custom_objects={"iou_metric": iou_metric}, compile=False)
        return model
    except Exception:
        return None

def enhance_image(img_pil):
    img = np.array(img_pil.convert("RGB"))
    img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2), interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    img = cv2.convertScaleAbs(img, alpha=1.15, beta=12)
    return img

def classify_land_use(mask):
    mask_bin = (mask > 0).astype(np.uint8)
    ratio = np.sum(mask_bin) / mask_bin.size
    if ratio < 0.05:
        return "Commercial", "Very low density - likely commercial/industrial"
    elif ratio >= 0.35:
        return "Commercial", "Very high density - likely commercial district"
    else:
        return "Residential", "Medium density - typical residential area"

def create_colored_overlay(enhanced_img, mask):
    mask_img = (mask.squeeze() * 255).astype(np.uint8)
    resized_img = cv2.resize(enhanced_img, (128, 128))
    mask_colored = cv2.applyColorMap(mask_img, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(resized_img, 0.6, mask_colored, 0.4, 0)
    return overlay


#  House Price Model Loader

@st.cache_resource
def load_house_model():
    model_path = Path("house_price_prediction_model.pkl")
    if not model_path.exists():
        return None
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


#  HEADER 

st.markdown('<div class="app-title">Reality AI</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Intelligent Property Analysis System</div>', unsafe_allow_html=True)

steps = ["satellite", "price", "forecast"]
step_labels = ["🛰️ Land Use Analysis", "🏡 Price Prediction", "📈 Market Forecast"]

col_nav = st.columns(3)
for i, (step, label) in enumerate(zip(steps, step_labels)):
    with col_nav[i]:
        button_key = f"nav_{step}"
        if st.session_state.step == step:
            st.markdown(f"""<style>div.stButton[data-testid="stColumn"] > div > button[data-testid="stKey-{button_key}"] {{ background: linear-gradient(90deg, #4facfe, #00f2fe); color: black !important; font-weight: 700; border: 2px solid #00f2fe; }}</style>""", unsafe_allow_html=True)
        if st.button(label, key=button_key, use_container_width=True):
            st.session_state.step = step
            st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)


#  STEP 1 — Satellite Analysis

if st.session_state.step == "satellite":
    st.markdown('<div class="simple-card">', unsafe_allow_html=True)
    st.markdown("### 🛰️ Land Use Classification")

    sat_model = load_satellite_model()
    uploaded_img = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png"], key="sat_img")

    if uploaded_img:
        try:
            img = Image.open(uploaded_img).convert("RGB")
            enhanced = enhance_image(img)
            col_disp1, col_disp2 = st.columns(2)
            with col_disp1:
                st.write("**Original Image**")
                st.image(img, use_container_width=True)
            with col_disp2:
                st.write("**Enhanced Image**")
                st.image(enhanced, use_container_width=True)

            st.markdown("---")
            if st.button("Analyze Image", use_container_width=True):
                with st.spinner("Analyzing..."):
                    try:
                        if sat_model:
                            resized = cv2.resize(enhanced, (128, 128))
                            x = resized.astype(np.float32) / 255.0
                            if hasattr(sat_model, "input_shape") and sat_model.input_shape[-1] == 4:
                                ndvi = np.mean(x, axis=-1, keepdims=True)
                                x = np.concatenate([x, ndvi], axis=-1)
                            pred = sat_model.predict(np.expand_dims(x, axis=0), verbose=0)
                            mask = (pred[0] > 0.5).astype(np.uint8)
                            label, exp = classify_land_use(mask)
                        else:
                            mask = np.ones((128, 128, 1))
                            label, exp = classify_land_use(mask)
                            exp = "Heuristic classification (Model not loaded)."
                        
                        st.session_state.sat_classification = label
                        color_map = {"Commercial": "#EB952C", "Residential": "#0A637B"}
                        color = color_map.get(label, "#3498DB")

                        st.markdown(f"""
                            <div style='text-align:center; margin:20px 0; background: linear-gradient(135deg, {color} 0%, {color}DD 100%);
                            padding:30px; border-radius:15px; color:white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                                <h2 style='margin:0; font-size:32px; font-weight:bold;'>Land Use: {label}</h2>
                                <p style='margin:10px 0 0 0; font-size:16px; opacity:0.95;'>{exp}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.success("Analysis Complete!")

                        if sat_model:
                            mask_image = (mask.squeeze() * 255).astype(np.uint8)
                            mask_pil = Image.fromarray(mask_image)
                            mask_buf = BytesIO()
                            mask_pil.save(mask_buf, format="PNG")
                            mask_buf.seek(0)
                            st.session_state.mask_image = mask_buf

                            overlay = create_colored_overlay(enhanced, mask)
                            overlay_pil = Image.fromarray(overlay)
                            overlay_buf = BytesIO()
                            overlay_pil.save(overlay_buf, format="PNG")
                            overlay_buf.seek(0)
                            st.session_state.overlay_image = overlay_buf
                            
                            col_seg1, col_seg2 = st.columns(2)
                            with col_seg1: st.image(mask_pil, caption="Building Mask", use_container_width=True)
                            with col_seg2: st.image(overlay_pil, caption="Overlay", use_container_width=True)
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")

            if st.session_state.sat_classification:
                st.markdown("---")
                if st.button("Continue to Price Prediction (Step 2/3)", type="primary", use_container_width=True):
                    st.session_state.step = "price"
                    st.rerun()
        except Exception as e:
             st.error(f"Error loading image: {e}")
    st.markdown('</div>', unsafe_allow_html=True)


#  STEP 2 — Price Prediction

elif st.session_state.step == "price":
    st.markdown('<div class="simple-card">', unsafe_allow_html=True)
    st.markdown("### 🏡 Property Price Prediction")

    if st.session_state.sat_classification:
        badge_class = "badge-commercial" if st.session_state.sat_classification == "Commercial" else "badge-residential"
        st.markdown(f'<div class="badge {badge_class}">🛰️ Detected Land Use: {st.session_state.sat_classification}</div>', unsafe_allow_html=True)

    model = load_house_model()
    tab1, tab2 = st.tabs(["📂 Batch Prediction", "✍️ Single Property Entry"])

    with tab1:
        uploaded = st.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx", "xls"], key="price_upload")
        if uploaded and model:
            df, err = safe_load_data(uploaded)
            if err:
                st.error(f"Error loading file: {err}")
            else:
                st.success(f"File loaded successfully: {len(df)} properties found.")
                st.dataframe(df.head(), use_container_width=True)
                if st.button("Generate Batch Predictions", key="batch_predict_btn", type="primary", use_container_width=True):
                    with st.spinner("Calculating..."):
                        try:
                            features = list(model.feature_names_in_)
                            X = pd.DataFrame(index=df.index)
                            for f in features:
                                match = get_close_matches(f, df.columns, n=1, cutoff=0.6)
                                if match: X[f] = pd.to_numeric(df[match[0]], errors="coerce").fillna(0)
                                else: X[f] = 0
                            preds = model.predict(X)
                            df["Predicted_Price"] = preds
                            st.session_state.predicted_df = df.copy()
                            st.success("Batch Predictions Complete!")
                            col1, col2 = st.columns(2)
                            col1.metric("Avg Price", f"${preds.mean():,.0f}")
                            col2.metric("Max Prediction", f"${preds.max():,.0f}")
                        except Exception as e:
                            st.error(f"Prediction Error: {e}")
        elif uploaded and model is None:
             st.warning("Model not available.")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            b = st.number_input("Bedrooms", 0, 20, 3, key="man_bed")
            sqft = st.number_input("Square Footage", 50, 20000, 1200, key="man_sqft")
            floors = st.number_input("Number of Floors", 1, 5, 1, key="man_floors")
        with col2:
            bath = st.number_input("Bathrooms", 0.0, 10.0, 2.0, 0.5, key="man_bath")
            year = st.number_input("Year Built", 1800, 2025, 2000, key="man_year")
            region_name = st.text_input("Region", value=st.session_state.user_region or "", placeholder="e.g., New York", key="man_region")

        if st.button("Calculate Single Price", key="single_predict_btn", use_container_width=True):
            with st.spinner("Calculating..."):
                if model:
                    try:
                        feats = list(model.feature_names_in_)
                        row = {}
                        for f in feats:
                            key = f.lower()
                            if "bed" in key: row[f] = b
                            elif "bath" in key: row[f] = bath
                            elif "sqft" in key or "area" in key: row[f] = sqft
                            elif "year" in key: row[f] = year
                            elif "floor" in key: row[f] = floors
                            else: row[f] = 0
                        X = pd.DataFrame([row])
                        pred = model.predict(X)[0]
                        if region_name: st.session_state.user_region = region_name
                        df = pd.DataFrame([{"Bedrooms": b, "Bathrooms": bath, "Sqft": sqft, "Floors": floors, "Year": year, "Region": region_name or "N/A", "Predicted_Price": pred}])
                        st.session_state.predicted_df = df.copy()
                        st.success("Calculation Complete!")
                        st.metric("Estimated Price", f"${pred:,.0f}")
                    except Exception as e: st.error(f"Error: {e}")
                else:
                    heuristic = sqft * 150 + (b * 15000) + (bath * 10000)
                    st.metric("Heuristic Estimate", f"${heuristic:,.0f}")
                    st.session_state.predicted_df = pd.DataFrame([{"Bedrooms": b, "Bathrooms": bath, "Sqft": sqft, "Floors": floors, "Year": year, "Region": region_name or "N/A", "Predicted_Price": heuristic}])
                    if region_name: st.session_state.user_region = region_name

    if st.session_state.predicted_df is not None:
        st.markdown("---")
        display_df = st.session_state.predicted_df.copy()
        if "Predicted_Price" in display_df.columns:
             display_df["Predicted_Price"] = display_df["Predicted_Price"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(display_df, use_container_width=True)
        col_res1, col_res2 = st.columns(2)
        with col_res1: create_download_button_excel(st.session_state.predicted_df, "Price_Predictions.xlsx", "📥 Download Results")
        with col_res2:
            if st.button("Continue to Market Forecast (Step 3/3)", type="primary", use_container_width=True):
                st.session_state.step = "forecast"
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


#  STEP 3 — Forecasting

elif st.session_state.step == "forecast":
    st.markdown('<div class="simple-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Market Forecasting")

    if st.session_state.user_region:
        st.info(f"📍 Region auto-filled: **{st.session_state.user_region}**")

    uploaded = st.file_uploader("Upload Historical Data (Time-Series)", type=["csv", "xlsx"], key="forecast_upload")

    if uploaded:
        df, err = safe_load_data(uploaded)
        if err: st.error(f"Error: {err}")
        else:
            st.success(f"Loaded {len(df)} records.")
            st.dataframe(df.head(), use_container_width=True)
            default_region = st.session_state.user_region if st.session_state.user_region else ""
            col_input1, col_input2 = st.columns(2)
            with col_input1: region = st.text_input("Enter Target Region Name", value=default_region, key="forecast_region_input")
            with col_input2: months = st.slider("Forecast Horizon (Months)", 1, 72, 12)

            if st.button("Generate Forecast", type="primary", use_container_width=True):
                if not region: st.error("Please enter a region name.")
                else:
                    with st.spinner(f"Training Prophet model for {region}..."):
                        try:
                            if "RegionName" not in df.columns:
                                st.error("Dataset must contain a 'RegionName' column.")
                                st.stop()
                            if region not in df["RegionName"].astype(str).values:
                                st.error(f"Region '{region}' not found in data.")
                                st.stop()
                            price_col = [c for c in df.columns if "price" in c.lower() or "zhvi" in c.lower()]
                            if not price_col:
                                st.error("No suitable price column found.")
                                st.stop()
                            price_col = price_col[0]

                            region_df = df[df["RegionName"].astype(str) == region][["Date", price_col]].rename(columns={"Date": "ds", price_col: "y"})
                            region_df["ds"] = pd.to_datetime(region_df["ds"])
                            region_df.dropna(subset=['y'], inplace=True)
                            
                            model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                            model.fit(region_df)
                            future = model.make_future_dataframe(periods=months, freq="MS")
                            forecast = model.predict(future)

                            st.session_state.forecast_df = forecast
                            st.session_state.forecast_region = region
                            st.success("Forecast Complete!")

                            latest_price = region_df["y"].iloc[-1] if not region_df.empty else 0
                            forecast_price = forecast["yhat"].iloc[-1]
                            change = ((forecast_price - latest_price) / latest_price) * 100 if latest_price != 0 else np.nan
                            change_str = f"{change:+.1f}%" if not np.isnan(change) else "N/A"

                            col_met1, col_met2, col_met3 = st.columns(3)
                            col_met1.metric("Current Price", f"${latest_price:,.0f}")
                            col_met2.metric(f"Forecast ({months}mo)", f"${forecast_price:,.0f}")
                            col_met3.metric("Expected Change", change_str)

                            fig = model.plot_components(forecast)
                            buf = BytesIO()
                            fig.savefig(buf, format="png")
                            buf.seek(0)
                            st.session_state.forecast_plot = buf
                            st.pyplot(fig)
                            plt.close()
                            
                            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(months), use_container_width=True)
                            create_download_button_excel(forecast, f"Forecast_{region}.xlsx", "📥 Download Forecast Data")
                        except Exception as e:
                            st.error(f"Error during forecasting: {e}")
    st.markdown('</div>', unsafe_allow_html=True)


#  FOOTER

st.markdown("<hr>", unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    if st.button("🔄 Reset Application", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key not in ['entered']:
                del st.session_state[key]
        st.session_state.step = "satellite"
        st.rerun()

with c2:
    if any([st.session_state.sat_classification, st.session_state.predicted_df is not None, st.session_state.forecast_df is not None]):
        try:
            pdf_buffer = create_final_report_pdf()
            st.download_button(label="📥 Download Final Report (PDF)", data=pdf_buffer, file_name="Reality_AI_Report.pdf", mime="application/pdf", use_container_width=True, type="primary")
        except Exception:
            pass

with c3:
    st.markdown("<div style='text-align:right; color:#777; font-size:0.8rem;'>Reality AI</div>", unsafe_allow_html=True)