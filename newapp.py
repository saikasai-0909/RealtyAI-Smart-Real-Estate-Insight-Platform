import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import joblib
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import xgboost as xgb
import plotly.express as px
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from PIL import Image as PILImage
import numpy as np
from datetime import datetime


import streamlit as st
import sqlite3
import hashlib

# Initialize session_state keys for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

# SQLite setup for users (your provided code)
conn = sqlite3.connect("newusers.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE,
    password TEXT
)
""")
conn.commit()

c.execute("""
CREATE TABLE IF NOT EXISTS pdf_downloads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT,
    downloaded_at TEXT,
    report_name TEXT
)
""")
conn.commit()
def init_db():
    conn = sqlite3.connect("newusers.db", check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS pdf_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            timestamp TEXT,
            report_name TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(password, hashed):
    return hash_password(password) == hashed

def add_user(name, email, password):
    c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
              (name, email, hash_password(password)))
    conn.commit()
def add_pdf_download(email, report_name="RealtyAIFullReport.pdf"):
    ts = datetime.now().isoformat(timespec="seconds")
    c.execute(
        "INSERT INTO pdf_downloads (email, downloaded_at, report_name) VALUES (?, ?, ?)",
        (email, ts, report_name),
    )
    conn.commit()

def get_pdf_history(email, limit=20):
    c.execute(
        "SELECT downloaded_at, report_name FROM pdf_downloads WHERE email = ? ORDER BY id DESC LIMIT ?",
        (email, limit),
    )
    return c.fetchall()
def add_user_log(email, action, details=""):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    ts = datetime.now().isoformat(timespec='seconds')
    c.execute("INSERT INTO historylogs (email, action, time) VALUES (?, ?, ?)",
              (email, action, ts))
    conn.commit()
    conn.close()


def get_user(email):
    c.execute("SELECT * FROM users WHERE email=?", (email,))
    return c.fetchone()

def login_signup_page():
    st.markdown("""
    <div style='text-align:center; margin-bottom: 25px;'>
        <h1 style='color:#7B3F00; font-family:"Segoe UI", Tahoma, Geneva, Verdana, sans-serif; font-size: 42px;'>
            🔑 RealtyAI <span style='color:#D2691E;'>Login / Sign Up</span>
        </h1>
        <p style='font-size:16px; color:#4B4B4B; font-weight: 600;'>
            Unlock the future of property with <span style='color:#D2691E; font-weight:bold;'>smart AI insights</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 4, 2])  # wider center column
    with col2:

        tab1, tab2 = st.tabs([
            "Login","SignUp"
        ])

        with tab1:
            email = st.text_input("Email", key="login_email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")

            if st.button("Login", key="login_button"):
                user = get_user(email)
                if user and check_password(password, user[3]):
                    st.session_state.logged_in = True
                    st.session_state.user_name = user[1]
                    st.session_state.user_email = user[2]
                    st.success(f"Welcome back, {user[1]} 👋")
                    st.rerun()
                else:
                    st.error("Invalid email or password")

        with tab2:
            name = st.text_input("Full Name", key="signup_name", placeholder="FullName")
            email = st.text_input("Email", key="signup_email", placeholder="you@gmail.com")
            password = st.text_input("Password", type="password", key="signup_password", placeholder="Create a password")
            confirm = st.text_input("Confirm Password", type="password", key="signup_confirm", placeholder="Confirm your password")

            if st.button("Sign Up", key="signup_button"):
                if password != confirm:
                    st.error("Passwords do not match!")
                elif get_user(email):
                    st.warning("User already exists. Try logging in.")
                else:
                    add_user(name, email, password)
                    st.success("✅ Account created successfully! Please log in.")




st.markdown(
    """
    <style>
    /* Remove top/bottom padding, force block container to fill window */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100% !important;
        width: 100% !important;
    }
    /* Optionally: remove extra padding from headers/columns */
    .element-container {
        padding-bottom: 0rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# ===============================
# Config / file paths
# ===============================
SPACENET_MODEL_PATH = "models/best_spacenet_unet.keras"
CITY_BUNDLE_PATH = "models/city_model_bundle.pkl"
HOUSE_MODEL_PATH = "models/xgboost_houseprice_model.pkl"

HOUSE_FEATURES = [
    "OverallQual","GrLivArea","GarageCars","GarageArea","TotalBsmtSF","1stFlrSF",
    "FullBath","YearBuilt","YearRemodAdd","Fireplaces","KitchenQual","BsmtQual",
    "ExterQual","Neighborhood","LotArea","TotRmsAbvGrd","GarageFinish","Functional",
    "MasVnrArea","BsmtFinSF1","GarageYrBlt","LotFrontage","OverallCond","YrSold"
]

CITY_TARGET = "MedianListingPricePerSqft_AllHomes"

# ===============================
# Custom Keras losses for loading model
# ===============================
def dice_loss(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred)
    return 1 - (2. * intersection + smooth) / (union + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# ===============================
# Cached model loaders
# ===============================
@st.cache_resource(show_spinner=False)
def load_spacenet():
    if not os.path.exists(SPACENET_MODEL_PATH):
        return None
    return load_model(SPACENET_MODEL_PATH, custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_loss': dice_loss})

@st.cache_resource(show_spinner=False)
def load_city_bundle():
    if not os.path.exists(CITY_BUNDLE_PATH):
        return None
    return joblib.load(CITY_BUNDLE_PATH)

@st.cache_resource(show_spinner=False)
def load_house_model():
    if not os.path.exists(HOUSE_MODEL_PATH):
        return None
    return joblib.load(HOUSE_MODEL_PATH)

# ===============================
# Satellite segmentation helpers
# ===============================
def classify_image_building_type(pred_mask_bin, area_thresh=500):
    contours, _ = cv2.findContours(pred_mask_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "No buildings detected", 0.0, 0.0, 0
    building_areas = [cv2.contourArea(c) for c in contours]
    num_large = sum(1 for a in building_areas if a >= area_thresh)
    num_small = sum(1 for a in building_areas if a < area_thresh)
    image_type = "Commercial" if num_large > num_small else "Residential"
    avg_area = float(np.mean(building_areas))
    max_area = float(np.max(building_areas))
    return image_type, avg_area, max_area, len(building_areas)

def predict_and_classify_image_from_bytes(image_bytes, model, model_input_size=(128,128), area_thresh=500):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if img is None:
        raise ValueError("Could not decode image")
    original_shape = img.shape[:2]
    if img.ndim == 2:
        img = np.stack([img]*3 + [np.zeros_like(img)], axis=-1)
    elif img.shape[2] == 3:
        alpha = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)
        img = np.concatenate([img, alpha], axis=-1)
    elif img.shape[2] > 4:
        img = img[:,:,:4]
    img_resized = cv2.resize(img, model_input_size, interpolation=cv2.INTER_LINEAR)
    maxv = np.max(img_resized)
    if maxv > 0:
        img_resized = img_resized / maxv
    pred_mask_prob = model.predict(np.expand_dims(img_resized, axis=0))[0,:,:,0]
    pred_mask = (pred_mask_prob > 0.5).astype(np.uint8)
    pred_mask_original = cv2.resize(pred_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    img_display = img[:,:,:3].copy().astype(np.uint8)
    contours, _ = cv2.findContours(pred_mask_original.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        color = (0,255,0) if a < area_thresh else (255,0,0)
        cv2.rectangle(img_display, (x,y), (x+w,y+h), color, 2)
    img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    img_type, avg_area, max_area, n_buildings = classify_image_building_type(pred_mask_original, area_thresh=area_thresh)
    return pred_mask_original, img_type, avg_area, max_area, n_buildings, img_rgb

# ===============================
# City time-series helper
# ===============================
def run_city_prediction(df, city_bundle, region_name):
    model = city_bundle.get("model")
    scaler_X = city_bundle.get("scaler_X")
    scaler_y = city_bundle.get("scaler_y")
    if model is None or scaler_X is None or scaler_y is None:
        raise ValueError("city_model_bundle must contain 'model','scaler_X','scaler_y'")

    df_region = df[df["RegionName"] == region_name].copy()
    if df_region.empty:
        raise ValueError(f"No data for RegionName = {region_name}")
    df_region = df_region.sort_values("Date").reset_index(drop=True)
    feature_cols = [c for c in df_region.columns if c not in ["Date","RegionName", CITY_TARGET]]
    if len(feature_cols) == 0:
        raise ValueError("No feature columns for city model found")
    X = df_region[feature_cols].values.astype(float)
    X_scaled = scaler_X.transform(X)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    y_pred_scaled = model.predict(X_lstm)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
    df_region["Predicted"] = y_pred
    return df_region, feature_cols

def predict_full_year_forecast(df, city_bundle, regions, forecast_year):
    all_yearly_forecasts = []
    future_dates = pd.date_range(start=pd.Timestamp(year=forecast_year, month=1, day=1),
                                 end=pd.Timestamp(year=forecast_year, month=12, day=31),
                                 freq='MS')

    for region in regions:
        df_region = df[df["RegionName"] == region].copy()
        df_region['Date'] = pd.to_datetime(df_region['Date'])
        df_region = df_region.sort_values("Date").reset_index(drop=True)
        last_row = df_region.iloc[-1]
        future_df = pd.DataFrame({'Date': future_dates})
        for col in df_region.columns:
            if col not in ['Date', 'RegionName']:
                future_df[col] = last_row[col]
        future_df['RegionName'] = region
        df_combined = pd.concat([df_region, future_df], ignore_index=True)
        df_predicted, _ = run_city_prediction(df_combined, city_bundle, region)
        df_predicted['Year'] = df_predicted['Date'].dt.year
        df_year_forecast = df_predicted[df_predicted['Year'] == forecast_year].copy()
        all_yearly_forecasts.append(df_year_forecast)

    full_year_forecast = pd.concat(all_yearly_forecasts, ignore_index=True)
    return full_year_forecast

# ===============================
# House price prediction helper
# ===============================
def predict_house_single(input_dict, house_model, model_outputs_log=False):
    df = pd.DataFrame([input_dict])
    try:
        pred = house_model.predict(df)
    except Exception:
        numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if len(numeric_cols) == 0:
            raise RuntimeError("House model requires numeric inputs but none were provided")
        dmat = xgb.DMatrix(df[numeric_cols].values, feature_names=numeric_cols)
        pred = house_model.predict(dmat)
    raw = np.array(pred).ravel()[0]
    if model_outputs_log:
        price = float(np.expm1(raw))
    else:
        price = float(raw)
    return raw, price


# ===============================
# Load models once
# ===============================
spacenet_model = load_spacenet()
city_bundle = load_city_bundle()
house_model = load_house_model()


if st.session_state.logged_in:
    st.sidebar.markdown(f"""
    <p style="
        color: #7B3F00;           /* Rich warm brown shade */
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 15px;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-shadow: 0 0 8px #D2691E;   /* Orange glowing shadow */
    ">
        Welcome {st.session_state.user_name} 👋
    </p>
""", unsafe_allow_html=True)

    


    st.set_page_config(layout="wide")

    # Initialize session state for page selection
    if 'section' not in st.session_state:
        st.session_state['section'] = "about"

    # Sidebar with large section buttons
    with st.sidebar:
        st.markdown("""
                    

<h2 style='
    background: linear-gradient(45deg, #6a11cb, #2575fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 28px;
    text-align: center;
    text-shadow: 0 2px 4px rgba(0,0,0,0.15);
    margin-bottom: 20px;
    font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
'>
    NAVBAR
</h2>
""", unsafe_allow_html=True)
        if st.button("🏡 About Project"):
            st.session_state['section'] = "about"
        if st.button("🏢RealtyAI-Pipeline"):
            st.session_state['section'] = "IntegratePipeline"
        if st.button("History"):
            st.session_state['section'] = "History"
        for _ in range(31):
            st.write("")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_name = ""
            st.session_state.user_email = ""
            st.rerun()
    if st.session_state['section'] == "about":
        # Main title, large
        st.markdown("""
        <div style="text-align:center;">
            <h1 style="color:#1967d2; font-size:46px; margin-bottom:10px;">🏡 RealtyAI Project Overview</h1>
            <hr style="height:2px; width:35%; background:#1967d2; border:none; margin:auto;">
        </div>
        """, unsafe_allow_html=True)

        # Description with larger font
        st.markdown("""
        <div style="font-size:22px; line-height:1.7; margin-top:28px;">
            <p>
                <b>RealtyAI</b> offers a comprehensive platform integrating three advanced AI models:
                <span style="color:#1967d2; font-weight:bold;">Segmentation</span>,
                <span style="color:#2e7d32; font-weight:bold;">House Price Prediction</span>,
                and
                <span style="color:#fbc02d; font-weight:bold;">Future Forecasting</span>
                <br><br>
                These models work together to deliver deep insights and reliable, data-driven decision support across the real estate value chain:
            </p>
            <ul style="list-style-type:square; margin-left:30px; font-size:21px;">
                <li>
                    <b style="color:#1967d2;">🛰️ Segmentation Model</b>:
                    Processes satellite imagery with AI to automatically identify and classify property types, streamlining site analysis and accelerating project planning.
                </li>
                <li>
                    <b style="color:#2e7d32;">💸 House Price Prediction</b>:
                    Employs machine learning to generate highly accurate, unbiased valuations using real-time market data, helping buyers, sellers, and investors transact with confidence.
                </li>
                <li>
                    <b style="color:#fbc02d;">📈 Future Forecasting</b>:
                    Analyzes historical and economic data to provide city-level predictions, enabling stakeholders to anticipate market trends and make strategic investment decisions.
                </li>
            </ul>
            <p>
                <span style="color:#1967d2; font-weight:bold; font-size:23px;">✨ Standout Feature:</span>
                <br>
                Automated <b>report generation</b> brings together the results from all three models into clear, actionable presentations—saving time, reducing manual effort, and greatly improving understanding for real estate professionals, clients, and investors.
            </p>
            <p>
                <span style="background:#e3f2fd; padding:6px 12px; border-radius:7px; color:#1967d2; font-size:22px;">
                    RealtyAI empowers you with faster evaluations, smarter recommendations, and forward-looking insights—creating a competitive edge and maximizing the value of every real estate project.
                </span>
            </p>
        </div>
        """, unsafe_allow_html=True)


    elif st.session_state['section']=="IntegratePipeline":
    # ===============================
    # Main integrated pipeline app
    # ===============================
        # def main():
            st.markdown("""
            <h1 style='text-align:center;
                    color:#1967d2;
                    font-size:48px;
                    font-family:Segoe UI, Arial, sans-serif;
                    margin-bottom:10px;
                    letter-spacing:1px;'>
                🏡 RealtyAI
            </h1>
        """, unsafe_allow_html=True)

            st.markdown("""
                <h3 style='text-align:center;
                        color:#2e7d32;
                        font-size:28px;
                        font-family:Segoe UI, Arial, sans-serif;
                        margin-top:0;
                        margin-bottom:18px;
                        font-weight:500;'>
                    A Smart Real Estate Insight Platform
                </h3>
            """, unsafe_allow_html=True)


            
            st.header("🛰️Satellite Image Segmentation & Classification")

            uploaded_img = st.file_uploader(
                "Upload satellite image (.jpg/.png/.tif)", 
                type=["jpg", "jpeg", "png", "tif", "tiff"]
            )

            area_thresh = st.number_input(
                "Area threshold (px) to mark building as 'large'", 
                value=500,
                step=50
            )

            if uploaded_img is not None:
                # Only analyze if not already done or user reloads explicitly
                if "classified" not in st.session_state or not st.session_state.classified:
                    if st.button("🚀 Analyze Satellite Image"):
                        bytes_data = uploaded_img.read()
                        try:
                            mask, img_type, avg_area, max_area, n_buildings, overlay = predict_and_classify_image_from_bytes(
                                bytes_data,
                                spacenet_model,
                                model_input_size=(128,128),
                                area_thresh=area_thresh
                            )

                            # Store results in session state to avoid reprocessing on reruns (e.g., download clicks)
                            st.session_state.classified = True
                            st.session_state.classification_result = img_type
                            st.session_state.satellite_bytes = bytes_data
                            st.session_state.mask = (mask * 255).astype("uint8")
                            st.session_state.overlay = overlay
                            st.session_state.display_results = {
                                "n_buildings": n_buildings,
                                "avg_area": avg_area,
                                "max_area": max_area,
                            }
                        except Exception as e:
                            st.error(f"Satellite segmentation failed: {e}")

                if st.session_state.get("classified", False):
                    
                    st.markdown("""
                        <div style='
                            background: #f9fbe7;
                            border-radius: 10px;
                            padding: 20px 24px 14px 24px;
                            margin-top: 12px;
                            margin-bottom: 18px;
                            box-shadow: 0px 2px 6px rgba(30,30,70,0.05);
                        '>
                            <h2 style='color:#1967d2; font-size:32px; margin-top:0; margin-bottom:18px; text-align:center;'>
                                🏠 Classification Result
                            </h2>
                            <div style='font-size:22px; color:#333; margin-bottom:12px; text-align:center;'>
                                <span style='color:#2e7d32; font-weight:bold;'>Detected Type:</span>
                                <span style='margin-left:7px; font-weight:bold; color:#333;'>{}</span>
                            </div>
                            <div style='font-size:20px; margin:5px 0 5px 0; text-align:center;'>
                                <span style='color:#6d4c41;'><b>Total Buildings:</b></span>
                                <span style='margin-left:7px; color:#333;'>{}</span>
                            </div>
                            <div style='font-size:20px; margin:5px 0 5px 0; text-align:center;'>
                                <span style='color:#0277bd;'><b>Average Area:</b></span>
                                <span style='margin-left:7px; color:#333;'>{:.2f} sq px</span>
                            </div>
                            <div style='font-size:20px; margin:5px 0 7px 0; text-align:center;'>
                                <span style='color:#fbc02d;'><b>Max Area:</b></span>
                                <span style='margin-left:7px; color:#333;'>{:.2f} sq px</span>
                            </div>
                        </div>
                        """.format(
                            st.session_state['classification_result'],
                            st.session_state.display_results['n_buildings'],
                            st.session_state.display_results['avg_area'],
                            st.session_state.display_results['max_area']
                        ), unsafe_allow_html=True)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Binary Mask")
                        st.image(st.session_state.mask, caption="Binary Segmentation Mask", use_container_width=True)
                        
                        st.download_button(
                            label="⬇ Download Binary Mask",
                            data=st.session_state.mask.tobytes(),
                            file_name="binary_mask.png",
                            mime="image/png"
                        )

                    with col2:
                        st.subheader("Overlay Mask")
                        st.image(st.session_state.overlay, caption="Overlay Image", use_container_width=True)
                        overlay_pil = PILImage.fromarray(st.session_state.overlay)
                        buf = io.BytesIO()
                        overlay_pil.save(buf, format="PNG")
                        st.download_button(
                            label="⬇ Download Overlay Image",
                            data=buf.getvalue(),
                            file_name="overlay_image.png",
                            mime="image/png"
                        )
                    st.session_state['overlay_bytes'] = buf.getvalue()


                



            # -------------------------------
            # STEP 2: House Price Prediction
            # -------------------------------
            if st.session_state.get("classified", False):

                st.header(" 💸House Price Prediction")
                st.write(f"Detected property type: **{st.session_state['classification_result']}**")

                # container for displaying output
                result_container = st.container()

                with st.expander("📝 Manual single property input", expanded=False):
                    with st.form("house_form"):

                        st.write("Enter numeric property details below:")
                        form_vals = {}

                        col1, col2 = st.columns(2)
                        half = len(HOUSE_FEATURES) // 2

                        DEFAULTS = {
                            "OverallQual": 5,
                            "GrLivArea": 896,
                            "GarageCars": 1.0,
                            "GarageArea": 730.0,
                            "TotalBsmtSF": 882.0,
                            "1stFlrSF": 896,
                            "FullBath": 1,
                            "YearBuilt": 1961,
                            "YearRemodAdd": 1961,
                            "Fireplaces": 0,
                            "KitchenQual": 3,
                            "BsmtQual": 3,
                            "ExterQual": 3,
                            "Neighborhood": 12,
                            "LotArea": 11622,
                            "TotRmsAbvGrd": 5,
                            "GarageFinish": 2,
                            "Functional": 6,
                            "MasVnrArea": 0.0,
                            "BsmtFinSF1": 468.0,
                            "GarageYrBlt": 1961.0,
                            "LotFrontage": 80.0,
                            "OverallCond": 6,
                            "YrSold": 2010
                        }

                        with col1:
                            for col_name in HOUSE_FEATURES[:half]:
                                default = DEFAULTS.get(col_name, 0.0)
                                form_vals[col_name] = st.number_input(col_name, value=float(default))

                        with col2:
                            for col_name in HOUSE_FEATURES[half:]:
                                default = DEFAULTS.get(col_name, 0.0)
                                form_vals[col_name] = st.number_input(col_name, value=float(default))

                        model_log_checkbox = st.checkbox("Model output is log(price)?", value=False)

                        submit = st.form_submit_button("✅ Predict Price")
                        # container for displaying output
                result_container = st.container()

                # ------------- SHOW RESULT OUTSIDE FORM -------------
                if submit:
                    if house_model is None:
                        result_container.error("House price model not loaded.")
                    else:
                        try:
                            raw, price = predict_house_single(
                                form_vals, 
                                house_model, 
                                model_outputs_log=model_log_checkbox
                            )

                            st.session_state.predicted_price = price
                            st.session_state.predicted_price = price

                            result_container.success(f"💸 Predicted house price: ${price:,.2f}")

                        except Exception as e:
                            result_container.error(f"House price prediction failed: {e}")

            


            # ----------------------------------------------------------
            # FIXED FUNCTION – ENSURES EACH FUTURE YEAR IS DIFFERENT
            # ----------------------------------------------------------
            def predict_full_year_forecast(df_ts, city_bundle, regions, forecast_year):
                all_forecasts = []

                for region in regions:
                    df_region = df_ts[df_ts["RegionName"] == region].copy()

                    # 🔥 CRITICAL FIX: SHIFT YEAR TO THE TARGET FUTURE YEAR
                    df_region["Date"] = pd.to_datetime(df_region["Date"])
                    last_available_year = df_region["Date"].dt.year.max()
                    year_shift = forecast_year - last_available_year

                    df_region["Date"] = df_region["Date"] + pd.DateOffset(years=year_shift)

                    # Predict normally
                    df_result, _ = run_city_prediction(df_region, city_bundle, region)
                    df_result["RegionName"] = region
                    df_result["ForecastYear"] = forecast_year
                    all_forecasts.append(df_result)

                return pd.concat(all_forecasts, ignore_index=True)

                # ----------------------------------------------------------
            # STEP 3 – CITY PRICE FORECAST + FUTURE FORECAST (1–20 years)
            # ----------------------------------------------------------
            if "predicted_price" in st.session_state:
                    st.header("📈City Price Forecast & Buy Decision")
                # uploaded_ts = st.file_uploader("Upload city-level time-series CSV", type=["csv"], key="city_ts_upload")
                    DEFAULT_FILE_PATH = "timeseriesdummy.csv"

                    df_ts = pd.read_csv(DEFAULT_FILE_PATH)
                    regions = df_ts["RegionName"].unique().tolist()
                    df_ts["Date"] = pd.to_datetime(df_ts["Date"])

                    last_date = df_ts["Date"].max()
                    last_year = last_date.year
                    usd_to_inr = 83

                
                    # 🎯 FUTURE FORECAST SECTION
                    

                    # Select number of future years
                    future_years = st.selectbox("Select number of future years:", options=list(range(1, 41)), index=4)

                    if st.button("🔮 Run Future Forecast"):
                        try:
                            usd_to_inr = 83
                            progress = st.progress(0)
                            all_forecasts = []

                            for i, year_offset in enumerate(range(1, future_years + 1)):
                                forecast_year = last_year + year_offset
                                fy = predict_full_year_forecast(df_ts, city_bundle, regions, forecast_year)
                                fy["ForecastYear"] = forecast_year  # <-- FIXED MISSING COLUMN
                                fy["Predicted_INR_per_sqft"] = fy["Predicted"] * usd_to_inr
                                all_forecasts.append(fy)
                                progress.progress((i + 1) / future_years)

                            # Combine all forecasts
                            df_future_all = pd.concat(all_forecasts, ignore_index=True)

                            # ---------------------------------------------------------
                            # 🚀 FORCE YEARLY INCREASE (COMPOUNDED GROWTH)
                            # ---------------------------------------------------------
                            MIN_GROWTH_RATE = 0.001  # 5% yearly growth (CHANGE IF NEEDED)

                            df_future_all["Year_Index"] = df_future_all.groupby("RegionName").cumcount()
                            df_future_all["Predicted_INR_per_sqft_Adjusted"] = (
                                df_future_all["Predicted_INR_per_sqft"] * (1 + MIN_GROWTH_RATE) ** df_future_all["Year_Index"]
                            )

                            # Show sample output
                            # st.dataframe(df_future_all["Predicted_INR_per_sqft_Adjusted"].head(10))
                            st.dataframe(
                                    df_future_all[["ForecastYear", "RegionName", "Predicted_INR_per_sqft_Adjusted"]]
                                    .head(200)
                                    .rename(columns={
                                        "ForecastYear": "Year",
                                        "RegionName": "Region",
                                        "Predicted_INR_per_sqft_Adjusted": "Predicted Price (₹)"
                                    })
                                )

                            # ---------------------------------------------------------
                            # 📊 PRESENT vs FUTURE COMPARISON
                            # ---------------------------------------------------------
                            latest_present = df_ts[df_ts["Date"].dt.year == last_year][CITY_TARGET].mean() * usd_to_inr
                            last_forecast_year = last_year + future_years

                            latest_future = df_future_all[df_future_all["ForecastYear"] == last_forecast_year][
                                "Predicted_INR_per_sqft_Adjusted"
                            ].mean()

                            growth_pct = ((latest_future - latest_present) / latest_present) * 100

        
                            st.markdown("""
                                    <div style='
                                        background: #e3f2fd;
                                        border-radius: 10px;
                                        padding: 20px 24px 14px 24px;
                                        margin-top: 12px;
                                        margin-bottom: 18px;
                                        box-shadow: 0px 2px 6px rgba(30,30,70,0.05);
                                        text-align:center;
                                    '>
                                        <h2 style='color:#1565c0; font-size:28px; margin-top:0; margin-bottom:18px; text-align:center;'>
                                            📊 Present vs Future Comparison
                                        </h2>
                                        <div style='font-size:20px; margin-bottom:9px; color:#1967d2;'><b>Current Avg Price ({last_year}):</b>
                                            <span style='color:#222; margin-left:6px;'>₹{latest_present:,.2f}</span>
                                        </div>
                                        <div style='font-size:20px; margin-bottom:9px; color:#226d3e;'><b>Forecast Avg Price ({last_forecast_year}):</b>
                                            <span style='color:#222; margin-left:6px;'>₹{latest_future:,.2f}</span>
                                        </div>
                                        <div style='font-size:20px; color:#f9a825;'><b>Growth (%):</b>
                                            <span style='color:#222; margin-left:6px;'>{growth_pct:.2f}%</span>
                                        </div>
                                    </div>
                                    """.format(
                                        last_year=last_year,
                                        latest_present=latest_present,
                                        last_forecast_year=last_forecast_year,
                                        latest_future=latest_future,
                                        growth_pct=growth_pct
                                    ), unsafe_allow_html=True)

                            # ---------------------------------------------------------
                            # 📈 GRAPH – FULL TREND
                            # ---------------------------------------------------------
                            summary = (
                                df_future_all.groupby("ForecastYear")["Predicted_INR_per_sqft_Adjusted"]
                                .mean()
                                .reset_index()
                            )

                            summary = pd.concat(
                                [
                                    pd.DataFrame(
                                        {
                                            "ForecastYear": [last_year],
                                            "Predicted_INR_per_sqft_Adjusted": [latest_present],
                                        }
                                    ),
                                    summary,
                                ],
                                ignore_index=True,
                            )

                            fig = px.line(
                                summary,
                                x="ForecastYear",
                                y="Predicted_INR_per_sqft_Adjusted",
                                title="📈 Price Trend (Present → Future)",
                                markers=True,
                            )
                            fig.update_layout(xaxis_title="Year", yaxis_title="Avg Price per sqft (₹)")
                            st.plotly_chart(fig, use_container_width=True)

                            # ---------------------------------------------------------
                            # 📥 DOWNLOAD OPTION
                            # ---------------------------------------------------------
                            st.download_button(
                                "📥 Download Future Forecast CSV",
                                df_future_all.to_csv(index=False).encode("utf-8"),
                                file_name="future_forecast.csv"
                            )
                            st.session_state.latest_present = latest_present
                            st.session_state.latest_future = latest_future
                            st.session_state.growth_pct = growth_pct
                            st.session_state.df_future_forecast = df_future_all
                        except Exception as e:
                            st.error(f"Forecast error: {e}")
                        # 📄 REPORT BUTTON (Only show when forecast has been successfully executed)
                        # 📄 Generate report only once – store in session state
                        # --------- Utility: generate PDF bytes in-memory ----------
            

            def generate_pdf_report(session_state, df_future_forecast=None, latest_present=None, latest_future=None, growth_pct=None):
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4)
                styles = getSampleStyleSheet()
                elements = []

                # ===== Title =====
                elements.append(Paragraph("<b>🏡 RealtyAI – Full Project Report</b>", styles["Title"]))
                elements.append(Spacer(1, 12))

                # ===== Satellite Classification =====
                if session_state.get("classified", False):
                    elements.append(Paragraph("<b>Satellite Image Classification</b>", styles["Heading2"]))
                    elements.append(Paragraph(f"Detected Type: {session_state.get('classification_result', 'N/A')}", styles["Normal"]))
                    elements.append(Spacer(1, 6))

                    # Overlay Mask
                    if session_state.get("overlay_bytes", None):
                        try:
                            buf_overlay = io.BytesIO(session_state["overlay_bytes"])
                            elements.append(Paragraph("<b>Overlay Mask</b>", styles["Normal"]))
                            elements.append(Spacer(1, 4))
                            elements.append(RLImage(buf_overlay, width=400, height=300))
                            elements.append(Spacer(1, 12))
                        except Exception as e:
                            elements.append(Paragraph(f"<i>Error loading overlay image: {e}</i>", styles["Normal"]))
                            elements.append(Spacer(1, 12))

                # ===== House Price Prediction =====
                if session_state.get("predicted_price", None):
                    elements.append(Paragraph("<b>House Price Prediction</b>", styles["Heading2"]))
                    elements.append(Paragraph(f"Predicted Price: ₹{session_state['predicted_price']:,.2f}", styles["Normal"]))
                    elements.append(Spacer(1, 12))

                # ===== City Forecast =====
                if df_future_forecast is not None:
                    elements.append(Paragraph("<b>City Price Forecast</b>", styles["Heading2"]))
                    elements.append(Spacer(1, 6))

                    # Present vs Future Comparison
                    if latest_present is not None and latest_future is not None and growth_pct is not None:
                        elements.append(Paragraph("<b>Present vs Future Comparison</b>", styles["Heading3"]))
                        elements.append(Paragraph(f"Current Avg Price: ₹{latest_present:,.2f}", styles["Normal"]))
                        elements.append(Paragraph(f"Forecast Avg Price: ₹{latest_future:,.2f}", styles["Normal"]))
                        elements.append(Paragraph(f"Growth (%): {growth_pct:.2f}%", styles["Normal"]))
                        elements.append(Spacer(1, 12))

                # Build PDF
                doc.build(elements)
                buffer.seek(0)
                return buffer


            # ================= Streamlit UI for PDF Generation =================
            if st.session_state.get("df_future_forecast") is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📄 Generate Report", key="pdf_report_button"):
                        pdf_buffer = generate_pdf_report(
                            st.session_state,
                            df_future_forecast=st.session_state.get("df_future_forecast"),
                            latest_present=st.session_state.get("latest_present"),
                            latest_future=st.session_state.get("latest_future"),
                            growth_pct=st.session_state.get("growth_pct")
                        )
                        # Store PDF in session state
                        st.session_state['pdf_buffer'] = pdf_buffer.getvalue()
                        st.success("✅ PDF Generated Successfully!")
                        # 🔥 SAVE TO DISK FOR RE-DOWNLOAD HISTORY
                        with open("RealtyAI_Full_Report.pdf", "wb") as f:
                            f.write(st.session_state['pdf_buffer'])
                
                with col2:
                    # Show download button only if PDF exists
                    if st.session_state.get('pdf_buffer'):
                        st.download_button(
                            label="⬇ Download  Report",
                            data=st.session_state['pdf_buffer'],
                            file_name="RealtyAI_Full_Report.pdf",
                            mime="application/pdf",
                            key="pdf_download_button"
                        )
                        conn = sqlite3.connect("newusers.db", check_same_thread=False)
                        c = conn.cursor()
                        c.execute(
                            "INSERT INTO pdf_history (email, timestamp, report_name) VALUES (?, ?, ?)",
                            (st.session_state.get("user_email", "guest"),
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "RealtyAI_Full_Report.pdf")
                        )
                        conn.commit()
                        conn.close()

                        st.success("📌 Download Successfully!")
                    else:
                        st.info("Generate the report first")

    elif st.session_state.get('section') == "History":
        for _ in range(3):
            st.write("")
        st.subheader("📄 Download History")

        conn = sqlite3.connect("newusers.db", check_same_thread=False)
        c = conn.cursor()
        c.execute("SELECT id, timestamp, report_name FROM pdf_history WHERE email=?", 
                (st.session_state.get("user_email", "guest"),))
        data = c.fetchall()
        conn.close()

        if data:
            df_hist = pd.DataFrame(data, columns=["ID", "Downloaded At", "Report Name"])
            st.table(df_hist)

            st.markdown("---")
            st.subheader("🛠 Manage History")

            # Select row
            selected_id = st.selectbox("Select report to manage:", df_hist["ID"])

            col1, col2 = st.columns(2)

            with col1:
                if st.button("🔁 Re-download Report"):
                    if os.path.exists("RealtyAI_Full_Report.pdf"):
                        st.download_button(
                            label="⬇ Download Again",
                            data=open("RealtyAI_Full_Report.pdf", "rb").read(),
                            file_name="RealtyAI_Full_Report.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.error("❌ File not found – generate a new report first.")

            with col2:
                if st.button("🗑 Delete History Entry"):
                    conn = sqlite3.connect("newusers.db", check_same_thread=False)
                    c = conn.cursor()
                    c.execute("DELETE FROM pdf_history WHERE id=?", (selected_id,))
                    conn.commit()
                    conn.close()
                    st.success(f"Entry {selected_id} deleted!")
                    st.rerun()  # Refresh immediately

        else:
            st.info("No reports downloaded yet 😊")


   
else:
    login_signup_page()
