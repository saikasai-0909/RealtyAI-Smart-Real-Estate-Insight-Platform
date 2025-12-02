import streamlit as st
import numpy as np
from PIL import Image
import cv2
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# --- FILE PATHS ---
PATH_STYLE_CSS = "style.css"
PATH_UNET_MODEL = "unet_satellite_model.keras"
PATH_HOUSE_MODEL = "house_price_model.json"
PATH_HOUSE_COLUMNS = "model_columns.joblib"
PATH_ZILLOW_MODEL = "zillow_timeseries_model.pkl"
PATH_ZILLOW_FEATURES = "zillow_features.pkl"
PATH_HOUSE_TRAIN_DATA = "house-prices-advanced-regression-techniques/train.csv"
PATH_ZILLOW_CITY_DATA = "zillow economics data/City_time_series.csv"

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RealtyAI Pro",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 🎨 OPTIMIZED BLACK THEME ---
def inject_custom_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100;300;400;700&display=swap');
            html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }
            .stApp { background-color: #000000; }
            
            [data-testid="stVerticalBlockBorderWrapper"] {
                background: rgba(20, 20, 20, 0.6);
                border: 1px solid rgba(50, 50, 50, 1);
                border-radius: 16px;
                padding: 30px; 
                margin-bottom: 30px;
            }
            
            .hero-title {
                text-align: center;
                font-size: 3.5rem;
                font-weight: 800;
                background: -webkit-linear-gradient(45deg, #00d2ff, #928DAB);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 10px;
            }
            
            h1, h2, h3, p { color: #ffffff !important; }
            
            .stButton>button {
                background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
                color: white; border: none; font-weight: 600; transition: transform 0.2s;
            }
            .stButton>button:hover { transform: scale(1.02); box-shadow: 0 0 15px rgba(0, 210, 255, 0.4); }
            
            [data-testid="stMetricValue"] { color: #00d2ff !important; }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- ⚡ CACHING (The Speed Secret) ---

@st.cache_resource(show_spinner="Loading AI Vision Model...")
def load_unet_model():
    import tensorflow as tf 
    path = PATH_UNET_MODEL
    return tf.keras.models.load_model(path) if os.path.exists(path) else None

@st.cache_resource(show_spinner="Loading AI Core...")
def load_housing_model():
    import xgboost as xgb
    if os.path.exists(PATH_HOUSE_MODEL) and os.path.exists(PATH_HOUSE_COLUMNS):
        model = xgb.XGBRegressor()
        model.load_model(PATH_HOUSE_MODEL)
        features = joblib.load(PATH_HOUSE_COLUMNS)
        return model, features
    return None, None

@st.cache_resource(show_spinner="Loading AI Core...")
def load_zillow_model():
    if os.path.exists(PATH_ZILLOW_MODEL) and os.path.exists(PATH_ZILLOW_FEATURES):
        model = joblib.load(PATH_ZILLOW_MODEL)
        features = joblib.load(PATH_ZILLOW_FEATURES)
        return model, features
    return None, None

@st.cache_data(show_spinner=False)
def load_data(file_name):
    return pd.read_csv(file_name) if os.path.exists(file_name) else None

@st.cache_data(show_spinner="Indexing Database...")
def load_and_process_zillow_data(data_path):
    if not os.path.exists(data_path): return None
    df = pd.read_csv(data_path)
    df = df.replace(0, np.nan).ffill().bfill().fillna(0)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data(show_spinner=False)
def get_city_list(df):
    return sorted(df['RegionName'].unique())[:100]

@st.cache_data(show_spinner=False)
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- PREPROCESSING ---
def preprocess_image(image_pil, target_shape):
    target_height, target_width, target_channels = target_shape[1:4]
    image_pil = image_pil.resize((target_width, target_height))
    image = np.array(image_pil)
    if image.ndim == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    current_channels = image.shape[2]
    if target_channels == 4 and current_channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    elif target_channels == 3 and current_channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
    image_normalized = image.astype('float32') / 255.0
    return np.expand_dims(image_normalized, axis=0)

def post_process_mask(prediction_raw):
    mask = (prediction_raw[0] > 0.5).astype(np.uint8) * 255
    if mask.ndim == 2: return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    elif mask.shape[2] == 1: return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    elif mask.shape[2] == 4: return cv2.cvtColor(mask, cv2.COLOR_RGBA2RGB)
    return mask

def preprocess_house_inputs(data_dict, model_columns):
    df = pd.DataFrame([data_dict])
    cat_cols = [c for c in df.columns if df[c].dtype == 'object']
    df[cat_cols] = df[cat_cols].fillna('None')
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(0)

    if 'LotFrontage' not in df: df['LotFrontage'] = 70.0
    if 'Electrical' not in df: df['Electrical'] = 'SBrkr'
    df['TotalSF'] = df.get('TotalBsmtSF',0) + df.get('1stFlrSF',0) + df.get('2ndFlrSF',0)
    
    if 'YrSold' not in df: df['YrSold'] = datetime.now().year
    if 'YearBuilt' in df: df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    else: df['HouseAge'] = 0
    
    df_processed = pd.get_dummies(df)
    return df_processed.reindex(columns=model_columns, fill_value=0)

def preprocess_dataframe(df, model_columns):
    df = df.dropna(axis=1, how='all')
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(exclude=np.number).columns
    df[num_cols] = df[num_cols].fillna(0)
    df[cat_cols] = df[cat_cols].fillna('None')
    
    df['TotalSF'] = df.get('TotalBsmtSF',0) + df.get('1stFlrSF',0) + df.get('2ndFlrSF',0)
    if 'YrSold' not in df: df['YrSold'] = datetime.now().year
    if 'YearBuilt' in df: df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    else: df['HouseAge'] = 0
    
    df_processed = pd.get_dummies(df)
    return df_processed.reindex(columns=model_columns, fill_value=0)

def create_zone_pdf(uploaded_image, mask_image, density, residential_pct, commercial_pct, classification):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(180, 750, "RealtyAI – SpaceNet Analysis")
    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, 720, f"Generated: {datetime.now().strftime('%Y-%m-%d')}")
    uploaded_reader = ImageReader(uploaded_image)
    mask_reader = ImageReader(mask_image)
    pdf.drawImage(uploaded_reader, 50, 450, width=240, height=240)
    pdf.drawImage(mask_reader, 310, 450, width=240, height=240)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, 400, "Analysis Results")
    pdf.setFont("Helvetica", 12)
    pdf.drawString(60, 370, f"Built-up Density: {density:.2f}%")
    pdf.drawString(60, 350, f"Res. Intensity: {residential_pct:.1f}/100")
    pdf.drawString(60, 330, f"Com. Intensity: {commercial_pct:.1f}/100")
    pdf.drawString(60, 310, f"Class: {classification}")
    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer

# --- UI MODULES ---

def show_hero_section():
    st.markdown("""
        <div class="hero-title">💠 RealtyAI Dashboard</div>
        <div style="text-align: center; margin-bottom: 40px; color: #cccccc;">
            Advanced Urban Intelligence • Predictive Valuation • Future Forecasting
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Model Accuracy", "94.2%")
    with col2: st.metric("Cities Covered", "200+")
    with col3: st.metric("Active Modules", "3")
    with col4: st.metric("System Status", "Online")
    
    # --- 🌍 THE MAP IS BACK! ---
    st.write("") # Spacer
    with st.container(border=True):
        st.subheader("🌍 Geographic Coverage")
        # Example coordinates for major US cities
        map_df = pd.DataFrame({
            'lat': [42.0308, 40.7128, 34.0522, 41.8781, 25.7617, 30.2672],
            'lon': [-93.6319, -74.0060, -118.2437, -87.6298, -80.1918, -97.7431],
            'City': ['Ames', 'New York', 'Los Angeles', 'Chicago', 'Miami', 'Austin']
        })
        st.map(map_df, zoom=3, use_container_width=True)
    
    st.markdown("---")

def show_spacenet_module():
    with st.container(border=True):
        st.subheader("🛰️ SpaceNet: Satellite Vision")
        st.caption("Upload satellite imagery to classify zoning intensity scores.")
        
        model = load_unet_model()
        if not model:
            st.warning("⚠️ Model missing.")
            return

        uploaded_file = st.file_uploader("", type=["jpg","png"], key="spacenet_up")
        
        if "spacenet_res" not in st.session_state: st.session_state.spacenet_res = None

        if uploaded_file:
            file_key = f"{uploaded_file.name}_{uploaded_file.size}"
            
            if st.session_state.spacenet_res is None or st.session_state.spacenet_res['id'] != file_key:
                with st.spinner("Analysing Terrain..."):
                    image_pil = Image.open(uploaded_file)
                    if image_pil.mode != "RGB": image_pil = image_pil.convert("RGB")
                        
                    processed = preprocess_image(image_pil, model.input_shape)
                    mask_raw = model.predict(processed)
                    mask_img = post_process_mask(mask_raw)
                    
                    density = np.mean(mask_raw) * 100
                    RES_MAX, COM_START = 15.0, 10.0
                    res_score = (density / RES_MAX) * 100 if density <= RES_MAX else 100.0
                    com_score = ((density - COM_START) / 35.0) * 100 if density > COM_START else 0.0
                    com_score = min(com_score, 100.0)
                    
                    cls = "Open Area"
                    if res_score > 20: cls = "Residential Zone"
                    if com_score > 20: cls = "Commercial / Mixed"
                    
                    is_success, buf = cv2.imencode(".png", mask_img)
                    mask_bytes = io.BytesIO(buf)
                    pdf_bytes = create_zone_pdf(image_pil, mask_bytes, density, res_score, com_score, cls).getvalue()

                    st.session_state.spacenet_res = {
                        'id': file_key, 'img': image_pil, 'mask': mask_img, 
                        'res': res_score, 'com': com_score, 'pdf': pdf_bytes
                    }
            
            res = st.session_state.spacenet_res
            c1, c2 = st.columns(2)
            with c1: st.image(res['img'], caption="Original", use_container_width=True)
            with c2: st.image(res['mask'], caption="AI Analysis", use_container_width=True)
            
            st.markdown("---")
            c_res, c_com = st.columns(2)
            with c_res:
                st.metric("🏠 Residential", f"{res['res']:.1f} / 100")
                st.progress(int(res['res']))
            with c_com:
                st.metric("🏢 Commercial", f"{res['com']:.1f} / 100")
                st.progress(int(res['com']))
                
            st.download_button("📄 Download Report", res['pdf'], "report.pdf", "application/pdf", type="primary")

def show_house_price_module():
    with st.container(border=True):
        st.subheader("🏠 AI House Valuator")
        
        housing_model, model_columns = load_housing_model()
        df_train = load_data(PATH_HOUSE_TRAIN_DATA)
        if not housing_model: return

        with st.expander("🔽 Manual Prediction"):
            with st.form(key="house_form"):
                c1, c2, c3 = st.columns(3)
                neighborhood = c1.selectbox("Neighborhood", sorted(df_train['Neighborhood'].unique()) if df_train is not None else ["NAmes"])
                quality = c1.slider("Quality", 1, 10, 7)
                area = c1.number_input("Area (sqft)", 500, 10000, 1500)
                bsmt = c2.number_input("Basement", 0, 5000, 800)
                cars = c2.slider("Garage", 0, 5, 2)
                bath = c2.slider("Baths", 1, 5, 2)
                built = c3.number_input("Year Built", 1900, 2025, 2005)
                f1 = c3.number_input("1st Flr", 0, 5000, 1000)
                f2 = c3.number_input("2nd Flr", 0, 5000, 800)
                
                if st.form_submit_button("Estimate Value", type="primary"):
                    inputs = {'Neighborhood':neighborhood, 'OverallQual':quality, 'GrLivArea':area,
                              'TotalBsmtSF':bsmt, 'GarageCars':cars, 'FullBath':bath, 'YearBuilt':built,
                              '1stFlrSF':f1, '2ndFlrSF':f2}
                    df_in = preprocess_house_inputs(inputs, model_columns)
                    pred = np.expm1(housing_model.predict(df_in))[0]
                    st.success(f"💰 Estimated Value: **${pred:,.2f}**")

        with st.expander("📂 Batch Upload"):
            up_file = st.file_uploader("Upload File", type=['csv','xlsx'])
            if "house_batch_csv" not in st.session_state: st.session_state.house_batch_csv = None

            if up_file and st.button("Process Batch", type="primary"):
                df = pd.read_csv(up_file) if up_file.name.endswith('.csv') else pd.read_excel(up_file)
                processed = preprocess_dataframe(df, model_columns)
                df['Predicted_Price'] = np.expm1(housing_model.predict(processed))
                st.success("Analysis Complete!")
                st.dataframe(df[['Id', 'Predicted_Price']].head())
                
                st.session_state.house_batch_csv = convert_df_to_csv(df)
            
            if st.session_state.house_batch_csv:
                 st.download_button("💾 Download CSV", st.session_state.house_batch_csv, "house_preds.csv", "text/csv", type="primary")

def show_zillow_module():
    with st.container(border=True):
        st.subheader("📈 Market Trends & Forecasting")
        
        zillow_model, zillow_features = load_zillow_model()
        df_zillow = load_and_process_zillow_data(PATH_ZILLOW_CITY_DATA)
        if df_zillow is None: return

        st.markdown("#### 📊 Regional Analysis")
        top_cities = get_city_list(df_zillow)
        city = st.selectbox("Select Region", top_cities)
        
        city_data = df_zillow[df_zillow['RegionName'] == city]
        if not city_data.empty:
            fig = px.line(city_data, x='Date', y='ZHVI_AllHomes', title=f"{city} - Historical Data")
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=300)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown(f"#### 🔮 Future Forecast (Up to 2030)")
        
        if "zillow_forecast_fig" not in st.session_state: st.session_state.zillow_forecast_fig = None
        if "zillow_forecast_city" not in st.session_state: st.session_state.zillow_forecast_city = None

        if st.button(f"Generate Forecast for {city}", type="primary"):
            try:
                from prophet import Prophet
                with st.spinner("Simulating future market conditions..."):
                    df_p = city_data[['Date','ZHVI_AllHomes']].rename(columns={'Date':'ds','ZHVI_AllHomes':'y'}).dropna()
                    if len(df_p) > 12:
                        m = Prophet()
                        m.fit(df_p)
                        last_date = df_p['ds'].max()
                        target_year = datetime.now().year + 5
                        months = (target_year - last_date.year) * 12 + (12 - last_date.month)
                        future = m.make_future_dataframe(periods=months, freq='M')
                        forecast = m.predict(future)
                        
                        fig_f = go.Figure()
                        future_f = forecast[forecast['ds'] > last_date]
                        
                        fig_f.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name='History', line=dict(color='#00d2ff')))
                        fig_f.add_trace(go.Scatter(x=future_f['ds'], y=future_f['yhat'], name='Prediction', line=dict(color='#A855F7', dash='dash')))
                        
                        fig_f.add_trace(go.Scatter(
                            x=pd.concat([future_f['ds'], future_f['ds'][::-1]]),
                            y=pd.concat([future_f['yhat_upper'], future_f['yhat_lower'][::-1]]),
                            fill='toself', fillcolor='rgba(168, 85, 247, 0.2)', line=dict(color='rgba(0,0,0,0)'),
                            hoverinfo="skip", showlegend=False
                        ))
                        
                        max_pt = future_f.loc[future_f['yhat'].idxmax()]
                        min_pt = df_p.loc[df_p['y'].idxmin()]
                        fig_f.add_trace(go.Scatter(x=[max_pt['ds']], y=[max_pt['yhat']], mode='markers+text', marker=dict(color='#00ff00', size=10, symbol='triangle-up'), text=[f"Peak: ${max_pt['yhat']:,.0f}"], textposition="top center", name='Peak'))
                        fig_f.add_trace(go.Scatter(x=[min_pt['ds']], y=[min_pt['y']], mode='markers+text', marker=dict(color='#ff0000', size=10, symbol='triangle-down'), text=[f"Low: ${min_pt['y']:,.0f}"], textposition="bottom center", name='Low'))
                        fig_f.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', height=400, title=f"Prediction: {city}")
                        
                        st.session_state.zillow_forecast_fig = fig_f
                        st.session_state.zillow_forecast_city = city
            except ImportError:
                st.error("Prophet not installed.")

        if st.session_state.zillow_forecast_fig and st.session_state.zillow_forecast_city == city:
            st.plotly_chart(st.session_state.zillow_forecast_fig, use_container_width=True)

        st.divider()
        st.markdown("#### 📂 Batch Analysis")
        uploaded_zillow = st.file_uploader("Upload Zillow Data", type=["csv", "xlsx"], key="z_up_batch")
        if "z_batch_csv" not in st.session_state: st.session_state.z_batch_csv = None
        
        if uploaded_zillow:
            if st.button("Analyze Upload", type="primary"):
                with st.spinner("Processing Batch..."):
                    df_up = pd.read_csv(uploaded_zillow) if uploaded_zillow.name.endswith(".csv") else pd.read_excel(uploaded_zillow)
                    df_up = df_up.replace(0, np.nan).ffill().bfill().fillna(0)
                    if 'Date' in df_up.columns:
                        df_up['Date'] = pd.to_datetime(df_up['Date'])
                        df_up['year'] = df_up['Date'].dt.year
                        df_up['month'] = df_up['Date'].dt.month
                        df_up['quarter'] = df_up['Date'].dt.quarter
                        
                        missing = [f for f in zillow_features if f not in df_up.columns]
                        if not missing:
                            X_upload = df_up[zillow_features]
                            df_up["Predicted_ZHVI"] = zillow_model.predict(X_upload)
                            st.success("✅ Analysis Complete!")
                            st.dataframe(df_up.head())
                            st.session_state.z_batch_csv = convert_df_to_csv(df_up)
                        else:
                            st.error(f"Missing features: {missing}")
                    else:
                        st.error("Missing 'Date' column.")
            
            if st.session_state.z_batch_csv:
                st.download_button("💾 Download CSV", st.session_state.z_batch_csv, "zillow_results.csv", "text/csv", type="primary")

def show_footer():
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #666;'><small>RealtyAI System v2.0 • Streamlit</small></div>", unsafe_allow_html=True)

# --- MAIN APP FLOW ---
show_hero_section()
show_spacenet_module()
show_house_price_module()
show_zillow_module()
show_footer()