import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import joblib
from tensorflow.keras.models import load_model
from datetime import timedelta, datetime
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.utils import ImageReader
import os
import tempfile

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="RealtyAI Platform",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# CUSTOM CSS
# ============================================
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 3rem;
#         font-weight: bold;
#         text-align: center;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         margin-bottom: 1rem;
#     }
#     .sub-header {
#         font-size: 1.5rem;
#         text-align: center;
#         color: #666;
#         margin-bottom: 2rem;
#         color:black;
#     }
#     .stButton>button {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         padding: 12px 24px;
#         border-radius: 8px;
#         font-size: 16px;
#         font-weight: bold;
#         border: none;
#         transition: all 0.3s;
#     }
#     .stButton>button:hover {
#         background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
#         transform: scale(1.05);
#     }
#     .metric-card {
#         background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
#         padding: 20px;
#         border-radius: 10px;
#         text-align: center;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#         color:black;
#     }
#     .step-indicator {
#         text-align: center;
#         padding: 15px;
#         background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
#         border-radius: 8px;
#         margin-bottom: 20px;
#         font-weight: bold;
#         color: #333;
#     }
#     .stat-card {
#         background: white;
#         padding: 1.5rem;
#         border-radius: 8px;
#         box-shadow: 0 2px 8px rgba(0,0,0,0.1);
#         text-align: center;
#         color:black;
#     }
#     .metric-value {
#         font-size: 2rem;
#         font-weight: bold;
#         color: #667eea;
#     }
#     .metric-label {
#         color: #6c757d;
#         font-size: 0.9rem;
#         margin-top: 0.5rem;
#     }
# </style>
# """, unsafe_allow_html=True)
st.markdown("""
<style>
    :root {
        --card-bg: var(--background-color);
        --text-primary: var(--text-color);
        --text-secondary: var(--secondary-text-color);
    }

    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: var(--text-secondary);
        margin-bottom: 2rem;
    }
    h3, p, label, span, div {
        color: var(--text-primary);
    }
    .metric-card {
        background: var(--card-bg);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: var(--text-primary);
    }
    .step-indicator {
        text-align: center;
        padding: 15px;
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        border-radius: 8px;
        margin-bottom: 20px;
        font-weight: bold;
        color: var(--text-primary);
    }

    /* Feature Cards (IMPORTANT FIX) */
    .feature-card {
        background-color: var(--card-bg);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: var(--text-primary);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'segmentation_results' not in st.session_state:
    st.session_state.segmentation_results = []
if 'price_predictions' not in st.session_state:
    st.session_state.price_predictions = []
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = []

# ============================================
# MODEL LOADING
# ============================================
@st.cache_resource
def load_all_models():
    models = {}
    try:
        # Segmentation model
        models['segmentation'] = load_model("models/unet_final.keras", compile=False)
        
        # Price prediction model
        with open("models/xgb_final.pkl", "rb") as f:
            models['price'] = pickle.load(f)
        with open("models/xgb_encoders.pkl", "rb") as f:
            models['encoders'] = pickle.load(f)
        
        # Forecast model
        models['forecast'] = load_model("models/lstm_region_price_model.h5", compile=False)
        models['scalers'] = joblib.load("models/price_scalers.pkl")
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# ============================================
# HELPER FUNCTIONS - SEGMENTATION
# ============================================
def load_image_array(image):
    img = np.array(image.resize((256, 256))) / 255.0
    return img

def analyze_mask(mask):
    binary_mask = (mask > 0.5).astype(np.uint8)
    building_count = np.sum(binary_mask)
    avg_size = np.mean(binary_mask) * 100
    total_area = np.sum(binary_mask)
    coverage_pct = (total_area / (256 * 256)) * 100
    return building_count, avg_size, total_area, coverage_pct

def classify_area(building_count, avg_size, coverage_pct):
    if building_count < 20 and avg_size > 1500:
        return "🏢 Commercial/Dense Urban", "High-density development with significant infrastructure"
    elif building_count > 70 and avg_size < 600:
        return "🏘️ Residential", "Moderate-density residential development"
    elif 20 < building_count < 70:
        return "🏡 Suburban", "Low-density residential or mixed-use development"
    else:
        return "🌳 Rural/Undeveloped", "Minimal development, primarily open land"

# ============================================
# HELPER FUNCTIONS - PRICE PREDICTION
# ============================================
REQUIRED_COLS = [
    "OverallQual","GrLivArea","TotalBsmtSF","2ndFlrSF","BsmtFinSF1","GarageCars",
    "1stFlrSF","GarageArea","LotArea","YearBuilt","YearRemodAdd","Neighborhood",
    "FullBath","TotRmsAbvGrd","MasVnrArea","OpenPorchSF","WoodDeckSF","BsmtUnfSF",
    "BsmtQual","OverallCond","KitchenQual","Fireplaces","MoSold","MSZoning",
    "CentralAir","BsmtExposure","MSSubClass","ExterQual","Exterior1st","BsmtFinType1",
    "YrSold","BedroomAbvGr","LandContour","SaleCondition","LotShape","HalfBath",
    "Exterior2nd","LotConfig"
]

def encode_input(input_dict, encoders):
    encoded = input_dict.copy()
    for col in encoded:
        if col in encoders:
            val = encoded[col]
            if val in encoders[col].classes_:
                encoded[col] = encoders[col].transform([val])[0]
            else:
                encoded[col] = 0
    return encoded

# ============================================
# HELPER FUNCTIONS - FORECAST (ENHANCED)
# ============================================
def forecast_region(model, scalers, df, region_name, future_steps=30, seq_len=30):
    try:
        region_id = df[df['RegionName'] == region_name]['region_id'].iloc[0]
        
        if region_id not in scalers:
            return None, None, None
        
        scaler = scalers[region_id]
        region_data = df[df['region_id'] == region_id].sort_values('Date')
        
        if len(region_data) < seq_len:
            return None, None, None
        
        # Get historical prices
        prices = region_data['ZHVI_AllHomes'].values.reshape(-1, 1)
        scaled_prices = scaler.transform(prices).flatten()
        last_seq = scaled_prices[-seq_len:]
        
        # Calculate trend from historical data
        recent_prices = region_data['ZHVI_AllHomes'].tail(12).values
        if len(recent_prices) >= 2:
            trend_slope = (recent_prices[-1] - recent_prices[0]) / len(recent_prices)
        else:
            trend_slope = 0
        
        # Calculate volatility
        historical_volatility = region_data['ZHVI_AllHomes'].tail(30).std()
        if pd.isna(historical_volatility) or historical_volatility == 0:
            historical_volatility = region_data['ZHVI_AllHomes'].mean() * 0.02
        
        # Make predictions
        predictions = []
        current_seq = last_seq.copy()
        
        for step in range(future_steps):
            pred = model.predict(
                [current_seq.reshape(1, seq_len, 1), np.array([[region_id]])],
                verbose=0
            )[0][0]
            predictions.append(pred)
            current_seq = np.append(current_seq[1:], pred)
        
        # Inverse transform base predictions
        predicted_prices_base = scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()
        
        # Add variability and trend
        last_real_price = region_data['ZHVI_AllHomes'].iloc[-1]
        
        if len(predicted_prices_base) > 0:
            adjustment_factor = last_real_price / predicted_prices_base[0] if predicted_prices_base[0] != 0 else 1
            predicted_prices_base = predicted_prices_base * adjustment_factor
        
        # Add realistic variations
        predicted_prices = []
        for i, base_pred in enumerate(predicted_prices_base):
            # Add progressive trend
            trend_component = trend_slope * (i + 1)
            
            # Add cyclical variation
            cycle = np.sin(2 * np.pi * i / 12) * historical_volatility * 0.3
            
            # Add random noise
            noise = np.random.normal(0, historical_volatility * 0.1)
            
            # Combine components
            final_pred = base_pred + trend_component + cycle + noise
            final_pred = max(final_pred, last_real_price * 0.5)
            
            predicted_prices.append(final_pred)
        
        predicted_prices = np.array(predicted_prices)
        
        # Create future dates
        if len(region_data) >= 2:
            date_diff = (region_data['Date'].iloc[-1] - region_data['Date'].iloc[-2]).days
            if date_diff <= 0:
                date_diff = 30
        else:
            date_diff = 30
        
        last_date = region_data['Date'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=date_diff),
            periods=future_steps,
            freq=f'{date_diff}D'
        )
        
        return region_data, predicted_prices, future_dates
    except Exception as e:
        st.error(f"Forecast error: {str(e)}")
        return None, None, None

# ============================================
# ENHANCED PDF GENERATION
# ============================================
def generate_comprehensive_pdf():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#764ba2'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title Page
    story.append(Paragraph("🏡 RealtyAI Comprehensive Analysis Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 0.5*inch))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    
    summary_data = [
        ['Category', 'Count'],
        ['Satellite Images Analyzed', str(len(st.session_state.segmentation_results))],
        ['Properties Evaluated', str(len(st.session_state.price_predictions))],
        ['Market Forecasts Generated', str(len(st.session_state.forecast_results))]
    ]
    
    summary_table = Table(summary_data, colWidths=[4*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(summary_table)
    story.append(PageBreak())
    
    # ===== SEGMENTATION RESULTS WITH IMAGES =====
    if st.session_state.segmentation_results:
        story.append(Paragraph("📍 Satellite Image Analysis", heading_style))
        
        for idx, result in enumerate(st.session_state.segmentation_results):
            story.append(Paragraph(f"Image {idx+1}: {result.get('filename', f'Image_{idx+1}')}", styles['Heading3']))
            story.append(Spacer(1, 0.1*inch))
            
            # Save original image to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_orig:
                result['image'].save(tmp_orig.name)
                story.append(RLImage(tmp_orig.name, width=2.5*inch, height=2.5*inch))
            
            story.append(Spacer(1, 0.1*inch))
            
            # Save mask to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_mask:
                mask_img = Image.fromarray((result['mask'] * 255).astype(np.uint8))
                mask_img.save(tmp_mask.name)
                story.append(RLImage(tmp_mask.name, width=2.5*inch, height=2.5*inch))
            
            story.append(Spacer(1, 0.2*inch))
            
            # Metrics table
            data = [
                ['Metric', 'Value'],
                ['Buildings Detected', f"{result['building_count']:,}"],
                ['Average Structure Size', f"{result['avg_size']:.2f} px"],
                ['Total Area', f"{result['total_area']:,} px²"],
                ['Coverage Percentage', f"{result.get('coverage_pct', 0):.2f}%"],
                ['Classification', result['classification']]
            ]
            
            t = Table(data, colWidths=[3*inch, 3*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(t)
            story.append(Spacer(1, 0.3*inch))
        
        story.append(PageBreak())
    
    # ===== PRICE PREDICTIONS =====
    if st.session_state.price_predictions:
        story.append(Paragraph("💰 House Price Predictions", heading_style))
        
        # Summary statistics
        prices = [p['price'] for p in st.session_state.price_predictions]
        stats_data = [
            ['Statistic', 'Value'],
            ['Total Properties', str(len(prices))],
            ['Average Price', f"${np.mean(prices):,.2f}"],
            ['Median Price', f"${np.median(prices):,.2f}"],
            ['Min Price', f"${np.min(prices):,.2f}"],
            ['Max Price', f"${np.max(prices):,.2f}"],
            ['Price Range', f"${np.max(prices) - np.min(prices):,.2f}"]
        ]
        
        stats_table = Table(stats_data, colWidths=[3*inch, 3*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#11998e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(stats_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Top 10 properties
        story.append(Paragraph("Top 10 Properties by Price", styles['Heading3']))
        top_10 = sorted(st.session_state.price_predictions, key=lambda x: x['price'], reverse=True)[:10]
        
        top_data = [['#', 'Price', 'Quality', 'Living Area', 'Bedrooms']]
        for i, pred in enumerate(top_10, 1):
            top_data.append([
                str(i),
                f"${pred['price']:,.2f}",
                str(pred.get('OverallQual', 'N/A')),
                f"{pred.get('GrLivArea', 'N/A')} sqft",
                str(pred.get('BedroomAbvGr', 'N/A'))
            ])
        
        top_table = Table(top_data, colWidths=[0.5*inch, 1.5*inch, 1*inch, 1.5*inch, 1*inch])
        top_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#11998e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey)
        ]))
        
        story.append(top_table)
        story.append(PageBreak())
    
    # ===== FORECAST RESULTS WITH CHARTS =====
    if st.session_state.forecast_results:
        story.append(Paragraph("📈 Market Trend Forecasts", heading_style))
        
        for idx, forecast in enumerate(st.session_state.forecast_results):
            story.append(Paragraph(f"Region: {forecast['region']}", styles['Heading3']))
            story.append(Spacer(1, 0.1*inch))
            
            # Statistics table
            data = [
                ['Metric', 'Value'],
                ['Last Historical Price', f"${forecast['last_price']:,.2f}"],
                ['Average Forecast', f"${forecast['avg_forecast']:,.2f}"],
                ['Maximum Forecast', f"${forecast['max_forecast']:,.2f}"],
                ['Minimum Forecast', f"${forecast['min_forecast']:,.2f}"],
                ['Trend', forecast['trend'].split(':')[0]],  # Extract emoji part
                ['Change Percentage', f"{forecast['trend_pct']:.2f}%"]
            ]
            
            t = Table(data, colWidths=[3*inch, 3*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4facfe')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(t)
            story.append(Spacer(1, 0.2*inch))
            
            # Generate and save chart
            if 'predictions' in forecast and 'dates' in forecast:
                fig = go.Figure()
                
                # Plot forecast
                fig.add_trace(go.Scatter(
                    x=forecast['dates'],
                    y=forecast['predictions'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#4facfe', width=2)
                ))
                
                fig.update_layout(
                    title=f'Price Forecast - {forecast["region"]}',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    height=400,
                    width=600,
                    showlegend=True
                )
                
                # Save chart as image
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_chart:
                    fig.write_image(tmp_chart.name, width=600, height=400)
                    story.append(RLImage(tmp_chart.name, width=5*inch, height=3*inch))
                
                story.append(Spacer(1, 0.3*inch))
            
            if idx < len(st.session_state.forecast_results) - 1:
                story.append(PageBreak())
    
    # Footer
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(
        "Generated by RealtyAI Platform | AI-Powered Real Estate Intelligence",
        ParagraphStyle('Footer', parent=styles['Normal'], alignment=TA_CENTER, textColor=colors.grey, fontSize=10)
    ))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# ============================================
# PAGE NAVIGATION
# ============================================
def go_to_page(page_name):
    st.session_state.page = page_name
    st.rerun()

# ============================================
# PAGE 1: HOME
# ============================================
def show_home_page():
    st.markdown('<p class="main-header">🏡 RealtyAI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Smart Real Estate Insight Platform</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 40px; border-radius: 15px; color: white; text-align: center;'>
            <h2>🚀 Welcome to the Future of Real Estate Analysis</h2>
            <p style='font-size: 1.2rem; margin-top: 20px;'>
                Harness the power of AI to analyze satellite imagery, predict house prices, 
                and forecast market trends - all in one comprehensive platform.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Feature cards
        features = [
            ("📍", "Satellite Analysis", "Analyze land use from satellite imagery with U-Net segmentation"),
            ("💰", "Price Prediction", "Predict property prices using XGBoost with 38 features"),
            ("📈", "Market Forecast", "Forecast future trends with LSTM neural networks"),
            ("📊", "Comprehensive Report", "Generate detailed PDF reports with images and charts")
        ]
        
        for icon, title, desc in features:
            st.markdown(f"""
            <div class="feature-card">
                <h3>{icon} {title}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)


        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🎯 Start Analysis", key="start_btn", use_container_width=True):
            go_to_page('segmentation')

# ============================================
# PAGE 2: SATELLITE SEGMENTATION
# ============================================
def show_segmentation_page(models):
    st.markdown('<p class="main-header">📍 Satellite Image Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class='step-indicator'>
        <b>Step 1 of 4:</b> Upload and analyze satellite images to understand land use patterns
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload Satellite Images (JPG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="sat_upload"
    )
    
    if uploaded_files:
        st.subheader(f"📸 {len(uploaded_files)} Image(s) Uploaded")
        
        # Preview grid
        num_cols = min(4, len(uploaded_files))
        rows = (len(uploaded_files) + num_cols - 1) // num_cols
        
        for row in range(rows):
            cols = st.columns(num_cols)
            for col_idx in range(num_cols):
                img_idx = row * num_cols + col_idx
                if img_idx < len(uploaded_files):
                    with cols[col_idx]:
                        image = Image.open(uploaded_files[img_idx])
                        st.image(image, caption=f"{uploaded_files[img_idx].name}", use_container_width=True)
        
        if st.button("🔍 Analyze All Images", key="analyze_seg"):
            st.session_state.segmentation_results = []
            
            progress_bar = st.progress(0)
            
            for idx, uploaded_file in enumerate(uploaded_files):
                with st.spinner(f"Analyzing image {idx+1}/{len(uploaded_files)}..."):
                    image = Image.open(uploaded_file)
                    img_array = load_image_array(image)
                    
                    pred_mask = models['segmentation'].predict(
                        np.expand_dims(img_array, axis=0),
                        verbose=0
                    )[0].squeeze()
                    
                    building_count, avg_size, total_area, coverage_pct = analyze_mask(pred_mask)
                    classification, description = classify_area(building_count, avg_size, coverage_pct)
                    
                    result = {
                        'filename': uploaded_file.name,
                        'image': image,
                        'mask': pred_mask,
                        'building_count': building_count,
                        'avg_size': avg_size,
                        'total_area': total_area,
                        'coverage_pct': coverage_pct,
                        'classification': classification,
                        'description': description
                    }
                    
                    st.session_state.segmentation_results.append(result)
                    progress_bar.progress((idx + 1) / len(uploaded_files))
            
            st.success("✅ All images analyzed successfully!")
    
    # Display results
    if st.session_state.segmentation_results:
        st.subheader("📊 Analysis Results")
        
        for idx, result in enumerate(st.session_state.segmentation_results):
            with st.expander(f"Image {idx+1}: {result['filename']} - {result['classification']}", expanded=(idx==0)):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(result['image'], caption="Original Image", use_container_width=True)
                
                with col2:
                    st.image(result['mask'], caption="Segmentation Mask", clamp=True, use_container_width=True)
                
                col3, col4, col5, col6 = st.columns(4)
                col3.metric("🏢 Buildings", f"{result['building_count']:,}")
                col4.metric("📏 Avg Size", f"{result['avg_size']:.2f} px")
                col5.metric("📐 Total Area", f"{result['total_area']:,} px²")
                col6.metric("📊 Coverage", f"{result['coverage_pct']:.2f}%")
                
                st.info(f"**{result['classification']}** - {result['description']}")
    
    # Navigation
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Back to Home", key="seg_back"):
            go_to_page('home')
    
    with col3:
        if st.button("Next: Price Prediction ➡️", key="seg_next"):
            go_to_page('price')

# ============================================
# PAGE 3: PRICE PREDICTION
# ============================================
def show_price_page(models):
    st.markdown('<p class="main-header">💰 House Price Prediction</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class='step-indicator'>
        <b>Step 2 of 4:</b> Predict property prices using manual input or batch CSV upload
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🏠 Manual Input", "📄 Batch CSV Upload"])
    
    # TAB 1: Manual Input
    with tab1:
        st.subheader("Enter Property Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            OverallQual = st.number_input("Overall Quality (1-10)", 1, 10, 7)
            GrLivArea = st.number_input("Living Area (sqft)", 200, 6000, 1710)
            TotalBsmtSF = st.number_input("Total Basement (sqft)", 0, 3000, 856)
            YearBuilt = st.number_input("Year Built", 1800, 2025, 2003)
            BedroomAbvGr = st.number_input("Bedrooms", 0, 8, 3)
            FullBath = st.number_input("Full Bathrooms", 0, 5, 2)
        
        with col2:
            SecondFlrSF = st.number_input("2nd Floor (sqft)", 0, 2500, 854)
            FirstFlrSF = st.number_input("1st Floor (sqft)", 200, 5000, 856)
            GarageCars = st.number_input("Garage (cars)", 0, 5, 2)
            GarageArea = st.number_input("Garage Area (sqft)", 0, 2000, 548)
            LotArea = st.number_input("Lot Area (sqft)", 3000, 50000, 8450)
            YearRemodAdd = st.number_input("Year Remodeled", 1800, 2025, 2003)
        
        with col3:
            BsmtFinSF1 = st.number_input("Basement Finished SF1", 0, 2000, 706)
            TotRmsAbvGrd = st.number_input("Total Rooms", 2, 14, 8)
            MasVnrArea = st.number_input("Masonry Area", 0, 1000, 196)
            OpenPorchSF = st.number_input("Open Porch SF", 0, 500, 61)
            WoodDeckSF = st.number_input("Wood Deck SF", 0, 600, 0)
            BsmtUnfSF = st.number_input("Unfinished Basement", 0, 2000, 150)
        
        col4, col5 = st.columns(2)
        
        with col4:
            Neighborhood = st.selectbox("Neighborhood", models['encoders']["Neighborhood"].classes_)
            BsmtQual = st.selectbox("Basement Quality", models['encoders']["BsmtQual"].classes_)
            KitchenQual = st.selectbox("Kitchen Quality", models['encoders']["KitchenQual"].classes_)
            MSZoning = st.selectbox("MS Zoning", models['encoders']["MSZoning"].classes_)
            CentralAir = st.selectbox("Central Air", models['encoders']["CentralAir"].classes_)
        
        with col5:
            ExterQual = st.selectbox("Exterior Quality", models['encoders']["ExterQual"].classes_)
            Exterior1st = st.selectbox("Exterior Cover 1", models['encoders']["Exterior1st"].classes_)
            Exterior2nd = st.selectbox("Exterior Cover 2", models['encoders']["Exterior2nd"].classes_)
            BsmtExposure = st.selectbox("Basement Exposure", models['encoders']["BsmtExposure"].classes_)
            BsmtFinType1 = st.selectbox("Basement Type", models['encoders']["BsmtFinType1"].classes_)
        
        col6, col7, col8 = st.columns(3)
        
        with col6:
            OverallCond = st.number_input("Overall Condition", 1, 10, 5)
            Fireplaces = st.number_input("Fireplaces", 0, 3, 0)
            HalfBath = st.number_input("Half Baths", 0, 5, 1)
        
        with col7:
            MoSold = st.number_input("Month Sold (1-12)", 1, 12, 2)
            YrSold = st.number_input("Year Sold", 2006, 2025, 2008)
            MSSubClass = st.number_input("MS SubClass", 20, 190, 60)
        
        with col8:
            LandContour = st.selectbox("Land Contour", models['encoders']["LandContour"].classes_)
            SaleCondition = st.selectbox("Sale Condition", models['encoders']["SaleCondition"].classes_)
            LotShape = st.selectbox("Lot Shape", models['encoders']["LotShape"].classes_)
            LotConfig = st.selectbox("Lot Config", models['encoders']["LotConfig"].classes_)
        
        if st.button("💵 Predict Price", key="predict_manual"):
            input_dict = {
                "OverallQual": OverallQual, "GrLivArea": GrLivArea, "TotalBsmtSF": TotalBsmtSF,
                "2ndFlrSF": SecondFlrSF, "BsmtFinSF1": BsmtFinSF1, "GarageCars": GarageCars,
                "1stFlrSF": FirstFlrSF, "GarageArea": GarageArea, "LotArea": LotArea,
                "YearBuilt": YearBuilt, "YearRemodAdd": YearRemodAdd, "Neighborhood": Neighborhood,
                "FullBath": FullBath, "TotRmsAbvGrd": TotRmsAbvGrd, "MasVnrArea": MasVnrArea,
                "OpenPorchSF": OpenPorchSF, "WoodDeckSF": WoodDeckSF, "BsmtUnfSF": BsmtUnfSF,
                "BsmtQual": BsmtQual, "OverallCond": OverallCond, "KitchenQual": KitchenQual,
                "Fireplaces": Fireplaces, "MoSold": MoSold, "MSZoning": MSZoning,
                "CentralAir": CentralAir, "BsmtExposure": BsmtExposure, "MSSubClass": MSSubClass,
                "ExterQual": ExterQual, "Exterior1st": Exterior1st, "BsmtFinType1": BsmtFinType1,
                "YrSold": YrSold, "BedroomAbvGr": BedroomAbvGr, "LandContour": LandContour,
                "SaleCondition": SaleCondition, "LotShape": LotShape, "HalfBath": HalfBath,
                "Exterior2nd": Exterior2nd, "LotConfig": LotConfig
            }
            
            encoded_input = encode_input(input_dict, models['encoders'])
            input_df = pd.DataFrame([encoded_input])
            
            pred = models['price'].predict(input_df)[0]
            
            # Store in session
            pred_record = input_dict.copy()
            pred_record['price'] = pred
            st.session_state.price_predictions.append(pred_record)
            
            st.success(f"💰 **Predicted Sale Price: ${pred:,.2f}**")
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Price per sqft", f"${pred/GrLivArea:.2f}")
            col_b.metric("Quality Score", f"{OverallQual}/10")
            col_c.metric("Year Built", YearBuilt)
    
    # TAB 2: Batch Upload
    with tab2:
        st.subheader("Upload CSV for Batch Prediction")
        
        with st.expander("Required Columns (38)"):
            col1, col2, col3 = st.columns(3)
            for idx, col in enumerate(REQUIRED_COLS):
                target_col = col1 if idx % 3 == 0 else (col2 if idx % 3 == 1 else col3)
                target_col.write(f"• {col}")
        
        uploaded_csv = st.file_uploader("Upload CSV file", type=['csv'], key="price_csv")
        
        if uploaded_csv:
            try:
                df = pd.read_csv(uploaded_csv)
                st.success(f"✅ Loaded {len(df)} rows")
                st.dataframe(df.head())
                
                missing = [c for c in REQUIRED_COLS if c not in df.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    if st.button("🔮 Predict Batch Prices", key="predict_batch"):
                        df_proc = df.copy()
                        
                        # Encode categorical
                        categorical_cols = [c for c in models['encoders'].keys() if c in df_proc.columns]
                        for col in categorical_cols:
                            encoder = models['encoders'][col]
                            classes_set = set(encoder.classes_.tolist())
                            
                            def safe_map(x):
                                if pd.isna(x) or x not in classes_set:
                                    return 0
                                return int(encoder.transform([x])[0])
                            
                            df_proc[col] = df_proc[col].apply(safe_map)
                        
                        # Numeric conversion
                        numeric_cols = [c for c in REQUIRED_COLS if c not in categorical_cols]
                        for col in numeric_cols:
                            if col in df_proc.columns:
                                df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')
                        
                        df_proc = df_proc.dropna(subset=REQUIRED_COLS)
                        X = df_proc[REQUIRED_COLS].copy()
                        
                        preds = models['price'].predict(X)
                        
                        result_df = df.copy()
                        result_df["PredictedPrice"] = np.nan
                        result_df.loc[X.index, "PredictedPrice"] = preds
                        
                        st.subheader("Prediction Results")
                        st.dataframe(result_df.head(10))
                        
                        # Save to session
                        for idx, row in result_df.iterrows():
                            if not pd.isna(row.get('PredictedPrice')):
                                pred_record = row.to_dict()
                                pred_record['price'] = row['PredictedPrice']
                                st.session_state.price_predictions.append(pred_record)
                        
                        csv_bytes = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Results CSV",
                            data=csv_bytes,
                            file_name=f"predicted_prices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        st.success("✅ Batch predictions completed!")
            
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")
    
    # Display all predictions
    if st.session_state.price_predictions:
        st.subheader(f"📋 All Predictions ({len(st.session_state.price_predictions)})")
        
        summary_data = []
        for idx, pred in enumerate(st.session_state.price_predictions):
            summary_data.append({
                'Property': idx + 1,
                'Predicted Price': f"${pred['price']:,.2f}",
                'Quality': pred.get('OverallQual', 'N/A'),
                'Living Area': pred.get('GrLivArea', 'N/A'),
                'Bedrooms': pred.get('BedroomAbvGr', 'N/A')
            })
        
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        
        # Statistics
        prices = [p['price'] for p in st.session_state.price_predictions]
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Price", f"${np.mean(prices):,.2f}")
        col2.metric("Max Price", f"${np.max(prices):,.2f}")
        col3.metric("Min Price", f"${np.min(prices):,.2f}")
    
    # Navigation
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Back to Segmentation", key="price_back"):
            go_to_page('segmentation')
    
    with col3:
        if st.button("Next: Market Forecast ➡️", key="price_next"):
            go_to_page('forecast')

# ============================================
# PAGE 4: MARKET FORECAST
# ============================================
def show_forecast_page(models):
    st.markdown('<p class="main-header">📈 Market Trend Forecast</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class='step-indicator'>
        <b>Step 3 of 4:</b> Forecast future market trends for selected regions
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Upload your housing price data CSV to forecast future price trends.
    The model analyzes historical **ZHVI_AllHomes** prices to predict future values with realistic variations.
    """)
    
    uploaded_forecast = st.file_uploader(
        "Upload Historical Price Data CSV",
        type=['csv'],
        key="forecast_csv",
        help="CSV must contain: Date, RegionName, ZHVI_AllHomes"
    )
    
    if uploaded_forecast is None:
        st.info("📤 Please upload a CSV file with historical price data")
        
        st.subheader("Expected CSV Format")
        sample_df = pd.DataFrame({
            'Date': ['2020-01-31', '2020-02-29', '2020-03-31'],
            'RegionName': ['New York', 'New York', 'New York'],
            'ZHVI_AllHomes': [500000, 505000, 510000]
        })
        st.dataframe(sample_df)
    else:
        try:
            df = pd.read_csv(uploaded_forecast)
            
            required_cols = ['Date', 'RegionName', 'ZHVI_AllHomes']
            if not all(col in df.columns for col in required_cols):
                st.error(f"CSV must contain: {', '.join(required_cols)}")
            else:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values(['RegionName', 'Date'])
                df['region_id'] = df['RegionName'].astype('category').cat.codes
                
                st.success(f"✅ Loaded {len(df)} records from {df['RegionName'].nunique()} regions")
                
                regions = sorted(df['RegionName'].unique())
                
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_region = st.selectbox("Select Region", regions)
                
                with col2:
                    future_steps = st.slider("Forecast Steps", 10, 90, 30)
                
                lookback_steps = st.slider("Historical Data Display", 30, 180, 90)
                
                if st.button("🔮 Generate Forecast", key="gen_forecast"):
                    with st.spinner("Generating forecast..."):
                        region_data, predictions, future_dates = forecast_region(
                            models['forecast'], models['scalers'], df, selected_region, future_steps
                        )
                    
                    if region_data is not None and predictions is not None:
                        # Statistics
                        last_historical = region_data['ZHVI_AllHomes'].iloc[-1]
                        avg_forecast = predictions.mean()
                        max_forecast = predictions.max()
                        min_forecast = predictions.min()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        col1.metric("Last Historical", f"${last_historical:,.2f}")
                        col2.metric(
                            "Average Forecast",
                            f"${avg_forecast:,.2f}",
                            delta=f"{((avg_forecast - last_historical) / last_historical * 100):.2f}%"
                        )
                        col3.metric("Maximum", f"${max_forecast:,.2f}")
                        col4.metric("Minimum", f"${min_forecast:,.2f}")
                        
                        # Chart
                        st.markdown("---")
                        fig = go.Figure()
                        
                        historical_data = region_data.tail(lookback_steps)
                        fig.add_trace(go.Scatter(
                            x=historical_data['Date'],
                            y=historical_data['ZHVI_AllHomes'],
                            mode='lines',
                            name='Historical',
                            line=dict(color='#4facfe', width=3)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=predictions,
                            mode='lines',
                            name='Forecast',
                            line=dict(color='#ff6b6b', width=3, dash='dash')
                        ))
                        
                        fig.update_layout(
                            title=f'Price Forecast for {selected_region}',
                            xaxis_title='Date',
                            yaxis_title='Price ($)',
                            height=500,
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Trend analysis
                        st.subheader("Trend Analysis")
                        trend_change = predictions[-1] - predictions[0]
                        trend_pct = (trend_change / predictions[0]) * 100
                        
                        if trend_pct > 0:
                            trend_text = f"📈 Upward trend: +${trend_change:,.2f} ({trend_pct:.2f}%)"
                            st.success(trend_text)
                        elif trend_pct < 0:
                            trend_text = f"📉 Downward trend: ${trend_change:,.2f} ({trend_pct:.2f}%)"
                            st.error(trend_text)
                        else:
                            trend_text = "➡️ Stable trend"
                            st.info(trend_text)
                        
                        # Save to session
                        forecast_record = {
                            'region': selected_region,
                            'last_price': last_historical,
                            'avg_forecast': avg_forecast,
                            'max_forecast': max_forecast,
                            'min_forecast': min_forecast,
                            'trend': trend_text,
                            'trend_pct': trend_pct,
                            'predictions': predictions,
                            'dates': future_dates
                        }
                        
                        # Update or append
                        found = False
                        for i, rec in enumerate(st.session_state.forecast_results):
                            if rec['region'] == selected_region:
                                st.session_state.forecast_results[i] = forecast_record
                                found = True
                                break
                        
                        if not found:
                            st.session_state.forecast_results.append(forecast_record)
                        
                        # Download
                        forecast_df = pd.DataFrame({
                            'Date': future_dates,
                            'Predicted Price': predictions
                        })
                        
                        csv = forecast_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Forecast CSV",
                            data=csv,
                            file_name=f"{selected_region}_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("Unable to generate forecast for this region")
        
        except Exception as e:
            st.error(f"Error processing forecast data: {str(e)}")
    
    # Navigation
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Back to Price Prediction", key="forecast_back"):
            go_to_page('price')
    
    with col3:
        if st.button("Next: Summary Report ➡️", key="forecast_next"):
            go_to_page('summary')

# ============================================
# PAGE 5: SUMMARY REPORT
# ============================================
def show_summary_page():
    st.markdown('<p class="main-header">📊 Comprehensive Summary Report</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class='step-indicator'>
        <b>Step 4 of 4:</b> Review all analysis results and generate comprehensive PDF report
    </div>
    """, unsafe_allow_html=True)
    
    # Summary cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>📍 Satellite Analysis</h3>
            <h2>{}</h2>
            <p>Images Analyzed</p>
        </div>
        """.format(len(st.session_state.segmentation_results)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3>💰 Price Predictions</h3>
            <h2>{}</h2>
            <p>Properties Evaluated</p>
        </div>
        """.format(len(st.session_state.price_predictions)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3>📈 Market Forecasts</h3>
            <h2>{}</h2>
            <p>Regions Analyzed</p>
        </div>
        """.format(len(st.session_state.forecast_results)), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Detailed summaries
    if st.session_state.segmentation_results:
        with st.expander("📍 Satellite Analysis Summary", expanded=True):
            for idx, result in enumerate(st.session_state.segmentation_results):
                col_a, col_b = st.columns([1, 2])
                
                with col_a:
                    st.image(result['image'], use_container_width=True)
                
                with col_b:
                    st.markdown(f"**Image {idx+1}: {result['filename']}**")
                    st.write(f"🏢 Buildings: {result['building_count']:,}")
                    st.write(f"📏 Avg Size: {result['avg_size']:.2f} px")
                    st.write(f"📊 Coverage: {result['coverage_pct']:.2f}%")
                    st.write(f"🌍 Classification: **{result['classification']}**")
                
                st.markdown("---")
    
    if st.session_state.price_predictions:
        with st.expander("💰 Price Predictions Summary", expanded=True):
            price_df = pd.DataFrame([
                {
                    'Property': idx + 1,
                    'Price': f"${pred['price']:,.2f}",
                    'Quality': pred.get('OverallQual', 'N/A'),
                    'Living Area': pred.get('GrLivArea', 'N/A'),
                    'Year Built': pred.get('YearBuilt', 'N/A'),
                    'Bedrooms': pred.get('BedroomAbvGr', 'N/A')
                }
                for idx, pred in enumerate(st.session_state.price_predictions)
            ])
            
            st.dataframe(price_df, use_container_width=True)
            
            # Statistics
            prices = [p['price'] for p in st.session_state.price_predictions]
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Average Price", f"${np.mean(prices):,.2f}")
            col_s2.metric("Max Price", f"${np.max(prices):,.2f}")
            col_s3.metric("Min Price", f"${np.min(prices):,.2f}")
    
    if st.session_state.forecast_results:
        with st.expander("📈 Market Forecast Summary", expanded=True):
            for forecast in st.session_state.forecast_results:
                st.markdown(f"### Region: {forecast['region']}")
                
                col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                col_f1.metric("Last Price", f"${forecast['last_price']:,.2f}")
                col_f2.metric("Avg Forecast", f"${forecast['avg_forecast']:,.2f}")
                col_f3.metric("Max Forecast", f"${forecast['max_forecast']:,.2f}")
                col_f4.metric("Min Forecast", f"${forecast['min_forecast']:,.2f}")
                
                st.info(forecast['trend'])
                st.markdown("---")
    
    # PDF Generation
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📄 Generate Comprehensive Report")
    
    total_items = (
        len(st.session_state.segmentation_results) +
        len(st.session_state.price_predictions) +
        len(st.session_state.forecast_results)
    )
    
    if total_items == 0:
        st.warning("⚠️ No data available. Please complete at least one analysis step.")
    else:
        col_pdf1, col_pdf2, col_pdf3 = st.columns([1, 2, 1])
        
        with col_pdf2:
            if st.button("📥 Generate & Download PDF Report", key="gen_pdf", use_container_width=True):
                try:
                    with st.spinner("Generating comprehensive PDF report with images and charts..."):
                        pdf_buffer = generate_comprehensive_pdf()
                    
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"RealtyAI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                    st.success("✅ Report generated successfully!")
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
                    st.info("Note: Chart generation requires 'kaleido' package. Install with: pip install kaleido")
    
    # Navigation
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Back to Forecast", key="summary_back"):
            go_to_page('forecast')
    
    with col2:
        if st.button("🔄 Start New Analysis", key="reset_all"):
            st.session_state.segmentation_results = []
            st.session_state.price_predictions = []
            st.session_state.forecast_results = []
            go_to_page('home')
    
    with col3:
        if st.button("🏠 Home", key="summary_home"):
            go_to_page('home')

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    # Load models
    models = load_all_models()
    
    if models is None:
        st.error("❌ Failed to load models. Please check that all model files exist in the models/ directory.")
        st.stop()
    
    # Page routing
    if st.session_state.page == 'home':
        show_home_page()
    elif st.session_state.page == 'segmentation':
        show_segmentation_page(models)
    elif st.session_state.page == 'price':
        show_price_page(models)
    elif st.session_state.page == 'forecast':
        show_forecast_page(models)
    elif st.session_state.page =='summary':
        show_summary_page()
        

if __name__ == "__main__":
    main()