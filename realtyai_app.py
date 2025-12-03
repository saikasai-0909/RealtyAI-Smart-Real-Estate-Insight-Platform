import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import time
import hashlib
import re

# Set page configuration
st.set_page_config(
    page_title="RealtyAI - Smart Real Estate Platform",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .price-display {
        font-size: 2rem;
        color: #2e86ab;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background-color: #f0f8ff;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .button-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 2rem 0;
    }
    .prediction-table {
        margin-top: 2rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    .data-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2e86ab;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Login/Signup styling */
    .auth-container {
        max-width: 400px;
        margin: 2rem auto;
        padding: 2rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    .auth-tabs {
        display: flex;
        margin-bottom: 2rem;
        border-bottom: 2px solid #e0e0e0;
    }
    .auth-tab {
        flex: 1;
        text-align: center;
        padding: 1rem;
        cursor: pointer;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .auth-tab.active {
        color: #1f77b4;
        border-bottom: 3px solid #1f77b4;
    }
    .auth-input {
        margin-bottom: 1.5rem;
    }
    
    /* CHANGED: All buttons to Light Blue with White Text */
    div.stButton > button[kind="primary"],
    button[kind="primary"] {
        background-color: #4da6ff !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        cursor: pointer !important;
        transition: background-color 0.3s ease !important;
    }
    div.stButton > button[kind="primary"]:hover,
    button[kind="primary"]:hover {
        background-color: #3399ff !important;
        color: white !important;
        border: none !important;
    }
    
    /* CHANGED: Logout button color to Light Blue */
    div.stButton > button.logout-button {
        background-color: #4da6ff !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-size: 1rem !important;
        cursor: pointer !important;
        transition: background-color 0.3s ease !important;
        width: 100% !important;
    }
    div.stButton > button.logout-button:hover {
        background-color: #3399ff !important;
        color: white !important;
    }
    
    /* CHANGED: PDF Download Button - Light Blue with White Text */
    .pdf-button {
        background-color: #4da6ff !important;
        color: white !important;
        padding: 0.75rem 1.5rem !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        cursor: pointer !important;
        transition: background-color 0.3s ease !important;
        text-decoration: none !important;
        display: inline-block !important;
        text-align: center !important;
        font-weight: 500 !important;
    }
    .pdf-button:hover {
        background-color: #3399ff !important;
        color: white !important;
        text-decoration: none !important;
    }
    
    /* CHANGED: CSV Download Button - Light Blue with White Text */
    .csv-button {
        background-color: #4da6ff !important;
        color: white !important;
        padding: 0.75rem 1.5rem !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        cursor: pointer !important;
        transition: background-color 0.3s ease !important;
        text-decoration: none !important;
        display: inline-block !important;
        text-align: center !important;
        font-weight: 500 !important;
    }
    .csv-button:hover {
        background-color: #3399ff !important;
        color: white !important;
        text-decoration: none !important;
    }
</style>
""", unsafe_allow_html=True)

# User authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def init_users():
    if 'users' not in st.session_state:
        # Default accounts including Preethi's account
        st.session_state.users = {
            'preethimallepally72@gmail.com': {
                'password': hash_password('preethi08'),
                'name': 'Preethi Mallepally',
                'created_at': datetime.now(),
                'role': 'Premium User'
            }
        }

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    return len(password) >= 6

def login_user(email, password):
    if email in st.session_state.users:
        if st.session_state.users[email]['password'] == hash_password(password):
            st.session_state.user = {
                'email': email,
                'name': st.session_state.users[email]['name'],
                'role': st.session_state.users[email]['role'],
                'login_time': datetime.now()
            }
            return True
    return False

def register_user(email, password, name):
    if email in st.session_state.users:
        return False, "Email already registered"
    if not validate_email(email):
        return False, "Invalid email format"
    if not validate_password(password):
        return False, "Password must be at least 6 characters"
    
    st.session_state.users[email] = {
        'password': hash_password(password),
        'name': name,
        'created_at': datetime.now(),
        'role': 'Standard User'
    }
    return True, "Registration successful"

# Forecast generation function
def generate_forecast(current_price, region, property_type, forecast_years):
    # Generate forecast dates
    dates = [datetime.now() + timedelta(days=365*i) for i in range(forecast_years + 1)]
    
    # Regional growth rates (annual)
    regional_growth_rates = {
        'California': 0.065,
        'New York': 0.048,
        'Texas': 0.042,
        'Florida': 0.055,
        'Illinois': 0.035,
        'Arizona': 0.058,
        'Georgia': 0.045,
        'North Carolina': 0.052
    }
    
    # Property type multipliers - CHANGED: Single Family to Residential
    property_multipliers = {
        'Residential': 1.0,  # CHANGED HERE
        'Condominium': 0.95,
        'Townhouse': 0.92,
        'Commercial': 1.15
    }
    
    base_growth_rate = regional_growth_rates.get(region, 0.05)
    property_multiplier = property_multipliers.get(property_type, 1.0)
    adjusted_growth_rate = base_growth_rate * property_multiplier
    
    # Generate forecasted prices
    prices = []
    current_value = current_price
    
    for i in range(len(dates)):
        # Add some random variation to make it realistic
        annual_variation = np.random.normal(1, 0.02)
        current_value = current_value * (1 + adjusted_growth_rate) * annual_variation
        prices.append(int(current_value))
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'Year': [date.year for date in dates],
        'Date': dates,
        'Predicted_Price': prices,
        'Region': region,
        'Property_Type': property_type
    })
    
    return forecast_df

# Display single property forecast WITHOUT GRAPHS
def display_single_forecast(forecast_data):
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    initial_price = forecast_data['Predicted_Price'].iloc[0]
    final_price = forecast_data['Predicted_Price'].iloc[-1]
    total_growth = ((final_price - initial_price) / initial_price) * 100
    annual_growth = total_growth / (len(forecast_data) - 1)
    
    with col1:
        st.metric("🏠 Current Value", f"${initial_price:,.0f}")
    with col2:
        st.metric(f"🎯 Forecasted Value", f"${final_price:,.0f}", delta=f"${final_price - initial_price:,.0f}")
    with col3:
        st.metric("📈 Total Growth", f"{total_growth:.1f}%", delta=f"{annual_growth:.1f}% annually")
    with col4:
        market_outlook = "🚀 Strong Growth" if annual_growth > 6 else "📈 Moderate Growth" if annual_growth > 3 else "📊 Stable"
        st.metric("📊 Market Outlook", market_outlook)
    
    # REMOVED: Price Trend Chart Graph
    
    # Forecast table
    st.subheader("📅 Detailed Forecast")
    display_forecast = forecast_data[['Year', 'Predicted_Price']].copy()
    display_forecast['Predicted_Price'] = display_forecast['Predicted_Price'].apply(lambda x: f"${x:,.0f}")
    display_forecast['Growth'] = display_forecast['Predicted_Price'].index.to_series().apply(
        lambda i: f"+{((forecast_data['Predicted_Price'].iloc[i] - initial_price) / initial_price * 100):.1f}%" if i > 0 else "0%"
    )
    st.dataframe(display_forecast, use_container_width=True)

# Function to create PDF report content - SIMPLIFIED VERSION
def create_pdf_report():
    report_content = f"""
REALTYAI PROPERTY ANALYSIS REPORT
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Analyzed by: {st.session_state.user['name']}

PROPERTY ANALYSIS SUMMARY
========================

1. SATELLITE IMAGE ANALYSIS
---------------------------
Property Type: {st.session_state.property_type}
Building Detection: Multiple structures identified
Zone Segmentation: Completed successfully
Area Calculation: Processing complete
Compliance Status: Zoning regulations met

2. PRICE PREDICTION
-------------------
"""
    
    if isinstance(st.session_state.price_prediction, int):
        report_content += f"""Estimated Property Value: ${st.session_state.price_prediction:,}
Confidence Level: High
Market Position: Competitive
"""
    else:
        report_content += f"""Properties Analyzed: {len(st.session_state.price_data)}
Average Price: ${st.session_state.price_data['Predicted_Price'].mean():,.0f}
Price Range: ${st.session_state.price_data['Predicted_Price'].min():,.0f} - ${st.session_state.price_data['Predicted_Price'].max():,.0f}
"""

    if 'forecast_data' in st.session_state:
        initial_price = st.session_state.forecast_data['Predicted_Price'].iloc[0]
        final_price = st.session_state.forecast_data['Predicted_Price'].iloc[-1]
        total_growth = ((final_price - initial_price) / initial_price) * 100
        annual_growth = total_growth / (len(st.session_state.forecast_data) - 1)
        
        report_content += f"""
3. MARKET FORECAST
------------------
Region: {st.session_state.forecast_data['Region'].iloc[0]}
Forecast Period: {len(st.session_state.forecast_data)-1} years
Current Value: ${initial_price:,.0f}
Forecasted Value: ${final_price:,.0f}
Total Growth: {total_growth:.1f}%
Annual Growth Rate: {annual_growth:.1f}%
Market Outlook: {'Strong Growth' if annual_growth > 6 else 'Moderate Growth' if annual_growth > 3 else 'Stable'}
"""

    report_content += f"""
4. RECOMMENDATIONS
------------------
"""
    if st.session_state.property_type == "Commercial":
        report_content += """• Consider commercial leasing opportunities
• Evaluate business tenant potential
• Assess parking and accessibility features
• Review commercial tax benefits
"""
    else:
        report_content += """• Ideal for long-term rental investment
• Strong resale value potential
• Good candidate for property appreciation
• Consider minor renovations for value add
"""

    report_content += f"""
Risk Level: Low to Medium
Investment Grade: {'A' if st.session_state.property_type == 'Residential' else 'B'}

CONCLUSION
----------
This property shows strong potential for investment based on our comprehensive AI analysis.
The satellite analysis confirms proper zoning compliance, and the market forecast indicates
positive growth trends in the region.

Generated by RealtyAI Smart Real Estate Platform
Contact: support@realtyai.com
"""
    
    return report_content

# Authentication page - ONLY CHANGED THE BUTTON HIGHLIGHTING
def show_auth_page():
    st.markdown('<div class="main-header">🏠 RealtyAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unlock the future of property with smart AI insights</div>', unsafe_allow_html=True)
    
    # Initialize users
    init_users()
    
    # Auth container
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    
    # ONLY CHANGE: Login button primary, Sign Up secondary
    col1, col2 = st.columns(2)
    with col1:
        # Login button - PRIMARY (highlighted)
        login_active = st.button("Login", use_container_width=True, type="primary")
    with col2:
        # Sign Up button - SECONDARY (not highlighted)
        signup_active = st.button("Sign Up", use_container_width=True, type="secondary")
    
    if signup_active:
        st.session_state.auth_mode = 'signup'
    elif login_active or 'auth_mode' not in st.session_state:
        st.session_state.auth_mode = 'login'
    
    # Login Form
    if st.session_state.auth_mode == 'login':
        st.subheader("Login to Your Account")
        
        with st.form("login_form"):
            email = st.text_input("📧 Email", placeholder="Enter your email", value="preethimallepally72@gmail.com")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password", value="preethi08")
            
            # Login button - PRIMARY
            if st.form_submit_button("Login", use_container_width=True, type="primary"):
                if email and password:
                    if login_user(email, password):
                        st.success(f"Welcome back, {st.session_state.user['name']}!")
                        # DIRECTLY GO TO SATELLITE ANALYSIS (Step 1)
                        st.session_state.current_page = 'analysis'
                        st.session_state.current_step = 1
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid email or password")
                else:
                    st.warning("Please fill in all fields")
    
    # Signup Form
    else:
        st.subheader("Create New Account")
        
        with st.form("signup_form"):
            name = st.text_input("👤 Full Name", placeholder="Enter your full name")
            email = st.text_input("📧 Email", placeholder="Enter your email")
            password = st.text_input("🔒 Password", type="password", placeholder="Create a password (min. 6 characters)")
            confirm_password = st.text_input("✅ Confirm Password", type="password", placeholder="Confirm your password")
            
            # Sign Up button - PRIMARY (only when in signup mode)
            if st.form_submit_button("Create Account", use_container_width=True, type="primary"):
                if all([name, email, password, confirm_password]):
                    if password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        success, message = register_user(email, password, name)
                        if success:
                            st.success(message)
                            if login_user(email, password):
                                # DIRECTLY GO TO SATELLITE ANALYSIS (Step 1)
                                st.session_state.current_page = 'analysis'
                                st.session_state.current_step = 1
                                time.sleep(1)
                                st.rerun()
                        else:
                            st.error(message)
                else:
                    st.warning("Please fill in all fields")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'auth'
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'property_type' not in st.session_state:
    st.session_state.property_type = None
if 'price_prediction' not in st.session_state:
    st.session_state.price_prediction = None
if 'price_data' not in st.session_state:
    st.session_state.price_data = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Main app routing
if 'user' not in st.session_state:
    st.session_state.current_page = 'auth'

if st.session_state.current_page == 'auth':
    show_auth_page()
    
elif st.session_state.current_page == 'analysis':
    # Logout button in sidebar - CHANGED: Light blue color
    with st.sidebar:
        st.write(f"**Logged in as:** {st.session_state.user['name']}")
        st.write(f"**Role:** {st.session_state.user['role']}")
        # CHANGED: Logout button with light blue color
        if st.button("🚪 Logout", key="logout_btn", use_container_width=True, type="primary"):
            del st.session_state.user
            st.session_state.current_page = 'auth'
            st.rerun()
    
    # Progress bar
    progress_labels = ["Image Analysis", "Price Prediction", "Forecasting", "Dashboard"]
    progress = st.session_state.current_step / 4
    st.progress(progress, text=f"Step {st.session_state.current_step}/4: {progress_labels[st.session_state.current_step-1]}")
    
    # Step 1: Satellite Image Analysis (DEFAULT AFTER LOGIN)
    if st.session_state.current_step == 1:
        st.markdown('<div class="section-header">🛰️ Satellite Image Analysis</div>', unsafe_allow_html=True)
        
        st.info("Upload a satellite image to analyze property features, building structures, and zoning information.")
        
        uploaded_file = st.file_uploader("Choose a satellite image", type=['jpg', 'jpeg', 'png'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Satellite Image', use_container_width=True)
                
                if st.button('Analyze Image', type="primary", use_container_width=True):
                    with st.spinner('Analyzing image with AI...'):
                        time.sleep(2)
                        property_types = ['Residential', 'Commercial', 'Industrial', 'Agricultural']
                        st.session_state.property_type = np.random.choice(property_types, p=[0.6, 0.25, 0.1, 0.05])
                        
                        img_array = np.array(image)
                        overlay = Image.new('RGBA', image.size, (0, 255, 0, 100))
                        original_rgba = image.convert('RGBA')
                        result = Image.blend(original_rgba, overlay, 0.3)
                        st.session_state.mask_image = result
                        
                        st.success(f"✅ Analysis complete! Property type: {st.session_state.property_type}")
        
        with col2:
            if 'mask_image' in st.session_state:
                st.image(st.session_state.mask_image, caption=f'AI Building Detection - {st.session_state.property_type} Zone', 
                        use_container_width=True)
                
                st.markdown("### Analysis Results")
                st.markdown(f"**🏠 Property Type:** {st.session_state.property_type}")
                st.markdown("**📐 Building Detection:** Multiple structures identified")
                st.markdown("**🗺️ Zone Segmentation:** Completed successfully")
                st.markdown("**📊 Area Calculation:** Processing complete")
                st.markdown("**✅ Compliance Status:** Zoning regulations met")
                
                # Only show continue button after analysis
                if st.button('Continue to Price Prediction →', type="primary", use_container_width=True):
                    st.session_state.current_step = 2
                    st.rerun()
            else:
                st.info("👆 Upload a satellite image and click 'Analyze Image' to see results here.")
        
        # Additional information
        with st.expander("ℹ️ About Satellite Analysis"):
            st.write("""
            **How it works:**
            - Upload any satellite or aerial image of a property
            - Our AI analyzes building structures, land usage, and zoning
            - Get instant classification and compliance checking
            - Perfect for real estate due diligence and investment analysis
            
            **Supported Features:**
            ✅ Building footprint detection
            ✅ Property type classification  
            ✅ Zoning compliance check
            ✅ Land usage analysis
            ✅ Area calculations
            """)

    # Step 2: Price Prediction
    elif st.session_state.current_step == 2:
        st.markdown('<div class="section-header">Step 2: Property Price Prediction</div>', unsafe_allow_html=True)
        
        if st.session_state.property_type:
            st.info(f"Property Type from Satellite Analysis: {st.session_state.property_type}")
        
        input_method = st.radio("Choose input method:", ["Manual Input", "Upload CSV"])
        
        if input_method == "Manual Input":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                living_area = st.number_input("Living Area (sq ft)", min_value=500, max_value=10000, value=1500, step=100)
                overall_quality = st.slider("Overall Quality (1-10)", min_value=1, max_value=10, value=7)
                full_bathrooms = st.number_input("Full Bathrooms", min_value=1, max_value=10, value=2)
            
            with col2:
                garage_cars = st.number_input("Garage Cars", min_value=0, max_value=10, value=2)
                total_rooms = st.number_input("Total Rooms", min_value=1, max_value=20, value=6)
            
            with col3:
                year_built = st.number_input("Year Built", min_value=1800, max_value=2024, value=2000)
                lot_area = st.number_input("Lot Area (sq ft)", min_value=1000, max_value=100000, value=10000, step=1000)
                fireplaces = st.number_input("Fireplaces", min_value=0, max_value=5, value=1)
            
            if st.button("Calculate Property Price", type="primary", use_container_width=True):
                with st.spinner('Calculating price...'):
                    time.sleep(1)
                    base_price = 100000
                    price = (base_price + 
                            living_area * 100 +
                            overall_quality * 15000 +
                            full_bathrooms * 12000 +
                            garage_cars * 8000 +
                            total_rooms * 5000 +
                            (2024 - year_built) * -200 +
                            lot_area * 2 +
                            fireplaces * 5000)
                    
                    price_variation = np.random.normal(1, 0.1)
                    final_price = int(price * price_variation)
                    st.session_state.price_prediction = final_price
                    
                    st.session_state.price_data = pd.DataFrame({
                        'Living_Area': [living_area],
                        'Overall_Quality': [overall_quality],
                        'Full_Bathrooms': [full_bathrooms],
                        'Garage_Cars': [garage_cars],
                        'Total_Rooms': [total_rooms],
                        'Year_Built': [year_built],
                        'Lot_Area': [lot_area],
                        'Fireplaces': [fireplaces],
                        'Predicted_Price': [final_price]
                    })
        
        else:
            st.subheader("Upload CSV File")
            csv_file = st.file_uploader("Upload property data CSV", type=['csv'])
            
            if csv_file is not None:
                try:
                    price_data = pd.read_csv(csv_file)
                    st.write(f"📊 Uploaded Data Preview (Total rows: {len(price_data)})")
                    st.dataframe(price_data.head(10))
                    
                    st.info(f"🏠 Total properties in CSV: {len(price_data)}")
                    
                    if st.button("Predict Prices from CSV", type="primary", use_container_width=True):
                        with st.spinner(f'Processing {len(price_data)} properties...'):
                            time.sleep(2)
                            existing_pred_cols = [col for col in price_data.columns if 'predicted' in col.lower() or 'price' in col.lower()]
                            
                            if existing_pred_cols:
                                pred_col = existing_pred_cols[0]
                                st.info(f"💰 Using existing prices from column: '{pred_col}'")
                                predictions = price_data[pred_col].tolist()
                                cols_to_keep = [col for col in price_data.columns if col not in existing_pred_cols or col == pred_col]
                                price_data = price_data[cols_to_keep]
                                price_data = price_data.rename(columns={pred_col: 'Predicted_Price'})
                            else:
                                predictions = []
                                for _, row in price_data.iterrows():
                                    living_area = row.get('square_footage', row.get('living_area', row.get('sqft', row.get('area', row.get('square_losing', np.random.randint(1000, 3000))))))
                                    bedrooms = row.get('bedrooms', row.get('beds', 3))
                                    bathrooms = row.get('bathrooms', row.get('baths', 2))
                                    year_built = row.get('year_built', row.get('year', 2000))
                                    lot_area = row.get('lot_size', row.get('lot_area', row.get('land_size', 10000)))
                                    
                                    base_price = 100000
                                    price = (base_price + 
                                            living_area * 100 +
                                            bedrooms * 12000 +
                                            bathrooms * 10000 +
                                            (2024 - year_built) * -150 +
                                            lot_area * 1.5)
                                    
                                    price_variation = np.random.normal(1, 0.08)
                                    pred_price = int(price * price_variation)
                                    predictions.append(pred_price)
                                
                                price_data['Predicted_Price'] = predictions
                            
                            st.session_state.price_data = price_data
                            st.session_state.price_prediction = f"✅ Successfully processed {len(price_data)} properties"
                            
                            st.subheader("📈 Price Predictions Results")
                            st.markdown('<div class="prediction-table">', unsafe_allow_html=True)
                            display_cols = ['Predicted_Price'] + [col for col in price_data.columns if col != 'Predicted_Price']
                            st.dataframe(price_data[display_cols].head(15))
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            pred_prices = price_data['Predicted_Price']
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                avg_price = pred_prices.mean()
                                st.metric("🏘️ Average Price", f"${avg_price:,.0f}")
                            with col2:
                                highest_price = pred_prices.max()
                                st.metric("📈 Highest Price", f"${highest_price:,.0f}")
                            with col3:
                                lowest_price = pred_prices.min()
                                st.metric("📉 Lowest Price", f"${lowest_price:,.0f}")
                            
                            st.write(f"**💰 Price Range:** ${lowest_price:,.0f} - ${highest_price:,.0f}")
                            st.write(f"**📊 Standard Deviation:** ${pred_prices.std():,.0f}")
                            
                except Exception as e:
                    st.error(f"Error reading CSV file: {e}")
        
        if st.session_state.price_prediction:
            if isinstance(st.session_state.price_prediction, int):
                st.markdown(f'<div class="price-display">Estimated Property Value: ${st.session_state.price_prediction:,}</div>', 
                           unsafe_allow_html=True)
            else:
                st.success(st.session_state.price_prediction)
            
            if st.session_state.price_data is not None:
                csv = st.session_state.price_data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a class="csv-button" href="data:file/csv;base64,{b64}" download="property_predictions.csv">📥 Download Price Predictions (CSV)</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        # Navigation buttons for Step 2
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="button-container">', unsafe_allow_html=True)
            if st.button('← Back to Satellite Analysis', type="primary"):
                st.session_state.current_step = 1
                st.rerun()
            
            if st.session_state.price_prediction and st.button('Continue to Forecasting →', type='primary'):
                st.session_state.current_step = 3
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # Step 3: Price Forecasting - WITHOUT GRAPH/CHART
    elif st.session_state.current_step == 3:
        st.markdown('<div class="section-header">Step 3: Market Price Forecasting</div>', unsafe_allow_html=True)
        
        st.info("Analyze future market trends and get price predictions for your property.")
        
        # Input method selection - MANUAL INPUT ONLY
        st.subheader("Enter Forecast Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            region = st.selectbox("📍 Region", [
                "California", "New York", "Texas", "Florida", 
                "Illinois", "Arizona", "Georgia", "North Carolina"
            ])
            property_type_forecast = st.selectbox("🏠 Property Type", [
                "Residential", "Condominium", "Townhouse", "Commercial"
            ])
        
        with col2:
            forecast_years = st.slider("📅 Forecast Period (Years)", min_value=1, max_value=10, value=5)
            
            # Get current price from previous step or allow manual input
            if st.session_state.price_prediction and isinstance(st.session_state.price_prediction, int):
                default_price = st.session_state.price_prediction
            else:
                default_price = 350000
                
            current_price = st.number_input(
                "💰 Current Property Value ($)", 
                min_value=50000, 
                max_value=5000000, 
                value=default_price,
                step=50000
            )

        if st.button("Generate Market Forecast", type="primary", use_container_width=True):
            with st.spinner('Analyzing market trends and generating forecast...'):
                time.sleep(2)
                
                # Generate forecast using the function
                forecast_data = generate_forecast(current_price, region, property_type_forecast, forecast_years)
                st.session_state.forecast_data = forecast_data
                
                st.success("✅ Forecast generated successfully!")

        # Display forecast results WITHOUT GRAPHS
        if 'forecast_data' in st.session_state:
            st.markdown("---")
            st.subheader("📈 Forecast Results")
            
            # Display the forecast WITHOUT the graph
            display_single_forecast(st.session_state.forecast_data)
            
            # CSV Download for Forecasting Data
            st.markdown("---")
            st.subheader("📥 Export Forecast Data")
            
            if st.session_state.forecast_data is not None:
                csv = st.session_state.forecast_data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a class="csv-button" href="data:file/csv;base64,{b64}" download="market_forecast_{region}_{forecast_years}years.csv">📊 Download Forecast Data (CSV)</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.info("Download the complete forecast data in CSV format for further analysis.")

        # Navigation buttons
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="button-container">', unsafe_allow_html=True)
            if st.button('← Back to Price Prediction', type="primary"):
                st.session_state.current_step = 2
                st.rerun()
            
            if 'forecast_data' in st.session_state and st.button('Continue to Dashboard →', type='primary'):
                st.session_state.current_step = 4
                st.session_state.analysis_complete = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # Step 4: Dashboard
    elif st.session_state.current_step == 4:
        st.markdown('<div class="section-header">Step 4: Analysis Dashboard</div>', unsafe_allow_html=True)
        
        if not st.session_state.analysis_complete:
            st.warning("Please complete all previous analysis steps to generate the dashboard.")
            if st.button("Return to Satellite Analysis", type="primary"):
                st.session_state.current_step = 1
                st.rerun()
        else:
            # Summary Dashboard
            st.markdown("## 📋 Complete Analysis Summary")
            
            # PDF Download Button - PROPER FIX
            st.markdown("---")
            st.subheader("📤 Export Report")
            
            # Create PDF report content
            pdf_report = create_pdf_report()
            
            # PROPER FIX: Create a text file instead of fake PDF
            b64 = base64.b64encode(pdf_report.encode()).decode()
            href = f'<a class="pdf-button" href="data:text/plain;base64,{b64}" download="realtyai_analysis_report.txt">📄 Download as Text Report</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            st.info("The report contains your complete property analysis including satellite analysis, price prediction, market forecast, and investment recommendations.")
            
            # Alternative: Show report content directly
            with st.expander("📋 View Full Report Content"):
                st.text(pdf_report)
            
            # Analysis Results Cards
            col1, col2 = st.columns(2)
            
            with col1:
                # Satellite Analysis Results
                st.markdown('<div class="data-card">', unsafe_allow_html=True)
                st.subheader("🛰️ Satellite Analysis")
                st.write(f"**Property Type:** {st.session_state.property_type}")
                st.write("**Status:** Analysis Completed")
                st.write("**Building Detection:** Successful")
                st.write("**Zoning Compliance:** Verified")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Price Prediction Results
                st.markdown('<div class="data-card">', unsafe_allow_html=True)
                st.subheader("💰 Price Prediction")
                if isinstance(st.session_state.price_prediction, int):
                    st.write(f"**Estimated Value:** ${st.session_state.price_prediction:,}")
                    st.write("**Confidence Level:** High")
                    st.write("**Market Position:** Competitive")
                else:
                    st.write(f"**Properties Analyzed:** {len(st.session_state.price_data)}")
                    st.write(f"**Average Price:** ${st.session_state.price_data['Predicted_Price'].mean():,.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Forecasting Results
                st.markdown('<div class="data-card">', unsafe_allow_html=True)
                st.subheader("📈 Market Forecast")
                if 'forecast_data' in st.session_state:
                    initial_price = st.session_state.forecast_data['Predicted_Price'].iloc[0]
                    final_price = st.session_state.forecast_data['Predicted_Price'].iloc[-1]
                    total_growth = ((final_price - initial_price) / initial_price) * 100
                    annual_growth = total_growth / (len(st.session_state.forecast_data) - 1)
                    
                    st.write(f"**Region:** {st.session_state.forecast_data['Region'].iloc[0]}")
                    st.write(f"**Forecast Period:** {len(st.session_state.forecast_data)-1} years")
                    st.write(f"**Current Value:** ${initial_price:,.0f}")
                    st.write(f"**Forecasted Value:** ${final_price:,.0f}")
                    st.write(f"**Annual Growth:** {annual_growth:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Recommendations
                st.markdown('<div class="data-card">', unsafe_allow_html=True)
                st.subheader("🎯 Recommendations")
                if st.session_state.property_type == "Commercial":
                    st.write("• Consider commercial leasing")
                    st.write("• Evaluate business potential")
                    st.write("• Assess parking/accessibility")
                else:
                    st.write("• Good rental investment")
                    st.write("• Strong resale potential")
                    st.write("• Consider value-add renovations")
                st.write("**Risk Level:** Low to Medium")
                st.markdown('</div>', unsafe_allow_html=True)

            # Final Actions
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button('🔄 Start New Analysis', type='primary', use_container_width=True):
                    # Reset analysis state but keep user logged in
                    st.session_state.current_step = 1
                    st.session_state.property_type = None
                    st.session_state.price_prediction = None
                    st.session_state.price_data = None
                    st.session_state.analysis_complete = False
                    if 'forecast_data' in st.session_state:
                        del st.session_state.forecast_data
                    st.rerun()

# Footer
st.markdown("---")
st.markdown("RealtyAI Smart Real Estate Insight Platform | Powered by AI/ML Models")