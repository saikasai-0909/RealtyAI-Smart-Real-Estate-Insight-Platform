import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import base64
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.stats import skew
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image as RLImage,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import tempfile
import hashlib  # <-- for authentication

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AI Property Intelligence",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== CUSTOM CSS ====================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .sub-header {
        font-size: 2rem;
        font-weight: 600;
        color: #2d3748;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
        text-align: center;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .success-banner {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(17, 153, 142, 0.3);
    }
    
    .info-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(102, 126, 234, 0.2);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .question-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .home-bg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 4rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
    }
    
    .home-bg h1, .home-bg p {
        color: #ffffff !important;
    }
    
    .home-description {
        color: #ffffff !important;
        font-size: 1.3rem;
        text-align: center;
        margin-top: 1rem;
    }
    
    .pipeline-step {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
    }
    
    .workflow-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        min-height: 200px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .workflow-card:hover {
        transform: scale(1.05);
    }
    
    .workflow-card h4, .workflow-card p {
        color: white !important;
    }
    
    /* Fix for dark mode text visibility */
    .info-card h2, .info-card h3, .info-card h4, .info-card p {
        color: #2d3748 !important;
    }
    
    .info-card ul li {
        color: #4a5568 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================== SESSION STATE INIT ====================
if "classification" not in st.session_state:
    st.session_state.classification = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "house_inputs" not in st.session_state:
    st.session_state.house_inputs = {}
if "house_predictions" not in st.session_state:
    st.session_state.house_predictions = None
if "timeseries_predictions" not in st.session_state:
    st.session_state.timeseries_predictions = None
if "timeseries_inputs" not in st.session_state:
    st.session_state.timeseries_inputs = {}
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# ==================== AUTHENTICATION ====================

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# Hardcoded users (username: hashed_password)
USERS = {
    "admin": hash_password("admin123"),
    "user1": hash_password("password123"),
    "demo": hash_password("demo123"),
}

def check_credentials(username, password):
    """Verify username and password"""
    if username in USERS:
        return USERS[username] == hash_password(password)
    return False

def initialize_auth():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None

def show_login_page():
    """Display login form"""
    st.markdown(
        """
        <div class="home-bg">
            <h1 class="main-header">🔐 Login to AI Property Intelligence</h1>
            <p class="home-description">Enter your credentials to access the platform</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            """
            <div class="info-card" style="margin-top: 2rem;">
                <h3 style="color: #667eea; text-align: center;">Sign In</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter username")
            password = st.text_input(
                "🔒 Password", type="password", placeholder="Enter password"
            )

            col_a, col_b = st.columns(2)
            with col_a:
                submit = st.form_submit_button(
                    "🚀 Login", use_container_width=True
                )
            with col_b:
                reset = st.form_submit_button(
                    "🔄 Reset", use_container_width=True
                )

            if submit:
                if username and password:
                    if check_credentials(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                else:
                    st.warning("⚠️ Please enter both username and password")

        # Demo credentials display
        st.markdown(
            """
            <div style="background: rgba(102, 126, 234, 0.1); 
                        padding: 1rem; border-radius: 10px; 
                        margin-top: 1rem; border-left: 4px solid #667eea;">
                <p style="color: #2d3748; font-size: 0.9rem; margin: 0;">
                    <strong>Demo Credentials:</strong><br>
                    Username: <code>demo</code> | Password: <code>demo123</code>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

def show_logout_button():
    """Display logout button in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div style="background: rgba(255, 255, 255, 0.1); 
                    padding: 1rem; border-radius: 10px; 
                    text-align: center;">
            <p style="color: #ffffff; margin: 0;">
                👤 Logged in as<br>
                <strong>{st.session_state.username}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()

# ==================== HELPER FUNCTIONS ====================
def preprocess_house_data(input_dict):
    """Preprocess user inputs to match model requirements"""
    df = pd.DataFrame([input_dict])

    required_cols = {
        "MSSubClass": "60",
        "MSZoning": "RL",
        "LotFrontage": 65.0,
        "LotArea": 8000,
        "Street": "Pave",
        "Alley": "None",
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "Utilities": "AllPub",
        "LotConfig": "Inside",
        "LandSlope": "Gtl",
        "Neighborhood": "NAmes",
        "Condition1": "Norm",
        "Condition2": "Norm",
        "BldgType": "1Fam",
        "HouseStyle": "2Story",
        "OverallQual": 5,
        "OverallCond": "5",
        "YearBuilt": 1995,
        "YearRemodAdd": 2000,
        "RoofStyle": "Gable",
        "RoofMatl": "CompShg",
        "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd",
        "MasVnrType": "None",
        "MasVnrArea": 0,
        "ExterQual": "TA",
        "ExterCond": "TA",
        "Foundation": "PConc",
        "BsmtQual": "TA",
        "BsmtCond": "TA",
        "BsmtExposure": "No",
        "BsmtFinType1": "GLQ",
        "BsmtFinSF1": 500,
        "BsmtFinType2": "Unf",
        "BsmtFinSF2": 0,
        "BsmtUnfSF": 300,
        "TotalBsmtSF": 800,
        "Heating": "GasA",
        "HeatingQC": "Ex",
        "CentralAir": "Y",
        "Electrical": "SBrkr",
        "1stFlrSF": 1000,
        "2ndFlrSF": 800,
        "LowQualFinSF": 0,
        "GrLivArea": 1800,
        "BsmtFullBath": 1,
        "BsmtHalfBath": 0,
        "FullBath": 2,
        "HalfBath": 1,
        "BedroomAbvGr": 3,
        "KitchenAbvGr": 1,
        "KitchenQual": "TA",
        "TotRmsAbvGrd": 7,
        "Functional": "Typ",
        "Fireplaces": 1,
        "FireplaceQu": "None",
        "GarageType": "Attchd",
        "GarageYrBlt": 1995,
        "GarageFinish": "RFn",
        "GarageCars": 2,
        "GarageArea": 500,
        "GarageQual": "TA",
        "GarageCond": "TA",
        "PavedDrive": "Y",
        "WoodDeckSF": 0,
        "OpenPorchSF": 50,
        "EnclosedPorch": 0,
        "3SsnPorch": 0,
        "ScreenPorch": 0,
        "PoolArea": 0,
        "PoolQC": "None",
        "Fence": "None",
        "MiscFeature": "None",
        "MiscVal": 0,
        "MoSold": "6",
        "YrSold": "2010",
        "SaleType": "WD",
        "SaleCondition": "Normal",
    }

    for col, default_val in required_cols.items():
        if col not in df.columns:
            df[col] = default_val

    all_data = df.copy()

    for col in [
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "MasVnrType",
    ]:
        if col in all_data.columns:
            all_data[col] = all_data[col].fillna("None")

    for col in [
        "GarageYrBlt",
        "GarageArea",
        "GarageCars",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "BsmtFullBath",
        "BsmtHalfBath",
        "MasVnrArea",
    ]:
        if col in all_data.columns:
            all_data[col] = all_data[col].fillna(0)

    if "Utilities" in all_data.columns:
        all_data = all_data.drop(["Utilities"], axis=1)

    for col in ["MSSubClass", "OverallCond", "YrSold", "MoSold"]:
        if col in all_data.columns:
            all_data[col] = all_data[col].astype(str)

    ordinal_cols = [
        "ExterQual",
        "ExterCond",
        "BsmtQual",
        "BsmtCond",
        "HeatingQC",
        "KitchenQual",
        "FireplaceQu",
        "GarageQual",
        "GarageCond",
        "PoolQC",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "Functional",
        "Fence",
    ]

    for c in ordinal_cols:
        if c in all_data.columns:
            lbl = LabelEncoder()
            all_data[c] = lbl.fit_transform(all_data[c].astype(str))

    if all(
        col in all_data.columns for col in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]
    ):
        all_data["TotalSF"] = (
            all_data["TotalBsmtSF"]
            + all_data["1stFlrSF"]
            + all_data["2ndFlrSF"]
        )

    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    if len(numeric_feats) > 0:
        skewed_feats = (
            all_data[numeric_feats]
            .apply(lambda x: skew(x.dropna()))
            .sort_values(ascending=False)
        )
        skewed_features = skewed_feats[abs(skewed_feats) > 0.75].index
        all_data[skewed_features] = np.log1p(all_data[skewed_features])

    all_data = pd.get_dummies(all_data)

    return all_data

def generate_comprehensive_csv():
    """Generate comprehensive CSV with all data"""
    data = []

    # Classification data
    if st.session_state.classification:
        data.append(["CLASSIFICATION RESULTS", ""])
        data.append(["Property Type", st.session_state.classification])
        data.append(
            ["Confidence", f"{st.session_state.confidence*100:.2f}%"]
        )
        data.append(["", ""])

    # House pricing inputs and predictions
    if st.session_state.house_inputs:
        data.append(["HOUSE PRICING - INPUT PARAMETERS", ""])
        for key, value in st.session_state.house_inputs.items():
            data.append([key, str(value)])
        data.append(["", ""])

    if st.session_state.house_predictions:
        data.append(["HOUSE PRICING - PREDICTIONS", ""])
        pred = st.session_state.house_predictions
        data.append(["Current Price", f"${pred['current']:,.0f}"])
        data.append(["+1 Year Price", f"${pred['year1']:,.0f}"])
        data.append(["+2 Years Price", f"${pred['year2']:,.0f}"])
        data.append(["+3 Years Price", f"${pred['year3']:,.0f}"])
        data.append(["", ""])

    # Time series inputs and predictions
    if st.session_state.timeseries_inputs:
        data.append(["MARKET FORECAST - INPUT PARAMETERS", ""])
        for key, value in st.session_state.timeseries_inputs.items():
            data.append([key, str(value)])
        data.append(["", ""])

    if st.session_state.timeseries_predictions:
        data.append(["MARKET FORECAST - PREDICTIONS", ""])
        pred = st.session_state.timeseries_predictions
        data.append(
            ["Current Market Value", f"${pred['current_value']:,.0f}"]
        )
        data.append(["Forecasted Value", f"${pred['forecast']:,.0f}"])
        data.append(["Expected Change", f"{pred['change']:.2f}%"])

    df = pd.DataFrame(data, columns=["Parameter", "Value"])
    return df.to_csv(index=False).encode()

def generate_pdf_report():
    """Generate comprehensive PDF report with image"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor("#667eea"),
        spaceAfter=30,
        alignment=TA_CENTER,
    )

    # Title
    story.append(
        Paragraph("🏘️ AI Property Intelligence Report", title_style)
    )
    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.3 * inch))

    # Add uploaded satellite image if available
    if st.session_state.uploaded_image:
        story.append(
            Paragraph("Satellite Image Analysis", styles["Heading2"])
        )
        try:
            # Save image temporarily
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".png"
            ) as tmp_file:
                img = Image.open(st.session_state.uploaded_image)
                img.save(tmp_file.name, format="PNG")

                # Add to PDF
                img_pdf = RLImage(tmp_file.name, width=4 * inch, height=4 * inch)
                story.append(img_pdf)
                story.append(Spacer(1, 0.2 * inch))
        except Exception as e:
            story.append(
                Paragraph(
                    f"Could not include image: {str(e)}", styles["Normal"]
                )
            )
        story.append(Spacer(1, 0.3 * inch))

    # Classification Results
    if st.session_state.classification:
        story.append(
            Paragraph(
                "Property Classification Results", styles["Heading2"]
            )
        )
        class_data = [
            ["Property Type", st.session_state.classification],
            [
                "Confidence",
                f"{st.session_state.confidence*100:.2f}%",
            ],
            ["Model", "SpaceNet CNN"],
            ["Accuracy", "82.94%"],
        ]
        class_table = Table(class_data, colWidths=[3 * inch, 3 * inch])
        class_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#667eea"),
                    ),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold",
                    ),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        story.append(class_table)
        story.append(Spacer(1, 0.3 * inch))

    # House Pricing Inputs
    if st.session_state.house_inputs:
        story.append(
            Paragraph(
                "House Pricing - Input Parameters", styles["Heading2"]
            )
        )
        input_data = [
            [key, str(value)]
            for key, value in st.session_state.house_inputs.items()
        ]
        input_table = Table(input_data, colWidths=[3 * inch, 3 * inch])
        input_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                ]
            )
        )
        story.append(input_table)
        story.append(Spacer(1, 0.3 * inch))

    # House Price Predictions
    if st.session_state.house_predictions:
        story.append(
            Paragraph("Price Predictions", styles["Heading2"])
        )
        pred = st.session_state.house_predictions
        price_data = [
            ["Period", "Predicted Price"],
            ["Current", f"${pred['current']:,.0f}"],
            ["+1 Year", f"${pred['year1']:,.0f}"],
            ["+2 Years", f"${pred['year2']:,.0f}"],
            ["+3 Years", f"${pred['year3']:,.0f}"],
        ]
        price_table = Table(price_data, colWidths=[3 * inch, 3 * inch])
        price_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#11998e"),
                    ),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold",
                    ),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        story.append(price_table)
        story.append(Spacer(1, 0.3 * inch))

    # Time Series Inputs
    if st.session_state.timeseries_inputs:
        story.append(
            Paragraph(
                "Market Forecast - Input Parameters", styles["Heading2"]
            )
        )
        ts_input_data = [
            [key, str(value)]
            for key, value in st.session_state.timeseries_inputs.items()
        ]
        ts_input_table = Table(
            ts_input_data, colWidths=[3 * inch, 3 * inch]
        )
        ts_input_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                ]
            )
        )
        story.append(ts_input_table)
        story.append(Spacer(1, 0.3 * inch))

    # Time Series Forecast
    if st.session_state.timeseries_predictions:
        story.append(
            Paragraph(
                "Market Forecast Results", styles["Heading2"]
            )
        )
        pred = st.session_state.timeseries_predictions
        ts_data = [
            ["Metric", "Value"],
            [
                "Current Market Value",
                f"${pred['current_value']:,.0f}",
            ],
            [
                "Forecasted Market Value",
                f"${pred['forecast']:,.0f}",
            ],
            ["Expected Change", f"{pred['change']:+.2f}%"],
            ["Model", "LSTM"],
            ["Accuracy", "97.85%"],
        ]
        ts_table = Table(ts_data, colWidths=[3 * inch, 3 * inch])
        ts_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#f5576c"),
                    ),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold",
                    ),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        story.append(ts_table)

    doc.build(story)
    buffer.seek(0)
    return buffer

def send_email_report(recipient_email, pdf_buffer, csv_buffer):
    """Send report via email - CONFIGURED VERSION"""
    try:
        # ✅ EMAIL CONFIGURATION - UPDATE THESE VALUES
        sender_email = "hardikagarwal0508@gmail.com"  # REPLACE with your Gmail
        sender_password = "xfvs saql rwka incu"  # REPLACE with your app password

        # Validate email configuration (optional sentinel check kept but will not trigger)
        if (
            sender_email == "YOUR_EMAIL@gmail.com"
            or sender_password == "YOUR_APP_PASSWORD"
        ):
            st.error(
                "❌ Email not configured! Please update sender_email and sender_password in the code."
            )
            st.info(
                """
            📧 **Setup Instructions:**
            1. Go to Google Account → Security
            2. Enable 2-Step Verification
            3. Generate App Password (Select 'Mail' and your device)
            4. Copy the 16-digit password
            5. Update lines with sender_email and sender_password in the code
            """
            )
            return False

        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = "Real Estate Property Intelligence Report"

        classification = (
            st.session_state.classification
            if st.session_state.classification
            else "N/A"
        )
        confidence = (
            f"{st.session_state.confidence*100:.2f}%"
            if st.session_state.confidence
            else "N/A"
        )

        body = f"""
Dear User,

Please find attached your comprehensive AI Property Intelligence Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.

Report Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Property Classification: {classification}
• Confidence Level: {confidence}

The attached documents include:
✓ Detailed PDF Report with satellite image analysis
✓ Complete CSV data export

Best regards,
Team RealtyAI Smart Real Estate Insight Platform
        """

        msg.attach(MIMEText(body, "plain"))

        # Attach PDF
        pdf_part = MIMEBase("application", "octet-stream")
        pdf_part.set_payload(pdf_buffer.read())
        encoders.encode_base64(pdf_part)
        pdf_part.add_header(
            "Content-Disposition", "attachment; filename=property_report.pdf"
        )
        msg.attach(pdf_part)

        # Attach CSV
        csv_part = MIMEBase("application", "octet-stream")
        csv_part.set_payload(csv_buffer)
        encoders.encode_base64(csv_part)
        csv_part.add_header(
            "Content-Disposition", "attachment; filename=property_report.csv"
        )
        msg.attach(csv_part)

        # Send email
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()

        return True
    except smtplib.SMTPAuthenticationError:
        st.error(
            "❌ Email authentication failed! Check your email and app password."
        )
        return False
    except Exception as e:
        st.error(f"❌ Email sending failed: {str(e)}")
        return False

@st.cache_resource
def load_pickle_model(model_path):
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_keras_model(model_path):
    try:
        import tensorflow as tf

        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# ==================== HOME PAGE ====================
def show_homepage():
    # Hero Section
    st.markdown(
        """
        <div class="home-bg">
            <h1 class="main-header">🏘️ AI Property Intelligence Platform</h1>
            <p class="home-description">
                Evaluate property conditions, predict price trends, and segment satellite imagery
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Overview
    st.markdown(
        """
        <div class="info-card">
            <h2 style="color: #667eea; text-align: center;">🎯 What This Platform Does</h2>
            <p style="font-size: 1.1rem; line-height: 1.8; text-align: center; color: #2d3748;">
                Our AI platform analyzes satellite images to classify property types, 
                predicts accurate property prices, and forecasts market trends using 
                advanced machine learning models. Perfect for property buyers, investors, 
                and urban planners.
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Features
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="info-card" style="border-left: 5px solid #667eea; min-height: 300px;">
                <h3 style="color: #667eea;">🛰️ Image Classification</h3>
                <p style="color: #2d3748;"><strong>Model:</strong> CNN (SpaceNet)</p>
                <p style="color: #2d3748;"><strong>Accuracy:</strong> 82.94%</p>
                <br>
                <p style="color: #4a5568;">Automatically classifies satellite imagery into residential or commercial properties using deep learning.</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="info-card" style="border-left: 5px solid #11998e; min-height: 300px;">
                <h3 style="color: #11998e;">🏠 Price Prediction</h3>
                <p style="color: #2d3748;"><strong>Model:</strong> XGBoost</p>
                <p style="color: #2d3748;"><strong>R² Score:</strong> 0.9974</p>
                <br>
                <p style="color: #4a5568;">Predicts property prices with 99.74% accuracy using 79+ features including location, size, and quality metrics.</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="info-card" style="border-left: 5px solid #f5576c; min-height: 300px;">
                <h3 style="color: #f5576c;">📈 Market Forecasting</h3>
                <p style="color: #2d3748;"><strong>Model:</strong> LSTM</p>
                <p style="color: #2d3748;"><strong>Accuracy:</strong> 97.85%</p>
                <br>
                <p style="color: #4a5568;">Forecasts housing market trends using time series analysis of 8 key market indicators.</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    # How It Works
    st.markdown("---")
    st.markdown(
        """
        <div class="info-card">
            <h2 style="color: #667eea; text-align: center; margin-bottom: 2rem;">🔄 How It Works</h2>
        </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="workflow-card">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📤</div>
                <h4 style="color: white !important;">Upload Image</h4>
                <p style="color: rgba(255,255,255,0.9) !important;">Provide satellite imagery</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="workflow-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
                <h4 style="color: white !important;">Classification</h4>
                <p style="color: rgba(255,255,255,0.9) !important;">AI identifies property type</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="workflow-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
                <h4 style="color: white !important;">Prediction</h4>
                <p style="color: rgba(255,255,255,0.9) !important;">Get price & trend forecasts</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="workflow-card">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📥</div>
                <h4 style="color: white !important;">Download</h4>
                <p style="color: rgba(255,255,255,0.9) !important;">Export detailed reports</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

# ==================== PIPELINE PAGE ====================
def show_pipeline():
    st.markdown(
        '<h2 class="sub-header">🔄 Intelligent Property Analysis Pipeline</h2>',
        unsafe_allow_html=True,
    )

    # Step 1: Upload Satellite Image
    st.markdown(
        """
        <div class="pipeline-step">
            <h3>📤 Step 1: Upload Satellite Image</h3>
            <p>Upload an aerial/satellite image of the property for classification</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    uploaded_image = st.file_uploader("", type=["png", "jpg", "jpeg"], key="pipeline_img")

    if uploaded_image:
        # Store uploaded image in session state
        st.session_state.uploaded_image = uploaded_image

        col1, col2 = st.columns([1, 1])
        with col1:
            img = Image.open(uploaded_image)
            st.image(img, caption="Uploaded Satellite Image", use_column_width=True)

        with col2:
            st.markdown(
                """
                <div class="info-card">
                    <h4>📋 Image Details</h4>
                </div>
            """,
                unsafe_allow_html=True,
            )
            st.info(f"**Filename:** {uploaded_image.name}")
            st.info(f"**Size:** {img.size[0]} x {img.size[1]} px")
            st.info(f"**Format:** {img.format}")

        if st.button("🚀 Start Analysis", type="primary"):
            spacenet_model = load_keras_model(
                "/home/hardik/Desktop/python_intern/residential_commercial_model.keras"
            )

            if spacenet_model:
                with st.spinner("🔍 Analyzing property type..."):
                    uploaded_image.seek(0)
                    img_processed = Image.open(uploaded_image).convert("RGBA")
                    img_processed = img_processed.resize((256, 256))
                    img_array = np.array(img_processed).astype(np.float32) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    pred = spacenet_model.predict(img_array, verbose=0)
                    score = pred[0][0]

                    classification = "Commercial" if score > 0.5 else "Residential"
                    confidence = score if score > 0.5 else 1 - score

                    st.session_state.classification = classification
                    st.session_state.confidence = confidence

                    st.markdown(
                        f"""
                        <div class="success-banner">
                            ✅ Property Classified as: <strong>{classification}</strong> ({confidence*100:.2f}% confidence)
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

    # Step 2: House Pricing Questions
    if st.session_state.classification:
        st.markdown(
            """
            <div class="pipeline-step">
                <h3>🏠 Step 2: Property Details for Price Prediction</h3>
                <p>Answer these questions to get accurate price predictions</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

        with st.form("house_pricing_form"):
            st.markdown(
                '<div class="question-box">❓ Question 1: Property Characteristics</div>',
                unsafe_allow_html=True,
            )
            col1, col2 = st.columns(2)
            with col1:
                overall_qual = st.slider(
                    "Overall Quality (1-10)",
                    1,
                    10,
                    5,
                    help="Overall material and finish quality",
                )
                year_built = st.number_input("Year Built", 1800, 2024, 2000)
                lot_area = st.number_input("Lot Area (sq ft)", 1000, 50000, 8000)
            with col2:
                gr_liv_area = st.number_input(
                    "Living Area (sq ft)", 500, 6000, 1800
                )
                total_bsmt_sf = st.number_input(
                    "Basement Area (sq ft)", 0, 3000, 800
                )
                garage_area = st.number_input(
                    "Garage Area (sq ft)", 0, 1500, 500
                )

            st.markdown(
                '<div class="question-box">❓ Question 2: Rooms & Facilities</div>',
                unsafe_allow_html=True,
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                bedrooms = st.number_input("Bedrooms", 0, 10, 3)
                full_bath = st.number_input("Full Bathrooms", 0, 5, 2)
            with col2:
                half_bath = st.number_input("Half Bathrooms", 0, 3, 1)
                fireplaces = st.number_input("Fireplaces", 0, 4, 1)
            with col3:
                garage_cars = st.number_input(
                    "Garage Capacity (cars)", 0, 4, 2
                )
                tot_rms_abv_grd = st.number_input(
                    "Total Rooms Above Grade", 2, 15, 7
                )

            st.markdown(
                '<div class="question-box">❓ Question 3: Quality Ratings</div>',
                unsafe_allow_html=True,
            )
            col1, col2 = st.columns(2)
            with col1:
                kitchen_qual = st.selectbox(
                    "Kitchen Quality",
                    ["Ex", "Gd", "TA", "Fa", "Po"],
                    index=2,
                    help="Ex=Excellent, Gd=Good, TA=Typical, Fa=Fair, Po=Poor",
                )
                exter_qual = st.selectbox(
                    "Exterior Quality", ["Ex", "Gd", "TA", "Fa", "Po"], index=2
                )
            with col2:
                heating_qc = st.selectbox(
                    "Heating Quality", ["Ex", "Gd", "TA", "Fa", "Po"], index=0
                )
                central_air = st.selectbox(
                    "Central Air Conditioning", ["Y", "N"], index=0
                )

            submit_house = st.form_submit_button("💰 Predict House Price")

            if submit_house:
                house_model = load_pickle_model(
                    "/home/hardik/Desktop/python_intern/house_price.pkl"
                )

                if house_model:
                    with st.spinner("🔄 Calculating property value..."):
                        input_data = {
                            "OverallQual": overall_qual,
                            "YearBuilt": year_built,
                            "LotArea": lot_area,
                            "GrLivArea": gr_liv_area,
                            "TotalBsmtSF": total_bsmt_sf,
                            "GarageArea": garage_area,
                            "BedroomAbvGr": bedrooms,
                            "FullBath": full_bath,
                            "HalfBath": half_bath,
                            "Fireplaces": fireplaces,
                            "GarageCars": garage_cars,
                            "TotRmsAbvGrd": tot_rms_abv_grd,
                            "KitchenQual": kitchen_qual,
                            "ExterQual": exter_qual,
                            "HeatingQC": heating_qc,
                            "CentralAir": central_air,
                            "1stFlrSF": int(gr_liv_area * 0.6),
                            "2ndFlrSF": int(gr_liv_area * 0.4),
                        }

                        # Store inputs in session state
                        st.session_state.house_inputs = input_data.copy()

                        processed_data = preprocess_house_data(input_data)

                        try:
                            model_features = (
                                house_model.get_booster().feature_names
                            )
                            for feat in model_features:
                                if feat not in processed_data.columns:
                                    processed_data[feat] = 0
                            processed_data = processed_data[model_features]
                        except Exception:
                            # For models without get_booster/feature_names
                            pass

                        pred_log = house_model.predict(processed_data)
                        current_price = np.expm1(pred_log)[0]

                        appreciation_rate = 0.03
                        year1_price = current_price * (1 + appreciation_rate)
                        year2_price = current_price * (1 + appreciation_rate) ** 2
                        year3_price = current_price * (1 + appreciation_rate) ** 3

                        st.session_state.house_predictions = {
                            "current": current_price,
                            "year1": year1_price,
                            "year2": year2_price,
                            "year3": year3_price,
                        }

                        st.markdown(
                            '<div class="success-banner">✅ Price Prediction Complete!</div>',
                            unsafe_allow_html=True,
                        )

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(
                                f"""
                                <div class="result-card">
                                    <h4>Current Price</h4>
                                    <h2>${current_price:,.0f}</h2>
                                </div>
                            """,
                                unsafe_allow_html=True,
                            )
                        with col2:
                            st.markdown(
                                f"""
                                <div class="result-card">
                                    <h4>+1 Year</h4>
                                    <h2>${year1_price:,.0f}</h2>
                                </div>
                            """,
                                unsafe_allow_html=True,
                            )
                        with col3:
                            st.markdown(
                                f"""
                                <div class="result-card">
                                    <h4>+2 Years</h4>
                                    <h2>${year2_price:,.0f}</h2>
                                </div>
                            """,
                                unsafe_allow_html=True,
                            )
                        with col4:
                            st.markdown(
                                f"""
                                <div class="result-card">
                                    <h4>+3 Years</h4>
                                    <h2>${year3_price:,.0f}</h2>
                                </div>
                            """,
                                unsafe_allow_html=True,
                            )

                        # Visualization
                        fig = go.Figure()

                        years = ["Current", "+1 Year", "+2 Years", "+3 Years"]
                        prices = [
                            current_price,
                            year1_price,
                            year2_price,
                            year3_price,
                        ]

                        fig.add_trace(
                            go.Scatter(
                                x=years,
                                y=prices,
                                mode="lines+markers",
                                line=dict(color="#667eea", width=3),
                                marker=dict(size=12, color="#764ba2"),
                                name="Predicted Price",
                                fill="tozeroy",
                                fillcolor="rgba(102, 126, 234, 0.2)",
                            )
                        )

                        fig.update_layout(
                            title="Property Price Forecast",
                            xaxis_title="Time Period",
                            yaxis_title="Price ($)",
                            template="plotly_white",
                            height=400,
                            hovermode="x unified",
                        )
                        st.plotly_chart(fig, use_container_width=True)

    # Step 3: Time Series Questions
    if st.session_state.house_predictions:
        st.markdown(
            """
            <div class="pipeline-step">
                <h3>📈 Step 3: Market Trend Analysis</h3>
                <p>Provide market indicators for time series forecasting</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

        with st.form("timeseries_form"):
            st.markdown(
                '<div class="question-box">❓ Current Market Indicators</div>',
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)
            with col1:
                zhvi_per_sqft = st.number_input(
                    "ZHVI Per Sq Ft ($)",
                    50,
                    500,
                    150,
                    help="Zillow Home Value Index per square foot",
                )
                pct_decreasing = st.slider(
                    "% Homes Decreasing in Value", 0.0, 100.0, 20.0
                )
                pct_increasing = st.slider(
                    "% Homes Increasing in Value", 0.0, 100.0, 60.0
                )
                zhvi_3bedroom = st.number_input(
                    "3-Bedroom Home Value ($)", 100000, 1000000, 250000
                )

            with col2:
                zhvi_bottom = st.number_input(
                    "Bottom Tier Value ($)", 50000, 500000, 150000
                )
                zhvi_middle = st.number_input(
                    "Middle Tier Value ($)", 100000, 800000, 250000
                )
                zhvi_single = st.number_input(
                    "Single Family Value ($)", 150000, 1500000, 350000
                )
                zhvi_top = st.number_input(
                    "Top Tier Value ($)", 200000, 2000000, 500000
                )

            submit_timeseries = st.form_submit_button(
                "📊 Generate Market Forecast"
            )

            if submit_timeseries:
                lstm_model = load_keras_model(
                    "/home/hardik/Desktop/python_intern/lstm_model.keras"
                )

                if lstm_model:
                    with st.spinner("🔄 Analyzing market trends..."):
                        # Store inputs
                        st.session_state.timeseries_inputs = {
                            "ZHVI Per Sq Ft": zhvi_per_sqft,
                            "% Homes Decreasing": pct_decreasing,
                            "% Homes Increasing": pct_increasing,
                            "3-Bedroom Value": zhvi_3bedroom,
                            "Bottom Tier Value": zhvi_bottom,
                            "Middle Tier Value": zhvi_middle,
                            "Single Family Value": zhvi_single,
                            "Top Tier Value": zhvi_top,
                        }

                        feature_data = np.array(
                            [
                                [
                                    zhvi_per_sqft,
                                    pct_decreasing,
                                    pct_increasing,
                                    zhvi_3bedroom,
                                    zhvi_bottom,
                                    zhvi_middle,
                                    zhvi_single,
                                    zhvi_top,
                                ]
                            ]
                        )

                        sequence = np.tile(feature_data, (12, 1))
                        scaler_X = MinMaxScaler()
                        sequence_scaled = scaler_X.fit_transform(sequence)
                        sequence_scaled = np.expand_dims(
                            sequence_scaled, axis=0
                        )

                        pred_scaled = lstm_model.predict(
                            sequence_scaled, verbose=0
                        )

                        scaler_y = MinMaxScaler()
                        scaler_y.fit(
                            [
                                [zhvi_single * 0.8],
                                [zhvi_single * 1.2],
                            ]
                        )
                        prediction = scaler_y.inverse_transform(pred_scaled)[0][0]

                        change = (
                            (prediction - zhvi_single) / zhvi_single
                        ) * 100

                        st.session_state.timeseries_predictions = {
                            "current_value": zhvi_single,
                            "forecast": prediction,
                            "change": change,
                        }

                        st.markdown(
                            '<div class="success-banner">✅ Market Forecast Complete!</div>',
                            unsafe_allow_html=True,
                        )

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(
                                f"""
                                <div class="result-card">
                                    <h4>Current Market Value</h4>
                                    <h2>${zhvi_single:,.0f}</h2>
                                </div>
                            """,
                                unsafe_allow_html=True,
                            )
                        with col2:
                            st.markdown(
                                f"""
                                <div class="result-card">
                                    <h4>Forecasted Value</h4>
                                    <h2>${prediction:,.0f}</h2>
                                </div>
                            """,
                                unsafe_allow_html=True,
                            )
                        with col3:
                            st.markdown(
                                f"""
                                <div class="result-card">
                                    <h4>Expected Change</h4>
                                    <h2>{change:+.2f}%</h2>
                                </div>
                            """,
                                unsafe_allow_html=True,
                            )

                        # Radar Chart
                        fig = go.Figure()

                        categories = [
                            "Bottom Tier",
                            "Middle Tier",
                            "Top Tier",
                            "Single Family",
                            "3-Bedroom",
                        ]
                        values = [
                            zhvi_bottom,
                            zhvi_middle,
                            zhvi_top,
                            zhvi_single,
                            zhvi_3bedroom,
                        ]

                        fig.add_trace(
                            go.Scatterpolar(
                                r=values,
                                theta=categories,
                                fill="toself",
                                name="Market Values",
                                line=dict(color="#667eea", width=2),
                                fillcolor="rgba(102, 126, 234, 0.3)",
                            )
                        )

                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, max(values) * 1.2],
                                )
                            ),
                            title="Market Segment Analysis (Radar)",
                            template="plotly_white",
                            height=500,
                        )
                        st.plotly_chart(fig, use_container_width=True)

# ==================== DASHBOARD PAGE ====================
def show_dashboard():
    st.markdown(
        '<h2 class="sub-header">📊 Analytics Dashboard</h2>',
        unsafe_allow_html=True,
    )

    if not st.session_state.classification:
        st.warning(
            "⚠️ No analysis data available. Please complete the pipeline first!"
        )
        return

    # Model Performance
    st.markdown(
        """
        <div class="info-card">
            <h3 style="color: #667eea;">🎯 Model Performance Summary</h3>
        </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <h4>🛰️ SpaceNet Classifier</h4>
                <h2>82.94%</h2>
                <p>Classification Accuracy</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <h4>🏠 House Pricing</h4>
                <h2>99.74%</h2>
                <p>R² Score (0.9974)</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <h4>📈 Time Series LSTM</h4>
                <h2>97.85%</h2>
                <p>Prediction Accuracy</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Classification Results
    st.markdown("### 🛰️ Property Classification Results")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=st.session_state.confidence * 100,
                title={
                    "text": f"Confidence: {st.session_state.classification}"
                },
                delta={"reference": 80},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "#667eea"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "gray"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90,
                    },
                },
            )
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(
            f"""
            <div class="result-card">
                <h4>Classification</h4>
                <h2>{st.session_state.classification}</h2>
                <p>Confidence: {st.session_state.confidence*100:.2f}%</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    # House Price Results
    if st.session_state.house_predictions:
        st.markdown("---")
        st.markdown("### 🏠 House Price Predictions")

        predictions = st.session_state.house_predictions

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
                <div class="result-card">
                    <h4>Current</h4>
                    <h2>${predictions['current']:,.0f}</h2>
                </div>
            """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="result-card">
                    <h4>Year 1</h4>
                    <h2>${predictions['year1']:,.0f}</h2>
                </div>
            """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
                <div class="result-card">
                    <h4>Year 2</h4>
                    <h2>${predictions['year2']:,.0f}</h2>
                </div>
            """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"""
                <div class="result-card">
                    <h4>Year 3</h4>
                    <h2>${predictions['year3']:,.0f}</h2>
                </div>
            """,
                unsafe_allow_html=True,
            )

        # Visualization
        fig = go.Figure()

        years = ["Current", "Year 1", "Year 2", "Year 3"]
        prices = [
            predictions["current"],
            predictions["year1"],
            predictions["year2"],
            predictions["year3"],
        ]

        fig.add_trace(
            go.Scatter(
                x=years,
                y=prices,
                mode="lines+markers",
                fill="tozeroy",
                line=dict(color="#667eea", width=3),
                marker=dict(size=15, color="#764ba2"),
                name="Price Forecast",
            )
        )

        fig.update_layout(
            title="3-Year Property Value Projection",
            xaxis_title="Time Period",
            yaxis_title="Property Value ($)",
            template="plotly_white",
            height=400,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Growth Rate Analysis
        growth_rates = [
            (
                (predictions["year1"] - predictions["current"])
                / predictions["current"]
            )
            * 100,
            (
                (predictions["year2"] - predictions["year1"])
                / predictions["year1"]
            )
            * 100,
            (
                (predictions["year3"] - predictions["year2"])
                / predictions["year2"]
            )
            * 100,
        ]

        fig2 = go.Figure(
            data=[
                go.Bar(
                    x=["Year 1", "Year 2", "Year 3"],
                    y=growth_rates,
                    marker_color=["#11998e", "#667eea", "#f5576c"],
                    text=[f"{x:.2f}%" for x in growth_rates],
                    textposition="auto",
                )
            ]
        )

        fig2.update_layout(
            title="Annual Growth Rate (%)",
            yaxis_title="Growth Rate (%)",
            template="plotly_white",
            height=350,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Time Series Results
    if st.session_state.timeseries_predictions:
        st.markdown("---")
        st.markdown("### 📈 Market Forecast Results")

        prediction = st.session_state.timeseries_predictions

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"""
                <div class="result-card">
                    <h4>Current Market Value</h4>
                    <h2>${prediction['current_value']:,.0f}</h2>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="result-card">
                    <h4>Forecasted Value</h4>
                    <h2>${prediction['forecast']:,.0f}</h2>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div class="result-card">
                    <h4>Expected Change</h4>
                    <h2>{prediction['change']:+.2f}%</h2>
                </div>
            """,
                unsafe_allow_html=True,
            )

    # Download Section
    st.markdown("---")
    st.markdown("### 📥 Download & Share Reports")

    col1, col2 = st.columns(2)

    with col1:
        pdf_buffer = generate_pdf_report()
        st.download_button(
            "📄 Download PDF Report",
            pdf_buffer,
            "property_analysis_report.pdf",
            "application/pdf",
            use_container_width=True,
        )

    with col2:
        csv_data = generate_comprehensive_csv()
        st.download_button(
            "📋 Download CSV Report",
            csv_data,
            "property_complete_report.csv",
            "text/csv",
            use_container_width=True,
        )

    # Email Functionality
    st.markdown("---")
    st.markdown("### 📧 Email Report")

    st.info(
        "💡 **Note:** Configure your Gmail credentials in the code to enable email functionality"
    )

    with st.form("email_form"):
        recipient_email = st.text_input(
            "Recipient Email", placeholder="user@example.com"
        )
        send_button = st.form_submit_button(
            "📧 Send Report via Email", use_container_width=True
        )

        if send_button:
            if recipient_email and "@" in recipient_email:
                with st.spinner("📤 Sending email..."):
                    pdf_buf = generate_pdf_report()
                    csv_buf = generate_comprehensive_csv()

                    success = send_email_report(
                        recipient_email, pdf_buf, csv_buf
                    )

                    if success:
                        st.success(
                            f"✅ Report sent successfully to {recipient_email}!"
                        )
                    else:
                        st.error(
                            "❌ Email sending failed. Please check configuration."
                        )
            else:
                st.error("❌ Please enter a valid email address")

# ==================== MAIN APP ====================
def main():
    # Initialize auth
    initialize_auth()

    # If not authenticated, show login and stop
    if not st.session_state.authenticated:
        show_login_page()
        return

    # Sidebar
    show_logout_button()

    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 1rem;">
            <h1 style="color: #ffffff; font-size: 1.8rem;">🏘️ Navigation</h1>
        </div>
    """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "",
        ["🏠 Home", "🔄 Pipeline", "📊 Dashboard"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="background: rgba(255, 255, 255, 0.1); 
                    padding: 1.5rem; border-radius: 10px; 
                    border: 1px solid rgba(255, 255, 255, 0.2);">
            <h3 style="color: #ffffff; margin-bottom: 1rem;">ℹ️ About</h3>
            <p style="color: #e0e0e0; font-size: 0.9rem; line-height: 1.6;">
                AI platform that evaluates property conditions, predicts price trends, 
                and segments satellite images of real estate regions. This system is 
                useful for property buyers, investors, and urban planners.
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <p style="text-align: center; color: #888888; font-size: 0.85rem;">
            <strong>v5.0 Pro</strong><br>
            Advanced Features Enabled
        </p>
    """,
        unsafe_allow_html=True,
    )

    # Route to pages
    if page == "🏠 Home":
        show_homepage()
    elif page == "🔄 Pipeline":
        show_pipeline()
    elif page == "📊 Dashboard":
        show_dashboard()

if __name__ == "__main__":
    main()
