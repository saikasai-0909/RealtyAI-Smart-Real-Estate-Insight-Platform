# 🏡 RealtyAI - Smart Real Estate Insight Platform

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.50.0-FF4B4B.svg)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.19.1-FF6F00.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

> **AI-powered real estate analysis platform** combining satellite image segmentation, house price prediction, and market trend forecasting into one comprehensive solution.

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Docker](#-docker-deployment) • [Documentation](#-documentation)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
  - [Local Setup](#local-setup)
  - [Docker Setup](#docker-setup)
- [Usage Guide](#-usage-guide)
- [Manual Testing Checklist](#-manual-testing-checklist)
- [Project Structure](#-project-structure)
- [Models](#-models)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🌟 Overview

RealtyAI is an end-to-end AI platform that revolutionizes real estate analysis by integrating three powerful machine learning models:

1. **🛰️ Satellite Image Segmentation** - Analyze land use patterns from satellite imagery
2. **💰 House Price Prediction** - Predict property prices with 38+ features
3. **📈 Market Trend Forecasting** - Forecast future market trends using time series analysis

All packaged in a beautiful, user-friendly **single-page flow application** with comprehensive PDF reporting.

---

## ✨ Features

### 🛰️ **Satellite Image Analysis**
- **U-Net Semantic Segmentation** for building detection
- **Multi-image processing** with batch upload support
- **Automatic classification**: Commercial, Residential, Suburban, Rural
- **Detailed metrics**: Building count, coverage percentage, structure size
- **Visual comparison**: Original image + segmentation mask side-by-side
- **Export results**: Individual and batch analysis reports

### 💰 **House Price Prediction**
- **XGBoost regression model** with 38 property features
- **Dual input modes**:
  - Manual entry with intelligent form validation
  - Batch CSV upload for multiple properties
- **Real-time predictions** with instant results
- **Statistical analysis**: Mean, median, min, max prices
- **Interactive visualizations**: Price distribution charts
- **Top/Bottom performers**: Identify high and low-value properties
- **CSV export**: Download predictions with original data

### 📈 **Market Trend Forecasting**
- **LSTM neural networks** for time series prediction
- **Region-based forecasting** with customizable parameters
- **Realistic price variations**: Incorporates trend, seasonality, and volatility
- **Configurable forecast horizon**: 10-90 steps ahead
- **Interactive charts**: Historical vs. predicted prices with Plotly
- **Trend analysis**: Upward, downward, or stable trend detection
- **Multiple regions support**: Forecast multiple markets simultaneously
- **CSV export**: Download forecast data for external analysis

### 📊 **Comprehensive Reporting**
- **PDF generation** with professional formatting
- **Includes**:
  - Original satellite images and segmentation masks
  - Detailed property price predictions
  - Market trend forecast charts
  - Statistical summaries and insights
- **Downloadable formats**: PDF reports and CSV data
- **Session persistence**: All data saved across navigation

### 🎨 **User Experience**
- **Guided workflow**: Step-by-step 4-page flow
- **Modern UI**: Gradient backgrounds, cards, and animations
- **Responsive design**: Works on desktop and tablet
- **Real-time feedback**: Progress bars and loading indicators
- **Error handling**: Graceful error messages and validation
- **Session state**: Navigate freely without losing data

---

## 🛠️ Tech Stack

### **Frontend**
- **Streamlit 1.50.0** - Web application framework
- **Plotly 6.3.0** - Interactive visualizations
- **Custom CSS** - Modern, gradient-based styling

### **Backend / ML**
- **TensorFlow 2.19.1** - Deep learning (U-Net, LSTM)
- **Keras 3.11.2** - Neural network API
- **XGBoost 2.0.3** - Gradient boosting (price prediction)
- **PyTorch 2.5.1** - Additional ML capabilities
- **Scikit-learn 1.7.2** - ML utilities and preprocessing

### **Data Processing**
- **Pandas 2.3.3** - Data manipulation
- **NumPy 1.26.4** - Numerical operations
- **OpenCV 4.7.0** - Image processing
- **Pillow 11.3.0** - Image handling

### **Visualization & Reports**
- **ReportLab 4.4.3** - PDF generation
- **Kaleido 1.2.0** - Static image export from Plotly
- **Matplotlib 3.8.0** - Additional plotting
- **Seaborn 0.13.2** - Statistical visualizations

### **Deployment**
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration

---

## 🚀 Installation

### Prerequisites

- **Python 3.9+**
- **pip** (Python package manager)
- **Git** (for cloning repository)
- **8GB+ RAM** recommended
- **10GB+ free disk space**

### Local Setup

#### 1. Clone Repository

```bash
git clone https://github.com/yourusername/realty-ai.git
cd realty-ai
```

#### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: TensorFlow and PyTorch are large packages (~2GB). Installation may take 10-15 minutes.

#### 4. Download Models

Place the following model files in the `models/` directory:

- `unet_final.keras` (~200MB) - Satellite segmentation
- `xgb_final.pkl` (~50MB) - Price prediction
- `xgb_encoders.pkl` (~5MB) - Feature encoders
- `lstm_region_price_model.h5` (~100MB) - Market forecast
- `price_scalers.pkl` (~5MB) - Price scalers

```
models/
├── unet_final.keras
├── xgb_final.pkl
├── xgb_encoders.pkl
├── lstm_region_price_model.h5
└── price_scalers.pkl
```

**Model files not included in repository due to size. Contact maintainer or train your own.**

#### 5. Run Application

```bash
streamlit run app.py
```

Application will open at `http://localhost:8501`

---

### Docker Setup

#### Prerequisites
- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
- **Docker Compose 2.0+**
- **8GB+ RAM** allocated to Docker

#### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/yourusername/realty-ai.git
cd realty-ai

# 2. Ensure models are in models/ directory

# 3. Build Docker image
docker-compose build

# 4. Run container
docker-compose up -d

# 5. Access application
# Open browser: http://localhost:8501

# 6. View logs
docker-compose logs -f

# 7. Stop container
docker-compose down
```

#### Docker Configuration

**Default settings** (in `docker-compose.yml`):
- **Port**: 8501
- **Memory**: 8GB limit, 4GB reserved
- **CPU**: 4 cores limit, 2 cores reserved
- **Restart policy**: Unless stopped
- **Health checks**: Enabled

**Customize** by editing `docker-compose.yml`:

```yaml
# Change port
ports:
  - "8080:8501"  # Use port 8080

# Increase memory
deploy:
  resources:
    limits:
      memory: 16G  # 16GB limit
```

---

## 📖 Usage Guide

### Application Workflow

```
🏠 Home → 📍 Satellite Analysis → 💰 Price Prediction → 📈 Market Forecast → 📊 Summary Report
```

### Step-by-Step Guide

#### **Step 1: Home Page**

- Welcome screen with feature overview
- Click **"Start Analysis"** to begin

#### **Step 2: Satellite Image Analysis**

1. **Upload Images**:
   - Click file uploader
   - Select one or multiple satellite images (JPG, PNG)
   - Images automatically displayed in grid

2. **Analyze**:
   - Click **"Analyze All Images"**
   - Wait for processing (2-5 seconds per image)
   - View results:
     - Original image vs. segmentation mask
     - Building count
     - Average structure size
     - Coverage percentage
     - Land classification

3. **Navigate**: Click **"Next: Price Prediction"** →

#### **Step 3: House Price Prediction**

**Option A: Manual Input**

1. Fill in property details across three columns:
   - **Column 1**: Quality, areas, year built, bedrooms
   - **Column 2**: Floor areas, garage, lot size
   - **Column 3**: Basement, rooms, masonry, outdoor spaces

2. Select categorical features:
   - Neighborhood, quality ratings
   - Exterior materials, basement type
   - Zoning, conditions

3. Click **"Predict Price"**
4. View predicted price and analytics

**Option B: Batch CSV Upload**

1. Click **"Batch CSV Upload"** tab
2. Upload CSV with required 38 columns
3. Click **"Predict Batch Prices"**
4. View results table
5. Download predictions CSV

**Required CSV columns**:
```
OverallQual, GrLivArea, TotalBsmtSF, 2ndFlrSF, BsmtFinSF1, GarageCars,
1stFlrSF, GarageArea, LotArea, YearBuilt, YearRemodAdd, Neighborhood,
FullBath, TotRmsAbvGrd, MasVnrArea, OpenPorchSF, WoodDeckSF, BsmtUnfSF,
BsmtQual, OverallCond, KitchenQual, Fireplaces, MoSold, MSZoning,
CentralAir, BsmtExposure, MSSubClass, ExterQual, Exterior1st, BsmtFinType1,
YrSold, BedroomAbvGr, LandContour, SaleCondition, LotShape, HalfBath,
Exterior2nd, LotConfig
```

**Navigate**: Click **"Next: Market Forecast"** →

#### **Step 4: Market Trend Forecast**

1. **Upload Data**:
   - Upload CSV with historical prices
   - Required columns: `Date`, `RegionName`, `ZHVI_AllHomes`

2. **Configure**:
   - Select region from dropdown
   - Choose forecast steps (10-90)
   - Set historical display range (30-180)

3. **Generate**:
   - Click **"Generate Forecast"**
   - View interactive chart
   - See statistics (last price, avg forecast, max, min)
   - Read trend analysis

4. **Download**:
   - Download forecast CSV for selected region

**Navigate**: Click **"Next: Summary Report"** →

#### **Step 5: Summary Report**

1. **Review Results**:
   - View all satellite analyses
   - Check all price predictions
   - Review all market forecasts
   - See summary statistics

2. **Generate PDF**:
   - Click **"Generate & Download PDF Report"**
   - Comprehensive report includes:
     - All satellite images and masks
     - Price prediction tables
     - Forecast charts
     - Statistical summaries

3. **Options**:
   - **Start New Analysis**: Reset and start over
   - **Back to Forecast**: Return to previous step
   - **Home**: Return to homepage

---
## 🧪 Manual Testing Checklist

- [ ] Upload and analyze satellite image from the test_data u may upload 1 or more images.
- [ ] Manual price prediction has default values can also be modified 
- [ ] Batch CSV upload and prediction: The batch upload tab can be used to predict csv file with columns  
- [ ] Forecast generation: Upload the ZILLOW_FINAL_TEST.csv and select the city that need to be forecasted.
- [ ] PDF report download
- [ ] Navigation between pages
- [ ] Session state persistence

## 📁 Project Structure

```
realty-ai/
│
├── app.py                          # Main application (single file)
│   ├── Model loading functions
│   ├── Helper functions
│   ├── Page navigation logic
│   ├── Segmentation page
│   ├── Price prediction page
│   ├── Forecast page
│   ├── Summary & PDF generation
│   └── Complete UI/UX
│
├── models/                         # AI model files
│   ├── unet_final.keras           # Satellite segmentation
│   ├── xgb_final.pkl              # Price prediction
│   ├── xgb_encoders.pkl           # Feature encoders
│   ├── lstm_region_price_model.h5 # Market forecast
│   └── price_scalers.pkl          # Price scalers
│
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker image config
├── docker-compose.yml              # Docker compose config
├── .dockerignore                   # Docker build exclusions
├── .gitignore                      # Git exclusions
│
├── test/                           
│   ├── 00014.png,00015.png,00016.png
│   ├── test_predictions.csv
│   └── ZILLOW_FINAL_TEST.csv
│
└── README.md                       
```

---

## 🤖 Models

### 1. Satellite Segmentation Model

**Architecture**: U-Net CNN
- **Input**: 256×256×3 RGB satellite images
- **Output**: 256×256 binary segmentation mask
- **Purpose**: Building detection and land use classification
- **Training**: Custom satellite imagery dataset
- **Performance**: High accuracy on urban/suburban areas

### 2. Price Prediction Model

**Algorithm**: XGBoost Gradient Boosting
- **Input**: 38 property features (numeric + categorical)
- **Output**: Predicted sale price (USD)
- **Features**:
  - Property characteristics (size, quality, age)
  - Location (neighborhood, zoning)
  - Amenities (garage, basement, outdoor spaces)
- **Training**: Ames Housing Dataset
- **Performance**: R² score > 0.9 on test set

### 3. Market Forecast Model

**Architecture**: LSTM Neural Network
- **Input**: 30-step price sequence + region ID
- **Output**: Next period price prediction
- **Purpose**: Time series forecasting with trend and seasonality
- **Training**: Zillow housing price data (ZHVI)
- **Performance**: MAE < 5% on validation set

---

## 🔧 Configuration

### Environment Variables

Optional configuration via `.env` file:

```bash
# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# TensorFlow Configuration
TF_CPP_MIN_LOG_LEVEL=2
TF_ENABLE_ONEDNN_OPTS=0

# Application Configuration
MAX_UPLOAD_SIZE=200  # MB
SESSION_TIMEOUT=3600  # seconds
```

### Model Paths

Update in `app.py` if models are in different location:

```python
@st.cache_resource
def load_all_models():
    models = {}
    models['segmentation'] = load_model("YOUR_PATH/unet_final.keras")
    # ... etc
```

---

## 📊 API Reference

### Core Functions

#### Satellite Segmentation

```python
def load_image_array(image):
    """Preprocess image for segmentation model"""
    # Returns: 256x256x3 normalized array

def analyze_mask(mask):
    """Extract metrics from segmentation mask"""
    # Returns: building_count, avg_size, total_area, coverage_pct

def classify_area(building_count, avg_size, coverage_pct):
    """Classify land use type"""
    # Returns: classification, description
```

#### Price Prediction

```python
def encode_input(input_dict, encoders):
    """Encode categorical features"""
    # Returns: encoded feature dictionary

def predict_price(model, features):
    """Predict house price"""
    # Returns: predicted_price (float)
```

#### Market Forecast

```python
def forecast_region(model, scalers, df, region_name, future_steps):
    """Generate market forecast for region"""
    # Returns: region_data, predictions, future_dates
```

#### PDF Generation

```python
def generate_comprehensive_pdf():
    """Create PDF report with all results"""
    # Returns: BytesIO buffer with PDF
```

---

## 🎨 Customization

### UI Styling

Modify CSS in `app.py`:

```python
st.markdown("""
<style>
    .stButton>button {
        background-color: #YOUR_COLOR;
        /* ... */
    }
</style>
""", unsafe_allow_html=True)
```

### Color Scheme

Current gradient colors:
- Primary: `#667eea` → `#764ba2` (Purple)
- Secondary: `#11998e` → `#38ef7d` (Green)
- Tertiary: `#4facfe` → `#00f2fe` (Blue)

### Feature Toggles

Enable/disable features in `app.py`:

```python
ENABLE_BATCH_UPLOAD = True
ENABLE_CLUSTERING = True
ENABLE_PDF_DOWNLOAD = True
```

---
---

## 🚀 Deployment


#### DigitalOcean
```bash
# Most cost-effective: $48/month
# 4GB RAM, 2 vCPUs droplet
```

**Complete deployment guides**: See `docs/` directory

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Getting Started

1. Fork the repository
2. Create feature branch: `git checkout -b feature/AmazingFeature`
3. Make changes
4. Commit: `git commit -m 'Add AmazingFeature'`
5. Push: `git push origin feature/AmazingFeature`
6. Open Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/Suriyaskrs/realty-ai.git
cd realty-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run in development mode
streamlit run app.py
```

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to functions
- Comment complex logic
- Keep functions focused and small
- Use meaningful variable names

### Pull Request Process

1. Update README.md with any new features
2. Update documentation in `docs/`
3. Add tests for new features
4. Ensure all tests pass
5. Update CHANGELOG.md

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 RealtyAI Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📧 Contact

**Project Maintainer**: Your Name
- GitHub: [@Suriyaskrs](https://github.com/Suriyaskrs)
- Email: suriyaskrs@gmail.com

**Project Link**: [https://github.com/Suriyaskrs/realty-ai](https://github.com/Suriyaskrs/realty-ai)

---

## 🙏 Acknowledgments

- **Infosys Springboard** - For this internship opportunity
- **Streamlit** - For the amazing web framework
- **TensorFlow Team** - For deep learning tools
- **XGBoost Contributors** - For the gradient boosting library
- **Zillow** - For housing price data (ZHVI)
- **Ames Housing Dataset** - For training data
- **Open Source Community** - For inspiration and support

---

## 📈 Roadmap

### Current Version: v1.0.0

### Planned Features

- [ ] **Authentication system** - User login and data privacy
- [ ] **Database integration** - PostgreSQL for persistent storage
- [ ] **Real-time data feeds** - Live market data integration
- [ ] **Comparative analysis** - Compare multiple properties/regions
- [ ] **Mobile app** - React Native mobile version
- [ ] **API endpoints** - RESTful API for external integration
- [ ] **Advanced visualizations** - 3D plots and interactive maps
- [ ] **Multi-language support** - i18n for global users
- [ ] **Custom model training** - UI for retraining models
- [ ] **Automated reports** - Scheduled report generation

### Version History

- **v1.0.0** (2025-12) - Initial release
  - Satellite image segmentation
  - House price prediction
  - Market trend forecasting
  - PDF report generation
  - Docker support

---

## 🐛 Known Issues

- PDF chart generation requires kaleido package
- Large CSV files (>10,000 rows) may cause memory issues
- TensorFlow warnings on CPU (expected, not errors)
- First prediction takes longer (model loading time)

See [Issues](https://github.com/Suriyaskrs/realty-ai/issues) for complete list.

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---



<div align="center">

**Built with ❤️ using Streamlit, TensorFlow, and XGBoost**

[⬆ Back to Top](#-realtyai---smart-real-estate-insight-platform)

</div>