🏡 RealtyAI — Intelligent Real Estate Analysis Suite
-------------------------------------------------------------------------------------------------------------------------------

A unified AI platform for Land Type Detection, House Price Prediction, Zillow Forecasting, and Automated PDF Reports.
--------------------------------------------------------------------------------------------------------------------------------
🌟 Features
-------------------------------------------------------------------------------------------------------------------------------
✔️ 1. Land Type Classification

Uses ResNet18 (PyTorch)

Detects: Residential / Commercial

Shows confidence score

Masked edge-image included in PDF
--------------------------------------------------------------------------------------------------------------------------------
✔️ 2. House Price Prediction

Gradient Boosting Regressor

Preprocessing with Sklearn Pipeline

Inputs: Area, quality, baths, years, neighborhood, garage, lot size
----------------------------------------------------------------------------------------------------------------------------------
✔️ 3. Zillow Market Forecasting
LightGBM – next-month prediction
Prophet – multi-year forecasting

Confidence intervals

Trend graph

Editable Zillow table inside UI
----------------------------------------------------------------------------------------------------------------------------------
✔️ 4. Investment Analysis

Investment vs predicted sale

Profit/Loss

% return
---------------------------------------------------------------------------------------------------------------------------------
✔️ 5. Automated PDF Report

PDF contains:

Header + branding

Land type + confidence

House price prediction

LightGBM next-month forecast

Prophet multi-year forecast

Investment summary

Original + masked satellite image

Zillow forecast sample rows
---------------------------------------------------------------------------------------------------------------------------------------------
✔️ 6. Multi-Page Streamlit UI

Home (Main Prediction Page)

User Profile

Saved Reports

About Page

Admin Controls (Model file management)
--------------------------------------------------------------------------------------------------------------------------
🧠 Tech Stack
----------------------------------------------------------------------------------------------------------------------
RealtyAI uses Streamlit for the frontend, PyTorch (ResNet18) for image classification, Gradient Boosting + Scikit-Learn for house price prediction, LightGBM and Prophet for Zillow forecasting, OpenCV + PIL for image processing, Matplotlib for visualization, Pandas and NumPy for data handling, and ReportLab for PDF generation.


      
               
    ## 📁 Project Structure

    RealtyAI/
    │── index.py                   # Main App (Full Pipeline UI)
    │
    │── pages/                     # Multi-Page UI Screens
    │     │── User_Profile.py
    │     │── Saved_Reports.py
    │     │── About.py
    │     │── Admin_Dashboard.py
    │
    │── models/                    # ML Models
    │     │── best_model.pth                 # ResNet land classifier
    │     │── house_prices_preprocessor.pkl  # Sklearn Pipeline
    │     │── gradient_boosting_house_price.pkl
    │     │── lightgbm_zillow_model.pkl
    │     │── features.json                  # Zillow model features
    │
    │── assets/
    │     │── logo.png
    │
    │── reports/                   # Auto-saved PDFs
    │
    │── sample_data/
          │── zillow_input_24_sample.csv






      

    
--------------------------------------------------------------------------------------

💻 How to Run
--------------------------------------------------------------------------------------
1. Install Dependencies
pip install -r requirements.txt

2. Run Streamlit App
streamlit run Home.py
---------------------------------------------------------------------------------------
🎯 Future Improvements
---------------------------------------------------------------------------------------
Authentication & user accounts

Deploy to cloud (AWS / Streamlit Cloud)

More visualization features

Advanced ML options
------------------------------------------------------------------------------------------
🎉 Developed by Sahithi Mandha (2025)
