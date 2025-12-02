# 🏠 RealtyAI - Complete Property Prediction Pipeline

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**An intelligent end-to-end ML system for property analysis, price forecasting, and automated reporting**

[Features](#features) • [Architecture](#architecture) • [Installation](#installation) • [Usage](#usage) • [Models](#models) • [Screenshots](#screenshots)

</div>

---

## 🎯 Overview

RealtyAI is a production-ready machine learning application that combines computer vision, time series forecasting, and gradient boosting to deliver comprehensive property analysis [file:2]. The system provides:

- **U-Net Segmentation** for residential vs. commercial property classification from satellite/aerial imagery
- **LSTM Forecasting** for multi-year property price predictions
- **XGBoost Regression** for accurate property valuation based on 50+ features
- **Automated PDF Reports** with visualizations and detailed analytics

## ✨ Features

### 🖼️ U-Net Image Segmentation
- Deep learning-based property classification
- Residential vs. Commercial detection with confidence scores
- Real-time image preprocessing and mask generation
- Supports PNG, JPG, JPEG formats [file:2]

### 📈 LSTM Time Series Forecasting
- Multi-year price forecasting (1-20 years)
- Excel/CSV data input support
- Automatic data normalization and sequence generation
- Handles variable-length historical data [file:2]

### 💰 XGBoost Price Prediction
- **Dual input modes**: Manual entry or batch CSV/Excel upload
- 50+ engineered features including property characteristics, location, and quality metrics
- Optimized hyperparameters for accurate valuation
- Log-transform predictions for stable price estimates [file:2]

### 📄 Professional PDF Reports
- Multi-page comprehensive reports with all predictions
- Embedded visualizations and data tables
- Property feature summaries
- Export-ready format for stakeholders [file:2]


## 🚀 Installation

### Prerequisites

Python 3.8+
TensorFlow 2.x
Streamlit
XGBoost
