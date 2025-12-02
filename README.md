<div align="center">

# 🏠 RealtyAI - Property Prediction Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**An end-to-end machine learning application for property classification, price forecasting, and valuation prediction**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Models](#-models) • [Contributing](#-contributing)

<img src="https://via.placeholder.com/800x400/2E86AB/FFFFFF?text=RealtyAI+Dashboard" alt="RealtyAI Dashboard" width="800"/>

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Models](#-models)
- [Project Structure](#-project-structure)
- [Sample Data](#-sample-data)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🌟 Overview

**RealtyAI** is a comprehensive property analysis platform that combines three powerful deep learning models into a seamless workflow:

1. **U-Net Segmentation** - Classifies properties as Residential or Commercial
2. **LSTM Forecasting** - Predicts future property price trends
3. **XGBoost Regression** - Estimates accurate property valuations

The application features an intuitive **Streamlit interface** and generates professional **PDF reports** with visualizations and detailed analysis [web:9].

---

## ✨ Features

### 🏗️ Multi-Model Pipeline
- **U-Net Segmentation**: Analyze property images with 256×256 pixel segmentation masks
- **LSTM Time Series**: Year-wise price forecasting with historical trend comparison
- **XGBoost Prediction**: 50+ feature-based property valuation

### 🎨 User Experience
- ✅ Side-by-side image comparison (original vs. segmented)
- ✅ Interactive Plotly graphs with historical vs. forecast visualization
- ✅ Real-time prediction with confidence scores
- ✅ Downloadable sample Excel templates for LSTM input

### 📄 Professional Reporting
- ✅ PDF export with embedded images and graphs
- ✅ Input data blueprint with feature summary tables
- ✅ Comprehensive analysis across all three models
- ✅ Year-wise forecast breakdown tables

### 🔧 Technical Highlights
- Built with **Streamlit** for rapid prototyping
- **TensorFlow/Keras** for deep learning models
- **XGBoost** for gradient boosting regression
- **ReportLab** for PDF generation
- **Plotly** for interactive visualizations

---

## 🏛️ Architecture

