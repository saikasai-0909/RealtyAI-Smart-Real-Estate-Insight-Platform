
import os
import io
import math
from functools import lru_cache
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import joblib
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# -------------------------
# Config & constants
# -------------------------
st.set_page_config(page_title="🏠 RealtyAI", layout="wide", initial_sidebar_state="expanded")


# Pixel scale chosen: B -> 0.5 meters per pixel
PIXEL_TO_M = 0.5  # meters per pixel
M2_TO_SQFT = 10.7639

# Model file names (must be in same folder)
SEG_MODEL_PATH = "unet_model.h5"
REG_MODEL_PATH = "xgb_model.joblib"
CITY_MODEL_PATH = "city_time_series.joblib"

# Sample images available in workspace (from conversation assets)
SAMPLE_IMAGE_1 = "/mnt/data/86137356-cc79-4f7f-81f4-32e7d4931f0b.png"
SAMPLE_IMAGE_2 = "/mnt/data/a677cec0-b43b-4364-84f4-51740af3512b.png"
SAMPLE_IMAGE_3 = "/mnt/data/fcef0bd4-4159-4dd7-bed3-a9dad3bcc110.png"

# -------------------------
# Utility / helper functions
# -------------------------
@st.cache_resource
def load_unet(path):
    if os.path.exists(path):
        try:
            from tensorflow.keras.models import load_model as km_load
            return km_load(path)
        except Exception as e:
            st.warning(f"Segmentation model failed to load: {e}")
            return None
    return None

@st.cache_resource
def load_reg(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Regression model failed to load: {e}")
            return None
    return None

@st.cache_resource
def load_city(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"City time-series model failed to load: {e}")
            return None
    return None

def read_image(input_obj):
    if isinstance(input_obj, str) and os.path.exists(input_obj):
        img = Image.open(input_obj).convert("RGB")
    else:
        img = Image.open(input_obj).convert("RGB")
    return np.array(img)

def seg_predict_mask(img_arr, seg_model):
    # expects seg_model to output probabilities map scaled 0..1, single channel
    try:
        h, w = img_arr.shape[:2]
        resized = cv2.resize(img_arr, (128, 128))
        inp = np.expand_dims(resized / 255.0, axis=0)
        pred = seg_model.predict(inp)[0]
        if pred.ndim == 3 and pred.shape[-1] > 1:
            pred = np.mean(pred, axis=-1)
        mask = (pred > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (w, h))
        return mask
    except Exception as e:
        st.warning(f"Segmentation inference issue: {e}")
        return np.zeros((img_arr.shape[0], img_arr.shape[1]), dtype=np.uint8)

def heuristic_mask(img_arr):
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 51, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mask = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.medianBlur(mask, 5)
    return mask

def contour_stats(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if not areas:
        return {"count":0, "avg":0.0, "max":0.0, "contours":contours, "areas":areas}
    return {"count":len(areas), "avg":float(np.mean(areas)), "max":float(np.max(areas)), "contours":contours, "areas":areas}

def pixel_area_to_sqft(pixel_count):
    # pixel_count * (px_to_m)^2 -> m^2 -> sqft
    m2 = pixel_count * (PIXEL_TO_M ** 2)
    return m2 * M2_TO_SQFT

def estimate_floors_from_ratio(footprint_sqft, grlivarea_sqft):
    # heuristic: floors = round(grlivarea / footprint)
    if footprint_sqft <= 0:
        return 1
    ratio = grlivarea_sqft / (footprint_sqft + 1e-6)
    floors = max(1, int(round(ratio)))
    return floors

def simple_condition_from_texture(mask, img_arr):
    # rough heuristic: many edges -> older/poor condition; few edges -> good
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # compute edges inside building footprints
    inside = cv2.bitwise_and(edges, edges, mask=(mask>0).astype(np.uint8)*255)
    edge_density = inside.sum() / max(1, mask.sum())
    # map to 1-10 scale inversely
    if mask.sum() == 0:
        return 5
    v = max(0.00001, edge_density)
    score = 10 - min(9, int(v * 50))
    return max(1, min(10, score))

# -------------------------
# Model features list (from your model)
# This is the exact list printed earlier from model.feature_names_in_
# Use this list to build a DataFrame and reindex to it.
# -------------------------
MODEL_FEATURES = [
'Id','MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1',
'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath',
'HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF',
'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold',
'MSZoning_C (all)','MSZoning_FV','MSZoning_RH','MSZoning_RL','MSZoning_RM','Street_Grvl','Street_Pave','Alley_Grvl','Alley_Pave',
'LotShape_IR1','LotShape_IR2','LotShape_IR3','LotShape_Reg','LandContour_Bnk','LandContour_HLS','LandContour_Low','LandContour_Lvl',
'Utilities_AllPub','Utilities_NoSeWa','LotConfig_Corner','LotConfig_CulDSac','LotConfig_FR2','LotConfig_FR3','LotConfig_Inside',
'LandSlope_Gtl','LandSlope_Mod','LandSlope_Sev',
'Neighborhood_Blmngtn','Neighborhood_Blueste','Neighborhood_BrDale','Neighborhood_BrkSide','Neighborhood_ClearCr','Neighborhood_CollgCr',
'Neighborhood_Crawfor','Neighborhood_Edwards','Neighborhood_Gilbert','Neighborhood_IDOTRR','Neighborhood_MeadowV','Neighborhood_Mitchel',
'Neighborhood_NAmes','Neighborhood_NPkVill','Neighborhood_NWAmes','Neighborhood_NoRidge','Neighborhood_NridgHt','Neighborhood_OldTown',
'Neighborhood_SWISU','Neighborhood_Sawyer','Neighborhood_SawyerW','Neighborhood_Somerst','Neighborhood_StoneBr','Neighborhood_Timber',
'Neighborhood_Veenker',
'Condition1_Artery','Condition1_Feedr','Condition1_Norm','Condition1_PosA','Condition1_PosN','Condition1_RRAe','Condition1_RRAn',
'Condition1_RRNe','Condition1_RRNn','Condition2_Artery','Condition2_Feedr','Condition2_Norm','Condition2_PosA','Condition2_PosN',
'Condition2_RRAe','Condition2_RRAn','Condition2_RRNn','BldgType_1Fam','BldgType_2fmCon','BldgType_Duplex','BldgType_Twnhs','BldgType_TwnhsE',
'HouseStyle_1.5Fin','HouseStyle_1.5Unf','HouseStyle_1Story','HouseStyle_2.5Fin','HouseStyle_2.5Unf','HouseStyle_2Story','HouseStyle_SFoyer',
'HouseStyle_SLvl','RoofStyle_Flat','RoofStyle_Gable','RoofStyle_Gambrel','RoofStyle_Hip','RoofStyle_Mansard','RoofStyle_Shed',
'RoofMatl_ClyTile','RoofMatl_CompShg','RoofMatl_Membran','RoofMatl_Metal','RoofMatl_Roll','RoofMatl_Tar&Grv','RoofMatl_WdShake','RoofMatl_WdShngl',
'Exterior1st_AsbShng','Exterior1st_AsphShn','Exterior1st_BrkComm','Exterior1st_BrkFace','Exterior1st_CBlock','Exterior1st_CemntBd',
'Exterior1st_HdBoard','Exterior1st_ImStucc','Exterior1st_MetalSd','Exterior1st_Plywood','Exterior1st_Stone','Exterior1st_Stucco',
'Exterior1st_VinylSd','Exterior1st_Wd Sdng','Exterior1st_WdShing','Exterior2nd_AsbShng','Exterior2nd_AsphShn','Exterior2nd_Brk Cmn',
'Exterior2nd_BrkFace','Exterior2nd_CBlock','Exterior2nd_CmentBd','Exterior2nd_HdBoard','Exterior2nd_ImStucc','Exterior2nd_MetalSd',
'Exterior2nd_Other','Exterior2nd_Plywood','Exterior2nd_Stone','Exterior2nd_Stucco','Exterior2nd_VinylSd','Exterior2nd_Wd Sdng',
'Exterior2nd_Wd Shng','MasVnrType_BrkCmn','MasVnrType_BrkFace','MasVnrType_Stone','ExterQual_Ex','ExterQual_Fa','ExterQual_Gd',
'ExterQual_TA','ExterCond_Ex','ExterCond_Fa','ExterCond_Gd','ExterCond_Po','ExterCond_TA','Foundation_BrkTil','Foundation_CBlock',
'Foundation_PConc','Foundation_Slab','Foundation_Stone','Foundation_Wood','BsmtQual_Ex','BsmtQual_Fa','BsmtQual_Gd','BsmtQual_TA',
'BsmtCond_Fa','BsmtCond_Gd','BsmtCond_Po','BsmtCond_TA','BsmtExposure_Av','BsmtExposure_Gd','BsmtExposure_Mn','BsmtExposure_No',
'BsmtFinType1_ALQ','BsmtFinType1_BLQ','BsmtFinType1_GLQ','BsmtFinType1_LwQ','BsmtFinType1_Rec','BsmtFinType1_Unf','BsmtFinType2_ALQ',
'BsmtFinType2_BLQ','BsmtFinType2_GLQ','BsmtFinType2_LwQ','BsmtFinType2_Rec','BsmtFinType2_Unf','Heating_Floor','Heating_GasA',
'Heating_GasW','Heating_Grav','Heating_OthW','Heating_Wall','HeatingQC_Ex','HeatingQC_Fa','HeatingQC_Gd','HeatingQC_Po','HeatingQC_TA',
'CentralAir_N','CentralAir_Y','Electrical_FuseA','Electrical_FuseF','Electrical_FuseP','Electrical_Mix','Electrical_SBrkr','KitchenQual_Ex',
'KitchenQual_Fa','KitchenQual_Gd','KitchenQual_TA','Functional_Maj1','Functional_Maj2','Functional_Min1','Functional_Min2','Functional_Mod',
'Functional_Sev','Functional_Typ','FireplaceQu_Ex','FireplaceQu_Fa','FireplaceQu_Gd','FireplaceQu_Po','FireplaceQu_TA','GarageType_2Types',
'GarageType_Attchd','GarageType_Basment','GarageType_BuiltIn','GarageType_CarPort','GarageType_Detchd','GarageFinish_Fin',
'GarageFinish_RFn','GarageFinish_Unf','GarageQual_Ex','GarageQual_Fa','GarageQual_Gd','GarageQual_Po','GarageQual_TA','GarageCond_Ex',
'GarageCond_Fa','GarageCond_Gd','GarageCond_Po','GarageCond_TA','PavedDrive_N','PavedDrive_P','PavedDrive_Y','PoolQC_Ex','PoolQC_Fa',
'PoolQC_Gd','Fence_GdPrv','Fence_GdWo','Fence_MnPrv','Fence_MnWw','MiscFeature_Gar2','MiscFeature_Othr','MiscFeature_Shed','MiscFeature_TenC',
'SaleType_COD','SaleType_CWD','SaleType_Con','SaleType_ConLD','SaleType_ConLI','SaleType_ConLw','SaleType_New','SaleType_Oth','SaleType_WD',
'SaleCondition_Abnorml','SaleCondition_AdjLand','SaleCondition_Alloca','SaleCondition_Family','SaleCondition_Normal','SaleCondition_Partial'
]

# convert to list of strings (ensure)
MODEL_FEATURES = [str(x) for x in MODEL_FEATURES]

# -------------------------
# Load models
# -------------------------
seg_model = load_unet(SEG_MODEL_PATH)
reg_model = load_reg(REG_MODEL_PATH)
city_model = load_city(CITY_MODEL_PATH)

# -------------------------
# Page layout & styling
# -------------------------
PRIMARY = "#0b3d91"  # deep enterprise blue
ACCENT = "#00b0ff"
BG = "#071229"
CARD = "rgba(255,255,255,0.03)"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, {BG}, #041223);
        color: #e6eef8;
        font-family: "Segoe UI", sans-serif;
    }}
    .card {{
        background: {CARD};
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.5);
        border: 1px solid rgba(255,255,255,0.03);
    }}
    .title {{
        color: {ACCENT};
        font-weight: 700;
    }}
    .muted {{
        color: rgba(230,238,248,0.7);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Sidebar: step progress + model status
# -------------------------
st.sidebar.markdown("### 🧭 RealtyAI")
st.sidebar.markdown("*Steps:*")
step = st.session_state.get("step", 1)
st.sidebar.markdown(f"1. Image → Classify {'✅' if step>1 else '🔒'}")
st.sidebar.markdown(f"2. Predict {'✅' if step>2 else '🔒'}")
st.sidebar.markdown(f"3. Forecast {'✅' if step>2 else '🔒'}")
st.sidebar.markdown("---")
st.sidebar.markdown("*Model Status*")
st.sidebar.write(f"Segmentation: {'Loaded' if seg_model is not None else 'Missing (heuristic)'}")
st.sidebar.write(f"Regression: {'Loaded' if reg_model is not None else 'Missing'}")
st.sidebar.write(f"City model: {'Loaded' if city_model is not None else 'Missing'}")
st.sidebar.markdown("---")
st.sidebar.markdown("Pixel scale: *1 px = 0.5 m* (preset)")

# -------------------------
# Main: Header
# -------------------------
st.markdown("<h1 class='title'>🏠 RealtyAI Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<div class='muted'>Upload satellite → Auto-extract features → Predict → Forecast</div>", unsafe_allow_html=True)

st.write(" ")

# -------------------------
# STEP 1 - Image upload and segmentation/classification
# -------------------------
st.markdown("## 1) Upload Satellite Image & Auto-Extract")
col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader("Upload satellite image (jpg/png) — or choose sample", type=["jpg","jpeg","png"])
    use_sample = st.checkbox("Use sample image (demo)", value=False)
    if use_sample:
        sample_choice = st.selectbox("Sample", ["Sample 1", "Sample 2", "Sample 3"])
        if sample_choice == "Sample 1":
            uploaded = SAMPLE_IMAGE_1
        elif sample_choice == "Sample 2":
            uploaded = SAMPLE_IMAGE_2
        else:
            uploaded = SAMPLE_IMAGE_3

with col2:
    st.markdown("### Quick tips")
    st.markdown("- Use reasonably zoomed satellite images (Google Satellite style).")
    st.markdown("- Pixel scale is set to *1 px = 0.5 m* for area calculations.")

if st.button("🔍 Detect & Auto-Extract") and uploaded:
    try:
        img_arr = read_image(uploaded)
        st.image(img_arr, caption="Input image", use_column_width=True)

        # segmentation mask (prefer model)
        if seg_model is not None:
            mask = seg_predict_mask(img_arr, seg_model)
        else:
            mask = heuristic_mask(img_arr)

        stats = contour_stats(mask)
        st.write(f"Detected building-like contours: *{stats['count']}*")

        # compute footprint area from largest contour
        if stats["count"] > 0:
            # choose largest contour
            largest_idx = int(np.argmax(stats["areas"]))
            largest_contour = stats["contours"][largest_idx]
            footprint_px = cv2.contourArea(largest_contour)
            footprint_sqft = pixel_area_to_sqft(footprint_px)
        else:
            footprint_px = 0
            footprint_sqft = 0.0

        # simple land area: bounding box of non-zero mask area
        ys, xs = np.where(mask > 0)
        if ys.size > 0:
            minx, maxx = xs.min(), xs.max()
            miny, maxy = ys.min(), ys.max()
            bbox_px = (maxx - minx) * (maxy - miny)
            lot_sqft = pixel_area_to_sqft(bbox_px)
        else:
            lot_sqft = 0.0

        # attempt to estimate grlivarea (approx) as footprint * floors_est (we'll set floors heuristic)
        # fallback: approximate GrLivArea ~ footprint * 1 (single floor) until user enters real value
        grliv_est_sqft = max(footprint_sqft, 200.0)

        # estimate floors (rounded ratio)
        floors_est = estimate_floors_from_ratio(footprint_sqft, grliv_est_sqft)

        # simple texture-based condition estimate
        cond_est = simple_condition_from_texture(mask, img_arr)

        # classify zone: heuristic using density & footprint size (or use classifier model if present)
        zone = "Residential"
        if stats["count"] == 0:
            zone = "Unknown"
        else:
            # if many large footprints -> commercial
            if stats["count"] >= 6 or footprint_sqft > 2500:
                zone = "Commercial"
            else:
                zone = "Residential"

        # overlay display: mask + contours
        overlay = img_arr.copy()
        # mask tint
        colored_mask = np.zeros_like(img_arr)
        colored_mask[:, :, 1] = (mask > 0).astype(np.uint8) * 100  # greenish mask
        overlay = cv2.addWeighted(overlay, 0.8, colored_mask, 0.4, 0)
        # draw contours in green
        if stats["contours"]:
            cv2.drawContours(overlay, stats["contours"], -1, (0,255,0), 2)

        st.image(overlay, caption="Segmentation mask + contours", use_column_width=True)
        # --- Display classification zone immediately below segmentation ---
        st.markdown(f"### 🏷️ Classified Zone: **{zone}**")

        # save auto-extracted results to session
        st.session_state.auto = {
            "footprint_px": footprint_px,
            "footprint_sqft": float(footprint_sqft),
            "lot_sqft": float(lot_sqft),
            "grliv_est_sqft": float(grliv_est_sqft),
            "floors_est": int(floors_est),
            "cond_est": int(cond_est),
            "zone": zone
        }
        st.success("Auto-extraction complete.")
        # unlock step 2
        st.session_state.step = 2

    except Exception as e:
        st.error(f"Auto-extract failed: {e}")
else:
    if not uploaded:
        st.info("Upload an image and click 'Detect & Auto-Extract' to start.")
    else:
        st.info("Click 'Detect & Auto-Extract' to extract features from the image.")

# -------------------------
# STEP 2 - Compact Smart UI for inputs + prediction
# -------------------------
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 2) House Price Prediction")

if st.session_state.get("step",1) < 2:
    st.info("Complete Step 1 to unlock this section.")
else:
    auto = st.session_state.get("auto", {})

    # Layout columns
    c1, c2 = st.columns([2,1])

    # Right-side auto-extracted card
    with c2:
        st.markdown("### ⚙️ Auto-extracted (AC-Pro)")
        st.write(f"*Zone:* {auto.get('zone','-')}")
        st.write(f"*Footprint (sqft):* {auto.get('footprint_sqft',0):,.0f}")
        st.write(f"*Lot (sqft):* {auto.get('lot_sqft',0):,.0f}")
        st.write(f"*Floors (est):* {auto.get('floors_est',1)}")
        st.write(f"*Condition (1-10):* {auto.get('cond_est',5)}")
        st.write("You can override any auto value below before prediction.")

    # Left: File Upload + Form
    with c1:

        # ---------------------- FILE UPLOAD (CSV / Excel) ----------------------
        st.subheader("📂 Upload CSV or Excel (Optional)")

        uploaded_step2 = st.file_uploader(
            "Upload file containing property details",
            type=["csv", "xlsx"],
            help="Upload a CSV or Excel file with GrLivArea, BedroomAbvGr, LotArea, etc."
        )

        auto_filled = {}

        if uploaded_step2 is not None:
            try:
                if uploaded_step2.name.endswith(".csv"):
                    df_up = pd.read_csv(uploaded_step2)
                else:
                    df_up = pd.read_excel(uploaded_step2)

                st.success("File uploaded successfully!")
                st.write("Preview of uploaded data:")
                st.dataframe(df_up.head())

                # Use first row as autofill
                row = df_up.iloc[0]

                for col in ["GrLivArea", "BedroomAbvGr", "FullBath", "GarageCars",
                            "LotArea", "YearBuilt", "OverallQual"]:
                    if col in row:
                        auto_filled[col] = float(row[col])

                st.info("Auto-filled values loaded. You can still edit them below.")
            except Exception as e:
                st.error(f"Error reading file: {e}")

        # ---------------------- INPUT FORM ----------------------
        with st.form("predict_form"):
            st.markdown("### Enter / Confirm Property Details")

            GrLivArea = st.number_input(
                "Living area (sq ft) - GrLivArea",
                value=int(auto_filled.get("GrLivArea", auto.get("grliv_est_sqft", 1500))),
                min_value=100
            )

            BedroomAbvGr = st.number_input(
                "Bedrooms - BedroomAbvGr",
                value=int(auto_filled.get("BedroomAbvGr", 3)),
                min_value=0
            )

            FullBath = st.number_input(
                "Full bathrooms - FullBath",
                value=int(auto_filled.get("FullBath", 2)),
                min_value=0
            )

            GarageCars = st.number_input(
                "Garage cars - GarageCars",
                value=int(auto_filled.get("GarageCars", 2)),
                min_value=0
            )

            LotArea = st.number_input(
                "Lot area (sq ft) - LotArea",
                value=int(auto_filled.get("LotArea", auto.get("lot_sqft", 10000))),
                min_value=0
            )

            YearBuilt = st.number_input(
                "Year built - YearBuilt",
                value=int(auto_filled.get("YearBuilt", 2000)),
                min_value=1800,
                max_value=2100
            )

            OverallQual = st.slider(
                "Overall quality (1-10) - OverallQual",
                1, 10,
                value=int(auto_filled.get("OverallQual", round(auto.get("cond_est", 5))))
            )

            Region = st.selectbox("Region (for forecasting)", ["CA","NY","TX","Other"], index=0)

            neighborhood_choice = st.selectbox(
                "Neighborhood (optional)",
                ["None","NAmes","Edwards","CollgCr","OldTown","NridgHt","Somerst"],
                index=0
            )

            submit = st.form_submit_button("🔮 Predict Price")

        # ---------------------- PREDICTION LOGIC ----------------------
        if submit:
            st.session_state.step = max(st.session_state.step, 3)

            X = pd.DataFrame(np.zeros((1, len(MODEL_FEATURES))), columns=MODEL_FEATURES, dtype=float)

            mapping = {
                "GrLivArea":"GrLivArea",
                "BedroomAbvGr":"BedroomAbvGr",
                "FullBath":"FullBath",
                "GarageCars":"GarageCars",
                "LotArea":"LotArea",
                "YearBuilt":"YearBuilt",
                "OverallQual":"OverallQual"
            }

            ui_values = {
                "GrLivArea": GrLivArea,
                "BedroomAbvGr": BedroomAbvGr,
                "FullBath": FullBath,
                "GarageCars": GarageCars,
                "LotArea": LotArea,
                "YearBuilt": YearBuilt,
                "OverallQual": OverallQual
            }

            for ui_k, model_k in mapping.items():
                if model_k in X.columns:
                    X.at[0, model_k] = ui_values[ui_k]

            # Auto-calculated engineered features
            if "TotalBsmtSF" in X.columns:
                X.at[0, "TotalBsmtSF"] = max(0, int(GrLivArea * 0.25))
            if "1stFlrSF" in X.columns:
                X.at[0, "1stFlrSF"] = max(0, int(GrLivArea * 0.6))
            if "2ndFlrSF" in X.columns:
                X.at[0, "2ndFlrSF"] = max(0, int(GrLivArea - X.at[0, "1stFlrSF"]))

            if neighborhood_choice != "None":
                col_name = f"Neighborhood_{neighborhood_choice}"
                if col_name in X.columns:
                    X.at[0, col_name] = 1.0

            import datetime
            now = datetime.datetime.now()
            if "YrSold" in X.columns:
                X.at[0, "YrSold"] = now.year
            if "MoSold" in X.columns:
                X.at[0, "MoSold"] = now.month

            X = X.fillna(0.0).astype(float)

            # Predict
            if reg_model is not None:
                try:
                    pred = reg_model.predict(X)[0]

                    st.success(f"🏷️ Predicted Property Price: ₹ {pred:,.2f}")
                    st.session_state.predicted_price = float(pred)

                    k1, k2, k3 = st.columns(3)
                    k1.metric("Predicted Price", f"₹ {pred:,.0f}")
                    k2.metric("Zone", auto.get("zone", "Unknown"))
                    k3.metric("Estimated Floors", auto.get("floors_est", 1))

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.warning("Model feature mismatch — using fallback heuristic")

                    pred = 50 * GrLivArea + 10000 * OverallQual + 5000 * BedroomAbvGr + 3000 * GarageCars
                    st.success(f"🏷️ Heuristic Predicted Price: ₹ {pred:,.2f}")
                    st.session_state.predicted_price = float(pred)

            else:
                st.warning("Regression model not loaded; using fallback heuristic.")
                pred = 50 * GrLivArea + 10000 * OverallQual + 5000 * BedroomAb

# -------------------------
# --------------------------
# STEP 3: Forecasting & Trend Analysis
# --------------------------

st.markdown("---")
st.header("Step 3: Forecast Price Trend & Region Selection")

if st.session_state.get("step", 0) < 3:
    st.info("Complete Steps 1 & 2 to unlock forecasting.")
else:

    st.markdown(
        "**Optional:** Upload a historical *city/region* time-series CSV to generate a forecast.<br>"
        "If not uploaded, a default projection will be used.",
        unsafe_allow_html=True
    )

    uploaded_csv = st.file_uploader(
        "Upload city/region time-series CSV (must include year/date column)",
        type=["csv"]
    )

    forecast_years = st.slider("Forecast horizon (years)", 1, 10, 5)
    forecast_btn = st.button("🚀 Generate Forecast Price Trend")

    if forecast_btn:
        proj_df = pd.DataFrame()
        fig = go.Figure()
        selected_col = None   # trend column

        # -------------------------------------------
        # CASE 1: CSV PROVIDED
        # -------------------------------------------
        if uploaded_csv is not None:
            try:
                df_hist = pd.read_csv(uploaded_csv)

                if df_hist.empty:
                    st.warning("CSV is empty. Switching to default forecast.")
                    raise Exception("Empty CSV")

                # Detect time column
                time_cols = [
                    c for c in df_hist.columns
                    if any(k in c.lower() for k in ["year", "date", "time"])
                ]

                if not time_cols:
                    st.warning("No valid date/year column found. Using default projection.")
                    raise Exception("No date column")

                time_col = time_cols[0]
                df_hist[time_col] = pd.to_datetime(df_hist[time_col], errors="coerce")
                df_hist["Year"] = df_hist[time_col].dt.year

                if df_hist["Year"].isna().all():
                    st.warning("Failed to extract Year. Using default projection.")
                    raise Exception("Invalid year format")

                # Detect numeric columns
                numeric_cols = df_hist.select_dtypes(include=["float", "int"]).columns.tolist()
                numeric_cols = [c for c in numeric_cols if c.lower() not in ["year"]]

                if not numeric_cols:
                    st.warning("No numeric columns available. Using default projection.")
                    raise Exception("No numeric column")

                # Region Trend Selector
                selected_col = st.selectbox(
                    "Select region trend column for forecasting",
                    numeric_cols,
                    index=0
                )

                if selected_col is None:
                    st.warning("Please select a trend column. Using default projection.")
                    raise Exception("No trend column selected")

                # Ensure enough data
                if df_hist["Year"].nunique() < 2:
                    st.warning("Too few data points. Using default projection.")
                    raise Exception("Insufficient data")

                # Linear Regression
                lr = LinearRegression()
                lr.fit(df_hist[["Year"]], df_hist[selected_col])

                max_year = int(df_hist["Year"].max())
                future_years = np.array(
                    list(range(max_year + 1, max_year + forecast_years + 1))
                ).reshape(-1, 1)

                forecast_vals = lr.predict(future_years)

                # Combine historical + forecast
                all_years = list(df_hist["Year"]) + list(future_years.flatten())
                all_vals = list(df_hist[selected_col]) + list(forecast_vals)

                fig.add_trace(go.Scatter(
                    x=all_years,
                    y=all_vals,
                    mode="lines+markers",
                    name=f"{selected_col} Trend (Historical + Forecast)"
                ))

                proj_df = pd.DataFrame({
                    "Year": all_years,
                    f"{selected_col}_Forecast": all_vals
                })

            except:
                uploaded_csv = None

        # ---------------------------------------------------
        # CASE 2: DEFAULT PROJECTION (NO CSV OR INVALID CSV)
        # ---------------------------------------------------
        if uploaded_csv is None or proj_df.empty:
            start_year = pd.Timestamp.now().year

            growth = st.number_input(
                "Default annual growth rate (%)",
                value=5.0, min_value=0.0, max_value=200.0, step=0.1
            )

            years_proj = [start_year + i for i in range(forecast_years)]
            proj_vals = [100 * (1 + growth/100)**i for i in range(forecast_years)]

            fig.add_trace(go.Scatter(
                x=years_proj,
                y=proj_vals,
                mode="lines+markers",
                name="Default Price Forecast"
            ))

            proj_df = pd.DataFrame({
                "Year": years_proj,
                "Projected_Trend": proj_vals
            })
            selected_col = "Projected_Trend"

        # ---------------------------------------------------
        # SHOW CHART
        # ---------------------------------------------------
        fig.update_layout(
            title="Forecast Price Trend (Historical + Projected)",
            xaxis_title="Year",
            yaxis_title="Price",
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------------------
        # SUMMARY STATISTICS
        # ---------------------------------------------------
        st.subheader("📊 Forecast Summary Statistics")

        forecast_only = proj_df.tail(forecast_years)
        avg_val = forecast_only.iloc[:, 1].mean()
        min_val = forecast_only.iloc[:, 1].min()
        max_val = forecast_only.iloc[:, 1].max()

        st.write(
            f"""
            **Average Forecasted Price:** ₹{avg_val:,.2f}  
            **Minimum Forecasted Price:** ₹{min_val:,.2f}  
            **Maximum Forecasted Price:** ₹{max_val:,.2f}
            """
        )

        # ---------------------------------------------------
        # DATA TABLE
        # ---------------------------------------------------
        st.subheader("Trend Data Table")
        st.dataframe(proj_df)

        # Download CSV
        buf = BytesIO()
        proj_df.to_csv(buf, index=False)
        buf.seek(0)

        st.download_button(
            "📥 Download Forecast CSV",
            data=buf,
            file_name="forecast_price_trend.csv",
            mime="text/csv"
        )

