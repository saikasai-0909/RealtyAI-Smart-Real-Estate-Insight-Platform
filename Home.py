import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import io
import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import base64
import warnings
warnings.filterwarnings("ignore")

# Prophet safe import
try:
    from prophet import Prophet
except Exception:
    Prophet = None



# =====================================================================
#                         UTILITY FUNCTIONS
# =====================================================================
def encode_image_base64(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

def fmt_currency(value):
    try:
        return f"₹ {float(value):,.2f}"
    except:
        return str(value)

def smart_load_pickle(path):
    """Safe loader for pickle/joblib files."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return joblib.load(path)
    except:
        with open(path, "rb") as f:
            raw = f.read()
        # numpy core fix
        try:
            raw = raw.replace(b"numpy.core", b"numpy._core")
        except:
            pass
        try:
            return pickle.loads(raw)
        except:
            with open(path, "rb") as f:
                return pickle.load(f)

def make_mask_from_pil(pil_img):
    """Create a masked/edge-detected version of the uploaded image."""
    try:
        arr = np.array(pil_img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        mask_rgb = np.stack([edges]*3, axis=-1)
        return Image.fromarray(mask_rgb)
    except:
        return pil_img.convert("L")

# =====================================================================
#                       PAGE CONFIG & HEADER
# =====================================================================
st.set_page_config(
    page_title="RealtyAI — Prediction",
    page_icon="🏡",
    layout="wide"
)

# Load Logo
logo_b64 = encode_image_base64("logo.png")



# Header
st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center;">
    <div>
        <h1 style="color:#FFD580; margin-bottom:0;">🏡 RealtyAI</h1>
        <p style="color:#BBD7FF; margin-top:4px;">
            Image → Property → Investment → Zillow → Forecast → PDF
        </p>
    </div>
    <div>
        {'<img src="data:image/png;base64,'+logo_b64+'" width="90"/>' if logo_b64 else ''}
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =====================================================================
#           MODEL PATHS (Loaded from Admin Dashboard)
# =====================================================================
# If admin page hasn't set these yet → provide defaults
st.session_state.setdefault("space_model_path", "best_model.pth")
st.session_state.setdefault("hp_prep_path", "house_prices_preprocessor.pkl")
st.session_state.setdefault("hp_model_path", "gradient_boosting_house_price.pkl")
st.session_state.setdefault("z_lgb_path", "lightgbm_zillow_model.pkl")
st.session_state.setdefault("z_feat_path", "features.json")

space_model_path = st.session_state["space_model_path"]
hp_prep_path     = st.session_state["hp_prep_path"]
hp_model_path    = st.session_state["hp_model_path"]
z_lgb_path       = st.session_state["z_lgb_path"]
z_feat_path      = st.session_state["z_feat_path"]

    # # ---- SETTINGS ----
    # st.markdown("<div class='section-label'>Forecast</div>", unsafe_allow_html=True)
    # forecast_years = st.slider("Forecast years (Prophet)", 1, 5, 3)

    # # ---- CONTACT ----
    # st.markdown("<div class='section-label'>Support</div>", unsafe_allow_html=True)
    # contact = st.button("📞 Contact / Help", use_container_width=True)
    # st.markdown("""
    #     <div style="font-size:0.8rem; color:#BBD7FF; margin-top:5px;">
    #         support@realtyai.com<br>
    #         Guide: docs.realtyai.com<br>
    #     </div>
    # """, unsafe_allow_html=True)

    # # ---- FOOTER ----
    # st.markdown("""
    #     <hr style="border: 1px solid rgba(255,255,255,0.06);">
    #     <div style="font-size:0.75rem; text-align:center; color:#6B7280;">
    #         RealtyAI © 2025<br>
    #         Developed by <b>Sahithi Mandha</b>
    #     </div>
    # """, unsafe_allow_html=True)

# -------------------------
# Sample filepath uploaded earlier in this session
# -------------------------
SAMPLE_ZILLOW_PATH = "/mnt/data/zillow_input_24_sample (1).csv"

# -------------------------
# 1) Image Upload (FIRST)
# -------------------------
st.markdown('<div class="app-card">', unsafe_allow_html=True)
st.header("Step 1 — Upload Image (required)")
uploaded_image = st.file_uploader("Upload Satellite / Aerial Image (jpg/jpeg/png)", type=["jpg","jpeg","png"], key="sat_up")
if uploaded_image is None:
    st.info("Upload an image to enable land classification and produce the report.")
else:
    try:
        pil_img = Image.open(uploaded_image).convert("RGB")
        st.image(pil_img, caption="Uploaded image", use_column_width=True)
    except Exception as e:
        st.error(f"Failed to read image: {e}")
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

# -------------------------
# 2) House details + Investment
# -------------------------
st.markdown('<div class="app-card">', unsafe_allow_html=True)
st.header("Step 2 — Property Details & Investment")
col1, col2 = st.columns(2)
with col1:
    OverallQual = st.slider("Overall Quality (1-10)", 1, 10, 7)
    GrLivArea = st.number_input("Living Area (sq ft)", min_value=200, max_value=8000, value=1800)
    GarageCars = st.slider("Garage Capacity (cars)", 0, 4, 2)
    GarageArea = st.number_input("Garage Area (sq ft)", min_value=0, max_value=2000, value=480)
with col2:
    TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=4000, value=900)
    FullBath = st.slider("Full Bathrooms", 0, 5, 2)
    YearBuilt = st.number_input("Year Built", 1900, 2025, 2005)
    YearRemodAdd = st.number_input("Year Remodeled", 1900, 2025, 2010)

col3, col4 = st.columns([2,1])
with col3:
    LotArea = st.number_input("Lot Area (sq ft)", min_value=500, max_value=100000, value=8500)
    Neighborhood = st.selectbox("Neighborhood (approx)", ["CollgCr","Veenker","Crawfor","NoRidge","Mitchel","Somerst","Timber","OldTown","BrkSide","Sawyer","NAmes","Gilbert","NWAmes"])
with col4:
    investment_price = st.number_input("Investment Price (₹) — enter your buying price / invested amount", min_value=0.0, step=1000.0, value=150000.0)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------
# 3) Zillow editor (prefilled) AFTER house details
# -------------------------
st.markdown('<div class="app-card">', unsafe_allow_html=True)
st.header("Step 3 — Zillow: 24-month manual input (prefilled) — You can edit any cell")
st.markdown("Table must have `Date` (YYYY-MM-DD) and `ZHVI_AllHomes` numeric values. Prefilled with a smooth trend; edit if needed.")

def make_prefilled_30(start_value=200000, monthly_growth=0.007):
    """Generate 30 months so lag_24 can be computed"""
    end = pd.to_datetime("today").to_period("M").to_timestamp("M")
    dates = pd.date_range(end=end, periods=30, freq="M")
    vals = []
    v = start_value * (1.0 / ((1 + monthly_growth) ** 29))
    for i in range(30):
        vals.append(round(v,2))
        v = v * (1 + monthly_growth)
    return pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "ZHVI_AllHomes": vals})

zillow_upload_opt = st.file_uploader("Optional: upload a Zillow CSV to replace the table (Date,ZHVI_AllHomes)", type=["csv"], key="zillow_upload_opt")

if "zillow_editor_table" not in st.session_state:
    if zillow_upload_opt is not None:
        try:
            df_tmp = pd.read_csv(zillow_upload_opt)
            if "Date" in df_tmp.columns and "ZHVI_AllHomes" in df_tmp.columns:
                df_tmp["Date"] = pd.to_datetime(df_tmp["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
                df_tmp = df_tmp.sort_values("Date").tail(30).reset_index(drop=True)
                df_tmp["ZHVI_AllHomes"] = pd.to_numeric(df_tmp["ZHVI_AllHomes"], errors="coerce")
                st.session_state["zillow_editor_table"] = df_tmp.copy()
            else:
                st.warning("Uploaded CSV missing required columns; using generated defaults.")
                st.session_state["zillow_editor_table"] = make_prefilled_30()
        except Exception:
            st.warning("Failed to load uploaded CSV; using generated defaults.")
            st.session_state["zillow_editor_table"] = make_prefilled_30()
    elif os.path.exists(SAMPLE_ZILLOW_PATH):
        try:
            s = pd.read_csv(SAMPLE_ZILLOW_PATH)
            if "Date" in s.columns and "ZHVI_AllHomes" in s.columns:
                s["Date"] = pd.to_datetime(s["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
                s["ZHVI_AllHomes"] = pd.to_numeric(s["ZHVI_AllHomes"], errors="coerce")
                st.session_state["zillow_editor_table"] = s.sort_values("Date").tail(30).reset_index(drop=True).copy()
            else:
                st.session_state["zillow_editor_table"] = make_prefilled_30()
        except Exception:
            st.session_state["zillow_editor_table"] = make_prefilled_30()
    else:
        st.session_state["zillow_editor_table"] = make_prefilled_30()

edited = st.data_editor(st.session_state["zillow_editor_table"], num_rows="dynamic", key="zillow_editor_widget", use_container_width=True)
if isinstance(edited, pd.DataFrame):
    try:
        edited["Date"] = pd.to_datetime(edited["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    st.session_state["zillow_editor_table"] = edited.copy()

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------
# 4) Forecast slider (MAIN UI, BEFORE Run)  <-- Option C selected
# -------------------------
st.markdown('<div class="app-card">', unsafe_allow_html=True)
st.header("Forecast Options")
forecast_years = st.slider("Forecast years (Prophet) — choose horizon used for Prophet forecast", 1, 5, 3, key="forecast_slider_main")
st.markdown("This slider controls the Prophet forecast horizon (in years). It is intentionally placed here (before Run).", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------
# 5) Run button (after all inputs)
# -------------------------
st.markdown('<div class="app-card" style="display:flex;justify-content:space-between;align-items:center">', unsafe_allow_html=True)
st.write("")  # spacing
run = st.button("Run: Land type → House price → Zillow forecast", key="run_pipeline")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Pipeline
# -------------------------
if run:
    # Validate image exists
    if uploaded_image is None:
        st.error("Please upload an image before running.")
        st.stop()

    progress = st.progress(0)
    progress.progress(5)

    # -------------------------
    # STEP A: Land classification (optional model)
    # -------------------------
    st.info("Step A — Land type classification")
    space_model = None
    land_type = "Unknown"
    land_conf = 0.0
    try:
        if os.path.exists(space_model_path):
            # build same head and load weights
            space_model = models.resnet18(pretrained=False)
            num_ftrs = space_model.fc.in_features
            space_model.fc = nn.Sequential(
                nn.Linear(num_ftrs,128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128,2),
                nn.LogSoftmax(dim=1)
            )
            state = torch.load(space_model_path, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            space_model.load_state_dict(state)
            space_model.eval()
    except Exception as e:
        st.warning(f"SpaceNet model not loaded: {e}")
        space_model = None

    # Predict or stub
    try:
        pil_img = Image.open(uploaded_image).convert("RGB")
        if space_model is not None:
            transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
            img_t = transform(pil_img).unsqueeze(0)
            with torch.no_grad():
                out = space_model(img_t)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                conf = float(np.max(probs))*100
            label_map = {0:"Residential", 1:"Commercial"}
            land_type = label_map.get(pred_idx, "Unknown")
            land_conf = conf
            st.success(f"Land Type: {land_type} ({conf:.2f}% confidence)")
        else:
            st.info("SpaceNet not provided — land type set to Unknown.")
    except Exception as e:
        st.warning(f"Land classification failed (continuing): {e}")

    st.session_state["land_type"] = land_type
    st.session_state["land_conf"] = land_conf
    progress.progress(20)

    # -------------------------
    # STEP B: House price prediction
    # -------------------------
    st.info("Step B — House price prediction")
    try:
        preprocessor = smart_load_pickle(hp_prep_path)
        house_model = smart_load_pickle(hp_model_path)
    except Exception as e:
        st.error(f"Could not load house preprocessor/model: {e}")
        st.stop()

    if hasattr(preprocessor, "feature_names_in_"):
        trained_cols = list(preprocessor.feature_names_in_)
    else:
        st.error("Preprocessor missing feature_names_in_. Provide a sklearn pipeline.")
        st.stop()

    base = pd.Series(index=trained_cols, dtype=object)
    inputs = {
        "OverallQual": OverallQual, "GrLivArea": GrLivArea, "GarageCars": GarageCars,
        "GarageArea": GarageArea, "TotalBsmtSF": TotalBsmtSF, "FullBath": FullBath,
        "YearBuilt": YearBuilt, "YearRemodAdd": YearRemodAdd, "LotArea": LotArea,
        "Neighborhood": Neighborhood
    }
    for c in trained_cols:
        if c in inputs:
            base[c] = inputs[c]
    for c in trained_cols:
        if pd.isna(base.get(c)):
            cname = c.lower()
            if any(tok in cname for tok in ["area","sf","sqft","lot","grliv","garage","bsmt"]):
                base[c] = 500
            elif "year" in cname or cname.startswith("yr"):
                base[c] = 2000
            elif "qual" in cname:
                base[c] = 6
            elif "bath" in cname:
                base[c] = 2
            else:
                base[c] = 0

    input_row = pd.DataFrame([base]).astype(object)
    for c in input_row.columns:
        try:
            input_row[c] = pd.to_numeric(input_row[c], errors="coerce")
        except Exception:
            pass

    try:
        Xp = preprocessor.transform(input_row)
        preds_log = house_model.predict(Xp)
        preds = np.expm1(preds_log) if np.any(preds_log < 100) else preds_log
        predicted_price = float(preds[0])
        st.success(f"Predicted House Price: {fmt_currency(predicted_price)}")
        st.session_state["predicted_price"] = predicted_price
    except Exception as e:
        st.error(f"House price prediction failed: {e}")
        st.stop()

    progress.progress(45)

    # -------------------------
    # STEP C: Zillow features + LightGBM
    # -------------------------
    st.info("Step C — Zillow: prepare features for LightGBM & Prophet")

    # load features.json; fallback to sensible defaults if missing
    try:
        if os.path.exists(z_feat_path):
            z_features = json.load(open(z_feat_path, "r"))
        else:
            raise FileNotFoundError
    except Exception:
        z_features = [
            "year","month","month_sin","month_cos","t","region_id",
            "lag_1","lag_2","lag_3","lag_4","lag_5","lag_6","lag_9","lag_12","lag_18","rmean_3","rmean_6"
        ]
        st.warning("features.json not found — using fallback feature set.")

    # Load lgb if exists
    lgb_loaded = False
    z_lgb = None
    try:
        if os.path.exists(z_lgb_path):
            z_lgb = joblib.load(z_lgb_path)
            lgb_loaded = True
        else:
            st.warning("Zillow LightGBM model file not found — skipping LightGBM prediction.")
    except Exception as e:
        st.warning(f"Failed to load LightGBM model: {e}")
        lgb_loaded = False

    # Read editor table
    z_tab = st.session_state.get("zillow_editor_table", pd.DataFrame()).copy()
    try:
        z_tab["Date"] = pd.to_datetime(z_tab["Date"], errors="coerce")
    except Exception:
        pass
    z_tab["ZHVI_AllHomes"] = pd.to_numeric(z_tab["ZHVI_AllHomes"], errors="coerce")
    z_tab = z_tab.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if z_tab.empty:
        st.error("Zillow editor has no valid rows.")
        st.stop()

    # Fill values and create contiguous monthly timeline
    z_tab["ZHVI_AllHomes"] = z_tab["ZHVI_AllHomes"].ffill().bfill()
    start = z_tab["Date"].min()
    end = z_tab["Date"].max()
    full_dates = pd.date_range(start=start, end=end, freq="M")
    if len(full_dates) < 6:
        # expand using prefill ending at 'end'
        pre = make_prefilled_30()
        dfz = pre.copy()
        dfz["Date"] = pd.to_datetime(dfz["Date"])
    else:
        dfz = pd.DataFrame({"Date": full_dates})
        dfz = dfz.merge(z_tab[["Date","ZHVI_AllHomes"]], on="Date", how="left")
        dfz["ZHVI_AllHomes"] = dfz["ZHVI_AllHomes"].ffill().bfill()
        if dfz["ZHVI_AllHomes"].isna().any():
            dfz["ZHVI_AllHomes"] = dfz["ZHVI_AllHomes"].fillna(z_tab["ZHVI_AllHomes"].median())

    # engineering
    dfz["year"] = dfz["Date"].dt.year
    dfz["month"] = dfz["Date"].dt.month
    dfz["month_sin"] = np.sin(2*np.pi*dfz["month"]/12)
    dfz["month_cos"] = np.cos(2*np.pi*dfz["month"]/12)
    dfz["t"] = (dfz["Date"] - dfz["Date"].min()).dt.days
    dfz["region_id"] = 0

    # create lags requested in z_features (if present)
    requested_lags = sorted([int(f.split("_")[1]) for f in z_features if isinstance(f,str) and f.startswith("lag_")]) if z_features else [1,2,3,4,5,6,9,12]
    requested_lags = list(dict.fromkeys(requested_lags))  # dedupe
    for lag in requested_lags:
        dfz[f"lag_{lag}"] = dfz["ZHVI_AllHomes"].shift(lag)

    dfz["rmean_3"] = dfz["ZHVI_AllHomes"].shift(1).rolling(3).mean()
    dfz["rmean_6"] = dfz["ZHVI_AllHomes"].shift(1).rolling(6).mean()

    # Keep raw for prophet
    dfz_raw = dfz.copy()

    # Align features for LightGBM
    model_feature_names = None
    if lgb_loaded and z_lgb is not None:
        try:
            model_feature_names = getattr(z_lgb, "feature_name_", None)
            if not model_feature_names and hasattr(z_lgb, "booster_"):
                model_feature_names = z_lgb.booster_.feature_name()
        except Exception:
            model_feature_names = None

    use_features = model_feature_names if model_feature_names else z_features

    # Identify missing features in dfz and fill with reasonable defaults
    missing = [f for f in use_features if f not in dfz.columns]
    if missing:
        for m in missing:
            if isinstance(m, str) and m.startswith("lag_"):
                try:
                    lag_n = int(m.split("_")[1])
                    dfz[m] = dfz["ZHVI_AllHomes"].shift(lag_n)
                except Exception:
                    dfz[m] = np.nan
            elif isinstance(m, str) and m.startswith("rmean_"):
                try:
                    r = int(m.split("_")[1])
                    dfz[m] = dfz["ZHVI_AllHomes"].shift(1).rolling(r).mean()
                except Exception:
                    dfz[m] = np.nan
            else:
                dfz[m] = 0
        dfz[missing] = dfz[missing].fillna(dfz["ZHVI_AllHomes"].median()).astype(float)

    # Ensure all columns exist and reorder
    dfz_model = dfz.copy()
    dfz_model = dfz_model.reindex(columns=use_features, fill_value=np.nan)

    # Drop rows with NaNs in model features, but attempt relaxation if necessary
    df_feat = dfz_model.dropna().reset_index(drop=True)
    if df_feat.empty:
        relax_features = use_features.copy()
        lags_in_use = sorted([int(f.split("_")[1]) for f in relax_features if isinstance(f,str) and f.startswith("lag_")])
        while df_feat.empty and lags_in_use:
            max_l = max(lags_in_use)
            relax_features = [f for f in relax_features if not (isinstance(f,str) and f.startswith("lag_") and int(f.split("_")[1])==max_l)]
            dfz_model2 = dfz.copy().reindex(columns=relax_features, fill_value=np.nan)
            df_feat = dfz_model2.dropna().reset_index(drop=True)
            lags_in_use = [l for l in lags_in_use if l != max_l]
        if df_feat.empty:
            dfz_model = dfz_model.fillna(method="ffill").fillna(method="bfill").fillna(dfz["ZHVI_AllHomes"].median())
            df_feat = dfz_model.dropna().reset_index(drop=True)

    if df_feat.empty:
        st.warning("LightGBM rows still empty; skipping LightGBM prediction.")
        z_next = None
        st.session_state["zillow_next"] = None
    else:
        x_row = df_feat.tail(1).copy()

        if lgb_loaded and z_lgb is not None and model_feature_names:
            for f in model_feature_names:
                if f not in x_row.columns:
                    x_row[f] = dfz["ZHVI_AllHomes"].median()
            x_row = x_row[model_feature_names]

        z_next = None
        if lgb_loaded and z_lgb is not None:
            try:
                z_next = float(z_lgb.predict(x_row)[0])
                st.success(f"LightGBM next-month Zillow prediction: {fmt_currency(z_next)}")
                st.session_state["zillow_next"] = z_next
            except Exception as e:
                st.warning(f"LightGBM prediction initial attempt failed: {e}")
                try:
                    model_cols = None
                    if hasattr(z_lgb, "feature_name_") and z_lgb.feature_name_():
                        model_cols = z_lgb.feature_name_()
                    elif hasattr(z_lgb, "booster_"):
                        model_cols = z_lgb.booster_.feature_name()
                    if model_cols:
                        for c in model_cols:
                            if c not in x_row.columns:
                                x_row[c] = dfz["ZHVI_AllHomes"].median()
                        x_row = x_row[model_cols]
                        z_next = float(z_lgb.predict(x_row)[0])
                        st.success(f"LightGBM next-month Zillow prediction (after alignment): {fmt_currency(z_next)}")
                        st.session_state["zillow_next"] = z_next
                    else:
                        st.warning("Could not determine model feature names for alignment.")
                        st.session_state["zillow_next"] = None
                except Exception as e2:
                    st.error(f"LightGBM prediction failed after attempts: {e2}")
                    st.session_state["zillow_next"] = None

    progress.progress(70)

    # -------------------------
    # PROPHET FORECASTING
    # -------------------------
    st.info("Step D — Prophet forecasting")
    prophet_df = dfz_raw[["Date","ZHVI_AllHomes"]].rename(columns={"Date":"ds","ZHVI_AllHomes":"y"})
    st.session_state["zillow_series_used_for_prophet"] = prophet_df.copy()
    if Prophet is None:
        st.warning("Prophet not installed — skipping multi-year forecast.")
        st.session_state["zillow_future"] = pd.DataFrame()
        st.session_state["zillow_last"] = 0.0
    else:
        try:
            m = Prophet(yearly_seasonality=True)
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=forecast_years*12, freq="M")
            forecast = m.predict(future)
            future_only = forecast[forecast["ds"] > prophet_df["ds"].max()].reset_index(drop=True)
            st.session_state["zillow_future"] = future_only.copy()
            st.session_state["zillow_last"] = float(future_only["yhat"].iloc[-1]) if len(future_only)>0 else 0.0
            st.success("Prophet forecasting completed.")
        except Exception as e:
            st.error(f"Prophet forecasting failed: {e}")
            st.session_state["zillow_future"] = pd.DataFrame()
            st.session_state["zillow_last"] = 0.0

    progress.progress(85)

    # -------------------------
    # Profitability calculation
    # -------------------------
    predicted_price = st.session_state.get("predicted_price", 0.0)
    invest = float(investment_price)
    profit_abs = predicted_price - invest
    profit_pct = (profit_abs / invest * 100) if invest != 0 else 0.0
    profitable = profit_abs > 0

    # Zillow index growth percent (from last historical to forecast end)
    z_last = st.session_state.get("zillow_last", None)
    z_hist_last = dfz_raw["ZHVI_AllHomes"].iloc[-1] if not dfz_raw.empty else None
    z_growth_pct = None
    if z_last is not None and z_hist_last is not None and z_hist_last > 0:
        z_growth_pct = (z_last - z_hist_last) / z_hist_last * 100

    # -------------------------
    # Show results
    # -------------------------
    st.markdown("## Results")
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Land Type", f"{st.session_state.get('land_type','Unknown')} ({st.session_state.get('land_conf',0.0):.1f}%)")
    with colB:
        st.metric("Predicted House Price", fmt_currency(predicted_price))
    with colC:
        st.metric("LightGBM Next-month Zillow", fmt_currency(st.session_state.get("zillow_next", 0.0) if st.session_state.get("zillow_next", None) else 0.0))

    # Profitability card
    st.markdown(f"""
    <div class="app-card" style="margin-top:12px">
      <h3>Investment analysis</h3>
      <div style="display:flex;gap:18px;">
        <div class="metric-card"><b>Investment</b><div>{fmt_currency(invest)}</div></div>
        <div class="metric-card"><b>Predicted Sale</b><div>{fmt_currency(predicted_price)}</div></div>
        <div class="metric-card"><b>Profit</b><div>{fmt_currency(profit_abs)} ({profit_pct:.2f}%)</div></div>
      </div>
      <div style="margin-top:8px;color:{'#A7F3D0' if profitable else '#FFD1D1'}"><b>{'Profitable' if profitable else 'Not profitable'}</b></div>
    </div>
    """, unsafe_allow_html=True)

    if z_growth_pct is not None:
        st.info(f"Zillow index projected growth over {forecast_years} year(s): {z_growth_pct:.2f}% (index end: {fmt_currency(z_last)})")

    progress.progress(92)

    # -------------------------
    # Prophet graph ONLY on UI
    # -------------------------
    if Prophet is not None and isinstance(st.session_state.get("zillow_future"), pd.DataFrame) and not st.session_state.get("zillow_future").empty:
        fut = st.session_state["zillow_future"]

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(prophet_df["ds"], prophet_df["y"], label="History", marker="o")

        if st.session_state.get("zillow_next", None) is not None:
            ax.scatter([prophet_df["ds"].max() + pd.DateOffset(months=1)], [st.session_state.get("zillow_next",0.0)], color="red", marker="X", s=80, label="LightGBM Next-month")

        ax.plot(fut["ds"], fut["yhat"], color="orange", linestyle="--", label=f"{forecast_years}-yr Forecast")
        ax.fill_between(fut["ds"], fut["yhat_lower"], fut["yhat_upper"], color="orange", alpha=0.2)
        ax.set_xlabel("Date")
        ax.set_ylabel("ZHVI_AllHomes")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        st.markdown("### Forecast sample (first 12 months)")
        display_df = fut[["ds", "yhat", "yhat_lower", "yhat_upper"]].head(12).copy()
        display_df["ds"] = display_df["ds"].dt.strftime("%Y-%m-%d")
        display_df = display_df.rename(columns={
            "ds": "Date",
            "yhat": "Forecast",
            "yhat_lower": "Lower",
            "yhat_upper": "Upper"
        })
        st.dataframe(display_df)
    else:
        if Prophet is None:
            st.info("Prophet is not installed. Forecast unavailable.")
        else:
            st.info("Forecast not available.")

    # -------------------------
    # Prepare masked image for PDF
    # -------------------------
    try:
        mask_pil = make_mask_from_pil(pil_img)
    except Exception:
        mask_pil = None

    # ---------------------------------------------------------
    # 📄 PERFECT PDF GENERATION — NO OVERLAP, CLEAN LAYOUT
    # ---------------------------------------------------------

    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch

    st.markdown("## 📎 Download PDF Report")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    W, H = letter

    

    # ---------- HEADER ----------
    c.setFillColorRGB(0.05, 0.08, 0.18)
    c.rect(0, H-70, W, 70, fill=1)
    c.setFillColorRGB(1, 0.83, 0.4)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(30, H-40, "RealtyAI — Property Report")

    # ---------- START COORD ----------
    y = H - 90


    # =====================================================
    # SECTION 1 — SUMMARY BOX (SAFE SPACING)
    # =====================================================
    c.setFillColorRGB(0.95, 0.95, 0.95)
    c.rect(30, y-100, W-60, 100, fill=1, stroke=0)

    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y-20, "Property Summary")

    c.setFont("Helvetica", 11)
    lines = [
        f"Land Type: {land_type} ({land_conf:.1f}%)",
        f"Predicted House Price: {fmt_currency(predicted_price)}",
        f"LightGBM Next-Month Zillow: {fmt_currency(st.session_state.get('zillow_next',0.0))}",
        f"Zillow {forecast_years}-Year End Value: {fmt_currency(z_last)}",
        f"Investment: {fmt_currency(invest)} → Profit: {fmt_currency(profit_abs)} ({profit_pct:.2f}%)",
    ]

    ty = y - 40
    for line in lines:
        c.drawString(40, ty, line)
        ty -= 15

    y = ty - 30   # SAFE SPACING BELOW BOX


    # =====================================================
    # SECTION 2 — ORIGINAL + MASK IMAGES
    # =====================================================
    try:
        # ORIGINAL
        orig_b = io.BytesIO()
        pil_img.thumbnail((300, 300))
        pil_img.save(orig_b, format="PNG")
        orig_b.seek(0)
        c.drawImage(ImageReader(orig_b), 40, y-280, width=260, height=260)

        # MASK
        mask_b = io.BytesIO()
        mask_pil.thumbnail((300, 300))
        mask_pil.save(mask_b, format="PNG")
        mask_b.seek(0)
        c.drawImage(ImageReader(mask_b), 320, y-280, width=260, height=260)

    except:
        pass

    y -= 320  # MOVE DOWN SAFELY


    # =====================================================
    # SECTION 3 — FORECAST GRAPH (ITS OWN PAGE)
    # =====================================================

    c.showPage()   # NEW CLEAN PAGE
    y = H - 60

    c.setFillColorRGB(0.05, 0.08, 0.18)
    c.rect(0, H-60, W, 60, fill=1)
    c.setFillColorRGB(1, 0.83, 0.4)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(30, H-35, "Zillow Forecast Graph")

    graph_b = io.BytesIO()
    fig.savefig(graph_b, format="PNG", dpi=150, bbox_inches="tight")
    graph_b.seek(0)

    c.drawImage(ImageReader(graph_b), 40, y-350, width=520, height=330)

    y -= 380


    # =====================================================
    # SECTION 4 — FORECAST TABLE (NEW PAGE)
    # =====================================================

    c.showPage()
    y = H - 60

    c.setFillColorRGB(0.05, 0.08, 0.18)
    c.rect(0, H-60, W, 60, fill=1)
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(30, H-35, "Zillow Monthly Forecast Table")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y-40, "Date")
    c.drawString(200, y-40, "Forecast (yhat)")
    y -= 60

    c.setFont("Helvetica", 11)
    zf = st.session_state["zillow_future"].head(20)

    for i, row in zf.iterrows():
        if y < 80:
            c.showPage()
            y = H - 80

        c.drawString(40, y, str(row["ds"].date()))
        c.drawString(200, y, fmt_currency(row["yhat"]))
        y -= 20


    # =====================================================
    # SAVE + CLOSE
    # =====================================================
    c.save()
    buf.seek(0)

    # SAVE TO /reports
    if not os.path.exists("reports"):
        os.makedirs("reports")

    file_name = f"RealtyAI_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    save_path = os.path.join("reports", file_name)

    with open(save_path, "wb") as f:
        f.write(buf.getbuffer())

    st.success(f"PDF saved to: {save_path}")

    # DOWNLOAD
    st.download_button("📥 Download PDF", buf,
                    file_name="RealtyAI_Report.pdf",
                    mime="application/pdf")


    progress.progress(100)
    st.balloons()


# Footer
st.markdown("---")
st.caption("Developed by Sahithi Mandha | RealtyAI © 2025")
