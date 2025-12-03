import streamlit as st

st.set_page_config(page_title="Admin Dashboard", page_icon="⚙️")

st.title("⚙️ Admin Dashboard")

st.markdown("### Manage Model Files for RealtyAI")

# Store in session_state so index.py can use them
st.session_state.setdefault("space_model_path", "best_model.pth")
st.session_state.setdefault("hp_prep_path", "house_prices_preprocessor.pkl")
st.session_state.setdefault("hp_model_path", "gradient_boosting_house_price.pkl")
st.session_state.setdefault("z_lgb_path", "lightgbm_zillow_model.pkl")
st.session_state.setdefault("z_feat_path", "features.json")

def update(key, value):
    st.session_state[key] = value

st.text_input("SpaceNet Model (.pth)", 
              value=st.session_state["space_model_path"], 
              on_change=update, 
              args=("space_model_path", st.session_state["space_model_path"]))

st.text_input("House Preprocessor (.pkl)", 
              value=st.session_state["hp_prep_path"], 
              on_change=update,
              args=("hp_prep_path", st.session_state["hp_prep_path"]))

st.text_input("House Model (.pkl)", 
              value=st.session_state["hp_model_path"], 
              on_change=update,
              args=("hp_model_path", st.session_state["hp_model_path"]))

st.text_input("Zillow LightGBM Model (.pkl)", 
              value=st.session_state["z_lgb_path"], 
              on_change=update,
              args=("z_lgb_path", st.session_state["z_lgb_path"]))

st.text_input("Zillow Features (.json)", 
              value=st.session_state["z_feat_path"], 
              on_change=update,
              args=("z_feat_path", st.session_state["z_feat_path"]))

st.success("Model paths updated successfully! These will be used in the Home Prediction page.")
