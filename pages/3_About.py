import streamlit as st

# ============================
# Animated Sidebar CSS
# ============================
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1224 0%, #0b162e 100%);
    padding: 18px 10px;
    animation: slideIn 0.7s ease-out;
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(8px);
}
@keyframes slideIn {
    0% { transform: translateX(-40px); opacity:0; }
    100% { transform: translateX(0); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# ============================
# About Page CSS
# ============================
st.markdown("""
<style>

.about-title {
    font-size: 2.7rem;
    font-weight: 800;
    color: #FFD580;
    text-shadow: 0 0 25px rgba(255,213,128,0.6);
    animation: fadeIn 1.2s ease;
}

@keyframes fadeIn {
    0% { opacity:0; transform: translateY(-10px); }
    100% { opacity:1; transform: translateY(0); }
}

/* Glowing Card */
.glow-card {
    background: rgba(255,255,255,0.04);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    transition: 0.3s ease;
    box-shadow: 0 0 0px rgba(255,213,128,0.0);
}

.glow-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 0 25px rgba(255,213,128,0.35);
    border-color: rgba(255,213,128,0.7);
}

/* Icon animation */
.icon {
    font-size: 2rem;
    margin-bottom: 10px;
    color: #93C5FD;
    transition: 0.3s ease;
}

.glow-card:hover .icon {
    transform: scale(1.25);
    color: #FFD580;
}
</style>
""", unsafe_allow_html=True)


# # ============================
# # Sidebar Navigation (simple)
# # ============================
# with st.sidebar:
#     st.markdown("<h2 style='color:#FFD580;'>About</h2>", unsafe_allow_html=True)
#     st.markdown("This page explains what RealtyAI does.")

# ============================
# Main About Content
# ============================
st.markdown("<h1 class='about-title'>✨ About RealtyAI</h1>", unsafe_allow_html=True)
st.write("### A powerful AI-driven real estate intelligence system built by **Sahithi Mandha**.")

st.write("")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='glow-card'>
        <div class='icon'>🏡</div>
        <h3 style='color:#FFD580;'>House Price AI</h3>
        <p>Predicts highly accurate sale prices using ML & real market patterns.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='glow-card'>
        <div class='icon'>🌍</div>
        <h3 style='color:#FFD580;'>Land Classification</h3>
        <p>Identifies land types from satellite images using deep learning.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='glow-card'>
        <div class='icon'>📈</div>
        <h3 style='color:#FFD580;'>Zillow Forecasting</h3>
        <p>Predicts future market value with LightGBM + Prophet forecasting.</p>
    </div>
    """, unsafe_allow_html=True)


st.write("")
st.write("")

# SECOND ROW
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
    <div class='glow-card'>
        <div class='icon'>📊</div>
        <h3 style='color:#FFD580;'>Investment Analysis</h3>
        <p>Instant profit/loss calculation using smart analytics.</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class='glow-card'>
        <div class='icon'>📄</div>
        <h3 style='color:#FFD580;'>PDF Reports</h3>
        <p>Generates professional real-estate property reports.</p>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("""
    <div class='glow-card'>
        <div class='icon'>🚀</div>
        <h3 style='color:#FFD580;'>Future Ready</h3>
        <p>Modular design ready for expansion into valuation dashboards.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("---")

st.markdown("""
### Built by **Sahithi Mandha**,  
for intelligent and transparent real-estate decision-making.
""")

