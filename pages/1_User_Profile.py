import streamlit as st
from PIL import Image
import io

# -------------------------
# PAGE TITLE
# -------------------------
st.markdown("""
<h1 style='color:#FFD580;'>👤 User Profile</h1>
<p style='color:#93C5FD;'>Manage your personal details, contact info & profile photo.</p>
<hr style="border: 1px solid rgba(255,255,255,0.08);">
""", unsafe_allow_html=True)



# -------------------------
# Load existing profile info
# -------------------------
saved_name = st.session_state.get("user_name", "")
saved_email = st.session_state.get("user_email", "")
saved_img = st.session_state.get("profile_img", None)

# -------------------------
# Two column layout
# -------------------------
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Personal Details")
    name = st.text_input("Full Name", value=saved_name)
    email = st.text_input("Email", value=saved_email)

with col2:
    st.subheader("Profile Photo")
    uploaded_img = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

    if uploaded_img:
        preview = Image.open(uploaded_img)
        st.image(preview, width=150)
    elif saved_img:
        st.image(saved_img, width=150)

# -------------------------
# Save Button
# -------------------------
if st.button("💾 Save Profile"):
    st.session_state["user_name"] = name
    st.session_state["user_email"] = email

    if uploaded_img:
        st.session_state["profile_img"] = Image.open(uploaded_img)

    st.success("Profile saved successfully!")

# -------------------------
# Summary
# -------------------------
st.markdown("""
<br><h3 style='color:#FFD580;'>Profile Summary</h3>
<div style="padding:14px;border-radius:10px;background:rgba(255,255,255,0.03);
            border:1px solid rgba(255,255,255,0.06);">
""", unsafe_allow_html=True)

c1, c2 = st.columns([1,1])

with c1:
    st.write("**Name:**", st.session_state.get("user_name", "Not set"))
    st.write("**Email:**", st.session_state.get("user_email", "Not set"))

with c2:
    if st.session_state.get("profile_img"):
        st.image(st.session_state["profile_img"], width=120)
    else:
        st.write("_No image uploaded_")

st.markdown("</div>", unsafe_allow_html=True)
