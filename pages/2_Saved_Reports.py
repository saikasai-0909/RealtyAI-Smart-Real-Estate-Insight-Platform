import streamlit as st
import os
import base64

def render_pdf(file_path):
    """Display PDF inline in Streamlit."""
    with open(file_path, "rb") as f:
        pdf_data = f.read()

    b64 = base64.b64encode(pdf_data).decode()
    pdf_display = f'<embed src="data:application/pdf;base64,{b64}" width="100%" height="700px" />'
    st.markdown(pdf_display, unsafe_allow_html=True)

# ---------------------------
# Page Title
# ---------------------------
st.title("📄 Saved Reports")

# Ensure folder exists
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

files = sorted(
    [f for f in os.listdir(REPORT_DIR) if f.endswith(".pdf")],
    reverse=True
)

if not files:
    st.info("No saved reports found yet. Generate one from Home page.")
else:
    st.write(f"Found **{len(files)}** saved reports.")

    for file_name in files:
        file_path = os.path.join(REPORT_DIR, file_name)

        with st.container(border=True):
            col1, col2 = st.columns([3,1])

            with col1:
                st.write(f"### 📘 {file_name}")
                st.caption(f"Saved at: `{file_path}`")

            with col2:
                if st.button("View", key=f"view_{file_name}"):
                    st.session_state["view_file"] = file_path
                st.download_button(
                    "Download",
                    data=open(file_path, "rb").read(),
                    file_name=file_name,
                    mime="application/pdf",
                    key=f"download_{file_name}"
                )
                if st.button("Delete", key=f"delete_{file_name}"):
                    os.remove(file_path)
                    st.warning(f"Deleted {file_name}")
                    st.rerun()

# Show PDF preview at bottom
if "view_file" in st.session_state:
    st.markdown("---")
    st.subheader(f"📑 Preview: {os.path.basename(st.session_state['view_file'])}")
    render_pdf(st.session_state["view_file"])
