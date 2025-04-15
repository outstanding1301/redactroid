import streamlit as st
import requests
import base64
import tempfile
import pandas as pd

API_DETECT_URL = "http://localhost:8000/detect"
API_REDACT_URL = "http://localhost:8000/redact"

# ---------------------------
# Session state 초기화
# ---------------------------
if "pii_data" not in st.session_state:
    st.session_state.pii_data = {}
if "redacted_pdf" not in st.session_state:
    st.session_state.redacted_pdf = None

# ---------------------------
# PDF 표시 함수 (iframe)
# ---------------------------
def displayPDF(file, ui_width=700):
    bytes_data = file.getvalue() if hasattr(file, "getvalue") else file
    base64_pdf = base64.b64encode(bytes_data).decode("utf-8")
    pdf_display = f'''
        <iframe src="data:application/pdf;base64,{base64_pdf}"
                width="{ui_width}" height="{int(ui_width * 4 / 3)}"
                type="application/pdf"></iframe>
    '''
    st.markdown(pdf_display, unsafe_allow_html=True)

# ---------------------------
# 사이드바 UI
# ---------------------------
with st.sidebar:
    st.title("🔐 PII Detection")

    uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])

    if uploaded_file is not None:
        if "last_uploaded_name" not in st.session_state or uploaded_file.name != st.session_state.last_uploaded_name:
            st.session_state.pii_data = {}
            st.session_state.redacted_pdf = None
            st.session_state.last_uploaded_name = uploaded_file.name

    if uploaded_file and st.button("🔍 Detect"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            files = {"file": (uploaded_file.name, f, "application/pdf")}
            with st.spinner("Sending to detection server..."):
                try:
                    response = requests.post(API_DETECT_URL, files=files)
                    response.raise_for_status()

                    st.session_state.pii_data = response.json()

                    st.success("✅ Detection completed!")
                    st.session_state.redacted_pdf = None  # 기존 Redact 결과는 제거

                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Detection failed: {e}")

    # 📊 결과 출력
    if st.session_state.pii_data:
        st.subheader("📊 Detected PII")
        pii_table = {
            "Field": [],
            "Values": []
        }
        for key, values in st.session_state.pii_data.items():
            pii_table["Field"].append(key)
            pii_table["Values"].append(", ".join(f"`{v}`" for v in values) if values else "")
        st.table(pd.DataFrame(pii_table).style.hide(axis="index"))

    # 🛡️ Redact
    if uploaded_file and st.session_state.pii_data:
        if st.button("🛡️ Redact"):
            pii_payload = {
                "name": ",".join(st.session_state.pii_data.get("name", [])),
                "phone": ",".join(st.session_state.pii_data.get("phone", [])),
                "rrn": ",".join(st.session_state.pii_data.get("rrn", [])),
                "email": ",".join(st.session_state.pii_data.get("email", [])),
                "address": ",".join(st.session_state.pii_data.get("address", [])),
            }

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            with open(tmp_path, "rb") as f:
                files = {"file": (uploaded_file.name, f, "application/pdf")}
                with st.spinner("Sending to redact server..."):
                    try:
                        response = requests.post(API_REDACT_URL, files=files, data=pii_payload)
                        response.raise_for_status()
                        st.session_state.redacted_pdf = response.content
                        st.success("✅ Redaction completed!")
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Redaction failed: {e}")

# ---------------------------
# 본문: PDF 나란히 표시
# ---------------------------
if uploaded_file:
    tab1, tab2 = st.tabs(["📘 Original PDF", "🧼 Redacted PDF"])
    with tab1:
        displayPDF(uploaded_file, ui_width=800)
    with tab2:
        if st.session_state.redacted_pdf:
            displayPDF(st.session_state.redacted_pdf, ui_width=800)
        else:
            st.info("Run redaction to preview result.")
