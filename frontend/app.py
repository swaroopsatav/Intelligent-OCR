import streamlit as st

from services.api_client import health_check

st.title("Intelligent OCR System")

if st.button("Check Backend"):

    result = health_check()

    st.success(
        result["service"]
    )