import streamlit as st

st.title("📄 AI Resume Analyzer")
st.write("Upload your resume to get started.")

uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")
else:
    st.info("👆 Please upload a resume file.")