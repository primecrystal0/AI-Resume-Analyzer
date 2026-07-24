import streamlit as st
import pdfplumber
import docx
import io

st.title("📄 AI Resume Analyzer")
st.write("Upload your resume to get started.")

uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])

def extract_text(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        text = []
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n".join(text)

    elif filename.endswith(".docx"):
        document = docx.Document(io.BytesIO(uploaded_file.read()))
        return "\n".join(p.text for p in document.paragraphs if p.text.strip())

    elif filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")
    resume_text = extract_text(uploaded_file)
    st.subheader("Extracted Text Preview")
    st.text_area("This is what we read from your resume:", resume_text, height=300)
else:
    st.info("👆 Please upload a resume file.")