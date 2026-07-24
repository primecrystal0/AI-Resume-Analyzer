import streamlit as st
import pdfplumber
import docx
import io
import re
from skills_db import SKILLS_DB

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


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#./@\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_skills(clean_resume_text):
    found = {}
    for category, skills in SKILLS_DB.items():
        matched = []
        for skill in skills:
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, clean_resume_text):
                matched.append(skill)
        if matched:
            found[category] = sorted(matched)
    return found


if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")

    raw_text = extract_text(uploaded_file)
    processed_text = clean_text(raw_text)
    found_skills = extract_skills(processed_text)

    st.subheader("🛠️ Skills Detected in Your Resume")
    if found_skills:
        for category, skills in found_skills.items():
            st.markdown(f"**{category}:** {', '.join(skills)}")
    else:
        st.warning("No known skills detected. Try a resume with a clear Skills section.")

else:
    st.info("👆 Please upload a resume file.")