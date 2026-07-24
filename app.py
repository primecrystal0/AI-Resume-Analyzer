import streamlit as st
import pdfplumber
import docx
import io
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from skills_db import SKILLS_DB
from job_roles import JOB_ROLES

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


def match_job_roles(clean_resume_text, top_n=5):
    role_names = list(JOB_ROLES.keys())
    role_docs = [" ".join(skills) for skills in JOB_ROLES.values()]

    documents = [clean_resume_text] + role_docs
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    resume_vector = tfidf_matrix[0:1]
    role_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(resume_vector, role_vectors)[0]

    results = []
    for i, role in enumerate(role_names):
        required_skills = set(JOB_ROLES[role])
        matched = {s for s in required_skills if re.search(r"\b" + re.escape(s) + r"\b", clean_resume_text)}
        missing = required_skills - matched

        overlap_score = len(matched) / len(required_skills) if required_skills else 0
        blended_score = (0.5 * similarities[i]) + (0.5 * overlap_score)

        results.append({
            "role": role,
            "score": round(blended_score * 100, 1),
            "matched_skills": sorted(matched),
            "missing_skills": sorted(missing),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")

    raw_text = extract_text(uploaded_file)
    processed_text = clean_text(raw_text)
    found_skills = extract_skills(processed_text)
    role_matches = match_job_roles(processed_text)

    st.subheader("🛠️ Skills Detected in Your Resume")
    if found_skills:
        for category, skills in found_skills.items():
            st.markdown(f"**{category}:** {', '.join(skills)}")
    else:
        st.warning("No known skills detected. Try a resume with a clear Skills section.")

    st.subheader("🎯 Best-Fit Job Roles")
    df = pd.DataFrame(role_matches)[["role", "score"]]
    df.columns = ["Job Role", "Match %"]
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.bar_chart(df.set_index("Job Role"))

    st.subheader("🔍 Detailed Breakdown")
    for match in role_matches:
        with st.expander(f"{match['role']} — {match['score']}% match"):
            st.markdown(f"**✅ Matched Skills:** {', '.join(match['matched_skills']) or 'None'}")
            st.markdown(f"**❌ Missing Skills:** {', '.join(match['missing_skills']) or 'None'}")

else:
    st.info("👆 Please upload a resume file.")