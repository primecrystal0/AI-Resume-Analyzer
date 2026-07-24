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

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.hero {
    padding: 2rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    color: white;
    margin-bottom: 1.5rem;
}
.hero h1 { margin: 0; font-size: 2.2rem; }
.hero p { margin: 0.4rem 0 0 0; opacity: 0.9; font-size: 1.05rem; }
.skill-badge {
    display: inline-block;
    background: #EEF2FF;
    color: #4338CA;
    border-radius: 999px;
    padding: 5px 14px;
    margin: 3px;
    font-size: 0.85rem;
    border: 1px solid #C7D2FE;
}
</style>
""", unsafe_allow_html=True)


SECTION_KEYWORDS = {
    "Experience": ["experience", "work history", "employment"],
    "Education": ["education", "academic", "qualification"],
    "Projects": ["project"],
    "Skills": ["skills", "technical skills"],
    "Contact Info": ["email", "phone", "linkedin", "@"],
    "Summary/Objective": ["summary", "objective", "profile"],
}

ACTION_VERBS = [
    "developed", "built", "designed", "implemented", "led", "managed",
    "created", "improved", "optimized", "automated", "analyzed", "launched"
]

st.markdown("""
<div class="hero">
    <h1>📄 AI Resume Analyzer</h1>
    <p>NLP-powered skill extraction, ML-based job matching, and resume improvement suggestions.</p>
</div>
""", unsafe_allow_html=True)

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

def compute_resume_score(raw_text, clean_text, found_skills, top_role_score):
    sections_present = sum(
        1 for keywords in SECTION_KEYWORDS.values() if any(kw in clean_text for kw in keywords)
    )
    section_score = (sections_present / len(SECTION_KEYWORDS)) * 25

    total_skills = sum(len(v) for v in found_skills.values())
    skill_score = min(total_skills / 15, 1.0) * 25

    verbs_found = sum(1 for v in ACTION_VERBS if v in clean_text)
    verb_score = min(verbs_found / 6, 1.0) * 12.5
    has_numbers = bool(re.search(r"\d+%|\$\d+|\b\d+\b", raw_text))
    number_score = 12.5 if has_numbers else 0
    impact_score = verb_score + number_score

    fit_score = (top_role_score / 100) * 25

    total = round(section_score + skill_score + impact_score + fit_score, 1)
    return total

def suggest_improvements(raw_text, clean_text, found_skills):
    suggestions = []

    for section, keywords in SECTION_KEYWORDS.items():
        if not any(kw in clean_text for kw in keywords):
            suggestions.append(f"Consider adding a clear **{section}** section — it wasn't detected.")

    word_count = len(raw_text.split())
    if word_count < 150:
        suggestions.append("Your resume looks quite short. Add more detail on projects and impact.")
    elif word_count > 1200:
        suggestions.append("Your resume is quite long — consider trimming to 1-2 pages.")

    if not re.search(r"\d+%|\$\d+|\b\d+\b", raw_text):
        suggestions.append("Add quantifiable achievements (e.g. 'improved accuracy by 15%').")

    verbs_found = [v for v in ACTION_VERBS if v in clean_text]
    if len(verbs_found) < 3:
        suggestions.append("Use more strong action verbs (developed, optimized, led, automated).")

    total_skills = sum(len(v) for v in found_skills.values())
    if total_skills < 5:
        suggestions.append("Very few technical skills detected — make sure your Skills section is specific.")

    if not suggestions:
        suggestions.append("Great job! Your resume covers sections, skills, and impact well.")

    return suggestions


if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")

    raw_text = extract_text(uploaded_file)
    processed_text = clean_text(raw_text)
    found_skills = extract_skills(processed_text)
    role_matches = match_job_roles(processed_text)
    suggestions = suggest_improvements(raw_text, processed_text, found_skills)

    total_skills_count = sum(len(v) for v in found_skills.values())
    top_role = role_matches[0]
    resume_score = compute_resume_score(raw_text, processed_text, found_skills, top_role["score"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Resume Score", f"{resume_score}/100")
    col2.metric("Skills Detected", total_skills_count)
    col3.metric("Top Job Match", top_role["role"], f"{top_role['score']}%")

    st.divider()

    st.subheader("🛠️ Skills Detected in Your Resume")
    if found_skills:
        for category, skills in found_skills.items():
            badges = "".join(f'<span class="skill-badge">{s}</span>' for s in skills)
            st.markdown(f"**{category}**", unsafe_allow_html=True)
            st.markdown(badges, unsafe_allow_html=True)
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

    st.subheader("💡 Suggestions to Improve Your Resume")
    for s in suggestions:
        st.markdown(f"- {s}")

else:
    st.info("👆 Please upload a resume file.")