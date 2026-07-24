# 📄 AI Resume Analyzer

An AI-powered resume analysis system built with **Python, NLP, and Machine Learning**, wrapped in a **Streamlit** web app. It extracts skills from an uploaded resume, recommends the best-fit job roles using TF-IDF + cosine similarity, scores the resume, and gives concrete suggestions to improve it — plus a downloadable PDF report.

## Features

- 📥 Upload resumes in **PDF, DOCX, or TXT** format
- 🧠 **NLP-based skill extraction** across 7 categories (programming languages, web dev, data science/ML, databases, cloud/DevOps, tools, soft skills)
- 🎯 **ML-based job-role matching** using TF-IDF vectorization + cosine similarity, blended with keyword overlap
- 📋 **Custom job-description matcher** — paste any job description and get a tailored match score
- 📊 Composite **resume score (0-100)** based on section completeness, skill richness, impact/action verbs, and role fit
- 💡 Rule-based **resume improvement suggestions**
- 📄 **Downloadable PDF report** of the full analysis
- 🎨 Custom, professional Streamlit UI (gradient header, skill badges, metric cards, charts)

## Tech Stack

- Python
- Streamlit (UI)
- scikit-learn (TF-IDF, cosine similarity)
- pdfplumber / python-docx (resume parsing)
- pandas (data display)
- reportlab (PDF report generation)

## Project Structure

```
AI-Resume-Analyzer/
├── app.py           # Main Streamlit app
├── skills_db.py      # Skills keyword database
├── job_roles.py       # Job role -> required skills mapping
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/primecrystal0/AI-Resume-Analyzer.git
cd AI-Resume-Analyzer
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## How It Works

1. **Parsing** – extracts raw text from the uploaded resume file.
2. **Skill Extraction (NLP)** – scans the cleaned text for known skill keywords across categories using regex word-boundary matching.
3. **Job Role Matching (ML)** – each job role's required-skills list is treated as a document. The resume and all role documents are vectorized with **TF-IDF**, then scored against each other with **cosine similarity**, blended with keyword overlap.
4. **Job Description Matching (ML)** – the same TF-IDF + cosine similarity technique applied between the resume and any specific job description the user pastes in.
5. **Resume Scoring** – a weighted composite score based on section completeness, number of skills detected, use of action verbs/quantifiable results, and best job-role fit.
6. **Improvement Suggestions** – rule-based checks surface actionable tips.

## Future Improvements

- Swap keyword matching for a transformer-based NER model for more robust skill extraction
- Add resume history/comparison across multiple uploads
- Support batch analysis of multiple resumes at once

## License

MIT