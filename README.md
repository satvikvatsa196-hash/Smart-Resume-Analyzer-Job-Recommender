# 🧠 Smart Resume Analyzer & Job Recommender

A production-grade NLP project built with **Streamlit**, **TF-IDF**, and **Cosine Similarity** that intelligently analyzes resumes and recommends the best-fit job roles.

---

## 🚀 Quick Start

```bash
# 1. Clone / extract this project
cd resume_analyzer

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Visit **http://localhost:8501** in your browser.

---

## 🏗️ Project Architecture

```
resume_analyzer/
│
├── app.py                    ← Main Streamlit UI (dark theme, full-featured)
│
├── data/
│   ├── skills_db.py          ← 300+ skills with categories & regex aliases
│   └── jobs_db.py            ← 50+ curated job descriptions (8 domains)
│
├── utils/
│   ├── extractor.py          ← PDF text extraction + NLP skill/exp detection
│   └── matcher.py            ← TF-IDF engine + job scoring + suggestions
│
├── requirements.txt
└── README.md
```

---

## 🧬 How the NLP Works

### 1. Text Extraction
`pdfplumber` extracts clean text from the uploaded PDF resume (handles multi-column layouts, tables, etc.)

### 2. Skill Extraction
A **regex-based pattern matcher** scans the resume text against a curated database of 300+ skills:
- Each skill has aliases (e.g., "Python" matches "python3", "py")
- Case-insensitive word-boundary matching prevents false positives
- Skills are categorized (ML, DevOps, Languages, etc.)

### 3. Job Matching — TF-IDF + Cosine Similarity
```
resume_vector = TF-IDF([resume_text + detected_skills × 3])
job_vectors   = TF-IDF([job_description_text] for each job)
tfidf_score   = cosine_similarity(resume_vector, job_vector)
skill_ratio   = matched_required_skills / total_required_skills
final_score   = 0.6 × tfidf_score + 0.4 × skill_ratio
```

The blended score ensures both semantic text similarity **and** explicit skill coverage matter.

### 4. Skill Gap Analysis
For each matched job, the system identifies:
- ✅ **Matched skills** — skills you already have
- ⚠️ **Missing skills** — required skills to develop
- 🌟 **Bonus skills** — nice-to-have skills you have

### 5. Improvement Suggestions
A rule-based engine analyzes:
- Most frequently missing skills across top matches
- Resume content quality (length, quantification, action verbs)
- Relevant certifications to pursue
- Portfolio and GitHub recommendations

---

## 📊 Features

| Feature | Details |
|---|---|
| PDF Upload | Handles multi-page PDFs with `pdfplumber` |
| Skill Extraction | 300+ skills, 8 categories, regex + alias matching |
| Job Database | 50+ roles, 8 domains, curated descriptions |
| Matching Engine | TF-IDF (1-2 grams) + cosine similarity |
| Visualizations | Radar chart, donut chart, histogram, gauge |
| Suggestions | Personalized, role-aware improvement tips |
| Filters | Domain filter, min score threshold |

---

## 🎨 UI Features

- **Dark theme** with custom CSS
- **4 tabs**: Job Matches, Skill Analysis, Visualizations, Suggestions
- **Interactive Plotly charts**: radar, pie, histogram, gauge
- **Expandable job cards** with matched/missing skill chips
- **Sidebar filters**: domain and minimum match score

---

## 🔧 Extending the Project

### Add More Jobs
Edit `data/jobs_db.py` and follow the existing schema:
```python
{
    "title": "Job Title",
    "domain": "Domain Name",
    "level": "Mid",
    "company_type": "Startup",
    "salary_range": "$100K – $150K",
    "description": "Human-readable description",
    "required_skills": ["Python", "Docker"],
    "nice_to_have": ["Kubernetes"],
    "description_text": "tfidf corpus text python docker kubernetes..."
}
```

### Add More Skills
Edit `data/skills_db.py`:
```python
{"name": "Rust", "category": "Programming Languages", "aliases": ["rust", "rust-lang"]}
```

### Improve Matching
In `utils/matcher.py`, you can swap TF-IDF for:
- **Sentence Transformers** (`sentence-transformers` library) for semantic embeddings
- **BM25** (`rank-bm25`) for better keyword matching
- **spaCy** NLP pipeline for entity-aware matching

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `pdfplumber` | PDF text extraction |
| `scikit-learn` | TF-IDF vectorization + cosine similarity |
| `plotly` | Interactive visualizations |
| `numpy` | Numerical operations |
| `pandas` | Data manipulation |

---

## 📸 App Screenshots (what to expect)

1. **Upload Page** — Dark hero header, drag-and-drop PDF upload
2. **Metrics Row** — Skills detected, jobs matched, top score, experience
3. **Job Matches Tab** — Ranked job cards with gauges and skill chips
4. **Skill Analysis Tab** — Detected skills by category + missing skills bar chart
5. **Visualizations Tab** — Radar chart by domain, skill donut, score histogram
6. **Suggestions Tab** — Personalized tips organized by category

---

*Built with Python · Streamlit · Scikit-learn · Plotly*
