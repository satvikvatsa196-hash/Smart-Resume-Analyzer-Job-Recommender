import streamlit as st
import pdfplumber
import re
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Analyzer & Job Recommender",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --primary: #6C63FF;
    --secondary: #FF6B6B;
    --accent: #4ECDC4;
    --bg: #0F0F1A;
    --surface: #1A1A2E;
    --surface2: #16213E;
    --text: #E8E8F0;
    --text-muted: #888899;
    --border: #2A2A4A;
    --success: #51CF66;
    --warning: #FFD43B;
    --danger: #FF6B6B;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

/* Hero Header */
.hero-header {
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(108,99,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6C63FF, #4ECDC4, #FF6B6B);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.hero-subtitle {
    color: var(--text-muted);
    font-size: 1.1rem;
    margin-top: 0.75rem;
    font-weight: 400;
}

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.3s;
}
.card:hover { border-color: var(--primary); }
.card-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.75rem;
}

/* Match Score Badge */
.score-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.9rem;
}
.score-high { background: rgba(81,207,102,0.15); color: #51CF66; border: 1px solid rgba(81,207,102,0.3); }
.score-mid  { background: rgba(255,212,59,0.15);  color: #FFD43B;  border: 1px solid rgba(255,212,59,0.3);  }
.score-low  { background: rgba(255,107,107,0.15); color: #FF6B6B;  border: 1px solid rgba(255,107,107,0.3);  }

/* Job Card */
.job-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
    cursor: default;
}
.job-card:hover { border-color: var(--primary); transform: translateX(4px); }
.job-title { font-size: 1.1rem; font-weight: 600; color: var(--text); }
.job-meta  { font-size: 0.85rem; color: var(--text-muted); margin-top: 0.25rem; }

/* Skill Chip */
.skill-chip {
    display: inline-block;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 0.2rem;
}
.skill-match   { background: rgba(78,205,196,0.15); color: #4ECDC4;  border: 1px solid rgba(78,205,196,0.3);  }
.skill-missing { background: rgba(255,107,107,0.1); color: #FF8E8E;  border: 1px solid rgba(255,107,107,0.25); }
.skill-bonus   { background: rgba(108,99,255,0.15); color: #A29BFE; border: 1px solid rgba(108,99,255,0.3); }

/* Steps */
.step-pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    background: var(--primary);
    color: white;
    border-radius: 50%;
    font-size: 0.8rem;
    font-weight: 700;
    margin-right: 0.5rem;
}

/* Suggestion Box */
.suggestion-item {
    background: var(--surface2);
    border-left: 3px solid var(--primary);
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
    color: var(--text);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Metric */
.metric-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6C63FF, #4ECDC4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-label { color: var(--text-muted); font-size: 0.8rem; margin-top: 0.25rem; }

/* Divider */
.divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

/* Override Streamlit defaults */
.stButton > button {
    background: linear-gradient(135deg, var(--primary), #5A52E0) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stFileUploader {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
}

h1, h2, h3, h4 { color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)

# ── NLP & Data Utilities ─────────────────────────────────────────────────────
from data.skills_db import SKILLS_DB
from data.jobs_db import JOBS_DATABASE
from utils.extractor import extract_text_from_pdf, extract_skills, extract_experience_years, extract_education, clean_text
from utils.matcher import compute_job_matches, generate_suggestions

# ── Session State ────────────────────────────────────────────────────────────
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "extracted_skills" not in st.session_state:
    st.session_state.extracted_skills = []
if "job_matches" not in st.session_state:
    st.session_state.job_matches = []
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0;">
        <div style="font-size: 1.4rem; font-weight: 700; color: #6C63FF;">🧠 ResumeAI</div>
        <div style="color: #888899; font-size: 0.85rem; margin-top: 0.25rem;">NLP-Powered Career Tool</div>
    </div>
    <hr style="border: none; border-top: 1px solid #2A2A4A; margin: 0.5rem 0 1.5rem;">
    """, unsafe_allow_html=True)

    st.markdown("**How it works**")
    steps = [
        ("1", "Upload your PDF resume"),
        ("2", "AI extracts your skills"),
        ("3", "Matches against job roles"),
        ("4", "Get tailored suggestions"),
    ]
    for num, txt in steps:
        st.markdown(f'<div style="display:flex;align-items:center;margin-bottom:0.6rem;color:#C8C8D8;font-size:0.88rem;"><span class="step-pill">{num}</span>{txt}</div>', unsafe_allow_html=True)

    st.markdown('<hr style="border:none;border-top:1px solid #2A2A4A;margin:1.5rem 0;">', unsafe_allow_html=True)

    st.markdown("**Filter Jobs**")
    domain_filter = st.multiselect(
        "Job Domain",
        options=list(set(j["domain"] for j in JOBS_DATABASE)),
        default=[],
        placeholder="All domains"
    )
    min_match = st.slider("Min Match Score (%)", 0, 100, 20)

    st.markdown('<hr style="border:none;border-top:1px solid #2A2A4A;margin:1.5rem 0;">', unsafe_allow_html=True)
    st.markdown('<div style="color:#888899;font-size:0.75rem;text-align:center;">Powered by TF-IDF + Cosine Similarity</div>', unsafe_allow_html=True)

# ── Main Layout ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">Resume Analyzer &<br>Job Recommender</h1>
    <p class="hero-subtitle">
        Upload your resume → extract skills via NLP → match against 50+ job roles → get actionable insights
    </p>
</div>
""", unsafe_allow_html=True)

# Upload section
col_upload, col_info = st.columns([3, 2], gap="large")

with col_upload:
    st.markdown('<div class="card"><div class="card-title">📄 Upload Resume</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your PDF resume here",
        type=["pdf"],
        help="Only PDF files are supported"
    )

    if uploaded_file:
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
            st.session_state.resume_text = text

        if text:
            st.success(f"✅ Extracted {len(text.split())} words from {uploaded_file.name}")
            with st.expander("📋 Preview extracted text"):
                st.text(text[:1500] + ("..." if len(text) > 1500 else ""))
        else:
            st.error("Could not extract text. Ensure the PDF has selectable text (not scanned).")

    st.markdown('</div>', unsafe_allow_html=True)

    # Analyze button
    if st.session_state.resume_text:
        if st.button("🚀 Analyze Resume & Find Jobs", use_container_width=True):
            with st.spinner("Running NLP analysis..."):
                skills = extract_skills(st.session_state.resume_text, SKILLS_DB)
                st.session_state.extracted_skills = skills
                matches = compute_job_matches(
                    st.session_state.resume_text,
                    skills,
                    JOBS_DATABASE,
                    domain_filter=domain_filter if domain_filter else None,
                    min_score=min_match / 100
                )
                st.session_state.job_matches = matches
                st.session_state.analyzed = True
            st.rerun()

with col_info:
    st.markdown('<div class="card"><div class="card-title">ℹ️ What we analyze</div>', unsafe_allow_html=True)
    features = [
        ("🔍", "Skill Extraction", "300+ technical & soft skills"),
        ("📊", "TF-IDF Matching", "Semantic similarity scoring"),
        ("💼", "50+ Job Roles", "Across 8 tech domains"),
        ("📈", "Gap Analysis", "Skills you're missing per role"),
        ("💡", "Suggestions", "Personalized improvement tips"),
    ]
    for icon, title, desc in features:
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;margin-bottom:0.75rem;">
            <span style="font-size:1.2rem;margin-right:0.75rem;">{icon}</span>
            <div>
                <div style="font-weight:600;color:#E8E8F0;font-size:0.9rem;">{title}</div>
                <div style="color:#888899;font-size:0.8rem;">{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Analysis Results ──────────────────────────────────────────────────────────
if st.session_state.analyzed and st.session_state.extracted_skills:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Metrics Row ───────────────────────────────────────────────────────────
    exp_years = extract_experience_years(st.session_state.resume_text)
    edu = extract_education(st.session_state.resume_text)
    skills = st.session_state.extracted_skills
    matches = st.session_state.job_matches

    m1, m2, m3, m4 = st.columns(4)
    for col, val, label in [
        (m1, len(skills), "Skills Detected"),
        (m2, len(matches), "Jobs Matched"),
        (m3, f"{matches[0]['score']*100:.0f}%" if matches else "0%", "Top Match Score"),
        (m4, f"{exp_years}y" if exp_years else edu, "Experience / Edu"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["💼 Job Matches", "🔬 Skill Analysis", "📊 Visualizations", "💡 Suggestions"])

    # Tab 1 — Job Matches
    with tab1:
        if not matches:
            st.info("No matches found. Try lowering the minimum match score in the sidebar.")
        else:
            st.markdown(f"**Found {len(matches)} matching roles** sorted by compatibility")
            st.markdown("")

            for i, job in enumerate(matches[:15]):
                score_pct = job["score"] * 100
                if score_pct >= 70:
                    badge_cls, badge_icon = "score-high", "🟢"
                elif score_pct >= 45:
                    badge_cls, badge_icon = "score-mid", "🟡"
                else:
                    badge_cls, badge_icon = "score-low", "🔴"

                matched = job.get("matched_skills", [])
                missing = job.get("missing_skills", [])[:5]

                with st.expander(f"{badge_icon} #{i+1}  {job['title']}  —  {score_pct:.0f}% match  ·  {job['domain']}"):
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.markdown(f"**{job['company_type']}** · {job['level']} · {job['salary_range']}")
                        st.markdown(f"<div style='color:#888899;font-size:0.9rem;margin-top:0.5rem;'>{job['description']}</div>", unsafe_allow_html=True)

                        if matched:
                            st.markdown("**✅ Your matching skills:**")
                            chips = "".join(f'<span class="skill-chip skill-match">{s}</span>' for s in matched[:12])
                            st.markdown(chips, unsafe_allow_html=True)

                        if missing:
                            st.markdown("**⚠️ Skills to develop:**")
                            chips = "".join(f'<span class="skill-chip skill-missing">{s}</span>' for s in missing)
                            st.markdown(chips, unsafe_allow_html=True)

                    with c2:
                        # Score gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=score_pct,
                            number={"suffix": "%", "font": {"size": 24, "color": "#E8E8F0"}},
                            gauge={
                                "axis": {"range": [0, 100], "tickcolor": "#888899"},
                                "bar": {"color": "#6C63FF"},
                                "bgcolor": "#1A1A2E",
                                "bordercolor": "#2A2A4A",
                                "steps": [
                                    {"range": [0, 40], "color": "rgba(255,107,107,0.2)"},
                                    {"range": [40, 70], "color": "rgba(255,212,59,0.2)"},
                                    {"range": [70, 100], "color": "rgba(81,207,102,0.2)"},
                                ],
                            },
                            title={"text": "Match", "font": {"color": "#888899", "size": 13}},
                        ))
                        fig.update_layout(
                            height=180, margin=dict(l=10, r=10, t=30, b=10),
                            paper_bgcolor="rgba(0,0,0,0)", font_color="#E8E8F0"
                        )
                        st.plotly_chart(fig, use_container_width=True)

    # Tab 2 — Skill Analysis
    with tab2:
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("#### 🎯 Detected Skills")
            categories = {}
            for skill in skills:
                cat = skill.get("category", "General")
                categories.setdefault(cat, []).append(skill["name"])

            for cat, skill_list in sorted(categories.items()):
                st.markdown(f"**{cat}**")
                chips = "".join(f'<span class="skill-chip skill-bonus">{s}</span>' for s in skill_list)
                st.markdown(chips, unsafe_allow_html=True)
                st.markdown("")

        with right:
            st.markdown("#### 📉 Top Missing Skills")
            all_missing: Counter = Counter()
            for job in matches[:10]:
                for s in job.get("missing_skills", []):
                    all_missing[s] += 1

            if all_missing:
                top_missing = all_missing.most_common(15)
                names = [x[0] for x in top_missing]
                counts = [x[1] for x in top_missing]

                fig = go.Figure(go.Bar(
                    x=counts[::-1], y=names[::-1],
                    orientation='h',
                    marker=dict(
                        color=counts[::-1],
                        colorscale=[[0, '#FF6B6B'], [0.5, '#FFD43B'], [1, '#51CF66']],
                        showscale=False
                    ),
                    text=[f"  {c} jobs" for c in counts[::-1]],
                    textposition='outside',
                    textfont=dict(color='#888899', size=11),
                ))
                fig.update_layout(
                    height=420,
                    margin=dict(l=10, r=80, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                    yaxis=dict(tickfont=dict(color='#C8C8D8', size=11)),
                    font_color="#E8E8F0",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No missing skills detected across top matches!")

    # Tab 3 — Visualizations
    with tab3:
        v1, v2 = st.columns(2, gap="large")

        with v1:
            st.markdown("#### 🌐 Match Score by Domain")
            domain_scores: dict = {}
            for job in matches:
                d = job["domain"]
                domain_scores.setdefault(d, []).append(job["score"] * 100)
            domain_avg = {d: np.mean(v) for d, v in domain_scores.items()}

            if domain_avg:
                fig = go.Figure(go.Scatterpolar(
                    r=list(domain_avg.values()),
                    theta=list(domain_avg.keys()),
                    fill='toself',
                    fillcolor='rgba(108,99,255,0.2)',
                    line=dict(color='#6C63FF', width=2),
                    marker=dict(color='#4ECDC4', size=8),
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color='#888899', size=9)),
                        angularaxis=dict(tickfont=dict(color='#C8C8D8', size=10)),
                        bgcolor='rgba(26,26,46,0.8)'
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=360, margin=dict(l=30, r=30, t=30, b=30),
                    font_color="#E8E8F0",
                )
                st.plotly_chart(fig, use_container_width=True)

        with v2:
            st.markdown("#### 📊 Skill Category Breakdown")
            cat_counts = {cat: len(s_list) for cat, s_list in categories.items()}
            if cat_counts:
                fig = go.Figure(go.Pie(
                    labels=list(cat_counts.keys()),
                    values=list(cat_counts.values()),
                    hole=0.55,
                    marker=dict(colors=[
                        '#6C63FF', '#4ECDC4', '#FF6B6B', '#FFD43B',
                        '#51CF66', '#A29BFE', '#FD79A8', '#00CEC9'
                    ]),
                    textfont=dict(size=11, color='white'),
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=360, margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(font=dict(color='#C8C8D8', size=11), bgcolor='rgba(0,0,0,0)'),
                    font_color="#E8E8F0",
                    annotations=[dict(text=f"<b>{len(skills)}</b><br>skills", x=0.5, y=0.5,
                                      font_size=16, font_color='#E8E8F0', showarrow=False)]
                )
                st.plotly_chart(fig, use_container_width=True)

        # Score distribution
        st.markdown("#### 📈 Match Score Distribution")
        scores = [j["score"] * 100 for j in matches]
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=scores, nbinsx=20,
            marker_color='#6C63FF',
            opacity=0.8,
            name='Frequency'
        ))
        fig.add_vline(x=np.mean(scores), line_dash="dash", line_color="#4ECDC4",
                      annotation_text=f"Avg: {np.mean(scores):.0f}%", annotation_font_color="#4ECDC4")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,26,46,0.6)",
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="Match Score (%)", color='#888899', gridcolor='#2A2A4A'),
            yaxis=dict(title="Count", color='#888899', gridcolor='#2A2A4A'),
            font_color="#E8E8F0",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tab 4 — Suggestions
    with tab4:
        suggestions = generate_suggestions(skills, matches, st.session_state.resume_text)
        st.markdown("#### 💡 Personalized Improvement Suggestions")
        st.markdown("")

        for section, items in suggestions.items():
            st.markdown(f"**{section}**")
            for item in items:
                st.markdown(f'<div class="suggestion-item">{item}</div>', unsafe_allow_html=True)
            st.markdown("")

elif not st.session_state.analyzed:
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;color:#888899;">
        <div style="font-size:4rem;margin-bottom:1rem;">📄</div>
        <div style="font-size:1.2rem;font-weight:600;color:#C8C8D8;margin-bottom:0.5rem;">Upload your resume to get started</div>
        <div style="font-size:0.9rem;">PDF format · Instant NLP analysis · No data stored</div>
    </div>
    """, unsafe_allow_html=True)
