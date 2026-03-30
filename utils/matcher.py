"""
Job matching engine using TF-IDF + cosine similarity.
Also generates improvement suggestions based on skill gaps.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _build_resume_skill_text(resume_text: str, extracted_skills: List[Dict]) -> str:
    """Combine resume text with explicitly detected skill names for richer TF-IDF."""
    skill_names = " ".join(s["name"].lower().replace("-", " ") for s in extracted_skills)
    # Repeat skills 3× to give them higher TF-IDF weight
    return f"{resume_text.lower()} {skill_names} {skill_names} {skill_names}"


def compute_job_matches(
    resume_text: str,
    extracted_skills: List[Dict],
    jobs_db: List[Dict],
    domain_filter: Optional[List[str]] = None,
    min_score: float = 0.0,
) -> List[Dict]:
    """
    For each job, compute:
    1. TF-IDF cosine similarity score
    2. Skill overlap (matched / missing)
    Returns sorted list of job match dicts.
    """
    # Optionally filter by domain
    jobs = jobs_db
    if domain_filter:
        jobs = [j for j in jobs_db if j["domain"] in domain_filter]
    if not jobs:
        jobs = jobs_db  # fallback

    resume_doc = _build_resume_skill_text(resume_text, extracted_skills)
    job_docs = [j.get("description_text", j["description"]) for j in jobs]

    # ── TF-IDF vectorisation ──────────────────────────────────────────────
    corpus = [resume_doc] + job_docs
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=8000,
        sublinear_tf=True,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except Exception:
        return []

    resume_vec = tfidf_matrix[0]
    job_vecs = tfidf_matrix[1:]
    similarities = cosine_similarity(resume_vec, job_vecs)[0]

    # ── Skill overlap ─────────────────────────────────────────────────────
    user_skill_names = {s["name"].lower() for s in extracted_skills}

    results = []
    for i, job in enumerate(jobs):
        raw_score = float(similarities[i])

        required = [s.lower() for s in job.get("required_skills", [])]
        nice = [s.lower() for s in job.get("nice_to_have", [])]

        matched = [s for s in required if s in user_skill_names]
        missing = [s for s in required if s not in user_skill_names]
        bonus   = [s for s in nice if s in user_skill_names]

        # Skill overlap bonus: weight the TF-IDF score with skill match ratio
        skill_ratio = len(matched) / max(len(required), 1)
        blended_score = 0.6 * raw_score + 0.4 * skill_ratio

        if blended_score < min_score:
            continue

        results.append({
            **job,
            "score": round(blended_score, 4),
            "tfidf_score": round(raw_score, 4),
            "skill_ratio": round(skill_ratio, 4),
            "matched_skills": [s.title() for s in matched],
            "missing_skills": [s.title() for s in missing],
            "bonus_skills": [s.title() for s in bonus],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def generate_suggestions(
    extracted_skills: List[Dict],
    job_matches: List[Dict],
    resume_text: str,
) -> Dict[str, List[str]]:
    """Generate actionable, personalised improvement suggestions."""
    from collections import Counter

    suggestions: Dict[str, List[str]] = {}
    user_skill_names = {s["name"].lower() for s in extracted_skills}
    text_lower = resume_text.lower()

    # ── 1. Skills to Add ─────────────────────────────────────────────────
    missing_counter: Counter = Counter()
    for job in job_matches[:10]:
        for skill in job.get("missing_skills", []):
            missing_counter[skill] += 1

    top_missing = [s for s, _ in missing_counter.most_common(8)]
    if top_missing:
        skill_items = []
        for skill in top_missing:
            count = missing_counter[skill]
            skill_items.append(
                f"🔧 <b>{skill}</b> — required in {count} of your top matches. "
                f"Add a project or certification to demonstrate this."
            )
        suggestions["🎯 High-Impact Skills to Learn"] = skill_items

    # ── 2. Resume Content Improvements ───────────────────────────────────
    content_tips = []

    word_count = len(resume_text.split())
    if word_count < 400:
        content_tips.append("📝 Your resume seems short. Aim for at least 400–600 words to give enough context about your experience.")
    elif word_count > 1200:
        content_tips.append("✂️ Your resume may be too long. Try to keep it to 1–2 pages by focusing on the most impactful accomplishments.")

    # Quantification check
    has_numbers = bool(__import__('re').search(r'\d+[%xX]|\$\d|\d+\s*(users|customers|records|requests|ms|seconds|TB|GB|K|M)', text_lower))
    if not has_numbers:
        content_tips.append("📊 Add quantified achievements: e.g., 'Reduced API latency by 40%' or 'Scaled service to 1M users'. Numbers dramatically improve resume impact.")

    # Action verbs
    action_verbs = ['built', 'designed', 'implemented', 'led', 'developed', 'architected', 'optimized', 'deployed', 'created', 'scaled']
    verb_count = sum(1 for v in action_verbs if v in text_lower)
    if verb_count < 4:
        content_tips.append("💬 Use strong action verbs: Built, Designed, Led, Architected, Optimized, Deployed, Scaled. Replace passive language like 'was responsible for'.")

    # Summary section
    if not any(w in text_lower for w in ['summary', 'objective', 'profile', 'about me']):
        content_tips.append("🪪 Add a 3–4 line professional summary at the top highlighting your expertise, specialization, and years of experience.")

    if content_tips:
        suggestions["📄 Resume Content Improvements"] = content_tips

    # ── 3. Certifications to Consider ────────────────────────────────────
    cert_map = {
        "aws": ("AWS Certified Solutions Architect", "https://aws.amazon.com/certification/"),
        "google cloud": ("Google Professional Cloud Engineer", "https://cloud.google.com/certification/"),
        "azure": ("Microsoft Azure Fundamentals (AZ-900)", "https://learn.microsoft.com/en-us/certifications/"),
        "kubernetes": ("Certified Kubernetes Administrator (CKA)", "https://www.cncf.io/certification/cka/"),
        "data science": ("Google Data Analytics Certificate", "https://grow.google/certificates/"),
        "machine learning": ("DeepLearning.AI TensorFlow Developer", "https://www.coursera.org/professional-certificates/tensorflow-in-practice"),
        "cybersecurity": ("CompTIA Security+", "https://www.comptia.org/certifications/security"),
        "project management": ("PMP Certification", "https://www.pmi.org/certifications/project-management-pmp"),
    }

    cert_tips = []
    for keyword, (cert_name, link) in cert_map.items():
        if keyword in " ".join(top_missing).lower():
            cert_tips.append(f"🏅 <b>{cert_name}</b> — strengthens your profile for {keyword.title()} roles.")

    # If no targeted certs, give generic ones based on top domain
    if not cert_tips and job_matches:
        top_domain = job_matches[0]["domain"]
        if "ML" in top_domain or "Data" in top_domain:
            cert_tips.append("🏅 <b>AWS Certified Machine Learning Specialty</b> — highly valued for Data/ML roles.")
        elif "Security" in top_domain:
            cert_tips.append("🏅 <b>CompTIA Security+</b> — foundational credential for cybersecurity roles.")
        elif "Cloud" in top_domain or "DevOps" in top_domain:
            cert_tips.append("🏅 <b>CKA (Certified Kubernetes Administrator)</b> — greatly valued for cloud/devops roles.")

    if cert_tips:
        suggestions["🎓 Certifications to Consider"] = cert_tips

    # ── 4. Portfolio & GitHub ─────────────────────────────────────────────
    portfolio_tips = []
    has_github = 'github' in text_lower
    has_portfolio = any(w in text_lower for w in ['portfolio', 'projects', 'kaggle', 'leetcode'])

    if not has_github:
        portfolio_tips.append("🐙 Add your <b>GitHub profile</b> link. Recruiters heavily scrutinize open-source contributions and project quality.")
    if not has_portfolio:
        portfolio_tips.append("💼 Create a <b>portfolio website</b> or Notion page showcasing 2–3 projects with problem statement, tech stack, and live demo links.")

    # Domain-specific portfolio advice
    if job_matches:
        top_domain = job_matches[0].get("domain", "")
        if "ML" in top_domain or "Data" in top_domain:
            portfolio_tips.append("📓 Publish a <b>Kaggle notebook</b> or Hugging Face Space demonstrating a real ML project end-to-end.")
        elif "Frontend" in top_domain or "Full-Stack" in top_domain:
            portfolio_tips.append("🎨 Deploy a live <b>full-stack project</b> on Vercel/Netlify with source on GitHub — this is your best interview talking point.")
        elif "Security" in top_domain:
            portfolio_tips.append("🔐 Create a <b>CTF writeup blog</b> (HackTheBox/TryHackMe) to demonstrate hands-on security skills.")

    if portfolio_tips:
        suggestions["🚀 Portfolio & Online Presence"] = portfolio_tips

    # ── 5. Career Growth ──────────────────────────────────────────────────
    growth_tips = []
    if job_matches:
        top_level = job_matches[0].get("level", "")
        if "Senior" in top_level or "Staff" in top_level:
            growth_tips.append("🔑 For senior roles, highlight <b>scope of impact</b>: mention team size led, system scale (e.g., 'handled 10K RPS'), or cost savings achieved.")
        if "Management" in job_matches[0].get("domain", ""):
            growth_tips.append("👥 Showcase <b>mentorship and team impact</b>: 'Mentored 3 junior engineers', 'Defined team OKRs', 'Reduced on-call incidents by 60%'.")

    if len(user_skill_names) < 10:
        growth_tips.append("🌱 You have fewer than 10 detected skills. Expand your tech vocabulary — pick one new framework or tool to learn this quarter.")
    elif len(user_skill_names) > 25:
        growth_tips.append("🎯 You have a broad skill set. Consider <b>specializing deeper</b> in 2–3 areas to stand out for senior/specialist roles.")

    if growth_tips:
        suggestions["📈 Career Growth Tips"] = growth_tips

    return suggestions
