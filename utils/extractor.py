"""
Utilities for extracting text and skills from resumes.
"""
import re
import io
from typing import List, Dict, Any

import pdfplumber


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract all text from an uploaded PDF file."""
    try:
        file_bytes = uploaded_file.read()
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
        return "\n".join(pages_text)
    except Exception as e:
        return ""


def clean_text(text: str) -> str:
    """Normalize whitespace and lower-case text for matching."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def extract_skills(resume_text: str, skills_db: List[Dict]) -> List[Dict]:
    """
    Extract skills from resume text by matching against the skills database.
    Returns list of matched skill dicts with name and category.
    """
    text_lower = resume_text.lower()
    found = {}

    for skill_entry in skills_db:
        name = skill_entry["name"]
        aliases = skill_entry.get("aliases", [])
        category = skill_entry.get("category", "General")

        search_terms = [name.lower()] + [a.lower() for a in aliases]

        for term in search_terms:
            # Build pattern: exact word-boundary match
            # Some aliases already contain regex patterns (e.g. r"\bc\b")
            try:
                if term.startswith(r'\b') or '\\b' in term:
                    pattern = term
                else:
                    # Escape for regex and wrap in word boundaries
                    escaped = re.escape(term)
                    pattern = r'\b' + escaped + r'\b'

                if re.search(pattern, text_lower):
                    if name not in found:
                        found[name] = {"name": name, "category": category}
                    break
            except re.error:
                # Fallback to simple substring match
                if term in text_lower:
                    if name not in found:
                        found[name] = {"name": name, "category": category}
                    break

    return list(found.values())


def extract_experience_years(resume_text: str) -> int:
    """Heuristically extract total years of experience from resume text."""
    text_lower = resume_text.lower()

    # Pattern: "X years of experience" / "X+ years"
    patterns = [
        r'(\d+)\+?\s*years?\s+of\s+(?:professional\s+)?experience',
        r'(\d+)\+?\s*years?\s+(?:in|of|working)',
        r'experience\s*[:\-]?\s*(\d+)\+?\s*years?',
    ]
    for pat in patterns:
        match = re.search(pat, text_lower)
        if match:
            return int(match.group(1))

    # Fallback: count year ranges like "2019 – 2023"
    year_ranges = re.findall(r'(20\d\d|19\d\d)\s*[-–—]\s*(20\d\d|present|current)', text_lower)
    if year_ranges:
        total = 0
        import datetime
        current_year = datetime.datetime.now().year
        for start_y, end_y in year_ranges:
            s = int(start_y)
            e = current_year if end_y in ('present', 'current') else int(end_y)
            total += max(0, e - s)
        return total if total > 0 else 0

    return 0


def extract_education(resume_text: str) -> str:
    """Extract highest education level from resume text."""
    text_lower = resume_text.lower()

    if any(w in text_lower for w in ['ph.d', 'phd', 'doctor of philosophy', 'doctorate']):
        return "PhD"
    if any(w in text_lower for w in ["master's", 'masters', 'm.s.', 'msc', 'm.tech', 'm.e.', 'mba']):
        return "Master's"
    if any(w in text_lower for w in ["bachelor's", 'bachelors', 'b.s.', 'b.e.', 'b.tech', 'b.sc', 'undergraduate']):
        return "Bachelor's"
    if any(w in text_lower for w in ['associate', 'diploma', 'certification', 'certified']):
        return "Cert/Diploma"

    return "N/A"
