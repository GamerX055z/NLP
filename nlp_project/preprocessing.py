import re
from collections import Counter


STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "using",
    "was",
    "were",
    "with",
}

SECTION_PATTERNS = {
    "skills": ("skills", "technical skills", "toolkit", "stack", "competencies"),
    "projects": ("projects", "project", "portfolio", "built", "developed"),
    "experience": ("experience", "employment", "internship", "worked", "work experience"),
    "education": ("education", "college", "university", "degree", "academic"),
    "certifications": ("certifications", "certification", "certified"),
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#/\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    return [token for token in normalized.split() if token and token not in STOP_WORDS]


def extract_keyphrases(text: str, limit: int = 8) -> list[str]:
    normalized = normalize_text(text)
    words = normalized.split()
    candidates = []

    for size in (3, 2):
        for index in range(len(words) - size + 1):
            phrase_words = words[index:index + size]
            if any(word in STOP_WORDS for word in phrase_words):
                continue
            if all(len(word) <= 2 for word in phrase_words):
                continue
            candidates.append(" ".join(phrase_words))

    phrase_counts = Counter(candidates)
    return [phrase for phrase, _ in phrase_counts.most_common(limit)]


def top_keywords(text: str, limit: int = 10) -> list[str]:
    normalized = normalize_text(text)
    tokens = tokenize(normalized)
    phrases = extract_keyphrases(normalized, limit=limit)
    counts = Counter(tokens)

    keywords: list[str] = []
    for phrase in phrases:
        if phrase not in keywords:
            keywords.append(phrase)

    for token, _ in counts.most_common(limit * 3):
        if token not in keywords and token not in SECTION_PATTERNS:
            keywords.append(token)
        if len(keywords) >= limit:
            break

    return keywords[:limit]


def extract_sections(text: str) -> dict[str, str]:
    sections = {section: "" for section in SECTION_PATTERNS}
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines:
        lowered = normalize_text(line)
        if ":" in line:
            raw_label, raw_value = line.split(":", 1)
            label = normalize_text(raw_label)
            value = raw_value.strip()
            for section, hints in SECTION_PATTERNS.items():
                if label in [normalize_text(hint) for hint in hints]:
                    sections[section] = f"{sections[section]} {value}".strip()
        else:
            for section, hints in SECTION_PATTERNS.items():
                if any(hint in lowered for hint in hints):
                    sections[section] = f"{sections[section]} {line}".strip()

    return {key: value.strip() for key, value in sections.items() if value.strip()}


def resume_completeness(text: str) -> tuple[int, list[str], dict[str, str]]:
    sections = extract_sections(text)
    present_sections = list(sections)
    weighted_points = {
        "skills": 25,
        "projects": 25,
        "experience": 25,
        "education": 15,
        "certifications": 10,
    }
    score = sum(weighted_points.get(section, 0) for section in present_sections)
    return min(100, score), present_sections, sections


def keyword_overlap(text: str, keywords: list[str]) -> tuple[list[str], list[str], float]:
    normalized = normalize_text(text)
    matched = [keyword for keyword in keywords if normalize_text(keyword) in normalized]
    missing = [keyword for keyword in keywords if keyword not in matched]
    coverage = (len(matched) / len(keywords) * 100) if keywords else 0.0
    return matched, missing, round(coverage, 2)


def extract_target_keywords(text: str, limit: int = 10) -> list[str]:
    base = top_keywords(text, limit=limit)
    phrases = extract_keyphrases(text, limit=limit // 2 + 2)

    keywords: list[str] = []
    for item in base + phrases:
        if item not in keywords:
            keywords.append(item)
        if len(keywords) >= limit:
            break
    return keywords
