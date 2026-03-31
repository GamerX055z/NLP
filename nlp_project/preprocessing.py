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
    "summary": ("summary", "profile", "professional summary", "objective"),
    "skills": ("skills", "technical skills", "toolkit", "stack", "competencies"),
    "projects": ("projects", "project", "portfolio", "built", "developed"),
    "experience": ("experience", "employment", "internship", "worked", "work experience"),
    "education": ("education", "college", "university", "degree", "academic"),
    "certifications": ("certifications", "certification", "certified"),
}

TERM_VARIANTS = {
    "javascript": ("js", "javascript", "ecmascript"),
    "typescript": ("ts", "typescript"),
    "react": ("react", "reactjs", "react.js"),
    "next.js": ("next", "nextjs", "next.js"),
    "node.js": ("node", "nodejs", "node.js"),
    "ci/cd": ("ci/cd", "cicd", "continuous integration", "continuous delivery"),
    "machine learning": ("machine learning", "ml"),
    "natural language processing": ("natural language processing", "nlp"),
    "ui/ux": ("ui/ux", "ui ux", "ux/ui", "user experience", "user interface"),
    "quality assurance": ("quality assurance", "qa"),
    "scikit learn": ("scikit learn", "scikit-learn", "sklearn"),
    "power bi": ("power bi", "powerbi"),
    "named entity recognition": ("named entity recognition", "ner"),
    "application security": ("application security", "appsec"),
    "amazon web services": ("amazon web services", "aws"),
    "google cloud platform": ("google cloud platform", "gcp"),
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
    hint_lookup = {
        section: [normalize_text(hint) for hint in hints]
        for section, hints in SECTION_PATTERNS.items()
    }
    active_section = None

    for line in lines:
        lowered = normalize_text(line)

        matched_heading = None
        for section, hints in hint_lookup.items():
            if lowered in hints:
                matched_heading = section
                break

        if matched_heading:
            active_section = matched_heading
            continue

        if ":" in line:
            raw_label, raw_value = line.split(":", 1)
            label = normalize_text(raw_label)
            value = raw_value.strip()
            for section, hints in hint_lookup.items():
                if label in hints:
                    sections[section] = f"{sections[section]} {value}".strip()
                    active_section = section
                    break
            else:
                if active_section:
                    sections[active_section] = f"{sections[active_section]} {line}".strip()
        else:
            for section, hints in hint_lookup.items():
                if any(hint in lowered for hint in hints):
                    sections[section] = f"{sections[section]} {line}".strip()
                    active_section = section
                    break
            else:
                if active_section:
                    sections[active_section] = f"{sections[active_section]} {line}".strip()

    return {key: value.strip() for key, value in sections.items() if value.strip()}


def build_weighted_resume_text(text: str) -> str:
    normalized = normalize_text(text)
    sections = extract_sections(text)
    section_weights = {
        "skills": 4,
        "projects": 3,
        "experience": 3,
        "education": 1,
        "certifications": 1,
    }

    weighted_parts = [normalized]
    for section, weight in section_weights.items():
        section_text = sections.get(section)
        if not section_text:
            continue
        normalized_section = normalize_text(section_text)
        weighted_parts.extend([f"{section} {normalized_section}"] * weight)

    keywords = top_keywords(text, limit=10)
    phrases = extract_keyphrases(text, limit=6)
    weighted_parts.extend(keywords)
    weighted_parts.extend(phrases)
    return " ".join(part for part in weighted_parts if part).strip()


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
    matched = [keyword for keyword in keywords if any(variant in normalized for variant in keyword_variants(keyword))]
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


def keyword_variants(keyword: str) -> list[str]:
    normalized_keyword = normalize_text(keyword)
    variants = {normalized_keyword}

    if normalized_keyword in TERM_VARIANTS:
        variants.update(normalize_text(value) for value in TERM_VARIANTS[normalized_keyword])

    if "+" in normalized_keyword:
        variants.add(normalized_keyword.replace("+", " plus "))
    if "/" in normalized_keyword:
        variants.add(normalized_keyword.replace("/", " "))
        variants.add(normalized_keyword.replace("/", " and "))
    if "." in normalized_keyword:
        variants.add(normalized_keyword.replace(".", ""))

    compact = normalized_keyword.replace(" ", "")
    if compact != normalized_keyword:
        variants.add(compact)

    return [variant.strip() for variant in variants if variant.strip()]
