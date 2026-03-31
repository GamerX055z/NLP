import json
from dataclasses import dataclass
from pathlib import Path

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import (
    extract_keyphrases,
    extract_sections,
    extract_target_keywords,
    keyword_overlap,
    normalize_text,
    resume_completeness,
    top_keywords,
)


@dataclass
class RoleProfile:
    role: str
    description: str
    keywords: list[str]

    @property
    def combined_text(self) -> str:
        return f"{self.description} {' '.join(self.keywords)}"


class RecruitmentNLPSystem:
    def __init__(self, role_profile_path: Path, training_data_path: Path):
        raw_profiles = json.loads(role_profile_path.read_text(encoding="utf-8"))
        self.role_profiles = [RoleProfile(**profile) for profile in raw_profiles]
        self.profile_lookup = {profile.role: profile for profile in self.role_profiles}

        role_texts = [profile.combined_text for profile in self.role_profiles]
        self.role_word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.role_char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        role_word_matrix = self.role_word_vectorizer.fit_transform(role_texts)
        role_char_matrix = self.role_char_vectorizer.fit_transform(role_texts)
        self.role_matrix = hstack([role_word_matrix, role_char_matrix])

        training_records = json.loads(training_data_path.read_text(encoding="utf-8"))
        training_examples = self._build_training_examples(training_records)
        training_texts = [record["text"] for record in training_examples]
        training_labels = [record["role"] for record in training_examples]

        self.classifier_word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1,
        )
        self.classifier_char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
        )
        training_word_matrix = self.classifier_word_vectorizer.fit_transform(training_texts)
        training_char_matrix = self.classifier_char_vectorizer.fit_transform(training_texts)
        training_matrix = hstack([training_word_matrix, training_char_matrix])

        self.classifier = LogisticRegression(max_iter=2500, class_weight="balanced")
        self.classifier.fit(training_matrix, training_labels)

    def analyze_resume(self, resume_text: str) -> dict:
        normalized_resume = normalize_text(resume_text)
        classifier_vector = self._classifier_transform([normalized_resume])
        role_vector = self._role_transform([normalized_resume])

        probabilities = self.classifier.predict_proba(classifier_vector)[0]
        similarity_scores = cosine_similarity(role_vector, self.role_matrix).flatten()
        similarity_by_role = {
            profile.role: float(score)
            for profile, score in zip(self.role_profiles, similarity_scores)
        }

        completeness_score, present_sections, structured_sections = resume_completeness(resume_text)
        extracted_keywords = top_keywords(resume_text, limit=12)
        phrases = extract_keyphrases(resume_text, limit=6)

        ranked_roles = []
        for role_name, probability in sorted(
            zip(self.classifier.classes_, probabilities),
            key=lambda item: item[1],
            reverse=True,
        ):
            profile = self.profile_lookup[role_name]
            similarity = similarity_by_role[role_name]
            matched, missing, coverage = keyword_overlap(normalized_resume, profile.keywords)
            score = self._blend_role_score(
                probability=float(probability),
                similarity=similarity,
                coverage=coverage / 100,
                completeness=completeness_score / 100,
            )
            ranked_roles.append(
                {
                    "role": role_name,
                    "score": round(score, 2),
                    "confidence": round(float(probability) * 100, 2),
                    "profile_similarity": round(similarity * 100, 2),
                    "keyword_coverage": coverage,
                    "matched_keywords": matched[:6],
                    "missing_keywords": missing[:5],
                }
            )

        top_role = ranked_roles[0]
        improvement_tips = self._build_improvement_tips(top_role["missing_keywords"], present_sections)

        return {
            "predicted_role": top_role["role"],
            "confidence": top_role["confidence"],
            "profile_similarity": top_role["profile_similarity"],
            "resume_strength": self._resume_strength(top_role["score"], completeness_score),
            "completeness_score": completeness_score,
            "present_sections": present_sections,
            "structured_sections": structured_sections,
            "keywords": extracted_keywords,
            "keyphrases": phrases,
            "recommendations": ranked_roles[:3],
            "gaps": top_role["missing_keywords"],
            "improvement_tips": improvement_tips,
        }

    def rank_resumes(self, job_description: str, resume_batch: str) -> list[dict]:
        resumes = self._parse_resume_batch(resume_batch)
        if not resumes:
            return []

        job_keywords = extract_target_keywords(job_description, limit=10)
        documents = [normalize_text(job_description)] + [
            normalize_text(item["content"]) for item in resumes
        ]
        ranking_word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        ranking_char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        word_matrix = ranking_word_vectorizer.fit_transform(documents)
        char_matrix = ranking_char_vectorizer.fit_transform(documents)
        matrix = hstack([word_matrix, char_matrix])

        job_vector = matrix[0]
        resume_vectors = matrix[1:]
        similarity_scores = cosine_similarity(job_vector, resume_vectors).flatten()

        ranked = []
        for item, similarity in sorted(
            zip(resumes, similarity_scores), key=lambda pair: pair[1], reverse=True
        ):
            analysis = self.analyze_resume(item["content"])
            matched_job_terms, missing_job_terms, job_keyword_coverage = keyword_overlap(
                item["content"], job_keywords
            )
            score = self._blend_ranking_score(
                similarity=float(similarity),
                confidence=analysis["confidence"] / 100,
                coverage=job_keyword_coverage / 100,
                completeness=analysis["completeness_score"] / 100,
            )
            ranked.append(
                {
                    "name": item["name"],
                    "score": round(score, 2),
                    "job_match": round(float(similarity) * 100, 2),
                    "predicted_role": analysis["predicted_role"],
                    "strength": analysis["resume_strength"],
                    "keyword_coverage": job_keyword_coverage,
                    "matched_job_terms": matched_job_terms[:5],
                    "missing_job_terms": missing_job_terms[:4],
                    "highlights": analysis["keywords"][:6],
                }
            )
        return ranked

    def project_summary(self) -> list[dict]:
        return [
            {
                "title": "Section-aware parsing",
                "detail": "Separates skills, projects, experience, education, and certifications to improve resume understanding.",
            },
            {
                "title": "Hybrid role prediction",
                "detail": "Combines word-level and character-level TF-IDF features with logistic regression for more stable predictions.",
            },
            {
                "title": "Evidence-based ranking",
                "detail": "Ranks candidates with job similarity, keyword coverage, and resume completeness instead of similarity alone.",
            },
        ]

    def _build_training_examples(self, training_records: list[dict]) -> list[dict]:
        examples = list(training_records)
        for profile in self.role_profiles:
            keyword_text = ", ".join(profile.keywords[:5])
            examples.append(
                {
                    "role": profile.role,
                    "text": f"Skills: {keyword_text}. Projects: {profile.description}. Experience: built practical systems in {profile.role.lower()}.",
                }
            )
            examples.append(
                {
                    "role": profile.role,
                    "text": f"Resume summary for {profile.role.lower()}: {profile.description} Keywords: {' '.join(profile.keywords)}.",
                }
            )
        return examples

    def _classifier_transform(self, texts: list[str]):
        word_matrix = self.classifier_word_vectorizer.transform(texts)
        char_matrix = self.classifier_char_vectorizer.transform(texts)
        return hstack([word_matrix, char_matrix])

    def _role_transform(self, texts: list[str]):
        word_matrix = self.role_word_vectorizer.transform(texts)
        char_matrix = self.role_char_vectorizer.transform(texts)
        return hstack([word_matrix, char_matrix])

    def _blend_role_score(
        self,
        probability: float,
        similarity: float,
        coverage: float,
        completeness: float,
    ) -> float:
        return (
            probability * 0.42
            + similarity * 0.28
            + coverage * 0.18
            + completeness * 0.12
        ) * 100

    def _blend_ranking_score(
        self,
        similarity: float,
        confidence: float,
        coverage: float,
        completeness: float,
    ) -> float:
        return (
            similarity * 0.48
            + confidence * 0.22
            + coverage * 0.2
            + completeness * 0.1
        ) * 100

    def _build_improvement_tips(self, missing_keywords: list[str], sections: list[str]) -> list[str]:
        tips = []
        if "projects" not in sections:
            tips.append("Add a projects section with concrete NLP or software outcomes.")
        if "experience" not in sections:
            tips.append("Include internships, work experience, or role-specific responsibilities.")
        if missing_keywords:
            tips.append(f"Highlight missing target skills such as {', '.join(missing_keywords[:3])}.")
        if "skills" not in sections:
            tips.append("Create a focused skills section so technical terms are easier to detect.")
        return tips[:4]

    def _parse_resume_batch(self, resume_batch: str) -> list[dict]:
        sections = [chunk.strip() for chunk in resume_batch.split("\n---\n") if chunk.strip()]
        parsed = []
        for index, section in enumerate(sections, start=1):
            lines = [line.strip() for line in section.splitlines() if line.strip()]
            if not lines:
                continue
            header = lines[0]
            content = " ".join(lines[1:]) if len(lines) > 1 else header
            parsed.append(
                {
                    "name": header.replace("Resume:", "").strip() or f"Candidate {index}",
                    "content": content,
                    "sections": extract_sections(section),
                }
            )
        return parsed

    def _resume_strength(self, role_score: float, completeness_score: int) -> str:
        combined = (role_score * 0.75) + (completeness_score * 0.25)
        if combined >= 78:
            return "Strong fit"
        if combined >= 58:
            return "Moderate fit"
        return "Emerging fit"
