import json
from dataclasses import dataclass
from pathlib import Path

from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import (
    build_weighted_resume_text,
    extract_keyphrases,
    extract_sections,
    extract_target_keywords,
    keyword_overlap,
    keyword_variants,
    normalize_text,
    resume_completeness,
    top_keywords,
)


@dataclass
class RoleProfile:
    role: str
    description: str
    keywords: list[str]
    family: str

    @property
    def combined_text(self) -> str:
        return f"{self.description} {' '.join(self.keywords)}"


class RecruitmentNLPSystem:
    def __init__(self, role_profile_path: Path, training_data_path: Path):
        raw_profiles = json.loads(role_profile_path.read_text(encoding="utf-8"))
        self.role_profiles = [
            RoleProfile(
                family=self._role_family(profile["role"]),
                **profile,
            )
            for profile in raw_profiles
        ]
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

        semantic_corpus = role_texts + training_texts
        self.semantic_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
        semantic_matrix = self.semantic_vectorizer.fit_transform(semantic_corpus)
        semantic_components = max(
            2,
            min(semantic_matrix.shape[0] - 1, semantic_matrix.shape[1] - 1, 64),
        )
        self.semantic_projector = TruncatedSVD(n_components=semantic_components, random_state=42)
        self.semantic_projector.fit(semantic_matrix)
        self.role_semantic_matrix = self._semantic_transform(role_texts)

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
        weighted_resume = build_weighted_resume_text(resume_text)
        classifier_vector = self._classifier_transform([weighted_resume])
        role_vector = self._role_transform([weighted_resume])
        semantic_vector = self._semantic_transform([weighted_resume])

        probabilities = self.classifier.predict_proba(classifier_vector)[0]
        similarity_scores = cosine_similarity(role_vector, self.role_matrix).flatten()
        semantic_scores = cosine_similarity(semantic_vector, self.role_semantic_matrix).flatten()
        similarity_by_role = {
            profile.role: float(score)
            for profile, score in zip(self.role_profiles, similarity_scores)
        }
        semantic_by_role = {
            profile.role: float(score)
            for profile, score in zip(self.role_profiles, semantic_scores)
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
            semantic_similarity = semantic_by_role[role_name]
            matched, missing, coverage = keyword_overlap(normalized_resume, profile.keywords)
            title_bonus = self._role_title_bonus(resume_text, role_name)
            family_bonus = self._family_alignment_bonus(normalized_resume, profile.family)
            score = self._blend_role_score(
                probability=float(probability),
                similarity=similarity,
                semantic_similarity=semantic_similarity,
                coverage=coverage / 100,
                completeness=completeness_score / 100,
                title_bonus=title_bonus,
                family_bonus=family_bonus,
            )
            ranked_roles.append(
                {
                    "role": role_name,
                    "family": profile.family,
                    "score": round(score, 2),
                    "confidence": round(float(probability) * 100, 2),
                    "profile_similarity": round(similarity * 100, 2),
                    "semantic_similarity": round(semantic_similarity * 100, 2),
                    "keyword_coverage": coverage,
                    "title_alignment": round(title_bonus * 100, 2),
                    "matched_keywords": matched[:6],
                    "missing_keywords": missing[:5],
                }
            )

        ranked_roles.sort(key=lambda item: item["score"], reverse=True)
        top_role = ranked_roles[0]
        improvement_tips = self._build_improvement_tips(top_role["missing_keywords"], present_sections)
        family_recommendations = self._family_recommendations(ranked_roles)

        return {
            "predicted_role": top_role["role"],
            "predicted_family": top_role["family"],
            "confidence": top_role["confidence"],
            "profile_similarity": top_role["profile_similarity"],
            "semantic_similarity": top_role["semantic_similarity"],
            "resume_strength": self._resume_strength(top_role["score"], completeness_score),
            "completeness_score": completeness_score,
            "present_sections": present_sections,
            "structured_sections": structured_sections,
            "keywords": extracted_keywords,
            "keyphrases": phrases,
            "recommendations": ranked_roles[:3],
            "family_recommendations": family_recommendations,
            "gaps": top_role["missing_keywords"],
            "improvement_tips": improvement_tips,
        }

    def rank_resumes(
        self,
        job_description: str,
        resume_batch: str,
        feedback_signals: dict | None = None,
    ) -> list[dict]:
        resumes = self._parse_resume_batch(resume_batch)
        if not resumes:
            return []

        job_keywords = extract_target_keywords(job_description, limit=10)
        weighted_job = self._weighted_job_representation(job_description)
        documents = [weighted_job] + [build_weighted_resume_text(item["content"]) for item in resumes]
        ranking_word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        ranking_char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        word_matrix = ranking_word_vectorizer.fit_transform(documents)
        char_matrix = ranking_char_vectorizer.fit_transform(documents)
        matrix = hstack([word_matrix, char_matrix])

        job_vector = matrix[0]
        resume_vectors = matrix[1:]
        similarity_scores = cosine_similarity(job_vector, resume_vectors).flatten()
        job_semantic_vector = self._semantic_transform([weighted_job])
        resume_semantic_vectors = self._semantic_transform(documents[1:])
        semantic_scores = cosine_similarity(job_semantic_vector, resume_semantic_vectors).flatten()

        ranked = []
        for item, similarity, semantic_similarity in sorted(
            zip(resumes, similarity_scores, semantic_scores),
            key=lambda pair: (pair[1] + pair[2]) / 2,
            reverse=True,
        ):
            analysis = self.analyze_resume(item["content"])
            matched_job_terms, missing_job_terms, job_keyword_coverage = keyword_overlap(
                item["content"], job_keywords
            )
            role_alignment = self._job_role_alignment(job_description, analysis["predicted_role"])
            feedback_boost = self._feedback_adjustment(
                analysis["predicted_role"],
                matched_job_terms + analysis["keywords"][:4],
                feedback_signals or {},
            )
            score = self._blend_ranking_score(
                similarity=float(similarity),
                semantic_similarity=float(semantic_similarity),
                confidence=analysis["confidence"] / 100,
                coverage=job_keyword_coverage / 100,
                completeness=analysis["completeness_score"] / 100,
                role_alignment=role_alignment,
                feedback_boost=feedback_boost,
            )
            ranked.append(
                {
                    "name": item["name"],
                    "score": round(score, 2),
                    "job_match": round(float(similarity) * 100, 2),
                    "semantic_match": round(float(semantic_similarity) * 100, 2),
                    "predicted_role": analysis["predicted_role"],
                    "predicted_family": analysis["predicted_family"],
                    "strength": analysis["resume_strength"],
                    "keyword_coverage": job_keyword_coverage,
                    "feedback_adjustment": round(feedback_boost * 100, 2),
                    "matched_job_terms": matched_job_terms[:5],
                    "missing_job_terms": missing_job_terms[:4],
                    "highlights": analysis["keywords"][:6],
                }
            )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked

    def project_summary(self) -> list[dict]:
        return [
            {
                "title": "Section-aware parsing",
                "detail": "Separates skills, projects, experience, education, and certifications to improve resume understanding.",
            },
            {
                "title": "Weighted role prediction",
                "detail": "Boosts skills, projects, and experience sections while combining word-level and character-level TF-IDF with logistic regression.",
            },
            {
                "title": "Semantic encoder layer",
                "detail": "Uses latent semantic embeddings over recruitment text so related wording matches even when exact keywords differ.",
            },
            {
                "title": "Evidence-based ranking",
                "detail": "Ranks candidates with lexical match, semantic match, keyword coverage, recruiter feedback memory, and resume completeness.",
            },
        ]

    def _build_training_examples(self, training_records: list[dict]) -> list[dict]:
        examples = list(training_records)
        for profile in self.role_profiles:
            keyword_text = ", ".join(profile.keywords[:5])
            variant_text = ", ".join(self._profile_aliases(profile))
            examples.append(
                {
                    "role": profile.role,
                    "text": (
                        f"Skills: {keyword_text}. Projects: {profile.description}. "
                        f"Experience: built practical systems in {profile.role.lower()}. "
                        f"Role family: {profile.family}. Title variants: {variant_text}."
                    ),
                }
            )
            examples.append(
                {
                    "role": profile.role,
                    "text": (
                        f"Resume summary for {profile.role.lower()}: {profile.description} "
                        f"Keywords: {' '.join(profile.keywords)}. Alternate titles: {variant_text}."
                    ),
                }
            )
            examples.append(
                {
                    "role": profile.role,
                    "text": (
                        f"Skills section: {' '.join(profile.keywords[:6])}. "
                        f"Projects section: delivered {profile.role.lower()} outcomes. "
                        f"Experience section: hands-on work in {profile.family.lower()}."
                    ),
                }
            )
        return examples

    def _classifier_transform(self, texts: list[str]):
        word_matrix = self.classifier_word_vectorizer.transform(texts)
        char_matrix = self.classifier_char_vectorizer.transform(texts)
        return hstack([word_matrix, char_matrix])

    def _semantic_transform(self, texts: list[str]):
        matrix = self.semantic_vectorizer.transform(texts)
        return self.semantic_projector.transform(matrix)

    def _role_transform(self, texts: list[str]):
        word_matrix = self.role_word_vectorizer.transform(texts)
        char_matrix = self.role_char_vectorizer.transform(texts)
        return hstack([word_matrix, char_matrix])

    def _blend_role_score(
        self,
        probability: float,
        similarity: float,
        semantic_similarity: float,
        coverage: float,
        completeness: float,
        title_bonus: float,
        family_bonus: float,
    ) -> float:
        return (
            probability * 0.28
            + similarity * 0.18
            + semantic_similarity * 0.22
            + coverage * 0.14
            + completeness * 0.12
            + title_bonus * 0.07
            + family_bonus * 0.05
        ) * 100

    def _blend_ranking_score(
        self,
        similarity: float,
        semantic_similarity: float,
        confidence: float,
        coverage: float,
        completeness: float,
        role_alignment: float,
        feedback_boost: float,
    ) -> float:
        return (
            similarity * 0.26
            + semantic_similarity * 0.24
            + confidence * 0.16
            + coverage * 0.16
            + completeness * 0.1
            + role_alignment * 0.1
            + feedback_boost * 0.08
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

    def _family_recommendations(self, ranked_roles: list[dict]) -> list[dict]:
        family_scores: dict[str, list[float]] = {}
        for role in ranked_roles[:8]:
            family_scores.setdefault(role["family"], []).append(role["score"])

        grouped = []
        for family, scores in family_scores.items():
            grouped.append(
                {
                    "family": family,
                    "score": round(sum(scores) / len(scores), 2),
                }
            )
        return sorted(grouped, key=lambda item: item["score"], reverse=True)[:3]

    def _role_family(self, role_name: str) -> str:
        mapping = {
            "Frontend Engineer": "Engineering",
            "Backend Engineer": "Engineering",
            "Full Stack Developer": "Engineering",
            "Mobile App Developer": "Engineering",
            "Cloud Engineer": "Cloud & Infrastructure",
            "DevOps Engineer": "Cloud & Infrastructure",
            "Site Reliability Engineer": "Cloud & Infrastructure",
            "Data Engineer": "Data & AI",
            "Data Scientist": "Data & AI",
            "Data Analyst": "Data & AI",
            "Machine Learning Engineer": "Data & AI",
            "NLP Engineer": "Data & AI",
            "MLOps Engineer": "Data & AI",
            "QA Engineer": "Quality & Security",
            "Automation Test Engineer": "Quality & Security",
            "Security Engineer": "Quality & Security",
            "Product Manager": "Product & Design",
            "UI/UX Designer": "Product & Design",
            "Solutions Architect": "Architecture & Strategy",
            "Software Architect": "Architecture & Strategy",
            "Technical Support Engineer": "Support & Operations",
        }
        return mapping.get(role_name, "General Software")

    def _resume_strength(self, role_score: float, completeness_score: int) -> str:
        combined = (role_score * 0.75) + (completeness_score * 0.25)
        if combined >= 78:
            return "Strong fit"
        if combined >= 58:
            return "Moderate fit"
        return "Emerging fit"

    def _profile_aliases(self, profile: RoleProfile) -> list[str]:
        aliases = {profile.role.lower(), profile.role.replace("Engineer", "Developer").lower()}
        for keyword in profile.keywords[:6]:
            aliases.update(keyword_variants(keyword))

        explicit = {
            "Frontend Engineer": {"frontend developer", "ui engineer", "web developer"},
            "Backend Engineer": {"backend developer", "software engineer backend", "api engineer"},
            "Full Stack Developer": {"full stack engineer", "fullstack developer"},
            "Mobile App Developer": {"android developer", "ios developer", "mobile engineer"},
            "DevOps Engineer": {"platform engineer", "infrastructure engineer"},
            "Cloud Engineer": {"cloud developer", "cloud infrastructure engineer"},
            "Site Reliability Engineer": {"sre engineer", "reliability engineer"},
            "Data Engineer": {"etl engineer", "analytics engineer"},
            "Data Scientist": {"applied scientist", "decision scientist"},
            "Data Analyst": {"business analyst", "reporting analyst"},
            "Machine Learning Engineer": {"ml engineer", "ai engineer"},
            "NLP Engineer": {"nlp developer", "language ai engineer"},
            "MLOps Engineer": {"ml ops engineer", "machine learning ops engineer"},
            "QA Engineer": {"quality engineer", "test engineer"},
            "Automation Test Engineer": {"qa automation engineer", "sdet"},
            "Security Engineer": {"application security engineer", "cybersecurity engineer"},
            "Product Manager": {"pm", "technical product manager"},
            "UI/UX Designer": {"product designer", "ux designer", "ui designer"},
            "Solutions Architect": {"technical architect", "solution architect"},
            "Software Architect": {"application architect", "technical architect"},
            "Technical Support Engineer": {"support engineer", "production support engineer"},
        }
        aliases.update(explicit.get(profile.role, set()))
        return sorted(alias for alias in aliases if alias)

    def _role_title_bonus(self, resume_text: str, role_name: str) -> float:
        normalized_resume = normalize_text(resume_text)
        profile = self.profile_lookup[role_name]
        aliases = self._profile_aliases(profile)
        title_hits = sum(1 for alias in aliases if alias and alias in normalized_resume)
        return min(1.0, title_hits / 3)

    def _family_alignment_bonus(self, normalized_resume: str, family_name: str) -> float:
        family_terms = {
            "Engineering": ["frontend", "backend", "full stack", "software engineering", "web"],
            "Data & AI": ["data", "analytics", "ml", "machine learning", "nlp", "model"],
            "Cloud & Infrastructure": ["cloud", "devops", "infrastructure", "deployment", "reliability"],
            "Quality & Security": ["testing", "qa", "quality", "security", "vulnerability"],
            "Product & Design": ["product", "roadmap", "wireframe", "prototype", "ux"],
            "Architecture & Strategy": ["architecture", "system design", "technical strategy", "scalability"],
            "Support & Operations": ["support", "troubleshooting", "incident", "customer"],
        }
        terms = family_terms.get(family_name, [])
        if not terms:
            return 0.0
        hits = sum(1 for term in terms if normalize_text(term) in normalized_resume)
        return min(1.0, hits / 3)

    def _weighted_job_representation(self, job_description: str) -> str:
        normalized = normalize_text(job_description)
        keywords = extract_target_keywords(job_description, limit=12)
        repeated_keywords = []
        for keyword in keywords:
            repeated_keywords.extend(keyword_variants(keyword)[:2])
        return " ".join([normalized] + repeated_keywords).strip()

    def _job_role_alignment(self, job_description: str, predicted_role: str) -> float:
        normalized_job = normalize_text(job_description)
        profile = self.profile_lookup.get(predicted_role)
        if not profile:
            return 0.0

        aliases = self._profile_aliases(profile)
        title_hit = any(alias in normalized_job for alias in aliases)
        keyword_hits = sum(
            1 for keyword in profile.keywords[:6] if any(variant in normalized_job for variant in keyword_variants(keyword))
        )
        keyword_score = min(1.0, keyword_hits / 4)
        return min(1.0, (0.55 if title_hit else 0.0) + (keyword_score * 0.45))

    def _feedback_adjustment(
        self,
        predicted_role: str,
        candidate_terms: list[str],
        feedback_signals: dict,
    ) -> float:
        if not feedback_signals:
            return 0.0

        shortlisted_roles = feedback_signals.get("shortlisted_roles", {})
        rejected_roles = feedback_signals.get("rejected_roles", {})
        shortlisted_terms = feedback_signals.get("shortlisted_terms", {})
        rejected_terms = feedback_signals.get("rejected_terms", {})

        role_signal = shortlisted_roles.get(predicted_role, 0) - rejected_roles.get(predicted_role, 0)
        term_signal = 0
        normalized_terms = {normalize_text(term) for term in candidate_terms if term}
        for term in normalized_terms:
            term_signal += shortlisted_terms.get(term, 0)
            term_signal -= rejected_terms.get(term, 0)

        raw_signal = (role_signal * 0.65) + (term_signal * 0.12)
        if raw_signal <= 0:
            return max(-0.35, raw_signal / 10)
        return min(0.35, raw_signal / 10)
