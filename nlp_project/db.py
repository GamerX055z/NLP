import json
import sqlite3
from pathlib import Path

from .preprocessing import normalize_text


class TalentLensDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._ensure_parent()
        self.init_db()

    def _ensure_parent(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def init_db(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('jobseeker', 'recruiter')),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    analysis_type TEXT NOT NULL CHECK(analysis_type IN ('jobseeker', 'recruiter')),
                    title TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS recruiter_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    candidate_name TEXT NOT NULL,
                    predicted_role TEXT NOT NULL,
                    decision TEXT NOT NULL CHECK(decision IN ('shortlist', 'reject')),
                    job_description TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );
                """
            )

    def create_user(self, name: str, email: str, password_hash: str, role: str) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO users (name, email, password_hash, role)
                VALUES (?, ?, ?, ?)
                """,
                (name, email.lower().strip(), password_hash, role),
            )
            return int(cursor.lastrowid)

    def get_user_by_email(self, email: str):
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM users WHERE email = ?",
                (email.lower().strip(),),
            ).fetchone()

    def get_user_by_id(self, user_id: int):
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()

    def save_analysis(
        self,
        user_id: int,
        analysis_type: str,
        title: str,
        input_text: str,
        result_payload: dict,
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO analyses (user_id, analysis_type, title, input_text, result_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    analysis_type,
                    title,
                    input_text,
                    json.dumps(result_payload, ensure_ascii=True),
                ),
            )
            return int(cursor.lastrowid)

    def list_recent_analyses(self, user_id: int, analysis_type: str, limit: int = 5) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, title, input_text, result_json, created_at
                FROM analyses
                WHERE user_id = ? AND analysis_type = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, analysis_type, limit),
            ).fetchall()

        analyses = []
        for row in rows:
            analyses.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "input_text": row["input_text"],
                    "result": json.loads(row["result_json"]),
                    "created_at": row["created_at"],
                }
            )
        return analyses

    def save_recruiter_feedback(
        self,
        user_id: int,
        candidate_name: str,
        predicted_role: str,
        decision: str,
        job_description: str,
        evidence_payload: dict,
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO recruiter_feedback (
                    user_id, candidate_name, predicted_role, decision, job_description, evidence_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    candidate_name,
                    predicted_role,
                    decision,
                    job_description,
                    json.dumps(evidence_payload, ensure_ascii=True),
                ),
            )
            return int(cursor.lastrowid)

    def recruiter_feedback_profile(self, user_id: int, limit: int = 80) -> dict:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT predicted_role, decision, job_description, evidence_json
                FROM recruiter_feedback
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()

        profile = {
            "shortlisted_roles": {},
            "rejected_roles": {},
            "shortlisted_terms": {},
            "rejected_terms": {},
            "total_feedback": len(rows),
        }

        for row in rows:
            role_bucket = (
                profile["shortlisted_roles"]
                if row["decision"] == "shortlist"
                else profile["rejected_roles"]
            )
            role_bucket[row["predicted_role"]] = role_bucket.get(row["predicted_role"], 0) + 1

            evidence = json.loads(row["evidence_json"])
            terms = set(evidence.get("matched_job_terms", [])) | set(evidence.get("highlights", []))
            target_bucket = (
                profile["shortlisted_terms"]
                if row["decision"] == "shortlist"
                else profile["rejected_terms"]
            )
            for term in terms:
                normalized_term = normalize_text(term)
                if normalized_term:
                    target_bucket[normalized_term] = target_bucket.get(normalized_term, 0) + 1

        return profile
