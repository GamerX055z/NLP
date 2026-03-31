from itertools import zip_longest
from pathlib import Path

from flask import Flask, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from nlp_project.db import TalentLensDB
from nlp_project.file_parsing import allowed_file, extract_text_from_upload
from nlp_project.recommender import RecruitmentNLPSystem


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "instance"
DEFAULT_JOB = (
    "Looking for an NLP engineer with Python, machine learning, transformers, "
    "spaCy, and text classification experience."
)
FAMILY_META = {
    "Engineering": {"icon": "Code", "slug": "engineering"},
    "Data & AI": {"icon": "AI", "slug": "data-ai"},
    "Cloud & Infrastructure": {"icon": "Cloud", "slug": "cloud"},
    "Quality & Security": {"icon": "Shield", "slug": "quality"},
    "Product & Design": {"icon": "Design", "slug": "product"},
    "Architecture & Strategy": {"icon": "Arc", "slug": "architecture"},
    "Support & Operations": {"icon": "Ops", "slug": "support"},
    "General Software": {"icon": "Tech", "slug": "general"},
}

app = Flask(__name__)
app.secret_key = "talentlens-dev-key"

system = RecruitmentNLPSystem(
    DATA_DIR / "role_profiles.json",
    DATA_DIR / "training_samples.json",
)
db = TalentLensDB(DB_DIR / "talentlens.db")


@app.context_processor
def inject_family_meta():
    def family_meta(family_name: str) -> dict:
        return FAMILY_META.get(family_name, FAMILY_META["General Software"])

    return {"family_meta": family_meta}


def current_user() -> dict | None:
    user_id = session.get("user_id")
    if not user_id:
        return None

    user = db.get_user_by_id(int(user_id))
    if not user:
        session.clear()
        return None

    return {
        "id": user["id"],
        "role": user["role"],
        "name": user["name"],
        "email": user["email"],
    }


def require_role(role: str):
    user = current_user()
    if not user or user["role"] != role:
        return None
    return user


def sample_resume() -> str:
    return (DATA_DIR / "sample_resumes.txt").read_text(encoding="utf-8").split("---")[0].strip()


def sample_batch() -> str:
    return (DATA_DIR / "sample_resumes.txt").read_text(encoding="utf-8")


def sample_candidates() -> list[dict]:
    entries = []
    for chunk in sample_batch().split("---"):
        block = chunk.strip()
        if not block:
            continue
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        name = lines[0].replace("Resume:", "").strip() or "Candidate"
        resume_text = "\n".join(lines[1:]).strip()
        entries.append({"name": name, "resume_text": resume_text})
    return entries


def parse_uploaded_resume(file_storage) -> tuple[str, str | None]:
    if not file_storage or not file_storage.filename:
        return "", None
    if not allowed_file(file_storage.filename):
        return "", f"{file_storage.filename}: unsupported file type"
    try:
        text = extract_text_from_upload(file_storage)
    except Exception:
        return "", f"{file_storage.filename}: could not extract text"
    if not text.strip():
        return "", f"{file_storage.filename}: no readable text found"
    return text.strip(), None


def candidate_entries_from_form() -> tuple[list[dict], list[str]]:
    names = request.form.getlist("candidate_name")
    resumes = request.form.getlist("candidate_resume")
    files = request.files.getlist("candidate_file")
    entries = []
    parse_notes = []

    for index, bundle in enumerate(zip_longest(names, resumes, files, fillvalue=None), start=1):
        name, resume_text, file_storage = bundle
        clean_name = (name or "").strip() or f"Candidate {index}"
        manual_text = (resume_text or "").strip()
        uploaded_text, note = parse_uploaded_resume(file_storage)
        if note:
            parse_notes.append(note)

        combined_parts = [part for part in [manual_text, uploaded_text] if part]
        combined_text = "\n\n".join(combined_parts).strip()
        if combined_text:
            entries.append({"name": clean_name, "resume_text": combined_text})

    return entries, parse_notes


def build_resume_batch(candidate_entries: list[dict]) -> str:
    blocks = []
    for entry in candidate_entries:
        blocks.append(f"Resume: {entry['name']}\n{entry['resume_text']}")
    return "\n---\n".join(blocks)


def save_jobseeker_history(user_id: int, resume_text: str, analysis: dict) -> None:
    title = f"{analysis['predicted_role']} analysis"
    payload = {
        "predicted_role": analysis["predicted_role"],
        "confidence": analysis["confidence"],
        "completeness_score": analysis["completeness_score"],
        "resume_strength": analysis["resume_strength"],
        "gaps": analysis["gaps"],
    }
    db.save_analysis(user_id, "jobseeker", title, resume_text, payload)


def save_recruiter_history(user_id: int, job_description: str, ranking: list[dict]) -> None:
    title = "Candidate ranking run"
    payload = {
        "top_candidates": ranking[:3],
        "total_candidates": len(ranking),
    }
    db.save_analysis(user_id, "recruiter", title, job_description, payload)


@app.route("/", methods=["GET"])
def home():
    return render_template("home.html", project_summary=system.project_summary(), user=current_user())


@app.route("/login/<user_role>", methods=["GET", "POST"])
def login(user_role: str):
    if user_role not in {"jobseeker", "recruiter"}:
        return redirect(url_for("home"))

    role_title = "Job Seeker" if user_role == "jobseeker" else "Recruiter"
    error = None

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        user = db.get_user_by_email(email)

        if not user or user["role"] != user_role or not check_password_hash(user["password_hash"], password):
            error = "Invalid credentials for this account type."
        else:
            session["user_id"] = int(user["id"])
            destination = "jobseeker_dashboard" if user_role == "jobseeker" else "recruiter_dashboard"
            return redirect(url_for(destination))

    return render_template(
        "login.html",
        user_role=user_role,
        role_title=role_title,
        mode="login",
        error=error,
    )


@app.route("/register/<user_role>", methods=["GET", "POST"])
def register(user_role: str):
    if user_role not in {"jobseeker", "recruiter"}:
        return redirect(url_for("home"))

    role_title = "Job Seeker" if user_role == "jobseeker" else "Recruiter"
    error = None

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        if not name or not email or not password:
            error = "Name, email, and password are required."
        elif len(password) < 6:
            error = "Use a password with at least 6 characters."
        elif db.get_user_by_email(email):
            error = "An account with this email already exists."
        else:
            user_id = db.create_user(
                name=name,
                email=email,
                password_hash=generate_password_hash(password),
                role=user_role,
            )
            session["user_id"] = user_id
            destination = "jobseeker_dashboard" if user_role == "jobseeker" else "recruiter_dashboard"
            return redirect(url_for(destination))

    return render_template(
        "login.html",
        user_role=user_role,
        role_title=role_title,
        mode="register",
        error=error,
    )


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.route("/jobseeker", methods=["GET"])
def jobseeker_dashboard():
    user = require_role("jobseeker")
    if not user:
        return redirect(url_for("login", user_role="jobseeker"))

    return render_template(
        "jobseeker.html",
        user=user,
        project_summary=system.project_summary(),
        default_resume=sample_resume(),
        resume_text=None,
        recent_analyses=db.list_recent_analyses(user["id"], "jobseeker"),
        parse_notes=[],
    )


@app.route("/jobseeker/analyze", methods=["POST"])
def jobseeker_analyze():
    user = require_role("jobseeker")
    if not user:
        return redirect(url_for("login", user_role="jobseeker"))

    manual_resume = request.form.get("resume_text", "").strip()
    uploaded_resume, note = parse_uploaded_resume(request.files.get("resume_file"))
    parse_notes = [note] if note else []
    if uploaded_resume:
        parse_notes.append("Uploaded resume file extracted successfully and included in the analysis.")

    resume_text = "\n\n".join(part for part in [manual_resume, uploaded_resume] if part).strip()
    resume_analysis = system.analyze_resume(resume_text) if resume_text else None
    recommendation = resume_analysis["recommendations"] if resume_analysis else []
    keywords = resume_analysis["keywords"] if resume_analysis else []

    if resume_analysis and resume_text:
        save_jobseeker_history(user["id"], resume_text, resume_analysis)

    return render_template(
        "jobseeker.html",
        user=user,
        resume_text=manual_resume,
        resume_analysis=resume_analysis,
        recommendation=recommendation,
        keywords=keywords,
        project_summary=system.project_summary(),
        default_resume=sample_resume(),
        recent_analyses=db.list_recent_analyses(user["id"], "jobseeker"),
        parse_notes=parse_notes,
    )


@app.route("/recruiter", methods=["GET"])
def recruiter_dashboard():
    user = require_role("recruiter")
    if not user:
        return redirect(url_for("login", user_role="recruiter"))

    feedback_profile = db.recruiter_feedback_profile(user["id"])

    return render_template(
        "recruiter.html",
        user=user,
        project_summary=system.project_summary(),
        default_job_description=DEFAULT_JOB,
        candidate_entries=sample_candidates(),
        recent_analyses=db.list_recent_analyses(user["id"], "recruiter"),
        feedback_profile=feedback_profile,
        parse_notes=[],
    )


@app.route("/recruiter/analyze", methods=["POST"])
def recruiter_analyze():
    user = require_role("recruiter")
    if not user:
        return redirect(url_for("login", user_role="recruiter"))

    job_description = request.form.get("job_description", "").strip()
    candidate_entries, parse_notes = candidate_entries_from_form()
    resume_batch = build_resume_batch(candidate_entries)
    feedback_profile = db.recruiter_feedback_profile(user["id"])
    ranking = (
        system.rank_resumes(job_description, resume_batch, feedback_signals=feedback_profile)
        if job_description and resume_batch
        else []
    )

    if ranking and job_description:
        save_recruiter_history(user["id"], job_description, ranking)

    return render_template(
        "recruiter.html",
        user=user,
        ranking=ranking,
        job_description=job_description,
        candidate_entries=candidate_entries or sample_candidates(),
        project_summary=system.project_summary(),
        default_job_description=job_description or DEFAULT_JOB,
        recent_analyses=db.list_recent_analyses(user["id"], "recruiter"),
        feedback_profile=feedback_profile,
        parse_notes=parse_notes,
    )


@app.route("/recruiter/feedback", methods=["POST"])
def recruiter_feedback():
    user = require_role("recruiter")
    if not user:
        return redirect(url_for("login", user_role="recruiter"))

    decision = request.form.get("decision", "").strip()
    if decision not in {"shortlist", "reject"}:
        return redirect(url_for("recruiter_dashboard"))

    candidate_name = request.form.get("candidate_name", "").strip() or "Candidate"
    predicted_role = request.form.get("predicted_role", "").strip() or "General Software"
    job_description = request.form.get("job_description", "").strip()

    matched_terms = [
        item.strip()
        for item in request.form.get("matched_job_terms", "").split(",")
        if item.strip()
    ]
    highlights = [
        item.strip()
        for item in request.form.get("highlights", "").split(",")
        if item.strip()
    ]

    db.save_recruiter_feedback(
        user_id=user["id"],
        candidate_name=candidate_name,
        predicted_role=predicted_role,
        decision=decision,
        job_description=job_description,
        evidence_payload={
            "matched_job_terms": matched_terms,
            "highlights": highlights,
        },
    )
    return redirect(url_for("recruiter_dashboard"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
