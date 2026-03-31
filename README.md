# Intelligent Resume Ranking and Job Recommendation System

This project is an NLP-based recruitment assistant built around the topic described in the provided review material: a single system that helps both job seekers and recruiters.

It performs two major tasks:

- Candidate-side job role recommendation from resume text
- Recruiter-side ranking of multiple resumes against a target job description

## Core NLP Pipeline

The application combines lightweight but explainable NLP methods:

- Text normalization and token filtering
- Keyword and keyphrase extraction
- TF-IDF vectorization
- Logistic regression for role prediction
- Cosine similarity for resume-to-job ranking
- Resume completeness scoring based on common section detection

## Features

- Predicts the most suitable job role from a single resume
- Produces confidence and overall fit indicators
- Shows matched and missing role keywords for explainability
- Extracts important keywords and phrases from resume text
- Ranks multiple candidates for a given job description
- Presents a clean Flask web interface for demo and submission use

## Project Structure

- `app.py`: Flask application entry point
- `nlp_project/preprocessing.py`: cleaning, phrase extraction, overlap scoring
- `nlp_project/recommender.py`: role prediction, suitability scoring, ranking logic
- `data/role_profiles.json`: role knowledge base
- `data/training_samples.json`: supervised examples for classifier training
- `data/sample_resumes.txt`: sample recruiter-side candidate set
- `templates/index.html`: frontend template
- `static/styles.css`: frontend styles

## How It Works

### 1. Resume preprocessing

The resume text is normalized to lowercase, cleaned, tokenized, and scanned for informative terms and resume sections such as skills, projects, education, and experience.

### 2. Role prediction

The system uses TF-IDF features with logistic regression to estimate the most likely role for the candidate. Role confidence is blended with role-profile similarity to produce a more stable recommendation score.

### 3. Resume explainability

For the predicted role, the system shows:

- matched keywords
- missing keywords
- extracted phrases
- completeness score

### 4. Candidate ranking

When a recruiter provides a job description and multiple resumes, the system computes TF-IDF cosine similarity between the job and each candidate. That score is blended with candidate suitability confidence to generate the final ranking.

## Run Locally

1. Create a Python virtual environment
2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Start the Flask app

```bash
python app.py
```

4. Open `http://127.0.0.1:5000`

## Suggested Submission Points

- Problem statement: automate resume screening and job recommendation
- NLP techniques: preprocessing, TF-IDF, logistic regression, cosine similarity
- Explainability: matched keywords, missing skills, phrase extraction
- Recruiter utility: ranked shortlist against a job description
- Candidate utility: role recommendation and suitability feedback
- Future scope: PDF parsing, semantic embeddings, larger labeled datasets, feedback-driven ranking

## Note On Source Alignment

The implementation was aligned mainly from the review document and the existing workspace scaffold. The other referenced PDFs could not be reliably extracted in this environment, so the code reflects the confirmed project theme rather than page-by-page reproduction of those files.
