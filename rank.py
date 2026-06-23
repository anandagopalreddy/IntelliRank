#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Ranking System
Senior AI Engineer (Founding Team) @ Redrob AI

Architecture:
  1. Multi-signal scoring with explicit JD alignment
  2. Honeypot detection (impossible profiles → hard disqualify)
  3. Availability/behavioral multiplier (signal layer)
  4. Weighted composite with transparent reasoning

Runs in ≤5 minutes on CPU, ≤16 GB RAM, no network.
"""

import json
import csv
import argparse
import sys
from datetime import date

# ─────────────────────────────────────────────
# JD SIGNAL DEFINITIONS
# ─────────────────────────────────────────────

STRONG_TITLE_KEYWORDS = [
    "ml engineer", "machine learning engineer", "ai engineer", "nlp engineer",
    "applied scientist", "applied ml", "research engineer", "data scientist",
    "search engineer", "ranking engineer", "recommendation", "recsys",
    "information retrieval", "ir engineer", "founding engineer",
    "staff engineer", "principal engineer",
]

ADJACENT_TITLE_KEYWORDS = [
    "software engineer", "backend engineer", "full stack", "platform engineer",
    "data engineer", "ml ops", "mlops", "devops engineer"
]

# Clear disqualifiers per JD ("title-chasers" + explicitly wrong domains)
DISQUALIFIER_TITLES = [
    "hr manager", "human resource", "marketing manager", "sales",
    "accountant", "finance", "mechanical engineer", "civil engineer",
    "electrical engineer", "content writer", "graphic designer",
    "business analyst", "operations manager",
    "customer support", "qa engineer", "quality assurance",
    "frontend engineer", "ui engineer", "ux engineer",
    "project manager", "product manager",
]

CONSULTING_COMPANIES = [
    "tata consultancy", "tcs", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "ltimindtree", "persistent systems", "mphasis",
    "birlasoft", "zensar", "cyient", "kpit"
]

# Core required skills — each bucket = one JD requirement
EMBEDDINGS_SKILLS = [
    "embeddings", "embedding", "sentence-transformers", "sentence transformers",
    "openai embeddings", "bge", "e5", "dense retrieval", "semantic search",
    "bi-encoder", "cross-encoder", "text embeddings", "ada embedding",
]

VECTOR_DB_SKILLS = [
    "vector database", "vector db", "pinecone", "weaviate", "qdrant",
    "milvus", "faiss", "elasticsearch", "opensearch", "chroma", "chromadb",
    "pgvector", "annoy", "scann", "vector search", "hybrid search",
    "approximate nearest neighbor", "ann", "hnsw",
]

RANKING_IR_SKILLS = [
    "information retrieval", "learning to rank", "ltr", "ranking", "reranking",
    "re-ranking", "bm25", "tf-idf", "ndcg", "mrr", "map", "recall@k",
    "retrieval augmented", "rag", "reciprocal rank",
]

LLM_NLP_SKILLS = [
    "llm", "large language model", "gpt", "llama", "mistral", "claude model",
    "transformers", "bert", "roberta", "fine-tuning", "fine tuning",
    "lora", "qlora", "peft", "instruction tuning", "nlp",
    "natural language processing", "text classification", "hugging face",
    "huggingface",
]

ML_FRAMEWORK_SKILLS = [
    "pytorch", "tensorflow", "jax", "scikit-learn", "sklearn",
    "xgboost", "lightgbm", "catboost",
]

EVAL_INFRA_SKILLS = [
    "mlops", "mlflow", "weights & biases", "wandb", "kubeflow",
    "a/b testing", "ab testing", "experimentation", "evaluation framework",
    "model serving", "triton", "bentoml", "ray serve",
]

PYTHON_SKILLS = ["python"]

ALL_CORE_BUCKETS = [
    ("embeddings", EMBEDDINGS_SKILLS),
    ("vector_db", VECTOR_DB_SKILLS),
    ("ranking_ir", RANKING_IR_SKILLS),
    ("llm_nlp", LLM_NLP_SKILLS),
    ("ml_frameworks", ML_FRAMEWORK_SKILLS),
    ("eval_infra", EVAL_INFRA_SKILLS),
    ("python", PYTHON_SKILLS),
]

BONUS_SKILLS = [
    "recommendation system", "recommender", "collaborative filtering",
    "distributed systems", "kafka", "spark", "flink",
    "kubernetes", "docker", "aws", "gcp", "azure",
    "open source", "research", "publications",
]

PREFERRED_LOCATIONS_IN = ["pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore", "bengaluru",
                           "ncr", "gurugram", "gurgaon", "chennai", "kolkata"]

# ─────────────────────────────────────────────
# HONEYPOT DETECTION
# ─────────────────────────────────────────────

def is_honeypot(cand: dict) -> tuple[bool, str]:
    """
    Detect impossible/synthetic honeypot profiles.
    Uses conservative thresholds to avoid false positives.
    """
    today = date.today()
    profile = cand.get("profile", {})
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])
    rs = cand.get("redrob_signals", {})
    edu = cand.get("education", [])

    yoe = profile.get("years_of_experience", 0)
    career_months = yoe * 12

    # 1. Expert skills with 0 duration — physically impossible
    expert_zero = [s for s in skills if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0]
    if len(expert_zero) >= 5:
        return True, f"{len(expert_zero)} expert skills with 0 months duration"

    # 2. YoE contradicts graduation year significantly
    if edu and yoe > 5:
        latest_grad = max((e.get("end_year", 0) for e in edu if e.get("end_year")), default=0)
        if latest_grad > 1990:
            implied_max_yoe = today.year - latest_grad
            if yoe > implied_max_yoe + 5:  # generous buffer
                return True, f"YoE {yoe} but graduated {latest_grad} (max plausible ~{implied_max_yoe})"

    # 3. Future start dates in career
    for job in career:
        start_str = job.get("start_date", "")
        if start_str:
            try:
                start = date.fromisoformat(start_str)
                if start > today:
                    return True, f"job start_date {start_str} is in the future"
            except ValueError:
                pass

    # 4. Sum of career durations wildly exceeds stated YoE (no overlaps possible beyond 2x)
    total_career_months = sum(j.get("duration_months", 0) for j in career)
    if career_months > 24 and total_career_months > career_months * 2.2:
        return True, f"career job durations sum {total_career_months}mo >> YoE {yoe}yr ({career_months:.0f}mo)"

    # 5. Brand new account with impossible number of expert skills
    signup = rs.get("signup_date", "")
    if signup:
        try:
            signup_date = date.fromisoformat(signup)
            days_on_platform = (today - signup_date).days
            expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
            if days_on_platform < 7 and len(expert_skills) >= 10:
                return True, f"account only {days_on_platform}d old with {len(expert_skills)} expert skills"
        except ValueError:
            pass

    return False, ""


# ─────────────────────────────────────────────
# SCORING COMPONENTS
# ─────────────────────────────────────────────

def score_title_role(cand: dict) -> float:
    """
    Title and career role alignment — primary anti-keyword-stuffer signal.
    Checks current title AND career history titles.
    """
    current_title = cand["profile"].get("current_title", "").lower()
    career = cand.get("career_history", [])
    career_titles = [j.get("title", "").lower() for j in career]
    all_titles = [current_title] + career_titles

    # Hard disqualifier: current title is clearly wrong function
    for bad in DISQUALIFIER_TITLES:
        if bad in current_title:
            # Check if they have solid AI/ML career history despite current title
            strong_past = sum(1 for t in career_titles if any(kw in t for kw in STRONG_TITLE_KEYWORDS))
            if strong_past >= 2:
                return 0.35  # redemption: temporarily in wrong role
            return 0.0

    score = 0.0

    # Strong ML/AI title in career history
    for kw in STRONG_TITLE_KEYWORDS:
        if any(kw in t for t in all_titles):
            score += 1.0
            break

    # Current title is strong (extra weight — it's the most recent)
    if any(kw in current_title for kw in STRONG_TITLE_KEYWORDS):
        score += 0.5

    # Adjacent title
    if score < 0.5:
        for kw in ADJACENT_TITLE_KEYWORDS:
            if any(kw in t for t in all_titles):
                score += 0.35
                break

    return min(score, 1.0)


def skill_name_matches(skill_name: str, keyword_list: list) -> bool:
    """Check if a skill name matches any keyword in the list."""
    sn = skill_name.lower()
    return any(kw in sn or sn in kw for kw in keyword_list)


def score_skills(cand: dict) -> tuple[float, list]:
    """
    Bucket-based skill scoring with trust multiplier.
    Covers all 7 JD-required skill buckets.
    Returns (score 0-1, matched_bucket_names).
    """
    skills = cand.get("skills", [])
    rs = cand.get("redrob_signals", {})
    assessment_scores = rs.get("skill_assessment_scores", {})

    # Build trusted skill set
    trusted_skills = {}
    for s in skills:
        name = s["name"].lower()
        prof = s.get("proficiency", "beginner")
        dur = s.get("duration_months", 0)
        end = s.get("endorsements", 0)

        # Base confidence from proficiency
        base = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.55, "beginner": 0.3}.get(prof, 0.3)

        # Duration trust — key anti-stuffing signal
        if dur == 0:
            base *= 0.35 if prof in ("expert", "advanced") else 0.6
        elif dur < 6:
            base *= 0.65
        elif dur >= 18:
            base *= 1.05
        elif dur >= 36:
            base *= 1.1

        # Endorsements
        if end >= 15:
            base *= 1.08
        elif end == 0 and prof in ("expert", "advanced"):
            base *= 0.85

        # Platform assessment override
        for assess_name, assess_score_val in assessment_scores.items():
            if assess_name.lower() in name or name in assess_name.lower():
                if assess_score_val >= 80:
                    base = max(base, 0.85)
                elif assess_score_val >= 60:
                    base = max(base, 0.65)
                break

        trusted_skills[name] = min(base, 1.0)

    # Score each required bucket — partial match allowed
    bucket_scores = {}
    matched_buckets = []

    for bucket_name, keyword_list in ALL_CORE_BUCKETS:
        best = 0.0
        for skill_name, trust in trusted_skills.items():
            if skill_name_matches(skill_name, keyword_list):
                best = max(best, trust)
        bucket_scores[bucket_name] = best
        if best >= 0.3:
            matched_buckets.append(bucket_name)

    # Also check career descriptions for implicit skill evidence
    career = cand.get("career_history", [])
    for job in career:
        desc = job.get("description", "").lower()
        for bucket_name, keyword_list in ALL_CORE_BUCKETS:
            if bucket_scores.get(bucket_name, 0) < 0.3:
                if any(kw in desc for kw in keyword_list):
                    bucket_scores[bucket_name] = max(bucket_scores.get(bucket_name, 0), 0.35)
                    if bucket_name not in matched_buckets:
                        matched_buckets.append(bucket_name)

    # Bonus skills
    bonus = 0.0
    for skill_name, trust in trusted_skills.items():
        if any(b in skill_name or skill_name in b for b in BONUS_SKILLS):
            bonus += trust * 0.15

    # Score: bucket coverage weighted
    bucket_weights = {
        "embeddings": 0.20, "vector_db": 0.20, "ranking_ir": 0.18,
        "llm_nlp": 0.15, "ml_frameworks": 0.12, "eval_infra": 0.10,
        "python": 0.05
    }
    weighted_score = sum(bucket_scores.get(b, 0) * w for b, w in bucket_weights.items())
    weighted_score += min(bonus, 0.15)

    return min(weighted_score, 1.0), matched_buckets


def score_experience(cand: dict) -> float:
    """
    Experience quality: YoE range, product vs services, production ML signals,
    career tenure (anti job-hopper).
    """
    profile = cand["profile"]
    career = cand.get("career_history", [])
    yoe = profile.get("years_of_experience", 0)

    # YoE scoring — sweet spot 5-9 per JD
    if 5 <= yoe <= 9:
        yoe_score = 1.0
    elif 4 <= yoe < 5 or 9 < yoe <= 11:
        yoe_score = 0.8
    elif 3 <= yoe < 4 or 11 < yoe <= 14:
        yoe_score = 0.55
    elif 2 <= yoe < 3:
        yoe_score = 0.3
    elif yoe < 2:
        yoe_score = 0.1
    else:  # >14: experienced but JD wants someone hands-on
        yoe_score = 0.45

    # Product vs services company
    product_months = 0
    services_months = 0
    total_months = 0

    for job in career:
        industry = job.get("industry", "").lower()
        company = job.get("company", "").lower()
        dur = job.get("duration_months", 1)
        total_months += dur

        is_consulting = any(c in company for c in CONSULTING_COMPANIES)
        is_services_ind = any(s in industry for s in ["it services", "consulting", "outsourcing", "bpo"])

        if is_consulting or is_services_ind:
            services_months += dur
        else:
            product_months += dur

    if total_months > 0:
        product_ratio = product_months / total_months
    else:
        product_ratio = 0.5

    # JD: "entire career at consulting = hard disqualifier"
    if services_months > 0 and product_months == 0:
        product_ratio = 0.05

    # Production ML evidence in descriptions
    prod_ml_score = 0.0
    for job in career:
        desc = job.get("description", "").lower()
        prod_keywords = [
            "deployed", "production", "serving", "real users", "scale",
            "ranking", "retrieval", "recommendation", "search system",
            "embedding pipeline", "vector index", "latency",
            "a/b test", "evaluation framework", "inference"
        ]
        matches = sum(1 for kw in prod_keywords if kw in desc)
        if matches >= 4:
            prod_ml_score = min(prod_ml_score + 0.5, 1.0)
        elif matches >= 2:
            prod_ml_score = min(prod_ml_score + 0.25, 1.0)

    # Career tenure (title-hopper penalty)
    if career:
        tenures = [j.get("duration_months", 0) for j in career]
        avg_tenure = sum(tenures) / len(tenures)
        if avg_tenure >= 24:
            tenure_score = 1.0
        elif avg_tenure >= 18:
            tenure_score = 0.85
        elif avg_tenure >= 12:
            tenure_score = 0.65
        else:
            tenure_score = 0.35
    else:
        tenure_score = 0.5

    exp_score = (
        yoe_score * 0.30 +
        product_ratio * 0.30 +
        prod_ml_score * 0.25 +
        tenure_score * 0.15
    )
    return min(exp_score, 1.0)


def score_availability(cand: dict) -> float:
    """
    Behavioral availability composite.
    Per JD: 'inactive for 6 months = not actually available — downweight'.
    """
    rs = cand.get("redrob_signals", {})
    today = date.today()

    # Recency of last activity
    last_active_str = rs.get("last_active_date", "")
    days_inactive = 365  # default pessimistic
    if last_active_str:
        try:
            last_active = date.fromisoformat(last_active_str)
            days_inactive = (today - last_active).days
        except ValueError:
            pass

    if days_inactive <= 7:
        recency = 1.0
    elif days_inactive <= 30:
        recency = 0.9
    elif days_inactive <= 60:
        recency = 0.75
    elif days_inactive <= 90:
        recency = 0.6
    elif days_inactive <= 180:
        recency = 0.45
    else:
        recency = 0.2

    # Open to work
    otw = 1.0 if rs.get("open_to_work_flag", False) else 0.65

    # Recruiter response rate
    rr = rs.get("recruiter_response_rate", 0.5)
    response_score = 0.25 + 0.75 * rr

    # Notice period (JD: sub-30d preferred, can buy out 30d)
    notice = rs.get("notice_period_days", 60)
    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.85
    elif notice <= 90:
        notice_score = 0.7
    elif notice <= 120:
        notice_score = 0.55
    else:
        notice_score = 0.35

    # Interview completion rate (shows up when committed)
    icr = rs.get("interview_completion_rate", 0.75)
    icr_score = 0.4 + 0.6 * icr

    # Profile completeness
    completeness = rs.get("profile_completeness_score", 70) / 100.0

    avail = (
        recency * 0.30 +
        otw * 0.22 +
        response_score * 0.22 +
        notice_score * 0.15 +
        icr_score * 0.07 +
        completeness * 0.04
    )
    return min(avail, 1.0)


def score_location(cand: dict) -> float:
    """Location fit: India-based preferred, Pune/Noida/metros strongly preferred."""
    profile = cand["profile"]
    rs = cand.get("redrob_signals", {})
    country = profile.get("country", "").lower()
    location = profile.get("location", "").lower()
    willing_relocate = rs.get("willing_to_relocate", False)

    if country == "india":
        if any(loc in location for loc in PREFERRED_LOCATIONS_IN):
            return 1.0
        elif willing_relocate:
            return 0.88
        else:
            return 0.72
    elif country in ["usa", "uk", "germany", "australia", "singapore", "uae", "canada"]:
        return 0.2  # case-by-case, no visa sponsorship
    else:
        return 0.25


def score_education(cand: dict) -> float:
    """Education quality — tertiary signal."""
    edu = cand.get("education", [])
    if not edu:
        return 0.45

    tier_map = {"tier_1": 1.0, "tier_2": 0.82, "tier_3": 0.62, "tier_4": 0.42, "unknown": 0.52}
    degree_map = {
        "phd": 1.0, "ph.d": 1.0, "doctorate": 1.0,
        "m.tech": 0.92, "mtech": 0.92, "m.e.": 0.92,
        "m.s.": 0.88, "ms": 0.88, "m.sc": 0.85,
        "mba": 0.72, "pgdm": 0.70,
        "b.tech": 0.72, "btech": 0.72, "b.e.": 0.72,
        "b.s.": 0.68, "bs": 0.68,
        "b.sc": 0.60, "bsc": 0.60,
    }
    relevant_fields = [
        "computer science", "cs", "information technology", "it",
        "data science", "machine learning", "artificial intelligence",
        "mathematics", "statistics", "electrical", "electronics", "instrumentation"
    ]

    best = 0.0
    for e in edu:
        tier_score = tier_map.get(e.get("tier", "unknown"), 0.52)
        degree = e.get("degree", "").lower()
        field = e.get("field_of_study", "").lower()

        deg_score = 0.62
        for dkw, dscore in degree_map.items():
            if dkw in degree:
                deg_score = dscore
                break

        field_score = 1.0 if any(f in field for f in relevant_fields) else 0.45

        s = tier_score * 0.40 + deg_score * 0.35 + field_score * 0.25
        best = max(best, s)

    return min(best, 1.0)


def score_engagement(cand: dict) -> float:
    """Platform engagement quality signals."""
    rs = cand.get("redrob_signals", {})

    github = rs.get("github_activity_score", -1)
    github_score = (github / 100.0) if github >= 0 else 0.35

    saved = rs.get("saved_by_recruiters_30d", 0)
    saved_score = min(saved / 5.0, 1.0)

    verified = (
        0.40 * int(rs.get("verified_email", False)) +
        0.35 * int(rs.get("verified_phone", False)) +
        0.25 * int(rs.get("linkedin_connected", False))
    )

    views = rs.get("profile_views_received_30d", 0)
    views_score = min(views / 15.0, 1.0)

    return min(
        github_score * 0.35 +
        saved_score * 0.25 +
        verified * 0.25 +
        views_score * 0.15,
        1.0
    )


# ─────────────────────────────────────────────
# COMPOSITE SCORING
# ─────────────────────────────────────────────

WEIGHTS = {
    "title":        0.26,
    "skills":       0.26,
    "experience":   0.24,
    "availability": 0.14,
    "location":     0.05,
    "education":    0.03,
    "engagement":   0.02,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 0.001


def score_candidate(cand: dict) -> tuple[float, dict, str]:
    """Returns (composite_score, components, reasoning). Score=-1 means honeypot."""
    honeypot, hp_reason = is_honeypot(cand)
    if honeypot:
        return -1.0, {}, f"HONEYPOT: {hp_reason}"

    title_score = score_title_role(cand)
    skill_score, matched_buckets = score_skills(cand)
    exp_score = score_experience(cand)
    avail_score = score_availability(cand)
    loc_score = score_location(cand)
    edu_score = score_education(cand)
    eng_score = score_engagement(cand)

    # Title=0 means completely wrong function; still produce a score but very low
    if title_score == 0.0:
        composite = (skill_score * 0.10 + exp_score * 0.05 + avail_score * 0.05)
    else:
        composite = (
            title_score * WEIGHTS["title"] +
            skill_score * WEIGHTS["skills"] +
            exp_score * WEIGHTS["experience"] +
            avail_score * WEIGHTS["availability"] +
            loc_score * WEIGHTS["location"] +
            edu_score * WEIGHTS["education"] +
            eng_score * WEIGHTS["engagement"]
        )

    components = {
        "title": title_score, "skills": skill_score, "experience": exp_score,
        "availability": avail_score, "location": loc_score,
        "education": edu_score, "engagement": eng_score,
    }

    reasoning = build_reasoning(cand, components, matched_buckets, composite)
    return composite, components, reasoning


def build_reasoning(cand: dict, components: dict, matched_buckets: list, score: float) -> str:
    """Stage 4-compliant reasoning: specific, honest, JD-connected."""
    today = date.today()
    profile = cand["profile"]
    rs = cand.get("redrob_signals", {})
    career = cand.get("career_history", [])

    title = profile.get("current_title", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "Unknown")
    country = profile.get("country", "Unknown")

    last_active_str = rs.get("last_active_date", "")
    try:
        days_inactive = (today - date.fromisoformat(last_active_str)).days if last_active_str else 999
    except ValueError:
        days_inactive = 999

    notice = rs.get("notice_period_days", 60)
    rr = rs.get("recruiter_response_rate", 0.5)
    otw = rs.get("open_to_work_flag", False)

    parts = []

    # Title + YoE
    if components["title"] >= 0.8:
        parts.append(f"{title}, {yoe:.1f}yr exp")
    elif components["title"] >= 0.35:
        parts.append(f"{title} ({yoe:.1f}yr, adjacent role)")
    else:
        parts.append(f"{title} ({yoe:.1f}yr, mismatched function)")

    # Skills buckets covered
    if matched_buckets:
        nice_names = {
            "embeddings": "embeddings", "vector_db": "vector DB",
            "ranking_ir": "ranking/IR", "llm_nlp": "LLM/NLP",
            "ml_frameworks": "ML frameworks", "eval_infra": "MLOps/eval",
            "python": "Python"
        }
        bk_str = ", ".join(nice_names.get(b, b) for b in matched_buckets[:4])
        parts.append(f"covers {len(matched_buckets)}/7 JD skill buckets ({bk_str})")
    else:
        parts.append("no JD skill bucket overlap")

    # Location
    loc_display = location if country == "India" else f"{location}, {country}"
    if components["location"] >= 0.95:
        parts.append(f"India/{location}")
    elif components["location"] >= 0.75:
        parts.append(f"willing to relocate from {loc_display}")
    else:
        parts.append(f"non-India ({loc_display})")

    # Availability
    if days_inactive <= 14 and otw:
        parts.append("actively searching, recently active")
    elif days_inactive > 180:
        parts.append(f"inactive {days_inactive}d — availability uncertain")
    if rr < 0.25:
        parts.append(f"low response rate ({rr:.0%})")
    if notice > 90:
        parts.append(f"{notice}d notice period")
    elif notice <= 30 and score > 0.4:
        parts.append(f"short notice ({notice}d)")

    return "; ".join(parts)[:280]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def load_candidates(path: str) -> list:
    import gzip
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt"
    with opener(path, mode, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rank_candidates(candidates: list) -> list:
    scored = []
    honeypot_count = 0
    for cand in candidates:
        composite, components, reasoning = score_candidate(cand)
        if composite < 0:
            honeypot_count += 1
            continue
        scored.append({
            "candidate_id": cand["candidate_id"],
            "score": composite,
            "reasoning": reasoning,
        })
    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    print(f"  Scored {len(scored)} | Honeypots removed: {honeypot_count}", file=sys.stderr)
    return scored[:100]


def write_submission(ranked: list, out_path: str):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, cand in enumerate(ranked):
            score = round(cand["score"], 6)
            reasoning = cand["reasoning"].replace('"', "'")
            writer.writerow([cand["candidate_id"], i + 1, score, reasoning])


def main():
    parser = argparse.ArgumentParser(description="Redrob AI Candidate Ranker")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(f"Loading {args.candidates}...", file=sys.stderr)
    candidates = load_candidates(args.candidates)
    print(f"  {len(candidates)} candidates loaded", file=sys.stderr)

    print("Ranking...", file=sys.stderr)
    ranked = rank_candidates(candidates)

    write_submission(ranked, args.out)
    print(f"\nTop 10:", file=sys.stderr)
    for i, r in enumerate(ranked[:10]):
        print(f"  #{i+1} {r['candidate_id']} {r['score']:.4f} | {r['reasoning'][:90]}", file=sys.stderr)
    print(f"\nSubmission written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
