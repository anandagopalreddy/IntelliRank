# Redrob Hackathon — Intelligent Candidate Ranking System

**Challenge:** Intelligent Candidate Discovery & Ranking  
**JD:** Senior AI Engineer (Founding Team) @ Redrob AI

## Reproduce the submission

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

- Runtime: ~40–50 seconds on a modern CPU (100K candidates)
- Memory: <2 GB peak
- No GPU, no network, no external dependencies

## Architecture

### Core philosophy
Keyword-matching fails this JD because the JD explicitly warns against it. The system reasons about **what the JD means**, not just what it says.

### Scoring components (7 signals, 100% explainable)

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Title/Role | 26% | Current title + career titles — primary anti-keyword-stuffer gate |
| Skills | 26% | 7 JD-required bucket coverage with trust multiplier (endorsements × duration) |
| Experience | 24% | YoE in 5–9yr sweet spot, product company %, production ML evidence in descriptions |
| Availability | 14% | last_active_date recency, open_to_work, recruiter_response_rate, notice_period |
| Location | 5% | India-based + Pune/Noida/metro preferred |
| Education | 3% | Tier × degree level × field relevance |
| Engagement | 2% | GitHub activity, saved_by_recruiters, verification, profile views |

### Anti-keyword-stuffer design
The title component (26%) acts as a gating signal. A "Marketing Manager" with 15 AI skill keywords scores near 0 despite the keywords. Skills scoring includes a **trust multiplier**: `expert` with `duration_months=0` is penalized 65% — you cannot claim expertise in something you have 0 months of use in.

### Honeypot detection (4 checks)
1. **Expert skills with 0 duration** (≥5 instances → honeypot)
2. **YoE contradicts graduation year** (>5yr buffer)
3. **Future start dates** in career history
4. **Career duration sum > 2.2× stated YoE** (accounting for legitimate overlap)

Result: **7,585 honeypots removed** from 100K pool (~7.6%, well under 10% disqualification threshold).

### Availability modifier
Per JD note: "a perfect-on-paper candidate who hasn't logged in for 6 months... is not actually available." `last_active_date` recency carries 30% of the availability sub-score. A candidate inactive for 180+ days gets a 0.2 recency multiplier regardless of other signals.

### Skill trust scoring
```
trust = proficiency_base × duration_factor × endorsement_factor × assessment_factor

proficiency_base: expert=1.0, advanced=0.8, intermediate=0.55, beginner=0.3
duration_factor: 0mo=0.35 (if expert/advanced), <6mo=0.65, ≥18mo=1.05, ≥36mo=1.10
endorsement_factor: 0 endorsements with high prof = ×0.85, ≥15 endorsements = ×1.08
```

## Validation

```bash
python validate_submission.py submission.csv
# → Submission is valid.
```

## Files

```
rank.py                    Main ranker (single file, stdlib only)
submission.csv             Output: top 100 ranked candidates
requirements.txt           Dependencies (none external)
README.md                  This file
submission_metadata.yaml   Competition metadata
```
