"""
Oral-habits questionnaire — questions, answer options, points, risk levels and advice copied unchanged from notebook
Section 82 (`RISK_QUESTIONS`, `RISK_LEVELS`, `ask_oral_habits`, `general_advice`).

The habit-based risk score is kept SEPARATE from the image analysis: it never changes a CNN output and it is not a
diagnosis.
"""
RISK_QUESTIONS = [
    {"key": "brush_twice", "text": "Do you brush your teeth twice a day?", "options": {"yes": 0, "no": 2},
     "advice": {"no": "Brush twice a day (morning and before bed) for 2 minutes."}},
    {"key": "junk_sugary", "text": "How often do you eat junk food or sugary foods / drinks?",
     "options": {"rarely": 0, "once a day": 1, "several times a day": 3},
     "advice": {"once a day": "Reduce junk and sugary food; prefer it with meals.",
                "several times a day": "Cut down junk / sugary food and drinks — frequent sugar intake is a major caries risk."}},
    {"key": "snack_night", "text": "Do you eat or drink something sugary between meals or after brushing at night?",
     "options": {"no": 0, "sometimes": 1, "often": 2},
     "advice": {"sometimes": "Avoid sugary snacks between meals and after night brushing.",
                "often": "Avoid sugary snacks between meals and after night brushing."}},
    {"key": "fluoride", "text": "Do you use a fluoride toothpaste?", "options": {"yes": 0, "not sure": 1, "no": 1},
     "advice": {"not sure": "Use a fluoride toothpaste.", "no": "Use a fluoride toothpaste."}},
    {"key": "dental_visit", "text": "When did you last visit a dentist?",
     "options": {"within 6 months": 0, "6-12 months ago": 1, "more than 1 year / never": 2},
     "advice": {"6-12 months ago": "Plan a routine dental check-up.",
                "more than 1 year / never": "Visit a dentist for a check-up (recommended every 6–12 months)."}},
    {"key": "pain", "text": "Do you have tooth pain, or sensitivity to sweet / hot / cold?", "options": {"no": 0, "yes": 2},
     "advice": {"yes": "Pain or sensitivity should be checked by a dentist soon."}},
    {"key": "tobacco", "text": "Do you smoke or chew tobacco?", "options": {"no": 0, "yes": 1},
     "advice": {"yes": "Stopping tobacco use improves oral health."}},
]
RISK_MAX = sum(max(q["options"].values()) for q in RISK_QUESTIONS)
RISK_LEVELS = [(3, "LOW"), (7, "MODERATE"), (RISK_MAX, "HIGH")]
assert RISK_MAX == 13

QUESTION_TOPICS = {"brush_twice": "Toothbrushing", "junk_sugary": "Sugary food / drinks", "snack_night": "Sugar between meals / at night",
                   "fluoride": "Fluoride toothpaste", "dental_visit": "Last dental visit", "pain": "Pain / sensitivity",
                   "tobacco": "Tobacco use"}


def level_ranges():
    """[('LOW', 0, 3), ('MODERATE', 4, 7), ('HIGH', 8, 13)] — derived from RISK_LEVELS."""
    out, lo = [], 0
    for limit, name in RISK_LEVELS:
        out.append((name, lo, limit)); lo = limit + 1
    return out


def score_answers(chosen):
    """chosen = {question key: option text (lower-case, exactly as in RISK_QUESTIONS)} -> risk dict (notebook ask_oral_habits)."""
    missing = [q["key"] for q in RISK_QUESTIONS if chosen.get(q["key"]) not in q["options"]]
    if missing:
        raise ValueError(f"unanswered or invalid questions: {missing}")
    chosen = {q["key"]: chosen[q["key"]] for q in RISK_QUESTIONS}
    score = int(sum(q["options"][chosen[q["key"]]] for q in RISK_QUESTIONS))
    level = next(name for limit, name in RISK_LEVELS if score <= limit)
    advice = list(dict.fromkeys(q["advice"][chosen[q["key"]]] for q in RISK_QUESTIONS if chosen[q["key"]] in q.get("advice", {})))
    points = {q["key"]: q["options"][chosen[q["key"]]] for q in RISK_QUESTIONS}
    return {"answers": chosen, "points": points, "score": score, "max": RISK_MAX, "level": level, "advice": advice}


def general_advice(res, risk):
    """Questionnaire advice + general advice (notebook general_advice)."""
    adv = list(risk["advice"]) if risk else []
    s1 = ((res or {}).get("stage1") or {}).get("prediction")
    if s1 in ("CARIES", "UNCERTAIN") or (res or {}).get("overall") == "INVALID INPUT":
        adv.insert(0, "Book a dental examination to confirm or rule out the image findings.")
    adv += ["Brush twice daily with a fluoride toothpaste and limit sugary snacks and drinks.",
            "Regular dental check-ups (every 6–12 months) catch caries early."]
    return list(dict.fromkeys(adv))
