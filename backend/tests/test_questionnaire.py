"""Questionnaire scoring against the project specification table (which matches notebook Section 82)."""
import itertools

import pytest

from pipeline.questionnaire import RISK_MAX, RISK_QUESTIONS, general_advice, level_ranges, score_answers

SPEC = {  # specification table: question -> {answer: points}
    "brush_twice": {"yes": 0, "no": 2},
    "junk_sugary": {"rarely": 0, "once a day": 1, "several times a day": 3},
    "snack_night": {"no": 0, "sometimes": 1, "often": 2},
    "fluoride": {"yes": 0, "not sure": 1, "no": 1},
    "dental_visit": {"within 6 months": 0, "6-12 months ago": 1, "more than 1 year / never": 2},
    "pain": {"no": 0, "yes": 2},
    "tobacco": {"no": 0, "yes": 1},
}


def test_questions_match_spec():
    assert [q["key"] for q in RISK_QUESTIONS] == list(SPEC)
    for q in RISK_QUESTIONS:
        assert q["options"] == SPEC[q["key"]]
    assert RISK_MAX == 13
    assert level_ranges() == [("LOW", 0, 3), ("MODERATE", 4, 7), ("HIGH", 8, 13)]


def test_all_648_answer_combinations():
    keys = list(SPEC)
    n = 0
    for combo in itertools.product(*[list(SPEC[k]) for k in keys]):
        ans = dict(zip(keys, combo))
        r = score_answers(ans)
        expected = sum(SPEC[k][a] for k, a in ans.items())
        assert r["score"] == expected
        assert r["level"] == ("LOW" if expected <= 3 else "MODERATE" if expected <= 7 else "HIGH")
        assert r["points"] == {k: SPEC[k][a] for k, a in ans.items()}
        n += 1
    assert n == 648


def test_best_and_worst_case_and_advice():
    best = score_answers({k: next(a for a, p in v.items() if p == 0) for k, v in SPEC.items()})
    assert best["score"] == 0 and best["level"] == "LOW" and best["advice"] == []
    worst = score_answers({k: max(v, key=v.get) for k, v in SPEC.items()})
    assert worst["score"] == 13 and worst["level"] == "HIGH"
    assert "Brush twice a day (morning and before bed) for 2 minutes." in worst["advice"]
    assert len(worst["advice"]) == len(set(worst["advice"]))   # de-duplicated like the notebook


def test_notebook_example_run_score_9_high():
    # answers recorded in the notebook's Section 82.1 output: 2,2,2,2,2,2,2 -> HIGH 9/13
    r = score_answers({"brush_twice": "no", "junk_sugary": "once a day", "snack_night": "sometimes", "fluoride": "not sure",
                       "dental_visit": "6-12 months ago", "pain": "yes", "tobacco": "yes"})
    assert (r["score"], r["level"]) == (9, "HIGH")
    assert list(r["points"].values()) == [2, 1, 1, 1, 1, 2, 1]


def test_invalid_or_missing_answers_rejected():
    good = {k: list(v)[0] for k, v in SPEC.items()}
    with pytest.raises(ValueError):
        score_answers({**good, "pain": "maybe"})
    bad = dict(good); bad.pop("tobacco")
    with pytest.raises(ValueError):
        score_answers(bad)


def test_general_advice_adds_dentist_visit_for_caries():
    risk = score_answers({k: list(v)[0] for k, v in SPEC.items()})
    adv = general_advice({"stage1": {"prediction": "CARIES"}}, risk)
    assert adv[0].startswith("Book a dental examination")
    assert not general_advice({"stage1": {"prediction": "HEALTHY"}}, risk)[0].startswith("Book")
