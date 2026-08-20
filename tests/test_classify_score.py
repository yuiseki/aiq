"""`aiq classify` must be able to emit a score, not just a hard label.

A cascade ("screen cheaply, then send only the uncertain records to an LLM")
is expressible as a pipeline only if the threshold decision can be made by
the next command. That needs a number in the record:

    aiq embed | aiq classify --score-field score | jq 'select(.score>=0.05)'

With only `predict`, the cutoff would have to be hard-coded inside a
bespoke script, which is exactly what we are trying to avoid.
"""
import json
import pathlib

import pytest

from conftest import REPO_ROOT, run_python


def _write_model(path, kind="proba"):
    """A 2-class model over 4-dim vectors, stored in aiq's model layout."""
    import random

    import joblib
    from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier

    random.seed(0)
    X, y = [], []
    for i in range(80):
        cls = i % 2
        X.append([random.gauss(cls, 0.1) for _ in range(4)])
        y.append(cls)
    model = LogisticRegression() if kind == "proba" else PassiveAggressiveClassifier()
    model.fit(X, y)
    joblib.dump({"model": model, "class_dict": {"c0": 0, "c1": 1}}, path)


def _rows(n=6):
    import random

    random.seed(1)
    out = []
    for i in range(n):
        cls = i % 2
        out.append({"seg": f"s{i}",
                    "embedding": [random.gauss(cls, 0.1) for _ in range(4)]})
    return "\n".join(json.dumps(r) for r in out) + "\n"


def _run(env, model, extra=""):
    code = ("import aiq.classify as C; C.classify(model_path=%r, no_warn=True%s)"
            % (str(model), extra))
    return run_python(code, env, stdin=_rows(), timeout=180)


def test_no_score_field_by_default(ansi_env, tmp_path):
    """Existing behaviour must not change when the option is unused."""
    m = tmp_path / "m.joblib"
    _write_model(m)
    r = _run(ansi_env, m)
    assert r.returncode == 0, r.stderr
    recs = [json.loads(l) for l in r.stdout.splitlines() if l.strip()]
    assert len(recs) == 6
    assert all("score" not in rec for rec in recs)
    assert all(rec["label"] in ("c0", "c1") for rec in recs)


def test_score_field_is_probability_of_predicted_label(ansi_env, tmp_path):
    m = tmp_path / "m.joblib"
    _write_model(m)
    r = _run(ansi_env, m, extra=", score_field='score'")
    assert r.returncode == 0, r.stderr
    recs = [json.loads(l) for l in r.stdout.splitlines() if l.strip()]
    assert len(recs) == 6
    for rec in recs:
        assert isinstance(rec["score"], float)
        # the predicted label is the argmax, so its probability is >= 0.5
        assert 0.5 <= rec["score"] <= 1.0


def test_score_label_pins_which_class_is_reported(ansi_env, tmp_path):
    """The cascade needs P(is_disaster), not P(whatever won)."""
    m = tmp_path / "m.joblib"
    _write_model(m)
    a = _run(ansi_env, m, extra=", score_field='score', score_label='c1'")
    b = _run(ansi_env, m, extra=", score_field='score', score_label='c0'")
    assert a.returncode == 0, a.stderr
    assert b.returncode == 0, b.stderr
    pa = [json.loads(l)["score"] for l in a.stdout.splitlines() if l.strip()]
    pb = [json.loads(l)["score"] for l in b.stdout.splitlines() if l.strip()]
    assert len(pa) == len(pb) == 6
    for x, y in zip(pa, pb):
        assert x + y == pytest.approx(1.0, abs=1e-6)
    # a pinned class must produce values on both sides of 0.5 for mixed input
    assert min(pa) < 0.5 < max(pa)


def test_unknown_score_label_fails_loudly(ansi_env, tmp_path):
    m = tmp_path / "m.joblib"
    _write_model(m)
    r = _run(ansi_env, m, extra=", score_field='score', score_label='nope'")
    assert r.returncode != 0
    assert "nope" in r.stderr


def test_margin_model_is_accepted_but_flagged(ansi_env, tmp_path):
    """PassiveAggressiveClassifier (what `aiq train` builds) has no
    predict_proba. Emitting its decision_function is still useful, but the
    value is an unbounded margin, so a 0..1 threshold would be wrong and the
    difference has to be visible on stderr."""
    m = tmp_path / "m.joblib"
    _write_model(m, kind="margin")
    r = _run(ansi_env, m, extra=", score_field='score'")
    assert r.returncode == 0, r.stderr
    recs = [json.loads(l) for l in r.stdout.splitlines() if l.strip()]
    assert len(recs) == 6
    assert all(isinstance(rec["score"], float) for rec in recs)
    assert "decision_function" in r.stderr
