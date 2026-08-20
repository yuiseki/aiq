"""stdout must stay parseable even when the environment forces colour.

Agents and CI runners frequently set FORCE_COLOR, and `rich` obeys it even
when stdout is a pipe. Every command that emits records must therefore write
plain bytes to stdout, while human-facing progress stays on stderr.
"""
import json
import os
import pathlib

import pytest

from conftest import REPO_ROOT, run_python

SRC = pathlib.Path(REPO_ROOT) / "src" / "aiq"

FAKE_LABEL = r'''
import sys, asyncio
import aiq.label as L

class _Msg:
    def __init__(self, c): self.content = c
class _Choice:
    def __init__(self, c): self.message = _Msg(c)
class _Resp:
    def __init__(self, c): self.choices = [_Choice(c)]
class _Completions:
    async def create(self, **kw): return _Resp("ニュース")
class _Chat:
    completions = _Completions()
class FakeClient:
    def __init__(self, *a, **k): self.chat = _Chat()

L.AsyncOpenAI = FakeClient
L.label(
    input_type="json",
    input_field="text",
    label_options=["ニュース", "音楽"],
    output_field="label",
    model="fake",
)
'''

FAKE_EMBED = r'''
import aiq.embed as E

class FakeModel:
    def encode(self, texts, show_progress=False):
        import numpy as np
        return [np.array([0.1, 0.2, 0.3]) for _ in texts]
class FakeRegistry:
    @staticmethod
    def from_registry(name): return FakeModel()

E.EmbeddingModel = FakeRegistry
E.embed(input_type="json", input_field="text")
'''


def _assert_clean_jsonl(stdout: str, expected_lines: int):
    assert "\x1b" not in stdout, f"ANSI escape leaked into stdout: {stdout!r}"
    lines = [l for l in stdout.splitlines() if l.strip()]
    assert len(lines) == expected_lines
    for line in lines:
        json.loads(line)  # must not raise
    return [json.loads(l) for l in lines]


def test_label_stdout_has_no_ansi(ansi_env):
    stdin = "\n".join(
        json.dumps({"seg": f"s{i}", "text": f"本文{i}", "gold": "音楽"})
        for i in range(3)
    )
    r = run_python(FAKE_LABEL, ansi_env, stdin=stdin + "\n")
    assert r.returncode == 0, r.stderr
    records = _assert_clean_jsonl(r.stdout, 3)
    # input fields must survive
    assert all(rec["gold"] == "音楽" and rec["label"] == "ニュース" for rec in records)
    assert {rec["seg"] for rec in records} == {"s0", "s1", "s2"}


def test_embed_stdout_has_no_ansi(ansi_env):
    stdin = "\n".join(
        json.dumps({"seg": f"s{i}", "text": f"本文{i}", "gold": "音楽"})
        for i in range(3)
    )
    r = run_python(FAKE_EMBED, ansi_env, stdin=stdin + "\n")
    assert r.returncode == 0, r.stderr
    records = _assert_clean_jsonl(r.stdout, 3)
    assert all(rec["embedding"] == [0.1, 0.2, 0.3] for rec in records)
    assert all(rec["gold"] == "音楽" for rec in records)


def test_classify_stdout_has_no_ansi(ansi_env, tmp_path):
    """train (offline) -> classify (offline), both with FORCE_COLOR set."""
    import random

    random.seed(0)
    rows = []
    for i in range(120):
        cls = i % 2
        vec = [random.gauss(cls, 0.1) for _ in range(8)]
        rows.append({"seg": f"s{i}", "gold": "音楽", "label": f"c{cls}", "embedding": vec})
    stdin = "\n".join(json.dumps(r) for r in rows) + "\n"

    model_path = tmp_path / "model.joblib"
    train_code = (
        "import aiq.train as T; T.train(model_path=%r, n_classes=2, batch_size=16, timeout=1.0)"
        % str(model_path)
    )
    r = run_python(train_code, ansi_env, stdin=stdin)
    assert r.returncode == 0, r.stderr
    assert model_path.exists()

    classify_code = (
        "import aiq.classify as C; C.classify(model_path=%r, no_warn=True)" % str(model_path)
    )
    r = run_python(classify_code, ansi_env, stdin=stdin, timeout=180)
    assert r.returncode == 0, r.stderr
    records = _assert_clean_jsonl(r.stdout, 120)
    assert all("embedding" not in rec for rec in records)
    assert all(rec["gold"] == "音楽" for rec in records)


def test_no_module_writes_rich_to_stdout():
    """Guard against reintroducing a stdout Console."""
    for path in SRC.glob("*.py"):
        text = path.read_text()
        assert "Console(file=sys.stdout)" not in text, path
        # no hard-coded global state file used as a path literal
        assert '"/tmp/aiq' not in text, path
        assert "'/tmp/aiq" not in text, path


def test_early_pipe_close_is_not_an_error(ansi_env, tmp_path):
    """`aiq ... | head -N` must not produce a traceback."""
    import json as _json
    import random
    import subprocess
    import sys as _sys

    random.seed(1)
    rows = [
        {"seg": f"s{i}", "gold": "音楽", "label": f"c{i % 2}",
         "embedding": [random.gauss(i % 2, 0.1) for _ in range(8)]}
        for i in range(120)
    ]
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
    model = tmp_path / "m.joblib"

    r = run_python(
        "import aiq.train as T; T.train(model_path=%r, n_classes=2, batch_size=16,"
        " timeout=1.0, test_size=0.25)" % str(model),
        ansi_env, stdin=data.read_text(),
    )
    assert r.returncode == 0, r.stderr

    proc = subprocess.run(
        ["bash", "-c",
         f'cat "{data}" | "{_sys.executable}" -m aiq.aiq classify '
         f'--model-path "{model}" --no-warn 2>/dev/null | head -3'],
        env=ansi_env, capture_output=True, text=True, timeout=300, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0
    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    assert len(lines) == 3
    for line in lines:
        json.loads(line)
    assert "Traceback" not in proc.stderr
