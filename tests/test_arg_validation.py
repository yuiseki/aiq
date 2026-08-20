"""Invalid values for Literal-typed arguments must fail loudly.

`fire` does not check annotations, so `--input-type jsonl` used to be
silently treated as the default ("text"), which sent whole JSONL lines to
the LLM and dropped every other field of the record. Silent
behaviour changes are the worst possible failure mode for an agent.
"""
import json
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

AIQ = [sys.executable, "-m", "aiq.aiq"]


def run_aiq(args, stdin="", env_extra=None, timeout=120):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(REPO_ROOT, "src")
    env.setdefault("OPENAI_API_KEY", "dummy")
    env.update(env_extra or {})
    return subprocess.run(
        AIQ + args, input=stdin, env=env, capture_output=True, text=True,
        timeout=timeout, cwd=REPO_ROOT,
    )


@pytest.mark.parametrize("bad", ["jsonl", "JSON", "", "csv"])
def test_label_rejects_bad_input_type(bad):
    r = run_aiq(["label", "--input-type", bad, "--label-options", "[a,b]"])
    assert r.returncode != 0
    assert "Invalid value for --input-type" in (r.stderr + r.stdout)
    assert r.stdout.strip() == "" or "Invalid value" in r.stdout


@pytest.mark.parametrize("bad", ["jsonl", "TEXT", "csv"])
def test_embed_rejects_bad_input_type(bad):
    r = run_aiq(["embed", "--input-type", bad])
    assert r.returncode != 0
    assert "Invalid value for --input-type" in (r.stderr + r.stdout)


def test_validate_choice_accepts_valid_values():
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
    from aiq.common import validate_choice

    assert validate_choice("input_type", "json", ["json", "text"]) == "json"
    with pytest.raises(ValueError) as exc:
        validate_choice("input_type", "jsonl", ["json", "text"])
    assert "--input-type" in str(exc.value)
