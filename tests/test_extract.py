"""`aiq extract`: one LLM call per record producing several fields at once.

`aiq label` answers "which of these categories?" and caps the reply at 8
tokens. Real pipeline stages need more than that from a single pass: a
boolean, a category, a short summary that keeps proper nouns intact. Doing
that with `label` would mean one LLM call per field over the same text.

Keeping the output shape in a JSON Schema file (rather than in code) is what
turns a lens -- disaster, human security, festivals -- into data.
"""
import json
import pathlib

import pytest

from conftest import REPO_ROOT, run_python

SCHEMA = {
    "type": "object",
    "properties": {
        "is_disaster": {"type": "boolean"},
        "category": {"type": "string"},
        "topic": {"type": "string"},
    },
    "required": ["is_disaster", "category", "topic"],
}

# A fake OpenAI client that echoes back a fixed object, and records the
# request so the test can assert on how the schema was passed.
FAKE = r'''
import json, sys
import aiq.extract as X

CALLS = []

class _Msg:
    def __init__(self, c): self.content = c
class _Choice:
    def __init__(self, c): self.message = _Msg(c)
class _Resp:
    def __init__(self, c): self.choices = [_Choice(c)]
class _Completions:
    async def create(self, **kw):
        CALLS.append(kw)
        return _Resp(json.dumps(
            {"is_disaster": True, "category": "地震", "topic": "熊本県益城町で震度5強"}
        ))
class _Chat:
    completions = _Completions()
class FakeClient:
    def __init__(self, *a, **k): self.chat = _Chat()

X.AsyncOpenAI = FakeClient
try:
    X.extract(%(kwargs)s)
finally:
    sys.stderr.write("CALLS=" + json.dumps(CALLS[:1]) + "\n")
'''


def _run(env, tmp_path, schema=SCHEMA, **kwargs):
    schema_file = tmp_path / "schema.json"
    schema_file.write_text(json.dumps(schema))
    instruction = tmp_path / "instruction.txt"
    instruction.write_text("次の放送文字起こしを分類してください。")
    kwargs.setdefault("input_type", "json")
    kwargs.setdefault("input_field", "text")
    kwargs.setdefault("schema_file", str(schema_file))
    kwargs.setdefault("instruction_file", str(instruction))
    kwargs.setdefault("model", "fake")
    kwargs.setdefault("api_key", "none")
    code = FAKE % {"kwargs": ", ".join(f"{k}={v!r}" for k, v in kwargs.items())}
    stdin = "\n".join(
        json.dumps({"seg": f"s{i}", "text": f"本文{i}"}) for i in range(3)
    ) + "\n"
    return run_python(code, env, stdin=stdin)


def _records(stdout):
    lines = [l for l in stdout.splitlines() if l.strip()]
    assert "\x1b" not in stdout, f"ANSI escape leaked into stdout: {stdout!r}"
    return [json.loads(l) for l in lines]


def test_schema_fields_are_merged_into_the_record(ansi_env, tmp_path):
    r = _run(ansi_env, tmp_path)
    assert r.returncode == 0, r.stderr
    recs = _records(r.stdout)
    assert len(recs) == 3
    for rec in recs:
        assert rec["is_disaster"] is True
        assert rec["category"] == "地震"
        assert rec["topic"] == "熊本県益城町で震度5強"
        # input fields must survive so the caller keeps the correspondence
        assert rec["seg"].startswith("s")
        assert rec["text"].startswith("本文")


def test_schema_is_sent_as_response_format(ansi_env, tmp_path):
    """Constraining the decoder beats asking politely in the prompt."""
    r = _run(ansi_env, tmp_path)
    assert r.returncode == 0, r.stderr
    call = json.loads(r.stderr.split("CALLS=")[1].splitlines()[0])[0]
    rf = call["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"]["properties"].keys() >= {
        "is_disaster", "category", "topic"
    }


def test_prompt_mode_does_not_send_response_format(ansi_env, tmp_path):
    """Endpoints without json_schema support still have to be usable."""
    r = _run(ansi_env, tmp_path, response_format="prompt")
    assert r.returncode == 0, r.stderr
    call = json.loads(r.stderr.split("CALLS=")[1].splitlines()[0])[0]
    assert "response_format" not in call
    # the schema has to reach the model some other way
    sent = json.dumps(call["messages"], ensure_ascii=False)
    assert "is_disaster" in sent


def test_output_field_nests_instead_of_merging(ansi_env, tmp_path):
    r = _run(ansi_env, tmp_path, output_field="lens")
    assert r.returncode == 0, r.stderr
    recs = _records(r.stdout)
    assert all(rec["lens"]["category"] == "地震" for rec in recs)
    assert all("category" not in rec for rec in recs)


def test_field_collision_is_an_error(ansi_env, tmp_path):
    """Merging must not silently destroy an input field."""
    schema_file = tmp_path / "schema.json"
    schema_file.write_text(json.dumps(SCHEMA))
    instruction = tmp_path / "instruction.txt"
    instruction.write_text("分類してください。")
    code = FAKE % {"kwargs": ", ".join(
        f"{k}={v!r}" for k, v in dict(
            input_type="json", input_field="text",
            schema_file=str(schema_file), instruction_file=str(instruction),
            model="fake", api_key="none",
        ).items())}
    stdin = json.dumps({"seg": "s0", "text": "本文", "category": "既存の値"}) + "\n"
    r = run_python(code, ansi_env, stdin=stdin)
    assert r.returncode != 0
    assert "category" in r.stderr


def test_empty_input_text_is_passed_through_not_dropped(ansi_env, tmp_path):
    """A pipeline stage must emit one record per input record. Dropping the
    ones it cannot process breaks the caller's ability to join results back
    to their inputs, and does it silently."""
    schema_file = tmp_path / "schema.json"
    schema_file.write_text(json.dumps(SCHEMA))
    instruction = tmp_path / "instruction.txt"
    instruction.write_text("分類してください。")
    code = FAKE % {"kwargs": ", ".join(
        f"{k}={v!r}" for k, v in dict(
            input_type="json", input_field="text",
            schema_file=str(schema_file), instruction_file=str(instruction),
            model="fake", api_key="none",
        ).items())}
    stdin = "\n".join([
        json.dumps({"seg": "s0", "text": "本文"}),
        json.dumps({"seg": "s1", "text": ""}),
        json.dumps({"seg": "s2", "text": "本文"}),
    ]) + "\n"
    r = run_python(code, ansi_env, stdin=stdin)
    assert r.returncode == 0, r.stderr
    recs = _records(r.stdout)
    assert {rec["seg"] for rec in recs} == {"s0", "s1", "s2"}
    empty = [rec for rec in recs if rec["seg"] == "s1"][0]
    assert empty["category"] is None
    assert empty["is_disaster"] is None
