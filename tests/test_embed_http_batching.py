"""`aiq embed` must batch without reordering, dropping, or corrupting records.

Two things are being guarded here. First, the HTTP route: `aiq embed` can
send batches to an OpenAI-compatible endpoint instead of running a model
locally, and the records that come back have to keep their identifiers and
their original fields. Second, batching itself: a batch is a place where the
correspondence between input and output can silently get lost, so order,
field preservation and empty-text skipping are checked explicitly, with
stdout still parseable under FORCE_COLOR.
"""
import json
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, run_python

# A fake OpenAI client whose embedding for a text is derived from the text, so
# a mismatch between an input record and its embedding is detectable. It also
# returns `data` in shuffled order to prove the caller does not rely on the
# order of the response.
FAKE_API = r'''
import aiq.embed as E

class _Item:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding
class _Resp:
    def __init__(self, data): self.data = data
class _Embeddings:
    def __init__(self, calls): self.calls = calls
    def create(self, model=None, input=None):
        assert isinstance(input, list), "input must be a batch (a list)"
        self.calls.append(list(input))
        items = [_Item(i, [float(len(t)), float(ord(t[-1])), 3.0])
                 for i, t in enumerate(input)]
        return _Resp(list(reversed(items)))
class FakeClient:
    calls = []
    def __init__(self, *a, **k):
        FakeClient.kwargs = k
        self.embeddings = _Embeddings(FakeClient.calls)

E.OpenAI = FakeClient
'''

def _run(body: str, env, stdin: str):
    return run_python(FAKE_API + body, env, stdin=stdin)


def _records(stdout: str, expected_lines: int):
    assert "\x1b" not in stdout, f"ANSI escape leaked into stdout: {stdout!r}"
    lines = [l for l in stdout.splitlines() if l.strip()]
    assert len(lines) == expected_lines, stdout[:2000]
    return [json.loads(l) for l in lines]


def test_api_route_preserves_identifiers_and_fields(ansi_env):
    rows = [{"seg": f"s{i}", "text": f"本文{i}", "gold": "音楽"} for i in range(5)]
    r = _run(
        'E.embed(input_type="json", input_field="text",'
        ' model="fake-embed", api_base_url="http://127.0.0.1:30190/v1",'
        ' batch_size=2)',
        ansi_env,
        "\n".join(json.dumps(x) for x in rows) + "\n",
    )
    assert r.returncode == 0, r.stderr
    records = _records(r.stdout, 5)
    assert [rec["seg"] for rec in records] == [x["seg"] for x in rows]
    for rec, src in zip(records, rows):
        assert rec["text"] == src["text"]
        assert rec["gold"] == "音楽"
        # the embedding belongs to *this* record's text
        assert rec["embedding"] == [float(len(src["text"])),
                                    float(ord(src["text"][-1])), 3.0]


@pytest.mark.parametrize("batch_size", [1, 2, 3, 7, 64])
def test_output_order_matches_input_order(ansi_env, batch_size):
    """Every batch size, including ones that do not divide the input."""
    rows = [{"seg": f"s{i}", "text": f"text-{chr(97 + i)}"} for i in range(13)]
    r = _run(
        f'E.embed(input_type="json", input_field="text", model="fake-embed",'
        f' api_base_url="http://x/v1", batch_size={batch_size})',
        ansi_env,
        "\n".join(json.dumps(x) for x in rows) + "\n",
    )
    assert r.returncode == 0, r.stderr
    records = _records(r.stdout, 13)
    assert [rec["seg"] for rec in records] == [x["seg"] for x in rows]
    assert [rec["text"] for rec in records] == [x["text"] for x in rows]
    for rec in records:
        assert rec["embedding"][1] == float(ord(rec["text"][-1]))


def test_empty_text_is_skipped_and_the_stream_continues(ansi_env):
    """An empty field must not truncate the rest of the stream."""
    rows = [
        {"seg": "s0", "text": "first"},
        {"seg": "s1", "text": ""},
        {"seg": "s2", "text": "third"},
        {"seg": "s3", "text": ""},
        {"seg": "s4", "text": "fifth"},
    ]
    r = _run(
        'E.embed(input_type="json", input_field="text", model="fake-embed",'
        ' api_base_url="http://x/v1", batch_size=2)',
        ansi_env,
        "\n".join(json.dumps(x) for x in rows) + "\n",
    )
    assert r.returncode == 0, r.stderr
    records = _records(r.stdout, 3)
    assert [rec["seg"] for rec in records] == ["s0", "s2", "s4"]


def test_local_route_batches_and_keeps_order(ansi_env):
    """The local ONNX route batches too, with the same guarantees."""
    body = r'''
import numpy as np
class FakeModel:
    batches = []
    def encode(self, texts, show_progress=False):
        assert isinstance(texts, list)
        FakeModel.batches.append(len(texts))
        return [np.array([float(len(t)), float(ord(t[-1])), 3.0]) for t in texts]
class FakeRegistry:
    @staticmethod
    def from_registry(name): return FakeModel()
E.EmbeddingModel = FakeRegistry
E.embed(input_type="json", input_field="text", batch_size=4)
import sys
print("batches:", FakeModel.batches, file=sys.stderr)
'''
    rows = [{"seg": f"s{i}", "text": f"text-{chr(97 + i)}"} for i in range(10)]
    r = _run(body, ansi_env, "\n".join(json.dumps(x) for x in rows) + "\n")
    assert r.returncode == 0, r.stderr
    records = _records(r.stdout, 10)
    assert [rec["seg"] for rec in records] == [x["seg"] for x in rows]
    # 4 + 4 + 2, i.e. actually batched rather than one call per record
    assert "batches: [4, 4, 2]" in r.stderr


def test_first_batch_dimension_is_reported_on_stderr(ansi_env):
    rows = [{"seg": f"s{i}", "text": f"本文{i}"} for i in range(3)]
    r = _run(
        'E.embed(input_type="json", input_field="text", model="fake-embed",'
        ' api_base_url="http://x/v1", batch_size=64)',
        ansi_env,
        "\n".join(json.dumps(x) for x in rows) + "\n",
    )
    assert r.returncode == 0, r.stderr
    _records(r.stdout, 3)
    # the dimension has to be visible before `aiq train` is handed the stream
    assert "3" in r.stderr and "fake-embed" in r.stderr


def test_api_base_url_requires_a_model():
    """Never guess the model name: the endpoint would embed with the wrong one."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(REPO_ROOT, "src")
    r = subprocess.run(
        [sys.executable, "-m", "aiq.aiq", "embed", "--api-base-url", "http://x/v1"],
        input="", env=env, capture_output=True, text=True, timeout=120,
        cwd=REPO_ROOT,
    )
    assert r.returncode != 0
    assert "--model" in (r.stderr + r.stdout)


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_batch_size_must_be_positive(bad):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(REPO_ROOT, "src")
    r = subprocess.run(
        [sys.executable, "-m", "aiq.aiq", "embed", "--batch-size", bad],
        input="", env=env, capture_output=True, text=True, timeout=120,
        cwd=REPO_ROOT,
    )
    assert r.returncode != 0
    assert "batch_size" in (r.stderr + r.stdout)


def test_openai_base_url_env_does_not_hijack_the_default_route(ansi_env):
    """`OPENAI_BASE_URL` exported for `aiq label` must not reroute `embed`."""
    env = dict(ansi_env)
    env["OPENAI_BASE_URL"] = "http://127.0.0.1:1/v1"  # nothing listens here
    body = r'''
import numpy as np
class FakeModel:
    def encode(self, texts, show_progress=False):
        return [np.array([0.5, 0.5]) for _ in texts]
class FakeRegistry:
    @staticmethod
    def from_registry(name): return FakeModel()
E.EmbeddingModel = FakeRegistry
E.embed(input_type="json", input_field="text")
'''
    r = _run(body, env, json.dumps({"seg": "s0", "text": "hello"}) + "\n")
    assert r.returncode == 0, r.stderr
    records = _records(r.stdout, 1)
    assert records[0]["embedding"] == [0.5, 0.5]
