"""Two pipelines on one machine must not interfere.

The commands used to hand `AIQ_INPUT_SIZE` to each other through a single
fixed file, `/tmp/aiq.status`, which every invocation truncated on startup.
Two concurrent pipelines therefore corrupted each other's state. The value is
only used for the "n / total" progress display, so the implicit channel was
removed: the total is now passed explicitly (`--input-size`) or through the
per-process `AIQ_INPUT_SIZE` environment variable.
"""
import concurrent.futures
import glob
import json
import os
import random
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from aiq.common import get_input_size  # noqa: E402


def _env(extra=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(REPO_ROOT, "src")
    env["FORCE_COLOR"] = "3"
    env.pop("AIQ_INPUT_SIZE", None)
    env.update(extra or {})
    return env


def test_no_shared_state_file_is_created(tmp_path):
    before = set(glob.glob("/tmp/aiq*"))
    r = subprocess.run(
        [sys.executable, "-m", "aiq.aiq", "--", "--help"],
        env=_env(), capture_output=True, text=True, timeout=120, cwd=REPO_ROOT,
    )
    assert r.returncode == 0, r.stderr
    assert set(glob.glob("/tmp/aiq*")) == before


def test_input_size_sources():
    assert get_input_size(50) == 50            # explicit flag wins
    os.environ["AIQ_INPUT_SIZE"] = "7"
    try:
        assert get_input_size() == 7           # environment fallback
        assert get_input_size(3) == 3
    finally:
        del os.environ["AIQ_INPUT_SIZE"]
    assert get_input_size() is None            # unknown -> "???"


def _make_data(path, n, dim, offset, seed):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        cls = i % 2
        rows.append({
            "seg": f"{offset}-{i}",
            "gold": "音楽",
            "label": f"c{cls}",
            "embedding": [rng.gauss(cls, 0.1) for _ in range(dim)],
        })
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return rows


def _run_pipeline(data_path, model_path, input_size, dim):
    """`cat data | aiq train` followed by `cat data | aiq classify`."""
    train = subprocess.run(
        ["bash", "-c",
         f'cat "{data_path}" | "{sys.executable}" -m aiq.aiq train '
         f'--model-path "{model_path}" --n-classes 2 --batch-size 16 '
         f'--timeout 1.0 --input-size {input_size}'],
        env=_env(), capture_output=True, text=True, timeout=300, cwd=REPO_ROOT,
    )
    classify = subprocess.run(
        ["bash", "-c",
         f'cat "{data_path}" | "{sys.executable}" -m aiq.aiq classify '
         f'--model-path "{model_path}" --no-warn --input-size {input_size}'],
        env=_env(), capture_output=True, text=True, timeout=300, cwd=REPO_ROOT,
    )
    return train, classify, dim


def test_two_concurrent_pipelines_stay_independent(tmp_path):
    """Different embedding dimensions: any cross-talk shows up as an error."""
    specs = []
    for idx, (n, dim) in enumerate([(96, 8), (128, 16)]):
        data = tmp_path / f"data{idx}.jsonl"
        _make_data(data, n, dim, idx, seed=idx)
        specs.append((data, tmp_path / f"model{idx}.joblib", n, dim))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda s: _run_pipeline(*s), specs))

    for (data, model, n, dim), (train, classify, _) in zip(specs, results):
        assert train.returncode == 0, train.stderr
        assert classify.returncode == 0, classify.stderr
        lines = [l for l in classify.stdout.splitlines() if l.strip()]
        assert len(lines) == n, classify.stderr[-2000:]
        records = [json.loads(l) for l in lines]
        assert all(r["gold"] == "音楽" for r in records)
        # the model that was trained on this pipeline's data, not the other's
        import joblib
        assert joblib.load(model)["embedding_dim"] == dim
