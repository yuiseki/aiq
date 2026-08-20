"""`aiq label` must emit one record per input record.

`if not input_text: return` dropped the record entirely. That is the same
class of defect as the `embed` bug where an empty text ended the stream: a
stage that quietly emits fewer records than it consumed leaves the caller
unable to join results back to inputs, with nothing on stderr to say so.
"""
import json

from conftest import run_python

FAKE = r'''
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
L.label(input_type="json", input_field="text",
        label_options=["ニュース", "音楽"], model="fake", api_key="none")
'''


def test_empty_text_is_passed_through_with_a_null_label(ansi_env):
    stdin = "\n".join([
        json.dumps({"seg": "s0", "text": "本文"}),
        json.dumps({"seg": "s1", "text": ""}),
        json.dumps({"seg": "s2", "text": "本文"}),
    ]) + "\n"
    r = run_python(FAKE, ansi_env, stdin=stdin)
    assert r.returncode == 0, r.stderr
    recs = [json.loads(l) for l in r.stdout.splitlines() if l.strip()]
    assert {rec["seg"] for rec in recs} == {"s0", "s1", "s2"}
    empty = [rec for rec in recs if rec["seg"] == "s1"][0]
    assert empty["label"] is None
