import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def ansi_env():
    """Environment that forces `rich` to emit colour even when piped.

    FORCE_COLOR is what makes the original bug reproducible: `rich` honours
    it and overrides its own tty detection, so anything printed through a
    Console lands in the pipe with ANSI escapes.
    """
    env = dict(os.environ)
    env["FORCE_COLOR"] = "3"
    env["CLICOLOR_FORCE"] = "1"
    env["PYTHONPATH"] = os.path.join(REPO_ROOT, "src")
    env.pop("AIQ_INPUT_SIZE", None)
    return env


def run_python(code: str, env: dict, stdin: str = "", timeout: int = 120):
    return subprocess.run(
        [sys.executable, "-c", code],
        input=stdin,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=REPO_ROOT,
    )
