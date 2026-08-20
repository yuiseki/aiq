import os
import sys
from typing import Any, Iterable

from rich.live import Live as _Live
from rich.status import Status as _RichStatus


def write_stdout(line: str) -> None:
    """Write one machine-readable record to stdout.

    Deliberately bypasses `rich`: `rich.Console` honours environment
    variables such as FORCE_COLOR / CLICOLOR_FORCE and will emit ANSI escape
    sequences even when stdout is a pipe, which corrupts the JSON stream that
    the next command in the pipeline has to parse. Human-facing output still
    goes through `rich`, on stderr.
    """
    try:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        # The downstream command closed the pipe (`... | head -3`, or a
        # consumer that died). Stop quietly like a normal unix filter instead
        # of dumping a traceback plus an "Exception ignored" message at exit.
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except OSError:
            pass
        raise SystemExit(141)


def validate_choice(name: str, value: Any, allowed: Iterable[Any]) -> Any:
    """Validate an argument that is declared as a `Literal` in the signature.

    `fire` does not enforce type annotations, so an unknown value would
    otherwise silently fall through to another branch and change what the
    command does without any error at all. Failing loudly is the only safe
    option for non-interactive callers.
    """
    allowed = list(allowed)
    if value not in allowed:
        raise ValueError(
            f"Invalid value for --{name.replace('_', '-')}: {value!r}. "
            f"Must be one of: {', '.join(repr(a) for a in allowed)}"
        )
    return value


def get_input_size(explicit: int | None = None) -> int | None:
    """Total number of input records, if known, for the progress display.

    This used to be handed between commands through a single fixed file
    (`/tmp/aiq.status`) that every invocation truncated at startup, which
    made two concurrent pipelines corrupt each other. There is no shared
    mutable state any more: the total is either passed explicitly
    (`--input-size`) or read from the AIQ_INPUT_SIZE environment variable,
    both of which are per-process and therefore parallel-safe.
    """
    if explicit is not None:
        return int(explicit)
    value = os.environ.get("AIQ_INPUT_SIZE")
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


class SafeStatus(_RichStatus):
    """A `rich` status spinner that leaves stdout alone.

    `rich.status.Status` wraps a `Live` display created with the default
    `redirect_stdout=True`, which globally replaces `sys.stdout` with a proxy
    that funnels writes into the (stderr) console. That silently moved every
    output record out of the pipe and into the progress display, so
    `aiq classify` produced no parseable stdout at all while its spinner was
    running. Recreate the `Live` with both redirects disabled.
    """

    def __init__(
        self,
        status,
        *,
        console=None,
        spinner: str = "dots",
        spinner_style="status.spinner",
        speed: float = 1.0,
        refresh_per_second: float = 12.5,
    ):
        super().__init__(
            status,
            console=console,
            spinner=spinner,
            spinner_style=spinner_style,
            speed=speed,
            refresh_per_second=refresh_per_second,
        )
        self._live = _Live(
            self.renderable,
            console=console,
            refresh_per_second=refresh_per_second,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False,
        )
