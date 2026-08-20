import sys
from typing import Literal, Optional
import json
import numpy as np
import rich
from rich.console import Console
import contextlib
from onnx_embedding_models import EmbeddingModel
from aiq.common import SafeStatus as Status, write_stdout, validate_choice

def embed(
    input_type: Literal["text", "json"] = "json",
    input_field: str | None = "text",
    output_field: str = "embedding",
    model_name: str = "snowflake-xs",
    batch_size: int = 64,
    skip_errors: bool = False,
    progress: bool = False, # do not turn on if piping to another process, they'll interfere
    file: Optional[str] = None
):
    validate_choice("input_type", input_type, ["text", "json"])
    console = Console(file=sys.stderr)
    # validate arguments before loading the model, so a bad invocation fails
    # fast instead of after a model download
    if input_type == "json" and input_field is None:
        raise ValueError("input_type is 'json' but input_field is not provided")
    if input_type == "text" and input_field is not None:
        console.print("\n[yellow]Warning: input_type is 'text' but input_field is provided. Ignoring input_field.[/yellow]\n")
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    model = EmbeddingModel.from_registry(model_name)
    examples_read = 0

    if file is None:
        file_handle = sys.stdin
    else:
        file_handle = open(file, "r")

    def flush(batch: list[tuple[dict, str]]) -> None:
        """Embed one batch and write the records out in input order."""
        nonlocal examples_read
        if not batch:
            return
        try:
            # `encode` takes a list, so one call per batch replaces what used
            # to be one call per record
            vectors = model.encode([text for _, text in batch], show_progress=False)
        except Exception as e:
            if skip_errors:
                return
            raise RuntimeError(
                f"Error embedding batch of {len(batch)} records with "
                f"{model_name}: {str(e)}"
            ) from e
        for (input_json, _), vector in zip(batch, vectors):
            write_stdout(json.dumps({
                **input_json,
                output_field: np.asarray(vector).tolist()
            }))
            examples_read += 1

    with (
        Status("", console=console, spinner_style=rich.style.Style(color="purple")) if progress else contextlib.nullcontext()
    ) as status:
        batch: list[tuple[dict, str]] = []
        for line in file_handle:
            if not line:
                # End of file reached
                break
            try:
                if input_type == "json":
                    input_json = json.loads(line)
                    input_text = input_json[input_field]
                else:
                    input_json = {"text": line}
                    input_text = line
                if not input_text:
                    # skip this record; `return` here used to abort the whole
                    # stream and silently truncate the output
                    continue
                batch.append((input_json, input_text))
            except Exception as e:
                if not skip_errors:
                    raise RuntimeError(f"Error processing line: {line}. Error: {str(e)}") from e
                continue
            if len(batch) >= batch_size:
                # only ever one batch in memory: this stays a stream
                flush(batch)
                batch = []
                if progress and status is not None:
                    status.update(f"Embedded {examples_read} examples...")
        flush(batch)
        if progress and status is not None:
            status.update(f"Embedded {examples_read} examples...")

    if file is not None:
        file_handle.close()
