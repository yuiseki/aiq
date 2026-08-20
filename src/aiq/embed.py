import sys
import select
from abc import ABC, abstractmethod
from typing import Literal, Union, Optional
import json
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
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
    model = EmbeddingModel.from_registry(model_name)
    examples_read = 0
    # with Status("", console=console) if  as status:
    # if show progress, then use status otherwise contextlib.nullcontext
    if file is None:
        file_handle = sys.stdin
    else:
        file_handle = open(file, "r")

    with (
        Status("", console=console, spinner_style=rich.style.Style(color="purple")) if progress else contextlib.nullcontext()
    ) as status:
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
                embedding = model.encode([input_text], show_progress=False)[0]
                write_stdout(json.dumps({
                    **input_json,
                    output_field: embedding.tolist()
                }))
                examples_read += 1
                if progress and status is not None:
                    status.update(f"Embedded {examples_read} examples...")
            except Exception as e:
                if not skip_errors:
                    raise RuntimeError(f"Error processing line: {line}. Error: {str(e)}") from e
