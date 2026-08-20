import sys
from typing import Literal, Optional
import json
import numpy as np
import rich
from rich.console import Console
import contextlib
import os
from onnx_embedding_models import EmbeddingModel
from openai import OpenAI
from aiq.common import SafeStatus as Status, write_stdout, validate_choice


class LocalEmbedder:
    """Local ONNX embedding model from the `onnx_embedding_models` registry."""

    def __init__(self, model_name: str):
        self.name = model_name
        self.model = EmbeddingModel.from_registry(model_name)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        # `encode` already takes a list, so one call per batch replaces what
        # used to be one call per record.
        vectors = self.model.encode(texts, show_progress=False)
        return [np.asarray(vector).tolist() for vector in vectors]


class ApiEmbedder:
    """Embeddings from an OpenAI-compatible HTTP endpoint.

    Lets `aiq embed` reuse an embedding server that is already running
    (llama.cpp, vLLM, TEI, OpenAI itself) instead of downloading and running a
    model locally. One request carries a whole batch, so the round-trip cost is
    amortised over `batch_size` records.
    """

    def __init__(self, model: str, api_base_url: Optional[str], api_key: Optional[str]):
        self.name = model
        self.model = model
        self.client = OpenAI(
            # local servers usually ignore the key, but the client insists on
            # one being present
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or "not-needed",
            base_url=api_base_url or os.environ.get("OPENAI_BASE_URL", None),
        )

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        # the API is documented to return the embeddings in input order, but
        # `index` is authoritative, so sort by it rather than trusting order
        data = sorted(response.data, key=lambda item: item.index)
        if len(data) != len(texts):
            raise RuntimeError(
                f"Endpoint returned {len(data)} embeddings for {len(texts)} inputs"
            )
        return [list(item.embedding) for item in data]


def embed(
    input_type: Literal["text", "json"] = "json",
    input_field: str | None = "text",
    output_field: str = "embedding",
    model_name: str = "snowflake-xs",
    model: Optional[str] = None,
    api_base_url: Optional[str] = None,
    api_key: Optional[str] = None,
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

    # An HTTP endpoint is only used when it is asked for explicitly, so the
    # default behaviour (local ONNX) cannot change under a caller who happens
    # to have OPENAI_BASE_URL exported for `aiq label`.
    use_api = api_base_url is not None or model is not None
    if use_api:
        if model is None:
            raise ValueError(
                "--api-base-url requires --model (the model name the endpoint "
                "expects), so the embeddings cannot silently come from the "
                "wrong model"
            )
        embedder = ApiEmbedder(model=model, api_base_url=api_base_url, api_key=api_key)
    else:
        embedder = LocalEmbedder(model_name)

    examples_read = 0

    if file is None:
        file_handle = sys.stdin
    else:
        file_handle = open(file, "r")

    def flush(batch: list[tuple[dict, str]]) -> None:
        """Embed one batch and write it out in input order."""
        nonlocal examples_read
        if not batch:
            return
        try:
            vectors = embedder.encode_batch([text for _, text in batch])
        except Exception as e:
            if skip_errors:
                return
            raise RuntimeError(
                f"Error embedding batch of {len(batch)} records with "
                f"{embedder.name}: {str(e)}"
            ) from e
        for (input_json, _), vector in zip(batch, vectors):
            write_stdout(json.dumps({
                **input_json,
                output_field: vector
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
                # only one batch is ever held in memory: this stays a stream
                flush(batch)
                batch = []
                if progress and status is not None:
                    status.update(f"Embedded {examples_read} examples...")
        flush(batch)
        if progress and status is not None:
            status.update(f"Embedded {examples_read} examples...")

    if file is not None:
        file_handle.close()
