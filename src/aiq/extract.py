"""Extract several fields from each record with one LLM call.

`aiq label` answers a single question ("which of these categories?") and caps
the reply at a few tokens. A pipeline stage usually needs more from the same
text at once -- a boolean, a category, a short summary that keeps proper
nouns intact -- and running `label` once per field would mean re-reading the
text once per field too.

The output shape lives in a JSON Schema file rather than in code, which is
what lets one command serve unrelated lenses (disasters, human security,
festivals) with no new Python.
"""
import asyncio
import contextlib
import json
import os
import sys
from typing import Literal, Optional

from openai import AsyncOpenAI
from rich.console import Console
from rich.style import Style

from aiq.common import SafeStatus as Status, validate_choice, write_stdout

RESPONSE_FORMATS = ["json_schema", "prompt"]


def read_schema(path: str) -> dict:
    with open(path, "r") as f:
        schema = json.load(f)
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError(
            f"{path}: the schema must be a JSON Schema object "
            '(i.e. {"type": "object", "properties": {...}})'
        )
    if not schema.get("properties"):
        raise ValueError(f"{path}: the schema has no properties, so nothing would be extracted")
    return schema


def read_text_file(path: Optional[str], what: str) -> str:
    if path is None:
        return ""
    with open(path, "r") as f:
        text = f.read().strip()
    if not text:
        raise ValueError(f"{path}: the {what} file is empty")
    return text


def build_messages(text, instruction, system, schema, response_format):
    """The prompt. In `prompt` mode the schema is the only thing telling the
    model what to emit, so it has to be spelled out in the message itself."""
    user = f"{instruction}\n\n{text}" if instruction else text
    if response_format == "prompt":
        user += (
            "\n\nRespond with a single JSON object matching this schema, "
            "with no prelude, commentary or code fence:\n"
            + json.dumps(schema, ensure_ascii=False)
        )
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


def parse_object(content: str, schema: dict) -> dict:
    """Take the object out of the reply and keep only the declared fields.

    Even with a constrained decoder a model may wrap the object in a code
    fence, so strip that. Unknown extra keys are dropped rather than merged:
    the schema is the contract for what this stage adds to a record, and
    silently widening it would change downstream field names without notice.
    """
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: -3]
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError(f"the model did not return a JSON object: {content!r}")
    missing = [k for k in schema.get("required", []) if k not in obj]
    if missing:
        raise ValueError(
            f"the model omitted required field(s) {missing}: {content!r}"
        )
    return {k: obj.get(k) for k in schema["properties"]}


def extract(
    schema_file: str,
    input_type: Literal["json", "text"] = "text",
    input_field: Optional[str] = None,
    instruction_file: Optional[str] = None,
    system_file: Optional[str] = None,
    output_field: Optional[str] = None,  # None merges the fields into the record
    response_format: Literal["json_schema", "prompt"] = "json_schema",
    model: str = "gpt-4o-mini",
    file: Optional[str] = None,
    max_concurrency: int = 10,
    max_tokens: int = 512,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    skip_errors: bool = False,
    progress: bool = False,  # Do not turn on if piping to another process
):
    validate_choice("input_type", input_type, ["json", "text"])
    validate_choice("response_format", response_format, RESPONSE_FORMATS)
    if input_type == "json" and input_field is None:
        raise ValueError("input_type is 'json' but input_field is not provided")
    if input_type == "text" and input_field is not None:
        sys.stderr.write(
            "Warning: input_type is 'text' but input_field is provided. Ignoring input_field.\n"
        )

    schema = read_schema(schema_file)
    fields = list(schema["properties"])
    instruction = read_text_file(instruction_file, "instruction")
    system = read_text_file(system_file, "system prompt")
    console = Console(file=sys.stderr)
    examples_read = 0

    client = AsyncOpenAI(
        api_key=api_key or os.environ["OPENAI_API_KEY"],
        base_url=api_base_url or os.environ.get("OPENAI_BASE_URL", None),
    )

    request_extras = {}
    if response_format == "json_schema":
        request_extras["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "extract",
                "schema": schema,
                "strict": True,
            },
        }

    file_handle = sys.stdin if file is None else open(file, "r")

    def emit(record, values):
        if output_field is not None:
            write_stdout(json.dumps({**record, output_field: values}, ensure_ascii=False))
            return
        collisions = [k for k in values if k in record]
        if collisions:
            raise ValueError(
                "the schema would overwrite existing field(s) "
                f"{collisions} in the input record. Use --output-field to nest "
                "the result under one key instead."
            )
        write_stdout(json.dumps({**record, **values}, ensure_ascii=False))

    async def run():
        nonlocal examples_read
        semaphore = asyncio.Semaphore(max_concurrency)
        tasks = []

        async def process_line(line: str):
            nonlocal examples_read
            try:
                if input_type == "json":
                    record = json.loads(line)
                    text = record[input_field]
                else:
                    record = {"text": line.rstrip("\n")}
                    text = record["text"]
                if not text:
                    # One record in, one record out. Dropping the ones that
                    # cannot be processed would silently break the caller's
                    # ability to join results back to their inputs.
                    emit(record, {k: None for k in fields})
                    examples_read += 1
                    return
                response = await client.chat.completions.create(
                    model=model,
                    messages=build_messages(
                        text, instruction, system, schema, response_format
                    ),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **request_extras,
                )
                values = parse_object(response.choices[0].message.content, schema)
                emit(record, values)
                examples_read += 1
                if progress and status is not None:
                    status.update(f"Extracted from {examples_read} examples...")
            except Exception as e:
                if not skip_errors:
                    raise RuntimeError(
                        f"Error processing line: {line}. Error: {str(e)}"
                    ) from e
                console.print(f"[yellow]Warning: skipping a record: {e}[/yellow]")

        status_context = (
            Status("", console=console, spinner_style=Style(color="purple"))
            if progress
            else contextlib.nullcontext()
        )
        with status_context as status:
            for line in file_handle:
                if not line.strip():
                    continue
                await semaphore.acquire()
                task = asyncio.create_task(process_line(line))
                task.add_done_callback(lambda t: semaphore.release())
                tasks.append(task)
            if tasks:
                await asyncio.gather(*tasks)

    try:
        asyncio.run(run())
    finally:
        if file is not None:
            file_handle.close()
