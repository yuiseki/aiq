import sys
import select
import json
from rich.console import Console
import time
import joblib
import numpy
from .common import SafeStatus as Status, get_input_size, write_stdout

def _score_reader(model, class_dict, score_label, console):
    """Return a function record -> float, or None if no score was asked for.

    Preferring `predict_proba` matters: a threshold like 0.05 is only
    meaningful on a probability. `decision_function` returns an unbounded
    signed margin, so the same number means something else entirely; that
    substitution is announced on stderr rather than made silently.
    """
    if score_label is not None and score_label not in class_dict:
        raise ValueError(
            f"Unknown --score-label: {score_label!r}. "
            f"Must be one of: {', '.join(repr(k) for k in class_dict)}"
        )

    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)

        def read(X, predicted_idx):
            probs = model.predict_proba([X])[0]
            wanted = class_dict[score_label] if score_label is not None else predicted_idx
            return float(probs[classes.index(wanted)])

        return read

    if hasattr(model, "decision_function"):
        console.print(
            "[yellow]Warning: this model has no predict_proba; "
            "--score-field reports its decision_function instead. That is an "
            "unbounded margin, not a probability, so a 0..1 threshold does not "
            "apply to it.[/yellow]"
        )
        classes = list(model.classes_)

        def read(X, predicted_idx):
            raw = model.decision_function([X])[0]
            wanted = class_dict[score_label] if score_label is not None else predicted_idx
            if numpy.ndim(raw) == 0:
                # binary case: one signed margin, positive meaning the second
                # class. It is a 0-d numpy scalar, so it cannot be indexed.
                margin = float(raw)
                return margin if wanted == classes[1] else -margin
            return float(raw[classes.index(wanted)])

        return read

    raise ValueError(
        "--score-field was given but this model exposes neither "
        "predict_proba nor decision_function."
    )


def classify(
    model_path: str,
    label_field: str = "label",
    input_field: str = "embedding",
    remove_input: bool = True,
    skip_errors: bool = False,
    no_warn: bool = False,
    input_size: int | None = None,  # total records, only used for the progress display
    score_field: str | None = None,  # write the score here; omit for label only
    score_label: str | None = None,  # score this class instead of the predicted one
):
    console = Console(file=sys.stderr)
    if not no_warn:
        console.print("[yellow]⚠️  Warning: Loading models uses pickle, which can execute arbitrary Python code. " +
            "If you don't trust the source of this model file, press CTRL + C to exit.[/yellow]")
        time.sleep(5)
    model_obj = joblib.load(model_path)
    model = model_obj["model"]
    label2idx = model_obj["class_dict"]
    idx2label = {v: k for k, v in label2idx.items()}
    read_score = (
        _score_reader(model, label2idx, score_label, console)
        if score_field is not None
        else None
    )

    with Status("Starting classification...", console=console) as status:
        examples_read = 0
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 5)
            if not ready:
                # No new data within the timeout period
                if examples_read == 0:
                    # If we haven't read any examples, wait a bit longer
                    continue
                else:
                    # If we've processed some data, assume we're done
                    break

            line = sys.stdin.readline()
            loaded = None
            if not line:
                # End of file reached
                break

            # Process the line
            try:
                loaded = json.loads(line)
            except Exception as e:
                if not skip_errors:
                    console.print(f"[red]Error: Could not parse line {examples_read} as JSON. Example: {line}[/red]")
                    raise e
                else:
                    console.print(f"[yellow]Warning: Could not parse line {examples_read} as JSON. Example: {line}. Skipping.[/yellow]")
            assert loaded is not None, "Error parsing line"
            X = loaded[input_field]
            y = model.predict([X])[0]
            label = idx2label[y]
            loaded[label_field] = label
            if read_score is not None:
                loaded[score_field] = read_score(X, y)
            if remove_input:
                del loaded[input_field]
            write_stdout(json.dumps(loaded))
            examples_read += 1

            total = get_input_size(input_size)
            total_steps = total if total is not None else "???"
            status.update(f"Classified {examples_read} / {total_steps} examples...")

    console.print("Done.")
