"""Inspect CLINC150 and classify support messages with a human-review option."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import NamedTuple
from urllib.request import urlopen

DATA_URL = (
    "https://raw.githubusercontent.com/clinc/oos-eval/"
    "828f8093932c8fe6ca7936c3d2e52903b1c523de/data/data_full.json"
)
DATA_SHA256 = "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"
DATA_PATH = Path(__file__).resolve().parent / "data" / "clinc150.json"
MAX_ROUTED_ERROR = 0.05
MAX_OOS_ROUTING = 0.10


class RoutingMetrics(NamedTuple):
    threshold: float
    coverage: float
    routed_error: float | None
    oos_routing: float


def load_dataset():
    """Load the original dataset and verify the exact version used."""
    if DATA_PATH.exists():
        contents = DATA_PATH.read_bytes()
    else:
        with urlopen(DATA_URL, timeout=30) as response:
            contents = response.read()

    if hashlib.sha256(contents).hexdigest() != DATA_SHA256:
        raise ValueError(f"Unexpected dataset contents. Check {DATA_PATH}.")

    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(exist_ok=True)
        DATA_PATH.write_bytes(contents)

    return json.loads(contents)


def show_dataset(dataset):
    for split in ("train", "val", "test", "oos_train", "oos_val", "oos_test"):
        rows = dataset[split]
        label_count = len({label for _, label in rows})
        print(f"{split}: {len(rows):,} examples, {label_count} labels")

    message, label = dataset["train"][0]
    print(f"Example message: {message}")
    print(f"Correct label: {label}")


def train_baseline(dataset):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    train_messages, train_labels = zip(*dataset["train"])
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2)),
        LogisticRegression(max_iter=500),
    )
    return model.fit(train_messages, train_labels)


def score_messages(model, messages):
    scores = model.predict_proba(messages)
    best_columns = scores.argmax(axis=1)
    return model.classes_[best_columns], scores.max(axis=1)


def routing_metrics(threshold, known_scores, known_correct, oos_scores):
    known_routed = known_scores > threshold
    oos_routed = oos_scores > threshold
    routed_count = known_routed.sum() + oos_routed.sum()
    wrong_count = (known_routed & ~known_correct).sum() + oos_routed.sum()
    return RoutingMetrics(
        threshold,
        routed_count / (len(known_scores) + len(oos_scores)),
        wrong_count / routed_count if routed_count else None,
        oos_routed.mean(),
    )


def select_threshold(rows):
    eligible = [
        row
        for row in rows
        if row.routed_error is not None
        and row.routed_error <= MAX_ROUTED_ERROR
        and row.oos_routing <= MAX_OOS_ROUTING
    ]
    # Rows are in ascending threshold order. Ties keep the smaller threshold.
    return max(eligible, key=lambda row: row.coverage, default=rows[-1])


def evaluate_validation(model, dataset):
    from sklearn.metrics import f1_score

    known_messages, known_labels = zip(*dataset["val"])
    oos_messages, _ = zip(*dataset["oos_val"])
    known_predictions, known_scores = score_messages(model, known_messages)
    _, oos_scores = score_messages(model, oos_messages)
    known_correct = known_predictions == known_labels
    rows = [
        routing_metrics(index / 100, known_scores, known_correct, oos_scores)
        for index in range(101)
    ]
    always_route = routing_metrics(-1, known_scores, known_correct, oos_scores)
    selected = select_threshold(rows)

    validation_f1 = f1_score(known_labels, known_predictions, average="macro")
    return validation_f1, always_route, rows, selected


def show_validation(validation_f1, always_route, rows, selected):
    print(f"Validation macro F1 (known intents, before review): {validation_f1:.3f}")
    print("Policy                     Routed   Error if routed   Out-of-scope routed")
    for label, row in (
        ("Always route", always_route),
        ("Threshold 0.10", rows[10]),
        (f"Selected threshold {selected.threshold:.2f}", selected),
        ("Threshold 0.50", rows[50]),
    ):
        error = "n/a" if row.routed_error is None else f"{row.routed_error:.1%}"
        print(f"{label:<26} {row.coverage:>6.1%} {error:>17} {row.oos_routing:>21.1%}")


def predict_message(model, message, threshold):
    labels, scores = score_messages(model, [message])
    score = float(scores[0])
    intent = str(labels[0]) if score > threshold else None
    return intent, score


def show_prediction(model, message, threshold):
    intent, score = predict_message(model, message, threshold)
    print(f"Model score: {score:.3f}")
    if intent is not None:
        print(f"Suggested intent: {intent}")
    else:
        print("Needs human review")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message", help="English support message to classify")
    args = parser.parse_args()

    if args.message is not None and not args.message.strip():
        parser.error("--message must contain some text")

    dataset = load_dataset()
    if args.message is None:
        show_dataset(dataset)
    else:
        model = train_baseline(dataset)
        validation_f1, always_route, rows, selected = evaluate_validation(model, dataset)
        show_validation(validation_f1, always_route, rows, selected)
        show_prediction(model, args.message.strip(), selected.threshold)


if __name__ == "__main__":
    main()
