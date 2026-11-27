"""Evaluate the validation-selected routing policy on the official test splits."""

from sklearn.metrics import f1_score

from triage import (
    evaluate_validation,
    load_dataset,
    routing_metrics,
    score_messages,
    train_baseline,
)


def normalise(message):
    return " ".join(message.casefold().split())


def main():
    dataset = load_dataset()
    model = train_baseline(dataset)
    _, _, _, selected = evaluate_validation(model, dataset)
    threshold = selected.threshold

    known_messages, known_labels = zip(*dataset["test"])
    oos_messages, _ = zip(*dataset["oos_test"])
    known_predictions, known_scores = score_messages(model, known_messages)
    oos_predictions, oos_scores = score_messages(model, oos_messages)
    known_correct = known_predictions == known_labels

    macro_f1 = f1_score(known_labels, known_predictions, average="macro")
    print(
        f"Official test split: {len(known_messages):,} known "
        f"and {len(oos_messages):,} out of scope"
    )
    print(f"Known-intent macro F1 before review: {macro_f1:.3f}")
    print("Policy                     Routed   Error if routed   Out-of-scope routed")
    for label, cutoff in (("Always route", -1), (f"Threshold {threshold:.2f}", threshold)):
        row = routing_metrics(cutoff, known_scores, known_correct, oos_scores)
        error = "n/a" if row.routed_error is None else f"{row.routed_error:.1%}"
        print(f"{label:<26} {row.coverage:>6.1%} {error:>17} {row.oos_routing:>21.1%}")

    known_misroutes = (known_scores > threshold) & ~known_correct
    oos_misroutes = oos_scores > threshold
    routed_count = (known_scores > threshold).sum() + oos_misroutes.sum()
    print(f"Threshold routed {routed_count:,} messages")
    print(
        f"Routing errors: {known_misroutes.sum()} wrong known labels "
        f"and {oos_misroutes.sum()} out-of-scope messages"
    )
    if known_misroutes.any():
        misroute_indices = known_misroutes.nonzero()[0]
        index = misroute_indices[known_scores[misroute_indices].argmax()]
        print(f"Highest-scoring known error: {known_messages[index]!r}")
        print(
            f"  Expected {known_labels[index]}, routed to "
            f"{known_predictions[index]} ({known_scores[index]:.3f})"
        )
    if oos_misroutes.any():
        index = oos_scores.argmax()
        print(f"Highest-scoring out-of-scope error: {oos_messages[index]!r}")
        print(f"  Routed to {oos_predictions[index]} ({oos_scores[index]:.3f})")

    seen_messages = {
        normalise(message)
        for split in ("train", "val", "oos_val")
        for message, _ in dataset[split]
    }
    overlap = sum(
        normalise(message) in seen_messages
        for messages in (known_messages, oos_messages)
        for message in messages
    )
    print(f"Test rows with exact normalised text seen in training or validation: {overlap}")


if __name__ == "__main__":
    main()
