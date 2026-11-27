"""Compare both models using their validation-selected routing thresholds."""

from semantic import load_saved_classifier
from triage import (
    evaluate_validation,
    load_dataset,
    predict_message,
    routing_metrics,
    score_messages,
    train_baseline,
)


def show_test_result(name, model, dataset, threshold):
    messages, labels = zip(*dataset["test"])
    outside, _ = zip(*dataset["oos_test"])
    predictions, scores = score_messages(model, messages)
    _, outside_scores = score_messages(model, outside)
    result = routing_metrics(threshold, scores, predictions == labels, outside_scores)
    error = "n/a" if result.routed_error is None else f"{result.routed_error:.1%}"

    print(f"\n{name} · threshold {threshold:.2f}")
    print(f"Test messages routed: {result.coverage:.1%}")
    print(f"Test error among routed: {error}")
    print(f"Out-of-scope messages routed: {result.oos_routing:.1%}")
    intent, score = predict_message(model, "ignore call", threshold)
    print(f"'ignore call': {intent or 'Needs human review'} ({score:.3f})")


def main():
    dataset = load_dataset()
    baseline = train_baseline(dataset)
    _, _, _, policy = evaluate_validation(baseline, dataset)
    show_test_result("TF-IDF baseline", baseline, dataset, policy.threshold)

    print("\nLoading the saved MiniLM classifier…", flush=True)
    model, threshold = load_saved_classifier()
    show_test_result("MiniLM sentence vectors", model, dataset, threshold)


if __name__ == "__main__":
    main()
