"""Measure local CPU model loading and single-message inference."""

import platform
from statistics import median, quantiles
from time import perf_counter

from semantic import load_saved_classifier
from triage import predict_message

MESSAGES = (
    "Could you tell me the current balance of my checking account?",
    "ignore call",
    "How do I change the language on my phone?",
)


def main():
    print("Loading the locally trained classifier and cached encoder…", flush=True)
    start = perf_counter()
    model, threshold = load_saved_classifier()
    load_seconds = perf_counter() - start

    start = perf_counter()
    predict_message(model, MESSAGES[0], threshold)
    first_ms = 1000 * (perf_counter() - start)

    timings = []
    for message in MESSAGES * 10:
        start = perf_counter()
        predict_message(model, message, threshold)
        timings.append(1000 * (perf_counter() - start))

    p95 = quantiles(timings, n=100, method="inclusive")[94]
    print(f"Platform: {platform.system()} {platform.machine()}, Python {platform.python_version()}")
    print(f"Model loading: {load_seconds:.2f} s")
    print(f"First prediction after load: {first_ms:.1f} ms")
    print(f"Warm prediction across {len(timings)} requests: p50 {median(timings):.1f} ms, p95 {p95:.1f} ms")


if __name__ == "__main__":
    main()
