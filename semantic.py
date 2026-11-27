"""Train and load the MiniLM classifier used by the web app."""

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from triage import evaluate_validation, load_dataset, train_baseline

ENCODER_ID = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "8b3219a92973c328a8e22fadcfa821b5dc75636a"
MODEL_PATH = Path(__file__).resolve().parent / "data" / "minilm-triage.joblib"


def make_semantic_model(classifier, *, show_progress=False):
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(
        ENCODER_ID,
        revision=ENCODER_REVISION,
        device="cpu",
        model_kwargs={"use_safetensors": True},
    )
    encoder_lock = Lock()

    def encode(messages):
        # Streamlit sessions share this encoder and its tokenizer.
        with encoder_lock:
            return encoder.encode(
                list(messages),
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            )

    return make_pipeline(FunctionTransformer(encode), classifier)


def train_semantic(dataset, target_coverage, *, show_progress=False):
    classifier = LogisticRegression(max_iter=500)
    model = make_semantic_model(classifier, show_progress=show_progress)
    messages, labels = zip(*dataset["train"])
    model.fit(messages, labels)
    _, _, candidates, _ = evaluate_validation(model, dataset)
    policy = min(candidates, key=lambda row: abs(row.coverage - target_coverage))
    return model, policy.threshold


def load_saved_classifier():
    """Load the classifier generated locally by this training script."""
    classifier, threshold = joblib.load(MODEL_PATH)
    return make_semantic_model(classifier), threshold


def main():
    dataset = load_dataset()
    print("Training the TF-IDF baseline for validation coverage…", flush=True)
    baseline = train_baseline(dataset)
    _, _, _, policy = evaluate_validation(baseline, dataset)

    print("Training the MiniLM classifier and selecting its threshold…", flush=True)
    model, threshold = train_semantic(dataset, policy.coverage, show_progress=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=MODEL_PATH.parent) as folder:
        temporary_path = Path(folder) / MODEL_PATH.name
        joblib.dump((model[-1], threshold), temporary_path)
        temporary_path.replace(MODEL_PATH)

    print(f"Saved classifier to {MODEL_PATH}")
    print(f"Routing threshold: {threshold:.2f}")


if __name__ == "__main__":
    main()
