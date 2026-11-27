"""Streamlit interface for the support message classifier."""

import logging

import streamlit as st

from semantic import MODEL_PATH, load_saved_classifier
from triage import predict_message


@st.cache_resource(show_spinner=False)
def load_classifier():
    return load_saved_classifier()


def main():
    st.set_page_config(page_title="Support Triage")
    st.title("Support Triage")
    st.write("Check how an English support message would be routed.")

    with st.form("triage"):
        message = st.text_area(
            "Support message",
            placeholder="Could you tell me the current balance of my checking account?",
            max_chars=1000,
        )
        submitted = st.form_submit_button("Analyse message", type="primary")

    if not submitted:
        return

    message = message.strip()
    if not message:
        st.warning("Enter a support message to analyse.")
        return

    if not MODEL_PATH.is_file():
        st.error("Train the classifier with python semantic.py before using the app.")
        return

    try:
        with st.spinner("Analysing message…"):
            model, threshold = load_classifier()
            intent, score = predict_message(model, message, threshold)
    except (OSError, ValueError):
        logging.exception("Could not analyse the support message")
        st.error("The classifier could not process this request. Try again.")
        return

    if intent is None:
        st.warning("Needs human review")
    else:
        st.success(f"Suggested category: {intent.replace('_', ' ')}")
    st.caption(f"Model score: {score:.3f} · Routing threshold: {threshold:.2f}")


if __name__ == "__main__":
    main()
