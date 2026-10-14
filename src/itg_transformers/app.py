from functools import cache

import gradio as gr
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from itg_transformers.config import FINE_TUNED_DIR, LABELS
from itg_transformers.data import clean_text
from itg_transformers.embeddings import pick_device
from itg_transformers.finetuning import load_classifier
from itg_transformers.inference import Prediction, predict_fine_tuned
from itg_transformers.tokenization import load_tokenizer

EXAMPLES = [
    "This movie was absolutely fantastic!",
    "Terrible, waste of my time.",
    "The meeting is moved to Monday.",
    "Thanks manager for putting me on the schedule for Sunday",
]


@cache
def load_fine_tuned() -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """The day 5 model, loaded once on the first request."""
    model = load_classifier(str(FINE_TUNED_DIR))
    model.to(pick_device())  # ty: ignore[invalid-argument-type]
    return model, load_tokenizer(str(FINE_TUNED_DIR))


def format_prediction(prediction: Prediction) -> str:
    label = LABELS[prediction["prediction"]].capitalize()
    lines = [f"Prediction: {label}", "", "Probabilities:"]
    for index, probability in enumerate(prediction["probabilities"]):
        lines.append(f"{LABELS[index].capitalize()}: {probability * 100:.2f}%")
    return "\n".join(lines)


def predict_sentiment(text: str) -> str:
    """Gradio callback: the sentiment of one text as readable text."""
    if not text.strip():
        return "Enter a text to analyse."
    model, tokenizer = load_fine_tuned()
    [prediction] = predict_fine_tuned(text, model, tokenizer, clean_text)
    return format_prediction(prediction)


def build_demo() -> gr.Interface:
    return gr.Interface(
        fn=predict_sentiment,
        inputs=gr.Textbox(lines=3, placeholder="Enter a text to analyse..."),
        outputs=gr.Textbox(label="Result"),
        title="Sentiment Analysis with DistilBERT",
        description=(
            "DistilBERT fine-tuned on tweets (tweet_eval sentiment). Enter "
            "a text and the model tells whether it is negative, neutral or "
            "positive. The model is overconfident: even at 98% it is wrong "
            "about one time in ten."
        ),
        examples=EXAMPLES,
        flagging_mode="never",
    )


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
