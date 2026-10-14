# Sentiment Analysis with DistilBERT

A one-week project on how transformers work, built around one task:
classifying the sentiment of tweets as **negative**, **neutral** or
**positive**. Every day adds a layer, from tokenization to a fine-tuned
model with a web demo.

- **Data:** [`tweet_eval` / sentiment](https://huggingface.co/datasets/cardiffnlp/tweet_eval)
  (SemEval-2017): 45,615 train, 2,000 validation and 12,284 test tweets.
- **Model:** [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased),
  6 layers, 66M parameters.

## Results

Both models were trained on the same 80% of the train split and evaluated
on the remaining 20% (9,123 tweets) and on the official test split, which
covers other topics (US politics of 2016-2017).

| Model | Macro F1, 20% split | Accuracy, 20% split | Macro F1, official test |
|---|---|---|---|
| Always "neutral" | 0.208 | 0.453 | 0.217 |
| Baseline: frozen DistilBERT + logistic regression | 0.635 | 0.663 | 0.648 |
| **Fine-tuned DistilBERT** (3 epochs) | **0.694** | **0.712** | **0.677** |

Improvement of fine-tuning over the baseline: **+9.3%** macro F1 on the
20% split, **+4.5%** on the official test split.

Macro F1 is the main metric: the classes are imbalanced (16% / 45% / 39%
in train), and always predicting neutral already gives 45% accuracy.

What the error analysis found (`results/error_analysis.txt`):

- 93% of the errors involve the neutral class, the noisiest class of the
  labels; negative and positive are rarely confused with each other.
- Sarcasm and negation are hard; negation raises the error rate from 27%
  to 37%.
- The model learned a topic bias: neutral tweets about Islam, the IRA or
  ISIS tend to be labelled negative.
- The model is overconfident: at 98% confidence it is still wrong about
  one time in ten.

## Project structure

```
app.py                      Gradio demo (entry point)
src/itg_transformers/
    config.py               model name, paths, labels, MAX_LENGTH = 64
    data.py                 download, cleaning and loading of the dataset
    eda.py                  helpers of the data analysis
    tokenization.py         day 1: tokenizer, tokenize_texts, explain_tokenization
    embeddings.py           day 2: load_model, get_embeddings, similarity
    attention.py            day 3: attention heatmaps and statistics
    finetuning.py           day 5: SentimentDataset, train_epoch, evaluate
    inference.py            day 6: predict_fine_tuned, predict_baseline
    app.py                  day 7: Gradio interface
notebooks/
    00_eda.ipynb            data analysis and text cleaning
    01_tokenization.ipynb   ... one notebook per day ...
    07_error_analysis.ipynb
results/                    metrics, comparison and error analysis (text files)
tests/                      unit tests, no downloads needed
models/                     trained models (not in git, created by notebooks 05-06)
data/                       dataset and cached embeddings (not in git)
figures/                    saved plots (not in git)
```

| Day | Topic | Notebook |
|---|---|---|
| 0 | Data analysis, cleaning of export artifacts | `00_eda.ipynb` |
| 1 | Tokenization | `01_tokenization.ipynb` |
| 2 | Hidden states and CLS embeddings | `02_hidden_states.ipynb` |
| 3 | Attention | `03_attention.ipynb` |
| 4 | Baseline without training the transformer | `04_baseline.ipynb` |
| 5 | Fine-tuning | `05_finetuning.ipynb` |
| 6 | Inference and model comparison | `06_comparison.ipynb` |
| 7 | Error analysis | `07_error_analysis.ipynb` |

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync          # create .venv and install the dependencies
make data        # download the dataset to data/
```

Then run the notebooks in order. `05_finetuning.ipynb` trains the model
(about 17 minutes on an Apple GPU) and saves it to
`models/fine_tuned_model/`; `06_comparison.ipynb` saves the baseline to
`models/baseline_model.pkl`. Both reuse the saved files on later runs.

## Demo

```bash
make demo        # or: uv run python app.py
```

Open http://127.0.0.1:7860, type a text and get the predicted sentiment
with the probabilities of all three classes. The demo needs the trained
model in `models/fine_tuned_model/`.

## Using the model in code

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "models/fine_tuned_model"
)
tokenizer = AutoTokenizer.from_pretrained("models/fine_tuned_model")
model.eval()

inputs = tokenizer(
    "I love this phone!", return_tensors="pt", truncation=True, max_length=64
)
with torch.no_grad():
    probs = model(**inputs).logits.softmax(dim=-1)[0]

pred = int(probs.argmax())
print(model.config.id2label[pred], f"{probs[pred]:.3f}")  # positive 0.995
```

Or with the project's helpers, which also clean the text and batch the
input: `itg_transformers.inference.predict_fine_tuned`.

## Development

```bash
make check       # ruff lint + format check, ty type check, pytest
make fix         # apply ruff fixes and formatting
```

## Requirements

- Python 3.13+
- PyTorch, transformers, datasets, scikit-learn, pandas
- gradio (demo), matplotlib and seaborn (plots)

All versions are pinned in `uv.lock`.
