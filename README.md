# Fake Review Detection

Detects fake / computer-generated product reviews using classical machine
learning on TF-IDF features, served through a FastAPI backend, and surfaced
live on Amazon product pages through a Chrome extension.

> Originally built as a Winter of Code (WoC 7.0) project. Rebuilt with a
> proper code structure, experiment tracking, a live API, and a browser
> extension — this README documents *what* was built, *why* each decision
> was made, and *what problems came up along the way*, so it's useful both
> as a personal reference and as a learning resource for anyone reading the
> code later.

---

## Table of contents

1. [Problem statement](#problem-statement)
2. [Live demo](#live-demo)
3. [Architecture](#architecture)
4. [Dataset](#dataset)
5. [Preprocessing — and why there are two pipelines](#preprocessing--and-why-there-are-two-pipelines)
6. [Models — what was tried and why](#models--what-was-tried-and-why)
7. [Results](#results)
8. [Experiment tracking (MLflow)](#experiment-tracking-mlflow)
9. [Transformer model (DistilBERT)](#transformer-model-distilbert)
10. [API](#api)
11. [Chrome extension](#chrome-extension)
12. [Challenges faced](#challenges-faced)
13. [Project structure](#project-structure)
14. [Setup](#setup)
15. [Future improvements](#future-improvements)
16. [Tech stack](#tech-stack)

---

## Problem statement

Online shopping platforms are full of reviews written specifically to
manipulate buying decisions — some generated in bulk, paid for, or
written by bots, rather than by people who used the product. These are
often called "computer-generated" (CG) reviews, as opposed to genuine,
"original" (OR) reviews from real buyers.

The goal here is a binary text classification problem:

> Given the text of a review, predict whether it is **real (OR)** or
> **fake / computer-generated (CG)**.

This matters for two audiences: shoppers (who want to trust what they
read before buying) and platforms (who want to detect coordinated fake
review campaigns at scale).

## Live demo

The Chrome extension auto-scans Amazon product review pages and tags each
review inline:

- ✓ Likely Real (green badge)
- ⚠ Likely Fake (red badge, with confidence %)

It can also be used manually via the extension popup — paste any review
text and get an instant prediction, useful for testing text that isn't on
a live Amazon page.

## Architecture

```
┌─────────────────────────────┐
│   Chrome Extension           │
│   (runs on amazon.com/.in)   │
│                               │
│  content.js  →  scans page,  │
│                 finds review │
│                 text, injects│
│                 badges       │
│  popup.js    →  manual check │
└───────────────┬───────────────┘
                │  POST /predict  { text: "..." }
                ▼
┌─────────────────────────────┐
│   FastAPI backend            │
│   (api/main.py)              │
│   localhost:8000              │
└───────────────┬───────────────┘
                │
                ▼
┌─────────────────────────────┐
│   Inference (src/predict.py) │
│   1. clean + lemmatize text  │
│   2. TF-IDF vectorize        │
│   3. run through best model  │
│      (Linear SVM)            │
└─────────────────────────────┘
```

Training happens offline (`src/train_classical.py`) and produces two
artifacts the API loads at request time: a fitted `TfidfVectorizer` and the
best-performing trained model.

## Dataset

- `data/fakeReviewData.csv` — 40,432 Amazon reviews, perfectly balanced:
  20,216 labeled `CG` (computer-generated) and 20,216 labeled `OR`
  (original/real).
- Columns: `category` (product category), `rating` (1-5 stars), `label`
  (CG/OR), `text_` (review text).
- No missing values.

A balanced dataset like this is convenient — it means plain **accuracy**
is a meaningful metric here (with an imbalanced dataset, accuracy can be
misleading — e.g. a model that always predicts "real" would score 95% on
a dataset that's 95% real reviews, while being useless).

## Preprocessing — and why there are two pipelines

Different model families expect text in different shapes, so this project
uses **two separate preprocessing pipelines** rather than forcing one
approach to serve both:

### 1. Classical pipeline (`src/preprocessing/classical.py`)

Used for all TF-IDF based models (Logistic Regression, SVM, Random Forest,
Naive Bayes, XGBoost). Steps:

1. Convert emojis to text (`:)` → `smiling_face`) — emoji use patterns
   can differ between fake and real reviews.
2. Expand contractions (`don't` → `do not`).
3. Lowercase, strip URLs/HTML/punctuation.
4. Tokenize.
5. Remove stopwords — **except negation words** (`not`, `no`, `never`...).
6. **Negation tagging**: the word immediately after a negation gets
   prefixed, e.g. `"not good"` → `not NEG_good`. Without this, standard
   stopword removal would delete "not" entirely, and `"not good"` becomes
   just `"good"` — flipping the sentiment the model sees. This one design
   choice meaningfully affects how well the model separates genuine
   criticism from generic praise.
7. Lemmatize (`running` → `run`).

The output is a plain space-joined string, ready for `TfidfVectorizer`.

### 2. Transformer-lite pipeline (`src/preprocessing/transformer_lite.py`)

Used for DistilBERT. Deliberately minimal — only strips URLs/HTML and
extra whitespace. **No lemmatization, no stopword removal.**

This isn't an oversight — it's the correct approach for transformers.
Subword tokenizers (WordPiece/BPE) are pretrained on natural, unedited
text. Feeding them stopword-stripped, lemmatized text actually *removes
information* the pretrained model already knows how to use (word order,
function words, inflections) and can hurt performance rather than help it.
Preprocessing for classical ML and preprocessing for transformers are
solving different problems: TF-IDF needs help reducing noise and
dimensionality; a transformer needs the sentence to look like real
language.

## Models — what was tried and why

Five classical models were trained on identical TF-IDF features, so the
comparison isolates model choice as the only variable:

| Model | Why it was included |
|---|---|
| **Logistic Regression** | Simple, fast, interpretable baseline. Coefficients can be inspected to see which words push toward "fake". |
| **Linear SVM** | Tends to do well on high-dimensional sparse text data (TF-IDF vectors are exactly that) — historically a strong baseline for text classification specifically. |
| **Multinomial Naive Bayes** | The classic, extremely fast text-classification baseline. Assumes word independence (technically wrong for language, but works surprisingly well for bag-of-words text). Useful as a "how much are the other models actually buying us" reference point. |
| **Random Forest** | Non-linear, captures feature interactions the linear models can't. Included to see whether non-linearity actually helps on this kind of data (spoiler: it didn't — see Results). |
| **XGBoost** | Gradient boosting, generally a strong performer on tabular/structured data. Included for the same reason as Random Forest — testing whether a more powerful non-linear model beats the simpler linear ones here. |

A sixth model, **DistilBERT** (a transformer), was added separately as a
stretch comparison — see [Transformer model](#transformer-model-distilbert).

**Why F1-score, not accuracy, decided the winner:** the dataset is
balanced, so accuracy is usable, but F1 (harmonic mean of precision and
recall) was used as the primary comparison metric because it penalizes
models that trade off precision for recall or vice versa. Precision here
means "of the reviews we flagged fake, how many actually were" (false
positives = real reviews wrongly flagged, which erodes user trust in the
tool). Recall means "of the actual fake reviews, how many did we catch"
(false negatives = fake reviews slipping through). Neither error is free,
so F1 was preferred over optimizing either alone.

## Results

Trained on TF-IDF features (10,000 max features, uni+bigrams, min_df=3),
80/20 train/test split, stratified by label:

| Model | Accuracy | Precision | Recall | F1 | Train time |
|---|---|---|---|---|---|
| **Linear SVM** ⭐ | 89.43% | 89.36% | 89.51% | **89.44%** | 0.2s |
| Logistic Regression | 88.82% | 88.69% | 88.99% | 88.84% | 0.2s |
| Naive Bayes | 86.26% | 87.13% | 85.09% | 86.10% | 0.01s |
| XGBoost | 83.57% | 86.32% | 79.77% | 82.92% | 41.6s |
| Random Forest | 80.87% | 84.57% | 75.51% | 79.79% | 6.9s |

Full hyperparameters and confusion matrices for every run are saved in
`models/results.json`.

**What this shows:** the two linear models (SVM, Logistic Regression)
clearly beat the tree-based models here. This tracks with a well-known
pattern in NLP: **TF-IDF features are high-dimensional and sparse**
(thousands of mostly-zero columns, one per word/bigram), and linear
models tend to handle that shape of data better than tree-based models,
which split on one feature at a time and struggle to find useful splits
across thousands of sparse, mostly-empty columns. Random Forest and
XGBoost need many more trees/deeper trees to compensate — at a real cost
in training time (Random Forest also had to convert the sparse matrix to
dense internally, which is where most of that extra cost comes from) —
without actually catching up to the linear models' accuracy on this task.
This isn't a universal rule (tree models often win on dense, tabular,
non-text data) — it's specific to sparse bag-of-words-style text features.

## Experiment tracking (MLflow)

Every training run — hyperparameters, all four metrics, training time,
and (where supported) the model artifact itself — is logged to MLflow
automatically by `src/train_classical.py`. View it with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

This gives a sortable comparison table and per-run detail pages, useful
for justifying model choice with actual logged numbers rather than a
one-off print statement.

## Transformer model (DistilBERT)

`src/train_transformer.py` fine-tunes `distilbert-base-uncased` as a
sixth model, to compare a pretrained transformer against the from-scratch
TF-IDF models.

**Status:** written and ready to run, but not executed in the environment
this project was originally built in (no GPU available there) — it should
be run and its results added to this table before being presented as a
completed comparison.

Configured to be CPU-runnable by default (6,000-sample subset, 2 epochs,
~20-40 min on a laptop CPU) since fine-tuning the full 32k-review training
set without a GPU would take considerably longer. Bump
`TRAIN_SAMPLE_SIZE`/`EPOCHS` in the file if running with a GPU (e.g. in
Google Colab's free tier).

```bash
pip install -r requirements-transformer.txt
cd src && python train_transformer.py
```

## API

FastAPI backend (`api/main.py`), run with:

```bash
uvicorn api.main:app --reload --port 8000
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check |
| `/predict` | POST | `{"text": "..."}` → single prediction |
| `/predict/batch` | POST | `{"reviews": ["...", "..."]}` → list of predictions |

Example:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Amazing product, changed my life, 5 stars!!!"}'
```

Response:
```json
{"label": "fake", "is_fake": true, "confidence": 0.78, "model_used": "linear_svm"}
```

## Chrome extension

A Manifest V3 extension (`extension/`) with two ways to use it:

1. **Auto-scan** (`content.js`) — runs on Amazon product review pages,
   finds each review via Amazon's `[data-hook='review-body']` attribute,
   sends its text to the local API, and injects a colored badge next to
   it. Uses a `MutationObserver` (debounced) to catch reviews that load
   in dynamically as you scroll or paginate.
2. **Manual check** (`popup.html`/`popup.js`) — paste any text into the
   extension popup and get an instant prediction, independent of what
   page you're on.

Currently scoped to `amazon.com`/`amazon.in` — both the page-matching
rules (`manifest.json`) and the review selector (`content.js`) are
Amazon-specific and would need a different selector to support another
site (e.g. Flipkart, Yelp), since every site marks up its reviews
differently.

## Challenges faced

Documenting these because they were the actual non-obvious parts of
building this — the kind of thing worth being able to explain in an
interview:

- **Chrome's Private Network Access policy blocked the extension.**
  Modern Chrome blocks `https://amazon.com` from calling `localhost`
  by default, as a security measure against malicious sites probing a
  user's local network. Fixed by adding an
  `Access-Control-Allow-Private-Network: true` response header via
  FastAPI middleware — CORS's `allow_origins=["*"]` alone wasn't enough.

- **Random Forest hit memory limits during training.** On sparse
  TF-IDF input, scikit-learn's Random Forest implementation converts
  data to a dense array internally. At the original planned feature
  count (15,000 TF-IDF features × 40k rows), that dense array alone
  would be several GB — larger than the training machine's available
  RAM. Fixed by reducing `max_features` in the vectorizer and capping
  tree count/depth, trading a small amount of potential accuracy for a
  training run that actually completes.

- **A naive stopword-removal step was silently breaking negation.**
  Standard stopword lists include "not", "no", "never" — removing them
  turns `"not good"` into `"good"`, which is backwards for a task where
  understanding what a reviewer is actually saying matters. Solved with
  the negation-tagging step described above.

- **The content script initially flagged fragments, not reviews.** An
  early version of the review selector matched every `<span>` inside a
  review block, not just the review itself — Amazon's markup nests many
  spans per review for styling. Each nested span got sent to the API and
  badged separately, so a single review ended up with a dozen+ duplicate
  badges. Fixed by narrowing the selector to the one stable
  `[data-hook='review-body']` element per review, and debouncing the
  `MutationObserver` so badge insertion (itself a DOM mutation) doesn't
  keep re-triggering the scan.

## Project structure

```
data/                          raw dataset
src/
  preprocessing/
    classical.py                preprocessing for the 5 TF-IDF models
    transformer_lite.py         preprocessing for DistilBERT
  train_classical.py            trains + MLflow-tracks all 5 classical models
  train_transformer.py          fine-tunes DistilBERT (optional, see above)
  predict.py                    inference using the best classical model
  predict_transformer.py        inference using DistilBERT (optional)
models/                        saved vectorizer, best model, results.json
api/
  main.py                       FastAPI app
extension/                     Chrome extension (Manifest V3)
notebooks_legacy/              original WoC checkpoint notebooks, kept for history
mlflow.db                      MLflow experiment tracking database
```

## Setup

```bash
pip install -r requirements.txt
cd src && python train_classical.py      # retrains all 5 models, ~1 min
cd .. && uvicorn api.main:app --reload --port 8000
```

Load the extension: `chrome://extensions` → enable Developer Mode →
**Load unpacked** → select the `extension/` folder. Then browse any
Amazon product review page with the API running.

## Future improvements

- Finish running and logging the DistilBERT comparison
- Extend the extension to more review sites beyond Amazon
- Publish the extension to the Chrome Web Store
- Deploy the API (currently localhost-only) so the extension works without
  running a local server
- Add SHAP/coefficient-based explainability to the popup ("flagged
  because of these words") for user trust

## Tech stack

Python, scikit-learn, XGBoost, MLflow, NLTK, FastAPI, Chrome Extension
(Manifest V3, vanilla JS). Optional: HuggingFace Transformers, PyTorch.