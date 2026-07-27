"""
Classical preprocessing pipeline — used for TF-IDF + traditional ML models
(Logistic Regression, SVM, Random Forest, Naive Bayes, XGBoost).

Design choice: negation handling is added on top of the original project's
pipeline. Plain stopword removal strips words like "not", "no", "never" —
which flips the meaning of reviews such as "not good" -> "good". We tag the
word following a negation so the model can still learn from it.
"""

import re
import string
import emoji
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

STOPWORDS = set(stopwords.words("english"))
NEGATIONS = {"not", "no", "never", "n't", "cannot", "cant", "none", "nobody", "nothing"}
# keep negation words even though they're normally stopwords
STOPWORDS = STOPWORDS - NEGATIONS

LEMMATIZER = WordNetLemmatizer()

URL_RE = re.compile(r"http\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
MULTISPACE_RE = re.compile(r"\s+")


def demojize(text: str) -> str:
    return emoji.demojize(text, delimiters=(" ", " "))


def expand_contractions(text: str) -> str:
    return contractions.fix(text)


def clean_text(text: str) -> str:
    text = str(text)
    text = demojize(text)
    text = expand_contractions(text)
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def tag_negations(tokens: list[str]) -> list[str]:
    """Prefix the word after a negation with NEG_ so it becomes its own
    feature, e.g. ['not', 'good'] -> ['not', 'NEG_good']."""
    out = []
    negate_next = False
    for tok in tokens:
        if negate_next and tok not in NEGATIONS:
            out.append(f"NEG_{tok}")
            negate_next = False
        else:
            out.append(tok)
        if tok in NEGATIONS:
            negate_next = True
    return out


def preprocess(text: str, use_negation_tagging: bool = True) -> str:
    """Full classical preprocessing: clean -> tokenize -> stopword removal
    -> negation tagging -> lemmatize. Returns a space-joined string ready
    for TfidfVectorizer."""
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    if use_negation_tagging:
        tokens = tag_negations(tokens)

    final = []
    for tok in tokens:
        if tok.startswith("NEG_"):
            final.append("NEG_" + LEMMATIZER.lemmatize(tok[4:]))
        else:
            final.append(LEMMATIZER.lemmatize(tok))
    return " ".join(final)


if __name__ == "__main__":
    sample = "This product is NOT good at all!! Waste of money :( http://example.com"
    print(preprocess(sample))
