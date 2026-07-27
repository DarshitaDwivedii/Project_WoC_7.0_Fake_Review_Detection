"""
Transformer-side preprocessing — deliberately minimal.

Unlike the classical pipeline, we do NOT lemmatize or remove stopwords here.
Transformer tokenizers (WordPiece/BPE) are trained on natural, un-stripped
text — removing stopwords or lemmatizing actually *hurts* performance
because it breaks the word patterns the pretrained model already knows.
We only strip things that are pure noise: URLs, HTML, and excess whitespace.
"""

import re

URL_RE = re.compile(r"http\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
MULTISPACE_RE = re.compile(r"\s+")


def preprocess(text: str) -> str:
    text = str(text)
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


if __name__ == "__main__":
    sample = "This product is NOT good at all!! Waste of money :( http://example.com"
    print(preprocess(sample))
