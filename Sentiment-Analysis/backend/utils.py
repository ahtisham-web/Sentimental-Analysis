"""
Shared text-processing utilities.
Used by BOTH training/train_model.py and backend/main.py so preprocessing
is guaranteed identical between training and inference (no train/serve skew).
"""
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

for pkg in ["stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(pkg if "/" in pkg else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOPWORDS = set(stopwords.words("english"))
# Negation words are deliberately kept: "not good" must not collapse to "good"
NEGATIONS = {"not", "no", "nor", "never", "n't"}
STOPWORDS -= NEGATIONS

LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()


def clean_text(text: str) -> str:
    """Data cleaning: lowercase, strip URLs/mentions/numbers/punctuation, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)      # remove URLs
    text = re.sub(r"@\w+", " ", text)                 # remove mentions
    text = re.sub(r"\d+", " ", text)                  # remove numbers
    text = re.sub(r"[^a-z\s']", " ", text)             # remove special characters
    text = text.translate(str.maketrans("", "", string.punctuation.replace("'", "")))
    text = re.sub(r"\s+", " ", text).strip()           # collapse extra spaces
    return text


def tokenize(text: str) -> list:
    return word_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def stem_tokens(tokens: list) -> list:
    """Stemming (Porter). Faster but cruder than lemmatization -- kept here for
    comparison/demonstration since the project spec calls it out explicitly."""
    return [STEMMER.stem(t) for t in tokens]


def lemmatize_tokens(tokens: list) -> list:
    """Lemmatization -- used in the production pipeline (more linguistically accurate)."""
    return [LEMMATIZER.lemmatize(t) for t in tokens]


def preprocess_text(text: str, method: str = "lemmatize") -> str:
    """
    Full pipeline: clean -> tokenize -> remove stopwords -> stem/lemmatize.
    method: "lemmatize" (default, used by the trained model) or "stem".
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens) if method == "stem" else lemmatize_tokens(tokens)
    return " ".join(tokens)
