from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from app.utils.text import normalize_whitespace, remove_non_alphanumeric, simple_tokenize

stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()


def preprocess_text(text: str) -> str:
    lowered = text.casefold()
    cleaned = remove_non_alphanumeric(lowered)
    normalized = normalize_whitespace(cleaned)
    without_stopwords = stopword_remover.remove(normalized)
    tokenized = simple_tokenize(without_stopwords)
    stemmed_tokens = [stemmer.stem(token) for token in tokenized if token]

    return " ".join(token for token in stemmed_tokens if token).strip()
