from Sastrawi.Dictionary.ArrayDictionary import ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemover import StopWordRemover
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from app.utils.text import normalize_whitespace, remove_non_alphanumeric, simple_tokenize

stemmer = StemmerFactory().create_stemmer()

# Kata negasi ini penting untuk analisis sentimen — jangan dihapus sebagai stopword.
# Sastrawi secara default menghapus "tidak", "belum", "tanpa" sehingga
# "tidak sesuai" menjadi "sesuai" (positif palsu). Dengan mengeluarkan kata-kata
# ini dari daftar stopword, konteks negasi tetap terjaga.
_NEGATION_WORDS_TO_KEEP = {"tidak", "belum", "tanpa"}
_factory = StopWordRemoverFactory()
_custom_stopwords = [w for w in _factory.get_stop_words() if w not in _NEGATION_WORDS_TO_KEEP]
stopword_remover = StopWordRemover(ArrayDictionary(_custom_stopwords))


def preprocess_text(text: str) -> str:
    lowered = text.casefold()
    cleaned = remove_non_alphanumeric(lowered)
    normalized = normalize_whitespace(cleaned)
    without_stopwords = stopword_remover.remove(normalized)
    tokenized = simple_tokenize(without_stopwords)
    stemmed_tokens = [stemmer.stem(token) for token in tokenized if token]

    return " ".join(token for token in stemmed_tokens if token).strip()
