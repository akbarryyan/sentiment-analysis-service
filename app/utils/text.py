import re


NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9\s]+")
MULTISPACE_PATTERN = re.compile(r"\s+")


def remove_non_alphanumeric(text: str) -> str:
    return NON_ALPHANUMERIC_PATTERN.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    return MULTISPACE_PATTERN.sub(" ", text).strip()


def simple_tokenize(text: str) -> list[str]:
    if not text.strip():
        return []

    return [token for token in text.split(" ") if token]
