"""Split long pathology reports into chunks the text encoders can read (77-127 tokens)."""
from __future__ import annotations

import re

SENTENCE_END = re.compile(r"(?<=[.;:!?])\s+")


def chunk_text(text: str, max_words: int) -> list[str]:
    """Greedily pack whole sentences into chunks of at most max_words words."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []
    chunks, cur = [], []
    for sent in SENTENCE_END.split(text):
        words = sent.split()
        while len(words) > max_words:  # one very long "sentence" (tables, lists): hard split
            if cur:
                chunks.append(" ".join(cur))
                cur = []
            chunks.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if len(cur) + len(words) > max_words and cur:
            chunks.append(" ".join(cur))
            cur = []
        cur.extend(words)
    if cur:
        chunks.append(" ".join(cur))
    return chunks
