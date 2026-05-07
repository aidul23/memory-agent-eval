"""Lightweight, dependency-free text similarity helpers.

The platform deliberately avoids embedding models in its baseline memory
implementations - that keeps the comparison between memory systems centred
on *retrieval policy* rather than on embedding quality. Researchers who want
embeddings can subclass any memory and override ``_score``.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_STOPWORDS = frozenset(
    """a an the and or but of for to in on at with from by is are was were be been
    being this that these those it its as if then so than too very can will would
    should could may might do does did has have had not no""".split()
)


def tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if t.lower() not in _STOPWORDS}


def jaccard(a: str, b: str) -> float:
    """Jaccard similarity between two strings' token sets, in [0, 1]."""
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    union = ta | tb
    return len(inter) / len(union)


def overlap(a: str, b: str) -> float:
    """Asymmetric overlap of ``a`` against ``b`` - useful for short queries."""
    ta, tb = tokenize(a), tokenize(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)
