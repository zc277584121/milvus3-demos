"""Application-side word-level diff used only as a reading aid.

This is not a Milvus return value. It compares the incoming (checked) text
against a retrieved document and produces a list of segments marked removed,
added, or equal. The comparison is token-by-token and best-effort; it is meant
to help a human inspect what changed, not to prove overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

DiffKind = Literal["equal", "removed", "added"]


@dataclass(frozen=True)
class DiffSegment:
    kind: DiffKind
    text: str

    def public_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "text": self.text}


@dataclass(frozen=True)
class DiffView:
    document_id: str
    removed_count: int
    added_count: int
    segments: tuple[DiffSegment, ...]

    def public_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "removed_count": self.removed_count,
            "added_count": self.added_count,
            "segments": [segment.public_dict() for segment in self.segments],
        }


def _tokenize(text: str) -> list[str]:
    return text.split()


def _chunks(tokens: list[str], size: int = 1) -> list[str]:
    return [" ".join(tokens[index : index + size]) for index in range(0, len(tokens), size)]


def compute_diff(
    *,
    incoming_text: str,
    document_id: str,
    document_text: str,
) -> DiffView:
    """Diff the retrieved document against the incoming text.

    ``removed`` marks words present in the retrieved document but absent from
    the incoming text; ``added`` marks words present in the incoming text but
    absent from the retrieved document. This orientation matches the concept
    image: the retrieved (existing) document is the baseline and the incoming
    (checked) text is the changed side.
    """

    if not incoming_text.strip():
        return DiffView(
            document_id=document_id,
            removed_count=0,
            added_count=len(document_text.split()),
            segments=(DiffSegment(kind="added", text=document_text),),
        )
    if not document_text.strip():
        return DiffView(
            document_id=document_id,
            removed_count=len(incoming_text.split()),
            added_count=0,
            segments=(DiffSegment(kind="removed", text=incoming_text),),
        )

    matcher = SequenceMatcher(
        a=_tokenize(document_text),
        b=_tokenize(incoming_text),
        autojunk=False,
    )
    segments: list[DiffSegment] = []
    removed_count = 0
    added_count = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            piece = " ".join(_tokenize(document_text)[i1:i2])
            if piece:
                segments.append(DiffSegment(kind="equal", text=piece))
        elif tag == "delete":
            words = _tokenize(document_text)[i1:i2]
            removed_count += len(words)
            segments.append(DiffSegment(kind="removed", text=" ".join(words)))
        elif tag == "insert":
            words = _tokenize(incoming_text)[j1:j2]
            added_count += len(words)
            segments.append(DiffSegment(kind="added", text=" ".join(words)))
        else:  # replace
            removed_words = _tokenize(document_text)[i1:i2]
            added_words = _tokenize(incoming_text)[j1:j2]
            removed_count += len(removed_words)
            added_count += len(added_words)
            if removed_words:
                segments.append(DiffSegment(kind="removed", text=" ".join(removed_words)))
            if added_words:
                segments.append(DiffSegment(kind="added", text=" ".join(added_words)))
    return DiffView(
        document_id=document_id,
        removed_count=removed_count,
        added_count=added_count,
        segments=tuple(segments),
    )
