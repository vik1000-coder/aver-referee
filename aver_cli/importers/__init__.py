"""Trace importers: vendor shapes in, ``aver-evidence/v1`` out.

Every importer is a pure function over already-parsed records with a ``detect``
predicate, and every one answers exactly one question per run and guarded action:
did the action fire? Vendor fields never reach the referee. Anything the export
does not say is ``None`` — never a violation (docs/104 §5).

``--format auto`` tries the detectors in the order below, which is the order
docs/104 §5 fixes: the more specific envelope first, the generic shape last.
"""

from __future__ import annotations

from aver_cli.importers import base, generic, langfuse, langsmith, otel, promptfoo
from aver_cli.importers.base import (
    ImportContext,
    ImportResult,
    SideUndecidable,
    UnknownFormat,
)

#: Detector order is part of the contract (docs/104 §5): otel, langsmith,
#: langfuse, promptfoo, jsonl.
FORMATS: tuple[tuple[str, object], ...] = (
    ("otel", otel),
    ("langsmith", langsmith),
    ("langfuse", langfuse),
    ("promptfoo", promptfoo),
    ("jsonl", generic),
)
FORMAT_NAMES: tuple[str, ...] = tuple(name for name, _ in FORMATS) + ("auto",)


def detect_format(payload: object) -> str:
    for name, module in FORMATS:
        if module.detect(payload):  # type: ignore[attr-defined]
            return name
    raise UnknownFormat(
        "no importer recognised this file. Pass --format explicitly, or convert it "
        "to aver-evidence/v1 JSONL (docs/104 §3)."
    )


def read_with(fmt: str, payload: object, context: ImportContext) -> ImportResult:
    for name, module in FORMATS:
        if name == fmt:
            return module.read(payload, context)  # type: ignore[attr-defined]
    raise UnknownFormat(f"unknown format {fmt!r}; known formats are {FORMAT_NAMES}")


__all__ = [
    "FORMATS",
    "FORMAT_NAMES",
    "ImportContext",
    "ImportResult",
    "SideUndecidable",
    "UnknownFormat",
    "base",
    "detect_format",
    "read_with",
]
