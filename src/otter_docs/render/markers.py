"""Marker-based document injection.

A generated document interleaves human prose with regenerated
sections. Each section is fenced:

    <!-- BEGIN GENERATED:system_overview -->
    ...renderer output, replaced every run...
    <!-- END GENERATED:system_overview -->

`inject` replaces only the text *between* a matching BEGIN/END pair;
everything else in the document — including human prose between
different sections — is preserved byte-for-byte. If the section's
markers don't exist, the new block is appended at the end so a
hand-written doc gets bootstrapped incrementally.

`bootstrap_document` writes a fresh starter doc with every renderer's
markers already placed, so `otter-docs init` produces a working
SYSTEM.md with zero hand-placement.
"""

from __future__ import annotations

import re

BEGIN = "<!-- BEGIN GENERATED:{name} -->"
END = "<!-- END GENERATED:{name} -->"


def _block_re(name: str) -> re.Pattern[str]:
    return re.compile(
        re.escape(BEGIN.format(name=name))
        + r".*?"
        + re.escape(END.format(name=name)),
        re.DOTALL,
    )


def inject(document: str, *, name: str, body: str) -> str:
    """Return `document` with section `name`'s generated block set to `body`.

    Idempotent: running twice with the same body yields the same
    document. Human prose outside the BEGIN/END pair is untouched.
    Missing markers → the block is appended (with a leading blank
    line if the doc is non-empty).
    """
    begin = BEGIN.format(name=name)
    end = END.format(name=name)
    block = f"{begin}\n{body}\n{end}"
    pattern = _block_re(name)
    if pattern.search(document):
        return pattern.sub(lambda _m: block, document, count=1)
    # Append. Keep exactly one blank line between existing content and
    # the new block; don't add leading whitespace to an empty doc.
    if document and not document.endswith("\n"):
        document += "\n"
    sep = "\n" if document else ""
    return f"{document}{sep}{block}\n"


def bootstrap_document(*, title: str, sections: list[str]) -> str:
    """Produce a starter document with all section markers in place.

    The body of each section is a placeholder line that the first
    real render replaces. Human-editable prose lives outside the
    markers — we seed an editable intro paragraph.
    """
    parts: list[str] = [f"# {title}", ""]
    parts.append(
        "_This document is maintained by otter-docs. Edit freely **outside** "
        "the `BEGIN GENERATED` / `END GENERATED` markers — content inside "
        "them is regenerated on every run._"
    )
    parts.append("")
    for name in sections:
        parts.append(f"## {name.replace('_', ' ').title()}")
        parts.append("")
        parts.append(BEGIN.format(name=name))
        parts.append("_not yet rendered — run otter-docs render_")
        parts.append(END.format(name=name))
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def sections_in(document: str) -> list[str]:
    """List the generated-section names present in a document, in order."""
    return re.findall(r"<!-- BEGIN GENERATED:([A-Za-z0-9_]+) -->", document)
