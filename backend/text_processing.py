import re


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Splits text into overlapping chunks of roughly `chunk_size` characters,
    trying to break on sentence/paragraph boundaries.

    Fixes vs previous version:
      - Actually splits the document into paragraphs (previous version treated
        the whole document as a single "paragraph", causing O(n^2)-ish behavior
        on large documents and thousands of debug prints).
      - Removed all per-chunk/per-paragraph print() calls (huge slowdown on
        large documents).
      - Guarantees forward progress on every iteration, so it can never raise
        a RuntimeError / crash the ingestion pipeline (previous version could
        raise and bubble up as a 500).
    """
    text = normalize_text(text)
    if not text:
        return []

    # Real paragraph split (was missing before — this is the main perf fix)
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not raw_paragraphs:
        raw_paragraphs = [text]

    # Merge short heading-like lines with the following paragraph
    merged = []
    i = 0
    while i < len(raw_paragraphs):
        current = raw_paragraphs[i]
        is_heading = (
            len(current) < 80
            and (
                current.startswith("#")
                or current.endswith(":")
                or re.fullmatch(r"[A-Z][A-Za-z0-9\s\-()]+", current)
            )
        )
        if is_heading and i + 1 < len(raw_paragraphs):
            merged.append(current + "\n\n" + raw_paragraphs[i + 1])
            i += 2
        else:
            merged.append(current)
            i += 1

    paragraphs = merged
    chunks: list[str] = []
    current = ""

    def hard_split(block: str) -> list[str]:
        """Split an oversized block into <= chunk_size pieces, always making progress."""
        pieces = []
        pos = 0
        n = len(block)
        while pos < n:
            end = min(pos + chunk_size, n)
            window = block[pos:end]

            if end < n:
                split_at = window.rfind(". ")
                if split_at == -1:
                    split_at = window.rfind("\n")
                # Only trust the sentence/newline split if it's not absurdly early
                if split_at != -1 and split_at >= chunk_size // 2:
                    end = pos + split_at + 1

            piece = block[pos:end].strip()
            if piece:
                pieces.append(piece)

            next_pos = end - overlap
            # Guarantee forward progress no matter what
            if next_pos <= pos:
                next_pos = end
            pos = next_pos

        return pieces

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= chunk_size:
            current = f"{current}\n\n{paragraph}".strip() if current else paragraph
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(paragraph) > chunk_size:
            chunks.extend(hard_split(paragraph))
        else:
            current = paragraph

    if current:
        chunks.append(current)

    # De-duplicate while preserving order (defensive; normal inputs won't hit this)
    seen = set()
    unique = []
    for chunk in chunks:
        key = chunk.strip()
        if key and key not in seen:
            unique.append(key)
            seen.add(key)

    return unique


"""
NOTE for app.py (Streamlit) — entities tab currently does:

    for index, entity in enumerate(entities):
        cols[index % 3].success(entity)

`entities` is a dict[str, list[str]] like {"equipment_tags": ["P-101"], ...}.
Iterating it only yields the keys, never the values. Fix:

    for index, (category, values) in enumerate(entities.items()):
        cols[index % 3].success(f"{category}: {', '.join(values)}")
"""


def excerpt(text: str, max_chars: int = 500) -> str:
    cleaned = normalize_text(text)
    if len(cleaned) <= max_chars:
        return cleaned

    cut = cleaned.rfind(".", 0, max_chars)

    if cut == -1:
        cut = max_chars

    return cleaned[:cut].strip() + "..."