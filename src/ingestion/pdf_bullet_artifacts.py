"""Conservative normalization for known PDF list-marker glyph artifacts."""

import re

from src.ingestion.parser_contracts import (
    CanonicalPage,
    CanonicalTextTransformation,
)

_VISIBLE_BULLET = "➢"
_LIST_MARKERS = (
    "\x01•●‣⁃◦▪▫○■□◆◇"
    "→➔➜➢➤➞➥⇨➡"
    "-–—*+"
)
_LIST_MARKER = re.compile(
    rf"(?m)(?<![^\r\n])([{re.escape(_LIST_MARKERS)}])"
    r"(?=[ \t]*(?:\r?\n|[^\s]))"
)
_RECOMMENDATION_HEADING = re.compile(
    r"(?im)^\s*(?:\d+(?:\.\d+)*\s*)?recommendations?\s*$"
)


def normalize_pdf_bullet_artifacts(page: CanonicalPage) -> CanonicalPage:
    """Normalize known glyphs only at a heading-anchored or repeated list."""
    candidates: dict[int, tuple[int, ...]] = {}
    for index, block in enumerate(page.blocks):
        if block.block_type in {"TEXT", "OTHER", "TABLE"}:
            offsets = tuple(match.start(1) for match in _LIST_MARKER.finditer(block.text))
            if offsets:
                candidates[index] = offsets
    if not candidates:
        return page

    validated: set[tuple[int, int]] = set()
    candidate_indices = sorted(candidates)
    text_indices = [
        index
        for index, block in enumerate(page.blocks)
        if block.block_type in {"TEXT", "OTHER", "TABLE"}
    ]
    text_position = {block_index: index for index, block_index in enumerate(text_indices)}
    for position, block_index in enumerate(candidate_indices):
        block = page.blocks[block_index]
        offsets = candidates[block_index]
        # A nearby explicit heading supports even a one-item recommendation list.
        heading_context = any(
            _RECOMMENDATION_HEADING.search(previous.text)
            and abs(block.source_coordinates.y0 - previous.source_coordinates.y1) <= 48
            for previous_index in text_indices[
                max(0, text_position[block_index] - 2) : text_position[block_index]
            ]
            for previous in (page.blocks[previous_index],)
        )
        if heading_context:
            validated.update((block_index, offset) for offset in offsets)

        # Without a heading, require a repeated, adjacent, aligned marker run.
        run = [block_index]
        cursor = position
        while cursor + 1 < len(candidate_indices):
            current_index = candidate_indices[cursor]
            next_index = candidate_indices[cursor + 1]
            current = page.blocks[current_index]
            following = page.blocks[next_index]
            if (
                text_position[next_index] != text_position[current_index] + 1
                or not 0
                < abs(
                    following.source_coordinates.y0
                    - current.source_coordinates.y0
                )
                <= 60
                or abs(
                    following.source_coordinates.x0
                    - current.source_coordinates.x0
                )
                > 24
            ):
                break
            run.append(next_index)
            cursor += 1
        if len(run) >= 2:
            validated.update(
                (run_index, offset)
                for run_index in run
                for offset in candidates[run_index]
            )

    if not validated:
        return page

    blocks = list(page.blocks)
    for block_index, offsets in candidates.items():
        block = blocks[block_index]
        replacements = [
            offset
            for offset in offsets
            if (block_index, offset) in validated
            and block.text[offset] != _VISIBLE_BULLET
        ]
        if not replacements:
            continue
        block = blocks[block_index]
        raw_text = block.raw_text if block.raw_text is not None else block.text
        derived = list(block.text)
        edits = list(block.text_transformations)
        for offset in replacements:
            original_character = block.text[offset]
            if original_character == _VISIBLE_BULLET:
                continue
            derived[offset] = _VISIBLE_BULLET
            edits.append(
                CanonicalTextTransformation(
                    original_character=original_character,
                    replacement=_VISIBLE_BULLET,
                    source_page=page.page_number,
                    start_offset=offset,
                    end_offset=offset + 1,
                    reason="PDF list-marker glyph normalized after contextual validation",
                )
            )
        blocks[block_index] = block.model_copy(
            update={
                "text": "".join(derived),
                "raw_text": raw_text,
                "text_transformations": tuple(
                    sorted(edits, key=lambda edit: edit.start_offset)
                ),
            }
        )
    return page.model_copy(update={"blocks": tuple(blocks)})
