# BLP-1 deterministic chunking

BLP-1 (Block-Anchored Ladder Packer) is the implemented chunking strategy for
canonical ingestion output. The implementation currently uses algorithm
version **1.2.0**. Its entry point is `chunk_document()` in
`src/ingestion/chunking.py`; it returns a `ChunkingResult` containing a
`ChunkedDocument` and a `ChunkingReport`.

## Pipeline

```text
CanonicalDocument
  -> page routing and block eligibility
  -> lossless M4 segmentation into source-relative atoms
  -> M5 packing for TEXT and OTHER
  -> M6 isolated packing for TABLE and IMAGE
  -> chunks with source ranges and a per-page report
```

The pipeline is deterministic for the same canonical input and configuration.
It does not parse PDFs itself or write chunks to Neo4j.

## Configuration and policy identity

`ChunkingConfig` is frozen, rejects unknown fields, and validates behavior
settings. Defaults include:

| Setting | Default | Effect |
| --- | --- | --- |
| `max_chunk_chars` | `1500` | Hard maximum measured in Python string characters, including synthetic separators. |
| `synthetic_separator` | `"\n\n"` | Added only between different source blocks packed into one ordinary-text chunk. |
| `marker_breaks` | `True` | Enables provision boundaries and M5 strong-marker chunk closing. |
| `marker_break_min_pct` | `25` | M5 strong-marker closing threshold as a percentage of the character budget. |
| `grapheme_backoff_chars` | `32` | Maximum backward search from a hard cut for a supported safe boundary. |
| `include_image_text` | `True` | Makes nonblank canonical IMAGE text eligible for isolated chunking. |

The configuration digest is SHA-256 over canonical JSON with sorted keys and
fixed separators. It includes all settings, including provision-marker
patterns, abbreviations, separators, thresholds, and grapheme backoff. The
policy identifier is `5c-<semantic-version>+<first-12-hex-of-config-digest>`;
the current semantic version is `1.2.0`. An implementation change that
changes chunk output under the same settings requires a semantic-version
change; an optimization that preserves output does not.

## Page routing and eligibility

Pages are handled independently and remain in canonical order. `BLANK`,
`VECTOR_ONLY`, and `FAILED` parser states remain visible in the report. A failed
page does not stop successful pages from being chunked. Failed-page details and
warnings are copied from the parser contracts.

For parsed pages, nonblank `TEXT`, `OTHER`, and `TABLE` blocks are eligible.
Nonblank `IMAGE` blocks are eligible when `include_image_text=True`.
`VECTOR` is never text-eligible. Empty and whitespace-only source blocks, image
text disabled by configuration, and vectors are recorded as skipped blocks;
eligible source text is never trimmed or normalized.

## M3 and M4: lossless atoms

`src/ingestion/segmentation.py` provides the independent boundary primitives:

1. Blank-line paragraph boundaries.
2. Configured line-initial provision markers, including weak list labels.
3. Deterministic sentence heuristics with abbreviations, initials, and
   plausible sentence starts.
4. CRLF, LF, and lone-CR line boundaries.
5. Whitespace-run word boundaries.
6. A bounded grapheme-safe hard cut.

M4 processes an oversized source block with the recursive ladder in that
order. A fitting block becomes one atom. Each split chooses the first tier
with an interior boundary, then its child spans continue below that tier in
the ladder. This preserves the ladder's precedence while avoiding rescanning
tiers already exhausted for a span. The final hard-cut tier guarantees
progress. Atoms contain half-open `[start, end)` offsets into the original
block; their text is always the original source slice. Unsafe grapheme
fallbacks are recorded in the report diagnostics.

Sentence detection is heuristic. Grapheme handling deliberately covers common
combining marks, variation selectors, emoji modifiers and tags, ZWJ/ZWNJ
sequences, regional-indicator pairs, CRLF, and Devanagari virama conjuncts; it
is not a complete implementation of Unicode UAX #29. Bounded fallback may
therefore split a cluster, and those cases are counted.

## M5: TEXT and OTHER packing

Only `TEXT` and `OTHER` blocks share the ordinary-text packing group. Packing
never crosses a page boundary. Before appending an atom, a strong provision
marker at a block start or line start can close a nonempty chunk when its
current length is at least
`floor(max_chunk_chars * marker_break_min_pct / 100)`. Strong markers include
numbered clauses and legal headings. Weak list labels such as `(a)`, `(iv)`,
and `(1)` are segmentation boundaries but do not trigger this closing rule.

The proposed length includes `synthetic_separator` when the next atom comes
from a different block. If it exceeds the hard limit, the current chunk is
closed and the atom is retried. Consecutive atoms from the same block receive
no separator; adjacent source ranges in that chunk are merged into one
`ChunkSource` range.

## M6: TABLE and IMAGE groups

Each `TABLE` block is packed independently and never combines with ordinary
text or another table. A table at or below the limit becomes one chunk. An
oversized table is split into rows using CRLF, LF, and lone CR terminators;
each terminator stays attached to its preceding row. Complete rows are greedily
packed while they fit.

An oversized row uses this fallback order:

1. Column gaps: tabs, runs of at least two spaces, and space-padded pipe
   delimiters.
2. Word boundaries.
3. The bounded grapheme-safe hard cut.

All fragments remain contiguous slices of the original table text. No table
header is repeated. `oversized_table_rows` counts oversized rows that required
fallback, and `unsafe_grapheme_cuts` counts degraded hard cuts.

An eligible `IMAGE` text block is isolated from all other block types. Fitting
text becomes one chunk; oversized text uses M4 segmentation and is packed only
with atoms from that same image block. With image text disabled, or with no
usable image text, no image chunk is emitted and the reason remains in the
skipped-block report.

## Provenance and reports

`ChunkSource` uses half-open source offsets and the original block's full text
length. Chunk text consists only of source slices and, for M5 ordinary text,
configured separators. Separators have no provenance range. Chunk indices are
zero-based and contiguous; source order follows page number and canonical
reading order. Each page is packed separately, so a chunk cannot reference
multiple pages.

`ChunkingReport` includes parser statuses, page outcomes, failures, warnings,
skipped blocks, configuration identity, and diagnostics. Overall outcomes are:

| Outcome | Meaning |
| --- | --- |
| `COMPLETED` | Eligible content was chunked and no page failed. |
| `NO_CONTENT` | No chunks were produced and no page failed, including blank or wholly ineligible input. |
| `PARTIAL` | Some chunks were produced alongside a failed page or explicitly unsupported eligible content. |
| `FAILED` | No chunks were produced and at least one page failed. |

Warnings alone do not change the overall outcome. A page with chunks is
`CHUNKED`; blank, vector-only, failed, image-without-text, and no-eligible-text
pages retain their specific skip outcomes.

## Verification

Focused tests live in:

- `tests/test_ingestion_chunk_contracts.py` for source and chunk models.
- `tests/test_ingestion_chunking_models.py` for configuration, digest, policy,
  and report contracts.
- `tests/test_ingestion_chunking_routing.py` for page routing and eligibility.
- `tests/test_ingestion_segmentation.py` for M3/M4 boundaries and atom spans.
- `tests/test_ingestion_chunking_packing.py` for M5/M6 packing, exact coverage,
  separators, isolation, and reporting.

Parser structure is an input limitation: chunking preserves canonical text
and boundaries, but cannot recover table structure or reading order that the
parser did not preserve. Parser integration fixtures are under
`tests/fixtures/ingestion_corpus/`.
