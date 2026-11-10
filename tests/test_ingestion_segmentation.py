import pytest

import src.ingestion.segmentation as segmentation_module
from src.ingestion.chunking import ChunkingConfig
from src.ingestion.segmentation import (
    BoundaryTier,
    HardCutResult,
    grapheme_safe_hard_cut,
    line_boundaries,
    paragraph_boundaries,
    provision_marker_boundaries,
    sentence_boundaries,
    segment_oversized_text,
    word_boundaries,
)


def assert_lossless(text, boundaries):
    assert boundaries == tuple(sorted(set(boundaries)))
    assert all(0 < position < len(text) for position in boundaries)
    points = (0, *boundaries, len(text))
    assert "".join(text[start:end] for start, end in zip(points, points[1:])) == text


@pytest.mark.parametrize(
    "detector",
    [
        paragraph_boundaries,
        provision_marker_boundaries,
        sentence_boundaries,
        line_boundaries,
        word_boundaries,
    ],
)
@pytest.mark.parametrize("text", ["", "x", "no candidate boundary", "a\nb\n\nc"])
def test_boundary_detectors_return_valid_lossless_boundaries(detector, text):
    boundaries = detector(text)
    assert boundaries == detector(text)
    assert_lossless(text, boundaries)


def test_paragraph_boundaries_require_a_blank_line_and_keep_delimiter():
    single_line_break = "First line\ncontinued paragraph"
    text = "First paragraph\n \t\nSecond paragraph\r\n\r\nThird paragraph"

    assert paragraph_boundaries(single_line_break) == ()
    boundaries = paragraph_boundaries(text)
    assert len(boundaries) == 2
    assert text[: boundaries[0]].endswith("\n \t\n")
    assert text[: boundaries[1]].endswith("\r\n\r\n")
    assert_lossless(text, boundaries)


def test_provision_marker_boundaries_match_supported_markers_only_at_line_start():
    text = (
        "Preamble\n1. First\n1) Second\n1.1 Third\n1.1.1 Fourth\n"
        "Article IV\nSection 5\nAnnex A\nAnnexure A\n2023. The court\n"
        "A mention of Article IV in a sentence"
    )
    config = ChunkingConfig()

    boundaries = provision_marker_boundaries(text, config)

    matched_pieces = [text[position:].splitlines()[0].strip() for position in boundaries]
    assert matched_pieces == [
        "1. First",
        "1) Second",
        "1.1 Third",
        "1.1.1 Fourth",
        "Article IV",
        "Section 5",
        "Annex A",
        "Annexure A",
    ]
    assert_lossless(text, boundaries)


def test_provision_marker_detector_is_deterministic():
    text = "Lead\nArticle IV\n1.1 Term"
    assert provision_marker_boundaries(text) == provision_marker_boundaries(text)
    assert provision_marker_boundaries(
        text, ChunkingConfig(marker_breaks=False)
    ) == ()


def test_provision_markers_exclude_indented_year_headings_but_keep_clause_numbers():
    text = (
        "2023. The court considered the evidence.\n"
        "    2023. The court considered the evidence.\n"
        "1. The first provision.\n"
        "    1. The indented provision."
    )

    boundaries = provision_marker_boundaries(text)
    matched_pieces = [text[position:].splitlines()[0].strip() for position in boundaries]

    assert matched_pieces == ["1. The first provision.", "1. The indented provision."]
    assert_lossless(text, boundaries)


def test_sentence_boundaries_cover_abbreviations_initials_numbers_quotes_and_danda():
    text = (
        "Dr. Smith met J. A. Doe. He said, \u201cProceed!\u201d Then left. "
        "The ratio is 1.5. Next, clause 1.2 applies. समाप्त। अगला भाग। Done."
    )

    boundaries = sentence_boundaries(text)
    pieces = [text[start:end] for start, end in zip((0, *boundaries), (*boundaries, len(text)))]

    assert len(boundaries) == 7
    assert pieces[0].startswith("Dr. Smith met J. A. Doe.")
    assert "\u201cProceed!\u201d " in pieces[1]
    assert pieces[2].startswith("Then left.")
    assert pieces[3].startswith("The ratio is 1.5.")
    assert pieces[4].startswith("Next, clause 1.2 applies.")
    assert "समाप्त। " in pieces[5]
    assert pieces[6].startswith("अगला भाग।")
    assert_lossless(text, boundaries)


def test_sentence_boundaries_recognize_question_marks():
    text = "Was it? Yes."
    boundaries = sentence_boundaries(text)
    assert len(boundaries) == 1
    assert text[: boundaries[0]] == "Was it? "
    assert_lossless(text, boundaries)


def test_sentence_boundaries_do_not_treat_clause_numbers_as_sentences():
    text = "1. Heading. Next sentence."
    boundaries = sentence_boundaries(text)

    assert len(boundaries) == 1
    assert text[: boundaries[0]] == "1. Heading. "
    assert_lossless(text, boundaries)


def test_sentence_boundaries_handle_repeated_initials_without_prefix_scans():
    phrase = "A. B. Smith signed. "
    text = phrase * 1200

    boundaries = sentence_boundaries(text)

    assert len(boundaries) == 1199
    pieces = [text[start:end] for start, end in zip((0, *boundaries), (*boundaries, len(text)))]
    assert all(piece.endswith("signed. ") for piece in pieces[:-1])
    assert all("A. B. Smith" in piece for piece in pieces)
    assert_lossless(text, boundaries)


def test_sentence_boundaries_contextualize_terminal_abbreviations():
    title = "Dr. Smith signed the agreement. The filing followed."
    terminal = "The company operates in the U.S. The agreement starts today."
    internal = "The U.S. company filed the report."

    title_boundaries = sentence_boundaries(title)
    terminal_boundaries = sentence_boundaries(terminal)
    internal_boundaries = sentence_boundaries(internal)

    assert len(title_boundaries) == 1
    assert title[: title_boundaries[0]] == "Dr. Smith signed the agreement. "
    assert len(terminal_boundaries) == 1
    assert terminal[: terminal_boundaries[0]] == "The company operates in the U.S. "
    assert internal_boundaries == ()
    assert_lossless(title, title_boundaries)
    assert_lossless(terminal, terminal_boundaries)
    assert_lossless(internal, internal_boundaries)


def test_sentence_boundaries_keep_legal_citations_and_split_common_words():
    text = (
        "State of Punjab v. Mohinder Singh. "
        "Civil Appeal No. 1234 was filed. "
        "Art. 5 applies. The terms. The claims."
    )

    boundaries = sentence_boundaries(text)
    pieces = [
        text[start:end]
        for start, end in zip((0, *boundaries), (*boundaries, len(text)))
    ]

    assert pieces == [
        "State of Punjab v. Mohinder Singh. ",
        "Civil Appeal No. 1234 was filed. ",
        "Art. 5 applies. ",
        "The terms. ",
        "The claims.",
    ]
    assert_lossless(text, boundaries)


def test_line_boundaries_recognize_only_crlf_lf_and_lone_cr():
    text = "one\r\ntwo\nthree\rfour\u2028five"

    boundaries = line_boundaries(text)

    pieces = [text[start:end] for start, end in zip((0, *boundaries), (*boundaries, len(text)))]
    assert pieces == ["one\r\n", "two\n", "three\r", "four\u2028five"]
    assert_lossless(text, boundaries)


def test_word_boundaries_keep_whitespace_runs_in_the_previous_piece():
    text = "first  \tsecond\n\nthird"

    boundaries = word_boundaries(text)

    assert [text[:boundaries[0]], text[boundaries[0] : boundaries[1]]] == [
        "first  \t",
        "second\n\n",
    ]
    assert_lossless(text, boundaries)


@pytest.mark.parametrize(
    ("text", "max_chars", "expected_position"),
    [
        ("e\u0301x", 2, 2),
        ("👍🏽x", 2, 2),
        ("❤️x", 2, 2),
        ("👩\u200d🚀x", 3, 3),
        ("🇮🇳x", 2, 2),
        ("a\r\nb", 2, 1),
    ],
)
def test_hard_cut_preserves_common_grapheme_sequences(
    text, max_chars, expected_position
):
    result = grapheme_safe_hard_cut(text, 0, max_chars)

    assert isinstance(result, HardCutResult)
    assert result.position == expected_position
    assert result.position <= max_chars
    assert result.is_grapheme_safe
    assert result.backoff_chars == max_chars - expected_position
    assert text[: result.position] + text[result.position :] == text


@pytest.mark.parametrize(
    "conjunct",
    ["\u0915\u094d\u0937", "\u0915\u094d\u0930"],
)
def test_hard_cut_keeps_devanagari_virama_conjuncts_together(conjunct):
    text = "x" + conjunct + "z"

    result = grapheme_safe_hard_cut(text, 0, 3)

    assert result.position == 1
    assert result.is_grapheme_safe
    assert result.backoff_chars == 2
    assert text[: result.position] + text[result.position :] == text


def test_hard_cut_reports_unsafe_when_limit_falls_inside_indic_conjunct():
    text = "\u0915\u094d\u0937x"

    result = grapheme_safe_hard_cut(text, 0, 2)

    assert result.position == 2
    assert not result.is_grapheme_safe
    assert text[: result.position] + text[result.position :] == text


def test_hard_cut_keeps_devanagari_zwj_and_zwnj_conjunct_sequences_together():
    for joiner in ("\u200d", "\u200c"):
        conjunct = f"\u0915\u094d{joiner}\u0937"
        text = "x" + conjunct + "z"

        result = grapheme_safe_hard_cut(text, 0, 3)

        assert result.position == 1
        assert result.is_grapheme_safe


def test_combining_mark_at_hard_limit_is_safe_or_reported_unsafe():
    text = "e\u0301x"
    safe = grapheme_safe_hard_cut(text, 0, 2)
    unsafe = grapheme_safe_hard_cut(
        text, 0, 1, config=ChunkingConfig(grapheme_backoff_chars=1)
    )

    assert safe.position == 2 and safe.is_grapheme_safe
    assert unsafe.position == 1 and not unsafe.is_grapheme_safe


def test_hard_cut_records_when_bounded_backoff_cannot_find_safe_boundary():
    text = "a" + "\u0301" * 20 + "b"

    result = grapheme_safe_hard_cut(
        text, 0, 8, config=ChunkingConfig(grapheme_backoff_chars=4)
    )

    assert result.position == 8
    assert result.position <= 8
    assert not result.is_grapheme_safe
    assert result.backoff_chars == 0
    assert text[: result.position] + text[result.position :] == text


def test_hard_cut_is_deterministic_and_handles_empty_and_single_character_text():
    assert grapheme_safe_hard_cut("", 0, 1) == HardCutResult(0, True, 0)
    assert grapheme_safe_hard_cut("x", 0, 1) == HardCutResult(1, True, 0)
    first = grapheme_safe_hard_cut("text\u0301", 0, 3)
    assert first == grapheme_safe_hard_cut("text\u0301", 0, 3)


def test_hard_cut_uses_absolute_source_offsets():
    text = "a e\u0301x"
    result = grapheme_safe_hard_cut(text, 2, 2)

    assert result.position == 4
    assert text[2 : result.position] == "e\u0301"
    assert result.is_grapheme_safe


def test_segmentation_returns_single_atom_when_text_fits():
    text = "small source block"

    result = segment_oversized_text(text, ChunkingConfig(max_chunk_chars=32))

    assert [(atom.start, atom.end, atom.created_by) for atom in result.atoms] == [
        (0, len(text), None)
    ]
    assert result.atom_texts == (text,)
    assert result.splits == ()


def test_segmentation_returns_no_atoms_for_empty_input():
    result = segment_oversized_text("")
    assert result.atoms == ()
    assert result.splits == ()
    assert result.atom_texts == ()


def test_segmentation_exact_budget_input_remains_one_atom():
    text = "x" * 16
    result = segment_oversized_text(text, ChunkingConfig(max_chunk_chars=16))

    assert [(atom.start, atom.end) for atom in result.atoms] == [(0, 16)]
    assert result.splits == ()
    assert result.atom_texts == (text,)


def test_segmentation_uses_recursive_paragraph_then_hard_cut_and_exact_spans():
    text = "A" * 10 + "\n\n" + "B" * 37
    result = segment_oversized_text(text, ChunkingConfig(max_chunk_chars=16))

    assert result.splits[0].tier is BoundaryTier.PARAGRAPH
    assert any(split.tier is BoundaryTier.GRAPHEME_HARD_CUT for split in result.splits)
    assert all(atom.end - atom.start <= 16 for atom in result.atoms)
    assert result.atom_texts == tuple(text[atom.start : atom.end] for atom in result.atoms)
    assert "".join(result.atom_texts) == text
    assert all(left.end == right.start for left, right in zip(result.atoms, result.atoms[1:]))


def test_segmentation_uses_provision_markers_including_weak_list_labels():
    text = "P" * 20 + "\n(a) alpha\n(iv) roman\n(1) numeric"
    result = segment_oversized_text(text, ChunkingConfig(max_chunk_chars=16))

    assert any(split.tier is BoundaryTier.PROVISION_MARKER for split in result.splits)
    assert all(atom.end - atom.start <= 16 for atom in result.atoms)
    assert "".join(result.atom_texts) == text


@pytest.mark.parametrize(
    ("text", "expected_tier"),
    [
        (
            "First sentence. Still first paragraph.\n\nSecond sentence. More text.",
            BoundaryTier.PARAGRAPH,
        ),
        (
            "Introductory sentence continues here.\nSection 5 Governing rule.\n"
            "Another provision follows here.",
            BoundaryTier.PROVISION_MARKER,
        ),
    ],
)
def test_segmentation_uses_higher_priority_tier_when_boundaries_compete(
    text, expected_tier
):
    result = segment_oversized_text(text, ChunkingConfig(max_chunk_chars=16))

    assert result.splits[0].tier is expected_tier
    assert "".join(result.atom_texts) == text


def test_segmentation_marker_tier_respects_marker_breaks_false():
    text = "P" * 20 + "\nSection 5 Applies"
    result = segment_oversized_text(
        text, ChunkingConfig(max_chunk_chars=16, marker_breaks=False)
    )

    assert all(split.tier is not BoundaryTier.PROVISION_MARKER for split in result.splits)
    assert "".join(result.atom_texts) == text
    assert all(atom.end - atom.start <= 16 for atom in result.atoms)


def test_segmentation_follows_sentence_line_and_word_tiers_before_hard_cut():
    sentence_text = "First sentence. Second sentence. Third sentence."
    sentence_result = segment_oversized_text(
        sentence_text, ChunkingConfig(max_chunk_chars=16)
    )
    assert sentence_result.splits[0].tier is BoundaryTier.SENTENCE

    line_text = "A" * 12 + "\n" + "B" * 12 + "\n" + "C" * 12
    line_result = segment_oversized_text(line_text, ChunkingConfig(max_chunk_chars=16))
    assert line_result.splits[0].tier is BoundaryTier.LINE

    word_text = "A" * 10 + " " + "B" * 10 + " " + "C" * 10
    word_result = segment_oversized_text(word_text, ChunkingConfig(max_chunk_chars=16))
    assert word_result.splits[0].tier is BoundaryTier.WORD
    for result, original in (
        (sentence_result, sentence_text),
        (line_result, line_text),
        (word_result, word_text),
    ):
        assert "".join(result.atom_texts) == original
        assert all(atom.end - atom.start <= 16 for atom in result.atoms)


def test_segmentation_uses_hard_cut_and_reports_unsafe_progress_when_required():
    text = "A" * 15 + "e\u0301" + "B" * 30
    result = segment_oversized_text(
        text,
        ChunkingConfig(max_chunk_chars=16, grapheme_backoff_chars=0),
    )

    assert any(
        split.tier is BoundaryTier.GRAPHEME_HARD_CUT
        and split.is_grapheme_safe is False
        for split in result.splits
    )
    assert all(atom.end - atom.start <= 16 for atom in result.atoms)
    assert "".join(result.atom_texts) == text


def test_segmentation_is_deterministic_for_same_source_and_configuration():
    text = "Intro.\nSection 5 Terms.\n\nMore text without punctuation " + "x" * 30
    config = ChunkingConfig(max_chunk_chars=16)

    assert segment_oversized_text(text, config) == segment_oversized_text(text, config)


def test_long_unbroken_fallback_precomputes_grapheme_boundaries_once(monkeypatch):
    original = segmentation_module._grapheme_boundaries
    calls = 0

    def counted(text):
        nonlocal calls
        calls += 1
        return original(text)

    monkeypatch.setattr(segmentation_module, "_grapheme_boundaries", counted)
    text = "x" * 30_000
    result = segment_oversized_text(text, ChunkingConfig(max_chunk_chars=16))

    assert calls == 1
    assert len(result.atoms) == 1_875
    assert all(atom.end - atom.start <= 16 for atom in result.atoms)
    assert "".join(result.atom_texts) == text
