import pytest
from pydantic import ValidationError

from src.domain.ontology import (
    ALLOWED_TRIPLES,
    EpistemicMode,
    NodeType,
    NODE_TYPE_DESCRIPTIONS,
    RelationType,
    RELATION_DESCRIPTIONS,
    validate_relation,
)


EXPECTED_TRIPLES = {
    (NodeType.ORGANIZATION, RelationType.OWNS, NodeType.APPLICATION),
    (NodeType.APPLICATION, RelationType.IN_SCOPE_OF, NodeType.POLICY),
    (NodeType.POLICY, RelationType.DEFINES, NodeType.OBLIGATION),
    (NodeType.CONTROL, RelationType.SATISFIES, NodeType.OBLIGATION),
    (NodeType.FINDING, RelationType.EVALUATES, NodeType.CONTROL),
    (NodeType.FINDING, RelationType.AFFECTS, NodeType.APPLICATION),
    (NodeType.EVIDENCE, RelationType.SUPPORTS, NodeType.FINDING),
    (NodeType.EVIDENCE, RelationType.EXTRACTED_FROM, NodeType.CHUNK),
    (NodeType.RISK, RelationType.INDICATED_BY, NodeType.FINDING),
    (NodeType.DOCUMENT, RelationType.HAS_CHUNK, NodeType.CHUNK),
    (NodeType.CHUNK, RelationType.NEXT_CHUNK, NodeType.CHUNK),
}


@pytest.mark.parametrize("triple", sorted(EXPECTED_TRIPLES, key=lambda row: row[1].name))
def test_every_allowed_triple_is_accepted(triple):
    source, relation, target = triple
    assert validate_relation(source.value, relation.value, target.value)


def test_allowed_triple_map_matches_intended_directed_ontology():
    actual = {
        (source, relation, target)
        for relation, endpoints in ALLOWED_TRIPLES.items()
        for source, target in endpoints
    }
    assert actual == EXPECTED_TRIPLES


@pytest.mark.parametrize(
    ("source", "relation", "target"),
    [
        ("ORGANIZATION", "DEFINES", "APPLICATION"),
        ("CHUNK", "HAS_CHUNK", "DOCUMENT"),
        ("DOCUMENT", "EXTRACTED_FROM", "EVIDENCE"),
        ("EVIDENCE", "EXTRACTED_FROM", "DOCUMENT"),
        ("CHUNK", "NEXT_CHUNK", "DOCUMENT"),
        ("DOCUMENT", "NEXT_CHUNK", "CHUNK"),
    ],
)
def test_invalid_triples_and_reversed_directions_are_rejected(source, relation, target):
    assert validate_relation(source, relation, target) is False


@pytest.mark.parametrize(
    ("source", "relation", "target"),
    [
        ("UNKNOWN_ENTITY", "OWNS", "APPLICATION"),
        ("ORGANIZATION", "UNKNOWN_RELATION", "APPLICATION"),
        ("ORGANIZATION", "OWNS", "UNKNOWN_ENTITY"),
        ("", "OWNS", "APPLICATION"),
        (" ORGANIZATION ", "OWNS", "APPLICATION"),
    ],
)
def test_unknown_names_are_rejected(source, relation, target):
    assert validate_relation(source, relation, target) is False


def test_non_string_relation_argument_is_rejected_by_validate_call():
    with pytest.raises(ValidationError):
        validate_relation(NodeType.ORGANIZATION, "OWNS", "APPLICATION")


def test_each_relation_type_has_an_allowed_triple():
    assert set(ALLOWED_TRIPLES) == set(RelationType)
    assert all(ALLOWED_TRIPLES[relation] for relation in RelationType)
    assert set(RELATION_DESCRIPTIONS) == set(RelationType)
    assert all(RELATION_DESCRIPTIONS[relation].strip() for relation in RelationType)


def test_enum_values_are_machine_readable_and_descriptions_are_separate():
    assert NodeType("ORGANIZATION") is NodeType.ORGANIZATION
    assert RelationType("EXTRACTED_FROM") is RelationType.EXTRACTED_FROM
    assert EpistemicMode("VERBATIM_EXTRACTION") is EpistemicMode.VERBATIM_EXTRACTION
    assert set(NODE_TYPE_DESCRIPTIONS) == set(NodeType)
    assert set(RELATION_DESCRIPTIONS) == set(RelationType)


def test_relation_semantics_capture_direction_and_mapping_meaning():
    assert "risk is indicated by the finding" in RELATION_DESCRIPTIONS[RelationType.INDICATED_BY]
    assert "chunk" in RELATION_DESCRIPTIONS[RelationType.EXTRACTED_FROM]
    assert "does not prove" in RELATION_DESCRIPTIONS[RelationType.SATISFIES]


def test_epistemic_modes_do_not_include_entity_resolution_process():
    assert {mode.name for mode in EpistemicMode} == {
        "VERBATIM_EXTRACTION",
        "RULE_DERIVED",
        "LLM_INFERRED",
    }
