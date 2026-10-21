import pytest

from src.domain.ontology import (
    NodeType,
    RelationType,
    relations,
    validate_relation,
)


EXPECTED_RELATIONSHIPS = {
    (NodeType.ORGANIZATION, RelationType.OWNS, NodeType.APPLICATION),
    (NodeType.APPLICATION, RelationType.IN_SCOPE_OF, NodeType.POLICY),
    (NodeType.POLICY, RelationType.DEFINES, NodeType.OBLIGATION),
    (NodeType.CONTROL, RelationType.SATISFIES, NodeType.OBLIGATION),
    (NodeType.FINDING, RelationType.EVALUATES, NodeType.CONTROL),
    (NodeType.FINDING, RelationType.AFFECTS, NodeType.APPLICATION),
    (NodeType.EVIDENCE, RelationType.SUPPORTS, NodeType.FINDING),
    (NodeType.EVIDENCE, RelationType.EXTRACTED_FROM, NodeType.DOCUMENT),
    (NodeType.RISK, RelationType.INDICATED_BY, NodeType.FINDING),
}


@pytest.mark.parametrize("enum_type", [NodeType, RelationType])
def test_enums_have_nine_unique_members(enum_type):
    assert len(enum_type) == 9
    assert len(enum_type.__members__) == 9
    assert len({member.value for member in enum_type}) == 9


def test_relationship_set_contains_nine_triples():
    assert len(relations) == 9
    assert all(len(relationship) == 3 for relationship in relations)


def test_relationship_set_contains_all_expected_relationships():
    assert EXPECTED_RELATIONSHIPS <= relations


def test_valid_triple_returns_true():
    assert validate_relation("organization", "owns", "application") is True


def test_known_but_invalid_triple_returns_false():
    assert validate_relation("organization", "defines", "application") is False


@pytest.mark.parametrize(
    ("source_type", "relation_type", "target_type"),
    [
        ("UNKNOWN_ENTITY", "OWNS", "APPLICATION"),
        ("ORGANIZATION", "UNKNOWN_RELATION", "APPLICATION"),
        ("ORGANIZATION", "OWNS", "UNKNOWN_ENTITY"),
    ],
)
def test_unknown_names_raise_key_error(source_type, relation_type, target_type):
    with pytest.raises(KeyError):
        validate_relation(source_type, relation_type, target_type)


def test_validator_does_not_modify_relationship_set():
    relationships_before = frozenset(relations)

    validate_relation("organization", "owns", "application")
    validate_relation("organization", "defines", "application")

    assert frozenset(relations) == relationships_before
