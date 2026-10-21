import pytest

from src.domain.graph_schema import NODE_PROPERTIES, validate_node
from src.domain.ontology import NodeType


def test_node_properties_defines_all_node_types():
    assert set(NODE_PROPERTIES) == {node_type.name for node_type in NodeType}


def test_every_node_type_requires_id():
    assert all("id" in properties for properties in NODE_PROPERTIES.values())


def test_evidence_requires_text_and_document_id():
    assert {"text", "document_id"} <= NODE_PROPERTIES["EVIDENCE"]


def test_valid_organization_is_accepted():
    assert validate_node("ORGANIZATION", {"id": "org-1", "name": "Acme"}) is True


def test_evidence_missing_document_id_is_rejected():
    with pytest.raises(ValueError):
        validate_node("EVIDENCE", {"id": "evidence-1", "text": "Source passage"})


def test_unknown_node_type_raises_value_error():
    with pytest.raises(ValueError):
        validate_node("UNKNOWN", {"id": "node-1"})


def test_empty_id_raises_value_error():
    with pytest.raises(ValueError):
        validate_node("ORGANIZATION", {"id": "", "name": "Acme"})


def test_missing_name_raises_value_error():
    with pytest.raises(ValueError):
        validate_node("ORGANIZATION", {"id": "org-1"})


def test_whitespace_only_name_raises_value_error():
    with pytest.raises(ValueError):
        validate_node("ORGANIZATION", {"id": "org-1", "name": "   "})


def test_valid_node_accepts_additional_department_property():
    assert validate_node(
        "ORGANIZATION",
        {"id": "org-1", "name": "Acme", "department": "Compliance"},
    ) is True
