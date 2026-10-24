import pytest

from src.domain.graph_schema import NODE_PROPERTIES, validate_node
from src.domain.ontology import NodeType


def test_node_properties_defines_all_node_types():
    assert set(NODE_PROPERTIES) == {node_type.value for node_type in NodeType}


def test_every_node_type_requires_id():
    assert all("id" in properties for properties in NODE_PROPERTIES.values())


def test_evidence_requires_text_and_chunk_id():
    assert {"text", "chunk_id"} <= NODE_PROPERTIES["EVIDENCE"]


def test_chunk_accepts_zero_based_integer_index():
    required = {"id", "text", "document_id", "chunk_index"}
    assert required <= NODE_PROPERTIES["CHUNK"]
    assert validate_node(
        "CHUNK",
        {"id": "chunk-1", "text": "A passage", "document_id": "doc-1", "chunk_index": 0},
    ) is True


def test_chunk_rejects_negative_index():
    with pytest.raises(ValueError, match="non-negative"):
        validate_node(
            "CHUNK", {"id": "chunk-1", "text": "A passage", "document_id": "doc-1", "chunk_index": -1}
        )


def test_chunk_rejects_boolean_index():
    with pytest.raises(ValueError, match="must be an integer"):
        validate_node(
            "CHUNK", {"id": "chunk-1", "text": "A passage", "document_id": "doc-1", "chunk_index": True}
        )


def test_chunk_missing_required_property_is_rejected():
    with pytest.raises(ValueError, match="chunk_index"):
        validate_node(
            "CHUNK", {"id": "chunk-1", "text": "A passage", "document_id": "doc-1"}
        )


def test_valid_organization_is_accepted():
    assert validate_node("ORGANIZATION", {"id": "org-1", "name": "Acme"}) is True


def test_evidence_missing_chunk_id_is_rejected():
    with pytest.raises(ValueError, match="chunk_id"):
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


def test_unknown_node_properties_are_rejected():
    with pytest.raises(ValueError, match="Unknown node properties"):
        validate_node("ORGANIZATION", {"id": "org-1", "name": "Acme", "department": "Compliance"})


def test_integer_text_is_rejected():
    with pytest.raises(ValueError, match="Property text must be a string"):
        validate_node(
            "CHUNK", {"id": "chunk-1", "text": 123, "document_id": "doc-1", "chunk_index": 0}
        )
