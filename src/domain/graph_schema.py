from typing import Final
NODE_PROPERTY_TYPES: Final[dict[str, dict[str, type]]] = {
    "ORGANIZATION": {"id": str, "name": str},
    "APPLICATION": {"id": str, "name": str},
    "POLICY": {"id": str, "title": str, "version": str},
    "OBLIGATION": {"id": str, "description": str, "requirement_text": str},
    "CONTROL": {"id": str, "name": str, "description": str},
    "FINDING": {"id": str, "description": str, "status": str},
    "RISK": {"id": str, "name": str, "description": str},
    "EVIDENCE": {"id": str, "text": str, "chunk_id": str},
    "DOCUMENT": {"id": str, "filename": str, "content_hash": str},
    "CHUNK": {"id": str, "text": str, "document_id": str, "chunk_index": int},
}

# Required property names are the keys of each type map.
NODE_PROPERTIES: Final[dict[str, set[str]]] = {
    node_type: set(property_types)
    for node_type, property_types in NODE_PROPERTY_TYPES.items()
}


def validate_node(node_type: str, properties: dict) -> bool:
    try:
        property_types = NODE_PROPERTY_TYPES[node_type.upper()]
    except (KeyError, AttributeError):
        raise ValueError(f"Invalid node_type: {node_type}") from None

    supplied = set(properties)
    missing = set(property_types) - supplied
    if missing:
        raise ValueError(f"Required node properties are missing: {missing}")

    unknown = supplied - set(property_types)
    if unknown:
        raise ValueError(f"Unknown node properties: {unknown}")

    for name, value in properties.items():
        expected_type = property_types[name]
        if expected_type is int:
            if type(value) is not int:
                raise ValueError(f"Property {name} must be an integer")
            if value < 0:
                raise ValueError("Property chunk_index must be non-negative")
        elif not isinstance(value, expected_type):
            raise ValueError(f"Property {name} must be a string")
        elif not value.strip():
            raise ValueError("String properties cannot be empty or whitespace-only")
    return True
