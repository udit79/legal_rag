NODE_PROPERTIES={
    "ORGANIZATION":{"id","name"},
    "APPLICATION":{"id","name"},
    "POLICY":{"id","title","version"},
    "OBLIGATION":{"id","description","requirement_text"},
    "CONTROL":{"id","name","description"},
    "FINDING":{"id","description","status"},
    "RISK":{"id","name","description"},
    "EVIDENCE":{"id","text","document_id"},
    "DOCUMENT":{"id","filename","content_hash"}
}
def validate_node(node_type: str, properties: dict) -> bool:
    try:
        node_property=NODE_PROPERTIES[node_type.upper()]
    except KeyError:
        raise ValueError(f"Invalid node_type: {node_type}")
    incoming_properties=set(properties.keys())
    property_diff=node_property-incoming_properties
    if property_diff==set():
        for _,v in properties.items():
            if v is None or (isinstance(v,str) and not v.strip()):
                raise ValueError("Property keys cannot be null, empty, or whitespace-only.")
            if not isinstance(v,str):
                raise ValueError("Property keys must be a string")
        return True
    raise ValueError(f"Required node properties are missing: {property_diff}")
