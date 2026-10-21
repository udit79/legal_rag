from enum import Enum
from pydantic import StrictStr,validate_call
class NodeType(Enum):
    ORGANIZATION="The organization being assessed"
    APPLICATION="A system or application"
    POLICY="An internal policy or external requirement document"
    OBLIGATION="A requirement that must be satisfied"
    CONTROL="A safeguard designed to satisfy an obligation"
    FINDING="An observation from an audit or assessment"
    RISK="A potential adverse outcome"
    EVIDENCE="A source passage supporting a claim or finding"
    DOCUMENT="original source file"
class RelationType(Enum):
    OWNS="owns"
    IN_SCOPE_OF="in scope of"
    DEFINES="defines"
    SATISFIES="satisfies"
    EVALUATES="evaluates"
    AFFECTS="affects"
    SUPPORTS="supports"
    EXTRACTED_FROM="extracted from"
    INDICATED_BY="indicated by"
relations=set([
    (NodeType.ORGANIZATION, RelationType.OWNS, NodeType.APPLICATION),
    (NodeType.APPLICATION, RelationType.IN_SCOPE_OF, NodeType.POLICY),
    (NodeType.POLICY, RelationType.DEFINES, NodeType.OBLIGATION),
    (NodeType.CONTROL, RelationType.SATISFIES, NodeType.OBLIGATION),
    (NodeType.FINDING, RelationType.EVALUATES, NodeType.CONTROL),
    (NodeType.FINDING, RelationType.AFFECTS, NodeType.APPLICATION),
    (NodeType.EVIDENCE, RelationType.SUPPORTS, NodeType.FINDING),
    (NodeType.EVIDENCE, RelationType.EXTRACTED_FROM, NodeType.DOCUMENT),
    (NodeType.RISK, RelationType.INDICATED_BY, NodeType.FINDING)
])
@validate_call
def validate_relation(source_type:StrictStr, relation_type:StrictStr, target_type:StrictStr)->bool:
    source_type=source_type.upper()
    relation_type=relation_type.upper()
    target_type=target_type.upper()
    relation=(NodeType[source_type],RelationType[relation_type],NodeType[target_type])
    if relation in relations:
        return True
    else:
        return False
