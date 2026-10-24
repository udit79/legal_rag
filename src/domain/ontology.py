from enum import Enum
from typing import Final
from pydantic import StrictStr, validate_call

class NodeType(Enum):
    ORGANIZATION = "ORGANIZATION"
    APPLICATION = "APPLICATION"
    POLICY = "POLICY"
    OBLIGATION = "OBLIGATION"
    CONTROL = "CONTROL"
    FINDING = "FINDING"
    RISK = "RISK"
    EVIDENCE = "EVIDENCE"
    DOCUMENT = "DOCUMENT"
    CHUNK = "CHUNK"

NODE_TYPE_DESCRIPTIONS: Final[dict[NodeType, str]] = {
    NodeType.ORGANIZATION: "The organization being assessed",
    NodeType.APPLICATION: "A system or application",
    NodeType.POLICY: "An internal policy or external requirement document",
    NodeType.OBLIGATION: "A requirement that must be satisfied",
    NodeType.CONTROL: "A safeguard designed to satisfy an obligation",
    NodeType.FINDING: "An observation from an audit or assessment",
    NodeType.RISK: "A potential adverse outcome",
    NodeType.EVIDENCE: "A source passage supporting a claim or finding",
    NodeType.DOCUMENT: "An original source file",
    NodeType.CHUNK: "An ordered text segment belonging to a source document",
}

class RelationType(Enum):
    OWNS = "OWNS"
    IN_SCOPE_OF = "IN_SCOPE_OF"
    DEFINES = "DEFINES"
    SATISFIES = "SATISFIES"
    EVALUATES = "EVALUATES"
    AFFECTS = "AFFECTS"
    SUPPORTS = "SUPPORTS"
    EXTRACTED_FROM = "EXTRACTED_FROM"
    INDICATED_BY = "INDICATED_BY"
    HAS_CHUNK = "HAS_CHUNK"
    NEXT_CHUNK = "NEXT_CHUNK"

class EpistemicMode(Enum):
    VERBATIM_EXTRACTION = "VERBATIM_EXTRACTION"
    RULE_DERIVED = "RULE_DERIVED"
    LLM_INFERRED = "LLM_INFERRED"

class EntityResolutionMethod(Enum):
    EXACT_IDENTIFIER = "EXACT_IDENTIFIER"
    NORMALIZED_NAME = "NORMALIZED_NAME"
    ALIAS = "ALIAS"
    HUMAN_REVIEW = "HUMAN_REVIEW"

RELATION_DESCRIPTIONS: Final[dict[RelationType, str]] = {
    RelationType.OWNS: "The organization owns or is responsible for the application.",
    RelationType.IN_SCOPE_OF: "The application is governed by or assessed against the policy.",
    RelationType.DEFINES: "The policy states the obligation.",
    RelationType.SATISFIES: "Maps a control to an obligation; this mapping alone does not prove compliance or effectiveness.",
    RelationType.EVALUATES: "The finding assesses the control's implementation or operation.",
    RelationType.AFFECTS: "The finding has an impact on the application.",
    RelationType.SUPPORTS: "The evidence supports the finding.",
    RelationType.EXTRACTED_FROM: "The evidence passage was extracted from the chunk.",
    RelationType.INDICATED_BY: "The risk is indicated by the finding.",
    RelationType.HAS_CHUNK: "The document contains the chunk; chunk_index provides its order.",
    RelationType.NEXT_CHUNK: "The source chunk immediately precedes the target chunk.",
}

EPISTEMIC_MODE_DESCRIPTIONS: Final[dict[EpistemicMode, str]] = {
    EpistemicMode.VERBATIM_EXTRACTION: "A claim quoted or directly extracted from source text.",
    EpistemicMode.RULE_DERIVED: "A claim derived by an explicit deterministic rule.",
    EpistemicMode.LLM_INFERRED: "A claim inferred by a language model.",
}

# Directed endpoint types. Existing business relationship directions are retained.
# HAS_CHUNK means DOCUMENT contains CHUNK. NEXT_CHUNK means the source chunk
# immediately precedes the target; neither relation is symmetric or transitive.
ALLOWED_TRIPLES: Final[dict[RelationType, frozenset[tuple[NodeType, NodeType]]]] = {
    RelationType.OWNS: frozenset({(NodeType.ORGANIZATION, NodeType.APPLICATION)}),
    RelationType.IN_SCOPE_OF: frozenset({(NodeType.APPLICATION, NodeType.POLICY)}),
    RelationType.DEFINES: frozenset({(NodeType.POLICY, NodeType.OBLIGATION)}),
    RelationType.SATISFIES: frozenset({(NodeType.CONTROL, NodeType.OBLIGATION)}),
    RelationType.EVALUATES: frozenset({(NodeType.FINDING, NodeType.CONTROL)}),
    RelationType.AFFECTS: frozenset({(NodeType.FINDING, NodeType.APPLICATION)}),
    RelationType.SUPPORTS: frozenset({(NodeType.EVIDENCE, NodeType.FINDING)}),
    RelationType.EXTRACTED_FROM: frozenset({(NodeType.EVIDENCE, NodeType.CHUNK)}),
    RelationType.INDICATED_BY: frozenset({(NodeType.RISK, NodeType.FINDING)}),
    RelationType.HAS_CHUNK: frozenset({(NodeType.DOCUMENT, NodeType.CHUNK)}),
    RelationType.NEXT_CHUNK: frozenset({(NodeType.CHUNK, NodeType.CHUNK)}),
}

if set(NODE_TYPE_DESCRIPTIONS) != set(NodeType):
    raise RuntimeError("Every node type must have a description")
if set(ALLOWED_TRIPLES) != set(RelationType):
    raise RuntimeError("Every relationship type must have allowed endpoints")
if set(RELATION_DESCRIPTIONS) != set(RelationType):
    raise RuntimeError("Every relationship type must have a description")
if set(EPISTEMIC_MODE_DESCRIPTIONS) != set(EpistemicMode):
    raise RuntimeError("Every epistemic mode must have a description")

@validate_call
def validate_relation(
    source_type: StrictStr, relation_type: StrictStr, target_type: StrictStr
) -> bool:
    try:
        source = NodeType(source_type.upper())
        relation = RelationType(relation_type.upper())
        target = NodeType(target_type.upper())
    except (KeyError, ValueError, AttributeError):
        return False
    return (source, target) in ALLOWED_TRIPLES[relation]
