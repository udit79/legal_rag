# Enterprise Risk Domain Model

## Purpose and scope

The enterprise risk system represents how an organization’s applications relate
to policies and obligations, how controls are intended to address obligations,
and how assessment findings, evidence, and risks are connected. It models these
relationships and source-document provenance; a recorded relationship alone
does not establish compliance or control effectiveness.

## Node types

| Node type | Meaning |
| --- | --- |
| `ORGANIZATION` | The organization being assessed. |
| `APPLICATION` | A system or application. |
| `POLICY` | An internal policy or external requirement document. |
| `OBLIGATION` | A requirement that must be satisfied. |
| `CONTROL` | A safeguard designed to satisfy an obligation. |
| `FINDING` | An observation from an audit or assessment. |
| `RISK` | A potential adverse outcome. |
| `EVIDENCE` | A source passage supporting a claim or finding. |
| `DOCUMENT` | An original source file. |
| `CHUNK` | An ordered text segment belonging to a source document. |

## Relationship triples

Each triple is **source — relationship → target**; direction is significant.

| Source | Relationship | Target |
| --- | --- | --- |
| `ORGANIZATION` | `OWNS` | `APPLICATION` |
| `APPLICATION` | `IN_SCOPE_OF` | `POLICY` |
| `POLICY` | `DEFINES` | `OBLIGATION` |
| `CONTROL` | `SATISFIES` | `OBLIGATION` |
| `FINDING` | `EVALUATES` | `CONTROL` |
| `FINDING` | `AFFECTS` | `APPLICATION` |
| `EVIDENCE` | `SUPPORTS` | `FINDING` |
| `EVIDENCE` | `EXTRACTED_FROM` | `CHUNK` |
| `RISK` | `INDICATED_BY` | `FINDING` |
| `DOCUMENT` | `HAS_CHUNK` | `CHUNK` |
| `CHUNK` | `NEXT_CHUNK` | `CHUNK` |

## Validation behavior

`validate_relation(source_type, relation_type, target_type)` accepts enum names
as strings, case-insensitively. It returns `True` for a defined triple and
`False` for an unknown name or a combination that is not defined. Enum values
are stable uppercase identifiers; descriptions are maintained separately.

## Findings, evidence, and control failures

A **finding** records an assessment observation about a control or application.
**Evidence** is source material that supports a finding; it is not itself the
finding or a conclusion. A **verified control failure** is a substantiated
conclusion that a control did not operate as required, based on assessment and
sufficient evidence. Neither a finding nor evidence alone necessarily means a
verified failure.

## Modelling decisions

- **Separate `POLICY` and `OBLIGATION`:** A policy or external requirement is a
  source document; an obligation is a specific requirement derived from it.
  This allows one policy to define multiple obligations and keeps the source
  distinct from the actionable requirement.
- **Separate `FINDING` and `EVIDENCE`:** Findings are assessment observations;
  evidence passages are supporting material extracted from source documents.
  Keeping them distinct preserves provenance and allows a finding to be
  supported by evidence without treating quoted material as an assessment
  conclusion.
- **`SATISFIES` does not prove effectiveness:** The relationship records that a
  control is intended to address an obligation. It does not show that the
  control is implemented, operating, or effective; those require assessment
  and appropriate evidence.
