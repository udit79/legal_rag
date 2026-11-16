import asyncio
import copy
import json

import pytest

from src.graph.semantic_schema import required_semantic_constraint_shapes
from src.graph.visual_observation_writer import (
    VisualObservationConflict,
    _VISUAL_CONSTRAINT_SHAPES,
    _write_visual_observation_transaction,
    persist_visual_observations,
)
from tests.graph.test_visual_observations import _asset, _document, _proposal
from src.graph.visual_observations import (
    build_visual_observation,
    extract_document_visual_observations,
)


def _observation():
    document, chunk, source = _document()
    return build_visual_observation(
        document=document, chunk=chunk, source=source, asset=_asset(),
        proposal=_proposal(), extraction_provider="openai", extraction_model="extractor-v1",
    )


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def __aiter__(self):
        async def iterate():
            for row in self.rows:
                yield row
        return iterate()

    async def single(self):
        return self.rows[0] if self.rows else None


def _constraints():
    shapes = required_semantic_constraint_shapes() | _VISUAL_CONSTRAINT_SHAPES
    return [{"entity_type": entity, "type": kind,
             "labels": list(labels), "properties": list(props)}
            for entity, kind, labels, props in shapes]


class _Tx:
    def __init__(self):
        self.observations = {}
        self.candidates = {}
        self.relationships = {}
        self.semantic = [{"properties": {"id": "organization:bank", "name": " Bank  "},
                          "labels": ["SEMANTIC_ENTITY", "ORGANIZATION"]}]

    async def run(self, query, **params):
        if "SHOW CONSTRAINTS" in query:
            return _Result([{"constraints": _constraints()}])
        if "MATCH (n:SEMANTIC_ENTITY)" in query:
            return _Result(self.semantic)
        if "UNWIND $ids AS id" in query:
            rows = []
            for identity in params["ids"]:
                matches = []
                if identity in self.observations:
                    matches.append(self.observations[identity])
                if identity in self.candidates:
                    matches.append(self.candidates[identity])
                rows.append({"id": identity, "matches": matches})
            return _Result(rows)
        if "MERGE (o:VISUAL_OBSERVATION" in query:
            self.observations.setdefault(params["id"], params["properties"])
            return _Result([{"properties": self.observations[params["id"]]}])
        if "MERGE (e:VISUAL_ENTITY_CANDIDATE" in query:
            self.candidates.setdefault(params["id"], params["properties"])
            rel = {"id": params["relationship_id"],
                   "trust_status": "UNVERIFIED_AI_VISUAL_OBSERVATION"}
            self.relationships.setdefault(params["relationship_id"], rel)
            return _Result([{"properties": self.candidates[params["id"]],
                             "relationship": rel,
                             "source_id": params["id"],
                             "target_id": params["observation_id"]}])
        if "MERGE (a)-[r:VISUALLY_RELATED_TO" in query:
            if params["source_id"] not in self.candidates or params["target_id"] not in self.candidates:
                return _Result([])
            self.relationships.setdefault(params["id"], params["properties"])
            return _Result([{"properties": self.relationships[params["id"]],
                             "source_id": params["source_id"], "target_id": params["target_id"]}])
        if "MERGE (e)-[r:VISUALLY_REFERENCES" in query:
            rel = {"id": params["id"], "trust_status": "UNVERIFIED_VISUAL_ENTITY_MATCH",
                   "match_method": "EXACT_NORMALIZED_NAME_AND_TYPE"}
            self.relationships.setdefault(params["id"], rel)
            return _Result([{"properties": rel, "source_id": params["candidate_id"],
                             "target_id": params["semantic_id"]}])
        raise AssertionError(query)


class _Session:
    def __init__(self):
        self.tx = _Tx()

    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass

    async def execute_write(self, callback):
        snapshot = copy.deepcopy(self.tx.__dict__)
        try:
            return await callback(self.tx)
        except Exception:
            self.tx.__dict__.clear()
            self.tx.__dict__.update(snapshot)
            raise


class _Db:
    def __init__(self): self.session = _Session()
    def get_session(self): return self.session


def test_visual_graph_persistence_is_idempotent_and_links_only_exact_existing_entities():
    tx = _Tx()
    first = asyncio.run(_write_visual_observation_transaction(tx, _observation()))
    replay = asyncio.run(_write_visual_observation_transaction(tx, _observation()))
    assert first == replay == {
        "observations": 1, "candidates": 2, "candidate_relationships": 1, "entity_links": 1,
    }
    assert len(tx.observations) == 1
    assert len(tx.candidates) == 2
    assert len(tx.relationships) == 4
    assert tx.observations[_observation().id]["trust_status"] == "UNVERIFIED_AI_VISUAL_OBSERVATION"
    assert tx.candidates[next(iter(tx.candidates))].get("verified_fact") is None


def test_visual_graph_persistence_rejects_changed_content_under_existing_identity():
    tx = _Tx()
    observation = _observation()
    asyncio.run(_write_visual_observation_transaction(tx, observation))
    conflict = observation.model_copy(update={"description_text": "different visual text"})
    with pytest.raises(VisualObservationConflict, match="immutable content"):
        asyncio.run(_write_visual_observation_transaction(tx, conflict))


def test_mocked_end_to_end_vlm_chunk_to_image_linked_graph_observation():
    document, _, _ = _document()
    db = _Db()

    class Extractor:
        provider = "openai"
        model = "visual-extractor-v1"
        async def extract(self, description):
            assert description == _observation().description_text
            return _proposal()

    class Resolver:
        async def resolve(self, run_id, keys):
            assert run_id == _observation().ingestion_run_id
            return {(2, "page-2-block-3"): _asset()}

    async def run():
        observations = await extract_document_visual_observations(
            Extractor(), document, Resolver()
        )
        counts = await persist_visual_observations(db, observations)
        return observations, counts

    observations, counts = asyncio.run(run())
    assert counts == {"observations": 1, "candidates": 2,
                      "candidate_relationships": 1, "entity_links": 1}
    assert len(db.session.tx.observations) == 1
    assert len(db.session.tx.candidates) == 2
    persisted = next(iter(db.session.tx.observations.values()))
    assert persisted["image_sha256"] == "b" * 64
    assert persisted["trust_status"] == "UNVERIFIED_AI_VISUAL_OBSERVATION"
    assert json.loads(persisted["source_coordinates_json"])["x0"] == 1.0
    assert "source_coordinates" not in persisted
    assert observations[0].source_coordinates["x0"] == 1.0

