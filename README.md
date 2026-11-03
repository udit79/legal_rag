# Graph RAG

Graph RAG is a Python service foundation for connecting source documents to a
validated risk-domain graph. The repository contains API health checks, Neo4j
connectivity, telemetry setup, graph-domain validation, canonical parser
adapters, and the BLP-1 deterministic chunking implementation. A full workflow
runner that connects every ingestion stage is still planned.

See [docs/architecture.md](docs/architecture.md) for the current component
structure, data contracts, and the boundary between implemented foundations
and planned workflow integration. See [docs/code_map.md](docs/code_map.md) for
the file tree and a file-by-file index of implemented classes and functions.
The implemented BLP-1 chunking strategy is described in
[docs/blp1_chunking.md](docs/blp1_chunking.md).
