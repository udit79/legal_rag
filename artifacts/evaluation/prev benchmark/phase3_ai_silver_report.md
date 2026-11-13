# Phase 3 AI-Silver Benchmark Report

**Classification:** AI-silver provisional development results. These labels are AI-generated and AI-source-checked, not human-verified ground truth. Do not present as an authoritative or statistically conclusive benchmark.

- Dataset: `phase3-ai-silver-development-evaluation` version `v1.0-provisional`
- Dataset SHA-256: `ed2a3dfafa3af2964fe22220f786e93863039ed4260c60ecc7c20acdb8a8b7ba`
- Corpus: Neo4j `rag`, 6 documents, 333 chunks
- Query set: 30 total (24 answerable, 6 unanswerable)
- Executed: 30 queries per strategy × 3 strategies = 90/90 query evaluations; zero failed, zero skipped.
- Cutoff: top-k = 10. Embedding: openai/text-embedding-3-small (1024d); vector index `chunk_embedding_204393822463970a`.

## Corpus snapshot

| Source SHA-256 | Ingestion run ID |
|---|---|
| `3c31f46fee98cac0c4323453e5109291a213b4de7fef8c058af9bf67f717433c` | `d5bc2bd4f66f4889aeeea0fb7ec5d40b` |
| `5c46c449018b8af374d140395e59ca67ce217f4d9a51e01b2454cba0a90df2c3` | `b4e8767410ab4888a574570b9d8a00c9` |
| `8ad6286676b298038b2dda8f6a6bc4014c2a3fe6f0b5c59b1b02fcaf77cb5f81` | `864eef9576a8461da85bf3361df4e099` |
| `90e73c9bc6c699a149aa772d09b467bf36e4d9eb90c629ed0eb1d69c51f9bf43` | `e391994767e943d8a5584cdc3dc387b0` |
| `9a4675dae853057f0a0e682ad1c1ae51ab0e78505d7b133ed70ab40876d79980` | `9ecbccf37f7f4f4f9cd238f9a6479c1f` |
| `dda561b9fd6420e1b99157b60f92bb0470a6154c6d559bbee3482ac255ac2c24` | `9bed2588d3e4494f8cac83fc4febba3e` |

## Aggregate metrics

| Strategy | Recall@10 | MRR | Document Recall@10 | Retrieved Document Precision | Unanswerable Abstention |
|---|---:|---:|---:|---:|---:|
| fulltext | 0.871528 | 0.716667 | 1.000000 | 0.560417 | 0.000000 |
| vector | 0.892361 | 0.885417 | 1.000000 | 0.704861 | 0.000000 |
| hybrid | 0.930556 | 0.868056 | 1.000000 | 0.635417 | 0.000000 |

All strategies used the same 30 query IDs, query order, corpus snapshot, `k=10`, embedding model/index and existing retriever settings. No retrieval tuning was performed. Metrics were independently recomputed from each strategy artifact and matched its stored values.

## Coverage and abstention

Each strategy has 24 answerable and 6 unanswerable queries, each with ten ranked results. There are no recorded failed or skipped query IDs. All six unanswerable queries returned results for all strategies, and each top score exceeded its recorded abstention threshold (0.01), so abstention is implemented and evaluated as **0/6** for each strategy. Thresholding is applied in each native score scale separately; the raw full-text, vector and RRF scores are not directly comparable.

Answer correctness and answer-citation metrics are **NOT_EVALUATED** because this retrieval benchmark does not supply generated answers or answer citations.

## Metric formulas

- **recall_at_k:** For each answerable query: distinct relevant chunk IDs in top-k divided by the number of expected relevant chunk IDs; macro-average over answerable queries.
- **mrr:** For each answerable query: reciprocal rank of the first retrieved expected relevant chunk, or 0 if none; macro-average over answerable queries.
- **document_recall_at_k:** For each answerable query: distinct expected document hashes represented in top-k divided by the number of expected document hashes; macro-average over answerable queries.
- **retrieved_document_precision:** For each answerable query: distinct retrieved document hashes that are expected divided by distinct retrieved document hashes; macro-average over answerable queries; 0 if no document retrieved.
- **unanswerable_query_abstention:** For each unanswerable query: 1 if retrieval is empty or the top result declared abstention score is <= the configured strategy-specific threshold, else 0; mean over unanswerable queries.

**Aggregation:** macro-average over 24 answerable queries for Recall@10, MRR, Document Recall@10 and Retrieved Document Precision; abstention averages only the 6 unanswerable queries. Multi-chunk labels are counted as a set for recall; MRR uses the first relevant rank.

## Provenance and files

- Full JSON with top-10 ranked evidence for each query: `artifacts/evaluation/phase3_ai_silver_benchmark.json`
- CSV with one row per strategy/query and complete ranked evidence payload: `artifacts/evaluation/phase3_ai_silver_per_query.csv`
- Source smoke report: `artifacts/graph_retrieval_smoke/latest.json`
- The complete per-query rankings include chunk/document IDs, raw evidence text, source coordinates, retrieval channels and native scores in the JSON artifact.
