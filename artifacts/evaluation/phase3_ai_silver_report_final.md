# Phase 3 AI-Silver Benchmark — Finalization

**Classification: AI-silver provisional development results.** The query labels were AI-generated and AI-source checked; they are not human-verified ground truth.

- Dataset: `phase3-ai-silver-development-evaluation` version `v1.0-provisional`
- Dataset SHA-256: `ed2a3dfafa3af2964fe22220f786e93863039ed4260c60ecc7c20acdb8a8b7ba`
- Coverage: 30 queries (24 answerable, 6 unanswerable) × 3 strategies = **90/90**, no failures or skips.
- Cutoff: top-k 10. The additional metrics below were computed from the existing saved top-10 rankings; retrieval was not rerun.

## Metrics

| Strategy | Recall@1 | Recall@3 | Recall@5 | Recall@10 | nDCG@10 | MRR | Document Recall@10 | Retrieved Document Precision | Unanswerable Abstention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fulltext | 0.423611 | 0.663194 | 0.809028 | 0.871528 | 0.732209 | 0.716667 | 1.000000 | 0.560417 | 0.000000 |
| vector | 0.524306 | 0.777778 | 0.822917 | 0.892361 | 0.825032 | 0.885417 | 1.000000 | 0.704861 | 0.000000 |
| hybrid | 0.503472 | 0.774306 | 0.864583 | 0.930556 | 0.832035 | 0.868056 | 1.000000 | 0.635417 | 0.000000 |

## Metric definitions

- **Recall@k:** for each answerable query, distinct expected relevant chunk IDs retrieved in the first k ranks divided by the count of expected relevant chunk IDs; macro-average over 24 answerable queries. All relevant chunks are included in the denominator.
- **MRR:** reciprocal rank of the first expected relevant chunk, or zero when none appears in top 10; macro-average over the 24 answerable queries.
- **nDCG@10:** binary chunk relevance, DCG = sum(relᵢ / log₂(i+1)); IDCG uses up to ten relevant chunks; macro-average over answerable queries.
- **Document Recall@10:** distinct expected document hashes in top 10 divided by expected document count; macro-average over answerable queries.
- **Retrieved Document Precision:** distinct expected document hashes in top 10 divided by distinct retrieved document hashes; macro-average over answerable queries (zero for empty retrieval).
- **Unanswerable abstention:** mean indicator over six unanswerable queries; abstain if no result or top score <= configured threshold.

## Abstention audit

The evaluator implements abstention. It compares the top result’s strategy-declared native score against the configured threshold and counts only unanswerable queries in this metric. The current threshold is 0.01 for each strategy. This is not calibrated across score types: full-text relevance, vector similarity, and RRF scores have different ranges and distributions. In the saved benchmark all six unanswerable queries returned results with top scores above 0.01 for each strategy, so the reported rate is 0/6. Treat that as threshold behavior on this AI-silver set, not evidence of calibrated abstention quality.

### Recommended calibration

1. Create a separate calibration set of human-reviewed answerable and corpus-unanswerable queries from the same frozen corpus. Do not reuse any benchmark query or paraphrase family; split by query intent/source topic and lock the benchmark set before calibration.
2. For each strategy independently, record the native top score and answerability label. Keep empty retrieval as a separate abstention case. Do not pool or compare the raw score values across strategies.
3. Before looking at calibration scores, choose an operating objective, such as at least 95% unanswerable-query abstention subject to no more than 5% false abstention on answerable queries, or a predeclared cost-weighted objective.
4. Choose the threshold on the calibration split only, report calibration sample sizes, score type, selected threshold, confusion matrix, and uncertainty intervals. If the target is infeasible, report the trade-off rather than silently relaxing the target.
5. Freeze each strategy-specific threshold, then evaluate once on the untouched locked benchmark. Report test abstention and answerable false-abstention separately. Any threshold change requires a new held-out evaluation set.
6. The current 30-query AI-silver benchmark (with only six provisional unanswerable labels) is too small and not human-verified, so do not use it to tune and then claim unbiased test abstention performance.

## Preservation and files

- Original benchmark JSON, CSV, and report were preserved unchanged. The final files are additional artifacts.
- Expanded JSON: `artifacts/evaluation/phase3_ai_silver_benchmark_final.json`
- Expanded per-query CSV: `artifacts/evaluation/phase3_ai_silver_per_query_final.csv`
- The expanded JSON includes the original top-10 rankings plus per-query Recall@1/3/5 and nDCG@10; CSV includes corresponding per-query columns.
- Answer correctness and answer-citation metrics remain NOT_EVALUATED.
- Fulltext, Vector and Hybrid score values retain their native scale and are not compared across strategies.
