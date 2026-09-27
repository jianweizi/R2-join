# Results

This file records the main evaluation results for R2-Join. All values are
reported with four decimal places.

## OpenData

Number of test queries: 1707.

| HIT@10 | RECALL@10 | MRR@10 | NDCG@10 |
| ---: | ---: | ---: | ---: |
| 0.7458 | 0.6273 | 0.3382 | 0.3863 |

## WebTable

Number of test queries: 1522.

| HIT@10 | RECALL@10 | MRR@10 | NDCG@10 |
| ---: | ---: | ---: | ---: |
| 0.5309 | 0.4765 | 0.2664 | 0.3061 |

## Odoo

| HIT@10 | RECALL@10 | MRR@10 | NDCG@10 |
| ---: | ---: | ---: | ---: |
| 0.5573 | 0.5573 | 0.2775 | 0.3435 |

## Ablation Study

Evaluation cutoff: 10.

### OpenData

| Method | HIT@10 | RECALL@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| R2-Join | 0.7458 | 0.6273 | 0.3382 | 0.3863 |
| w/o Reranker | 0.6356 | 0.5314 | 0.2323 | 0.2868 |
| w/o Rich Column Evidence | 0.7305 | 0.6205 | 0.3247 | 0.3750 |
| w/o Rank-Tiered Negative Sampling | 0.6385 | 0.5205 | 0.2173 | 0.2717 |
| w/o Hybrid Loss | 0.7288 | 0.6143 | 0.3030 | 0.3572 |

### WebTable

| Method | HIT@10 | RECALL@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| R2-Join | 0.5309 | 0.4765 | 0.2664 | 0.3061 |
| w/o Reranker | 0.4113 | 0.3646 | 0.1696 | 0.2081 |
| w/o Rich Column Evidence | 0.4139 | 0.3708 | 0.1743 | 0.2129 |
| w/o Rank-Tiered Negative Sampling | 0.3463 | 0.3025 | 0.1460 | 0.1762 |
| w/o Hybrid Loss | 0.4803 | 0.4316 | 0.2331 | 0.2727 |

## Efficiency

Mean online query latency reported in the paper:

| Dataset | Mean online latency |
| --- | ---: |
| OpenData | 540 ms |
| WebTable | 359 ms |
| Odoo | 181 ms |

These timings exclude offline candidate preprocessing, candidate encoding, and
embedding caching.
