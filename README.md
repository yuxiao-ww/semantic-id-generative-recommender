# Generative Recommendation with Semantic IDs

An end-to-end research implementation of a generative recommender that maps
catalog items to hierarchical Semantic IDs and predicts the next item as a
short token sequence.

The project uses Amazon Beauty interactions as a reproducible case study. It
contains the complete pipeline from data preprocessing and semantic embedding
through residual vector quantization, autoregressive recommendation,
constrained decoding, cold-start evaluation, and SASRec/popularity baselines.

## System design

```text
Amazon reviews + metadata
        |
        v
iterative 5-core + chronological user sequences
        |
        +---------------------------> SASRec / popularity baselines
        |
        v
BGE item-text embeddings (384-D, L2 normalized)
        |
        v
3-level RQ-VAE --------------------> hierarchical Semantic ID
        |                                      |
        |                            deterministic collision suffix
        |                                      |
        +--------------------------------------+
                                               v
user history -> decoder-only Transformer -> 4-token next-item ID
                                               |
                                               v
                                   trie-constrained beam search
                                               |
                                               v
                                  Recall@K / NDCG@K evaluation
```

## Verified experiment snapshot

The local experiment artifacts used to develop this repository produced:

- 22,363 users, 12,101 items, and 198,502 interactions after preprocessing.
- 91.0% weighted category purity across first-level Semantic ID buckets with at
  least five items.
- 1,169 items in colliding three-code buckets; the deterministic fourth token
  makes all 12,101 item IDs globally unique.
- A strict cold-start split with 1,210 held-out items and 843 test users.
- 89.5% of evaluable unseen items matched the dominant category of the warm
  items in their first-level code bucket, even though the quantizer was trained
  only on warm items.

These values are descriptive checks of representation quality and data
integrity. Ranking performance should be reproduced from checkpoints generated
in the target environment rather than inferred from them.

## Repository map

| Path | Purpose |
| --- | --- |
| `data/` | Preprocessing and sequential recommendation datasets |
| `semantic_id/` | BGE encoder, autoencoder, RQ-VAE, ID generation, and collision handling |
| `gr/` | Generative-recommender model, training, beam search, trie, and evaluation |
| `baselines/` | Popularity and SASRec baselines |
| `cold_start/` | Warm-only training, unseen-item assignment, and cold-start evaluation |
| `eval/` | Shared full-catalog Recall@K and NDCG@K evaluator |
| `tests/` | Fast invariants for tokenization, trie decoding, and quantization |

## Reproduce

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Download the Amazon Reviews 2014 Beauty 5-core review and metadata archives
into `data/raw/` as `reviews_Beauty_5.json.gz` and
`meta_Beauty.json.gz`. Generated datasets and checkpoints are intentionally
excluded from Git.

Run the main stages from the repository root:

```bash
python -m data.preprocess
python -m semantic_id.encoder
python -m semantic_id.train_rqvae
python -m semantic_id.build_sid
python -m semantic_id.resolve_collision
python -m gr.train
python -m gr.evaluate
```

Run baselines and the strict cold-start experiment:

```bash
python -m baselines.train_sasrec
python run_popularity.py
python -m cold_start.prepare_cold
python -m cold_start.train_rqvae_warm
python -m cold_start.build_sid_cold
python -m cold_start.train_gr_warm
python -m cold_start.train_sasrec_warm
python -m cold_start.eval_cold
```

Run the fast tests:

```bash
python -m unittest discover -s tests -v
```

## Engineering decisions

- **No ID leakage in cold-start evaluation.** Cold items are removed before the
  warm-only RQ-VAE and recommendation models are trained.
- **Stable preprocessing.** User/item ordering and equal-timestamp tie breaks
  are deterministic so regenerated splits remain comparable.
- **Collision-safe IDs.** Semantic quantization can legitimately map multiple
  items to one code path; a reproducible suffix preserves item-level identity.
- **Valid constrained generation.** The trie exposes only prefixes that lead to
  real catalog items, preventing non-existent Semantic IDs at inference time.
- **Comparable evaluation.** Generative retrieval and baselines use the same
  leave-one-out targets and Recall@K/NDCG@K definitions.

## Data and model artifacts

Amazon review data is not redistributed here. Checkpoints and generated data
are also excluded to keep the repository lightweight and to avoid publishing
third-party content. See the upstream dataset terms before downloading or
redistributing the source archives.
