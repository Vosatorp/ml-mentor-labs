# ML Mentor Labs

Public starter notebooks for hands-on exercises in the ML Mentor roadmaps.

The repository intentionally contains only task statements, small public or
built-in datasets, and deterministic self-checks. Editorial rubrics, solutions,
paid theory, and material from private interviews are not published here.

## Labs

1. `01-metrics-threshold.ipynb` — choose a classification threshold under explicit FP/FN costs.
2. `02-honest-binary-baseline.ipynb` — build a leak-free scikit-learn baseline.
3. `03-leakage-splits.ipynb` — compare random, group, and time-based validation.
4. `04-tree-ensembles.ipynb` — compare a tree, Random Forest, and gradient boosting under one budget.
5. `05-broken-pytorch-loop.ipynb` — repair and verify a PyTorch training loop.
6. `06-correlated-importance.ipynb` — inspect permutation importance with correlated features.
7. `07-bike-demand-production-capstone.ipynb` — build a time-aware Bike Sharing baseline and production contract.
8. `08-banking77-tfidf-error-analysis.ipynb` — compare word/character TF-IDF and audit errors across 77 intents.
9. `09-rag-failure-decomposition.ipynb` — measure retrieval and separate RAG failure buckets on a fixed SQuAD 2.0 subset.
10. `11-cifar10-pytorch-training-pipeline.ipynb` — build and verify a reproducible PyTorch training pipeline on CIFAR-10.

`10-llm-serving-benchmark.ipynb` and the collectors in `advanced/` are drafts.
They remain outside the public index until real, provenance-checked vLLM and
SGLang fixtures exist for one comparable protocol. Synthetic benchmark values
are not accepted.

Each notebook has a fixed seed, an explicit deliverable, and local assertions.
Run it in Colab or a Python 3.11+ environment. CPU is sufficient.

Labs 07–09 first look for the checksum-verified fixture in a repository
checkout. When a notebook is opened by itself in Colab, its setup may download
only the declared upstream source, verifies its SHA-256, and caches the verified
artifact locally. The validation job blocks that network fallback and proves
that the committed offline path works without secrets. Redirects are accepted
only when the final hostname matches the notebook allowlist; SHA-256 validation
keeps the downloaded bytes fail-closed.

Lab 11 intentionally does not package CIFAR-10 in Git. Its setup downloads the
official Python archive from `www.cs.toronto.edu`, verifies its current final
hostname `cave.cs.toronto.edu`, byte count and SHA-256, then relies on
torchvision's per-batch integrity checks.
The required 10k/2k path runs on CPU; full-data CUDA + AMP is optional.

## Dataset licenses

The repository-level MIT license applies to starter text and code only. Dataset
fixtures keep their own terms and attribution:

- UCI Bike Sharing: CC BY 4.0; see `data/uci-bike-sharing/LICENSE` and
  `data/uci-bike-sharing/ATTRIBUTION.md`.
- Banking77: CC BY 4.0; see `data/banking77/LICENSE` and `data/datasets.json`.
- Adapted SQuAD 2.0 subset: CC BY-SA 4.0; see `data/squad2/LICENSE` and
  `data/squad2/ATTRIBUTION.md` for the deterministic selection/modification
  notice.
- CIFAR-10: the authors do not publish a standard license on the official
  dataset page. The archive is not redistributed; `data/datasets.json` records
  the source, checksum, attribution and technical report.
- Yambda-50M likes subset: Apache 2.0; see `data/yambda/LICENSE` and
  `data/yambda/ATTRIBUTION.md` for the deterministic selection and modification
  notice.

The course-mechanics attribution for YDS Practical_DL and its MIT notice are
recorded in `THIRD_PARTY_NOTICES.md` and `third_party/`.

`labs-manifest.json` is the release source of truth. `publicIndex` contains only
public notebooks; `review_ready` and `draft` files are never exposed by that
index.

Continuous verification regenerates every notebook, compiles all code cells,
rejects committed outputs and executes the dependency/setup cells. Exercise and
self-check cells stay unexecuted because starters intentionally contain no
reference solutions.

## Regeneration

The notebooks are generated from `scripts/generate_notebooks.py` so cell order
and metadata remain deterministic.

```bash
python3 scripts/sync_datasets.py --check
python3 scripts/generate_notebooks.py
python3 scripts/validate_notebooks.py --execute-setup
```

Run `sync_datasets.py` without `--check` only for an intentional fixture
refresh followed by license, checksum, and notebook review.

## License

Starter code and task text are available under the MIT License. Dataset and
library licenses remain with their respective publishers.
