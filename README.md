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

Each notebook has a fixed seed, an explicit deliverable, and local assertions.
Run it in Colab or a Python 3.11+ environment. CPU is sufficient.

## Regeneration

The notebooks are generated from `scripts/generate_notebooks.py` so cell order
and metadata remain deterministic.

```bash
python3 scripts/generate_notebooks.py
```

## License

Starter code and task text are available under the MIT License. Dataset and
library licenses remain with their respective publishers.
