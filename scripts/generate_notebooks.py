from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str, *, role: str = "exercise") -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"ml_mentor": {"role": role}},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def notebook(title: str, cells: list[dict]) -> dict:
    return {
        "cells": [md(f"# {title}\n\nML Mentor · starter notebook · решения в репозитории отсутствуют.\n")] + cells,
        "metadata": {
            "colab": {"name": title, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


LABS: dict[str, dict] = {
    "01-metrics-threshold.ipynb": notebook("Метрики и порог решения", [
        md("""## Задача

Есть 20 заявок, истинная метка дефолта и score модели. Ложный пропуск дефолта стоит 8 условных единиц, лишняя ручная проверка — 1. Посчитайте confusion matrix, precision и recall для нескольких порогов и выберите порог с минимальной стоимостью.

**Deliverable:** таблица по порогам, выбранный порог и объяснение в 3–5 предложениях."""),
        code("""import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score

y_true = np.array([0,0,1,0,1,0,0,1,0,0,1,0,0,0,1,0,1,0,0,1])
y_score = np.array([.05,.12,.88,.22,.63,.31,.08,.71,.44,.18,.56,.09,.27,.37,.92,.14,.49,.33,.24,.81])
FP_COST = 1
FN_COST = 8
THRESHOLDS = np.arange(0.1, 0.91, 0.05)
""", role="setup"),
        code("""def evaluate_threshold(y_true, y_score, threshold):
    # TODO: верните dict с threshold, tn, fp, fn, tp, precision, recall, cost.
    # Используйте zero_division=0 для precision/recall.
    raise NotImplementedError

rows = [evaluate_threshold(y_true, y_score, t) for t in THRESHOLDS]
results = pd.DataFrame(rows)
results.sort_values(["cost", "threshold"]).head()
"""),
        code("""# Self-check: запускайте после реализации.
assert set(["threshold", "tn", "fp", "fn", "tp", "precision", "recall", "cost"]) <= set(results.columns)
assert (results[["tn", "fp", "fn", "tp"]].sum(axis=1) == len(y_true)).all()
best = results.sort_values(["cost", "threshold"]).iloc[0]
assert 0 <= best.precision <= 1 and 0 <= best.recall <= 1
print(best)
""", role="self_check"),
        md("""## Вывод

Запишите выбранный threshold, цену FP/FN, ближайший альтернативный threshold и причину выбора. Отдельно объясните, почему этот threshold нельзя подбирать на test."""),
    ]),
    "02-honest-binary-baseline.ipynb": notebook("Честный baseline бинарной классификации", [
        md("""## Задача

Постройте воспроизводимый baseline на встроенном датасете breast cancer. Разделите данные до fit любых преобразований, соберите `Pipeline(StandardScaler, LogisticRegression)`, посчитайте ROC-AUC и Average Precision, затем выберите порог под стоимость FP=1, FN=5.

**Deliverable:** одна таблица метрик, confusion matrix на выбранном пороге и короткий список того, что защищает решение от leakage."""),
        code("""import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

SEED = 42
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED
)
""", role="setup"),
        code("""# TODO: соберите и обучите Pipeline.
model = None

# TODO: получите test scores через predict_proba.
y_score = None
"""),
        code("""# Self-check: он проверяет контракт, но не подбирает решение за вас.
assert model is not None
assert y_score is not None and len(y_score) == len(y_test)
assert np.isfinite(y_score).all() and ((0 <= y_score) & (y_score <= 1)).all()
print({
    "roc_auc": roc_auc_score(y_test, y_score),
    "average_precision": average_precision_score(y_test, y_score),
})
""", role="self_check"),
        code("""# TODO: выберите threshold на отдельной validation-части или через out-of-fold predictions.
# Для лабораторной явно опишите упрощение, если повторно используете train для выбора порога.
chosen_threshold = None
assert chosen_threshold is not None and 0 < chosen_threshold < 1
y_pred = (y_score >= chosen_threshold).astype(int)
confusion_matrix(y_test, y_pred)
"""),
        md("""## Leakage check

Ответьте: почему `StandardScaler().fit(X)` до split даёт утечку? Какие ещё сущности — imputer, target encoder, feature selection, threshold — должны обучаться без test?"""),
    ]),
    "03-leakage-splits.ipynb": notebook("Leakage detective: random, group и time split", [
        md("""## Задача

В таблице несколько событий на одного пользователя, а target зависит от устойчивого пользовательского эффекта и времени. Сравните random split, `GroupShuffleSplit` по user_id и временную границу. Объясните, почему оценки отличаются.

**Deliverable:** таблица из трёх ROC-AUC, пересечение user_id между train/test и рекомендация для production validation."""),
        code("""import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit

rng = np.random.default_rng(42)
n_users, events_per_user = 250, 8
user_id = np.repeat(np.arange(n_users), events_per_user)
timestamp = np.tile(np.arange(events_per_user), n_users)
user_effect = rng.normal(size=n_users)[user_id]
x = user_effect + 0.15 * timestamp + rng.normal(scale=.7, size=len(user_id))
y = (user_effect + 0.08 * timestamp + rng.normal(size=len(user_id)) > 0).astype(int)
df = pd.DataFrame({"user_id": user_id, "timestamp": timestamp, "x": x, "target": y})
""", role="setup"),
        code("""def fit_score(train_idx, test_idx):
    # TODO: обучите LogisticRegression на столбце x и верните ROC-AUC.
    raise NotImplementedError

# TODO: сформируйте три пары индексов: random, group, time.
splits = {}
"""),
        code("""# Self-check.
assert set(splits) == {"random", "group", "time"}
rows = []
for name, (train_idx, test_idx) in splits.items():
    overlap = len(set(df.iloc[train_idx].user_id) & set(df.iloc[test_idx].user_id))
    rows.append({"split": name, "roc_auc": fit_score(train_idx, test_idx), "user_overlap": overlap})
report = pd.DataFrame(rows)
assert report.roc_auc.between(0, 1).all()
assert report.loc[report.split == "group", "user_overlap"].item() == 0
report
""", role="self_check"),
        md("""## Вывод

Опишите единицу независимого наблюдения, production-сценарий и возможный gap. Высокая метрика random split сама по себе не доказывает leakage; покажите конкретный канал пересечения."""),
    ]),
    "04-tree-ensembles.ipynb": notebook("Дерево, Random Forest и boosting при одном бюджете", [
        md("""## Задача

Сравните `DecisionTreeClassifier`, `RandomForestClassifier` и `HistGradientBoostingClassifier` на одном датасете, одинаковых split и метрике. Помимо качества измерьте время fit и размер сериализованной модели.

**Deliverable:** таблица mean/std ROC-AUC, fit time, model bytes и объяснение выбранного baseline."""),
        code("""import pickle
import time
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42),
}
""", role="setup"),
        code("""# TODO: для каждой модели выполните один cross_validate с scoring='roc_auc'.
# Затем fit на полном X только для оценки pickle size. Сохраните строки в results.
results = []
"""),
        code("""report = pd.DataFrame(results)
required = {"model", "roc_auc_mean", "roc_auc_std", "fit_seconds", "model_bytes"}
assert required <= set(report.columns)
assert set(report.model) == set(models)
assert report.roc_auc_mean.between(0, 1).all()
report.sort_values("roc_auc_mean", ascending=False)
""", role="self_check"),
        md("""## Вывод

Объясните, где одно дерево остаётся уместным, почему Random Forest снижает variance и почему boosting нельзя объявлять победителем без latency/maintenance ограничений."""),
    ]),
    "05-broken-pytorch-loop.ipynb": notebook("Code review: сломанный PyTorch train loop", [
        md("""## Задача

Исправьте цикл ниже. В нём намеренно смешаны ошибки управления градиентами, режимов модели и validation. CPU достаточно.

**Deliverable:** исправленный loop, decreasing train loss, повторяемая validation-метрика и список найденных ошибок."""),
        code("""import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
X = torch.randn(512, 4)
y = ((X[:, 0] - .7 * X[:, 1] + .2 * X[:, 2]) > 0).long()
train_loader = DataLoader(TensorDataset(X[:400], y[:400]), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X[400:], y[400:]), batch_size=64, shuffle=False)
model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Dropout(.2), nn.Linear(16, 2))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()
""", role="setup"),
        code("""# BROKEN: сделайте code review, затем замените реализацию.
def train_epoch(model, loader):
    losses = []
    model.eval()
    for xb, yb in loader:
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.step()
        loss.backward()
        losses.append(loss.item())
    return float(np.mean(losses))

def validate(model, loader):
    model.train()
    correct = 0
    for xb, yb in loader:
        logits = model(xb)
        correct += (logits.argmax(1) == yb).sum().item()
    return correct / len(loader.dataset)
"""),
        code("""# Self-check после исправления.
history = [train_epoch(model, train_loader) for _ in range(8)]
score_1 = validate(model, val_loader)
score_2 = validate(model, val_loader)
assert history[-1] < history[0], history
assert score_1 == score_2, "validation должна быть детерминирована при неизменной модели"
assert score_1 > 0.75, score_1
print({"first_loss": history[0], "last_loss": history[-1], "val_accuracy": score_1})
""", role="self_check"),
        md("""## Review checklist

Для каждой ошибки укажите механизм: почему `step()` до `backward()` не обновляет текущий градиент, зачем обнулять gradients, чем `train()` отличается от `eval()`, и почему validation выполняют без записи autograd graph."""),
    ]),
    "06-correlated-importance.ipynb": notebook("Permutation importance при коррелированных признаках", [
        md("""## Задача

Создайте информативный признак и его почти точную копию. Обучите Random Forest и сравните permutation importance до и после удаления дубля. Объясните, почему низкая индивидуальная importance не означает отсутствие сигнала.

**Deliverable:** таблица importances, корреляционная матрица и вывод о границах метода."""),
        code("""import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(42)
X_raw, y = make_classification(
    n_samples=1500, n_features=5, n_informative=2, n_redundant=0,
    class_sep=1.1, random_state=42
)
X = pd.DataFrame(X_raw, columns=["signal", "feature_1", "feature_2", "feature_3", "feature_4"])
X["signal_copy"] = X["signal"] + rng.normal(scale=.02, size=len(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.3, random_state=42)
""", role="setup"),
        code("""def fit_and_importance(columns):
    # TODO: fit RandomForestClassifier(random_state=42), посчитайте ROC-AUC
    # и permutation_importance(scoring='roc_auc', n_repeats=10, random_state=42).
    # Верните DataFrame columns=[feature, importance_mean, importance_std] и auc.
    raise NotImplementedError

with_copy, auc_with_copy = fit_and_importance(list(X.columns))
without_copy, auc_without_copy = fit_and_importance([c for c in X.columns if c != "signal_copy"])
"""),
        code("""assert with_copy.importance_mean.notna().all()
assert without_copy.importance_mean.notna().all()
assert 0 <= auc_with_copy <= 1 and 0 <= auc_without_copy <= 1
print(X[["signal", "signal_copy"]].corr())
display(with_copy.sort_values("importance_mean", ascending=False))
display(without_copy.sort_values("importance_mean", ascending=False))
""", role="self_check"),
        md("""## Вывод

Ответьте, что именно измеряет permutation importance, почему коррелированный дубль меняет результат и какие дополнительные проверки нужны перед удалением признака. SHAP можно сравнить дополнительно, но это не обязательная часть лабораторной."""),
    ]),
}


from new_labs import build_new_labs


NEW_LABS = build_new_labs(md=md, code=code, notebook=notebook)
ALL_LABS = {**LABS, **NEW_LABS}

RELEASE_METADATA = {
    "01-metrics-threshold.ipynb": {"version": "1.0.0", "releaseStatus": "public", "datasets": []},
    "02-honest-binary-baseline.ipynb": {"version": "1.0.0", "releaseStatus": "public", "datasets": []},
    "03-leakage-splits.ipynb": {"version": "1.0.0", "releaseStatus": "public", "datasets": []},
    "04-tree-ensembles.ipynb": {"version": "1.0.0", "releaseStatus": "public", "datasets": []},
    "05-broken-pytorch-loop.ipynb": {"version": "1.0.0", "releaseStatus": "public", "datasets": []},
    "06-correlated-importance.ipynb": {"version": "1.0.0", "releaseStatus": "public", "datasets": []},
    "07-bike-demand-production-capstone.ipynb": {
        "version": "1.0.0",
        "releaseStatus": "public",
        "datasets": ["uci-bike-sharing-hour"],
    },
    "08-banking77-tfidf-error-analysis.ipynb": {
        "version": "1.0.0",
        "releaseStatus": "public",
        "datasets": ["banking77"],
    },
    "09-rag-failure-decomposition.ipynb": {
        "version": "1.0.0",
        "releaseStatus": "public",
        "datasets": ["squad2-rag-subset"],
    },
    "10-llm-serving-benchmark.ipynb": {"version": "0.1.0", "releaseStatus": "draft", "datasets": []},
    "advanced/10a-vllm-benchmark.ipynb": {
        "version": "0.1.0",
        "releaseStatus": "draft",
        "datasets": ["squad2-rag-subset"],
    },
    "advanced/10b-sglang-benchmark.ipynb": {
        "version": "0.1.0",
        "releaseStatus": "draft",
        "datasets": ["squad2-rag-subset"],
    },
}


def normalized_json(data: object, *, indent: int = 2) -> str:
    return json.dumps(data, ensure_ascii=False, indent=indent, sort_keys=True) + "\n"


def write_json(relative_path: str, data: object) -> str:
    payload = normalized_json(data)
    path = ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def benchmark_protocol() -> dict:
    return {
        "schemaVersion": 1,
        "protocolId": "squad2-first-32-serving-v1",
        "status": "draft",
        "promptSource": {
            "datasetId": "squad2-rag-subset",
            "selection": "first 32 answerable questions in document order; prompt contains context and question",
            "license": "CC-BY-SA-4.0",
        },
        "workload": {
            "requestCount": 32,
            "maxOutputTokens": 64,
            "temperature": 0,
            "seed": 42,
            "warmupRequests": 4,
            "concurrencyLevels": [1, 4, 16],
        },
        "comparabilityRequirements": [
            "same physical host and GPU model/count",
            "same model artifact and immutable revision",
            "same numerical precision and quantization",
            "same prompts, request order, output-token cap and concurrency",
            "same warmup policy and client-side measurement method",
        ],
        "requiredRunMetadata": [
            "provenance",
            "engine",
            "engineVersion",
            "engineCommit",
            "modelId",
            "modelRevision",
            "gpu",
            "driver",
            "cuda",
            "environmentLock",
            "startedAt",
            "rawFixturePath",
            "rawFixtureSha256",
        ],
        "canonicalMetrics": [
            "successfulRequests",
            "inputTokens",
            "outputTokens",
            "requestThroughput",
            "outputTokenThroughput",
            "ttftP50Ms",
            "ttftP95Ms",
            "tpotP50Ms",
            "tpotP95Ms",
        ],
    }


def main() -> None:
    notebook_hashes: dict[str, str] = {}
    for name, data in ALL_LABS.items():
        payload = json.dumps(data, ensure_ascii=False, indent=1) + "\n"
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        digest = hashlib.sha256(payload.encode()).hexdigest()
        notebook_hashes[name] = digest
        print(f"{name}  sha256:{digest}")

    protocol = benchmark_protocol()
    protocol_sha256 = write_json("data/llm-serving-benchmarks/protocol.json", protocol)
    write_json(
        "data/llm-serving-benchmarks/manifest.json",
        {
            "schemaVersion": 1,
            "status": "draft",
            "protocolPath": "data/llm-serving-benchmarks/protocol.json",
            "protocolSha256": protocol_sha256,
            "runs": [],
        },
    )

    dataset_manifest_path = ROOT / "data" / "datasets.json"
    if not dataset_manifest_path.exists():
        raise FileNotFoundError("Run scripts/sync_datasets.py before generating notebook manifests")
    dataset_manifest_sha256 = hashlib.sha256(dataset_manifest_path.read_bytes()).hexdigest()
    entries = []
    for path in ALL_LABS:
        release = RELEASE_METADATA[path]
        entries.append(
            {
                "path": path,
                "version": release["version"],
                "releaseStatus": release["releaseStatus"],
                "license": "MIT",
                "datasets": release["datasets"],
                "notebookSha256": notebook_hashes[path],
                "containsSolutions": False,
                "containsExecutionOutputs": False,
            }
        )
    write_json(
        "labs-manifest.json",
        {
            "schemaVersion": 1,
            "datasetManifest": {
                "path": "data/datasets.json",
                "sha256": dataset_manifest_sha256,
            },
            "publicIndex": [path for path in ALL_LABS if RELEASE_METADATA[path]["releaseStatus"] == "public"],
            "labs": entries,
        },
    )


if __name__ == "__main__":
    main()
