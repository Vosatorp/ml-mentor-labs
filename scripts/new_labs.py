from __future__ import annotations

from collections.abc import Callable


CellFactory = Callable[..., dict]


def build_new_labs(*, md: CellFactory, code: CellFactory, notebook: CellFactory) -> dict[str, dict]:
    """Return post-v1 starter notebooks without changing the original six specs."""

    return {
        "07-bike-demand-production-capstone.ipynb": notebook(
            "Итоговая работа: прогноз спроса на велосипеды на ближайший час",
            [
                md(
                    """## Задача

Постройте воспроизводимое базовое решение для прогноза спроса на прокат велосипедов на ближайший час и оформите его так, как если бы модель готовилась к первому запуску в продакшне. Каждая строка описывает один завершившийся час; при реальном применении признаки календаря и прогноз погоды формируются для следующего часа. Данные упорядочены по времени, поэтому случайное перемешивание здесь запрещено.

В таблице `cnt` — целевая переменная, а `casual` и `registered` — две части уже состоявшегося спроса. Их сумма равна `cnt`, поэтому использовать эти столбцы как признаки нельзя: в момент прогноза их ещё нет.

**Что должно получиться:** хронологическое разбиение на обучающую, проверочную и тестовую выборки (`train/validation/test`), три базовых решения, MAE и RMSLE, проверка утечки данных, срезы ошибок, краткая карточка модели, контракт инференса и три проверки сдвига данных.

Источник: UCI Bike Sharing Dataset, лицензия CC BY 4.0. Атрибуция и контрольные суммы находятся в `data/datasets.json`. CPU достаточно; API-ключи не нужны. В локальной копии репозитория используются уже сохранённые данные. Если открыть только notebook в Colab, подготовительная ячейка один раз скачает официальный ZIP по закреплённому адресу, проверит SHA-256 и дальше будет работать локально."""
                ),
                code(
                    """import hashlib
import io
import time
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_log_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import sparse


UCI_SOURCE_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
UCI_SOURCE_SHA256 = "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"
HOUR_CSV_SHA256 = "e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_sha256(payload: bytes, expected: str, label: str) -> None:
    actual = sha256(payload)
    if actual != expected:
        raise RuntimeError(f"{label}: checksum changed: {actual}")


def require_final_hostname(response, expected: str) -> None:
    actual = urlparse(response.geturl()).hostname
    if actual != expected:
        raise RuntimeError(f"unexpected redirect host: {actual}")


def resolve_bike_path() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        path = candidate / "data" / "uci-bike-sharing" / "hour.csv"
        if path.is_file():
            require_sha256(path.read_bytes(), HOUR_CSV_SHA256, "Bike Sharing hour.csv")
            return path

    cache_path = Path("/tmp/ml-mentor-labs-datasets-v1/uci-bike-sharing/hour.csv")
    if cache_path.is_file():
        require_sha256(cache_path.read_bytes(), HOUR_CSV_SHA256, "cached Bike Sharing hour.csv")
        return cache_path
    request = urllib.request.Request(UCI_SOURCE_URL, headers={"User-Agent": "ML-Mentor-Labs/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        require_final_hostname(response, "archive.ics.uci.edu")
        archive_payload = response.read()
    require_sha256(archive_payload, UCI_SOURCE_SHA256, "UCI Bike Sharing ZIP")
    with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
        hour_payload = archive.read("hour.csv")
    require_sha256(hour_payload, HOUR_CSV_SHA256, "Bike Sharing hour.csv from ZIP")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(hour_payload)
    return cache_path


BIKE_PATH = resolve_bike_path()
TARGET = "cnt"
LEAKAGE_COLUMNS = {"casual", "registered"}
NON_FEATURE_COLUMNS = {"instant", "dteday", TARGET, *LEAKAGE_COLUMNS}
CATEGORICAL_FEATURES = ["season", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
NUMERIC_FEATURES = ["yr", "temp", "atemp", "hum", "windspeed"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

df = pd.read_csv(BIKE_PATH, parse_dates=["dteday"]).sort_values(["dteday", "hr"]).reset_index(drop=True)
df["timestamp"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
assert df["timestamp"].is_monotonic_increasing
assert (df["casual"] + df["registered"] == df[TARGET]).all()
assert not (set(FEATURES) & NON_FEATURE_COLUMNS)
assert df[TARGET].ge(0).all()
""",
                    role="setup",
                ),
                md(
                    """## 1. Хронологическая проверка

Отделите последние 15% наблюдений в тестовую выборку, предшествующие 15% — в проверочную, остальное — в обучающую. Используйте закреплённые границы ниже: первые `TRAIN_END` строк, затем строки до `VALIDATION_END`, затем остаток. Так два решения будут сравниваться на одних и тех же 70/15/15 отрезках. Тестовую выборку не используйте при выборе признаков, модели и гиперпараметров."""
                ),
                code(
                    """TRAIN_END = int(len(df) * 0.70)
VALIDATION_END = int(len(df) * 0.85)


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # TODO: верните непересекающиеся train, validation и test в хронологическом порядке.
    raise NotImplementedError


train_df, validation_df, test_df = chronological_split(df)
"""
                ),
                md(
                    """## 2. Три базовых решения в одном бюджете

Сравните на одном разбиении и одних признаках три решения:

1. сезонный наивный прогноз — значение спроса ровно 168 часов назад в последовательной проверке;
2. Ridge с отдельной обработкой числовых и категориальных признаков;
3. `HistGradientBoostingRegressor(random_state=42)`.

У обучаемых моделей одинаковый бюджет: одно обучение без поиска по широкой сетке. `HistGradientBoostingRegressor` не принимает разреженную матрицу, поэтому его `OneHotEncoder` должен использовать `sparse_output=False` (либо нужна другая корректная плотная предобработка). Любые преобразования обучаются только на обучающей выборке.

Для сезонного прогноза разрешено использовать целевую переменную из проверочной или тестовой выборки только тогда, когда соответствующий час уже наступил: для прогноза в момент `t` источник обязан быть ровно `t − 168h` и строго раньше `t`. Это имитирует почасовую последовательную проверку (rolling backtest). Если брать историю только из обучающей выборки, для поздних недель базовый прогноз ошибочно исчезнет. В данных есть пропущенные часы. Если точного источника нет, используйте закреплённую политику `no_prediction`: оставьте `NaN`, измерьте покрытие и не подменяйте источник соседним или будущим наблюдением.

Считайте MAE в исходных единицах спроса и RMSLE после обрезки отрицательных прогнозов до нуля. Метрики сезонного решения считайте только там, где прогноз доступен, и всегда показывайте рядом покрытие. RMSLE сильнее наказывает относительную ошибку в часы с небольшим спросом."""
                ),
                code(
                    """MODEL_NAMES = ("ridge", "hist_gradient_boosting")
SEASONAL_MISSING_POLICY = "no_prediction"


def seasonal_naive(
    observations: pd.DataFrame,
    prediction_rows: pd.DataFrame,
    *,
    missing_policy: str = SEASONAL_MISSING_POLICY,
) -> tuple[np.ndarray, pd.DataFrame]:
    # TODO: сопоставьте каждому prediction_timestamp источник ровно t - 168h.
    # Верните predictions и audit columns=[prediction_timestamp, source_timestamp, source_value, available].
    # observations может содержать весь backtest, но lookup имеет право читать только source_timestamp < prediction_timestamp.
    # Поддержите только явную политику no_prediction: отсутствующий источник остаётся NaN.
    raise NotImplementedError


def build_model(name: str) -> Pipeline:
    # TODO: соберите Pipeline с шагами preprocess и model.
    # Ridge может получать sparse one-hot, HistGradientBoosting обязан получать dense matrix.
    raise NotImplementedError


def regression_report(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    # TODO: верните mae, rmsle, coverage и n_predictions. NaN-прогнозы исключите из метрик,
    # но обязательно учтите в coverage; для RMSLE обрежьте доступный прогноз снизу нулём.
    raise NotImplementedError


models = {name: build_model(name) for name in MODEL_NAMES}
# TODO: получите predictions всех трёх решений на validation и одну строку метрик на модель.
validation_predictions = {}
validation_results = pd.DataFrame(
    columns=["model", "mae", "rmsle", "coverage", "n_predictions", "fit_seconds"]
)
seasonal_validation_audit = None
# TODO: выберите лучшую обучаемую модель только по validation MAE. Затем сравните её
# с seasonal baseline и зафиксируйте решение до открытия test.
CANDIDATE_MODEL_NAME = None
VALIDATION_GO_NO_GO = {
    "candidate": None,
    "baseline": "seasonal_naive",
    "rule": "promote_candidate_only_if_validation_mae_is_lower",
    "outcome": None,  # promote_candidate | keep_seasonal_baseline
}
"""
                ),
                md(
                    """## 3. Финальная проверка

После фиксации решения переобучите кандидата среди обучаемых моделей на обучающей и проверочной выборках и ровно один раз посчитайте его метрики на тестовой. Отдельно посчитайте метрики и покрытие сезонного решения на тесте. Не меняйте `VALIDATION_GO_NO_GO` по тестовому результату. Для кандидата посчитайте срезы MAE на проверочной и тестовой выборках по часу суток, признаку рабочего дня и погодной категории. Небольшой выигрыш общей метрики может скрывать систематический провал на важном сегменте.

Сохраните кандидата через `joblib` и явно назовите его кандидатом: сериализация не означает автоматический выпуск. Опишите входную схему артефакта. Исторические столбцы погоды здесь содержат уже реализованные наблюдения; в реальном прогнозе следующего часа потребуются прогнозные признаки погоды. Зафиксируйте это как оптимистичное ограничение офлайн-оценки."""
                ),
                code(
                    """# TODO: после фиксации решения получите кандидата и оба test-отчёта.
candidate_model = None
candidate_test_predictions = None
candidate_test_metrics = None
seasonal_test_predictions = None
seasonal_test_audit = None
seasonal_test_metrics = None
FINAL_REPORT = None
ARTIFACT_PATH = Path("/tmp/ml-mentor-bike-demand-candidate.joblib")
# TODO: сериализуйте candidate_model в ARTIFACT_PATH. Это кандидат, а не подтверждение выпуска.
INPUT_SCHEMA = [
    # TODO: по одному dict(name, dtype, nullable) на каждый признак из FEATURES.
]


def mae_slices(frame: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray, *, split: str) -> pd.DataFrame:
    # TODO: верните columns=[split, dimension, value, n, mae].
    # Имена dimension: hr -> hour, workingday -> workingday, weathersit -> weather.
    raise NotImplementedError


validation_slices = None  # кандидат на validation
test_slices = None  # кандидат на test
"""
                ),
                md(
                    """## 4. Контракт и наблюдаемость

Заполните структуры ниже короткими проверяемыми формулировками. Карточка модели должна объяснять назначение и границы базового решения. Контракт инференса перечисляет входы, выход и поведение при ошибке. Для каждой проверки сдвига данных укажите сигнал, окно наблюдения и порог реакции."""
                ),
                code(
                    """# TODO: замените None конкретными значениями и формулировками.
MODEL_CARD = {
    "purpose": None,
    "target_horizon": None,
    "validation_scheme": None,
    "primary_metric": None,
    "deployment_status": None,
    "known_limitations": None,
}
INFERENCE_CONTRACT = {
    "required_features": None,
    "prediction_type": None,
    "invalid_input_policy": None,
}
DRIFT_CHECKS = [
    # TODO: минимум три dict с keys signal, window, threshold, action.
]
"""
                ),
                code(
                    """# Self-check: запускайте после выполнения всех TODO.
assert len(train_df) + len(validation_df) + len(test_df) == len(df)
assert train_df.index.is_unique and validation_df.index.is_unique and test_df.index.is_unique
assert len(train_df) == TRAIN_END
assert len(validation_df) == VALIDATION_END - TRAIN_END
assert len(test_df) == len(df) - VALIDATION_END
assert train_df["timestamp"].reset_index(drop=True).equals(df.iloc[:TRAIN_END]["timestamp"].reset_index(drop=True))
assert validation_df["timestamp"].reset_index(drop=True).equals(
    df.iloc[TRAIN_END:VALIDATION_END]["timestamp"].reset_index(drop=True)
)
assert test_df["timestamp"].reset_index(drop=True).equals(df.iloc[VALIDATION_END:]["timestamp"].reset_index(drop=True))
assert train_df["timestamp"].max() < validation_df["timestamp"].min() < test_df["timestamp"].min()
assert not (set(FEATURES) & LEAKAGE_COLUMNS)
assert set(models) == set(MODEL_NAMES)
assert set(validation_predictions) == {"seasonal_naive", *MODEL_NAMES}
assert all(len(prediction) == len(validation_df) for prediction in validation_predictions.values())
assert seasonal_validation_audit is not None
assert set(seasonal_validation_audit.columns) == {
    "prediction_timestamp", "source_timestamp", "source_value", "available",
}
assert len(seasonal_validation_audit) == len(validation_df)
assert seasonal_validation_audit["prediction_timestamp"].reset_index(drop=True).equals(
    validation_df["timestamp"].reset_index(drop=True)
)
assert (seasonal_validation_audit["source_timestamp"] < seasonal_validation_audit["prediction_timestamp"]).all()
assert (
    seasonal_validation_audit["prediction_timestamp"] - seasonal_validation_audit["source_timestamp"]
).eq(pd.Timedelta(hours=168)).all()
assert seasonal_validation_audit["available"].equals(seasonal_validation_audit["source_value"].notna())
assert seasonal_validation_audit["available"].all()
expected_validation_source = df.set_index("timestamp")[TARGET].reindex(
    seasonal_validation_audit["source_timestamp"]
).to_numpy(dtype=float)
assert np.allclose(seasonal_validation_audit["source_value"], expected_validation_source, equal_nan=True)
assert np.allclose(
    validation_predictions["seasonal_naive"], seasonal_validation_audit["source_value"], equal_nan=True
)
assert all({"preprocess", "model"} <= set(model.named_steps) for model in models.values())
ridge_matrix = models["ridge"].named_steps["preprocess"].transform(validation_df[FEATURES].head(4))
hist_matrix = models["hist_gradient_boosting"].named_steps["preprocess"].transform(validation_df[FEATURES].head(4))
assert sparse.issparse(ridge_matrix) or isinstance(ridge_matrix, np.ndarray)
assert isinstance(hist_matrix, np.ndarray) and not sparse.issparse(hist_matrix)
assert set(validation_results["model"]) == {"seasonal_naive", *MODEL_NAMES}
metric_columns = {"mae", "rmsle", "coverage", "n_predictions"}
assert {"model", *metric_columns, "fit_seconds"} <= set(validation_results.columns)
assert validation_results[[*metric_columns, "fit_seconds"]].ge(0).all().all()
assert validation_results["coverage"].between(0, 1).all()
seasonal_validation_row = validation_results.set_index("model").loc["seasonal_naive"]
assert np.isclose(seasonal_validation_row["coverage"], seasonal_validation_audit["available"].mean())
assert int(seasonal_validation_row["n_predictions"]) == int(seasonal_validation_audit["available"].sum())
assert CANDIDATE_MODEL_NAME in MODEL_NAMES
expected_candidate = validation_results.query("model in @MODEL_NAMES").sort_values(["mae", "model"]).iloc[0]["model"]
assert CANDIDATE_MODEL_NAME == expected_candidate
assert set(VALIDATION_GO_NO_GO) == {"candidate", "baseline", "rule", "outcome"}
assert VALIDATION_GO_NO_GO["candidate"] == CANDIDATE_MODEL_NAME
assert VALIDATION_GO_NO_GO["baseline"] == "seasonal_naive"
assert VALIDATION_GO_NO_GO["rule"] == "promote_candidate_only_if_validation_mae_is_lower"
candidate_validation_mae = validation_results.set_index("model").loc[CANDIDATE_MODEL_NAME, "mae"]
expected_outcome = (
    "promote_candidate" if candidate_validation_mae < seasonal_validation_row["mae"]
    else "keep_seasonal_baseline"
)
assert VALIDATION_GO_NO_GO["outcome"] == expected_outcome
assert candidate_model is not None and candidate_test_predictions is not None
assert len(candidate_test_predictions) == len(test_df)
assert set(candidate_test_metrics) == metric_columns
assert all(np.isfinite(list(candidate_test_metrics.values())))
assert np.isclose(candidate_test_metrics["coverage"], 1.0)
assert int(candidate_test_metrics["n_predictions"]) == len(test_df)
assert seasonal_test_predictions is not None and len(seasonal_test_predictions) == len(test_df)
assert seasonal_test_audit is not None and set(seasonal_test_audit.columns) == {
    "prediction_timestamp", "source_timestamp", "source_value", "available",
}
assert len(seasonal_test_audit) == len(test_df)
assert seasonal_test_audit["prediction_timestamp"].reset_index(drop=True).equals(
    test_df["timestamp"].reset_index(drop=True)
)
assert (seasonal_test_audit["source_timestamp"] < seasonal_test_audit["prediction_timestamp"]).all()
assert (
    seasonal_test_audit["prediction_timestamp"] - seasonal_test_audit["source_timestamp"]
).eq(pd.Timedelta(hours=168)).all()
assert seasonal_test_audit["available"].equals(seasonal_test_audit["source_value"].notna())
assert int((~seasonal_test_audit["available"]).sum()) == 39
expected_test_source = df.set_index("timestamp")[TARGET].reindex(
    seasonal_test_audit["source_timestamp"]
).to_numpy(dtype=float)
assert np.allclose(seasonal_test_audit["source_value"], expected_test_source, equal_nan=True)
assert np.allclose(seasonal_test_predictions, seasonal_test_audit["source_value"], equal_nan=True)
assert set(seasonal_test_metrics) == metric_columns
assert np.isclose(seasonal_test_metrics["coverage"], seasonal_test_audit["available"].mean())
assert int(seasonal_test_metrics["n_predictions"]) == int(seasonal_test_audit["available"].sum())
assert FINAL_REPORT is not None and {
    "validation_go_no_go", "candidate_test", "seasonal_test",
    "seasonal_missing_policy", "seasonal_test_missing",
} <= set(FINAL_REPORT)
assert FINAL_REPORT["validation_go_no_go"] == VALIDATION_GO_NO_GO
assert FINAL_REPORT["candidate_test"] == candidate_test_metrics
assert FINAL_REPORT["seasonal_test"] == seasonal_test_metrics
assert FINAL_REPORT["seasonal_missing_policy"] == SEASONAL_MISSING_POLICY
assert FINAL_REPORT["seasonal_test_missing"] == 39
assert ARTIFACT_PATH.is_file()
restored_model = joblib.load(ARTIFACT_PATH)
assert len(restored_model.predict(test_df[FEATURES].head(2))) == 2
assert {item["name"] for item in INPUT_SCHEMA} == set(FEATURES)
assert all({"name", "dtype", "nullable"} == set(item) for item in INPUT_SCHEMA)
assert all(item["dtype"] == str(df[item["name"]].dtype) for item in INPUT_SCHEMA)
assert all(isinstance(item["nullable"], bool) for item in INPUT_SCHEMA)
expected_slice_columns = {"split", "dimension", "value", "n", "mae"}
assert validation_slices is not None and set(validation_slices.columns) == expected_slice_columns
assert test_slices is not None and set(test_slices.columns) == expected_slice_columns
assert set(validation_slices["dimension"]) == {"hour", "workingday", "weather"}
assert set(test_slices["dimension"]) == {"hour", "workingday", "weather"}
assert set(validation_slices["split"]) == {"validation"} and set(test_slices["split"]) == {"test"}
assert validation_slices[["n", "mae"]].ge(0).all().all()
assert test_slices[["n", "mae"]].ge(0).all().all()
assert validation_slices.groupby("dimension")["n"].sum().eq(len(validation_df)).all()
assert test_slices.groupby("dimension")["n"].sum().eq(len(test_df)).all()
assert all(MODEL_CARD.values()) and all(INFERENCE_CONTRACT.values())
limitations = str(MODEL_CARD["known_limitations"]).lower()
assert "погод" in limitations and "прогноз" in limitations and "оптимист" in limitations
assert len(DRIFT_CHECKS) >= 3
assert all({"signal", "window", "threshold", "action"} <= set(item) for item in DRIFT_CHECKS)
print({"validation": validation_results, "final_report": FINAL_REPORT})
""",
                    role="self_check",
                ),
            ],
        ),
        "08-banking77-tfidf-error-analysis.ipynb": notebook(
            "Banking77: TF-IDF baseline и разбор ошибок",
            [
                md(
                    """## Задача

Обучите прозрачное базовое решение для классификации банковских обращений по 77 намерениям. Здесь важна не только итоговая метрика: найдите классы, которые модель путает, прочитайте конкретные ошибки и предложите следующий эксперимент.

Официальную обучающую часть разделите на обучающую и проверочную выборки. Официальную тестовую часть откройте один раз после фиксации конвейера и правил анализа.

**Что должно получиться:** честное сравнение словных n-грамм длиной 1–2 и символьных `char_wb` n-грамм при одинаковом бюджете подбора, логистическая регрессия, доля правильных ответов (accuracy) и macro-F1, полнота по каждому классу, пять наиболее частых пар ошибок, ручная типология минимум 20 ошибок и один обоснованный следующий эксперимент.

Источник: Banking77 из официального репозитория PolyAI, ревизия закреплена в `data/datasets.json`, лицензия CC BY 4.0. CPU достаточно; API-ключи не нужны. В локальной копии репозитория используются сохранённые данные. Если открыть только notebook в Colab, подготовительная ячейка один раз скачает три официальных файла из закреплённого commit, проверит SHA-256 и дальше будет работать локально."""
                ),
                code(
                    """import hashlib
import json
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BANKING_FILES = {
    "train.csv": (
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/train.csv",
        "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    ),
    "test.csv": (
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/test.csv",
        "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    ),
    "categories.json": (
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/categories.json",
        "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
    ),
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_sha256(payload: bytes, expected: str, label: str) -> None:
    actual = sha256(payload)
    if actual != expected:
        raise RuntimeError(f"{label}: checksum changed: {actual}")


def require_final_hostname(response, expected: str) -> None:
    actual = urlparse(response.geturl()).hostname
    if actual != expected:
        raise RuntimeError(f"unexpected redirect host: {actual}")


def resolve_banking_dir() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        directory = candidate / "data" / "banking77"
        if all((directory / name).is_file() for name in BANKING_FILES):
            for name, spec in BANKING_FILES.items():
                require_sha256((directory / name).read_bytes(), spec[1], f"Banking77 {name}")
            return directory

    directory = Path("/tmp/ml-mentor-labs-datasets-v1/banking77")
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, expected_sha256) in BANKING_FILES.items():
        path = directory / name
        if path.is_file():
            try:
                require_sha256(path.read_bytes(), expected_sha256, f"cached Banking77 {name}")
                continue
            except RuntimeError:
                path.unlink()
        request = urllib.request.Request(url, headers={"User-Agent": "ML-Mentor-Labs/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            require_final_hostname(response, "raw.githubusercontent.com")
            payload = response.read()
        require_sha256(payload, expected_sha256, f"Banking77 {name}")
        path.write_bytes(payload)
    return directory


DATA_DIR = resolve_banking_dir()
OFFICIAL_TEST_PATH = DATA_DIR / "test.csv"
official_train = pd.read_csv(DATA_DIR / "train.csv")
categories = json.loads((DATA_DIR / "categories.json").read_text(encoding="utf-8"))

train_df, validation_df = train_test_split(
    official_train,
    test_size=0.2,
    stratify=official_train["category"],
    random_state=42,
)
assert official_train["category"].nunique() == len(categories) == 77
assert set(train_df.index).isdisjoint(validation_df.index)
""",
                    role="setup",
                ),
                md(
                    """## 1. Сравните два представления текста

Соберите два семейства конвейеров (`Pipeline`) из `TfidfVectorizer` и `LogisticRegression`: словные n-граммы длиной 1–2 и символьные `char_wb` n-граммы длиной 3–5. Назовите шаги `tfidf` и `classifier`. В обоих вариантах используйте `min_df=2`, `max_features=50_000`, `sublinear_tf=True` и логистическую регрессию с `max_iter=1000`, `random_state=42`. Меняется только представление текста и одно из двух значений `C`. Это одинаковый бюджет подбора: по два обучения на вариант, одно разбиение и одна метрика выбора. Все решения принимайте по macro-F1 на проверочной выборке; при равенстве используйте accuracy, затем имя варианта в алфавитном порядке и меньшее `C`. Не подглядывайте в тестовую выборку."""
                ),
                code(
                    """VARIANTS = ("word_1_2", "char_wb")
C_VALUES = (0.5, 2.0)


def build_intent_model(variant: str, *, c: float) -> Pipeline:
    # TODO: соберите TF-IDF + Logistic Regression pipeline для указанного variant.
    raise NotImplementedError


# TODO: выполните ровно len(C_VALUES) fit на каждый variant и сохраните все результаты.
candidate_results = pd.DataFrame(columns=["variant", "c", "accuracy", "macro_f1"])
candidate_models = {}
# TODO: выберите одну конфигурацию по validation macro-F1 и получите validation_pred.
SELECTED_VARIANT = None
SELECTED_C = None
model = None
validation_pred = None
validation_metrics = None
"""
                ),
                md(
                    """## 2. Разберите ошибки

Macro-F1 одинаково учитывает редкие и частые классы, поэтому не ограничивайтесь accuracy. Постройте таблицу полноты (recall) по классам и матрицу ошибок без диагонали. Затем прочитайте ошибочные тексты и присвойте им понятные категории, например: близкие намерения, слишком короткий запрос, неоднозначность или лексика, которой не было в обучающей выборке."""
                ),
                code(
                    """def per_class_recall(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    # TODO: верните columns=[category, support, recall], по одному ряду на класс.
    raise NotImplementedError


def top_confusions(y_true: pd.Series, y_pred: np.ndarray, *, limit: int = 5) -> pd.DataFrame:
    # TODO: верните самые частые ошибки columns=[actual, predicted, count], без диагонали.
    raise NotImplementedError


recall_table = None
confusion_pairs = None

# TODO: отберите минимум 20 ошибочных validation-примеров и заполните error_type и note.
error_audit = pd.DataFrame(columns=["text", "actual", "predicted", "error_type", "note"])
NEXT_EXPERIMENT = None
"""
                ),
                md(
                    """## 3. Одна финальная проверка на тестовой выборке

Когда конвейер, метрики и типология ошибок зафиксированы, прочитайте официальную тестовую часть, переобучите модель на полной официальной обучающей части и посчитайте test accuracy и macro-F1. После этого не меняйте решение по тестовому результату."""
                ),
                code(
                    """# TODO: выполните этот блок только после фиксации решения.
official_test = None
final_model = None
test_pred = None
test_metrics = None
"""
                ),
                code(
                    """# Self-check: запускайте после выполнения всех TODO.
assert model is not None and validation_pred is not None
assert len(validation_pred) == len(validation_df)
assert len(candidate_results) == len(VARIANTS) * len(C_VALUES)
assert set(candidate_results["variant"]) == set(VARIANTS)
assert candidate_results.groupby("variant").size().eq(len(C_VALUES)).all()
assert set(candidate_results["c"]) == set(C_VALUES)
expected_candidates = {(variant, c) for variant in VARIANTS for c in C_VALUES}
assert set(candidate_models) == expected_candidates
assert all({"tfidf", "classifier"} <= set(candidate.named_steps) for candidate in candidate_models.values())
assert all(candidate.named_steps["tfidf"].min_df == 2 for candidate in candidate_models.values())
assert all(candidate.named_steps["tfidf"].max_features == 50_000 for candidate in candidate_models.values())
assert all(candidate.named_steps["tfidf"].sublinear_tf is True for candidate in candidate_models.values())
assert all(candidate.named_steps["classifier"].max_iter == 1000 for candidate in candidate_models.values())
assert all(candidate.named_steps["classifier"].random_state == 42 for candidate in candidate_models.values())
assert all(np.isclose(candidate.named_steps["classifier"].C, c) for (_, c), candidate in candidate_models.items())
assert candidate_models[("word_1_2", C_VALUES[0])].named_steps["tfidf"].analyzer == "word"
assert candidate_models[("word_1_2", C_VALUES[0])].named_steps["tfidf"].ngram_range == (1, 2)
assert candidate_models[("char_wb", C_VALUES[0])].named_steps["tfidf"].analyzer == "char_wb"
assert candidate_models[("char_wb", C_VALUES[0])].named_steps["tfidf"].ngram_range == (3, 5)
assert candidate_results[["accuracy", "macro_f1"]].apply(lambda column: column.between(0, 1).all()).all()
assert SELECTED_VARIANT in VARIANTS and SELECTED_C in C_VALUES
expected_best = candidate_results.sort_values(
    ["macro_f1", "accuracy", "variant", "c"],
    ascending=[False, False, True, True],
).iloc[0]
assert (SELECTED_VARIANT, SELECTED_C) == (expected_best["variant"], expected_best["c"])
assert model is candidate_models[(SELECTED_VARIANT, SELECTED_C)]
assert set(validation_metrics) == {"accuracy", "macro_f1"}
assert 0 <= validation_metrics["accuracy"] <= 1
assert 0 <= validation_metrics["macro_f1"] <= 1
assert recall_table is not None and set(recall_table.columns) == {"category", "support", "recall"}
assert len(recall_table) == 77 and recall_table["recall"].between(0, 1).all()
assert confusion_pairs is not None and set(confusion_pairs.columns) == {"actual", "predicted", "count"}
assert len(confusion_pairs) == 5
assert (confusion_pairs["actual"] != confusion_pairs["predicted"]).all()
assert confusion_pairs["count"].gt(0).all()
assert confusion_pairs["count"].tolist() == sorted(confusion_pairs["count"], reverse=True)
assert len(error_audit) >= 20
assert error_audit[["error_type", "note"]].notna().all().all()
assert (error_audit["actual"] != error_audit["predicted"]).all()
assert isinstance(NEXT_EXPERIMENT, str) and NEXT_EXPERIMENT.strip()
assert official_test is not None and final_model is not None and test_pred is not None
assert len(test_pred) == len(official_test)
assert set(test_metrics) == {"accuracy", "macro_f1"}
print({"validation": validation_metrics, "test": test_metrics})
""",
                    role="self_check",
                ),
            ],
        ),
        "09-rag-failure-decomposition.ipynb": notebook(
            "RAG: разделяем ошибки поиска и ответа",
            [
                md(
                    """## Задача

Соберите небольшое воспроизводимое базовое RAG-решение и научитесь отвечать на главный диагностический вопрос: система не нашла нужный документ или нашла, но неправильно извлекла ответ?

Это намеренно простое локальное решение на TF-IDF и лексическом извлечении. Оно не имитирует качество LLM и нужно для проверки контура оценки до подключения дорогой генерации.

**Что должно получиться:** Recall@1/3 и MRR поиска, качество ответа и отказа от ответа, порог отказа, выбранный только по проверочной выборке, и взаимоисключающая таблица исходов: `retrieval_miss`, `bad_context`, `unsupported_claim`, `answering_failure`, `correct_abstention`, `success`.

Используется фиксированная адаптированная подвыборка SQuAD 2.0, лицензия CC BY-SA 4.0. Атрибуция, описание изменений, правило отбора и контрольная сумма находятся в `data/squad2/` и `data/datasets.json`. CPU достаточно; API-ключи не нужны. В локальной копии репозитория данные читаются без сети. Если открыть только notebook в Colab, подготовительная ячейка скачает официальный `dev-v2.0.json`, проверит SHA-256, воспроизведёт ту же подвыборку и проверит её итоговую SHA-256."""
                ),
                code(
                    """import hashlib
import json
import re
import string
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


SQUAD_SOURCE_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
SQUAD_SOURCE_SHA256 = "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8"
SQUAD_SUBSET_SHA256 = "053396eeff7f1c07d473f2bd62c4d0a28f5d34d37f49d5ffd7922a58fbb5494f"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_sha256(payload: bytes, expected: str, label: str) -> None:
    actual = sha256(payload)
    if actual != expected:
        raise RuntimeError(f"{label}: checksum changed: {actual}")


def require_final_hostname(response, expected: str) -> None:
    actual = urlparse(response.geturl()).hostname
    if actual != expected:
        raise RuntimeError(f"unexpected redirect host: {actual}")


def normalized_json(data: object) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\\n").encode("utf-8")


def build_squad_subset(source: bytes) -> bytes:
    raw = json.loads(source)
    documents_payload = []
    impossible_examples = 0
    for article in raw["data"]:
        for paragraph in article["paragraphs"]:
            answerable = [qa for qa in paragraph["qas"] if not qa["is_impossible"]]
            impossible = [qa for qa in paragraph["qas"] if qa["is_impossible"]]
            if not answerable:
                continue
            selected = [answerable[0]]
            if impossible and impossible_examples < 50:
                selected.append(impossible[0])
                impossible_examples += 1
            qas = []
            for qa in selected:
                seen = set()
                answers = []
                for answer in qa["answers"]:
                    key = (answer["text"], answer["answer_start"])
                    if key not in seen:
                        seen.add(key)
                        answers.append({"text": key[0], "answer_start": key[1]})
                qas.append(
                    {
                        "answers": answers,
                        "is_impossible": qa["is_impossible"],
                        "question": qa["question"],
                        "source_qa_id": qa["id"],
                    }
                )
            documents_payload.append(
                {
                    "context": paragraph["context"],
                    "document_id": f"squad2-{len(documents_payload):03d}",
                    "qas": qas,
                    "title": article["title"],
                }
            )
            if len(documents_payload) == 100:
                break
        if len(documents_payload) == 100:
            break
    if len(documents_payload) != 100 or impossible_examples != 50:
        raise RuntimeError("SQuAD subset source no longer satisfies the pinned selection recipe")
    return normalized_json(
        {
            "attribution": "Rajpurkar, Jia et al. Stanford Question Answering Dataset 2.0.",
            "dataset": "SQuAD 2.0",
            "documents": documents_payload,
            "license": "CC-BY-SA-4.0",
            "modificationNotice": (
                "Deterministic educational subset: first 100 eligible contexts in source order; "
                "one answerable QA per context; one impossible QA for the first 50 eligible contexts. "
                "Duplicate answer spans removed and fields normalized; source text unchanged."
            ),
            "selection": {
                "answerable_per_document": 1,
                "documents": 100,
                "impossible_examples": 50,
                "policy": "source order; first answerable QA per document; first impossible QA for first 50 eligible documents",
            },
            "source_sha256": SQUAD_SOURCE_SHA256,
            "source_version": raw["version"],
        }
    )


def resolve_squad_subset() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        path = candidate / "data" / "squad2" / "squad2-rag-subset.json"
        if path.is_file():
            require_sha256(path.read_bytes(), SQUAD_SUBSET_SHA256, "local SQuAD subset")
            return path
    cache_path = Path("/tmp/ml-mentor-labs-datasets-v1/squad2/squad2-rag-subset.json")
    if cache_path.is_file():
        try:
            require_sha256(cache_path.read_bytes(), SQUAD_SUBSET_SHA256, "cached SQuAD subset")
            return cache_path
        except RuntimeError:
            cache_path.unlink()
    request = urllib.request.Request(SQUAD_SOURCE_URL, headers={"User-Agent": "ML-Mentor-Labs/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        require_final_hostname(response, "rajpurkar.github.io")
        source = response.read()
    require_sha256(source, SQUAD_SOURCE_SHA256, "SQuAD 2.0 source")
    subset = build_squad_subset(source)
    require_sha256(subset, SQUAD_SUBSET_SHA256, "generated SQuAD subset")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(subset)
    return cache_path


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\\b(a|an|the)\\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, references: list[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not references:
        return float(not pred_tokens)
    scores = []
    for reference in references:
        ref_tokens = normalize_answer(reference).split()
        common = sum((min(pred_tokens.count(token), ref_tokens.count(token)) for token in set(pred_tokens)))
        if not pred_tokens or not ref_tokens:
            scores.append(float(pred_tokens == ref_tokens))
            continue
        precision = common / len(pred_tokens)
        recall = common / len(ref_tokens)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return max(scores, default=0.0)


SQUAD_SUBSET_PATH = resolve_squad_subset()
raw = json.loads(SQUAD_SUBSET_PATH.read_text(encoding="utf-8"))
assert raw["license"] == "CC-BY-SA-4.0" and raw["modificationNotice"]
documents = pd.DataFrame(
    [{"document_id": item["document_id"], "title": item["title"], "context": item["context"]} for item in raw["documents"]]
)
questions = pd.DataFrame(
    [
        {
            "document_id": item["document_id"],
            "question": qa["question"],
            "answers": [answer["text"] for answer in qa["answers"]],
            "is_impossible": qa["is_impossible"],
            "source_qa_id": qa["source_qa_id"],
        }
        for item in raw["documents"]
        for qa in item["qas"]
    ]
)
validation_mask = questions.groupby("is_impossible", sort=False).cumcount().mod(2).eq(0)
validation_questions = questions[validation_mask].reset_index(drop=True)
test_questions = questions[~validation_mask].reset_index(drop=True)
assert len(documents) == 100 and len(questions) == 150
assert set(validation_questions["source_qa_id"]).isdisjoint(test_questions["source_qa_id"])
assert validation_questions["is_impossible"].value_counts().to_dict() == {False: 50, True: 25}
assert test_questions["is_impossible"].value_counts().to_dict() == {False: 50, True: 25}
""",
                    role="setup",
                ),
                md(
                    """## 1. Поиск документов

Обучите TF-IDF только на корпусе документов. Для каждого вопроса верните ранжированный список `document_id` и оценку сходства. Recall@k считается по наличию исходного документа SQuAD в первых k результатах. MRR (mean reciprocal rank) усредняет обратную позицию исходного документа: 1 для первого места, 1/2 для второго и 0, если документ не найден в пределах оценочного списка. Это упрощённая измеримая проверка для лабораторной, а не универсальная оценка поиска в продакшне."""
                ),
                code(
                    """def build_retriever(corpus: pd.DataFrame):
    # TODO: верните обученный TfidfVectorizer и матрицу документов.
    raise NotImplementedError


def retrieve(question: str, *, top_k: int = 10) -> list[dict]:
    # TODO: верните список dict(document_id, score, context), score по убыванию.
    raise NotImplementedError


vectorizer, document_matrix = build_retriever(documents)
"""
                ),
                md(
                    """## 2. Простой ответ и отказ

Используйте готовую прозрачную функцию лексического ответа: она получает вопрос и первый найденный контекст, выбирает подходящее предложение, а затем короткий числовой, именной или связочный фрагмент. Это ограниченный англоязычный baseline с понятными правилами, а не скрытая QA-модель. Улучшать его не нужно — центральная работа лабораторной посвящена поиску, метрикам и классификации ошибок.

Функция возвращает фрагмент и оценку уверенности (`confidence`). Если уверенность ниже выбранного порога, итоговый ответ должен быть пустой строкой. Порог подбирается по сетке от 0 до 1 только на проверочной выборке, затем замораживается. Критерий выбора — среднее между долей успешных ответов на вопросах с ответом и долей корректных отказов на вопросах без ответа. Чтобы критерий не выбрал бесполезную систему «всегда отказываться», заранее зафиксировано ограничение: ответ должен быть выдан минимум для 60% проверочных вопросов с ответом. Не проходящие его пороги не участвуют в выборе. При равенстве выбирается меньший порог.

Порог успешного ответа `ANSWER_SUCCESS_F1 = 0.5` и минимальное покрытие `MIN_ANSWER_COVERAGE = 0.60` заданы до тестовой выборки и по её результатам не меняются. 60% — учебный продуктовый контракт именно этой лабораторной, а не универсальная рекомендация. В реальной системе минимальное покрытие заранее задаёт владелец продукта с учётом цены ошибки и допустимой доли отказов.

Не выдавайте это решение за генеративную модель: его задача — сделать причины ошибок видимыми. Классифицируйте каждый пример ровно в одну категорию:

- `retrieval_miss`: исходного документа нет в top-k;
- `bad_context`: исходный документ есть в top-k, но top-1 контекст другой;
- `unsupported_claim`: для вопроса без ответа система вернула непустой текст;
- `answering_failure`: top-1 контекст правильный, но token-F1 ответа ниже порога;
- `correct_abstention`: система корректно отказалась отвечать на вопрос без ответа;
- `success`: ответ на answerable-вопрос достиг порога token-F1.

Для вопросов с полем `is_impossible=true` сначала оценивается корректность отказа, поэтому они попадают в `unsupported_claim` или `correct_abstention` и не смешиваются с ошибками на вопросах, у которых есть ответ."""
                ),
                code(
                    """ANSWER_SUCCESS_F1 = 0.5
MIN_ANSWER_COVERAGE = 0.60
EVALUATION_TOP_K = 10
THRESHOLD_GRID = np.linspace(0.0, 1.0, 21)
LEXICAL_STOPWORDS = {
    "a", "an", "the", "what", "when", "where", "which", "who", "whom", "whose",
    "why", "how", "did", "does", "do", "is", "are", "was", "were", "be", "been",
    "in", "on", "at", "to", "of", "for", "from", "with", "and", "or", "by", "as",
    "this", "that", "it", "its", "their", "his", "her", "name", "called",
}


def lexical_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.-]*", text)


def extract_answer(question: str, context: str) -> tuple[str, float]:
    '''Прозрачный baseline: подходящее предложение, затем короткий типизированный фрагмент.

    Правила намеренно ограничены английскими вопросами SQuAD. Это scaffold для измеримого
    RAG-контура; студенту не нужно изобретать генератор ответа.
    '''
    question_tokens = set(normalize_answer(question).split()) - LEXICAL_STOPWORDS
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\\s+", context) if item.strip()]
    if not sentences or not question_tokens:
        return "", 0.0

    def sentence_overlap(candidate: str) -> float:
        candidate_tokens = set(normalize_answer(candidate).split())
        return len(question_tokens & candidate_tokens) / len(question_tokens)

    sentence = max(sentences, key=sentence_overlap)
    overlap = sentence_overlap(sentence)
    question_lower = question.lower()
    if re.search(r"\\b(when|what year|how many|how much|how long)\\b", question_lower):
        intent = "number"
    elif re.search(r"\\b(who|whose|what person|what (?:is|was) the name)\\b", question_lower):
        intent = "person"
    elif re.search(r"\\b(where|what country|what city|what place)\\b", question_lower):
        intent = "place"
    else:
        intent = "other"

    candidates: list[tuple[str, str, float]] = []

    def add_candidate(text: str, kind: str, prior: float) -> None:
        cleaned = " ".join(text.strip(" \\t\\n,;:.!?()[]").split())
        words = lexical_tokens(cleaned)
        if not words:
            return
        cleaned = " ".join(words[:6])
        if len(cleaned) <= 80:
            candidates.append((cleaned, kind, prior))

    number_pattern = r"(?<!\\w)(?:\\d{3,4}s?|\\d+(?:\\.\\d+)?(?:\\s*(?:%|percent|million|billion|years?|months?|days?|hours?))?)(?!\\w)"
    for match in re.finditer(number_pattern, sentence, re.IGNORECASE):
        add_candidate(match.group(), "number", 1.0)
    for match in re.finditer(r'["“]([^"”]{1,80})["”]', sentence):
        add_candidate(match.group(1), "quote", 0.6)
    proper_pattern = r"\\b(?:[A-Z][\\w'’.-]*(?:\\s+(?:(?:of|the|de|von|and)\\s+)?[A-Z][\\w'’.-]*){0,4})\\b"
    for match in re.finditer(proper_pattern, sentence):
        add_candidate(match.group(), "proper", 0.8)
    relation_pattern = r"\\b(?:is|are|was|were|became|becomes|mean(?:s|t)?|called|known as|named)\\s+([^,.;:!?]{1,80})"
    for match in re.finditer(relation_pattern, sentence, re.IGNORECASE):
        words = lexical_tokens(match.group(1))
        for length in range(1, min(6, len(words)) + 1):
            add_candidate(" ".join(words[:length]), "relation", 0.7)
    if intent == "place":
        for match in re.finditer(r"\\b(?:in|at|near|from|into|on)\\s+([^,.;:!?]{1,60})", sentence, re.IGNORECASE):
            words = lexical_tokens(match.group(1))
            for length in range(1, min(5, len(words)) + 1):
                add_candidate(" ".join(words[:length]), "place", 1.0)

    sentence_words = lexical_tokens(sentence)
    for start, word in enumerate(sentence_words):
        if re.fullmatch(r"\\d{2,4}s?", word) or (word[:1].isupper() and word.lower() not in LEXICAL_STOPWORDS):
            for length in range(1, min(5, len(sentence_words) - start) + 1):
                add_candidate(" ".join(sentence_words[start : start + length]), "window", 0.3)

    if not candidates:
        fallback = " ".join(sentence_words[:6])
        return fallback, float(np.clip(0.55 * overlap + 0.15, 0.0, 1.0))

    def candidate_score(item: tuple[str, str, float]) -> float:
        text, kind, prior = item
        words = normalize_answer(text).split()
        novelty = sum(word not in question_tokens for word in words) / max(len(words), 1)
        length_preference = 1 - min(abs(len(words) - 2) / 5, 1)
        typed = (
            (intent == "number" and kind == "number")
            or (intent == "person" and kind == "proper")
            or (intent == "place" and kind in {"place", "proper"})
        )
        echo = len(set(words) & question_tokens) / max(len(words), 1)
        return 2.0 * typed + prior + 0.45 * novelty + 0.15 * length_preference - 0.30 * echo

    answer, kind, _ = max(candidates, key=candidate_score)
    typed_span = float(
        (intent == "number" and kind == "number")
        or (intent == "person" and kind == "proper")
        or (intent == "place" and kind in {"place", "proper"})
    )
    confidence = float(np.clip(0.55 * overlap + 0.15 * typed_span + 0.15, 0.0, 1.0))
    return answer, confidence


def evaluate_questions(frame: pd.DataFrame, *, abstain_threshold: float, top_k: int = EVALUATION_TOP_K) -> pd.DataFrame:
    # TODO: для каждого вопроса сохраните retrieved_ids, gold_rank, reciprocal_rank,
    # retrieval_hit_at_1/3, prediction, answer_f1, correct_abstention и один outcome
    # из шести bucket, описанных выше.
    raise NotImplementedError


# TODO: заполните таблицу для всей THRESHOLD_GRID. answer_coverage — доля answerable-вопросов
# с непустым prediction; eligible означает answer_coverage >= MIN_ANSWER_COVERAGE.
# decision_score — среднее answer_success_rate и correct_abstention_rate.
threshold_results = pd.DataFrame(
    columns=[
        "threshold", "answer_coverage", "answer_success_rate",
        "correct_abstention_rate", "decision_score", "eligible",
    ]
)
ABSTAIN_THRESHOLD = None
validation_report = None
"""
                ),
                md(
                    """## 3. Замороженная проверка

После выбора порога один раз примените тот же контур к тестовой выборке. Отдельно покажите Recall@1/3, MRR по полному оценочному top-k, средний token-F1 на вопросах с ответом, долю правильных отказов и распределение всех фактически встретившихся исходов. Выведите до пяти примеров из каждой непустой категории ошибок. Не добавляйте искусственные строки ради заполнения категории."""
                ),
                code(
                    """# TODO: ABSTAIN_THRESHOLD больше не меняется после этого места.
test_report = None
summary = None
failure_examples = None
"""
                ),
                code(
                    """# Self-check: запускайте после выполнения всех TODO.
assert vectorizer is not None and document_matrix.shape[0] == len(documents)
probe = retrieve(validation_questions.iloc[0]["question"], top_k=EVALUATION_TOP_K)
assert len(probe) == EVALUATION_TOP_K and all({"document_id", "score", "context"} <= set(item) for item in probe)
assert [item["score"] for item in probe] == sorted([item["score"] for item in probe], reverse=True)
assert ABSTAIN_THRESHOLD is not None and 0 <= ABSTAIN_THRESHOLD <= 1
assert len(threshold_results) == len(THRESHOLD_GRID)
assert set(threshold_results.columns) == {
    "threshold", "answer_coverage", "answer_success_rate",
    "correct_abstention_rate", "decision_score", "eligible",
}
assert np.allclose(sorted(threshold_results["threshold"]), THRESHOLD_GRID)
assert threshold_results[
    ["answer_coverage", "answer_success_rate", "correct_abstention_rate", "decision_score"]
].apply(
    lambda column: column.between(0, 1).all()
).all()
assert threshold_results["eligible"].equals(
    threshold_results["answer_coverage"].ge(MIN_ANSWER_COVERAGE)
)
assert threshold_results["eligible"].any()
assert np.allclose(
    threshold_results["decision_score"],
    (threshold_results["answer_success_rate"] + threshold_results["correct_abstention_rate"]) / 2,
)
expected_threshold = threshold_results[threshold_results["eligible"]].sort_values(
    ["decision_score", "threshold"], ascending=[False, True]
).iloc[0]["threshold"]
assert np.isclose(ABSTAIN_THRESHOLD, expected_threshold)
required = {
    "source_qa_id", "retrieved_ids", "gold_rank", "reciprocal_rank",
    "retrieval_hit_at_1", "retrieval_hit_at_3", "prediction", "confidence",
    "answer_f1", "correct_abstention", "outcome",
}
assert validation_report is not None and required <= set(validation_report.columns)
assert test_report is not None and required <= set(test_report.columns)
assert validation_report["source_qa_id"].tolist() == validation_questions["source_qa_id"].tolist()
assert test_report["source_qa_id"].tolist() == test_questions["source_qa_id"].tolist()
allowed_outcomes = {
    "retrieval_miss", "bad_context", "unsupported_claim",
    "answering_failure", "correct_abstention", "success",
}
assert set(test_report["outcome"]) <= allowed_outcomes
assert test_report["reciprocal_rank"].between(0, 1).all()
assert (test_report["retrieval_hit_at_1"] <= test_report["retrieval_hit_at_3"]).all()
assert test_report["answer_f1"].between(0, 1).all()
validation_answerable = validation_report[~validation_questions["is_impossible"]]
assert validation_answerable["prediction"].ne("").mean() >= MIN_ANSWER_COVERAGE
assert (validation_report["outcome"] == "success").any()
assert (test_report["outcome"] == "success").any()
assert (test_report["outcome"] == "correct_abstention").any()
assert set(test_report["outcome"]) & {
    "retrieval_miss", "bad_context", "unsupported_claim", "answering_failure",
}
assert summary is not None and {
    "retrieval_recall_at_1", "retrieval_recall_at_3", "retrieval_mrr",
    "answer_f1", "answer_coverage", "abstention_accuracy", "outcome_counts",
} <= set(summary)
assert 0 <= summary["retrieval_mrr"] <= 1
assert summary["answer_coverage"] > 0
assert failure_examples is not None
failure_buckets = {"retrieval_miss", "bad_context", "unsupported_claim", "answering_failure"}
assert set(failure_examples["outcome"]) <= failure_buckets
assert failure_examples.groupby("outcome").size().le(5).all()
for outcome in set(test_report["outcome"]) & failure_buckets:
    assert (failure_examples["outcome"] == outcome).any()
print(summary)
""",
                    role="self_check",
                ),
            ],
        ),
        "10-llm-serving-benchmark.ipynb": notebook(
            "LLM serving: анализ реальных benchmark-запусков",
            [
                md(
                    """## Статус: draft

Эта лабораторная анализирует только сохранённые результаты реальных запусков vLLM и SGLang на одинаковом оборудовании и workload. В репозитории пока нет подтверждённых fixtures, поэтому публиковать сравнительные числа или делать вывод о победителе нельзя.

Синтетические метрики запрещены. Черновик станет доступным учащимся только после записи настоящих запусков, проверки provenance и отдельного review.

**Будущий deliverable:** проверка сопоставимости запусков, throughput, TTFT и TPOT с p50/p95, график latency-throughput и инженерный вывод с перечислением ограничений."""
                ),
                code(
                    """import hashlib
import json
from pathlib import Path

import pandas as pd


def find_repository_root() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / "data" / "llm-serving-benchmarks" / "manifest.json").exists():
            return candidate
    raise FileNotFoundError("Запустите notebook из checkout репозитория ml-mentor-labs")


ROOT = find_repository_root()
BENCHMARK_DIR = ROOT / "data" / "llm-serving-benchmarks"
benchmark_manifest = json.loads((BENCHMARK_DIR / "manifest.json").read_text(encoding="utf-8"))
recorded_runs = benchmark_manifest["runs"]
BENCHMARK_READY = bool(recorded_runs)
assert all(run.get("provenance") == "recorded" for run in recorded_runs)
""",
                    role="setup",
                ),
                md(
                    """## 1. Проверьте provenance до анализа

Каждый run обязан указывать engine/version/commit, модель и revision, GPU, driver/CUDA, контейнер или lockfile, параметры workload, временную метку, сырой fixture и SHA-256. Запуски можно сравнивать только при совпадении полей из `comparabilityRequirements` протокола."""
                ),
                code(
                    """def load_verified_runs(manifest: dict) -> pd.DataFrame:
    # TODO: если runs пуст, завершите работу без чисел; иначе проверьте SHA-256
    # каждого raw fixture и верните нормализованную таблицу реальных измерений.
    raise NotImplementedError


if not BENCHMARK_READY:
    raise RuntimeError("Нет проверенных реальных fixtures: сравнительные числа публиковать нельзя")

runs = load_verified_runs(benchmark_manifest)
"""
                ),
                md(
                    """## 2. Сравните только сопоставимые запуски

Покажите количество успешных запросов, input/output tokens, request throughput, output-token throughput, TTFT p50/p95 и TPOT p50/p95. Не объединяйте client-observed latency и внутренние engine-метрики под одним названием."""
                ),
                code(
                    """# TODO: создайте comparison только из сопоставимых recorded runs.
comparison = None
LIMITATIONS = None
"""
                ),
                code(
                    """# Self-check. В текущем draft он намеренно не проходит без реальных fixtures.
assert BENCHMARK_READY, "release gate: нужны реальные записанные запуски"
assert comparison is not None and len(comparison) >= 2
required = {"engine", "request_throughput", "output_token_throughput", "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "tpot_p95_ms"}
assert required <= set(comparison.columns)
assert (comparison[list(required - {"engine"})] >= 0).all().all()
assert isinstance(LIMITATIONS, str) and LIMITATIONS.strip()
""",
                    role="self_check",
                ),
            ],
        ),
        "advanced/10a-vllm-benchmark.ipynb": notebook(
            "Draft collector: vLLM serving benchmark",
            [
                md(
                    """## Статус: draft

Шаблон фиксирует реальный запуск официального vLLM benchmark CLI. Он не устанавливает сервер, не запускается в основном CI и не содержит результатов. Перед запуском закрепите версию vLLM, revision модели и окружение, затем используйте workload из `data/llm-serving-benchmarks/protocol.json`.

Сырые числа нельзя переносить вручную: сохраните JSON, логи команды и metadata рядом, затем добавьте их в fixture manifest с SHA-256."""
                ),
                code(
                    """import hashlib
import json
from pathlib import Path


def find_repository_root() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / "data" / "llm-serving-benchmarks" / "protocol.json").exists():
            return candidate
    raise FileNotFoundError("Запустите notebook из checkout репозитория ml-mentor-labs")


ROOT = find_repository_root()
BENCHMARK_DIR = ROOT / "data" / "llm-serving-benchmarks"
protocol = json.loads((BENCHMARK_DIR / "protocol.json").read_text(encoding="utf-8"))
ENGINE = "vllm"
""",
                    role="setup",
                ),
                md(
                    """## Запуск

Используйте официальный CLI той версии vLLM, которую записываете в metadata. Параметры запроса должны точно соответствовать protocol. Команду и вывод `pip freeze` или digest контейнера сохраните в run-каталоге. Если версия CLI не поддерживает нужный JSON output, collector остаётся draft — копировать числа с экрана нельзя."""
                ),
                code(
                    """# TODO: укажите существующие файлы после реального запуска.
RAW_RESULT_PATH = None
COMMAND_LOG_PATH = None
RUN_METADATA = {
    "provenance": "recorded",
    "engine": ENGINE,
    "engineVersion": None,
    "engineCommit": None,
    "modelId": None,
    "modelRevision": None,
    "gpu": None,
    "driver": None,
    "cuda": None,
    "environmentLock": None,
    "startedAt": None,
    "rawFixturePath": None,
    "rawFixtureSha256": None,
}
"""
                ),
                code(
                    """# Self-check до передачи fixture на review.
assert RAW_RESULT_PATH is not None and Path(RAW_RESULT_PATH).is_file()
assert COMMAND_LOG_PATH is not None and Path(COMMAND_LOG_PATH).is_file()
raw_sha256 = hashlib.sha256(Path(RAW_RESULT_PATH).read_bytes()).hexdigest()
RUN_METADATA["rawFixturePath"] = str(RAW_RESULT_PATH)
RUN_METADATA["rawFixtureSha256"] = raw_sha256
assert set(RUN_METADATA) == set(protocol["requiredRunMetadata"])
assert all(RUN_METADATA.values())
assert len(raw_sha256) == 64
print({"engine": ENGINE, "raw_sha256": raw_sha256})
""",
                    role="self_check",
                ),
            ],
        ),
        "advanced/10b-sglang-benchmark.ipynb": notebook(
            "Draft collector: SGLang serving benchmark",
            [
                md(
                    """## Статус: draft

Шаблон фиксирует реальный запуск официального SGLang benchmark CLI. Он не устанавливает сервер, не запускается в основном CI и не содержит результатов. Перед запуском закрепите версию SGLang, revision модели и окружение, затем используйте workload из `data/llm-serving-benchmarks/protocol.json`.

Сырые числа нельзя переносить вручную: сохраните JSON, логи команды и metadata рядом, затем добавьте их в fixture manifest с SHA-256."""
                ),
                code(
                    """import hashlib
import json
from pathlib import Path


def find_repository_root() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / "data" / "llm-serving-benchmarks" / "protocol.json").exists():
            return candidate
    raise FileNotFoundError("Запустите notebook из checkout репозитория ml-mentor-labs")


ROOT = find_repository_root()
BENCHMARK_DIR = ROOT / "data" / "llm-serving-benchmarks"
protocol = json.loads((BENCHMARK_DIR / "protocol.json").read_text(encoding="utf-8"))
ENGINE = "sglang"
""",
                    role="setup",
                ),
                md(
                    """## Запуск

Используйте официальный CLI той версии SGLang, которую записываете в metadata. Параметры запроса должны точно соответствовать protocol. Команду и вывод `pip freeze` или digest контейнера сохраните в run-каталоге. Если версия CLI не поддерживает нужный JSON output, collector остаётся draft — копировать числа с экрана нельзя."""
                ),
                code(
                    """# TODO: укажите существующие файлы после реального запуска.
RAW_RESULT_PATH = None
COMMAND_LOG_PATH = None
RUN_METADATA = {
    "provenance": "recorded",
    "engine": ENGINE,
    "engineVersion": None,
    "engineCommit": None,
    "modelId": None,
    "modelRevision": None,
    "gpu": None,
    "driver": None,
    "cuda": None,
    "environmentLock": None,
    "startedAt": None,
    "rawFixturePath": None,
    "rawFixtureSha256": None,
}
"""
                ),
                code(
                    """# Self-check до передачи fixture на review.
assert RAW_RESULT_PATH is not None and Path(RAW_RESULT_PATH).is_file()
assert COMMAND_LOG_PATH is not None and Path(COMMAND_LOG_PATH).is_file()
raw_sha256 = hashlib.sha256(Path(RAW_RESULT_PATH).read_bytes()).hexdigest()
RUN_METADATA["rawFixturePath"] = str(RAW_RESULT_PATH)
RUN_METADATA["rawFixtureSha256"] = raw_sha256
assert set(RUN_METADATA) == set(protocol["requiredRunMetadata"])
assert all(RUN_METADATA.values())
assert len(raw_sha256) == 64
print({"engine": ENGINE, "raw_sha256": raw_sha256})
""",
                    role="self_check",
                ),
            ],
        ),
    }
