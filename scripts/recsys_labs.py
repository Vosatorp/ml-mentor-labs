from __future__ import annotations

from collections.abc import Callable


CellFactory = Callable[..., dict]

FIXTURE_COMMIT = "a8542e1f42cecc21a1a7278955b6fd323ced8c9b"
FIXTURE_SHA256 = "ca4dbdffea7114131ad9da411419838ed9c0a830613de935deaeecaad14d4805"
FIXTURE_BYTES = 2_031_420
FIXTURE_URL = (
    "https://raw.githubusercontent.com/Vosatorp/ml-mentor-labs/"
    f"{FIXTURE_COMMIT}/data/yambda/yambda-50m-likes-compact.parquet"
)


def _setup_source() -> str:
    return f'''import hashlib
import subprocess
import sys
import time
import urllib.request
from importlib.metadata import PackageNotFoundError, version as installed_version
from pathlib import Path
from urllib.parse import urlparse


REQUIRED_PACKAGES = {{
    "pyarrow": "22.0.0",
    "implicit": "0.7.3",
    "catboost": "1.2.10",
}}


def install_missing_packages() -> None:
    missing = []
    for package, version in REQUIRED_PACKAGES.items():
        try:
            actual = installed_version(package)
        except PackageNotFoundError:
            missing.append(f"{{package}}=={{version}}")
            continue
        if actual != version:
            missing.append(f"{{package}}=={{version}}")
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *missing], check=True)


install_missing_packages()

import numpy as np
import pandas as pd
import pyarrow
import implicit
import catboost
from scipy import sparse


SEED = 42
FIXTURE_URL = "{FIXTURE_URL}"
FIXTURE_SHA256 = "{FIXTURE_SHA256}"
FIXTURE_BYTES = {FIXTURE_BYTES}
EXPECTED_COLUMNS = ["uid", "item_id", "timestamp", "is_organic"]


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_sha256(payload: bytes, expected: str, label: str) -> None:
    actual = sha256(payload)
    if actual != expected:
        raise RuntimeError(f"{{label}}: checksum changed: {{actual}}")


def require_final_hostname(response, expected: str) -> None:
    actual = urlparse(response.geturl()).hostname
    if actual != expected:
        raise RuntimeError(f"unexpected redirect host: {{actual}}")


def resolve_fixture_path() -> Path:
    for root in [Path.cwd(), *Path.cwd().parents]:
        candidate = root / "data" / "yambda" / "yambda-50m-likes-compact.parquet"
        if candidate.is_file():
            payload = candidate.read_bytes()
            if len(payload) != FIXTURE_BYTES:
                raise RuntimeError(f"fixture size changed: {{len(payload)}}")
            require_sha256(payload, FIXTURE_SHA256, "local Yambda fixture")
            return candidate

    cache_path = Path("/tmp/ml-mentor-labs-datasets-v1/yambda/yambda-50m-likes-compact.parquet")
    if cache_path.is_file():
        payload = cache_path.read_bytes()
        if len(payload) != FIXTURE_BYTES:
            raise RuntimeError(f"cached fixture size changed: {{len(payload)}}")
        require_sha256(payload, FIXTURE_SHA256, "cached Yambda fixture")
        return cache_path

    request = urllib.request.Request(FIXTURE_URL, headers={{"User-Agent": "ML-Mentor-Labs/1.0"}})
    with urllib.request.urlopen(request, timeout=120) as response:
        require_final_hostname(response, "raw.githubusercontent.com")
        payload = response.read()
    if len(payload) != FIXTURE_BYTES:
        raise RuntimeError(f"downloaded fixture size changed: {{len(payload)}}")
    require_sha256(payload, FIXTURE_SHA256, "downloaded Yambda fixture")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(payload)
    return cache_path


FIXTURE_PATH = resolve_fixture_path()
events = pd.read_parquet(FIXTURE_PATH, columns=EXPECTED_COLUMNS)
events = events.sort_values(["timestamp", "uid", "item_id"], kind="mergesort").reset_index(drop=True)
if list(events.columns) != EXPECTED_COLUMNS:
    raise RuntimeError(f"unexpected columns: {{list(events.columns)}}")
if len(events) != 268_631 or events.uid.nunique() != 1_999 or events.item_id.nunique() != 42_965:
    raise RuntimeError("unexpected fixture statistics")
if not events.timestamp.is_monotonic_increasing:
    raise RuntimeError("events must be sorted by time")

TRAIN_BOUNDARY = int(events.timestamp.quantile(0.70, interpolation="lower"))
VALIDATION_BOUNDARY = int(events.timestamp.quantile(0.85, interpolation="lower"))
train_events = events[events.timestamp <= TRAIN_BOUNDARY].copy()
validation_events = events[(events.timestamp > TRAIN_BOUNDARY) & (events.timestamp <= VALIDATION_BOUNDARY)].copy()
test_events = events[events.timestamp > VALIDATION_BOUNDARY].copy()

USER_IDS = np.sort(train_events.uid.unique())
ITEM_IDS = np.sort(train_events.item_id.unique())
USER_TO_INDEX = {{int(value): index for index, value in enumerate(USER_IDS)}}
ITEM_TO_INDEX = {{int(value): index for index, value in enumerate(ITEM_IDS)}}
INDEX_TO_ITEM = np.asarray(ITEM_IDS)


def to_user_item_matrix(frame: pd.DataFrame) -> sparse.csr_matrix:
    known = frame[frame.uid.isin(USER_TO_INDEX) & frame.item_id.isin(ITEM_TO_INDEX)]
    rows = known.uid.map(USER_TO_INDEX).to_numpy()
    columns = known.item_id.map(ITEM_TO_INDEX).to_numpy()
    values = np.ones(len(known), dtype=np.float32)
    matrix = sparse.csr_matrix((values, (rows, columns)), shape=(len(USER_IDS), len(ITEM_IDS)))
    matrix.data[:] = 1.0
    matrix.eliminate_zeros()
    return matrix


train_matrix = to_user_item_matrix(train_events)
print({{
    "versions": {{"pyarrow": pyarrow.__version__, "implicit": implicit.__version__, "catboost": catboost.__version__}},
    "events": len(events),
    "train": len(train_events),
    "validation": len(validation_events),
    "test": len(test_events),
    "users_in_train": train_matrix.shape[0],
    "items_in_train": train_matrix.shape[1],
}})
'''


def build_recsys_labs(*, md: CellFactory, code: CellFactory, notebook: CellFactory) -> dict[str, dict]:
    setup = _setup_source()
    return {
        "12-recsys-collaborative-baselines.ipynb": notebook(
            "Коллаборативные модели: от лога до честного бейзлайна",
            [
                md(
                    """## Что предстоит сделать

За 90 минут вы пройдёте путь от лога лайков до решения, которое можно защищать перед командой: построите популярность, item-to-item и implicit ALS, сравните их на одном временном разбиении и отдельно проверите сложные срезы. Обязательный путь рассчитан на CPU и должен выполняться не более 15 минут.

**Итог:** одна таблица с Recall@20, NDCG@20, покрытием каталога, временем обучения и применения; отдельные метрики для пользователей с короткой историей и редких объектов; решение, какой бейзлайн выпускать первым.

Случайное разбиение запрещено. Уже просмотренные объекты нужно удалить из рекомендаций. Фраза «ALS лучше» без разбора срезов и стоимости не считается инженерным выводом.

### На чём основана работа

Последовательность экспериментов опирается на практику [недели 1](https://github.com/yandexdataschool/recsys_course/blob/2026_spring/week01_intro/practice.ipynb) и [недели 3](https://github.com/yandexdataschool/recsys_course/blob/2026_spring/week03_ranking/practice/practice.ipynb) курса ШАД RecSys 2026. Текст, код и проверки написаны заново для ML Mentor. Данные — детерминированная выборка Yambda-50M, Apache 2.0; ревизия, преобразование и контрольные суммы описаны в `data/datasets.json`."""
                ),
                code(setup, role="setup"),
                md(
                    """## 1. Контракт событий и временное разбиение

Одна строка — факт лайка: пользователь `uid`, объект `item_id`, время `timestamp`, признак органического взаимодействия `is_organic`. Мы прогнозируем будущие лайки по прошлым. Поэтому граница времени общая для всех пользователей: модель не должна видеть события, которые в продукте ещё не произошли.

Проверьте границы, пересечения событий и доступность пользователей. Для основной оценки оставьте пользователей, у которых есть хотя бы пять лайков в train и хотя бы один лайк в validation и test. Пользователей без истории не удаляйте из отчёта: для них нужен fallback по популярности."""
                ),
                code(
                    """def build_evaluation_users(train, validation, test):
    # TODO: верните DataFrame columns=[uid, train_events, validation_events, test_events, segment].
    # segment: cold (нет train), short (5–19 train), established (20+ train).
    raise NotImplementedError


evaluation_users = build_evaluation_users(train_events, validation_events, test_events)
eligible_users = evaluation_users.query("train_events >= 5 and validation_events >= 1 and test_events >= 1").uid.tolist()
split_audit = {
    "train_max": int(train_events.timestamp.max()),
    "validation_min": int(validation_events.timestamp.min()),
    "validation_max": int(validation_events.timestamp.max()),
    "test_min": int(test_events.timestamp.min()),
    "eligible_users": len(eligible_users),
}
split_audit
"""
                ),
                md(
                    """## 2. Популярность

Это не заглушка, а обязательная контрольная модель. Посчитайте число уникальных пользователей для каждого объекта только по train, отсортируйте объекты детерминированно и для каждого пользователя верните первые 20 ещё не просмотренных объектов. Измерьте время подготовки и среднее время ответа на пользователя."""
                ),
                code(
                    """def fit_popularity(history: pd.DataFrame) -> pd.DataFrame:
    # TODO: верните columns=[item_id, score, rank]; score — число уникальных пользователей.
    raise NotImplementedError


def recommend_popularity(popularity: pd.DataFrame, history: pd.DataFrame, user_ids, k=20):
    # TODO: верните dict[uid, list[item_id]], исключив объекты из history этого пользователя.
    raise NotImplementedError


popularity_started = time.perf_counter()
popularity_model = fit_popularity(train_events)
popularity_fit_seconds = time.perf_counter() - popularity_started
popularity_recommendations = recommend_popularity(popularity_model, train_events, eligible_users, k=20)
"""
                ),
                md(
                    """## 3. Item-to-item по сходству взаимодействий

Представьте объект бинарным вектором пользователей. Близкими считаются объекты, которые лайкали похожие люди. Соберите `CosineRecommender(K=100)` из `implicit.nearest_neighbours`, обучите его на той же матрице и получите top-20. Объясните, зачем ограничивать число соседей и что произойдёт с редким объектом."""
                ),
                code(
                    """from implicit.nearest_neighbours import CosineRecommender


def fit_item_to_item(user_item_matrix: sparse.csr_matrix):
    # TODO: обучите CosineRecommender(K=100) и верните модель вместе со временем fit.
    raise NotImplementedError


def recommend_implicit_model(model, user_item_matrix, user_ids, k=20):
    # TODO: вызовите model.recommend для известных пользователей, filter_already_liked_items=True.
    # Преобразуйте внутренние индексы объектов обратно в исходные item_id.
    raise NotImplementedError


item_to_item_model, item_to_item_fit_seconds = fit_item_to_item(train_matrix)
item_to_item_recommendations = recommend_implicit_model(
    item_to_item_model, train_matrix, eligible_users, k=20
)
"""
                ),
                md(
                    """## 4. Implicit ALS

ALS раскладывает разреженную матрицу взаимодействий на векторы пользователей и объектов. Здесь отсутствие лайка не является доказанным дизлайком: это неизвестное взаимодействие. Обучите `AlternatingLeastSquares` с `factors=32`, `regularization=0.05`, `iterations=10`, `random_state=42` и CPU. Не подбирайте параметры по test."""
                ),
                code(
                    """from implicit.als import AlternatingLeastSquares


def fit_als(user_item_matrix: sparse.csr_matrix):
    # TODO: обучите закреплённую ALS и верните модель вместе со временем fit.
    raise NotImplementedError


als_model, als_fit_seconds = fit_als(train_matrix)
als_recommendations = recommend_implicit_model(als_model, train_matrix, eligible_users, k=20)
"""
                ),
                md(
                    """## 5. Общая оценка и стоимость

Для каждого пользователя релевантны его лайки из test. Считайте Recall@20 и NDCG@20 сначала по пользователям, затем усредняйте. Покрытие — доля уникальных рекомендованных объектов от каталога train. Измерьте среднее время формирования ответа. Все три модели должны оцениваться на одном наборе пользователей."""
                ),
                code(
                    """def ranking_metrics(recommendations, targets: pd.DataFrame, catalog_size: int, k=20):
    # TODO: верните общий dict и per_user DataFrame.
    # Общий dict содержит recall_at_20, ndcg_at_20, catalog_coverage и evaluated_users.
    raise NotImplementedError


def timed_recommend(callable_):
    # TODO: верните результат и среднее время на пользователя в миллисекундах.
    raise NotImplementedError


model_runs = {
    "popularity": (popularity_recommendations, popularity_fit_seconds),
    "item_to_item": (item_to_item_recommendations, item_to_item_fit_seconds),
    "als": (als_recommendations, als_fit_seconds),
}
comparison_rows = []
per_user_reports = {}
for model_name, (recommendations, fit_seconds) in model_runs.items():
    # TODO: посчитайте метрики и добавьте одну строку comparison_rows.
    pass

comparison = pd.DataFrame(comparison_rows)
comparison
"""
                ),
                md(
                    """## 6. Новые пользователи, редкие объекты и решение

Покажите минимум два среза: пользователи с 5–19 событиями в train и цели, относящиеся к нижним 50% объектов по популярности train. Для пользователя без истории примените popularity fallback и явно укажите, что персонализации в нём нет.

Выберите первый production-бейзлайн. Назовите выигрыш, проигрыш, вычислительную цену и условие пересмотра решения. Если ALS выигрывает общий Recall, но хуже на короткой истории или заметно дороже, это должно быть видно в выводе."""
                ),
                code(
                    """def build_slice_report(per_user_reports, train, test, evaluation_users):
    # TODO: верните таблицу model, slice, users, recall_at_20, ndcg_at_20.
    # Обязательные slice: all, short_history, rare_target.
    raise NotImplementedError


slice_report = build_slice_report(per_user_reports, train_events, test_events, evaluation_users)
cold_user_example = int(evaluation_users.sort_values(["train_events", "uid"]).iloc[0].uid)
cold_user_recommendations = recommend_popularity(popularity_model, train_events, [cold_user_example], k=20)

ENGINEERING_DECISION = {
    "ship_first": None,  # TODO: popularity | item_to_item | als
    "quality_evidence": None,
    "slice_risk": None,
    "compute_cost": None,
    "fallback": "popularity",
    "revisit_when": None,
}
"""
                ),
                code(
                    """# Self-check: проверяет полноту работы, но не выбирает модель за вас.
assert split_audit["train_max"] < split_audit["validation_min"]
assert split_audit["validation_max"] < split_audit["test_min"]
assert len(eligible_users) >= 1_000
assert set(comparison.model) == {"popularity", "item_to_item", "als"}
assert comparison.recall_at_20.between(0, 1).all()
assert comparison.ndcg_at_20.between(0, 1).all()
assert comparison.catalog_coverage.between(0, 1).all()
assert (comparison.fit_seconds >= 0).all() and (comparison.mean_inference_ms >= 0).all()
for uid in eligible_users:
    seen = set(train_events.loc[train_events.uid == uid, "item_id"])
    for recommendations, _ in model_runs.values():
        assert not (seen & set(recommendations[uid][:20]))
assert {"all", "short_history", "rare_target"} <= set(slice_report.slice)
assert all(ENGINEERING_DECISION[key] is not None for key in [
    "ship_first", "quality_evidence", "slice_risk", "compute_cost", "revisit_when"
])
print(comparison.sort_values("ndcg_at_20", ascending=False))
display(slice_report)
""",
                    role="self_check",
                ),
                md(
                    """## Что сдать

1. Заполненный notebook без удаления assertions.
2. Таблицу сравнения и таблицу срезов.
3. Короткое решение о первом выпуске: модель, fallback, стоимость и риск.
4. Пояснение, почему случайный split и вывод «ALS лучше» были бы недостаточны.

Не публикуйте выполненный notebook в репозитории ML Mentor: сохраните личную копию в Google Drive или выгрузите её локально."""
                ),
            ],
        ),
        "13-recsys-multistage-ranking.ipynb": notebook(
            "Каскад рекомендаций: кандидаты, признаки и CatBoost-ранжирование",
            [
                md(
                    """## Что предстоит сделать

За 120 минут вы соберёте полный рекомендательный каскад: три источника кандидатов, point-in-time признаки, эвристическое объединение и `CatBoostRanker`. Вы отдельно измерите потери на этапе кандидатов и качество итогового порядка. Обязательный CPU-прогон рассчитан не более чем на 15 минут.

**Итог:** схема каскада, таблица качества каждого слоя, разбор потерянных целей и решение о размере пула. Непоказанный объект нельзя называть доказанным негативом: это лишь неизвестное взаимодействие в implicit feedback.

### На чём основана работа

Структура эксперимента опирается на практику [недели 2](https://github.com/yandexdataschool/recsys_course/blob/2026_spring/week02_candgen/practice.ipynb) и [недели 3](https://github.com/yandexdataschool/recsys_course/blob/2026_spring/week03_ranking/practice/practice.ipynb) курса ШАД RecSys 2026. Формулировки, код и проверки ML Mentor написаны заново. Используется тот же закреплённый Apache-2.0 fixture Yambda, что и в предыдущей работе."""
                ),
                code(setup, role="setup"),
                md(
                    """## 1. Два временных снимка

Для обучения ранжировщика история заканчивается на первой границе, а целевые лайки берутся из validation. Для финальной проверки кандидаты и признаки строятся по истории `train + validation`, цели — из test. Нельзя один раз посчитать признаки по полному логу и потом разрезать строки: так в train попадёт будущее.

Сформируйте таблицу запросов `uid, query_timestamp` для обоих снимков. Ограничьте обязательный прогон 1 000 подходящих пользователей, выбранных детерминированно по SHA-256 от `uid`, чтобы время CPU оставалось предсказуемым."""
                ),
                code(
                    """def deterministic_users(user_ids, limit=1_000):
    # TODO: отсортируйте uid по SHA-256 строки recsys-cascade-v1:<uid> и возьмите limit.
    raise NotImplementedError


def build_snapshot(history: pd.DataFrame, targets: pd.DataFrame, user_ids):
    # TODO: верните history, targets и queries. query_timestamp строго больше max(history.timestamp)
    # и не больше min(targets.timestamp) для каждого пользователя.
    raise NotImplementedError


ranker_users = deterministic_users(
    sorted(set(train_events.uid) & set(validation_events.uid) & set(test_events.uid)),
    limit=1_000,
)
training_snapshot = build_snapshot(train_events, validation_events, ranker_users)
test_history = pd.concat([train_events, validation_events], ignore_index=True)
evaluation_snapshot = build_snapshot(test_history, test_events, ranker_users)
"""
                ),
                md(
                    """## 2. Три источника кандидатов

Для каждого снимка обучите популярность, item-to-item и ALS только на его истории. Каждый источник возвращает top-100 после удаления просмотренных объектов. Сохраните исходный `score` и `rank`: после объединения они станут признаками и позволят понять, какой генератор нашёл цель.

Здесь можно переиспользовать функции из предыдущей лабораторной, но notebook должен оставаться исполняемым сам по себе."""
                ),
                code(
                    """from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import CosineRecommender


SOURCES = ("popularity", "item_to_item", "als")
SOURCE_K = 100


def fit_candidate_models(history: pd.DataFrame):
    # TODO: обучите три источника на history и верните модели, матрицу и отображения ID.
    raise NotImplementedError


def generate_source_candidates(bundle, queries: pd.DataFrame, k=SOURCE_K):
    # TODO: верните long DataFrame columns=[uid, item_id, source, source_score, source_rank].
    # Для каждого uid и каждого источника не более k строк; просмотренные объекты исключены.
    raise NotImplementedError


training_bundle = fit_candidate_models(training_snapshot["history"])
training_source_candidates = generate_source_candidates(training_bundle, training_snapshot["queries"])
evaluation_bundle = fit_candidate_models(evaluation_snapshot["history"])
evaluation_source_candidates = generate_source_candidates(evaluation_bundle, evaluation_snapshot["queries"])
"""
                ),
                md(
                    """## 3. Объединение и потери кандидатов

Дедуплицируйте пары `(uid, item_id)`, но не теряйте происхождение: для каждого источника нужны отдельные score, rank и бинарный признак присутствия. Рассчитайте candidate Recall@100 для каждого источника и их объединения. Это верхняя граница для ранжировщика: он не вернёт цель, которой нет в пуле."""
                ),
                code(
                    """def merge_candidates(source_candidates: pd.DataFrame) -> pd.DataFrame:
    # TODO: одна строка на uid,item_id; columns для каждого источника:
    # <source>_score, <source>_rank, from_<source>. Отсутствующий rank заполните SOURCE_K + 1.
    raise NotImplementedError


def candidate_recall(source_candidates, targets, k=100):
    # TODO: посчитайте macro Recall@k по пользователям отдельно для источников и union.
    raise NotImplementedError


training_pool = merge_candidates(training_source_candidates)
evaluation_pool = merge_candidates(evaluation_source_candidates)
training_candidate_report = candidate_recall(training_source_candidates, training_snapshot["targets"], k=100)
evaluation_candidate_report = candidate_recall(evaluation_source_candidates, evaluation_snapshot["targets"], k=100)
"""
                ),
                md(
                    """## 4. Point-in-time признаки

Для каждой пары пользователь–кандидат посчитайте признаки только по истории снимка: длину истории пользователя, популярность объекта, давность последнего лайка пользователя, встречался ли объект среди последних 10 лайков, score/rank источников и число источников. `is_organic` можно агрегировать только по прошлым событиям.

Метка равна 1, если объект встречается среди целей этого запроса, иначе 0. Ноль здесь — не доказанный дизлайк, а рабочее приближение для обучения порядка внутри уже сформированного пула."""
                ),
                code(
                    """FEATURE_COLUMNS = [
    "user_history_size", "item_popularity", "user_recency_seconds",
    "in_last_10", "organic_share", "source_count",
    "popularity_score", "popularity_rank",
    "item_to_item_score", "item_to_item_rank",
    "als_score", "als_rank",
]


def point_in_time_features(pool, history, targets, queries):
    # TODO: верните pool + FEATURE_COLUMNS + label + group_id.
    # Для каждой строки докажите history.timestamp < query_timestamp.
    raise NotImplementedError


training_rows = point_in_time_features(
    training_pool, training_snapshot["history"], training_snapshot["targets"], training_snapshot["queries"]
)
evaluation_rows = point_in_time_features(
    evaluation_pool, evaluation_snapshot["history"], evaluation_snapshot["targets"], evaluation_snapshot["queries"]
)
"""
                ),
                md(
                    """## 5. Эвристика и CatBoostRanker

Сначала постройте прозрачную эвристику reciprocal rank fusion: сумма `1 / (60 + rank)` по источникам. Затем обучите `CatBoostRanker` с `loss_function="YetiRankPairwise"`, `iterations=150`, `depth=6`, `learning_rate=0.08`, `random_seed=42`, `verbose=False`, `thread_count=-1`. Перед fit отсортируйте строки по `group_id`, чтобы строки одного пользовательского запроса шли подряд.

Не подбирайте параметры по test. Если в обучающем запросе нет ни одного положительного кандидата, исключите всю группу и посчитайте долю таких потерь отдельно."""
                ),
                code(
                    """from catboost import CatBoostRanker


def reciprocal_rank_fusion(frame: pd.DataFrame) -> np.ndarray:
    # TODO: верните score по трём source rank; отсутствующий источник не даёт вклад.
    raise NotImplementedError


def fit_ranker(training_frame: pd.DataFrame):
    # TODO: удалите группы без positive, отсортируйте по group_id и обучите закреплённый ranker.
    # Верните model, fit_seconds и долю исключённых запросов.
    raise NotImplementedError


training_rows["heuristic_score"] = reciprocal_rank_fusion(training_rows)
evaluation_rows["heuristic_score"] = reciprocal_rank_fusion(evaluation_rows)
ranker, ranker_fit_seconds, dropped_training_group_share = fit_ranker(training_rows)
evaluation_rows["ranker_score"] = ranker.predict(evaluation_rows[FEATURE_COLUMNS])
"""
                ),
                md(
                    """## 6. Качество слоёв, задержка и размер пула

Сравните эвристику и ranker по Recall@10, NDCG@10, покрытию каталога и времени сортировки. Рядом покажите candidate Recall@100: если он низкий, перестановка кандидатов проблему не исправит. Повторите оценку union для размеров источника 20, 50 и 100 и выберите минимальный пул, который укладывается в допустимую потерю качества и задержку.

В письменном выводе разделите: потери генерации, потери ранжирования, ограничение implicit feedback и production-стоимость. Укажите fallback при недоступности одного источника или ranker."""
                ),
                code(
                    """def ranking_report(frame, targets, score_column, catalog_size, k=10):
    # TODO: macro Recall@k, NDCG@k, catalog coverage и mean sorting milliseconds per user.
    raise NotImplementedError


heuristic_report = ranking_report(
    evaluation_rows, evaluation_snapshot["targets"], "heuristic_score", evaluation_bundle["matrix"].shape[1], k=10
)
ranker_report = ranking_report(
    evaluation_rows, evaluation_snapshot["targets"], "ranker_score", evaluation_bundle["matrix"].shape[1], k=10
)
layer_report = pd.DataFrame([
    *evaluation_candidate_report.to_dict("records"),
    {"layer": "heuristic_top10", **heuristic_report},
    {"layer": "catboost_ranker_top10", **ranker_report},
])


def pool_size_experiment(source_candidates, targets, sizes=(20, 50, 100)):
    # TODO: для каждого размера источника верните union candidate recall, mean pool size и build milliseconds/user.
    raise NotImplementedError


pool_size_report = pool_size_experiment(evaluation_source_candidates, evaluation_snapshot["targets"])
CASCADE_DECISION = {
    "source_pool_size": None,  # TODO: 20 | 50 | 100
    "candidate_loss": None,
    "ranking_loss": None,
    "implicit_feedback_limit": None,
    "latency_tradeoff": None,
    "fallback": None,
}
"""
                ),
                code(
                    """# Self-check: проверяет границы эксперимента, но не подставляет вывод.
assert len(ranker_users) == 1_000 and len(set(ranker_users)) == 1_000
assert training_snapshot["history"].timestamp.max() < training_snapshot["targets"].timestamp.min()
assert evaluation_snapshot["history"].timestamp.max() < evaluation_snapshot["targets"].timestamp.min()
required_source_columns = {"uid", "item_id", "source", "source_score", "source_rank"}
assert required_source_columns <= set(training_source_candidates)
assert set(training_source_candidates.source) == set(SOURCES)
assert not training_pool.duplicated(["uid", "item_id"]).any()
assert not evaluation_pool.duplicated(["uid", "item_id"]).any()
assert set(FEATURE_COLUMNS) <= set(training_rows) and set(FEATURE_COLUMNS) <= set(evaluation_rows)
assert np.isfinite(training_rows[FEATURE_COLUMNS].to_numpy()).all()
assert training_rows.groupby("group_id").label.max().isin([0, 1]).all()
assert 0 <= dropped_training_group_share < 1
assert {"popularity", "item_to_item", "als", "union"} <= set(evaluation_candidate_report.layer)
assert {20, 50, 100} == set(pool_size_report.source_k)
assert all(CASCADE_DECISION[key] is not None for key in CASCADE_DECISION)
assert "неизвест" in CASCADE_DECISION["implicit_feedback_limit"].lower()
display(layer_report)
display(pool_size_report)
""",
                    role="self_check",
                ),
                md(
                    """## Что сдать

1. Заполненный notebook и схему `история → три источника → union → признаки → ranker → top-10`.
2. Candidate Recall@100 по каждому источнику и union.
3. Recall@10, NDCG@10, покрытие и задержку эвристики и ranker.
4. Эксперимент с пулами 20/50/100 и решение о размере пула.
5. Объяснение, почему непоказанные объекты — неизвестные взаимодействия, а не доказанные негативы.

Не загружайте решения в публичный репозиторий ML Mentor. Личную выполненную копию храните в Google Drive или локально."""
                ),
            ],
        ),
    }
