from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
LAB_MANIFEST_PATH = ROOT / "labs-manifest.json"
DATASET_MANIFEST_PATH = ROOT / "data" / "datasets.json"
BENCHMARK_MANIFEST_PATH = ROOT / "data" / "llm-serving-benchmarks" / "manifest.json"
REQUIRED_ROLES = {"setup", "exercise", "self_check"}
ALLOWED_RELEASE_STATUSES = {"public", "review_ready", "draft"}
CPU_STARTER_LABS = {
    "07-bike-demand-production-capstone.ipynb",
    "08-banking77-tfidf-error-analysis.ipynb",
    "09-rag-failure-decomposition.ipynb",
}
FORBIDDEN_TEXT = (
    "private interview",
    "recruiter contact",
    "salary fork",
    "reference solution",
    "service_role",
    "supabase_service",
)
NETWORK_MARKERS = (
    "http://",
    "https://",
    "requests.",
    "urllib.",
    "urlopen(",
    "wget ",
    "curl ",
)
FORBIDDEN_CPU_CODE = (
    "os.environ",
    "os.getenv",
    "getpass(",
    ".cuda(",
    "device='cuda'",
    'device="cuda"',
)
EXPECTED_BOOTSTRAP_MARKERS = {
    "07-bike-demand-production-capstone.ipynb": {
        "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
        "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401",
        "e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f",
    },
    "08-banking77-tfidf-error-analysis.ipynb": {
        "57ec275d8078af65b7731c2a98be812d844a6d6b",
        "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
        "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
        "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
    },
    "09-rag-failure-decomposition.ipynb": {
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
        "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",
        "053396eeff7f1c07d473f2bd62c4d0a28f5d34d37f49d5ffd7922a58fbb5494f",
    },
    "11-cifar10-pytorch-training-pipeline.ipynb": {
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce",
        "170_498_071",
    },
}
ALLOWED_BOOTSTRAP_URLS = {
    "07-bike-demand-production-capstone.ipynb": {
        "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
    },
    "08-banking77-tfidf-error-analysis.ipynb": {
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/train.csv",
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/test.csv",
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/categories.json",
    },
    "09-rag-failure-decomposition.ipynb": {
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
    },
    "11-cifar10-pytorch-training-pipeline.ipynb": {
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    },
}
EXPECTED_RESPONSE_HOSTS = {
    "07-bike-demand-production-capstone.ipynb": {"archive.ics.uci.edu"},
    "08-banking77-tfidf-error-analysis.ipynb": {"raw.githubusercontent.com"},
    "09-rag-failure-decomposition.ipynb": {"rajpurkar.github.io"},
    "11-cifar10-pytorch-training-pipeline.ipynb": {"cave.cs.toronto.edu"},
}
TRUST_HELPER_NAMES = {"require_sha256", "require_final_hostname"}
EXPECTED_CONTRACT_MARKERS = {
    "07-bike-demand-production-capstone.ipynb": {
        'TRAIN_END = int(len(df) * 0.70)',
        'VALIDATION_END = int(len(df) * 0.85)',
        'SEASONAL_MISSING_POLICY = "no_prediction"',
        "VALIDATION_GO_NO_GO",
        'assert int((~seasonal_test_audit["available"]).sum()) == 39',
        "оптимистичное ограничение офлайн-оценки",
    },
    "09-rag-failure-decomposition.ipynb": {
        "MIN_ANSWER_COVERAGE = 0.60",
        'threshold_results[threshold_results["eligible"]]',
        'assert (validation_report["outcome"] == "success").any()',
        'assert (test_report["outcome"] == "correct_abstention").any()',
        "учебный продуктовый контракт именно этой лабораторной",
    },
    "11-cifar10-pytorch-training-pipeline.ipynb": {
        'REQUIRED_DEVICE = torch.device("cpu")',
        "train_size=12_000",
        "test_size=2_000",
        "accumulation_steps=3",
        "torch.inference_mode()",
        "validation_accuracy >= 0.40",
        "checkpoint_sha256",
        "torch.amp.GradScaler",
        "Адаптировано из YDS Practical_DL",
        "Дополнительное русское объяснение — «мыш»",
    },
    "advanced/10a-vllm-benchmark.ipynb": {
        '"engineVersion": None',
        '"rawFixturePath": None',
        'set(protocol["requiredRunMetadata"])',
    },
    "advanced/10b-sglang-benchmark.ipynb": {
        '"engineVersion": None',
        '"rawFixturePath": None',
        'set(protocol["requiredRunMetadata"])',
    },
}
URL_LITERAL = re.compile(r"https?://[^\"'\s)]+")
EXPECTED_DATASET_LICENSES = {
    "uci-bike-sharing-hour": "CC-BY-4.0",
    "banking77": "CC-BY-4.0",
    "squad2-rag-subset": "CC-BY-SA-4.0",
    "cifar-10": "not-specified",
    "yambda-50m-likes-compact": "Apache-2.0",
}
REQUIRED_DATASET_ARTIFACTS = {
    "uci-bike-sharing-hour": {
        "data/uci-bike-sharing/hour.csv",
        "data/uci-bike-sharing/source-readme.txt",
        "data/uci-bike-sharing/LICENSE",
        "data/uci-bike-sharing/ATTRIBUTION.md",
    },
    "banking77": {
        "data/banking77/train.csv",
        "data/banking77/test.csv",
        "data/banking77/categories.json",
        "data/banking77/LICENSE",
    },
    "squad2-rag-subset": {
        "data/squad2/squad2-rag-subset.json",
        "data/squad2/LICENSE",
        "data/squad2/ATTRIBUTION.md",
    },
    "yambda-50m-likes-compact": {
        "data/yambda/yambda-50m-likes-compact.parquet",
        "data/yambda/LICENSE",
        "data/yambda/ATTRIBUTION.md",
    },
}
EXPECTED_OLD_NOTEBOOK_HASHES = {
    "01-metrics-threshold.ipynb": "fb219f371267fea2e8f33866aac18a8a132f23e439e6c9381f179cc40127b9a0",
    "02-honest-binary-baseline.ipynb": "a9849cec51322e308d2cb7892cdfc7a0da1343224cc515ccbf161e0e8b009354",
    "03-leakage-splits.ipynb": "db794aa074251b4273cb448872b8499c468fa4b5e4cf3ec050b1ef450584eabc",
    "04-tree-ensembles.ipynb": "07f209f1b63a02b0c32187f5423a72ffcc11a70a5b3157a91a6942c6a242be7d",
    "05-broken-pytorch-loop.ipynb": "ccbdc1849dfe16471b07a2aa7caec99968c084c07860f72825ee6b6d93054709",
    "06-correlated-importance.ipynb": "d58418e178e432a426c0e58056d8443028d4104ea19125e021665eade867a895",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def repository_path(relative_path: str) -> Path:
    path = (ROOT / relative_path).resolve()
    if ROOT.resolve() not in path.parents:
        raise ValueError(f"path escapes repository: {relative_path}")
    return path


def validate_setup_trust_contract(
    path: Path,
    *,
    setup_source: str,
    namespace: dict[str, object],
    execute_setup: bool,
) -> list[str]:
    """Validate that Colab downloads fail closed even under ``python -O``.

    Static checks keep the notebook contract reviewable. Executable probes make
    sure the named helpers really reject a corrupt payload and a redirected
    response instead of relying on ``assert`` statements that optimization can
    remove.
    """

    errors: list[str] = []
    expected_hosts = EXPECTED_RESPONSE_HOSTS.get(path.name, set())
    if not expected_hosts:
        return errors

    for helper_name in sorted(TRUST_HELPER_NAMES):
        if not re.search(rf"^def {re.escape(helper_name)}\(", setup_source, re.MULTILINE):
            errors.append(f"verified Colab bootstrap is missing {helper_name} helper")

    if re.search(r"^\s*assert\b[^\n]*sha256", setup_source, re.MULTILINE):
        errors.append("checksum trust guard must raise explicitly, not use assert")
    if re.search(r"^\s*assert\b[^\n]*urlparse", setup_source, re.MULTILINE):
        errors.append("redirect-host trust guard must raise explicitly, not use assert")
    if len(re.findall(r"\brequire_sha256\(", setup_source)) < 2:
        errors.append("verified Colab bootstrap does not call require_sha256")

    for host in sorted(expected_hosts):
        call_pattern = rf'require_final_hostname\(\s*response\s*,\s*["\']{re.escape(host)}["\']\s*\)'
        if not re.search(call_pattern, setup_source):
            errors.append(f"setup does not fail closed on redirect host {host}")

    if not execute_setup:
        return errors

    require_sha256 = namespace.get("require_sha256")
    require_final_hostname = namespace.get("require_final_hostname")
    if not callable(require_sha256):
        errors.append("executed setup did not define callable require_sha256")
    if not callable(require_final_hostname):
        errors.append("executed setup did not define callable require_final_hostname")
    if not callable(require_sha256) or not callable(require_final_hostname):
        return errors

    trusted_payload = b"ml-mentor-validator-trusted-payload"
    trusted_sha256 = sha256_bytes(trusted_payload)
    try:
        require_sha256(trusted_payload, trusted_sha256, "validator trusted fixture")
    except Exception as exc:  # noqa: BLE001 - explain a broken notebook guard
        errors.append(f"require_sha256 rejected a valid payload: {type(exc).__name__}: {exc}")

    try:
        require_sha256(trusted_payload, "0" * 64, "validator corrupt fixture")
    except RuntimeError:
        pass
    except Exception as exc:  # noqa: BLE001 - RuntimeError is part of the public contract
        errors.append(f"require_sha256 raised {type(exc).__name__}, expected RuntimeError")
    else:
        errors.append("require_sha256 accepted a corrupt payload")

    expected_host = sorted(expected_hosts)[0]

    class FixtureResponse:
        def __init__(self, url: str) -> None:
            self.url = url

        def geturl(self) -> str:
            return self.url

    try:
        require_final_hostname(FixtureResponse(f"https://{expected_host}/dataset"), expected_host)
    except Exception as exc:  # noqa: BLE001 - explain a broken notebook guard
        errors.append(f"require_final_hostname rejected the expected host: {type(exc).__name__}: {exc}")

    try:
        require_final_hostname(FixtureResponse("https://malicious.invalid/captured-dataset"), expected_host)
    except RuntimeError:
        pass
    except Exception as exc:  # noqa: BLE001 - RuntimeError is part of the public contract
        errors.append(f"require_final_hostname raised {type(exc).__name__}, expected RuntimeError")
    else:
        errors.append("require_final_hostname accepted a malicious redirect host")

    return errors


def validate_notebook(path: Path, *, execute_setup: bool, offline_cpu: bool) -> list[str]:
    errors: list[str] = []
    data = load_json(path)
    if data.get("nbformat") != 4:
        errors.append("nbformat must be 4")

    roles: set[str] = set()
    has_exercise_marker = False
    namespace: dict[str, object] = {"__name__": "__ml_mentor_lab_setup__"}
    setup_source = ""
    notebook_source = ""

    for index, cell in enumerate(data.get("cells", [])):
        source = "".join(cell.get("source", []))
        notebook_source += source
        lowered = source.lower()
        if any(marker in lowered for marker in FORBIDDEN_TEXT):
            errors.append(f"cell {index}: possible private or solution material")
        if cell.get("cell_type") != "code":
            continue

        if cell.get("execution_count") is not None or cell.get("outputs"):
            errors.append(f"cell {index}: committed execution output")
        try:
            compiled = compile(source, f"{path.relative_to(ROOT)}:cell-{index}", "exec")
        except SyntaxError as exc:
            errors.append(f"cell {index}: syntax error: {exc.msg}")
            continue

        role = cell.get("metadata", {}).get("ml_mentor", {}).get("role")
        if role not in REQUIRED_ROLES:
            errors.append(f"cell {index}: missing or unknown ml_mentor.role")
            continue
        roles.add(role)
        if offline_cpu and any(marker in lowered for marker in FORBIDDEN_CPU_CODE):
            errors.append(f"cell {index}: CPU/no-secret contract violation")
        if offline_cpu and role != "setup" and any(marker in lowered for marker in NETWORK_MARKERS):
            errors.append(f"cell {index}: network access is only allowed in verified setup bootstrap")
        if role == "setup":
            setup_source += source
        if role == "exercise" and (
            "TODO" in source or "NotImplementedError" in source or "BROKEN:" in source
        ):
            has_exercise_marker = True
        if execute_setup and role == "setup":
            try:
                with patch(
                    "urllib.request.urlopen",
                    side_effect=RuntimeError("network disabled while validating local offline setup"),
                ):
                    exec(compiled, namespace)
            except Exception as exc:  # noqa: BLE001 - report the exact notebook fixture failure
                errors.append(f"cell {index}: setup failed: {type(exc).__name__}: {exc}")

    missing_roles = REQUIRED_ROLES - roles
    if missing_roles:
        errors.append(f"missing roles: {', '.join(sorted(missing_roles))}")
    if not has_exercise_marker:
        errors.append("exercise does not contain an explicit starter marker")
    missing_contract_markers = {
        marker for marker in EXPECTED_CONTRACT_MARKERS.get(path.relative_to(ROOT).as_posix(), set())
        if marker not in notebook_source
    }
    if missing_contract_markers:
        errors.append(f"contract markers missing: {', '.join(sorted(missing_contract_markers))}")
    expected_markers = EXPECTED_BOOTSTRAP_MARKERS.get(path.name, set())
    missing_markers = {marker for marker in expected_markers if marker not in setup_source}
    if missing_markers:
        errors.append(f"verified Colab bootstrap markers missing: {', '.join(sorted(missing_markers))}")
    if expected_markers and "urlopen(" not in setup_source:
        errors.append("verified Colab bootstrap has no download fallback")
    allowed_urls = ALLOWED_BOOTSTRAP_URLS.get(path.name, set())
    discovered_urls = set(URL_LITERAL.findall(setup_source))
    if allowed_urls and discovered_urls != allowed_urls:
        extra = discovered_urls - allowed_urls
        missing = allowed_urls - discovered_urls
        if extra:
            errors.append(f"setup contains non-whitelisted URLs: {', '.join(sorted(extra))}")
        if missing:
            errors.append(f"setup is missing whitelisted URLs: {', '.join(sorted(missing))}")
    errors.extend(
        validate_setup_trust_contract(
            path,
            setup_source=setup_source,
            namespace=namespace,
            execute_setup=execute_setup,
        )
    )
    return errors


def validate_dataset_manifest() -> tuple[set[str], list[str]]:
    errors: list[str] = []
    if not DATASET_MANIFEST_PATH.is_file():
        return set(), ["data/datasets.json is missing"]

    manifest = load_json(DATASET_MANIFEST_PATH)
    if manifest.get("schemaVersion") != 2:
        errors.append("data/datasets.json: schemaVersion must be 2")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list):
        return set(), [*errors, "data/datasets.json: datasets must be a list"]

    dataset_ids: set[str] = set()
    for dataset in datasets:
        dataset_id = dataset.get("id")
        if not isinstance(dataset_id, str) or not dataset_id:
            errors.append("data/datasets.json: dataset without id")
            continue
        if dataset_id in dataset_ids:
            errors.append(f"data/datasets.json: duplicate dataset id {dataset_id}")
        dataset_ids.add(dataset_id)

        expected_license = EXPECTED_DATASET_LICENSES.get(dataset_id)
        if expected_license and dataset.get("license") != expected_license:
            errors.append(f"{dataset_id}: expected license {expected_license}")
        for key in ("name", "license", "licenseUrl", "attribution", "sourceUrl"):
            if not dataset.get(key):
                errors.append(f"{dataset_id}: missing {key}")

        delivery = dataset.get("delivery")
        if delivery not in {"packaged", "runtime_download"}:
            errors.append(f"{dataset_id}: invalid delivery {delivery}")
            continue

        artifacts = dataset.get("artifacts")
        if not isinstance(artifacts, list):
            errors.append(f"{dataset_id}: artifacts must be a list")
            continue
        if delivery == "runtime_download":
            if artifacts:
                errors.append(f"{dataset_id}: runtime_download must not package artifacts")
            if dataset.get("redistribution") != "not-packaged":
                errors.append(f"{dataset_id}: runtime_download must declare redistribution=not-packaged")
            if not re.fullmatch(r"[0-9a-f]{64}", str(dataset.get("sourceSha256", ""))):
                errors.append(f"{dataset_id}: runtime_download requires sourceSha256")
            if not isinstance(dataset.get("sourceBytes"), int) or dataset["sourceBytes"] <= 0:
                errors.append(f"{dataset_id}: runtime_download requires sourceBytes")
            source_host = urlparse(dataset["sourceUrl"]).hostname
            if urlparse(dataset["sourceUrl"]).scheme != "https" or not source_host:
                errors.append(f"{dataset_id}: runtime_download requires an HTTPS sourceUrl")
            if not dataset.get("sourceFinalHost"):
                errors.append(f"{dataset_id}: runtime_download requires sourceFinalHost")
            if dataset_id == "cifar-10" and dataset.get("sourceFinalHost") != "cave.cs.toronto.edu":
                errors.append("cifar-10: unexpected official redirect host")
            continue
        if not artifacts:
            errors.append(f"{dataset_id}: packaged dataset requires artifacts")
            continue
        artifact_paths = {artifact.get("path") for artifact in artifacts}
        missing_artifacts = REQUIRED_DATASET_ARTIFACTS.get(dataset_id, set()) - artifact_paths
        if missing_artifacts:
            errors.append(f"{dataset_id}: missing packaged artifacts {', '.join(sorted(missing_artifacts))}")
        for artifact in artifacts:
            relative_path = artifact.get("path")
            try:
                path = repository_path(relative_path)
            except (TypeError, ValueError) as exc:
                errors.append(f"{dataset_id}: invalid artifact path: {exc}")
                continue
            if not path.is_file():
                errors.append(f"{dataset_id}: missing artifact {relative_path}")
                continue
            payload = path.read_bytes()
            if artifact.get("bytes") != len(payload):
                errors.append(f"{dataset_id}: byte count mismatch for {relative_path}")
            if artifact.get("sha256") != sha256_bytes(payload):
                errors.append(f"{dataset_id}: checksum mismatch for {relative_path}")

        if dataset_id == "squad2-rag-subset":
            if not dataset.get("modificationNotice"):
                errors.append("squad2-rag-subset: modificationNotice is required")
            subset_path = ROOT / "data" / "squad2" / "squad2-rag-subset.json"
            if subset_path.is_file():
                subset = load_json(subset_path)
                if subset.get("license") != "CC-BY-SA-4.0" or not subset.get("modificationNotice"):
                    errors.append("squad2-rag-subset: fixture must preserve license and modification notice")
        if dataset_id == "yambda-50m-likes-compact":
            if not dataset.get("modificationNotice"):
                errors.append("yambda-50m-likes-compact: modificationNotice is required")
            expected_stats = {"events": 268631, "users": 1999, "items": 42965, "sourceEvents": 881456, "sourceUsers": 8283}
            if dataset.get("statistics") != expected_stats:
                errors.append("yambda-50m-likes-compact: unexpected statistics")

    packaged_license_expectations = {
        ROOT / "data" / "uci-bike-sharing" / "LICENSE": "CC BY 4.0",
        ROOT / "data" / "squad2" / "LICENSE": "CC BY-SA 4.0",
        ROOT / "data" / "squad2" / "ATTRIBUTION.md": "adaptation",
        ROOT / "data" / "yambda" / "LICENSE": "Apache License",
        ROOT / "data" / "yambda" / "ATTRIBUTION.md": "deterministic educational subset",
    }
    for path, marker in packaged_license_expectations.items():
        if not path.is_file() or marker.lower() not in path.read_text(encoding="utf-8").lower():
            errors.append(f"{path.relative_to(ROOT)}: missing license/attribution marker {marker}")

    third_party_expectations = {
        ROOT / "third_party" / "yandexdataschool-practical-dl-MIT.txt": "Permission is hereby granted",
        ROOT / "THIRD_PARTY_NOTICES.md": "YDS Practical_DL",
    }
    for path, marker in third_party_expectations.items():
        if not path.is_file() or marker.lower() not in path.read_text(encoding="utf-8").lower():
            errors.append(f"{path.relative_to(ROOT)}: missing third-party notice marker {marker}")

    missing_expected = set(EXPECTED_DATASET_LICENSES) - dataset_ids
    if missing_expected:
        errors.append(f"data/datasets.json: missing datasets {', '.join(sorted(missing_expected))}")
    return dataset_ids, errors


def validate_benchmark_manifest(*, serving_lab_public: bool) -> list[str]:
    errors: list[str] = []
    if not BENCHMARK_MANIFEST_PATH.is_file():
        return ["benchmark manifest is missing"]
    manifest = load_json(BENCHMARK_MANIFEST_PATH)
    if manifest.get("schemaVersion") != 1:
        errors.append("benchmark manifest schemaVersion must be 1")

    try:
        protocol_path = repository_path(manifest.get("protocolPath"))
    except (TypeError, ValueError) as exc:
        return [*errors, f"benchmark manifest has invalid protocolPath: {exc}"]
    if not protocol_path.is_file():
        errors.append("benchmark protocol is missing")
    elif manifest.get("protocolSha256") != sha256_bytes(protocol_path.read_bytes()):
        errors.append("benchmark protocol checksum mismatch")

    runs = manifest.get("runs")
    if not isinstance(runs, list):
        return [*errors, "benchmark runs must be a list"]
    engines: set[str] = set()
    required_run_fields = {
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
    }
    for index, run in enumerate(runs):
        missing = required_run_fields - set(run)
        if missing:
            errors.append(f"benchmark run {index}: missing {', '.join(sorted(missing))}")
            continue
        if run.get("provenance") != "recorded":
            errors.append(f"benchmark run {index}: provenance must be recorded")
        engine = run.get("engine")
        if engine not in {"vllm", "sglang"}:
            errors.append(f"benchmark run {index}: unknown engine {engine}")
        else:
            engines.add(engine)
        try:
            fixture_path = repository_path(run.get("rawFixturePath"))
        except (TypeError, ValueError) as exc:
            errors.append(f"benchmark run {index}: invalid fixture path: {exc}")
            continue
        if not fixture_path.is_file():
            errors.append(f"benchmark run {index}: fixture is missing")
        elif run.get("rawFixtureSha256") != sha256_bytes(fixture_path.read_bytes()):
            errors.append(f"benchmark run {index}: fixture checksum mismatch")

    if serving_lab_public and (len(runs) < 2 or engines != {"vllm", "sglang"}):
        errors.append("public serving lab requires recorded fixtures for both vLLM and SGLang")
    return errors


def validate_repository(*, execute_setup: bool) -> list[str]:
    failures: list[str] = []
    if not LAB_MANIFEST_PATH.is_file():
        return ["labs-manifest.json is missing"]

    dataset_ids, dataset_errors = validate_dataset_manifest()
    failures.extend(dataset_errors)

    manifest = load_json(LAB_MANIFEST_PATH)
    if manifest.get("schemaVersion") != 1:
        failures.append("labs-manifest.json: schemaVersion must be 1")
    labs = manifest.get("labs")
    if not isinstance(labs, list):
        return [*failures, "labs-manifest.json: labs must be a list"]

    dataset_ref = manifest.get("datasetManifest", {})
    if dataset_ref.get("path") != "data/datasets.json":
        failures.append("labs-manifest.json: unexpected dataset manifest path")
    elif DATASET_MANIFEST_PATH.is_file() and dataset_ref.get("sha256") != sha256_bytes(
        DATASET_MANIFEST_PATH.read_bytes()
    ):
        failures.append("labs-manifest.json: dataset manifest checksum mismatch")

    entries_by_path: dict[str, dict] = {}
    for entry in labs:
        relative_path = entry.get("path")
        if not isinstance(relative_path, str) or not relative_path:
            failures.append("labs-manifest.json: lab without path")
            continue
        if relative_path in entries_by_path:
            failures.append(f"labs-manifest.json: duplicate lab {relative_path}")
        entries_by_path[relative_path] = entry

        status = entry.get("releaseStatus")
        if status not in ALLOWED_RELEASE_STATUSES:
            failures.append(f"{relative_path}: invalid releaseStatus {status}")
        if not entry.get("version") or entry.get("license") != "MIT":
            failures.append(f"{relative_path}: missing version or MIT license")
        if entry.get("containsSolutions") is not False or entry.get("containsExecutionOutputs") is not False:
            failures.append(f"{relative_path}: unsafe starter metadata")
        unknown_datasets = set(entry.get("datasets", [])) - dataset_ids
        if unknown_datasets:
            failures.append(f"{relative_path}: unknown datasets {', '.join(sorted(unknown_datasets))}")

        try:
            notebook_path = repository_path(relative_path)
        except (TypeError, ValueError) as exc:
            failures.append(f"{relative_path}: invalid path: {exc}")
            continue
        if not notebook_path.is_file():
            failures.append(f"{relative_path}: notebook is missing")
            continue
        actual_hash = sha256_bytes(notebook_path.read_bytes())
        if entry.get("notebookSha256") != actual_hash:
            failures.append(f"{relative_path}: notebook checksum mismatch")
        expected_old_hash = EXPECTED_OLD_NOTEBOOK_HASHES.get(relative_path)
        if expected_old_hash and actual_hash != expected_old_hash:
            failures.append(f"{relative_path}: original starter bytes changed")
        for error in validate_notebook(
            notebook_path,
            execute_setup=execute_setup,
            offline_cpu=relative_path in CPU_STARTER_LABS,
        ):
            failures.append(f"{relative_path}: {error}")

    discovered = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    }
    manifest_paths = set(entries_by_path)
    if discovered != manifest_paths:
        missing = manifest_paths - discovered
        untracked = discovered - manifest_paths
        if missing:
            failures.append(f"manifest notebooks missing from disk: {', '.join(sorted(missing))}")
        if untracked:
            failures.append(f"notebooks absent from manifest: {', '.join(sorted(untracked))}")

    expected_public = [entry["path"] for entry in labs if entry.get("releaseStatus") == "public"]
    public_index = manifest.get("publicIndex")
    if public_index != expected_public:
        failures.append("publicIndex must contain every and only public lab, in manifest order")
    for relative_path in public_index if isinstance(public_index, list) else []:
        if entries_by_path.get(relative_path, {}).get("releaseStatus") != "public":
            failures.append(f"publicIndex exposes non-public lab {relative_path}")

    serving_lab_public = entries_by_path.get("10-llm-serving-benchmark.ipynb", {}).get("releaseStatus") == "public"
    failures.extend(validate_benchmark_manifest(serving_lab_public=serving_lab_public))
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-setup", action="store_true")
    args = parser.parse_args()

    failures = validate_repository(execute_setup=args.execute_setup)
    if failures:
        raise SystemExit("\n".join(failures))

    manifest = load_json(LAB_MANIFEST_PATH)
    status_counts = {
        status: sum(entry["releaseStatus"] == status for entry in manifest["labs"])
        for status in sorted(ALLOWED_RELEASE_STATUSES)
    }
    print(f"Validated {len(manifest['labs'])} starter notebooks: {status_counts}")


if __name__ == "__main__":
    main()
