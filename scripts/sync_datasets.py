from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


UCI_LICENSE = b"""Bike Sharing Dataset license notice

The Bike Sharing Dataset is distributed by the UCI Machine Learning
Repository under the Creative Commons Attribution 4.0 International license
(CC BY 4.0): https://creativecommons.org/licenses/by/4.0/legalcode

This notice does not replace the license text at the canonical URL.
"""

UCI_ATTRIBUTION = b"""# Attribution: UCI Bike Sharing Dataset

- Dataset: Bike Sharing Dataset (hourly aggregation)
- Authors: Hadi Fanaee-T and Joao Gama
- Publisher: UCI Machine Learning Repository
- DOI: https://doi.org/10.24432/C5W894
- Source snapshot: `bike+sharing+dataset.zip`, verified by SHA-256 in
  `data/datasets.json`
- License: CC BY 4.0

The repository keeps the publisher's `Readme.txt` unchanged. ML Mentor only
packages `hour.csv` for an educational starter notebook; the observations and
labels are not modified.
"""

SQUAD_LICENSE = b"""SQuAD 2.0 derivative fixture license notice

The source SQuAD 2.0 dataset and this adapted subset are distributed under the
Creative Commons Attribution-ShareAlike 4.0 International license
(CC BY-SA 4.0): https://creativecommons.org/licenses/by-sa/4.0/legalcode

This notice does not replace the license text at the canonical URL. Any
redistribution or adaptation of this fixture must preserve attribution and the
share-alike terms.
"""

SQUAD_ATTRIBUTION = b"""# Attribution and modification notice: SQuAD 2.0 subset

- Source dataset: Stanford Question Answering Dataset 2.0 (SQuAD 2.0)
- Authors: Pranav Rajpurkar, Robin Jia, Percy Liang and collaborators
- Project: https://rajpurkar.github.io/SQuAD-explorer/
- Source file: `dev-v2.0.json`, verified by SHA-256 in `data/datasets.json`
- License: CC BY-SA 4.0

ML Mentor created a deterministic educational subset. It keeps the first 100
eligible contexts in source order, one answerable question per context, and
one impossible question for the first 50 eligible contexts that contain one.
Duplicate answer spans are removed and internal field names are normalized.
Question, context and answer text are otherwise unchanged. This file is an
adaptation, so the CC BY-SA 4.0 share-alike requirement continues to apply.
"""


SOURCES = {
    "uci_bike_zip": {
        "url": "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
        "sha256": "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401",
    },
    "banking77_train": {
        "url": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/train.csv",
        "sha256": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    },
    "banking77_test": {
        "url": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/test.csv",
        "sha256": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    },
    "banking77_categories": {
        "url": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/categories.json",
        "sha256": "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
    },
    "banking77_license": {
        "url": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/LICENSE",
        "sha256": "7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661",
    },
    "squad2_dev": {
        "url": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
        "sha256": "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",
    },
}

EXPECTED_SOURCE_HOSTS = {
    "uci_bike_zip": "archive.ics.uci.edu",
    "banking77_train": "raw.githubusercontent.com",
    "banking77_test": "raw.githubusercontent.com",
    "banking77_categories": "raw.githubusercontent.com",
    "banking77_license": "raw.githubusercontent.com",
    "squad2_dev": "rajpurkar.github.io",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch(name: str) -> bytes:
    spec = SOURCES[name]
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "ML-Mentor-Labs/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        final_host = urlparse(response.geturl()).hostname
        if final_host != EXPECTED_SOURCE_HOSTS[name]:
            raise RuntimeError(f"{name}: unexpected redirect host {final_host}")
        payload = response.read()
    actual = sha256(payload)
    if actual != spec["sha256"]:
        raise RuntimeError(f"{name}: source checksum changed: {actual}")
    return payload


def normalized_json(data: object) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def build_squad_subset(source: bytes) -> bytes:
    raw = json.loads(source)
    documents: list[dict] = []
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
                seen: set[tuple[str, int]] = set()
                answers = []
                for answer in qa["answers"]:
                    key = (answer["text"], answer["answer_start"])
                    if key in seen:
                        continue
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

            documents.append(
                {
                    "context": paragraph["context"],
                    "document_id": f"squad2-{len(documents):03d}",
                    "qas": qas,
                    "title": article["title"],
                }
            )
            if len(documents) == 100:
                break
        if len(documents) == 100:
            break

    if len(documents) != 100 or impossible_examples != 50:
        raise RuntimeError(
            f"unexpected SQuAD subset shape: documents={len(documents)}, impossible={impossible_examples}"
        )

    return normalized_json(
        {
            "attribution": "Rajpurkar, Jia et al. Stanford Question Answering Dataset 2.0.",
            "dataset": "SQuAD 2.0",
            "documents": documents,
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
            "source_sha256": SOURCES["squad2_dev"]["sha256"],
            "source_version": raw["version"],
        }
    )


def build_artifacts() -> tuple[dict[str, bytes], dict]:
    bike_zip = fetch("uci_bike_zip")
    banking_train = fetch("banking77_train")
    banking_test = fetch("banking77_test")
    banking_categories = fetch("banking77_categories")
    banking_license = fetch("banking77_license")
    squad_source = fetch("squad2_dev")

    with zipfile.ZipFile(io.BytesIO(bike_zip)) as archive:
        bike_hour = archive.read("hour.csv")
        bike_readme = archive.read("Readme.txt")

    artifacts = {
        "data/uci-bike-sharing/hour.csv": bike_hour,
        "data/uci-bike-sharing/source-readme.txt": bike_readme,
        "data/uci-bike-sharing/LICENSE": UCI_LICENSE,
        "data/uci-bike-sharing/ATTRIBUTION.md": UCI_ATTRIBUTION,
        "data/banking77/train.csv": banking_train,
        "data/banking77/test.csv": banking_test,
        "data/banking77/categories.json": banking_categories,
        "data/banking77/LICENSE": banking_license,
        "data/squad2/squad2-rag-subset.json": build_squad_subset(squad_source),
        "data/squad2/LICENSE": SQUAD_LICENSE,
        "data/squad2/ATTRIBUTION.md": SQUAD_ATTRIBUTION,
    }

    def artifact(path: str) -> dict:
        payload = artifacts[path]
        return {"path": path, "bytes": len(payload), "sha256": sha256(payload)}

    manifest = {
        "schemaVersion": 1,
        "datasets": [
            {
                "id": "uci-bike-sharing-hour",
                "name": "Bike Sharing Dataset (hourly aggregation)",
                "license": "CC-BY-4.0",
                "licenseUrl": "https://creativecommons.org/licenses/by/4.0/legalcode",
                "attribution": "Fanaee-T, Hadi; Gama, Joao. Bike Sharing. UCI Machine Learning Repository. DOI: 10.24432/C5W894.",
                "sourceUrl": SOURCES["uci_bike_zip"]["url"],
                "sourceSha256": SOURCES["uci_bike_zip"]["sha256"],
                "artifacts": [
                    artifact("data/uci-bike-sharing/hour.csv"),
                    artifact("data/uci-bike-sharing/source-readme.txt"),
                    artifact("data/uci-bike-sharing/LICENSE"),
                    artifact("data/uci-bike-sharing/ATTRIBUTION.md"),
                ],
            },
            {
                "id": "banking77",
                "name": "Banking77",
                "license": "CC-BY-4.0",
                "licenseUrl": "https://creativecommons.org/licenses/by/4.0/legalcode",
                "attribution": "Casanueva et al. Efficient Intent Detection with Dual Sentence Encoders. PolyAI-LDN/task-specific-datasets.",
                "sourceRevision": "57ec275d8078af65b7731c2a98be812d844a6d6b",
                "sourceUrl": "https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data",
                "sourceFiles": [
                    {"url": SOURCES[name]["url"], "sha256": SOURCES[name]["sha256"]}
                    for name in ("banking77_train", "banking77_test", "banking77_categories", "banking77_license")
                ],
                "artifacts": [
                    artifact("data/banking77/train.csv"),
                    artifact("data/banking77/test.csv"),
                    artifact("data/banking77/categories.json"),
                    artifact("data/banking77/LICENSE"),
                ],
            },
            {
                "id": "squad2-rag-subset",
                "name": "SQuAD 2.0 fixed RAG subset",
                "license": "CC-BY-SA-4.0",
                "licenseUrl": "https://creativecommons.org/licenses/by-sa/4.0/legalcode",
                "attribution": "Rajpurkar, Jia et al. Stanford Question Answering Dataset 2.0.",
                "sourceUrl": SOURCES["squad2_dev"]["url"],
                "sourceSha256": SOURCES["squad2_dev"]["sha256"],
                "selection": "100 contexts in source order; one answerable QA each; one impossible QA for first 50 eligible contexts",
                "modificationNotice": (
                    "Deterministic adapted subset; selection and normalization details are stored "
                    "in data/squad2/ATTRIBUTION.md and inside the fixture."
                ),
                "artifacts": [
                    artifact("data/squad2/squad2-rag-subset.json"),
                    artifact("data/squad2/LICENSE"),
                    artifact("data/squad2/ATTRIBUTION.md"),
                ],
            },
        ],
    }
    return artifacts, manifest


def write_or_check(path: Path, payload: bytes, *, check: bool) -> None:
    if check:
        if not path.exists() or path.read_bytes() != payload:
            raise RuntimeError(f"generated artifact is stale: {path.relative_to(ROOT)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="download sources and verify committed artifacts")
    args = parser.parse_args()

    artifacts, manifest = build_artifacts()
    stale_artifacts = [DATA_ROOT / "uci-bike-sharing" / "day.csv"]
    for path in stale_artifacts:
        if args.check and path.exists():
            raise RuntimeError(f"stale generated artifact must be removed: {path.relative_to(ROOT)}")
        if not args.check and path.exists():
            path.unlink()
    for relative_path, payload in artifacts.items():
        write_or_check(ROOT / relative_path, payload, check=args.check)
    write_or_check(DATA_ROOT / "datasets.json", normalized_json(manifest), check=args.check)
    action = "Verified" if args.check else "Wrote"
    print(f"{action} {len(artifacts)} dataset artifacts")


if __name__ == "__main__":
    main()
