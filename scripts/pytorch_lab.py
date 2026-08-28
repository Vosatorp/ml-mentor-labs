from __future__ import annotations

from collections.abc import Callable


CellFactory = Callable[..., dict]


def build_pytorch_lab(*, md: CellFactory, code: CellFactory, notebook: CellFactory) -> dict[str, dict]:
    """Build the CIFAR-10 starter without embedding an answer key."""

    return {
        "11-cifar10-pytorch-training-pipeline.ipynb": notebook(
            "PyTorch: воспроизводимый цикл обучения на CIFAR-10",
            [
                md(
                    """## Что будет сделано

В этой лабораторной нужно провести code review сломанного цикла, собрать честный конвейер обучения небольшой свёрточной сети и проверить, что модель можно восстановить из checkpoint. Обязательная часть выполняется на CPU: фиксированные 10 000 изображений для обучения, 2 000 для проверки и три эпохи. Дополнительная часть использует полный CIFAR-10, GPU и автоматический выбор числовой точности (AMP).

**Итог работы:** таблица с loss, accuracy, временем, числом параметров, найденными ошибками и ограничениями эксперимента. При корректной реализации конечный train loss ниже начального, validation accuracy не ниже 40%, повторная проверка детерминирована, а восстановленная модель выдаёт те же logits на закреплённом batch.

### Происхождение учебной механики

- **Адаптировано из YDS Practical_DL:** последовательность «разобрать ошибочный код → собрать цикл обучения → проверить результат» и работа с небольшим vision-датасетом. Репозиторий Yandex Data School распространяется по MIT; notice находится в `third_party/yandexdataschool-practical-dl-MIT.txt`. Код, формулировки и проверки этой лабораторной написаны заново для ML Mentor.
- **Дополнительное русское объяснение — «мыш»:** статьи перечислены в конце как чтение. Их код, текст и иллюстрации не копируются: лицензия сайта CC BY-NC-ND 4.0 не разрешает такое использование в коммерческом материале.
- **Официальная документация PyTorch:** поведение `CrossEntropyLoss`, режимов модели, AMP и checkpoint проверяется по официальной документации.

CIFAR-10 загружается с сайта авторов, не хранится в Git и проверяется по SHA-256 до распаковки. Набор содержит 60 000 цветных изображений 32×32 в десяти классах: 50 000 train и 10 000 test. Используйте его только в рамках условий, указанных авторами на официальной странице; в `data/datasets.json` зафиксированы источник, атрибуция и технический отчёт."""
                ),
                md(
                    """## 1. Code review до запуска

Ниже намеренно сломанный фрагмент. Не запускайте его. Выпишите минимум семь дефектов и для каждого укажите последствие: неверный градиент, недетерминированная проверка, лишняя память, неправильная функция потерь или ошибочный шаг оптимизатора.

```python
def broken_epoch(model, loader, optimizer, criterion):
    model.eval()
    for images, target in loader:
        probabilities = model(images).softmax(dim=1)
        loss = criterion(probabilities, target)
        optimizer.step()
        loss.backward()
    return loss

def broken_validation(model, loader):
    model.train()
    correct = 0
    for images, target in loader:
        logits = model(images)
        correct += (logits.argmax(1) == target).sum().item()
    return correct / len(loader)
```

Особенно проверьте режимы `train()`/`eval()`, устройство тензоров, вход `CrossEntropyLoss`, порядок `zero_grad → forward → backward → step`, запись графа во время проверки и знаменатель accuracy."""
                ),
                code(
                    """import hashlib
import json
import os
import random
import tarfile
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


SEED = 42
REQUIRED_DEVICE = torch.device("cpu")
CIFAR10_SOURCE_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_SOURCE_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
CIFAR10_ARCHIVE_BYTES = 170_498_071
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = (destination / member.name).resolve()
            if destination != member_path and destination not in member_path.parents:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        archive.extractall(destination)


def ensure_cifar10(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    extracted = root / "cifar-10-batches-py"
    if extracted.is_dir():
        return root

    archive_path = root / "cifar-10-python.tar.gz"
    if archive_path.is_file():
        payload = archive_path.read_bytes()
        require_sha256(payload, CIFAR10_SOURCE_SHA256, "cached CIFAR-10 archive")
    else:
        request = urllib.request.Request(
            CIFAR10_SOURCE_URL,
            headers={"User-Agent": "ML-Mentor-Labs/1.0"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            require_final_hostname(response, "cave.cs.toronto.edu")
            payload = response.read()
        if len(payload) != CIFAR10_ARCHIVE_BYTES:
            raise RuntimeError(f"CIFAR-10 archive size changed: {len(payload)}")
        require_sha256(payload, CIFAR10_SOURCE_SHA256, "downloaded CIFAR-10 archive")
        archive_path.write_bytes(payload)

    _safe_extract_tar(archive_path, root)
    if not extracted.is_dir():
        raise RuntimeError("CIFAR-10 archive does not contain cifar-10-batches-py")
    return root


set_seed()
DATA_ROOT = Path.home() / ".cache" / "ml-mentor-labs" / "cifar10"
""",
                    role="setup",
                ),
                md(
                    """## 2. Данные и честное разбиение

Запустите ячейку ниже: первый запуск скачает около 163 MiB. Обучающая и проверочная выборки создаются из официальной train-части. Индексы выбираются стратифицированно и закреплены seed: по 1 000 train и 200 validation изображений каждого класса.

Создаются **два разных объекта** `CIFAR10`: случайный crop и отражение применяются только к train. Validation использует только преобразование в тензор и нормализацию. Если один объект датасета с аугментациями разделить через два `Subset`, случайные преобразования попадут в validation и метрика перестанет описывать одну и ту же выборку."""
                ),
                code(
                    """ensure_cifar10(DATA_ROOT)

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
validation_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_source = datasets.CIFAR10(DATA_ROOT, train=True, transform=train_transform, download=False)
validation_source = datasets.CIFAR10(DATA_ROOT, train=True, transform=validation_transform, download=False)
all_indices = np.arange(len(train_source))
all_targets = np.asarray(train_source.targets)
selected_indices, _ = train_test_split(
    all_indices,
    train_size=12_000,
    stratify=all_targets,
    random_state=SEED,
)
train_indices, validation_indices = train_test_split(
    selected_indices,
    test_size=2_000,
    stratify=all_targets[selected_indices],
    random_state=SEED,
)

train_dataset = Subset(train_source, train_indices.tolist())
validation_dataset = Subset(validation_source, validation_indices.tolist())
loader_generator = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0,
    generator=loader_generator,
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0,
)

sample_images, sample_targets = next(iter(validation_loader))
class_counts_train = np.bincount(all_targets[train_indices], minlength=10)
class_counts_validation = np.bincount(all_targets[validation_indices], minlength=10)
print({
    "train": len(train_dataset),
    "validation": len(validation_dataset),
    "shape": tuple(sample_images.shape),
    "dtype": str(sample_images.dtype),
    "classes": CIFAR10_CLASSES,
})
""",
                    role="exercise",
                ),
                code(
                    """assert len(train_dataset) == 10_000
assert len(validation_dataset) == 2_000
assert not (set(train_indices) & set(validation_indices))
assert tuple(sample_images.shape[1:]) == (3, 32, 32)
assert sample_images.dtype == torch.float32
assert sample_targets.dtype == torch.int64
assert class_counts_train.tolist() == [1_000] * 10
assert class_counts_validation.tolist() == [200] * 10
assert train_source.transform is not validation_source.transform
assert any(isinstance(step, transforms.RandomHorizontalFlip) for step in train_transform.transforms)
assert not any(isinstance(step, transforms.RandomHorizontalFlip) for step in validation_transform.transforms)
print("Разбиение и transforms проверены")
""",
                    role="self_check",
                ),
                md(
                    """## 3. Компактная CNN

Модель намеренно небольшая, чтобы обязательный путь оставался доступным на CPU. Она возвращает десять **raw logits**. `CrossEntropyLoss` сама применяет log-softmax; добавлять `softmax` перед loss нельзя. Вероятности нужны только после обучения, когда они действительно используются в отчёте или интерфейсе."""
                ),
                code(
                    """class CompactCifarCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(features.flatten(1))


set_seed()
model = CompactCifarCNN().to(REQUIRED_DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
MODEL_CONFIG = {
    "architecture": "CompactCifarCNN",
    "num_classes": 10,
    "optimizer": "AdamW",
    "learning_rate": 3e-3,
    "weight_decay": 1e-4,
    "seed": SEED,
}
parameter_count = sum(parameter.numel() for parameter in model.parameters())
assert model(torch.zeros(4, 3, 32, 32, device=REQUIRED_DEVICE)).shape == (4, 10)
print({"parameters": parameter_count, "device": str(REQUIRED_DEVICE)})
""",
                    role="setup",
                ),
                md(
                    """## 4. Реализация train и validation

Реализуйте две функции. `train_one_epoch` обязана поддерживать gradient accumulation. Делите loss на фактический размер текущей группы micro-batch: последняя группа может быть короче `accumulation_steps`. Шаг оптимизатора выполняется и для такого остатка. Метрики epoch считайте по исходному, не поделённому loss и по числу объектов.

`evaluate` переводит модель в `eval()`, использует `torch.inference_mode()` и не меняет параметры. Обе функции переносят images и targets на переданное устройство. Параметр `scaler` используется только в дополнительной GPU-части; на CPU он равен `None`."""
                ),
                code(
                    """def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    accumulation_steps: int = 1,
    scaler=None,
) -> dict[str, float]:
    # TODO: train(), device, zero_grad -> forward -> backward -> step.
    # Учтите последнюю неполную accumulation-группу и optional AMP scaler.
    raise NotImplementedError


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    # TODO: eval(), torch.inference_mode(), loss и accuracy по числу объектов.
    raise NotImplementedError
"""
                ),
                code(
                    """# Небольшая проверка контракта до трёх полных эпох.
probe_model = nn.Linear(4, 2)
probe_optimizer = torch.optim.SGD(probe_model.parameters(), lr=0.05)
probe_x = torch.randn(10, 4)
probe_y = (probe_x[:, 0] > 0).long()
probe_loader = DataLoader(torch.utils.data.TensorDataset(probe_x, probe_y), batch_size=3, shuffle=False)
before = [parameter.detach().clone() for parameter in probe_model.parameters()]
probe_train = train_one_epoch(
    probe_model,
    probe_loader,
    probe_optimizer,
    nn.CrossEntropyLoss(),
    torch.device("cpu"),
    accumulation_steps=3,
)
after = list(probe_model.parameters())
assert any(not torch.equal(left, right) for left, right in zip(before, after))
assert set(probe_train) == {"loss", "accuracy"}
assert np.isfinite(probe_train["loss"])
assert 0 <= probe_train["accuracy"] <= 1
print("Train-контракт и остаточный accumulation batch проверены")
""",
                    role="self_check",
                ),
                md(
                    """## 5. Три обязательные эпохи на CPU

Обязательный эксперимент всегда использует `REQUIRED_DEVICE`, то есть CPU. Это делает результат сравнимым и не превращает доступ к GPU в условие прохождения. Зафиксируйте время каждой эпохи и обе метрики."""
                ),
                code(
                    """history = []
required_started_at = time.perf_counter()
for epoch in range(1, 4):
    epoch_started_at = time.perf_counter()
    train_metrics = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        REQUIRED_DEVICE,
        accumulation_steps=1,
    )
    validation_metrics = evaluate(model, validation_loader, criterion, REQUIRED_DEVICE)
    row = {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "validation_loss": validation_metrics["loss"],
        "validation_accuracy": validation_metrics["accuracy"],
        "seconds": time.perf_counter() - epoch_started_at,
    }
    history.append(row)
    print(row)
required_seconds = time.perf_counter() - required_started_at
history_frame = pd.DataFrame(history)
"""
                ),
                code(
                    """assert len(history_frame) == 3
assert history_frame.iloc[-1].train_loss < history_frame.iloc[0].train_loss, history_frame
assert history_frame.iloc[-1].validation_accuracy >= 0.40, history_frame
validation_1 = evaluate(model, validation_loader, criterion, REQUIRED_DEVICE)
validation_2 = evaluate(model, validation_loader, criterion, REQUIRED_DEVICE)
assert validation_1 == validation_2, (validation_1, validation_2)
assert model.training is False
print({"required_seconds": required_seconds, **validation_2})
""",
                    role="self_check",
                ),
                md(
                    """## 6. Надёжный checkpoint

Checkpoint должен содержать состояния модели и оптимизатора, номер эпохи и конфигурацию эксперимента. Реализуйте атомарную запись через временный файл, сохраните SHA-256 рядом и проверяйте его **до** `torch.load`. При загрузке проверьте обязательные ключи и совпадение конфигурации. Тест ниже также создаёт повреждённую копию и убеждается, что она отвергается."""
                ),
                code(
                    """def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict,
) -> str:
    # TODO: torch.save во временный файл, os.replace, SHA-256 в path.with_suffix(path.suffix + '.sha256').
    raise NotImplementedError


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_config: dict,
    device: torch.device,
) -> dict:
    # TODO: сначала проверьте SHA-256, затем обязательные ключи/config и загрузите state_dict.
    raise NotImplementedError


CHECKPOINT_PATH = Path("/tmp/ml-mentor-cifar10-cpu-v1.pt")
checkpoint_sha256 = save_checkpoint(
    CHECKPOINT_PATH,
    model=model,
    optimizer=optimizer,
    epoch=3,
    config=MODEL_CONFIG,
)
"""
                ),
                code(
                    """set_seed()
restored_model = CompactCifarCNN().to(REQUIRED_DEVICE)
restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=3e-3, weight_decay=1e-4)
checkpoint = load_checkpoint(
    CHECKPOINT_PATH,
    model=restored_model,
    optimizer=restored_optimizer,
    expected_config=MODEL_CONFIG,
    device=REQUIRED_DEVICE,
)
model.eval()
restored_model.eval()
fixed_images = sample_images[:32].to(REQUIRED_DEVICE)
with torch.inference_mode():
    original_logits = model(fixed_images)
    restored_logits = restored_model(fixed_images)
assert checkpoint["epoch"] == 3
assert torch.equal(original_logits, restored_logits)

corrupt_path = CHECKPOINT_PATH.with_name("ml-mentor-cifar10-corrupt.pt")
corrupt_payload = bytearray(CHECKPOINT_PATH.read_bytes())
corrupt_payload[len(corrupt_payload) // 2] ^= 0x01
corrupt_path.write_bytes(corrupt_payload)
corrupt_path.with_suffix(corrupt_path.suffix + ".sha256").write_text(checkpoint_sha256, encoding="utf-8")
try:
    load_checkpoint(
        corrupt_path,
        model=CompactCifarCNN(),
        optimizer=torch.optim.AdamW(CompactCifarCNN().parameters(), lr=3e-3),
        expected_config=MODEL_CONFIG,
        device=REQUIRED_DEVICE,
    )
except RuntimeError:
    pass
else:
    raise AssertionError("Повреждённый checkpoint должен быть отвергнут")
print({"checkpoint_sha256": checkpoint_sha256, "logits_identical": True})
""",
                    role="self_check",
                ),
                md(
                    """## 7. Дополнительно: полный CIFAR-10, GPU и AMP

Этот раздел не влияет на обязательное прохождение. Если CUDA недоступна, ячейка корректно сообщает об этом и ничего не запускает. При наличии GPU используйте все 50 000 train-изображений и официальный test-набор только для финальной оценки. AMP ускоряет вычисления и уменьшает память, но меняет численную точность; сравнивайте время и качество, а не только факт успешного запуска.

Не подбирайте гиперпараметры по официальному test. Для полноценной разработки следует выделить validation из train, зафиксировать выбор, затем открыть test один раз."""
                ),
                code(
                    """GPU_REPORT = {"status": "skipped", "reason": "CUDA is unavailable"}
if torch.cuda.is_available():
    gpu_device = torch.device("cuda")
    full_train = datasets.CIFAR10(DATA_ROOT, train=True, transform=train_transform, download=False)
    official_test = datasets.CIFAR10(DATA_ROOT, train=False, transform=validation_transform, download=False)
    full_train_loader = DataLoader(full_train, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    official_test_loader = DataLoader(official_test, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    gpu_model = CompactCifarCNN().to(gpu_device)
    gpu_optimizer = torch.optim.AdamW(gpu_model.parameters(), lr=3e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    gpu_started_at = time.perf_counter()
    # TODO (optional): выполните минимум одну эпоху через train_one_epoch(..., scaler=scaler),
    # затем evaluate на official_test_loader и заполните GPU_REPORT.
else:
    print(GPU_REPORT)
"""
                ),
                md(
                    """## 8. Итоговый отчёт

Заполните найденные ошибки и ограничения конкретно. Хорошее ограничение связывает свойство эксперимента с риском вывода: например, «обучались на 20% train-данных, поэтому результат нельзя использовать как benchmark архитектуры». Не называйте три эпохи доказательством сходимости и не сравнивайте CPU/GPU только по времени без указания batch size, precision и объёма данных."""
                ),
                code(
                    """FOUND_ERRORS = [
    # TODO: минимум семь коротких записей из code review.
]
EXPERIMENT_LIMITATIONS = [
    # TODO: минимум четыре ограничения обязательного и optional экспериментов.
]
FINAL_REPORT = {
    "device": str(REQUIRED_DEVICE),
    "train_examples": len(train_dataset),
    "validation_examples": len(validation_dataset),
    "epochs": len(history_frame),
    "first_train_loss": float(history_frame.iloc[0].train_loss),
    "last_train_loss": float(history_frame.iloc[-1].train_loss),
    "validation_accuracy": float(validation_2["accuracy"]),
    "required_seconds": float(required_seconds),
    "model_parameters": parameter_count,
    "checkpoint_sha256": checkpoint_sha256,
    "found_errors": FOUND_ERRORS,
    "limitations": EXPERIMENT_LIMITATIONS,
    "gpu": GPU_REPORT,
}
assert len(FOUND_ERRORS) >= 7
assert len(EXPERIMENT_LIMITATIONS) >= 4
assert FINAL_REPORT["last_train_loss"] < FINAL_REPORT["first_train_loss"]
assert FINAL_REPORT["validation_accuracy"] >= 0.40
FINAL_REPORT
""",
                    role="self_check",
                ),
                md(
                    """## Дальше по теме

1. [YDS Practical_DL — automatic differentiation and modules](https://github.com/yandexdataschool/Practical_DL/tree/fall25/week02_autodiff) — первоисточник учебной механики и упражнений на устройство PyTorch.
2. [YDS Practical_DL — convolutional networks](https://github.com/yandexdataschool/Practical_DL/tree/fall25/week03_convnets) — практический контекст для CNN и обучения на изображениях.
3. [PyTorch: Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) — официальный базовый пример CIFAR-10.
4. [PyTorch: CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) — контракт raw logits и class indices.
5. [PyTorch: Automatic Mixed Precision](https://pytorch.org/docs/stable/notes/amp_examples.html) — официальный AMP и gradient scaling.
6. [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) и [технический отчёт](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) — описание и происхождение датасета.
7. [«мыш»: PyTorch с нуля](https://mouseml.github.io/blog/posts/pytorch-tensors/) и [«мыш»: нейронные сети — теория и практика](https://mouseml.github.io/blog/2025/04/18/nn/) — дополнительное русское объяснение; материалы доступны по CC BY-NC-ND 4.0 и здесь не воспроизводятся.

Источники и лицензии проверены 28 августа 2026 года."""
                ),
            ],
        )
    }
