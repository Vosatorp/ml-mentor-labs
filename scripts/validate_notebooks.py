from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ROLES = {"setup", "exercise", "self_check"}
FORBIDDEN_TEXT = (
    "private interview",
    "recruiter contact",
    "salary fork",
    "reference solution",
    "service_role",
    "supabase_service",
)


def validate(path: Path, *, execute_setup: bool) -> list[str]:
    errors: list[str] = []
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("nbformat") != 4:
        errors.append("nbformat must be 4")

    roles: set[str] = set()
    has_exercise_marker = False
    namespace: dict[str, object] = {"__name__": "__ml_mentor_lab_setup__"}

    for index, cell in enumerate(data.get("cells", [])):
        source = "".join(cell.get("source", []))
        lowered = source.lower()
        if any(marker in lowered for marker in FORBIDDEN_TEXT):
            errors.append(f"cell {index}: possible private or solution material")
        if cell.get("cell_type") != "code":
            continue

        if cell.get("execution_count") is not None or cell.get("outputs"):
            errors.append(f"cell {index}: committed execution output")
        try:
            compiled = compile(source, f"{path.name}:cell-{index}", "exec")
        except SyntaxError as exc:
            errors.append(f"cell {index}: syntax error: {exc.msg}")
            continue

        role = cell.get("metadata", {}).get("ml_mentor", {}).get("role")
        if role not in REQUIRED_ROLES:
            errors.append(f"cell {index}: missing or unknown ml_mentor.role")
            continue
        roles.add(role)
        if role == "exercise" and ("TODO" in source or "NotImplementedError" in source or "BROKEN:" in source):
            has_exercise_marker = True
        if execute_setup and role == "setup":
            try:
                exec(compiled, namespace)
            except Exception as exc:  # noqa: BLE001 - report the exact notebook fixture failure
                errors.append(f"cell {index}: setup failed: {type(exc).__name__}: {exc}")

    missing_roles = REQUIRED_ROLES - roles
    if missing_roles:
        errors.append(f"missing roles: {', '.join(sorted(missing_roles))}")
    if not has_exercise_marker:
        errors.append("exercise does not contain an explicit starter marker")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-setup", action="store_true")
    args = parser.parse_args()

    notebooks = sorted(ROOT.glob("*.ipynb"))
    if len(notebooks) != 6:
        raise SystemExit(f"expected 6 notebooks, found {len(notebooks)}")

    failures: list[str] = []
    for path in notebooks:
        for error in validate(path, execute_setup=args.execute_setup):
            failures.append(f"{path.name}: {error}")

    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Validated {len(notebooks)} starter notebooks")


if __name__ == "__main__":
    main()
