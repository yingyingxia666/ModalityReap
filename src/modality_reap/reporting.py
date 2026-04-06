from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)



def summarize_warnings(warnings: list[str]) -> dict[str, Any]:
    return {
        "count": len(warnings),
        "messages": warnings,
    }
