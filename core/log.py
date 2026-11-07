"""打回日志：环形缓冲 + 落盘，面板"日志"页与 /qc 指令共用。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_MAX_ENTRIES = 300


def _clip(text: Any, limit: int = 400) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


class CheckLog:
    def __init__(self, path: Path, max_entries: int = _MAX_ENTRIES) -> None:
        self.path = Path(path)
        self.max_entries = max(10, int(max_entries))
        self._entries: list[dict[str, Any]] = []

    def load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            self._entries = []
            return
        if isinstance(raw, list):
            self._entries = [e for e in raw if isinstance(e, dict)][: self.max_entries]

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._entries, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass

    def add(
        self,
        *,
        umo: str,
        action: str,
        violations: list[dict[str, Any]] | None = None,
        original: str = "",
        final: str = "",
        rounds: int = 0,
        error: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """写一条日志。action: rewrite | dry | give_up | pass | error。"""
        entry: dict[str, Any] = {
            "ts": time.time(),
            "umo": str(umo or ""),
            "action": action,
            "rounds": int(rounds),
            "violations": [
                {
                    "id": str(v.get("id") or ""),
                    "name": str(v.get("name") or ""),
                    "reason": _clip(v.get("reason"), 200),
                }
                for v in (violations or [])
                if isinstance(v, dict)
            ],
            "original": _clip(original),
            "final": _clip(final),
        }
        if error:
            entry["error"] = _clip(error, 300)
        if extra:
            entry.update(extra)
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]
        self.save()
        return entry

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), self.max_entries))
        return list(reversed(self._entries[-limit:]))

    def clear(self) -> int:
        removed = len(self._entries)
        self._entries.clear()
        self.save()
        return removed

    def stats(self) -> dict[str, Any]:
        rejections = sum(
            1 for e in self._entries
            if e.get("action") in {"rewrite", "give_up", "dry", "local"}
        )
        return {"total": len(self._entries), "rejections": rejections}
