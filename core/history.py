"""会话历史：上一句 + 最近 K 条 + 用户上一条。

只记录**最终真正发出**的版本 —— 改写前的草稿从不入史，
于是"重合检测"天然只在「上一条已发送消息 vs 改写后文本」之间进行，
不需要额外的删除逻辑（这正是你提的方案）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import as_int

_MAX_SESSIONS = 300


class SessionHistory:
    """每会话的历史环，支持落盘。"""

    def __init__(self, path: Path, keep_k: int = 5, persist: bool = True) -> None:
        self.path = Path(path)
        self.keep_k = max(1, as_int(keep_k, 5, low=1, high=50))
        self.persist = bool(persist)
        self._data: dict[str, dict[str, Any]] = {}

    # -- 存取 ---------------------------------------------------------------
    def load(self) -> None:
        if not self.persist:
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - 没有/坏了都当空历史
            self._data = {}
            return
        if isinstance(raw, dict):
            self._data = {
                str(key): value
                for key, value in raw.items()
                if isinstance(value, dict)
            }

    def save(self) -> None:
        if not self.persist:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001 - 落盘失败不影响聊天
            pass

    def configure(self, keep_k: int | None = None, persist: bool | None = None) -> None:
        if keep_k is not None:
            self.keep_k = max(1, as_int(keep_k, self.keep_k, low=1, high=50))
        if persist is not None:
            self.persist = bool(persist)

    # -- 读写 ---------------------------------------------------------------
    def _blank(self) -> dict[str, Any]:
        return {"prev": "", "recent": [], "user_prev": "", "ts": 0.0}

    def get(self, umo: str) -> dict[str, Any]:
        entry = self._data.get(str(umo))
        if not isinstance(entry, dict):
            return self._blank()
        recent = entry.get("recent")
        return {
            "prev": str(entry.get("prev") or ""),
            "recent": [str(x) for x in recent if isinstance(x, str)][: self.keep_k]
            if isinstance(recent, list) else [],
            "user_prev": str(entry.get("user_prev") or ""),
            "ts": float(entry.get("ts") or 0.0),
        }

    def context(self, umo: str):
        """给检测层的 RuleContext（避免 core 模块反向依赖调用方）。"""
        from .model import RuleContext

        entry = self.get(umo)
        return RuleContext(
            prev=entry["prev"],
            recent=list(entry["recent"]),
            user_prev=entry["user_prev"],
        ).safe()

    def record(self, umo: str, bare: str, user_prev: str | None = None) -> None:
        """记下一条真正发出的消息（裸文本，标签不入史）。"""
        key = str(umo)
        entry = self.get(key)
        bare = str(bare or "").strip()
        recent = entry["recent"]
        if bare:
            recent = [bare] + [x for x in recent if x != bare]
            recent = recent[: self.keep_k]
        entry["recent"] = recent
        entry["prev"] = recent[0] if recent else ""
        if user_prev is not None:
            entry["user_prev"] = str(user_prev or "").strip()
        entry["ts"] = time.time()
        self._data[key] = entry
        self._evict()
        self.save()

    def _evict(self) -> None:
        if len(self._data) <= _MAX_SESSIONS:
            return
        ordered = sorted(self._data.items(), key=lambda kv: kv[1].get("ts") or 0.0)
        for key, _ in ordered[: len(self._data) - _MAX_SESSIONS]:
            self._data.pop(key, None)

    def stats(self) -> dict[str, Any]:
        return {
            "sessions": len(self._data),
            "keep_k": self.keep_k,
            "persist": self.persist,
        }

    def clear(self, umo: str | None = None) -> int:
        if umo:
            removed = 1 if self._data.pop(str(umo), None) else 0
        else:
            removed = len(self._data)
            self._data.clear()
        self.save()
        return removed
