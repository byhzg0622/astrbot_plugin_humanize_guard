"""共享数据结构：违规项与检测上下文。

Violation 被三层共用 —— 规则层命中、AI 层裁决、改写提示词拼装，
所以它必须是一个谁都认识的普通对象。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Violation:
    """一条违规判定。"""

    id: str          # 检测项 id，例如 "end_period"
    name: str        # 人类可读名称，例如 "句末有句号"
    reason: str      # 具体依据，例如 "末尾字符是“。”"
    engine: str = "rule"   # rule | judge

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "reason": self.reason, "engine": self.engine}


@dataclass
class RuleContext:
    """检测时可见的会话上下文。

    - prev:        机器人上一条**已发出**的消息（改写前的草稿从不入史，
                   所以重合检测天然只和"上一句 + 改写后"比对，不会和自己撞车）
    - recent:      最近 K 条已发出消息，最新在前（index 0 == prev）
    - user_prev:   用户上一条消息（复述检测用）
    """

    prev: str = ""
    recent: list[str] = field(default_factory=list)
    user_prev: str = ""

    def safe(self) -> "RuleContext":
        """确保 recent 是干净的字符串列表。"""
        clean: list[str] = []
        for item in self.recent:
            if isinstance(item, str) and item:
                clean.append(item)
        return RuleContext(
            prev=self.prev if isinstance(self.prev, str) else "",
            recent=clean,
            user_prev=self.user_prev if isinstance(self.user_prev, str) else "",
        )
