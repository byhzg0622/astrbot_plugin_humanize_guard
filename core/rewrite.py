"""打回改写：按命中的检测项把"对应提示词"注入改写请求，督促模型重写。

只输出正文本身 —— 解释、引号、markdown 围栏全部在出口处清掉；
改写后的文本还要回流水线复检（只和上一句比，不看改写前的草稿）。
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from .model import RuleContext, Violation

_FENCE_RE = re.compile(r"^\s*\x60{3,}[a-zA-Z]*\s*|\s*\x60{3,}\s*$")
_LEAD_RE = re.compile(
    r"^\s*(?:改写后(?:的文本)?|修改后(?:的文本)?|以下是改写结果|改写结果)\s*[：:]\s*"
)


def build_rewrite(
    text: str,
    violations: list[Violation],
    hints: dict[str, str],
    ctx: RuleContext,
) -> tuple[str, str]:
    """拼装改写请求。返回 (system_prompt, prompt)。

    hints = 违规项对应的改写提示词（ItemStore.rewrite_hints），
    只带**本轮命中**的项 —— 提示词随命中的勾选项变化。
    """
    ctx = ctx.safe()
    problems: list[str] = []
    requirements: list[str] = []
    seen: set[str] = set()
    for index, violation in enumerate(violations, 1):
        key = violation.id
        if key in seen:
            continue
        seen.add(key)
        problems.append(f"{index}. [{violation.id}] {violation.name}：{violation.reason}")
        hint = str(hints.get(key) or "").strip()
        if hint:
            requirements.append(f"{index}. {hint}")

    system = (
        "你是一个文本改写器。把用户给出的聊天消息改写到合格。\n"
        "硬性规则：\n"
        "- 只修复下面指出的问题，不改变内容含义、信息量与说话人的人设语气；\n"
        "- 不添加新信息、不解释、不道歉、不加任何前后缀说明；\n"
        "- 保持口语感，像真人打字；\n"
        "- 只输出改写后的正文本身：不要引号、不要代码围栏、不要 markdown。"
    )
    parts = [
        "这条消息被质检打回，存在以下问题：",
        "\n".join(problems) if problems else "（未给出具体问题，请整体润色得更像真人）",
    ]
    if requirements:
        parts.append("改写时必须逐条落实：\n" + "\n".join(requirements))
    refs: list[str] = []
    if ctx.prev:
        refs.append(f"- 你上一条发出的消息（不要复读它的用词和开头结尾）：{ctx.prev[:160]}")
    if ctx.user_prev:
        refs.append(f"- 用户上一条消息（供理解语境，不要复述）：{ctx.user_prev[:160]}")
    if refs:
        parts.append("参考信息：\n" + "\n".join(refs))
    parts.append(f'待改写消息：\n"""\n{text}\n"""')
    return system, "\n\n".join(parts)


def sanitize(output: str) -> str:
    """清洗模型的改写输出：围栏、解释前缀、包裹引号。"""
    text = str(output or "").strip()
    if not text:
        return ""
    text = _FENCE_RE.sub("", text).strip()
    text = _LEAD_RE.sub("", text).strip()
    # 成对包裹引号整体剥掉（只剥最外层）
    for left, right in (("“", "”"), ("‘", "’"), ("「", "」"), ("『", "』"), ('"', '"'), ("'", "'")):
        if len(text) >= 2 and text.startswith(left) and text.endswith(right):
            text = text[1:-1].strip()
    # 退回第一行（模型嘴碎时只取正文）
    if "\n" in text and len(text.splitlines()) > 1:
        lines = [line for line in text.splitlines() if line.strip()]
        if lines and all(line.strip().startswith(("-", "*", "1.", "2.", "3.")) for line in lines):
            text = " ".join(line.strip().lstrip("-*0123456789.、 ") for line in lines)
    return text.strip()


async def run_rewrite(
    provider: Any,
    text: str,
    violations: list[Violation],
    hints: dict[str, str],
    ctx: RuleContext,
    timeout: float = 12.0,
) -> tuple[str, str]:
    """执行一次改写。返回 (改写结果, 错误信息)；出错时结果为空串。"""
    if provider is None:
        return "", "没有可用的模型 Provider"
    system, prompt = build_rewrite(text, violations, hints, ctx)
    try:
        response = await asyncio.wait_for(
            provider.text_chat(prompt=prompt, system_prompt=system),
            timeout=max(1.0, float(timeout)),
        )
    except Exception as exc:  # noqa: BLE001 - 改写失败必须回退原稿
        return "", f"{type(exc).__name__}: {exc}"
    raw = getattr(response, "completion_text", "") or ""
    if not isinstance(raw, str):
        raw = str(raw)
    return sanitize(raw), ""
