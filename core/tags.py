"""标签剥离与还原。

好感度 / 情绪等重复性标签：
- 检测时剥离 —— 不参与句末标点、重合、相同开头结尾的任何比对；
- 改写完成后原样加回 —— 标签是机器人体系的一部分，只在质检视野里隐身。
（按你的决策：只在检测时忽略，修改后还要加上去。）
"""

from __future__ import annotations

import re

#: 内置标签模式（追加在用户配置的额外模式之前，先到先得）
DEFAULT_TAG_PATTERNS: list[str] = [
    # 数值型标签：【好感度+5】 / [好感度:80] / (好感度 +3.5) / <好感度: +2>
    r"[【\[（(<]\s*[^\]】）>\n]{0,24}?(?:好感度|好感|亲密度|熟悉度|信赖度|亲密)"
    r"\s*[：:.+\-−]?\s*[+\-−]?\d+(?:\.\d+)?\s*%?\s*[】\]）)>]",
    # 纯名称型标签：【好感度】 / [亲密度]
    r"[【\[（(<]\s*(?:好感度|好感|亲密度|熟悉度|信赖度)\s*[】\]）)>]",
    # 情绪标签（与 ai_mind 的 EMOTION_TAG_RE 同构）：<emotion>害羞</emotion> / [mood]开心[/mood] / 【情绪：开心】
    r"[<\[【]\s*(?:emotion|mood|情绪|心情)\s*[>\]】][\s\S]{0,80}?"
    r"[<\[【]\s*/?\s*(?:emotion|mood|情绪|心情)\s*[>\]】]",
]

#: 协议占位（引用消息等）：进提示词会被当成"他说的话"，质检前后都要清掉
QUOTE_RE = re.compile(
    r"<quoted_message\b[^>]*>[\s\S]*?</quoted_message>|<quoted_message\b[^>]*/>",
    re.IGNORECASE,
)


def _compile(patterns: list[str]) -> list[re.Pattern]:
    out: list[re.Pattern] = []
    for raw in patterns:
        text = str(raw or "").strip()
        if not text:
            continue
        try:
            out.append(re.compile(text))
        except re.error:
            continue  # 用户写坏的正则跳过，绝不让质检崩掉
    return out


def strip_message_markup(text: str) -> str:
    """去掉协议占位（引用消息），保留正文。"""
    if not text:
        return ""
    return QUOTE_RE.sub("", text)


def strip_tags(text: str, extra_patterns: list[str] | None = None) -> tuple[str, list[str]]:
    """剥离标签，返回 (裸文本, 按原顺序找到的标签列表)。

    标签列表留着改写完成后加回；裸文本用于一切检测与改写。
    """
    if not text:
        return "", []
    patterns = list(DEFAULT_TAG_PATTERNS)
    patterns.extend(extra_patterns or [])
    found: list[str] = []
    current = text
    for pattern in _compile(patterns):
        pieces: list[str] = []
        last = 0
        for match in pattern.finditer(current):
            # 只在还没被剥掉的区域上收集，避免两个模式重复吃同一段
            if match.start() < last:
                continue
            pieces.append(current[last:match.start()])
            found.append(match.group(0))
            last = match.end()
        pieces.append(current[last:])
        current = "".join(pieces)
    # 连续空白整理一下（标签夹在中间会留下双空格），但不改结构
    current = re.sub(r"[ \t]{2,}", " ", current)
    current = re.sub(r"\n{3,}", "\n\n", current)
    return current.strip() if "\n" not in current else current.strip("\n"), found


def rejoin_tags(bare: str, tags: list[str]) -> str:
    """改写完成后把剥离的标签按原顺序加回（按你的决策：加回文本末尾）。"""
    if not tags:
        return bare
    tail = "".join(tags)
    if not bare:
        return tail
    return bare + tail
