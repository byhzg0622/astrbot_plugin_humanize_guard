"""AI 检测层：把**勾选中的检测项注入词**动态拼成提示词，交给模型裁决。

注入词随勾选变化 —— 调用方只传启用且注入词非空的项，
本模块按勾选顺序拼装 system_prompt，输出契约里允许的 id 集合同步收窄。
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .items import CheckItem
from .model import RuleContext, Violation

_FENCE_RE = re.compile(r"^\s*`{3,}(?:json)?|\x60{3,}\s*$", re.IGNORECASE)


@dataclass
class JudgeResult:
    violations: list[Violation] = field(default_factory=list)
    raw: str = ""
    error: str = ""
    system: str = ""
    prompt: str = ""

    @property
    def passed(self) -> bool:
        return not self.violations and not self.error


def _clip(text: str, limit: int = 120) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_judge_prompt(
    items: list[CheckItem],
    text: str,
    ctx: RuleContext,
) -> dict[str, Any]:
    """拼装检测模型的 system_prompt 与 user prompt。

    items = 勾选中的、注入词非空的检测项（由 ItemStore.enabled_judge_items 给出）。
    勾选不同 → 注入词不同 → 本函数的返回值不同。
    """
    ctx = ctx.safe()
    blocks: list[str] = []
    ids: list[str] = []
    for index, item in enumerate(items, 1):
        ids.append(item.id)
        blocks.append(f"【{index}】{item.name}（id: {item.id}）\n{item.prompt.strip()}")

    refs: list[str] = []
    if ctx.prev:
        refs.append(f"- 你上一条发出的消息：{_clip(ctx.prev, 200)}")
    older = [r for r in ctx.recent[1:] if r][:4]
    if older:
        refs.append("- 最近发出的其他消息（新→旧）：" + " / ".join(_clip(r, 60) for r in older))
    if ctx.user_prev:
        refs.append(f"- 用户上一条消息：{_clip(ctx.user_prev, 200)}")
    if not refs:
        refs.append("-（这个会话还没有可参考的历史）")

    system = (
        "你是「消息质检员」，判断一条聊天消息是否像真人随手打字发出的。\n"
        "本次启用的检查项如下，**只有命中列出的检查项才算违规**，其余一概不究：\n\n"
        + "\n\n".join(blocks)
        + "\n\n参考信息（部分检查项需要）：\n"
        + "\n".join(refs)
        + "\n\n只输出一个 JSON 对象，不要输出任何其他文字（不要代码围栏）：\n"
        '{"pass": true或false, "violations": [{"id": "检查项id", "reason": "一句话依据"}]}\n'
        f"\"id\" 只能取自：{json.dumps(ids, ensure_ascii=False)}\n"
        "硬性规则：没命中任何检查项就输出 {\"pass\": true, \"violations\": []}；"
        "拿不准的项宁可放过，不许牵强附会。"
    )
    prompt = f'待检消息：\n"""\n{text}\n"""'
    return {"system": system, "prompt": prompt, "ids": ids}


def parse_verdict(raw: str, allowed_ids: list[str]) -> tuple[list[Violation], str]:
    """把模型输出解析成违规列表。返回 (violations, error)。"""
    text = str(raw or "").strip()
    if not text:
        return [], "模型返回空输出"
    text = _FENCE_RE.sub("", text).strip()
    data: Any = None
    try:
        data = json.loads(text)
    except Exception:  # noqa: BLE001 - 有前后缀杂质时再抠
        pass
    if not isinstance(data, dict):
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                data = json.loads(text[start:end + 1])
            except Exception:  # noqa: BLE001
                data = None
    if not isinstance(data, dict):
        return [], f"无法解析模型输出为 JSON：{_clip(text, 160)}"

    allowed = {str(x) for x in allowed_ids}
    out: list[Violation] = []
    seen: set[str] = set()
    raw_violations = data.get("violations")
    if isinstance(raw_violations, list):
        for entry in raw_violations:
            if not isinstance(entry, dict):
                continue
            item_id = str(entry.get("id") or "").strip()
            if not item_id or item_id not in allowed or item_id in seen:
                continue
            seen.add(item_id)
            out.append(Violation(
                id=item_id,
                name=item_id,   # 名称由调用方按注册表补全
                reason=str(entry.get("reason") or "模型未给出依据"),
                engine="judge",
            ))
    if not out and data.get("pass") is False:
        out.append(Violation(
            id="ai_reject",
            name="AI 判定不通过",
            reason=str(data.get("reason") or "模型判定不合格但未给出具体检查项"),
            engine="judge",
        ))
    return out, ""


async def run_judge(
    provider: Any,
    items: list[CheckItem],
    text: str,
    ctx: RuleContext,
    timeout: float = 12.0,
) -> JudgeResult:
    """调用检测模型。任何异常都转成 error（fail-open，由调用方决定放行与否）。"""
    if provider is None:
        return JudgeResult(error="没有可用的模型 Provider")
    if not items:
        return JudgeResult()          # 没有启用的检测项 = AI 层直接放行
    built = build_judge_prompt(items, text, ctx)
    result = JudgeResult(system=built["system"], prompt=built["prompt"])
    try:
        response = await asyncio.wait_for(
            provider.text_chat(prompt=built["prompt"], system_prompt=built["system"]),
            timeout=max(1.0, float(timeout)),
        )
    except Exception as exc:  # noqa: BLE001 - 检测失败绝不能阻断回复
        result.error = f"{type(exc).__name__}: {exc}"
        return result
    raw = getattr(response, "completion_text", "") or ""
    if not isinstance(raw, str):
        raw = str(raw)
    result.raw = raw.strip()
    violations, error = parse_verdict(result.raw, built["ids"])
    result.violations = violations
    result.error = error
    return result
