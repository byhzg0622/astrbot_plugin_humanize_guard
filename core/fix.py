"""本地公式修复层（零模型调用）。

用户诉求：命中的检测项先在本层"套公式"改好 —— 能用确定性规则修掉的，
一次模型调用都不花；本层修不掉的（与上一句重合、复述用户、整体 AI 味、
自定义项…）才落到 AI 改写层。"公式识别套用，不满足再传 AI"。

安全阀（核心设计）：每个候选补丁都要过一遍规则层复检 ——
① 违规**数量必须严格变少**；② 不允许**新增任何违规项**。
两条有一条不满足，补丁直接丢弃、原文不动，绝不越修越糟。

本层只做"改完一定更好"的确定性修补：
删句末句号、合并重复标点、清残留字符、拆 Markdown 排版、补逗号断句、
去掉重复的语气词收尾/开头、删本条内部重复句、删清单里的套话与书面语……
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import as_int
from .items import CheckItem
from .model import RuleContext, Violation
from .rules import _NO_PUNCT_RE, find_phrases, normalize, run_rules
from .tags import strip_message_markup

# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------

#: 与规则层同一套分句口径（rules.split_clauses 用的分隔符）
_CLAUSE_SPLIT = re.compile(r"[，,、;；:：!！?？…。.~～\n\r]+")

_ZW_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
#: 句尾半截标签：【好感度+3 / [emotion 之类没闭合的残片
_HALF_TAG_RE = re.compile(
    r"[【\[<（(]\s*(?:好感度|好感|亲密度|熟悉度|信赖度|emotion|mood|情绪|心情)"
    r"[^\n】\]>）)]{0,24}$"
)

#: 明确"本地公式修不掉、必须交给 AI"的检测项
AI_ONLY_IDS: set[str] = {"overlap_prev", "echo_user", "too_long", "ai_flavor"}

#: 本地能修的检测项（面板/文档里展示用）
FIXABLE_IDS: list[str] = [
    "end_period", "punct_oddity", "residue", "structure", "mid_punct",
    "overlap_self", "same_tail", "same_head", "filler_dup",
    "boilerplate", "written_style", "identity_leak",
]


def _tidy(text: str) -> str:
    """删掉东西之后收拾标点：不留连续标点、不以标点开头结尾。"""
    out = re.sub(r"[ \t]{2,}", " ", text)
    out = re.sub(r"([，,、;；:：])\s*[，,、;；:：]+", r"\1", out)
    out = re.sub(r"^[，,、;；:：！!？?。.\s]+", "", out)
    out = re.sub(r"[，,、;；:：]\s*$", "", out)
    return out.strip()


def _clause_spans(text: str) -> list[tuple[int, int]]:
    """分句的 (起点, 终点) —— 与规则层 split_clauses 完全一致的分隔符。"""
    raw = text or ""
    spans: list[tuple[int, int]] = []
    pos = 0
    for match in _CLAUSE_SPLIT.finditer(raw):
        if match.start() > pos:
            spans.append((pos, match.start()))
        pos = match.end()
    if pos < len(raw):
        spans.append((pos, len(raw)))
    return spans


def _clauses_with_delims(text: str) -> list[tuple[str, str]]:
    """(正文, 后面的分隔符) 序列，便于整句删除后重新拼接。"""
    raw = text or ""
    out: list[tuple[str, str]] = []
    pos = 0
    for match in _CLAUSE_SPLIT.finditer(raw):
        out.append((raw[pos:match.start()], match.group(0)))
        pos = match.end()
    out.append((raw[pos:], ""))
    return out


# ---------------------------------------------------------------------------
# 各检测项的"公式"
# ---------------------------------------------------------------------------
# 统一签名：(原文, 规则给的依据, 该检测项参数, 上下文) -> 修好的文本 / None（修不了）

_TRAILING_PERIOD_RE = re.compile(r"[。.]+\s*$")

_DOTS_AS_ELLIPSIS = "…"

_HEAD_FILLERS = ("嗯", "唔", "诶", "哎", "那个", "这个", "哈哈", "嘿嘿", "哎呀", "emmm", "emm")

# ---------------------------------------------------------------------------
# 断句候选：逗号只落在"完整词块边界"，找不到合格位置就交给 AI（宁缺毋滥）
# ---------------------------------------------------------------------------
#: 这些**词块**前可以断：接续/转折/推论连接词，以及主语开头（口语里天然的分句点）
_BREAK_BEFORE_WORDS = (
    # 接续 / 转折 / 推论（在词块前断，语义最稳）
    "然后", "但是", "不过", "而且", "所以", "因为", "如果", "要是", "要不",
    "还是", "可是", "毕竟", "可能", "也许", "大概", "顺便", "回头", "对了",
    "另外", "话说", "等会", "一会儿", "算了", "反正", "其实",
    # 多字主语开头：换主语必是分句点
    "咱们", "我们", "你们", "他们", "咱俩", "大家", "主人", "兄弟", "宝贝",
)
#: 多字主语（换主语必是分句点，词块边界明确）
_SUBJECT_WORDS = ("咱们", "我们", "你们", "他们", "咱俩", "大家", "主人", "兄弟", "宝贝")
#: 单字主语/指示词（我/你/他/她/它/咱/这/那/别）：前一个字符必须也是完整词块
#:（不在词中间）才允许断，避免"我也是"被切成"，也是"
_SUBJECT_START = set("我你他她它咱这那别")
#: 这些字符后面断最自然（语气词收尾，语气词自身就是完整词块）
_BREAK_AFTER = "啊呀呢吧嘛哦诶嗯啦哈"
#: 前面是这些字时不能断（前面这个词还没说完）
_NO_SPLIT_AFTER = set("的地得很不没在和与或把被给太更最都也还又就才只挺好几个这那么什怎为因但而且若每某")
#: 后面是这些字时不能断（后面这个词还没开始）
_NO_SPLIT_BEFORE = set("的了着过吗吧呢啊呀得地们")


def fix_end_period(text: str, reason: str, params: dict, ctx: RuleContext) -> str | None:
    """句末句号：删掉结尾的“。”/“.”（连同“。。。。”一类的连发）。"""
    tail = (text or "").rstrip()
    if not tail or tail[-1] not in "。.":
        return None
    new = _TRAILING_PERIOD_RE.sub("", tail).rstrip()
    return new or None


def fix_punct_oddity(text: str, reason: str, params: dict, ctx: RuleContext) -> str | None:
    """连续重复标点/表情：合并成一个（句点连发收成省略号）。"""
    exclam_min = as_int(params.get("exclam_min"), 3, 2, 20)
    quest_min = as_int(params.get("quest_min"), 3, 2, 20)
    dots_min = as_int(params.get("dots_min"), 4, 3, 20)
    emoji_min = as_int(params.get("emoji_min"), 3, 3, 20)
    new = re.sub(r"([!！]){%d,}" % exclam_min, r"\1", text or "")
    new = re.sub(r"([?？]){%d,}" % quest_min, r"\1", new)
    new = re.sub(r"([。.]){%d,}" % dots_min, _DOTS_AS_ELLIPSIS, new)
    new = re.sub(
        r"([\u2600-\u27BF\u2B00-\u2BFF\U0001F300-\U0001FAFF])\1{%d,}" % (emoji_min - 1),
        r"\1", new,
    )
    return new if new != text else None


def fix_residue(text: str, reason: str, params: dict, ctx: RuleContext) -> str | None:
    """残留标记：协议占位、全角空格、零宽/控制字符、句尾半截标签。"""
    new = strip_message_markup(text or "")
    new = new.replace("\u3000", " ")
    new = _ZW_RE.sub("", new)
    new = _CTRL_RE.sub("", new)
    new = _HALF_TAG_RE.sub("", new)
    return new if new != text else None


def fix_structure(text: str, reason: str, params: dict, ctx: RuleContext) -> str | None:
    """结构化排版：拆掉 Markdown/列表/编号，打散“首先其次最后”三段式。"""
    new = text or ""
    new = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", new)
    new = re.sub(r"(?m)^\s*#{1,6}\s*", "", new)
    new = re.sub(r"(?m)^\s*[-*•]\s+", "", new)
    new = re.sub(r"(?m)^\s*\d+[.、)]\s+", "", new)
    new = new.replace(chr(96), "")
    new = re.sub(r"\[([^\]\n]+)\]\([^)\n]+\)", r"\1", new)
    new = new.replace("其次", "然后")
    new = re.sub(r"[①②③④⑤⑥⑦⑧⑨⑩]", "", new)
    new = re.sub(r"\n{3,}", "\n\n", new)
    return new if new != text else None


def fix_mid_punct(text: str, reason: str, params: dict, ctx: RuleContext) -> str | None:
    """句中缺标点：在过长的分句里挑最自然的位置补逗号（一次补到够为止）。"""
    max_run = as_int(params.get("max_run"), 15, 3, 300)
    raw = text or ""
    new = raw
    for _ in range(4):                              # 超长句最多补 4 个逗号
        pos = _pick_break(new, max_run)
        if pos is None:
            break
        if new[pos - 1] in " \t":                  # 半角空格：直接换成逗号更干净
            new = new[: pos - 1] + "，" + new[pos:].lstrip(" \t")
        else:
            new = new[:pos] + "，" + new[pos:]
    return new if new != raw else None


def _pick_break(raw: str, max_run: int) -> int | None:
    """在第一个“太长且无标点”的分句里，挑一个最自然的断句位置。

    断点必须同时满足两个条件（防止补出“越补越怪”的逗号）：
    ① 打分器认可（词块边界，详见 _break_score）；
    ② 断完之后**两边都**不超阈值 —— 只把长句劈成两个都不算长的短句，
       不做"一半还是 20 字"的无效补丁。
    """
    if not raw:
        return None
    for start, end in _clause_spans(raw):
        piece = raw[start:end]
        core = piece.strip()
        if len(core) <= max_run:
            continue
        if not re.search(r"[\u4e00-\u9fff]", core):
            continue
        if _NO_PUNCT_RE.fullmatch(core):
            continue
        left = start + (len(piece) - len(piece.lstrip()))
        right = start + len(piece.rstrip())
        if right - left <= max_run:
            continue
        middle = (left + right) / 2
        best: tuple[tuple[int, float], int] | None = None
        for pos in range(left + 4, right - 3):
            score = _break_score(raw, pos)
            if score is None:
                continue
            # 断完两边都不能还是超长（否则这一刀等于没切到位）
            if pos - left > max_run and right - pos > max_run:
                continue
            key = (score, abs(pos - middle))
            if best is None or key < best[0]:
                best = (key, pos)
        if best is not None:
            return best[1]
    return None


def _break_score(text: str, pos: int) -> int | None:
    """断句点打分：越小越自然；**None = 这个位置不许断**。

    原则（用户反馈"逗号规则鸡肋"后重写）：
    - 逗号只落在完整词块边界：半角空格、连接词词块前、多字主语前、
      单字主语前（要求前一个字符也不在词中间）、语气词收尾后；
    - **绝不**在词中间硬断——"也不是，什么大事"这种一刀切进去的断法一律禁止；
    - 找不到任何合格位置就返回 None，这条违规原样交给 AI 改写层（宁缺毋滥）。
    """
    if pos <= 0 or pos >= len(text):
        return None
    prev, nxt = text[pos - 1], text[pos]
    if prev.isspace():
        return 0                                    # 空格：作者本来就想停顿
    if prev in _NO_SPLIT_AFTER or nxt in _NO_SPLIT_BEFORE:
        return None                                 # 前词没收尾 / 后词没开始
    if prev in "，,、;；:：!！?？…。" or nxt in "，,、;；:：!！?？…。.":
        return None                                 # 已有标点，轮不到补逗号
    # ① 连接词词块前（语义最稳）
    for word in _BREAK_BEFORE_WORDS:
        if text.startswith(word, pos):
            return 1
    # ② 多字主语前（词块边界明确）
    for word in _SUBJECT_WORDS:
        if text.startswith(word, pos):
            return 2
    # ③ 单字主语前（"我/你/这"）：前一个字符必须已构成完整词块，避免切断前面的词
    if nxt in _SUBJECT_START and pos >= 2:
        before = text[pos - 2]
        if before not in _NO_SPLIT_AFTER and before not in _BREAK_AFTER:
            return 3
    # ④ 语气词收尾后
    if prev in _BREAK_AFTER:
        return 4
    return None                                     # 没有合格断点：不硬断


def fix_same_tail(text: str, reason: str, params: dict, ctx: RuleContext) -> str | None:
    """相同结尾：句尾是重复的语气词 → 去掉它。"""
    particles = str(params.get("particle_chars") or "呢呀哦啦嘛吧啊诶嘿哈~～")
    tail = (text or "").rstrip()
    if not tail or tail[-1] not in particles:
        return None
    new = tail[:-1].rstrip()
    return new or None


def fix_same_head(text: str, reason: str, params: dict, ctx: RuleContext) -> str | None:
    """相同开头：开头是套路的语气词/口头禅 → 去掉它。"""
    raw = text or ""
    for filler in _HEAD_FILLERS:
        if raw.startswith(filler) and len(raw) > len(filler) + 1:
            new = raw[len(filler):].lstrip("，,、 ")
            if new and new != raw:
                return new
    return None


_TOKEN_REASON_RE = re.compile(r"词组[“\"]([^”\"]{2,6})[”\"]")


def fix_filler_dup(text: str, reason: str, params: dict, ctx: RuleContext) -> str | None:
    """口头禅超频：同一个词组只留第一次出现。"""
    match = _TOKEN_REASON_RE.search(reason or "")
    token = match.group(1) if match else ""
    raw = text or ""
    if not token:
        return None
    first = raw.find(token)
    if first < 0:
        return None
    head = raw[: first + len(token)]
    tail = raw[first + len(token):].replace(token, "")
    new = _tidy(head + tail)
    return new if new and new != raw else None


def fix_overlap_self(text: str, reason: str, params: dict, ctx: RuleContext) -> str | None:
    """本条内部重复：同义的两句只留前一句，删掉后面那句。"""
    min_common = as_int(params.get("min_common"), 6, 3, 100)
    segments = _clauses_with_delims(text or "")
    norms = [normalize(piece) for piece, _ in segments]
    for i in range(len(segments)):
        if len(norms[i]) < min_common:
            continue
        for j in range(i + 1, len(segments)):
            a, b = norms[i], norms[j]
            if len(b) < min_common:
                continue
            if a == b or a in b or b in a:
                kept = [seg for index, seg in enumerate(segments) if index != j]
                new = _tidy("".join(piece + delim for piece, delim in kept))
                return new if new and new != text else None
    return None


def _phrase_fixer(text: str, reason: str, params: dict, ctx: RuleContext) -> str | None:
    """清单类（套话/书面语/身份泄漏）：把命中的词删掉，再收拾标点。"""
    phrases = params.get("phrases")
    if isinstance(phrases, str):
        phrases = [line.strip() for line in phrases.splitlines() if line.strip()]
    if not isinstance(phrases, (list, tuple, set)) or not phrases:
        return None
    hits = find_phrases(text or "", [str(p) for p in phrases], limit=len(phrases) + 1)
    if not hits:
        return None
    new = text or ""
    for hit in hits:
        new = re.sub(re.escape(hit), "", new, flags=re.IGNORECASE)
    new = _tidy(new)
    return new if new and new != text else None


#: 检测项 id → 本地修补公式
_FIXERS: dict[str, Callable[[str, str, dict, RuleContext], str | None]] = {
    "end_period": fix_end_period,
    "punct_oddity": fix_punct_oddity,
    "residue": fix_residue,
    "structure": fix_structure,
    "mid_punct": fix_mid_punct,
    "same_tail": fix_same_tail,
    "same_head": fix_same_head,
    "filler_dup": fix_filler_dup,
    "overlap_self": fix_overlap_self,
    "boilerplate": _phrase_fixer,
    "written_style": _phrase_fixer,
    "identity_leak": _phrase_fixer,
}


@dataclass
class RepairResult:
    """本地修补结果。"""

    text: str                                        # 修补后的文本（没修补就是原文）
    fixed: list[str] = field(default_factory=list)    # 被本地修好的检测项 id
    remaining: list[Violation] = field(default_factory=list)   # 复检后仍然存在的违规
    steps: list[str] = field(default_factory=list)    # 实际落地的补丁（id 顺序）

    @property
    def changed(self) -> bool:
        return bool(self.steps)


def can_fix(item_id: str) -> bool:
    """这一项本地能不能修。"""
    return str(item_id) in _FIXERS


def local_repair(
    text: str,
    violations: list[Violation],
    rule_items: list[CheckItem],
    ctx: RuleContext,
    max_steps: int = 8,
) -> RepairResult:
    """本地公式修补。

    text       —— 裸文本（标签已剥离）
    violations —— 规则层（或 AI 层）刚判出的违规
    rule_items —— 启用的规则项（既用来修补，也用来复检）
    """
    current = str(text or "")
    start_ids = list(dict.fromkeys(v.id for v in violations))
    if not current or not start_ids or not rule_items:
        return RepairResult(current, [], list(violations), [])

    params_by_id = {item.id: dict(item.params or {}) for item in rule_items}
    steps: list[str] = []
    for _ in range(max(1, int(max_steps))):
        current_violations = run_rules(current, rule_items, ctx)
        if not current_violations:
            break
        before_ids = [v.id for v in current_violations]
        accepted = False
        tried: set[str] = set()
        for violation in current_violations:
            if violation.id in tried:
                continue
            tried.add(violation.id)
            fixer = _FIXERS.get(violation.id)
            if fixer is None:
                continue
            try:
                candidate = fixer(current, violation.reason, params_by_id.get(violation.id, {}), ctx)
            except Exception:  # noqa: BLE001 - 单条公式炸了绝不能影响质检
                continue
            if not candidate or candidate == current:
                continue
            after = run_rules(candidate, rule_items, ctx)
            after_ids = [v.id for v in after]
            if set(after_ids) - set(before_ids):        # 修出了新问题 → 不要
                continue
            if len(after) >= len(current_violations):   # 没变少 → 不要
                continue
            current = candidate
            steps.append(violation.id)
            accepted = True
            break
        if not accepted:
            break

    remaining = run_rules(current, rule_items, ctx)
    remaining_ids = {v.id for v in remaining}
    fixed = [item_id for item_id in start_ids if item_id not in remaining_ids]
    return RepairResult(current, fixed, remaining, steps)
