"""规则层：全部内置检测项的代码实现（毫秒级、零 token）。

每个函数签名统一：

    check_xxx(text: str, ctx: RuleContext, params: dict) -> list[str]

返回违规理由列表（空 = 通过）。text 一律是**剥掉标签后的裸文本**。
异常单条吞掉 —— 任何一条规则出错都不能打断质检流程。
"""

from __future__ import annotations

import re
from typing import Any, Callable

from .config import as_float, as_int
from .items import CheckItem
from .model import RuleContext, Violation

# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

#: 天然无标点的长串（URL、路径、base64…）不算"缺标点"
_NO_PUNCT_RE = re.compile(r"[0-9a-zA-Z./:_\-?=&%#~]+")


def cjk_count(text: str) -> int:
    return len(_CJK_RE.findall(text))


def normalize(text: str) -> str:
    """比对用归一化：只留中英文与数字，转小写。"""
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", text or "").lower()


def split_clauses(text: str) -> list[str]:
    return [seg.strip() for seg in re.split(r"[，,、;；:：!！?？…。.~～\n\r]+", text or "") if seg.strip()]


def grams(text: str, n: int = 2) -> set[str]:
    return {text[i:i + n] for i in range(max(0, len(text) - n + 1))}


def coverage(a: set[str], b: set[str]) -> float:
    """a 被 b 覆盖的比例（取小的一方做分母），两边都空时为 0。"""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def longest_common(a: str, b: str) -> tuple[int, str]:
    """最长公共连续片段（长度, 片段）。输入应已 normalize（短文本，DP 足够快）。"""
    if not a or not b:
        return 0, ""
    if len(a) > 600 or len(b) > 600:
        a, b = a[:600], b[:600]      # 超长消息截断比对，保命要紧
    prev_row = [0] * (len(b) + 1)
    best_len = 0
    best_end = 0
    for i in range(1, len(a) + 1):
        row = [0] * (len(b) + 1)
        ca = a[i - 1]
        for j in range(1, len(b) + 1):
            if ca == b[j - 1]:
                row[j] = prev_row[j - 1] + 1
                if row[j] > best_len:
                    best_len = row[j]
                    best_end = i
        prev_row = row
    if best_len <= 0:
        return 0, ""
    return best_len, a[best_end - best_len:best_end]


def _phrase_lines(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def find_phrases(text: str, phrases: list[str], limit: int = 3) -> list[str]:
    """大小写不敏感的短语命中（清单通常含中英混合）。"""
    lower = (text or "").lower()
    hits: list[str] = []
    for phrase in phrases:
        needle = str(phrase).strip().lower()
        if needle and needle in lower:
            hits.append(str(phrase).strip())
            if len(hits) >= limit:
                break
    return hits


# ---------------------------------------------------------------------------
# 各检测项
# ---------------------------------------------------------------------------

def check_mid_punct(text: str, ctx: RuleContext, params: dict) -> list[str]:
    """句中缺少标点：单个分句连续超长且无任何断句标记。"""
    max_run = as_int(params.get("max_run"), 15, low=3, high=300)
    out: list[str] = []
    for seg in split_clauses(text):
        if len(seg) <= max_run:
            continue
        if cjk_count(seg) < 1:
            continue                      # 纯英文/数字/URL 不算
        if _NO_PUNCT_RE.fullmatch(seg):
            continue
        out.append(f"连续 {len(seg)} 字没有任何标点：“{seg[:18]}…”（阈值 {max_run} 字）")
    return out


def check_end_period(text: str, ctx: RuleContext, params: dict) -> list[str]:
    """句末有句号：只禁“。”和西文“.”（按你的决策）。"""
    tail = (text or "").rstrip()
    if tail and tail[-1] in "。.":
        return [f"末尾字符是“{tail[-1]}”"]
    return []


def check_overlap_prev(text: str, ctx: RuleContext, params: dict) -> list[str]:
    """与上一句（上一条已发送消息）重合过多。"""
    prev = (ctx.prev or "").strip()
    if not prev:
        return []
    min_common = as_int(params.get("min_common"), 8, low=3, high=200)
    gram_ratio = as_float(params.get("gram_ratio"), 0.5, low=0.05, high=1.0)
    cur_norm, prev_norm = normalize(text), normalize(prev)
    if len(cur_norm) < min_common or len(prev_norm) < min_common:
        return []
    out: list[str] = []
    lcs_len, lcs_text = longest_common(cur_norm, prev_norm)
    if lcs_len >= min_common:
        out.append(f"与上一句的公共片段达 {lcs_len} 字：“{lcs_text[:20]}”（阈值 {min_common} 字）")
    ratio = coverage(grams(cur_norm, 2), grams(prev_norm, 2))
    if ratio >= gram_ratio:
        out.append(f"与上一句的相邻两字重合率 {ratio:.0%}（阈值 {gram_ratio:.0%}）")
    return out


def check_overlap_self(text: str, ctx: RuleContext, params: dict) -> list[str]:
    """本条内部自我重复：两个分句几乎说的是同一句。"""
    min_common = as_int(params.get("min_common"), 6, low=3, high=100)
    clauses = [normalize(seg) for seg in split_clauses(text)]
    clauses = [seg for seg in clauses if len(seg) >= min_common]
    for i in range(len(clauses)):
        for j in range(i + 1, len(clauses)):
            a, b = clauses[i], clauses[j]
            if a == b or (len(a) >= min_common and (a in b or b in a)):
                return [f"第 {i + 1} 句与第 {j + 1} 句重复：“{a[:16]}” vs “{b[:16]}”"]
    return []


def check_same_head(text: str, ctx: RuleContext, params: dict) -> list[str]:
    """相同开头：与最近 K 条里多条开头一致。"""
    head_len = as_int(params.get("head_len"), 4, low=2, high=30)
    need = as_int(params.get("matches"), 2, low=1, high=20)
    cur = (text or "").strip()
    if len(cur) < head_len:
        return []
    head = cur[:head_len]
    hits = 0
    for old in ctx.recent:
        old = (old or "").strip()
        if len(old) >= head_len and old[:head_len] == head:
            hits += 1
    if hits >= need:
        return [f"开头「{head}」在最近 {len(ctx.recent)} 条里出现 {hits} 次（阈值 {need} 次）"]
    return []


def check_same_tail(text: str, ctx: RuleContext, params: dict) -> list[str]:
    """相同结尾：结尾 N 字重复，或同一语气词收尾。"""
    tail_len = as_int(params.get("tail_len"), 3, low=2, high=30)
    need = as_int(params.get("matches"), 2, low=1, high=20)
    particle_need = as_int(params.get("particle_matches"), 3, low=2, high=20)
    particles = str(params.get("particle_chars") or "呢呀哦啦嘛吧啊诶嘿哈~～")
    out: list[str] = []

    cur = (text or "").strip()
    if len(cur) >= tail_len:
        tail = cur[-tail_len:]
        hits = 0
        for old in ctx.recent:
            old = (old or "").strip()
            if len(old) >= tail_len and old[-tail_len:] == tail:
                hits += 1
        if hits >= need:
            out.append(f"结尾「{tail}」在最近 {len(ctx.recent)} 条里出现 {hits} 次（阈值 {need} 次）")

    if cur and cur[-1] in particles:
        last = cur[-1]
        hits = 0
        for old in ctx.recent:
            old = (old or "").strip()
            if old and old[-1] == last:
                hits += 1
        if hits >= particle_need:
            out.append(f"最近 {hits + 1} 条都以语气词“{last}”收尾（阈值 {particle_need} 次）")
    return out


def check_boilerplate(text: str, ctx: RuleContext, params: dict) -> list[str]:
    """AI 套话清单命中。"""
    phrases = _phrase_lines(params.get("phrases"))
    return [f"命中套话：“{hit}”" for hit in find_phrases(text, phrases)]


def check_structure(text: str, ctx: RuleContext, params: dict) -> list[str]:
    """结构化排版：编号列表 / Markdown / 三段式。"""
    out: list[str] = []
    raw = text or ""
    if len(re.findall(r"(?m)^\s*\d+[.、)]\s*\S", raw)) >= 2:
        out.append("出现编号列表（1. 2. 形式）")
    if len(re.findall(r"(?m)^\s*[-*•]\s+\S", raw)) >= 2:
        out.append("出现项目符号列表")
    if re.search(r"\*\*[^*\n]+\*\*", raw):
        out.append("出现 **加粗** 标记")
    if re.search(r"(?m)^#{1,6}\s", raw):
        out.append("出现 Markdown 标题")
    if re.search(chr(96) + r"[^" + chr(96) + r"\n]+" + chr(96), raw):
        out.append("出现反引号代码")
    if re.search(r"\[[^\]\n]+\]\([^)\n]+\)", raw):
        out.append("出现 Markdown 链接")
    if all(word in raw for word in ("首先", "其次", "最后")):
        out.append("出现“首先…其次…最后”三段式")
    if len(re.findall(r"[①②③④⑤⑥]", raw)) >= 2:
        out.append("出现圈号列表（①②）")
    return out[:3]


def check_identity_leak(text: str, ctx: RuleContext, params: dict) -> list[str]:
    """AI 身份泄漏。"""
    return [f"身份泄漏：“{hit}”" for hit in find_phrases(text, _phrase_lines(params.get("phrases")))]


def check_written_style(text: str, ctx: RuleContext, params: dict) -> list[str]:
    """书面语与翻译腔。"""
    return [f"书面语：“{hit}”" for hit in find_phrases(text, _phrase_lines(params.get("phrases")))]


def check_residue(text: str, ctx: RuleContext, params: dict) -> list[str]:
    """残留标记：协议占位、半截标签、异常空白。"""
    raw = text or ""
    out: list[str] = []
    if re.search(r"<quoted_message\b", raw, re.IGNORECASE):
        out.append("残留 quoted_message 引用占位")
    if re.search(r"<\s*/?\s*(?:emotion|mood)\b", raw, re.IGNORECASE):
        out.append("残留 emotion/mood 标签")
    if re.search(r"【\s*(?:好感度|好感|亲密度|熟悉度)", raw) and "】" not in raw:
        out.append("残留半截好感度标签")
    if "\u3000" in raw:
        out.append("含全角空格")
    if re.search(r"[\u200b-\u200f\ufeff]", raw):
        out.append("含零宽字符")
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", raw):
        out.append("含控制字符")
    return out


def check_punct_oddity(text: str, ctx: RuleContext, params: dict) -> list[str]:
    """连续重复标点 / 同一表情刷屏。"""
    raw = text or ""
    exclam_min = as_int(params.get("exclam_min"), 3, low=2, high=20)
    quest_min = as_int(params.get("quest_min"), 3, low=2, high=20)
    dots_min = as_int(params.get("dots_min"), 4, low=3, high=20)
    emoji_min = as_int(params.get("emoji_min"), 3, low=3, high=20)
    out: list[str] = []
    if re.search(r"([!！]){" + str(exclam_min) + r",}", raw):
        out.append(f"连续 {exclam_min} 个以上感叹号")
    if re.search(r"([?？]){" + str(quest_min) + r",}", raw):
        out.append(f"连续 {quest_min} 个以上问号")
    if re.search(r"([。.]){" + str(dots_min) + r",}", raw):
        out.append(f"连续 {dots_min} 个以上句点")
    flat = raw.replace("\ufe0f", "")
    if re.search(r"([\u2600-\u27BF\u1F300-\u1FAFF\u2B00-\u2BFF])\1{" + str(emoji_min - 1) + r",}", flat):
        out.append(f"同一表情连发 {emoji_min} 次以上")
    return out


def check_echo_user(text: str, ctx: RuleContext, params: dict) -> list[str]:
    """复述用户原话。"""
    user = (ctx.user_prev or "").strip()
    if not user:
        return []
    ratio = as_float(params.get("ratio"), 0.6, low=0.1, high=1.0)
    min_len = as_int(params.get("min_len"), 8, low=4, high=200)
    cur_norm, user_norm = normalize(text), normalize(user)
    if len(cur_norm) < min_len or len(user_norm) < min_len:
        return []
    value = coverage(grams(cur_norm, 2), grams(user_norm, 2))
    if value >= ratio:
        return [f"与用户上一条消息重合率 {value:.0%}（阈值 {ratio:.0%}），像在复读"]
    return []


def check_filler_dup(text: str, ctx: RuleContext, params: dict) -> list[str]:
    """口头禅超频：同一词组出现次数过多。"""
    count = as_int(params.get("count"), 3, low=2, high=20)
    stopwords = set(_phrase_lines(params.get("stopwords")))
    if len(text or "") < 8:
        return []
    core = re.sub(r"[^\u4e00-\u9fff]+", "", text or "")
    if len(core) < 4:
        return []
    best: tuple[int, str] = (0, "")
    for n in (2, 3):
        seen: set[str] = set()
        for i in range(len(core) - n + 1):
            token = core[i:i + n]
            if token in seen or token in stopwords:
                continue
            seen.add(token)
            hits = core.count(token)
            if hits > best[0]:
                best = (hits, token)
    if best[0] >= count and best[1]:
        return [f"词组“{best[1]}”在这条消息里出现了 {best[0]} 次（阈值 {count} 次）"]
    return []


def check_too_long(text: str, ctx: RuleContext, params: dict) -> list[str]:
    """单条过长。"""
    max_len = as_int(params.get("max_len"), 400, low=50, high=5000)
    length = len((text or "").strip())
    if length > max_len:
        return [f"共 {length} 字，超过 {max_len} 字上限"]
    return []


#: 检测项 id → 实现
RULE_FUNCS: dict[str, Callable[[str, RuleContext, dict], list[str]]] = {
    "mid_punct": check_mid_punct,
    "end_period": check_end_period,
    "overlap_prev": check_overlap_prev,
    "overlap_self": check_overlap_self,
    "same_head": check_same_head,
    "same_tail": check_same_tail,
    "boilerplate": check_boilerplate,
    "structure": check_structure,
    "identity_leak": check_identity_leak,
    "written_style": check_written_style,
    "residue": check_residue,
    "punct_oddity": check_punct_oddity,
    "echo_user": check_echo_user,
    "filler_dup": check_filler_dup,
    "too_long": check_too_long,
}


def run_rules(text: str, items: list[CheckItem], ctx: RuleContext) -> list[Violation]:
    """跑一遍规则层。单条规则异常直接跳过，绝不外抛。"""
    ctx = ctx.safe()
    out: list[Violation] = []
    for item in items:
        func = RULE_FUNCS.get(item.id)
        if func is None:
            continue
        try:
            reasons = func(text, ctx, item.params or {})
        except Exception:  # noqa: BLE001 - 规则炸了也不能打断质检
            continue
        for reason in reasons[:3]:
            out.append(Violation(id=item.id, name=item.name, reason=reason, engine="rule"))
    return out
