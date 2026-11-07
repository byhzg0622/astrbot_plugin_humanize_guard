"""检测项注册表：内置种子 + 用户自定义 + 勾选/注入词/改写词的持久化。

三条产品决策落在这一个文件里：
1. 新建检测项 → 自动生成"检测模型注入词"和"打回改写提示词"，两者都可改；
2. 检测项可选择性勾选 —— 勾选状态决定哪些注入词进入检测模型的提示词（judge.py 拼装）；
3. 内置项（含规则项）也带注入词：规则项由代码先判，注入词再让 AI 层复核，
   这样"勾选任何一项"都会如实改变注入词，所见即所得。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import as_bool

# ---------------------------------------------------------------------------
# 内置检测项种子
# ---------------------------------------------------------------------------
# engine: "rule" = 代码层有对应实现（rules.py RULE_FUNCS）；"judge" = 纯 AI 判定。
# prompt: 给检测模型的注入词（勾选后进入拼装，可改）
# rewrite: 打回时拼进改写请求的提示词（可改）

_DEFAULT_BOILERPLATE = [
    "作为一个AI", "作为一个ai", "作为AI", "作为人工智能", "人工智能助手",
    "我理解你的感受", "我理解你的难过", "我很理解", "希望这能帮", "希望对你有帮助",
    "希望这对你有帮助", "如果你还有其他", "如果你有任何其他", "随时告诉我",
    "随时联系我", "很高兴为你服务", "为你服务", "抱歉让你感到", "很抱歉让你",
    "我无法提供", "我无法回答", "作为一个语言模型", "作为一个助手",
    "希望你度过愉快", "祝你有美好的一天", "感谢你的提问", "感谢您的支持",
    "请注意，", "让我来帮你", "当然可以！",
]

_DEFAULT_IDENTITY = [
    "作为一个AI", "作为AI", "人工智能助手", "语言模型", "我只是一个程序",
    "我是程序", "你的AI助手", "ai助手", "AI助手", "作为助手", "机器人助手",
    "我被设计", "我的开发者", "程序设定", "人类用户", "用户你好", "您好，用户",
    "作为一个语言模型",
]

_DEFAULT_WRITTEN = [
    "值得注意的是", "综上所述", "总而言之", "总的来说", "众所周知",
    "在某种意义上", "不可否认", "换言之", "简而言之", "由此可见",
    "与此同时", "基于此", "首先需要指出", "从某种意义上",
]

_DEFAULT_FILLER_STOPWORDS = [
    "一个", "不是", "就是", "什么", "怎么", "因为", "所以", "但是", "这个", "那个",
    "可以", "知道", "觉得", "现在", "时候", "我们", "你们", "他们", "自己", "没有",
    "如果", "已经", "还是", "这样", "那样", "真的", "有点", "一下", "这么", "那么",
]

SEEDS: list[dict[str, Any]] = [
    # ---------------- 你点名的四条核心检测 ----------------
    {
        "id": "mid_punct",
        "name": "句中缺少标点",
        "engine": "rule",
        "enabled": True,
        "desc": "一个分句连续很长却没有任何断句标点，读起来一口气从头连到尾。",
        "prompt": (
            "句子中间缺少标点：一个分句连续超过约 15 个汉字没有任何逗号、顿号、分号等断句标记。"
            "短句、纯语气词句不算；按参数里的字数阈值判断。"
        ),
        "rewrite": "在过长的连续句里补上逗号/顿号断句，保持口语感，别整段一句到底。",
        "params": {"max_run": 15},
    },
    {
        "id": "end_period",
        "name": "句末有句号",
        "engine": "rule",
        "enabled": True,
        "desc": "消息最后一个字符是句号（只禁“。”和西文“.”，其他标点不管）。",
        "prompt": (
            "句末有句号：消息最后一个非空白字符是句号“。”或西文句点“.”。"
            "只看最后一个字符，句中允许出现标点；末尾是文字、语气词、省略号都不算。"
        ),
        "rewrite": "删掉句末的句号，让消息停在文字或语气词上（句中的逗号保留）。",
        "params": {},
    },
    {
        "id": "overlap_prev",
        "name": "与上一句重合过多",
        "engine": "rule",
        "enabled": True,
        "desc": "与机器人上一条已发送消息的公共片段过长（或二元组重合率过高）。",
        "prompt": (
            "与上一条自己发出的消息重合过多：去掉标点空白后，与参考信息里“上一条自己发出的消息”的"
            "公共连续片段达到约 8 字，或二元组（相邻两字）重合率超过一半。"
            "只和上一条已发送消息比，不和任何草稿版本比。"
        ),
        "rewrite": "换一种说法：与上一条消息的用词和句式至少换掉一半，开头结尾都不能沿用。",
        "params": {"min_common": 8, "gram_ratio": 0.5},
    },
    {
        "id": "same_head",
        "name": "相同开头",
        "engine": "rule",
        "enabled": True,
        "desc": "开头几个字和最近几条自己发出的消息一样（如每条都以“哈哈”开头）。",
        "prompt": (
            "相同开头：开头约 4 个字与参考信息里“最近发出的消息”中的任意一条相同，"
            "最近几条里出现 2 次及以上。只比对去掉标签后的正文开头。"
        ),
        "rewrite": "换一个开头，别重复最近几条用过的开头几个字。",
        "params": {"head_len": 4, "matches": 2},
    },
    {
        "id": "same_tail",
        "name": "相同结尾",
        "engine": "rule",
        "enabled": True,
        "desc": "结尾几个字或收尾语气词和最近几条一样（如每条都以“呢”结尾）。",
        "prompt": (
            "相同结尾：结尾约 3 个字与参考信息里“最近发出的消息”中的任意一条相同，出现 2 次及以上；"
            "或每条都以同一个语气词（呢/呀/哦/啦/嘛/吧/啊/诶/嘿/哈/～）收尾，出现 3 次及以上。"
            "好感度等标签已在正文外，不参与比对。"
        ),
        "rewrite": "换一个结尾方式，别总是用相同的几个字或同一个语气词收尾。",
        "params": {
            "tail_len": 3,
            "matches": 2,
            "particle_matches": 3,
            "particle_chars": "呢呀哦啦嘛吧啊诶嘿哈~～",
        },
    },
    {
        "id": "overlap_self",
        "name": "本条内部重复",
        "engine": "rule",
        "enabled": True,
        "desc": "一条消息里把同一个意思说了两遍（两个分句几乎一样）。",
        "prompt": (
            "本条消息内部自我重复：断句后有两个分句几乎相同（完全相同或互为包含，长度约 6 字以上），"
            "一个意思说了两遍。"
        ),
        "rewrite": "删掉本条内部重复的句子，一个意思只说一遍。",
        "params": {"min_common": 6},
    },
    # ---------------- 常见 AI 痕迹（规则层） ----------------
    {
        "id": "boilerplate",
        "name": "AI 套话",
        "engine": "rule",
        "enabled": True,
        "desc": "客服式寒暄与模板句（清单可在参数里增删）。",
        "prompt": (
            "AI 套话：出现客服式寒暄与模板句，如“希望这能帮到你”“随时告诉我”"
            "“很高兴为你服务”“我理解你的感受”等（完整清单见检测参数）。"
            "命中清单任意一条即算违规。"
        ),
        "rewrite": "删掉套话和寒暄模板，直接说正事。",
        "params": {"phrases": list(_DEFAULT_BOILERPLATE)},
    },
    {
        "id": "structure",
        "name": "结构化排版",
        "engine": "rule",
        "enabled": True,
        "desc": "编号列表、项目符号、Markdown 标记、首先其次最后三段式。",
        "prompt": (
            "结构化排版：出现编号列表（1. 2. / ①②）、项目符号（- 或 * 开头成行）、"
            "Markdown 标记（**加粗**、#标题、反引号代码、[链接](…)），"
            "或“首先…其次…最后”式三段论述。真人聊天不会这样排版。"
        ),
        "rewrite": "去掉编号、列表、加粗等排版痕迹，改成自然连贯的口语。",
        "params": {},
    },
    {
        "id": "identity_leak",
        "name": "AI 身份泄漏",
        "engine": "rule",
        "enabled": True,
        "desc": "自称 AI/助手/语言模型，或把对方叫“用户”（清单可增删）。",
        "prompt": (
            "AI 身份泄漏：自称“作为AI/人工智能/语言模型/助手/程序”，"
            "或把对方称作“用户/人类”，或提到“我的开发者/被设计”等暴露机器身份的表述"
            "（清单见检测参数）。"
        ),
        "rewrite": "删掉一切机器身份表述：不自称 AI/助手/程序，也不把对方叫“用户”。",
        "params": {"phrases": list(_DEFAULT_IDENTITY)},
    },
    {
        "id": "written_style",
        "name": "书面语与翻译腔",
        "engine": "rule",
        "enabled": True,
        "desc": "“综上所述/值得注意的是”类书面连接词（清单可增删）。",
        "prompt": (
            "书面语与翻译腔：出现“值得注意的是/综上所述/总而言之/众所周知/在某种意义上/不可否认/换言之”"
            "等书面连接词（清单见检测参数），句子像书面文章而不像随口说话。"
        ),
        "rewrite": "把书面连接词换成日常口语，像平时打字那样说话。",
        "params": {"phrases": list(_DEFAULT_WRITTEN)},
    },
    {
        "id": "residue",
        "name": "残留标记",
        "engine": "rule",
        "enabled": True,
        "desc": "未清理的协议占位、半截标签、全角/零宽/控制字符。",
        "prompt": (
            "残留标记：消息里有未清理的协议占位或标签残片（如 quoted_message 占位、未闭合的【好感度、"
            "emotion 标签），或全角空格、零宽字符、控制字符等异常空白。"
        ),
        "rewrite": "删掉残留的标记、占位符和异常空白字符。",
        "params": {},
    },
    {
        "id": "punct_oddity",
        "name": "标点/表情异常",
        "engine": "rule",
        "enabled": True,
        "desc": "连续重复标点（！！！/？？/。。。。）或同一表情连发。",
        "prompt": (
            "标点或表情异常：连续 3 个以上重复的感叹号/问号（“！！！”“？？”），"
            "连续 4 个以上的句点（“。。。。”），或同一个表情符号连发 3 次以上。"
            "两三个字的自然语气（如“好哇”）不算。"
        ),
        "rewrite": "去掉连续重复的标点和刷屏表情，一个就够。",
        "params": {"exclam_min": 3, "quest_min": 3, "dots_min": 4, "emoji_min": 3},
    },
    # ---------------- 可选检测（默认关，按需勾选） ----------------
    {
        "id": "echo_user",
        "name": "复述用户原话",
        "engine": "rule",
        "enabled": False,
        "desc": "和用户上一条消息高度相似，像复读机。",
        "prompt": (
            "复述对方原话：与参考信息里“用户上一条消息”的二元组重合率超过约 60%，"
            "基本是在复读对方说的话而不是回应（仅当用户消息足够长时判断）。"
        ),
        "rewrite": "别复读对方的原话，用自己的话回应。",
        "params": {"ratio": 0.6, "min_len": 8},
    },
    {
        "id": "filler_dup",
        "name": "口头禅超频",
        "engine": "rule",
        "enabled": False,
        "desc": "同一个词组一条消息里出现太多次（常见虚词清单除外）。",
        "prompt": (
            "口头禅超频：同一个 2-3 字词组（如“嘿嘿”“真的”“那个”）在一条消息里"
            "出现 3 次及以上，常见虚词清单内的不算。"
        ),
        "rewrite": "减少同一口头禅/叠词的重复，出现一次就够。",
        "params": {"count": 3, "stopwords": list(_DEFAULT_FILLER_STOPWORDS)},
    },
    {
        "id": "too_long",
        "name": "单条过长",
        "engine": "rule",
        "enabled": False,
        "desc": "一条消息像作文，超过设定字数。",
        "prompt": "单条过长：去掉标签后超过约 400 字，像作文而不像一口气打完的聊天消息。",
        "rewrite": "把这条拆短，挑最要紧的说，像平时打字一样分开发。",
        "params": {"max_len": 400},
    },
    # ---------------- AI 检测项（纯模型裁决） ----------------
    {
        "id": "ai_flavor",
        "name": "整体 AI 味",
        "engine": "judge",
        "enabled": True,
        "desc": "总评：客服腔、模板转折、空洞关心、总分总、句长过于均匀。",
        "prompt": (
            "整体 AI 味（总评）：通篇客服腔、模板化转折（生硬的“不过/对了/话说”开头）、"
            "空洞的关心套话、总分总结构、每句长度过于均匀、过度解释与过度礼貌。"
            "只在迹象明显时判违规，拿不准就放过。"
        ),
        "rewrite": (
            "整体去掉 AI 腔：删掉模板转折和客套，句子长短错落，像真人随口说的，"
            "但不要改变原意与人设语气。"
        ),
        "params": {},
    },
]

#: 规则参数在面板上的渲染说明（type: int / float / text / lines）
#: int/float 带 low/high：保存接口按这个范围钳制（越界钳到边界，不再回退默认值）
PARAM_SCHEMA: dict[str, list[dict[str, Any]]] = {
    "mid_punct": [
        {"key": "max_run", "label": "连续多少字算缺标点", "type": "int",
         "default": 15, "low": 3, "high": 300},
    ],
    "overlap_prev": [
        {"key": "min_common", "label": "公共连续片段≥多少字算重合", "type": "int", "default": 8},
        {"key": "gram_ratio", "label": "二元组重合率阈值(0~1)", "type": "float", "default": 0.5},
    ],
    "overlap_self": [
        {"key": "min_common", "label": "重复分句≥多少字", "type": "int", "default": 6},
    ],
    "same_head": [
        {"key": "head_len", "label": "开头比对字数", "type": "int", "default": 4},
        {"key": "matches", "label": "最近几条中命中几次算违规", "type": "int", "default": 2},
    ],
    "same_tail": [
        {"key": "tail_len", "label": "结尾比对字数", "type": "int", "default": 3},
        {"key": "matches", "label": "命中几次算违规", "type": "int", "default": 2},
        {"key": "particle_matches", "label": "同一语气词收尾几次算违规", "type": "int", "default": 3},
        {"key": "particle_chars", "label": "语气词集合", "type": "text", "default": "呢呀哦啦嘛吧啊诶嘿哈~～"},
    ],
    "boilerplate": [
        {"key": "phrases", "label": "套话清单（每行一条）", "type": "lines", "default": ""},
    ],
    "identity_leak": [
        {"key": "phrases", "label": "身份泄漏清单（每行一条）", "type": "lines", "default": ""},
    ],
    "written_style": [
        {"key": "phrases", "label": "书面语清单（每行一条）", "type": "lines", "default": ""},
    ],
    "echo_user": [
        {"key": "ratio", "label": "重合率阈值(0~1)", "type": "float", "default": 0.6},
        {"key": "min_len", "label": "用户消息至少多少字才比对", "type": "int", "default": 8},
    ],
    "filler_dup": [
        {"key": "count", "label": "同一词组出现几次算违规", "type": "int", "default": 3},
        {"key": "stopwords", "label": "豁免虚词（每行一条）", "type": "lines", "default": ""},
    ],
    "too_long": [
        {"key": "max_len", "label": "最多多少字", "type": "int", "default": 400},
    ],
    "punct_oddity": [
        {"key": "exclam_min", "label": "感叹号连续几个算异常", "type": "int", "default": 3},
        {"key": "quest_min", "label": "问号连续几个算异常", "type": "int", "default": 3},
        {"key": "dots_min", "label": "句点连续几个算异常", "type": "int", "default": 4},
        {"key": "emoji_min", "label": "同一表情连发几个算异常", "type": "int", "default": 3},
    ],
}


def gen_prompt(name: str, desc: str) -> str:
    """新建检测项时自动生成的检测模型注入词。"""
    name = str(name or "").strip() or "未命名检测项"
    desc = str(desc or "").strip() or "（未填写判定说明）"
    return (
        f"【{name}】\n{desc}\n"
        "若消息命中上述描述即为违规；拿不准时放过，不要牵强附会。"
    )


def gen_rewrite(name: str, desc: str) -> str:
    """新建检测项时自动生成的打回改写提示词。"""
    name = str(name or "").strip() or "未命名检测项"
    desc = str(desc or "").strip() or ""
    if desc:
        return f"修正「{name}」问题：{desc}"
    return f"修正「{name}」问题。"


@dataclass
class CheckItem:
    id: str
    name: str
    engine: str                # rule | judge
    desc: str = ""
    prompt: str = ""
    rewrite: str = ""
    enabled: bool = True
    builtin: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "engine": self.engine,
            "desc": self.desc,
            "prompt": self.prompt,
            "rewrite": self.rewrite,
            "enabled": bool(self.enabled),
            "builtin": bool(self.builtin),
            "params": dict(self.params),
            "param_schema": PARAM_SCHEMA.get(self.id, []),
        }


class ItemStore:
    """检测项的持久化：内置种子 + 覆盖 + 自定义。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._items: dict[str, CheckItem] = {}
        self._order: list[str] = []

    # -- 加载 / 保存 ---------------------------------------------------------
    def load(self) -> None:
        self._items = {}
        self._order = []
        # 1) 内置种子
        for seed in SEEDS:
            item = CheckItem(
                id=str(seed["id"]),
                name=str(seed["name"]),
                engine=str(seed.get("engine") or "rule"),
                desc=str(seed.get("desc") or ""),
                prompt=str(seed.get("prompt") or ""),
                rewrite=str(seed.get("rewrite") or ""),
                enabled=as_bool(seed.get("enabled"), True),
                builtin=True,
                params=dict(seed.get("params") or {}),
            )
            self._items[item.id] = item
            self._order.append(item.id)
        # 2) 文件覆盖 + 自定义项
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - 没有文件就用纯种子
            raw = {}
        stored = raw.get("items") if isinstance(raw, dict) else None
        if isinstance(stored, dict):
            for item_id, state in stored.items():
                if not isinstance(state, dict):
                    continue
                item = self._items.get(str(item_id))
                if item is None:
                    # 自定义项：完整记录
                    engine = str(state.get("engine") or "judge")
                    if engine not in {"rule", "judge"}:
                        engine = "judge"
                    item = CheckItem(
                        id=str(item_id),
                        name=str(state.get("name") or item_id),
                        engine=engine,
                        desc=str(state.get("desc") or ""),
                        prompt=str(state.get("prompt") or ""),
                        rewrite=str(state.get("rewrite") or ""),
                        enabled=as_bool(state.get("enabled"), True),
                        builtin=False,
                        params=dict(state.get("params") or {}),
                    )
                    self._items[item.id] = item
                    self._order.append(item.id)
                    continue
                # 内置项：只覆盖可编辑字段
                if "enabled" in state:
                    item.enabled = as_bool(state.get("enabled"), item.enabled)
                if "prompt" in state:
                    item.prompt = str(state.get("prompt") or "")
                if "rewrite" in state:
                    item.rewrite = str(state.get("rewrite") or "")
                if isinstance(state.get("params"), dict):
                    merged = dict(item.params)
                    merged.update(state["params"])
                    item.params = merged

    def save(self) -> None:
        state: dict[str, Any] = {}
        for item_id in self._order:
            item = self._items.get(item_id)
            if item is None:
                continue
            entry: dict[str, Any] = {
                "enabled": item.enabled,
                "prompt": item.prompt,
                "rewrite": item.rewrite,
                "params": item.params,
            }
            if not item.builtin:
                entry.update({
                    "name": item.name,
                    "engine": item.engine,
                    "desc": item.desc,
                })
            state[item_id] = entry
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps({"version": 1, "items": state}, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001 - 落盘失败不影响本次运行
            pass

    # -- 查询 ---------------------------------------------------------------
    def list(self) -> list[CheckItem]:
        return [self._items[i] for i in self._order if i in self._items]

    def get(self, item_id: str) -> CheckItem | None:
        return self._items.get(str(item_id))

    def enabled_rule_items(self) -> list[CheckItem]:
        return [i for i in self.list() if i.enabled and i.engine == "rule"]

    def enabled_judge_items(self) -> list[CheckItem]:
        """进入注入词拼装的检测项：勾选了的、且注入词非空的。"""
        return [i for i in self.list() if i.enabled and i.prompt.strip()]

    def enabled_items(self) -> list[CheckItem]:
        return [i for i in self.list() if i.enabled]

    def rewrite_hints(self, ids: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for item_id in ids:
            item = self._items.get(str(item_id))
            if item is not None and item.rewrite.strip():
                out[item.id] = item.rewrite.strip()
        return out

    # -- 修改 ---------------------------------------------------------------
    def update(
        self,
        item_id: str,
        *,
        enabled: bool | None = None,
        prompt: str | None = None,
        rewrite: str | None = None,
        params: dict[str, Any] | None = None,
        name: str | None = None,
        desc: str | None = None,
    ) -> CheckItem:
        item = self._items.get(str(item_id))
        if item is None:
            raise KeyError(f"检测项不存在：{item_id}")
        if enabled is not None:
            item.enabled = bool(enabled)
        if prompt is not None:
            item.prompt = str(prompt)
        if rewrite is not None:
            item.rewrite = str(rewrite)
        if isinstance(params, dict):
            if item.builtin:
                merged = dict(item.params)
                merged.update(params)
                item.params = merged
            else:
                item.params = dict(params)
        if name is not None and str(name).strip():
            item.name = str(name).strip()
        if desc is not None:
            item.desc = str(desc)
        self.save()
        return item

    def add_custom(self, name: str, desc: str, prompt: str = "", rewrite: str = "") -> CheckItem:
        """新建检测项：未给注入词/改写词时按名称与判定说明自动生成。"""
        name = str(name or "").strip()
        if not name:
            raise ValueError("检测项名称不能为空")
        desc = str(desc or "").strip()
        item_id = "custom_" + uuid.uuid4().hex[:8]
        while item_id in self._items:
            item_id = "custom_" + uuid.uuid4().hex[:8]
        item = CheckItem(
            id=item_id,
            name=name,
            engine="judge",
            desc=desc,
            prompt=str(prompt).strip() or gen_prompt(name, desc),
            rewrite=str(rewrite).strip() or gen_rewrite(name, desc),
            enabled=True,
            builtin=False,
            params={},
        )
        self._items[item.id] = item
        self._order.append(item.id)
        self.save()
        return item

    def delete(self, item_id: str) -> None:
        item = self._items.get(str(item_id))
        if item is None:
            raise KeyError(f"检测项不存在：{item_id}")
        if item.builtin:
            raise ValueError("内置检测项不能删除，只能取消勾选")
        self._items.pop(item.id, None)
        self._order = [i for i in self._order if i != item.id]
        self.save()

    def reset_builtin(self) -> int:
        """内置项恢复出厂（保留自定义项）。返回恢复的条数。"""
        seeds = {str(s["id"]): s for s in SEEDS}
        restored = 0
        for item_id, seed in seeds.items():
            item = self._items.get(item_id)
            if item is None:
                item = CheckItem(id=item_id, name=str(seed["name"]), engine=str(seed.get("engine") or "rule"))
                self._items[item_id] = item
                self._order.append(item.id)
            item.name = str(seed["name"])
            item.desc = str(seed.get("desc") or "")
            item.prompt = str(seed.get("prompt") or "")
            item.rewrite = str(seed.get("rewrite") or "")
            item.enabled = as_bool(seed.get("enabled"), True)
            item.builtin = True
            item.params = dict(seed.get("params") or {})
            restored += 1
        self.save()
        return restored

    def stats(self) -> dict[str, Any]:
        items = self.list()
        return {
            "total": len(items),
            "enabled": sum(1 for i in items if i.enabled),
            "rule_enabled": len(self.enabled_rule_items()),
            "judge_enabled": len(self.enabled_judge_items()),
            "custom": sum(1 for i in items if not i.builtin),
        }
