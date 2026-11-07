"""人味守卫（astrbot_plugin_humanize_guard）。

拦截 AI 即将发出的消息 → 三层处理（规则 → **本地公式修复** → AI 裁决/改写），
命中即打回，按命中检测项把对应提示词注入改写请求 → 复检通过才真正发出。

提速要点（v1.1）：
- **本地公式修复层**（core/fix.py）：能用确定性规则改好的问题（句末句号、连发标点、
  残留字符、Markdown 排版、缺标点断句、重复语气词、套话/书面语…）直接本地改好，
  一次模型调用都不花；每个补丁都要过规则复检（违规变少且不新增才采纳）。
  本地修不掉的（与上一句重合、复述用户、整体 AI 味、自定义项）才交给 AI 改写。
- 改写后用**规则层复检**（可开 AI 复检）、改写稿再过一遍本地公式，
  一条消息的模型调用数从 3~5 次降到 0~2 次。
- **分段插件兼容**：对话分段Pro 会把回复切段直发（绕过装饰钩子），
  人味守卫现在会①接管 Context.send_message，先按"整条"质检再由它切分；
  ②识别并放行切出来的碎片，不再逐段重复质检，也不往历史里灌碎片。

主动消息（context.send_message / platform.send_by_session 直发，不经过事件
pipeline）也走同一套质检：包装每个平台实例的 send_by_session，幂等、可还原、
异常一律 fail-open 原样照发。

关键决策（按需求确认稿）：
- 句末只禁“。”和“.”；
- 打回改写用独立模型调用（方案 A）+ 可选的“下轮反馈注入”；
- 检测/改写默认用当前会话在用的模型（可指定 provider）；
- 好感度等标签检测时剥离、改写后原样加回；
- 重合检测只比“上一条已发送消息 vs 改写后文本”，改写前草稿从不入史。
"""

from __future__ import annotations

import asyncio
import contextvars
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools
try:
    from astrbot.core.platform.message_session import MessageSesion
except ImportError:  # 极老版本路径不同：主动消息质检退化为按整段 umo 匹配名单
    MessageSesion = None  # type: ignore[assignment,misc]

from .api.webapi import PLUGIN_ID, PLUGIN_NAME, PLUGIN_VERSION, WebApiMixin
from .core.config import as_bool, as_float, as_int, as_str, as_lines, cfg_get
from .core.fix import can_fix, local_repair
from .core.history import SessionHistory
from .core.items import ItemStore
from .core.judge import run_judge
from .core.log import CheckLog
from .core.model import RuleContext, Violation
from .core.rewrite import run_rewrite
from .core.rules import run_rules
from .core.tags import rejoin_tags, strip_message_markup, strip_tags

try:
    from astrbot.api.star import register
except ImportError:  # 极老版本没有 register：退化成空装饰器
    def register(*_args: Any, **_kwargs: Any):
        def decorator(cls: Any) -> Any:
            return cls

        return decorator


#: 下轮反馈注入的标记（防止重复注入）
FEEDBACK_MARKER = "[人味质检反馈]"


#: 名单条目两端可以剥掉的“包装”字符（引号、括号、全角空格……）
_ENTRY_NOISE = " \t\r\n\u3000\"'“”‘’`[]（）()<>《》"
#: 会话 ID 分段用的分隔符：唯一会话模式下群会话是「发送者_群号」，
#: 其余平台还有 % / : . - 等写法，全部当分隔符拆开逐段比对
_SEG_SPLIT = re.compile(r"[_\-/:.%!#&\s\u3000]+")


def _clean_entry(part: str) -> str:
    """去掉条目两端的引号 / 括号 / 空白等无意义字符。"""
    return part.strip().strip(_ENTRY_NOISE)


def _split_ids(raw: Any) -> set[str]:
    """名单配置 → 条目集合（支持换行 / 中英文逗号 / 分号分隔）。"""
    if not raw:
        return set()
    if isinstance(raw, (list, tuple, set)):
        parts = [str(item) for item in raw]
    else:
        parts = re.split(r"[,;，；\n]+", str(raw))
    return {clean for clean in (_clean_entry(part) for part in parts) if clean}


#: LLM 打标的有效期（秒）：过期的打标不再认，避免误把指令回复当 AI 回复
MARK_TTL = 600.0
#: 打标表与锁表的容量上限
_MARK_CAP = 500
_LOCK_CAP = 300
#: 每会话记住几条"刚质检过的整条消息"（用来识别分段插件切出来的碎片）
_SEEN_KEEP = 4
#: 记住多少个会话（LRU）
_SEEN_SESSIONS = 300
#: 比对用归一化：只去空白，标点保留（碎片与整条的差别只在空白上）
_WS_RE = re.compile(r"\s+")

@dataclass
class RewriteOutcome:
    """打回改写循环的结果。"""

    text: str
    passed: bool
    violations: list[Violation]
    rounds: int = 0
    error: str = ""
    local: list[str] = field(default_factory=list)   # 本循环里被本地公式修好的项


HELP_TEXT = """人味守卫 · 指令
/qc 状态 —— 查看质检开关与统计
/qc 测试 <文本> —— 试跑检测（不改写、不发送）
/qc 日志 [条数] —— 最近的打回记录
/qc 帮助 —— 本说明
检测项勾选、注入词编辑、新建检测项都在面板里。"""


@register(PLUGIN_NAME, "workspace", "人味守卫：拦截发送消息质检打回，注入提示词督促改写", PLUGIN_VERSION)
class HumanizeGuard(WebApiMixin, Star):
    """质检主管线。"""

    def __init__(self, context: Context, config: Any = None) -> None:
        super().__init__(context)
        self.config = config if isinstance(config, dict) else {}
        self.data_dir = self._resolve_data_dir()
        self.items = ItemStore(self.data_dir / "items.json")
        self.items.load()
        self.history = SessionHistory(self.data_dir / "history.json")
        self.history.load()
        self.log = CheckLog(self.data_dir / "log.json")
        self.log.load()

        self._schema_cache: dict[str, Any] | None = None
        self._settings_saved_at = 0.0
        self._llm_marks: OrderedDict[str, float] = OrderedDict()
        self._feedback: dict[str, str] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._wrap_task: asyncio.Task | None = None   # 新平台补挂轮询（防包装丢失）
        self._wrapped: dict[int, dict[str, Any]] = {}  # id(平台) -> {"inst", "orig", "name"}
        self._wrap_guard: dict[int, Any] = {}          # id(平台) -> 占位（识别自己的包装）
        # 发送闸门：任务级隔离（ContextVar），避免多会话并发互相把质检吞掉
        self._send_guard: contextvars.ContextVar[bool] = contextvars.ContextVar(
            "humanize_guard_send", default=False
        )
        self._ctx_send_inner: Any = None                # 我们包住的那个 send_message
        self._ctx_send_wrapped = False
        # 刚质检过的整条消息：umo -> [(归一化文本, 时间戳)]，用来识别分段插件的碎片
        self._seen_texts: OrderedDict[str, list[tuple[str, float]]] = OrderedDict()

        # 运行参数（_reload_config 从配置读取）
        self.enabled = True
        self.enable_rules = True
        self.enable_judge = True
        self.judge_on_rule_fail = False
        self.skip_below = 4
        self.blacklist: set[str] = set()      # 黑名单（免检）
        self.whitelist: set[str] = set()      # 白名单（质检名单）
        self.whitelist_mode = False           # 白名单模式：只检名单内
        self.model_mode = "session"
        self.provider_id = ""
        self.model_timeout = 12.0
        self.max_rounds = 2
        self.dry_run = False
        self.feedback_enabled = True
        self.fast_fix = True          # 本地公式修复（省 AI 调用）
        self.recheck_judge = False    # 改写后用 AI 复检（默认只用规则层复检）
        self.dedup_fragments = True   # 识别分段插件切出来的碎片，不再逐段重复质检
        self.dedup_window = 60.0      # 碎片识别时间窗（秒）
        self.wrap_context_send = True # 主动发送先按整条质检再交给分段插件
        self.tag_patterns: list[str] = []
        self.debug_log = False
        self._reload_config()

        self._register_web_apis()
        logger.info(
            f"[人味守卫] 已加载：检测项 {self.items.stats()['enabled']}/{self.items.stats()['total']} 启用"
            f"（AI 层 {self.items.stats()['judge_enabled']} 项），数据目录 {self.data_dir}"
        )

    # ------------------------------------------------------------------
    # 生命周期与配置
    # ------------------------------------------------------------------
    def _resolve_data_dir(self) -> Path:
        getter = getattr(StarTools, "get_data_dir", None)
        if callable(getter):
            try:
                return Path(getter(PLUGIN_NAME))
            except Exception:  # noqa: BLE001
                pass
        return Path(__file__).resolve().parent / "data"

    async def initialize(self) -> None:
        self._wrap_platform_sends()
        self._assert_context_send()
        self._wrap_task = asyncio.create_task(self._wrap_platforms_loop())
        logger.info(f"[人味守卫] 就绪：规则层={'开' if self.enable_rules else '关'} "
                    f"AI 层={'开' if self.enable_judge else '关'} "
                    f"改写上限 {self.max_rounds} 轮"
                    f"{'（体检模式，只记不改）' if self.dry_run else ''}"
                    f"，主动消息拦截已就绪")

    @filter.on_astrbot_loaded(priority=-1000)
    async def on_astrbot_loaded(self) -> None:
        """框架加载完成后补一次包装。

        分段插件在自己 on_astrbot_loaded 里劫持 Context.send_message —— 谁后包谁在最外层。
        这里用更低的 priority 保证最后执行，把人味守卫包到最外层：主动消息先按整条质检，
        再由分段插件切分发送。
        """
        try:
            self._wrap_platform_sends()
            self._assert_context_send()
        except Exception as exc:  # noqa: BLE001 - 补挂失败不影响主流程
            logger.debug(f"[人味守卫] 加载后补挂失败：{exc}")

    async def terminate(self) -> None:
        if self._wrap_task is not None:
            self._wrap_task.cancel()
            self._wrap_task = None
        self._unwrap_platform_sends()
        self._unwrap_context_send()
        try:
            self.items.save()
            self.history.save()
            self.log.save()
        except Exception:  # noqa: BLE001
            pass
        logger.info("[人味守卫] 已卸载")

    def _reload_config(self) -> None:
        cfg = self.config
        self.enabled = as_bool(cfg_get(cfg, "enabled", True))
        self.enable_rules = as_bool(cfg_get(cfg, "detect.enable_rules", True))
        self.enable_judge = as_bool(cfg_get(cfg, "detect.enable_judge", True))
        self.judge_on_rule_fail = as_bool(cfg_get(cfg, "detect.judge_when_rules_fail", False))
        self.skip_below = as_int(cfg_get(cfg, "detect.skip_below", 4), 4, low=0, high=200)
        self.blacklist = _split_ids(cfg_get(cfg, "lists.blacklist", ""))
        self.whitelist = _split_ids(cfg_get(cfg, "lists.whitelist", ""))
        self.whitelist_mode = as_bool(cfg_get(cfg, "lists.whitelist_mode", False))
        mode = as_str(cfg_get(cfg, "model.mode", "session"), "session").strip().lower()
        self.model_mode = mode if mode in {"session", "custom"} else "session"
        self.provider_id = as_str(cfg_get(cfg, "model.provider_id", "")).strip()
        self.model_timeout = as_float(cfg_get(cfg, "model.timeout_seconds", 12.0), 12.0, low=1.0, high=300.0)
        self.max_rounds = as_int(cfg_get(cfg, "rewrite.max_rounds", 2), 2, low=0, high=10)
        self.dry_run = as_bool(cfg_get(cfg, "rewrite.dry_run", False))
        self.fast_fix = as_bool(cfg_get(cfg, "rewrite.fast_fix", True))
        self.recheck_judge = as_bool(cfg_get(cfg, "rewrite.recheck_judge", False))
        self.dedup_fragments = as_bool(cfg_get(cfg, "segment.dedup", True))
        self.dedup_window = as_float(cfg_get(cfg, "segment.dedup_window", 60.0), 60.0, low=5.0, high=3600.0)
        self.wrap_context_send = as_bool(cfg_get(cfg, "segment.wrap_context_send", True))
        self.feedback_enabled = as_bool(cfg_get(cfg, "feedback.enabled", True))
        keep_k = as_int(cfg_get(cfg, "history.keep_k", 5), 5, low=1, high=50)
        persist = as_bool(cfg_get(cfg, "history.persist", True))
        self.history.configure(keep_k=keep_k, persist=persist)
        self.tag_patterns = as_lines(cfg_get(cfg, "tags.patterns", ""))
        self.debug_log = as_bool(cfg_get(cfg, "debug.debug_log", False))

    # ------------------------------------------------------------------
    # 基础工具
    # ------------------------------------------------------------------
    @staticmethod
    def _event_key(event: AstrMessageEvent) -> str:
        message_id = str(getattr(event.message_obj, "message_id", "") or "")
        umo = event.unified_msg_origin
        return f"{umo}|{message_id}" if message_id else f"{umo}|*"

    def _mark_llm(self, event: AstrMessageEvent) -> None:
        key = self._event_key(event)
        self._llm_marks[key] = time.time()
        while len(self._llm_marks) > _MARK_CAP:
            self._llm_marks.popitem(last=False)

    def _consume_mark(self, key: str) -> bool:
        stamp = self._llm_marks.pop(key, None)
        if stamp is None:
            return False
        return (time.time() - stamp) <= MARK_TTL

    @staticmethod
    def _sender_id(event: AstrMessageEvent) -> str:
        try:
            return str(event.get_sender_id() or "")
        except Exception:  # noqa: BLE001 - 个别平台没有该方法
            return ""

    @staticmethod
    def _hit(entries: set[str], umo: str, sender: str, group: str = "") -> bool:
        """名单命中判定，四种写法都认：
        完整会话 ID / 发送者 ID / 群号 / 裸编号（自动匹配会话尾段及其分段）。

        关键坑（本次修的 bug）：AstrBot 开了「唯一会话 unique_session」时，
        群会话 ID 会被改写成 `发送者_群号`（如 default:GroupMessage:1402704406_1053676331），
        尾段整段是 "1402704406_1053676331" —— 只比整段的话，名单里写裸群号
        1053676331 永远匹配不上。所以还要：① 直接比对事件的群号；
        ② 把尾段按分隔符拆开逐段比对。
        """
        if not entries:
            return False
        if umo and umo in entries:
            return True
        if sender and sender in entries:
            return True
        if group and group in entries:
            return True
        if not umo:
            return False
        tail = umo.rsplit(":", 1)[-1]
        if not tail:
            return False
        if tail in entries:
            return True
        for seg in _SEG_SPLIT.split(tail):
            if seg and seg in entries:
                return True
        return False

    def _qc_allowed(self, umo: str, sender: str, group: str = "") -> bool:
        """黑白名单闸门：黑名单优先；白名单模式下只检名单内。"""
        if self._hit(self.blacklist, umo, sender, group):
            return False
        if self.whitelist_mode and not self._hit(self.whitelist, umo, sender, group):
            return False
        return True

    @staticmethod
    def _group_id(event: AstrMessageEvent) -> str:
        """取群号（私聊或不支持的平台返回空串），名单里写群号时用它精确命中。"""
        try:
            getter = getattr(event, "get_group_id", None)
            gid = getter() if callable(getter) else ""
        except Exception:  # noqa: BLE001 - 个别平台没有该方法
            return ""
        return str(gid or "").strip()

    def _lock_for(self, umo: str) -> asyncio.Lock:
        lock = self._locks.get(umo)
        if lock is None:
            if len(self._locks) > _LOCK_CAP:
                self._locks.clear()
            lock = asyncio.Lock()
            self._locks[umo] = lock
        return lock

    async def _resolve_provider(self, umo: str) -> Any:
        """检测/改写用的模型：custom 指定优先，否则跟随当前会话。"""
        if self.model_mode == "custom" and self.provider_id:
            try:
                provider = self.context.get_provider_by_id(provider_id=self.provider_id)
                if provider is not None:
                    return provider
            except Exception:  # noqa: BLE001
                pass
        try:
            return await self.context.get_using_provider_async(umo=umo)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _plain_text(chain: list[Any]) -> str:
        pieces = [
            comp.text for comp in chain
            if type(comp).__name__ == "Plain"
            and isinstance(getattr(comp, "text", None), str)
        ]
        return "\n".join(pieces).strip()

    @staticmethod
    def _apply_text(chain: list[Any], slots: list[int], text: str) -> bool:
        """把改写结果写回 chain：首个 Plain 承载全文，其余清空。"""
        if not slots or not text:
            return False
        try:
            chain[slots[0]].text = text
            for index in slots[1:]:
                chain[index].text = ""
            return True
        except Exception:  # noqa: BLE001 - 改不动就发原稿，绝不丢消息
            return False

    def _names_of(self, ids: list[str]) -> str:
        """检测项 id → 人类可读名称串（本地公式修复的下轮反馈用）。"""
        names: list[str] = []
        for item_id in ids:
            item = self.items.get(item_id)
            names.append(item.name if item is not None else str(item_id))
        return "、".join(dict.fromkeys(names))

    def _dedup_append(self, target: list[Violation], extra: list[Violation]) -> None:
        seen = {v.id for v in target}
        for violation in extra:
            if violation.id in seen:
                continue
            item = self.items.get(violation.id)
            if item is not None:
                violation.name = item.name
            target.append(violation)
            seen.add(violation.id)

    # ------------------------------------------------------------------
    # 钩子 1：LLM 请求前 —— 打标 + 下轮反馈注入
    # ------------------------------------------------------------------
    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        try:
            self._assert_context_send()          # 每次请求顺带把 send_message 包装补回来（分段插件可能后包）
            self._mark_llm(event)
            if not (self.enabled and self.feedback_enabled):
                return
            umo = event.unified_msg_origin
            if not self._qc_allowed(umo, self._sender_id(event), self._group_id(event)):
                self._feedback.pop(umo, None)   # 免检会话：上一轮的陈旧反馈直接丢掉
                return
            reason = self._feedback.pop(umo, None)
            if not reason:
                return
            current = req.system_prompt or ""
            if FEEDBACK_MARKER in current:
                return
            block = (
                f"{FEEDBACK_MARKER}你上一条回复被质检打回，系统已代为改写。"
                f"原因：{reason}。这一轮说话请注意别再出现同样的问题。"
            )
            req.system_prompt = (current + "\n\n" + block).strip() if current else block
            if self.debug_log:
                logger.debug(f"[人味守卫] 下轮反馈已注入 {umo}：{reason}")
        except Exception as exc:  # noqa: BLE001 - 钩子异常绝不能打断对话
            logger.error(f"[人味守卫] on_llm_request 出错：{exc}", exc_info=True)

    # ------------------------------------------------------------------
    # 钩子 2：消息发出前 —— 质检主管线
    # ------------------------------------------------------------------
    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent) -> None:
        try:
            await self._qc_pipeline(event)
        except Exception as exc:  # noqa: BLE001 - 管线炸了必须原稿照发
            logger.error(f"[人味守卫] 质检管线异常（原稿照发）：{exc}", exc_info=True)

    async def _qc_pipeline(self, event: AstrMessageEvent) -> None:
        if not self.enabled:
            return
        self._assert_context_send()
        if not self._consume_mark(self._event_key(event)):
            return                      # 非 LLM 轮次（指令/面板回复）不质检
        umo = event.unified_msg_origin
        if not self._qc_allowed(umo, self._sender_id(event), self._group_id(event)):
            return                      # 黑白名单放行
        try:
            if event.is_stopped():
                return
        except Exception:  # noqa: BLE001
            pass
        try:
            result = event.get_result()
        except Exception:  # noqa: BLE001
            return
        chain = getattr(result, "chain", None) or []
        slots = [
            index for index, comp in enumerate(chain)
            if type(comp).__name__ == "Plain"
            and isinstance(getattr(comp, "text", None), str)
            and comp.text.strip()
        ]
        if not slots:
            return
        original = "\n".join(chain[i].text for i in slots).strip()
        user_prev = strip_message_markup(event.message_str or "").strip()

        if len(original) < self.skip_below:
            # 短消息没什么可检的，但要入史供下一句比对（也记一下，便于识别分段碎片）
            self._remember_seen(umo, original)
            self.history.record(umo, original, user_prev)
            return

        async with self._lock_for(umo):
            await self._qc_chain(chain, slots, original, user_prev, umo)

    async def _qc_chain(
        self,
        chain: list[Any],
        slots: list[int],
        original: str,
        user_prev: str,
        umo: str,
    ) -> None:
        """质检主管线：被动回复（装饰钩子）与主动消息（send_by_session / send_message）共用。

        顺序：规则层 → **本地公式修复层**（零调用）→ AI 检测层 → 打回改写（必要时）。
        本地能用公式改好的，一次模型调用都不花；改不掉的才落到 AI 改写。
        """
        self._remember_seen(umo, original)
        ctx = self.history.context(umo)
        if user_prev:
            ctx.user_prev = user_prev      # 复述检测与 AI 参考都用"正在回应的这句"
        bare, tags = strip_tags(original, self.tag_patterns)
        original_bare = bare

        provider: Any = None
        violations: list[Violation] = []
        detected: list[Violation] = []      # 本地公式动手前的完整命中（日志/体检模式如实记录）
        local_fixed: list[str] = []
        # 体检模式只看不改、规则层关掉时也不动手：日志里如实记检测结果
        allow_local = self.fast_fix and self.enable_rules and not self.dry_run
        rule_items = self.items.enabled_rule_items()

        # ---------- 第一层：规则 ----------
        if self.enable_rules:
            violations = run_rules(bare, rule_items, ctx)
            detected = list(violations)

        # ---------- 第二层：本地公式修复（确定性修补，0 次模型调用）----------
        if violations and allow_local:
            outcome = local_repair(bare, violations, rule_items, ctx)
            if outcome.changed and outcome.text != bare:
                bare = outcome.text
                violations = outcome.remaining
                local_fixed = list(outcome.fixed)
                if self.debug_log:
                    logger.debug(
                        f"[人味守卫] 本地公式修复 {umo}：{'、'.join(outcome.steps)}"
                        f"（剩 {len(violations)} 项）"
                    )

        # ---------- 第三层：AI 检测 ----------
        if self.enable_judge and (not violations or self.judge_on_rule_fail):
            provider = await self._resolve_provider(umo)
            judge = await run_judge(
                provider, self.items.enabled_judge_items(), bare, ctx, self.model_timeout
            )
            if judge.error:
                if self.debug_log:
                    logger.debug(f"[人味守卫] AI 检测未完成（不影响放行）：{judge.error}")
            else:
                self._dedup_append(violations, judge.violations)
                detected = list(violations)
                # AI 层也可能点名规则项（比如句末句号）→ 再给它一次本地修好的机会
                if violations and allow_local:
                    again = local_repair(bare, violations, rule_items, ctx)
                    if again.changed and again.text != bare:
                        bare = again.text
                        violations = again.remaining
                        local_fixed = list(dict.fromkeys(local_fixed + again.fixed))

        if self.debug_log:
            summary = "、".join(f"{v.id}:{v.reason}" for v in violations) or "通过"
            logger.debug(f"[人味守卫] 检测 {umo}（{len(bare)} 字）→ {summary}")

        # ---------- 通过：写回（若本地修过）+ 入史放行 ----------
        if not violations:
            sent_text = original
            if bare != original_bare:
                sent_text = rejoin_tags(bare, tags)     # 修好后把标签原样加回
                if not self._apply_text(chain, slots, sent_text):
                    sent_text, bare = original, original_bare   # 写不回就发原稿
            if local_fixed:
                if self.debug_log:
                    logger.debug(f"[人味守卫] 本地修好了 {'、'.join(local_fixed)}，未调用模型")
                self.log.add(
                    umo=umo, action="local", original=original, final=sent_text,
                    violations=[v.to_dict() for v in detected],
                    extra={"fixed": "、".join(local_fixed)},
                )
                if self.feedback_enabled:
                    # 本地替你改了，也要让模型下一轮知道问题出在哪
                    self._feedback[umo] = self._names_of(local_fixed)
            elif self.debug_log:
                self.log.add(umo=umo, action="pass", original=original, final=sent_text)
            self._remember_seen(umo, sent_text)
            self.history.record(umo, bare, user_prev or None)
            return

        # ---------- 打回 ----------
        reject_names = "、".join(dict.fromkeys(v.name for v in violations))
        if self.dry_run:
            self.log.add(
                umo=umo, action="dry",
                violations=[v.to_dict() for v in detected],
                original=original, final=original,
            )
            self._remember_seen(umo, original)
            self.history.record(umo, original_bare, user_prev or None)
            return

        outcome = await self._rewrite_loop(umo, provider, bare, violations)
        final_bare = outcome.text
        passed = outcome.passed
        final_violations = outcome.violations
        rounds = outcome.rounds
        error = outcome.error
        local_fixed = list(dict.fromkeys(local_fixed + outcome.local))

        # ---------- 写回 / 记账 ----------
        sent_text = original
        if final_bare != original_bare:
            sent_text = rejoin_tags(final_bare, tags)   # 改写后把标签加回去
            if not self._apply_text(chain, slots, sent_text):
                sent_text, final_bare = original, original_bare   # 写不回就发原稿

        if passed and rounds == 0 and local_fixed:
            action = "local"                            # 纯本地公式修好，没花模型调用
        else:
            action = "rewrite" if passed else "give_up"
        extra: dict[str, Any] = {}
        if local_fixed:
            extra["fixed"] = "、".join(local_fixed)
        if not passed and final_violations:
            extra["unresolved"] = "、".join(
                dict.fromkeys(v.name for v in final_violations)
            )
        self.log.add(
            umo=umo, action=action,
            violations=[v.to_dict() for v in detected],
            original=original, final=sent_text,
            rounds=rounds, error=error, extra=extra,
        )
        self._remember_seen(umo, sent_text)
        self.history.record(umo, final_bare, user_prev or None)
        if self.feedback_enabled:
            self._feedback[umo] = reject_names
        if self.debug_log:
            state = "复检通过" if passed else ("改写失败，发原稿" if final_bare == original_bare else "到轮数上限，发最后一版")
            logger.debug(
                f"[人味守卫] 打回 {reject_names} → {rounds} 轮改写，{state}"
                + (f"；本地已修 {'、'.join(local_fixed)}" if local_fixed else "")
                + (f"（{error}）" if error else "")
            )

    # ------------------------------------------------------------------
    # 主动消息拦截：包装各平台实例的 send_by_session
    # ------------------------------------------------------------------
    # 主动消息（proactive_chat 的兜底/直发、其他插件的 context.send_message）
    # 走 platform.send_by_session 直发，不创建事件、不过 pipeline，
    # on_decorating_result 钩子根本不会触发。这里是所有主动发送的必经关口：
    # 发送前把 chain 里的文本抽出来过同一套质检，命中就打回改写，改完再发。
    # 设计要点：幂等（重复调用只包一层）、可还原（terminate 拆包）、
    # fail-open（包装内任何异常都放行原消息，绝不丢消息）。

    def _wrap_platform_sends(self) -> None:
        """给当前所有平台实例的 send_by_session 套上质检包装。"""
        try:
            insts = list(self.context.platform_manager.platform_insts)
        except Exception as exc:  # noqa: BLE001 - 拿不到平台列表就放弃，不影响主流程
            logger.warning(f"[人味守卫] 无法枚举平台实例，主动消息拦截未启用：{exc}")
            return
        for inst in insts:
            self._wrap_one_platform(inst)

    def _wrap_one_platform(self, inst: Any) -> None:
        """包装单个平台实例（幂等；不碰已经包过的实例）。"""
        try:
            key = id(inst)
            if key in self._wrapped:
                return
            orig = getattr(inst, "send_by_session", None)
            if not callable(orig):
                return
            plugin = self
            inst_name = self._platform_name(inst)

            async def guarded_send(session: Any, message_chain: Any, *_a: Any, **_kw: Any):
                """send_by_session 的质检替身：先质检改写，再交回原实现发送。"""
                # 防重入（任务级隔离）：整条消息已经质检过时，内层的分段发送直接透传
                if plugin._send_guard.get():
                    return await orig(session, message_chain, *_a, **_kw)
                token = plugin._send_guard.set(True)
                try:
                    try:
                        await plugin._qc_proactive(inst_name, session, message_chain)
                    except Exception as exc:  # noqa: BLE001 - 质检炸了原样照发
                        logger.error(f"[人味守卫] 主动消息质检异常（原样照发）：{exc}")
                    return await orig(session, message_chain, *_a, **_kw)
                finally:
                    plugin._send_guard.reset(token)

            # 注意不要用 __get__ 绑定：那会把实例再塞回第一个参数，
            # 实际签名就成了 (self, session, chain)，与基类/send 调用约定不符。
            inst.send_by_session = guarded_send
            self._wrapped[key] = {"inst": inst, "orig": orig, "name": inst_name}
            if self.debug_log:
                logger.debug(f"[人味守卫] 已拦截平台主动发送：{inst_name}")
        except Exception as exc:  # noqa: BLE001 - 单个平台失败不影响其他平台
            logger.warning(f"[人味守卫] 包装平台 send_by_session 失败：{exc}")

    def _platform_name(self, inst: Any) -> str:
        try:
            return str(inst.meta().id or inst.meta().name)
        except Exception:  # noqa: BLE001
            return "unknown"

    def _unwrap_platform_sends(self) -> None:
        """卸载：把所有包装还原成原始实现。"""
        for info in self._wrapped.values():
            try:
                info["inst"].send_by_session = info["orig"]
            except Exception:  # noqa: BLE001
                pass
        self._wrapped.clear()

    async def _wrap_platforms_loop(self) -> None:
        """轮询补挂：新接入的平台也会被包上（挂丢了也能补回来）。"""
        while True:
            try:
                await asyncio.sleep(30)
                if not self.enabled:
                    continue
                try:
                    insts = list(self.context.platform_manager.platform_insts)
                except Exception:  # noqa: BLE001
                    continue
                for inst in insts:
                    if id(inst) not in self._wrapped:
                        self._wrap_one_platform(inst)
                self._assert_context_send()
            except asyncio.CancelledError:
                return
            except Exception as exc:  # noqa: BLE001 - 轮询永不炸
                logger.debug(f"[人味守卫] 补挂轮询异常：{exc}")

    # ------------------------------------------------------------------
    # 分段插件（对话分段Pro 等）兼容：整条质检 + 碎片不重复质检
    # ------------------------------------------------------------------
    # 分段插件会把一条回复切段，而且经常直接调 Context.send_message /
    # platform.send_by_session 把每一段发出去（不走装饰钩子）。于是人味守卫
    # 只能看到"半句话"：句末句号、句中缺标点、与上一句重合这些依赖整句的
    # 检测全部失真，历史里还会被灌进一堆碎片；每段还各跑一遍 AI，慢得离谱。
    # 这里两个动作解决它：
    #   ① 把 Context.send_message 也包上：主动消息先按"整条"质检，再交给分段插件切；
    #   ② 记住刚质检过的整条消息：分段发出去的碎片（整条的子串）直接放行，
    #      既不重复质检、也不写历史 —— 一次回复最多一次质检、一次 AI 调用。

    @staticmethod
    def _norm_text(text: str) -> str:
        return _WS_RE.sub("", str(text or ""))

    def _remember_seen(self, umo: str, text: str) -> None:
        """记下"刚质检过的整条消息"，供碎片识别用。"""
        if not (self.dedup_fragments and umo):
            return
        norm = self._norm_text(text)
        if not norm:
            return
        key = str(umo)
        bucket = [pair for pair in self._seen_texts.get(key, []) if pair[0] != norm]
        bucket.insert(0, (norm, time.time()))
        self._seen_texts[key] = bucket[:_SEEN_KEEP]
        self._seen_texts.move_to_end(key)
        while len(self._seen_texts) > _SEEN_SESSIONS:
            self._seen_texts.popitem(last=False)

    def _is_fragment(self, umo: str, text: str) -> bool:
        """这条文本是不是"刚质检过的整条消息"被切出来的一段。"""
        if not (self.dedup_fragments and umo):
            return False
        norm = self._norm_text(text)
        if not norm:
            return False
        now = time.time()
        hit = False
        keep: list[tuple[str, float]] = []
        for prev, stamp in self._seen_texts.get(str(umo), []):
            if now - stamp > self.dedup_window:
                continue
            keep.append((prev, stamp))
            if hit:
                continue
            if norm == prev:
                hit = True                                   # 同一条重复投递
            elif len(norm) >= 2 and len(norm) < len(prev) and norm in prev:
                hit = True                                   # 是整条的一部分 → 分段碎片
        if keep:
            self._seen_texts[str(umo)] = keep
        return hit

    def _assert_context_send(self) -> None:
        """把 Context.send_message 包上（幂等、可还原）：主动消息先按整条质检。

        分段插件也是在这一层做劫持，谁后包谁在最外层。我们每次请求都补一次，
        保证自己包在最外层：先按整条质检，再由它切分发送。
        """
        if not (self.enabled and self.wrap_context_send):
            return
        try:
            ctx = self.context
            current = getattr(ctx, "send_message", None)
        except Exception:  # noqa: BLE001
            return
        if not callable(current) or getattr(current, "_humanize_guard_wrapper", False):
            return
        plugin = self
        inner = current

        async def guarded_send_message(session: Any, message_chain: Any, *_a: Any, **_kw: Any):
            if plugin._send_guard.get():
                return await inner(session, message_chain, *_a, **_kw)
            token = plugin._send_guard.set(True)
            try:
                try:
                    await plugin._qc_proactive("context", session, message_chain)
                except Exception as exc:  # noqa: BLE001 - 质检炸了原样照发
                    logger.error(f"[人味守卫] 主动消息质检异常（原样照发）：{exc}")
                return await inner(session, message_chain, *_a, **_kw)
            finally:
                plugin._send_guard.reset(token)

        guarded_send_message._humanize_guard_wrapper = True  # type: ignore[attr-defined]
        try:
            ctx.send_message = guarded_send_message
            self._ctx_send_inner = inner
            if not self._ctx_send_wrapped:
                logger.info(
                    "[人味守卫] 已接管 Context.send_message：主动消息先按整条质检，再交给分段插件切分"
                )
            self._ctx_send_wrapped = True
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[人味守卫] 接管 Context.send_message 失败：{exc}")

    def _unwrap_context_send(self) -> None:
        """卸载：还原 Context.send_message（只还原是我们包的那一次）。"""
        try:
            ctx = self.context
            current = getattr(ctx, "send_message", None)
            if callable(current) and getattr(current, "_humanize_guard_wrapper", False):
                ctx.send_message = self._ctx_send_inner
        except Exception:  # noqa: BLE001
            pass
        self._ctx_send_wrapped = False

    async def _qc_proactive(self, platform_id: str, session: Any, message_chain: Any) -> None:
        """主动消息质检：把 message_chain 里的 Plain 文本过一遍质检管线，命中就地改写。"""
        if not self.enabled:
            return
        if isinstance(session, str):
            umo = session
        else:
            umo = str(session)
        sender = ""
        group = ""
        try:
            parsed = MessageSesion.from_str(umo) if isinstance(session, str) else session
            group = str(getattr(parsed, "session_id", "") or "").rsplit("_", 1)[-1]
            # 唯一会话模式：群会话 session_id 形如 发送者_群号
            if "_" in str(getattr(parsed, "session_id", "")):
                sender = str(getattr(parsed, "session_id", "")).split("_", 1)[0]
        except Exception:  # noqa: BLE001
            pass
        if not self._qc_allowed(umo, sender, group):
            return

        chain = getattr(message_chain, "chain", None)
        if not isinstance(chain, list) or not chain:
            return
        slots = [
            index for index, comp in enumerate(chain)
            if type(comp).__name__ == "Plain"
            and isinstance(getattr(comp, "text", None), str)
            and comp.text.strip()
        ]
        if not slots:
            return
        original = "\n".join(chain[i].text for i in slots).strip()
        if not original:
            return

        # ① 分段碎片：整条刚刚质检过 → 不再逐段重复质检（"多传两三次 AI"的主因之一）
        if self._is_fragment(umo, original):
            if self.debug_log:
                logger.debug(f"[人味守卫] 跳过分段碎片（已按整条质检）：{original[:24]}")
            return

        # ② 防重复质检：proactive_chat 会先手动触发 decorating 钩子再直发。
        #    钩子路径若已质检并记录（几秒内），这里比对"上一条已发送"跳过，
        #    避免同一条消息被质检两遍；带时间窗是为了不误伤之后真正有意重发的内容。
        recent = self.history.get(umo)
        if (
            recent["prev"]
            and recent["prev"] == original
            and (time.time() - recent["ts"]) < 30.0
        ):
            self._remember_seen(umo, original)
            return

        if len(original) < self.skip_below:
            self._remember_seen(umo, original)
            self.history.record(umo, original)
            return

        async with self._lock_for(umo):
            # 与装饰钩子完全同一条管线（规则 → 本地公式 → AI 检测 → 打回改写）
            await self._qc_chain(chain, slots, original, "", umo)


    async def _rewrite_loop(
        self,
        umo: str,
        provider: Any,
        bare: str,
        violations: list[Violation],
    ) -> RewriteOutcome:
        """打回改写循环（被动回复与主动消息共用）。

        每一轮都**先给本地公式一次机会**：
        - AI 层点名的规则项（句末句号、连发标点…）本地就能改好，直接省掉这一轮模型调用；
        - 本地改完就干净了 → 直接通过，rounds 不加（日志记为「本地公式修复」）；
        - 本地改不掉（重合、复述、整体 AI 味…）→ 才真的调模型改写。
        改写稿回来后再过一遍本地公式，再复检；复检默认只看规则层（recheck_judge=False），
        省掉每轮一次 AI 检测调用。
        """
        final_bare = str(bare or "")
        final_violations = list(violations)
        rounds = 0
        error = ""
        passed = False
        local_steps: list[str] = []
        rule_items = self.items.enabled_rule_items()

        while True:
            ctx = self.history.context(umo)
            # ---- 本地公式优先 ----
            if self.fast_fix and self.enable_rules and final_violations:
                outcome = local_repair(final_bare, final_violations, rule_items, ctx)
                if outcome.changed and outcome.text != final_bare:
                    final_bare = outcome.text
                    final_violations = outcome.remaining
                    local_steps.extend(outcome.fixed)
                    if not final_violations:
                        passed = True
                        break

            if rounds >= self.max_rounds:
                break
            if provider is None:
                provider = await self._resolve_provider(umo)
            hint_ids = list(dict.fromkeys(v.id for v in final_violations))
            rewritten, err = await run_rewrite(
                provider, final_bare, final_violations,
                self.items.rewrite_hints(hint_ids), ctx, self.model_timeout,
            )
            if err or not rewritten:
                error = err or "改写返回空"
                break
            rounds += 1
            candidate, _ = strip_tags(rewritten, self.tag_patterns)
            candidate = candidate.strip()
            if not candidate:
                error = "改写结果清洗后为空"
                break
            # 改写稿也先过本地公式：常常能省下这一轮的复检与下一轮改写
            if self.fast_fix and self.enable_rules:
                polished = local_repair(candidate, run_rules(candidate, rule_items, ctx), rule_items, ctx)
                if polished.changed and polished.text != candidate:
                    candidate = polished.text
                    local_steps.extend(polished.fixed)
            checked: list[Violation] = []
            if self.enable_rules:
                checked = run_rules(candidate, rule_items, ctx)
            if self.enable_judge and self.recheck_judge and (not checked or self.judge_on_rule_fail):
                judge2 = await run_judge(
                    provider, self.items.enabled_judge_items(), candidate, ctx, self.model_timeout
                )
                if not judge2.error:
                    self._dedup_append(checked, judge2.violations)
            final_bare = candidate
            if not checked:
                passed = True
                final_violations = []
                break
            final_violations = checked

        return RewriteOutcome(
            text=final_bare,
            passed=passed,
            violations=final_violations,
            rounds=rounds,
            error=error,
            local=list(dict.fromkeys(local_steps)),
        )

    # ------------------------------------------------------------------
    # 试跑接口（面板 / 指令共用）
    # ------------------------------------------------------------------
    async def qc_detect(self, text: str, umo: str = "") -> dict[str, Any]:
        ctx = self.history.context(umo) if umo else RuleContext()
        bare, tags = strip_tags(text, self.tag_patterns)
        violations: list[Violation] = []
        if self.enable_rules:
            violations = run_rules(bare, self.items.enabled_rule_items(), ctx)
        judge_info: dict[str, Any] = {
            "ran": False, "error": "", "system": "", "prompt": "", "raw": "",
        }
        if self.enable_judge:
            provider = await self._resolve_provider(umo)
            judge = await run_judge(
                provider, self.items.enabled_judge_items(), bare, ctx, self.model_timeout
            )
            judge_info = {
                "ran": True, "error": judge.error,
                "system": judge.system, "prompt": judge.prompt, "raw": judge.raw,
            }
            if not judge.error:
                self._dedup_append(violations, judge.violations)
        return {
            "bare": bare,
            "tags": tags,
            "violations": [v.to_dict() for v in violations],
            "passed": not violations,
            "judge": judge_info,
            "rules_enabled": self.enable_rules,
            "judge_enabled": self.enable_judge,
            # 本地公式能直接修好的命中项：这些不必调模型改写
            "local_fixable": [v.id for v in violations if can_fix(v.id)],
            "fast_fix": self.fast_fix,
        }

    async def qc_rewrite_once(self, text: str, umo: str = "") -> dict[str, Any]:
        """试跑一次"检测 → 本地公式修复 →（必要时）AI 改写"。

        本地公式能修好的，就直接给本地结果，一次模型调用都不花。
        """
        detected = await self.qc_detect(text, umo)
        out: dict[str, Any] = {
            "detect": detected, "rewritten": "", "error": "", "recheck": None,
            "local": [], "used_ai": False,
        }
        if detected["passed"]:
            return out
        ctx = self.history.context(umo) if umo else RuleContext()
        violations = [
            Violation(
                id=str(v.get("id") or ""),
                name=str(v.get("name") or ""),
                reason=str(v.get("reason") or ""),
                engine=str(v.get("engine") or "rule"),
            )
            for v in detected["violations"]
        ]
        bare = detected["bare"]
        if self.fast_fix and self.enable_rules:
            outcome = local_repair(bare, violations, self.items.enabled_rule_items(), ctx)
            if outcome.changed and outcome.text != bare:
                bare = outcome.text
                violations = outcome.remaining
                out["local"] = outcome.fixed
                if not violations:
                    out["rewritten"] = bare
                    out["recheck"] = await self.qc_detect(bare, umo)
                    return out
        if not violations:
            out["rewritten"] = bare
            out["recheck"] = await self.qc_detect(bare, umo)
            return out
        provider = await self._resolve_provider(umo)
        hint_ids = list(dict.fromkeys(v.id for v in violations))
        rewritten, err = await run_rewrite(
            provider, bare, violations,
            self.items.rewrite_hints(hint_ids), ctx, self.model_timeout,
        )
        out["rewritten"] = rewritten
        out["error"] = err
        out["used_ai"] = True
        if rewritten:
            out["recheck"] = await self.qc_detect(rewritten, umo)
        return out

    # ------------------------------------------------------------------
    # 指令
    # ------------------------------------------------------------------
    @filter.command("qc", alias={"质检", "人味质检"})
    async def cmd_qc(self, event: AstrMessageEvent):
        """人味守卫：/qc 状态 | 测试 <文本> | 日志 [n] | 帮助"""
        raw = (event.message_str or "").strip()
        parts = raw.split(maxsplit=1)
        body = parts[1].strip() if len(parts) > 1 else ""
        head, _, arg = body.partition(" ")
        head = head.strip().lower()
        try:
            if head in ("", "状态", "status", "st"):
                text = self._status_text()
            elif head in ("测试", "test", "t"):
                if not arg.strip():
                    text = "用法：/qc 测试 <要检测的文本>"
                else:
                    text = await self._test_text(event, arg.strip())
            elif head in ("日志", "log", "l"):
                text = self._log_text(arg.strip())
            elif head in ("帮助", "help", "?", "？"):
                text = HELP_TEXT
            else:
                text = f"未知子命令：{head}\n\n{HELP_TEXT}"
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[人味守卫] 指令出错：{exc}", exc_info=True)
            text = f"质检指令出错了：{exc}"
        yield event.plain_result(text)

    def _status_text(self) -> str:
        stats = self.items.stats()
        lines = [
            "🛡 人味守卫 · 状态",
            f"总开关：{'开' if self.enabled else '关'}"
            f"｜规则层：{'开' if self.enable_rules else '关'}"
            f"｜AI 层：{'开' if self.enable_judge else '关'}"
            + ("｜体检模式" if self.dry_run else ""),
            f"检测项：启用 {stats['enabled']}/{stats['total']}"
            f"（AI 注入词 {stats['judge_enabled']} 项，自定义 {stats['custom']} 项）",
            f"改写：最多 {self.max_rounds} 轮｜模型：{'会话模型' if self.model_mode == 'session' else self.provider_id or '会话模型'}"
            f"｜超时 {self.model_timeout:g}s",
            f"历史：{self.history.stats()['sessions']} 个会话（近 {self.history.keep_k} 条）"
            f"｜打回日志：{self.log.stats()['total']} 条",
            f"下轮反馈注入：{'开' if self.feedback_enabled else '关'}"
            f"｜待注入 {len(self._feedback)} 个会话",
        ]
        mode = (
            f"白名单模式开：只检 {len(self.whitelist)} 个"
            if self.whitelist_mode else f"白名单模式关：全检（名单 {len(self.whitelist)} 个未启用）"
        )
        lines.append(
            f"提速：本地公式修复 {'开' if self.fast_fix else '关'}"
            f"｜改写后 AI 复检 {'开' if self.recheck_judge else '关（只跑规则层）'}"
            f"｜分段碎片去重 {'开' if self.dedup_fragments else '关'}"
        )
        lines.append(
            f"分段兼容：接管 Context.send_message {'开' if (self.wrap_context_send and self._ctx_send_wrapped) else ('开（待接管）' if self.wrap_context_send else '关')}"
            f"｜刚质检过的整条消息缓存 {len(self._seen_texts)} 个会话"
        )
        lines.append(f"名单：黑名单 {len(self.blacklist)} 个｜{mode}")
        lines.append(
            "名单条目可写：完整会话 ID / 群号 / 发送者 ID"
            "（唯一会话模式的「发送者_群号」自动拆段，裸群号也能命中）"
        )
        return "\n".join(lines)

    async def _test_text(self, event: AstrMessageEvent, arg: str) -> str:
        umo = event.unified_msg_origin
        result = await self.qc_detect(arg, umo)
        if result["passed"]:
            head = "✅ 通过：没有命中任何启用的检测项"
        else:
            head = "🚫 命中："
        lines = [head]
        for v in result["violations"]:
            mark = "（本地公式可修）" if v["id"] in result.get("local_fixable", []) else ""
            lines.append(f"- [{v['id']}] {v['name']}：{v['reason']}{mark}")
        if result["tags"]:
            lines.append("（检测时已剥离标签：" + " ".join(result["tags"]) + "）")
        judge = result["judge"]
        if judge["ran"]:
            if judge["error"]:
                lines.append(f"AI 层未完成：{judge['error']}")
            else:
                lines.append(f"AI 层已裁决，原始输出：{judge['raw'][:200]}")
        elif not result["judge_enabled"]:
            lines.append("（AI 层已关闭，仅规则层）")
        return "\n".join(lines)

    def _log_text(self, arg: str) -> str:
        try:
            limit = int(arg) if arg else 5
        except ValueError:
            limit = 5
        entries = self.log.recent(limit)
        if not entries:
            return "还没有打回记录。"
        lines = [f"📜 最近 {len(entries)} 条质检记录："]
        label = {"rewrite": "已改写", "give_up": "改写未达标(发末版)", "dry": "体检命中",
                 "local": "本地公式修复(未调AI)", "pass": "通过", "error": "异常"}
        for entry in entries:
            when = time.strftime("%m-%d %H:%M", time.localtime(entry.get("ts") or 0))
            action = label.get(entry.get("action"), entry.get("action"))
            names = "、".join(v.get("name", "") for v in entry.get("violations", [])) or "-"
            lines.append(f"[{when}] {action}｜{names}｜{entry.get('original', '')[:40]}")
        return "\n".join(lines)
