"""端到端冒烟：用"假的 astrbot 运行时"真实加载 main.py 并驱动两条钩子。

覆盖：LLM 打标、拦截、规则检测、打回改写、标签加回、复检兼容（草稿不入史）、
轮数上限兜底、下轮反馈注入、体检模式、白名单、指令、面板后端接口。
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import re
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from typing import Any

PLUGIN_DIR = Path(__file__).resolve().parent.parent
WORKSPACE = PLUGIN_DIR.parent
UMO = "aiocqhttp:GroupMessage:66666"
FENCE = chr(96) * 3

REGISTRY: dict[str, list[Any]] = {}

# ---------------------------------------------------------------------------
# 假的 AstrBot 运行时
# ---------------------------------------------------------------------------

DATA_ROOT = {"path": Path(tempfile.mkdtemp())}


class Plain:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeResult:
    def __init__(self, chain: list[Any]) -> None:
        self.chain = chain


class FakeReq:
    def __init__(self, system_prompt: str = "原有系统提示") -> None:
        self.system_prompt = system_prompt


class FakeEvent:
    def __init__(self, message: str = "", umo: str = UMO, mid: str | None = None,
                 uid: str = "10001", gid: str = "") -> None:
        self.message_str = message
        self.unified_msg_origin = umo
        self.message_obj = types.SimpleNamespace(message_id=mid or f"m{time.time_ns()}")
        self._uid = uid
        self._gid = gid
        self._result: Any = None
        self.stopped = False

    def get_sender_id(self) -> str:
        return self._uid

    def get_group_id(self) -> str:
        return self._gid

    def set_reply(self, text: str, *others: str) -> None:
        self._result = FakeResult([Plain(text), *[Plain(o) for o in others]])

    def get_result(self) -> Any:
        return self._result

    def is_stopped(self) -> bool:
        return self.stopped

    def plain_result(self, text: str) -> str:
        return text


class NoGroupEvent(FakeEvent):
    """模拟不提供 get_group_id 的平台（名单匹配只能靠会话 ID 分段）。"""

    def get_group_id(self):  # noqa: ANN201 - 故意没有返回类型
        raise AttributeError("该平台没有 get_group_id")


class BrokenEvent(FakeEvent):
    def get_result(self) -> Any:
        raise RuntimeError("结果读不出来")


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.completion_text = text


class FakeProvider:
    def __init__(self, script: list[Any] | None = None) -> None:
        self.script = list(script or [])
        self.calls: list[dict[str, Any]] = []

    async def text_chat(self, prompt: str | None = None, system_prompt: str | None = None, **_kw: Any):
        self.provider_last = None
        self.calls.append({"prompt": prompt or "", "system_prompt": system_prompt or ""})
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return FakeResponse(item)
        return FakeResponse("")


class FakeContext:
    def __init__(self, provider: Any = None) -> None:
        self.provider = provider
        self.routes: dict[str, Any] = {}

    def get_provider_by_id(self, provider_id: str | None = None) -> Any:
        return self.provider

    async def get_using_provider_async(self, umo: str | None = None) -> Any:
        return self.provider

    def register_web_api(self, path: str, handler: Any, methods: Any = None, desc: str = "") -> None:
        self.routes[path] = (handler, methods)


class FakePlatform:
    """假的平台适配器实例：send_by_session 记录调用。"""

    def __init__(self, pid: str = "wechatai") -> None:
        self.pid = pid
        self.sent: list[tuple[Any, Any]] = []

    def meta(self) -> Any:
        return types.SimpleNamespace(id=self.pid, name=self.pid)

    async def send_by_session(self, session: Any, message_chain: Any) -> None:
        self.sent.append((session, message_chain))


class FakeWebRequest:
    def __init__(self) -> None:
        self.query: dict[str, Any] = {}
        self.json_body: Any = None
        self.raw_body: bytes = b""

    async def json(self) -> Any:
        if self.json_body is None:
            raise ValueError("no json body")
        return self.json_body

    async def body(self) -> bytes:
        return self.raw_body


WEB_REQUEST = FakeWebRequest()


class SaveableConfig(dict):
    """带 save_config 的配置对象（模拟 AstrBotConfig）。"""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.save_count = 0

    def save_config(self, *a: Any, **kw: Any) -> None:
        self.save_count += 1


def _decorator_factory(name: str):
    def factory(*_args: Any, **_kwargs: Any):
        def decorate(func: Any) -> Any:
            REGISTRY.setdefault(name, []).append(func)
            return func

        return decorate

    return factory


class FakeFilterModule:
    def __getattr__(self, name: str) -> Any:
        return _decorator_factory(name)


def _install_fake_astrbot() -> None:
    astrbot = types.ModuleType("astrbot")
    astrbot.__path__ = []  # type: ignore[attr-defined]
    api = types.ModuleType("astrbot.api")
    api.__path__ = []  # type: ignore[attr-defined]
    api.logger = logging.getLogger("astrbot-fake-test")
    api.logger.setLevel(logging.CRITICAL)

    event_mod = types.ModuleType("astrbot.api.event")
    event_mod.AstrMessageEvent = FakeEvent  # type: ignore[attr-defined]
    event_mod.filter = FakeFilterModule()   # type: ignore[attr-defined]

    provider_mod = types.ModuleType("astrbot.api.provider")
    provider_mod.ProviderRequest = FakeReq  # type: ignore[attr-defined]

    star_mod = types.ModuleType("astrbot.api.star")

    class Star:
        def __init__(self, context: Any, config: Any = None) -> None:
            self.context = context

    class StarTools:
        @staticmethod
        def get_data_dir(name: str) -> str:
            return str(DATA_ROOT["path"] / name)

    def register(*_a: Any, **_k: Any):
        def decorator(cls: Any) -> Any:
            return cls

        return decorator

    star_mod.Context = object          # type: ignore[attr-defined]
    star_mod.Star = Star               # type: ignore[attr-defined]
    star_mod.StarTools = StarTools     # type: ignore[attr-defined]
    star_mod.register = register       # type: ignore[attr-defined]

    web_mod = types.ModuleType("astrbot.api.web")
    web_mod.request = WEB_REQUEST      # type: ignore[attr-defined]

    def json_response(data: Any = None, **_kw: Any) -> Any:
        return {"status": "ok", "data": data}

    def error_response(message: str, **_kw: Any) -> Any:
        return {"status": "error", "message": message}

    web_mod.json_response = json_response    # type: ignore[attr-defined]
    web_mod.error_response = error_response  # type: ignore[attr-defined]

    astrbot.api = api                    # type: ignore[attr-defined]
    api.event = event_mod                # type: ignore[attr-defined]
    api.provider = provider_mod          # type: ignore[attr-defined]
    api.star = star_mod                  # type: ignore[attr-defined]
    api.web = web_mod                    # type: ignore[attr-defined]

    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api
    sys.modules["astrbot.api.event"] = event_mod
    sys.modules["astrbot.api.provider"] = provider_mod
    sys.modules["astrbot.api.star"] = star_mod
    sys.modules["astrbot.api.web"] = web_mod


_install_fake_astrbot()
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

main_mod = importlib.import_module("astrbot_plugin_humanize_guard.main")
S_ItemStore = main_mod.ItemStore

FEEDBACK_MARKER = main_mod.FEEDBACK_MARKER


def make_config(**overrides: Any) -> SaveableConfig:
    cfg: dict[str, Any] = {
        "enabled": True,
        "detect": {
            "enable_rules": True,
            "enable_judge": False,
            "judge_when_rules_fail": False,
            "skip_below": 4,
            "whitelist": "",
        },
        "model": {"mode": "session", "provider_id": "", "timeout_seconds": 5.0},
        "rewrite": {"max_rounds": 2, "dry_run": False},
        "feedback": {"enabled": True},
        "history": {"keep_k": 5, "persist": True},
        "tags": {"patterns": ""},
        "debug": {"debug_log": False},
    }
    for path, value in overrides.items():
        node = cfg
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return SaveableConfig(cfg)


def unwrap(resp: Any) -> Any:
    if isinstance(resp, dict) and resp.get("status") == "ok":
        return resp.get("data")
    return resp


def run(coro: Any) -> Any:
    return asyncio.run(coro)


class SmokeCase(unittest.TestCase):
    """每个用例独立数据目录 + 可编排的假模型。"""

    def setUp(self):
        DATA_ROOT["path"] = Path(tempfile.mkdtemp())
        # 注意：REGISTRY 记录的是导入期的装饰器注册，不能在 setUp 里清空
        WEB_REQUEST.query = {}
        WEB_REQUEST.json_body = None
        WEB_REQUEST.raw_body = b""
        self.provider = FakeProvider()
        self.make_plugin()

    def make_plugin(self, **config_overrides: Any):
        self.config = make_config(**config_overrides)
        self.context = FakeContext(self.provider)
        self.plugin = main_mod.HumanizeGuard(self.context, self.config)
        return self.plugin

    async def drive(self, event: FakeEvent, req: FakeReq | None = None) -> FakeReq:
        req = req or FakeReq()
        await self.plugin.on_llm_request(event, req)
        await self.plugin.on_decorating_result(event)
        return req

    def reply_text(self, event: FakeEvent) -> str:
        return "\n".join(
            c.text for c in event.get_result().chain if type(c).__name__ == "Plain"
        )

    # ------------------------------------------------------------------
    # 钩子与打标
    # ------------------------------------------------------------------
    def test_hooks_registered(self):
        self.assertTrue(REGISTRY.get("on_llm_request"))
        self.assertTrue(REGISTRY.get("on_decorating_result"))
        self.assertTrue(REGISTRY.get("command"))

    def test_non_llm_reply_untouched(self):
        event = FakeEvent()
        event.set_reply("句末有句号。")
        run(self.plugin.on_decorating_result(event))       # 没有先走 on_llm_request
        self.assertEqual(self.reply_text(event), "句末有句号。")
        self.assertEqual(self.plugin.log.stats()["total"], 0)
        self.assertEqual(self.plugin.history.context(UMO).prev, "")

    def test_pipeline_fail_open_on_result_error(self):
        event = BrokenEvent()
        run(self.plugin.on_llm_request(event, FakeReq()))
        event.set_reply("随便吧。")
        run(self.plugin.on_decorating_result(event))       # 绝不抛异常

    # ------------------------------------------------------------------
    # 核心：拦截 → 打回 → 改写 → 标签加回
    # ------------------------------------------------------------------
    def test_clean_reply_passes_untouched(self):
        event = FakeEvent()
        text = "今天天气真不错，我们去公园吧"
        event.set_reply(text)
        run(self.drive(event))
        self.assertEqual(self.reply_text(event), text)
        self.assertEqual(self.plugin.history.context(UMO).prev, text)
        self.assertEqual(self.plugin.log.stats()["total"], 0)
        self.assertEqual(self.provider.calls, [])

    def test_reject_rewrite_and_tags_rejoined(self):
        # 本用例专门覆盖 AI 改写路径：关掉本地公式（默认开，能本地修好的不调模型）
        self.make_plugin(**{"rewrite.fast_fix": False})
        self.provider.script = ["你真棒"]
        event = FakeEvent()
        event.set_reply("你真棒。【好感度+3】")
        run(self.drive(event))

        # 1) 写回的是改写后文本 + 标签加回
        self.assertEqual(self.reply_text(event), "你真棒【好感度+3】")
        # 2) 改写请求里带了对应提示词，且看不到标签（标签不进改写视野）
        self.assertEqual(len(self.provider.calls), 1)
        call = self.provider.calls[0]
        self.assertIn("打回", call["prompt"])
        self.assertIn("删掉句末的句号", call["prompt"])
        self.assertNotIn("好感度", call["prompt"])
        self.assertIn("待改写消息", call["prompt"])
        # 3) 历史只存最终裸文本
        hist = self.plugin.history.context(UMO)
        self.assertEqual(hist.prev, "你真棒")
        # 4) 日志与反馈
        entries = self.plugin.log.recent(5)
        self.assertEqual(entries[0]["action"], "rewrite")
        self.assertEqual(entries[0]["violations"][0]["id"], "end_period")
        self.assertIn("句末有句号", self.plugin._feedback.get(UMO, ""))

    def test_feedback_injected_on_next_llm_request(self):
        self.provider.script = ["你真棒"]
        event1 = FakeEvent()
        event1.set_reply("你真棒。")
        run(self.drive(event1))
        self.assertIn(UMO, self.plugin._feedback)

        event2 = FakeEvent()
        req2 = FakeReq(system_prompt="原有系统提示")
        run(self.plugin.on_llm_request(event2, req2))
        self.assertIn(FEEDBACK_MARKER, req2.system_prompt)
        self.assertIn("原有系统提示", req2.system_prompt)
        self.assertIn("句末有句号", req2.system_prompt)
        self.assertNotIn(UMO, self.plugin._feedback)        # 一次性，用完即焚

        req3 = FakeReq(system_prompt="原有系统提示")
        run(self.plugin.on_llm_request(FakeEvent(), req3))
        self.assertNotIn(FEEDBACK_MARKER, req3.system_prompt)

    def test_dry_run_records_but_not_rewrite(self):
        self.make_plugin(**{"rewrite.dry_run": True})
        self.provider.script = ["不该被调用"]
        event = FakeEvent()
        event.set_reply("你真棒。")
        run(self.drive(event))
        self.assertEqual(self.reply_text(event), "你真棒。")  # 原稿照发
        self.assertEqual(self.provider.calls, [])            # 不改写
        self.assertEqual(self.plugin.log.recent(1)[0]["action"], "dry")

    def test_give_up_after_max_rounds_sends_last_version(self):
        self.make_plugin(**{"rewrite.max_rounds": 1, "rewrite.fast_fix": False})
        self.provider.script = ["还是不行。"]
        event = FakeEvent()
        event.set_reply("你真棒。【好感度+3】")
        run(self.drive(event))
        self.assertEqual(self.reply_text(event), "还是不行。【好感度+3】")
        entry = self.plugin.log.recent(1)[0]
        self.assertEqual(entry["action"], "give_up")
        self.assertIn("句末有句号", entry.get("unresolved", ""))
        self.assertIn(UMO, self.plugin._feedback)            # 仍要注入反馈

    def test_rewrite_error_falls_back_to_original(self):
        self.make_plugin(**{"rewrite.max_rounds": 1, "rewrite.fast_fix": False})
        self.provider.script = [RuntimeError("模型炸了")]
        event = FakeEvent()
        event.set_reply("你真棒。【好感度+3】")
        run(self.drive(event))
        self.assertEqual(self.reply_text(event), "你真棒。【好感度+3】")  # 原稿发出
        entry = self.plugin.log.recent(1)[0]
        self.assertEqual(entry["action"], "give_up")
        self.assertIn("模型炸了", entry.get("error", ""))

    def test_rewrite_recheck_only_against_sent_history(self):
        """重合检测兼容：复检只比上一句与改写后，草稿不入史。"""
        self.plugin.history.record(UMO, "今天天气真不错我们去公园走走吧", user_prev="走吗")
        self.provider.script = ["那我们改天去爬山咯"]
        event = FakeEvent()
        draft = "今天天气真不错我们去公园走走吧。"          # 句末违规 + 与上一句重合
        event.set_reply(draft)
        run(self.drive(event))

        # 打回时确实命中了重合（草稿 vs 上一句）
        entry = self.plugin.log.recent(1)[0]
        hit_ids = {v["id"] for v in entry["violations"]}
        self.assertIn("end_period", hit_ids)
        self.assertIn("overlap_prev", hit_ids)

        hist = self.plugin.history.context(UMO)
        self.assertEqual(hist.prev, "那我们改天去爬山咯")
        self.assertNotIn(draft, hist.recent)                  # 草稿从不入史
        # 改写后与上一句无重合 → 复检通过
        self.assertEqual(self.reply_text(event), "那我们改天去爬山咯")

    def test_multi_plain_collapse(self):
        self.make_plugin(**{"rewrite.fast_fix": False})
        self.provider.script = ["改好的全文"]
        event = FakeEvent()
        event.set_reply("前半段有句号。", "后半段也有句号。")
        run(self.drive(event))
        chain = event.get_result().chain
        self.assertEqual(chain[0].text, "改好的全文")
        self.assertEqual(chain[1].text, "")

    def test_short_message_skipped_but_recorded(self):
        event = FakeEvent()
        event.set_reply("嗯")
        run(self.drive(event))
        self.assertEqual(self.reply_text(event), "嗯")
        self.assertEqual(self.plugin.history.context(UMO).prev, "嗯")
        self.assertEqual(self.plugin.log.stats()["total"], 0)

    # ------------------------------------------------------------------
    # 黑白名单
    # ------------------------------------------------------------------
    def test_blacklist_skips_session(self):
        self.make_plugin(**{"lists.blacklist": UMO})
        event = FakeEvent()
        event.set_reply("你真棒。")
        run(self.drive(event))
        self.assertEqual(self.reply_text(event), "你真棒。")
        self.assertEqual(self.plugin.log.stats()["total"], 0)
        self.assertEqual(self.provider.calls, [])            # 连模型都不调

    def test_blacklist_matches_bare_number_and_sender(self):
        # 裸编号匹配会话尾段（本用例会话尾段是 66666）
        self.make_plugin(**{"lists.blacklist": "66666"})
        event = FakeEvent()
        event.set_reply("你真棒。")
        run(self.drive(event))
        self.assertEqual(self.reply_text(event), "你真棒。")
        self.assertEqual(self.plugin.log.stats()["total"], 0)

        # 发送者 ID 命中：换了会话，同一个人触发的回复照样豁免
        self.make_plugin(**{"lists.blacklist": "10001"})
        event2 = FakeEvent(umo="aiocqhttp:FriendMessage:99999", uid="10001")
        event2.set_reply("你真棒。")
        run(self.drive(event2))
        self.assertEqual(self.reply_text(event2), "你真棒。")
        self.assertEqual(self.plugin.log.stats()["total"], 0)

        # 不在名单内的会话照常质检
        self.make_plugin(**{"lists.blacklist": "10001"})
        self.provider.script = ["你真棒"]
        event3 = FakeEvent(umo="aiocqhttp:FriendMessage:99999", uid="20002")
        event3.set_reply("你真棒。")
        run(self.drive(event3))
        self.assertEqual(self.reply_text(event3), "你真棒")   # 被改写了

    def test_blacklist_group_number_beats_unique_session_umo(self):
        """唯一会话模式下会话 ID 是「发送者_群号」：裸群号也必须命中（本次修的 bug）。"""
        self.make_plugin(**{"lists.blacklist": "66666"})
        event = FakeEvent(umo="default:GroupMessage:10001_66666", uid="10001", gid="66666")
        event.set_reply("你真棒。")
        run(self.drive(event))
        self.assertEqual(self.reply_text(event), "你真棒。")
        self.assertEqual(self.plugin.log.stats()["total"], 0)
        self.assertEqual(self.provider.calls, [])            # 连模型都不调

        # 同群换个人说话：照样豁免（命中的是群号，不是发送者）
        event2 = FakeEvent(umo="default:GroupMessage:20002_66666", uid="20002", gid="66666")
        event2.set_reply("你真棒。")
        run(self.drive(event2))
        self.assertEqual(self.plugin.log.stats()["total"], 0)

        # 别的群照常质检
        self.make_plugin(**{"lists.blacklist": "66666"})
        self.provider.script = ["你真棒"]
        event3 = FakeEvent(umo="default:GroupMessage:30003_77777", uid="30003", gid="77777")
        event3.set_reply("你真棒。")
        run(self.drive(event3))
        self.assertEqual(self.reply_text(event3), "你真棒")   # 被改写了
        self.assertIn(self.plugin.log.recent(1)[0]["action"], ("rewrite", "local"))

    def test_blacklist_group_number_works_without_group_api(self):
        """平台不提供 get_group_id：靠会话尾段分段也能命中裸群号。"""
        self.make_plugin(**{"lists.blacklist": "66666"})
        event = NoGroupEvent(umo="default:GroupMessage:10001_66666", uid="10001")
        event.set_reply("你真棒。")
        run(self.drive(event))
        self.assertEqual(self.reply_text(event), "你真棒。")
        self.assertEqual(self.plugin.log.stats()["total"], 0)

    def test_whitelist_mode_matches_group_in_unique_session(self):
        """白名单模式同样按群号命中：名单内的群照常质检，名单外放行。"""
        self.make_plugin(**{"lists.whitelist_mode": True, "lists.whitelist": "66666"})
        self.provider.script = ["你真棒"]
        event = FakeEvent(umo="default:GroupMessage:10001_66666", uid="10001", gid="66666")
        event.set_reply("你真棒。")
        run(self.drive(event))
        self.assertEqual(self.reply_text(event), "你真棒")     # 被改写了
        self.assertIn(self.plugin.log.recent(1)[0]["action"], ("rewrite", "local"))

        self.make_plugin(**{"lists.whitelist_mode": True, "lists.whitelist": "66666"})
        baseline = self.plugin.log.stats()["total"]           # 日志是落盘的，先记基数
        event2 = FakeEvent(umo="default:GroupMessage:40004_88888", uid="40004", gid="88888")
        event2.set_reply("你真棒。")
        run(self.drive(event2))
        self.assertEqual(self.reply_text(event2), "你真棒。")
        self.assertEqual(self.plugin.log.stats()["total"], baseline)   # 没有新增记录

    def test_blacklisted_session_gets_no_feedback(self):
        """拉黑的会话连下轮反馈注入也要跳过。"""
        self.make_plugin(**{"lists.blacklist": UMO})
        self.plugin._feedback[UMO] = "句末有句号"
        req = FakeReq()
        run(self.plugin.on_llm_request(FakeEvent(), req))
        self.assertNotIn(FEEDBACK_MARKER, req.system_prompt)
        self.assertNotIn(UMO, self.plugin._feedback)

    def test_list_entries_allow_quotes_and_fullwidth_spaces(self):
        self.make_plugin(**{"lists.blacklist": "  \u201c66666\u201d \n1108524594 "})
        event = FakeEvent()
        event.set_reply("你真棒。")
        run(self.drive(event))
        self.assertEqual(self.reply_text(event), "你真棒。")
        self.assertEqual(self.plugin.log.stats()["total"], 0)

    def test_whitelist_mode_only_checks_listed(self):
        self.make_plugin(**{"lists.whitelist_mode": True, "lists.whitelist": "别的会话"})
        event = FakeEvent()                                   # 不在白名单 → 放行
        event.set_reply("你真棒。")
        run(self.drive(event))
        self.assertEqual(self.reply_text(event), "你真棒。")
        self.assertEqual(self.plugin.log.stats()["total"], 0)

        # 名单内（裸编号 66666）→ 正常质检、打回、改写
        self.make_plugin(**{"lists.whitelist_mode": True, "lists.whitelist": "66666"})
        self.provider.script = ["你真棒"]
        event2 = FakeEvent()
        event2.set_reply("你真棒。【好感度+3】")
        run(self.drive(event2))
        self.assertEqual(self.reply_text(event2), "你真棒【好感度+3】")
        self.assertIn(self.plugin.log.recent(1)[0]["action"], ("rewrite", "local"))

    def test_whitelist_mode_empty_list_passes_everything(self):
        self.make_plugin(**{"lists.whitelist_mode": True, "lists.whitelist": ""})
        event = FakeEvent()
        event.set_reply("你真棒。")
        run(self.drive(event))
        self.assertEqual(self.reply_text(event), "你真棒。")

    def test_blacklist_beats_whitelist(self):
        self.make_plugin(**{
            "lists.whitelist_mode": True,
            "lists.whitelist": UMO,
            "lists.blacklist": UMO,
        })
        event = FakeEvent()
        event.set_reply("你真棒。")
        run(self.drive(event))
        self.assertEqual(self.reply_text(event), "你真棒。")
        self.assertEqual(self.plugin.log.stats()["total"], 0)

    # ------------------------------------------------------------------
    # AI 检测层（开 judge）
    # ------------------------------------------------------------------
    def test_judge_layer_flags_and_rewrites(self):
        self.make_plugin(**{"detect.enable_judge": True})
        judge_fail = (
            FENCE + 'json\n{"pass": false, "violations": '
            '[{"id": "ai_flavor", "reason": "客服腔套话"}]}\n' + FENCE
        )
        self.provider.script = [
            judge_fail,
            "认识你很高兴！",
            '{"pass": true, "violations": []}',
        ]
        event = FakeEvent()
        event.set_reply("很高兴认识你，希望我们能聊得开心")   # 过规则、被 AI 层命中
        run(self.drive(event))

        # AI 调用带着注入词
        self.assertGreaterEqual(len(self.provider.calls), 1)
        self.assertIn("质检员", self.provider.calls[0]["system_prompt"])
        self.assertIn("整体 AI 味", self.provider.calls[0]["system_prompt"])
        # 命中 → 改写 → 复检通过
        self.assertEqual(self.reply_text(event), "认识你很高兴！")
        entry = self.plugin.log.recent(1)[0]
        self.assertEqual(entry["action"], "rewrite")
        self.assertEqual(entry["violations"][0]["name"], "整体 AI 味")   # 名称按注册表回填

    def test_judge_error_fails_open(self):
        self.make_plugin(**{"detect.enable_judge": True})
        self.provider.script = [RuntimeError("超时了")]
        event = FakeEvent()
        event.set_reply("这条消息过规则")                   # 规则通过、AI 层故障 → 放行
        run(self.drive(event))
        self.assertEqual(self.reply_text(event), "这条消息过规则")
        self.assertEqual(self.plugin.log.stats()["total"], 0)

    # ------------------------------------------------------------------
    # 指令
    # ------------------------------------------------------------------
    async def _call_cmd(self, text: str) -> str:
        event = FakeEvent(message=text)
        out = []
        async for piece in self.plugin.cmd_qc(event):
            out.append(piece)
        return "\n".join(out)

    def test_command_status(self):
        result = run(self._call_cmd("/qc 状态"))
        self.assertIn("人味守卫", result)
        self.assertIn("检测项", result)

    def test_command_test_subcommand(self):
        result = run(self._call_cmd("/qc 测试 今天有点累。"))
        self.assertIn("命中", result)
        self.assertIn("end_period", result)

    # ------------------------------------------------------------------
    # 面板后端
    # ------------------------------------------------------------------
    def _set_body(self, body: dict[str, Any]) -> None:
        WEB_REQUEST.json_body = body
        WEB_REQUEST.raw_body = b""

    def test_items_api_and_preview_follows_checkbox(self):
        resp = unwrap(run(self.plugin._api_items()))
        ids = [i["id"] for i in resp["items"]]
        self.assertIn("ai_flavor", ids)
        self.assertIn("mid_punct", ids)

        # 关掉 ai_flavor → 注入词预览立刻少一块
        self._set_body({"id": "ai_flavor", "enabled": False})
        unwrap(run(self.plugin._api_items_save()))
        WEB_REQUEST.json_body = None
        preview = unwrap(run(self.plugin._api_prompt_preview()))
        self.assertNotIn("整体 AI 味", preview["system"])
        self.assertIn("句中缺少标点", preview["system"])
        self.assertNotIn("ai_flavor", preview["ids"])

        # 再开 → 注入词回来
        self._set_body({"id": "ai_flavor", "enabled": True})
        unwrap(run(self.plugin._api_items_save()))
        WEB_REQUEST.json_body = None
        preview = unwrap(run(self.plugin._api_prompt_preview()))
        self.assertIn("整体 AI 味", preview["system"])

    def test_items_add_delete_api(self):
        self._set_body({"name": "句尾带英文逗号", "desc": "最后一个字符是英文逗号"})
        resp = unwrap(run(self.plugin._api_items_add()))
        self.assertTrue(resp["auto_prompt"])
        item_id = resp["item"]["id"]
        self.assertIn("句尾带英文逗号", resp["item"]["prompt"])
        self.assertIn(item_id, unwrap(run(self.plugin._api_prompt_preview()))["ids"])

        self._set_body({"id": item_id})
        unwrap(run(self.plugin._api_items_delete()))
        self.assertNotIn(item_id, [i["id"] for i in unwrap(run(self.plugin._api_items()))["items"]])

        # 内置项不可删（返回错误）
        self._set_body({"id": "end_period"})
        resp = run(self.plugin._api_items_delete())
        self.assertEqual(resp.get("status"), "error")

    def test_settings_save_hot_reloads(self):
        self.assertTrue(self.plugin.enabled)
        self._set_body({"values": {"enabled": False, "rewrite.max_rounds": 1}})
        resp = unwrap(run(self.plugin._api_settings_save()))
        self.assertIn("enabled", resp["saved"])
        self.assertEqual(resp["not_saved"], [])
        self.assertFalse(self.plugin.enabled)                # 热更新生效
        self.assertEqual(self.plugin.max_rounds, 1)
        self.assertEqual(self.config.save_count, 1)

    def test_test_endpoint(self):
        self._set_body({"text": "今天有点累。"})
        resp = unwrap(run(self.plugin._api_test()))
        self.assertFalse(resp["passed"])
        self.assertEqual(resp["violations"][0]["id"], "end_period")

    def test_rewrite_test_endpoint(self):
        self.provider.script = ["今天有点累"]
        self._set_body({"text": "今天有点累。"})
        resp = unwrap(run(self.plugin._api_rewrite_test()))
        self.assertEqual(resp["rewritten"], "今天有点累")
        self.assertTrue(resp["recheck"]["passed"])

    def test_status_endpoint(self):
        resp = unwrap(run(self.plugin._api_status()))
        self.assertTrue(resp["enabled"])
        self.assertIn("items", resp)
        self.assertIn("history", resp)
        self.assertIn("blacklist", resp)
        self.assertIn("whitelist", resp)
        self.assertIn("whitelist_mode", resp)
        self.assertFalse(resp["whitelist_mode"])



    # ------------------------------------------------------------------
    # 主动消息拦截（send_by_session 包装）
    # ------------------------------------------------------------------
    def attach_platform(self, *plats: FakePlatform) -> None:
        self.context.platform_manager = types.SimpleNamespace(
            platform_insts=list(plats)
        )

    def test_proactive_send_intercepted_and_rewritten(self):
        plat = FakePlatform()
        self.attach_platform(plat)
        self.plugin._wrap_platform_sends()
        self.provider.script = ["你真棒"]

        chain = types.SimpleNamespace(chain=[Plain("你真棒。")])
        run(plat.send_by_session("aiocqhttp:GroupMessage:66666", chain))

        # 打回改写后真正发送的是改写稿
        self.assertEqual(chain.chain[0].text, "你真棒")
        self.assertEqual(len(plat.sent), 1)
        self.assertIn(self.plugin.log.recent(1)[0]["action"], ("rewrite", "local"))
        self.assertEqual(self.plugin.history.context(UMO).prev, "你真棒")

    def test_proactive_send_passes_clean_text(self):
        plat = FakePlatform()
        self.attach_platform(plat)
        self.plugin._wrap_platform_sends()

        chain = types.SimpleNamespace(chain=[Plain("今天天气真不错，我们去公园吧")])
        run(plat.send_by_session(UMO, chain))

        self.assertEqual(chain.chain[0].text, "今天天气真不错，我们去公园吧")
        self.assertEqual(self.plugin.log.stats()["total"], 0)
        self.assertEqual(self.plugin.history.context(UMO).prev,
                         "今天天气真不错，我们去公园吧")

    def test_proactive_send_blacklist_skips_qc(self):
        plat = FakePlatform()
        self.attach_platform(plat)
        self.make_plugin(**{"lists.blacklist": UMO})
        self.plugin._wrap_platform_sends()

        chain = types.SimpleNamespace(chain=[Plain("你真棒。")])
        run(plat.send_by_session(UMO, chain))

        self.assertEqual(chain.chain[0].text, "你真棒。")   # 原样发出
        self.assertEqual(self.plugin.log.stats()["total"], 0)

    def test_proactive_send_fail_open_on_model_error(self):
        plat = FakePlatform()
        self.attach_platform(plat)
        self.make_plugin(**{"rewrite.fast_fix": False})
        self.plugin._wrap_platform_sends()
        self.provider.script = [RuntimeError("模型炸了")]

        chain = types.SimpleNamespace(chain=[Plain("你真棒。")])
        run(plat.send_by_session(UMO, chain))

        self.assertEqual(chain.chain[0].text, "你真棒。")   # 质检炸了也照发
        self.assertEqual(len(plat.sent), 1)

    def test_proactive_wrap_idempotent_and_unwrap(self):
        plat = FakePlatform()
        self.attach_platform(plat)
        self.plugin._wrap_platform_sends()
        first = plat.send_by_session
        self.plugin._wrap_platform_sends()                  # 再包一次
        self.assertIs(plat.send_by_session, first)          # 不会叠两层
        self.assertEqual(len(self.plugin._wrapped), 1)

        self.plugin._unwrap_platform_sends()                # 卸载还原
        self.assertEqual(len(self.plugin._wrapped), 0)
        chain = types.SimpleNamespace(chain=[Plain("你真棒。")])
        run(plat.send_by_session(UMO, chain))               # 还原后不走质检
        self.assertEqual(self.plugin.log.stats()["total"], 0)

    def test_proactive_send_non_plain_chain_untouched(self):
        plat = FakePlatform()
        self.attach_platform(plat)
        self.plugin._wrap_platform_sends()

        image = types.SimpleNamespace(url="http://x/1.png")  # 非 Plain 组件
        chain = types.SimpleNamespace(chain=[image])
        run(plat.send_by_session(UMO, chain))
        self.assertIs(chain.chain[0], image)

        chain2 = types.SimpleNamespace(chain=[])
        run(plat.send_by_session(UMO, chain2))              # 空链也不炸

    # ------------------------------------------------------------------
    # 提速：本地公式修复（不花模型调用）
    # ------------------------------------------------------------------
    def test_local_fix_end_period_without_ai(self):
        """句末句号：本地公式直接删掉，一次模型都不调。"""
        event = FakeEvent()
        event.set_reply("今天有点累。")
        run(self.drive(event))

        self.assertEqual(self.reply_text(event), "今天有点累")
        self.assertEqual(self.provider.calls, [])                 # 零模型调用
        entry = self.plugin.log.recent(1)[0]
        self.assertEqual(entry["action"], "local")
        self.assertIn("end_period", entry["fixed"])
        self.assertIn("句末有句号", self.plugin._feedback.get(UMO, ""))   # 下轮反馈照给
        self.assertEqual(self.plugin.history.context(UMO).prev, "今天有点累")

    def test_local_fix_rejoins_tags_and_punctuation_runs(self):
        event = FakeEvent()
        event.set_reply("太好了！！！！。【好感度+3】")
        run(self.drive(event))

        self.assertEqual(self.reply_text(event), "太好了！【好感度+3】")
        self.assertEqual(self.provider.calls, [])
        self.assertEqual(self.plugin.log.recent(1)[0]["action"], "local")

    def test_local_fix_mid_punct_splits_long_clause(self):
        event = FakeEvent()
        event.set_reply("今天天气真不错我们去公园走走吧顺便买点吃的一起回家好吗。")
        run(self.drive(event))

        text = self.reply_text(event)
        self.assertEqual(self.provider.calls, [])
        self.assertNotIn("。", text)
        self.assertIn("，", text)
        # 补完逗号后没有任何分句还超阈值
        clauses = [c for c in re.split(r"[，,、;；:：!！?？…。.~～\n\r]+", text) if c]
        self.assertLessEqual(max(len(c) for c in clauses), 15)

    def test_local_fix_leaves_ai_only_violation_to_model(self):
        """本地修不掉的（与上一句重合）才交给 AI；给 AI 的是修好的版本，提示词也只剩这一项。"""
        self.plugin.history.record(UMO, "今天天气真不错我们去公园走走吧", user_prev="走吗")
        self.provider.script = ["那我们改天去爬山"]
        event = FakeEvent()
        event.set_reply("今天天气真不错我们去公园走走吧。")
        run(self.drive(event))

        self.assertEqual(len(self.provider.calls), 1)
        prompt = self.provider.calls[0]["prompt"]
        self.assertIn("换一种说法", prompt)                        # 只剩重合项的改写提示词
        self.assertNotIn("删掉句末的句号", prompt)                  # 句末句号已被本地修掉
        self.assertIn("今天天气真不错我们去公园走走吧", prompt)      # 交给 AI 的是本地修过的版本
        entry = self.plugin.log.recent(1)[0]
        self.assertEqual(entry["action"], "rewrite")
        self.assertIn("end_period", entry["fixed"])
        self.assertEqual({v["id"] for v in entry["violations"]}, {"end_period", "overlap_prev"})
        self.assertEqual(self.reply_text(event), "那我们改天去爬山")

    def test_fast_fix_off_goes_straight_to_ai(self):
        self.make_plugin(**{"rewrite.fast_fix": False})
        self.provider.script = ["今天有点累"]
        event = FakeEvent()
        event.set_reply("今天有点累。")
        run(self.drive(event))
        self.assertEqual(len(self.provider.calls), 1)
        self.assertEqual(self.reply_text(event), "今天有点累")

    def test_recheck_rules_only_saves_one_judge_call(self):
        """改写后默认只跑规则层复检：检测+改写共 2 次调用（旧行为是 3 次）。"""
        self.make_plugin(**{"detect.enable_judge": True})
        judge_fail = (
            FENCE + 'json\n{"pass": false, "violations": '
            '[{"id": "ai_flavor", "reason": "客服腔套话"}]}\n' + FENCE
        )
        self.provider.script = [judge_fail, "认识你很高兴！"]
        event = FakeEvent()
        event.set_reply("很高兴认识你，希望我们能聊得开心")
        run(self.drive(event))
        self.assertEqual(len(self.provider.calls), 2)
        self.assertEqual(self.reply_text(event), "认识你很高兴！")

        # 打开 AI 复检 → 每轮多一次调用（检测 + 改写 + 复检 = 3）
        self.make_plugin(**{"detect.enable_judge": True, "rewrite.recheck_judge": True})
        base = len(self.provider.calls)
        self.provider.script = [judge_fail, "认识你很高兴！", '{"pass": true, "violations": []}']
        event2 = FakeEvent()
        event2.set_reply("很高兴认识你，希望我们能聊得开心")
        run(self.drive(event2))
        self.assertEqual(len(self.provider.calls) - base, 3)
        self.assertEqual(self.reply_text(event2), "认识你很高兴！")

    def test_dry_run_keeps_original_detection(self):
        """体检模式：只记不改，本地公式也不动手，日志里如实记检测结果。"""
        self.make_plugin(**{"rewrite.dry_run": True})
        event = FakeEvent()
        event.set_reply("今天有点累。")
        run(self.drive(event))
        self.assertEqual(self.reply_text(event), "今天有点累。")     # 原稿照发
        entry = self.plugin.log.recent(1)[0]
        self.assertEqual(entry["action"], "dry")
        self.assertEqual(entry["violations"][0]["id"], "end_period")

    # ------------------------------------------------------------------
    # 分段插件（对话分段Pro）兼容
    # ------------------------------------------------------------------
    def test_split_fragments_are_not_rechecked(self):
        """分段Pro 把整条回复切段直发：碎片不重复质检、不写历史、不再多传 AI。"""
        plat = FakePlatform()
        self.attach_platform(plat)
        self.plugin._wrap_platform_sends()

        event = FakeEvent()
        event.set_reply("今天有点累。\n明天再说吧。")     # 装饰钩子按整条质检（本地删句末句号）
        run(self.drive(event))
        self.assertEqual(self.reply_text(event), "今天有点累。\n明天再说吧")

        calls = len(self.provider.calls)
        logs = self.plugin.log.stats()["total"]
        history = self.plugin.history.context(UMO).prev

        # 分段插件按换行切开，逐段经 send_by_session 发出
        for piece in ("今天有点累。", "明天再说吧"):
            frag = types.SimpleNamespace(chain=[Plain(piece)])
            run(plat.send_by_session(UMO, frag))

        self.assertEqual(len(self.provider.calls), calls)              # 一次都没多调
        self.assertEqual(self.plugin.log.stats()["total"], logs)       # 碎片不进日志
        hist = self.plugin.history.context(UMO)
        self.assertEqual(hist.prev, history)                           # 碎片不写历史
        self.assertEqual(len(hist.recent), 1)
        self.assertNotIn("明天再说吧", hist.recent[1:])

    def test_split_fragments_skip_even_without_prior_hook(self):
        """先收到碎片本身（没有先质检整条）时仍照常质检，不会整段漏检。"""
        plat = FakePlatform()
        self.attach_platform(plat)
        self.plugin._wrap_platform_sends()
        frag = types.SimpleNamespace(chain=[Plain("这条没有上下文也要查。")])
        run(plat.send_by_session(UMO, frag))
        self.assertEqual(frag.chain[0].text, "这条没有上下文也要查")
        self.assertEqual(self.plugin.log.recent(1)[0]["action"], "local")

    def test_context_send_message_checks_whole_message(self):
        """主动发送：先按整条质检，再交给（假扮的）分段插件切段发出。"""
        sent: list[Any] = []

        async def inner_send(session: Any, chain: Any) -> bool:
            sent.append(chain)
            return True

        self.context.send_message = inner_send
        self.plugin._assert_context_send()
        self.assertTrue(self.plugin._ctx_send_wrapped)
        self.assertTrue(getattr(self.context.send_message, "_humanize_guard_wrapper", False))

        chain = types.SimpleNamespace(chain=[Plain("今天有点累。\n明天再说吧。")])
        run(self.context.send_message(UMO, chain))
        self.assertEqual(chain.chain[0].text, "今天有点累。\n明天再说吧")   # 整条被本地修好
        self.assertEqual(self.provider.calls, [])
        self.assertEqual(self.plugin.log.recent(1)[0]["action"], "local")
        self.assertEqual(len(sent), 1)                                       # 仍然照常发出

        # 幂等：再补挂一次不会叠层
        first = self.context.send_message
        self.plugin._assert_context_send()
        self.assertIs(self.context.send_message, first)

        # 卸载还原
        self.plugin._unwrap_context_send()
        self.assertFalse(getattr(self.context.send_message, "_humanize_guard_wrapper", False))

    def test_context_wrapper_skips_fragments_sent_by_splitter(self):
        """整条质检过之后，分段插件逐段直发时不再被当新消息质检。"""
        plat = FakePlatform()
        self.attach_platform(plat)
        self.plugin._wrap_platform_sends()

        async def inner_send(session: Any, chain: Any) -> bool:
            return True

        self.context.send_message = inner_send
        self.plugin._assert_context_send()

        chain = types.SimpleNamespace(chain=[Plain("今天有点累。\n明天再说吧。")])
        run(self.context.send_message(UMO, chain))
        logs = self.plugin.log.stats()["total"]
        calls = len(self.provider.calls)

        frag = types.SimpleNamespace(chain=[Plain("明天再说吧")])
        run(plat.send_by_session(UMO, frag))
        self.assertEqual(frag.chain[0].text, "明天再说吧")
        self.assertEqual(self.plugin.log.stats()["total"], logs)
        self.assertEqual(len(self.provider.calls), calls)

    # ------------------------------------------------------------------
    # 阈值参数保存（"面板改了 max_run 保存后还是 15" 的回归）
    # ------------------------------------------------------------------
    def _save_item(self, body: dict[str, Any]) -> Any:
        self._set_body(body)
        resp = unwrap(run(self.plugin._api_items_save()))
        WEB_REQUEST.json_body = None
        return resp

    def test_param_save_updates_effective_threshold(self):
        """把 mid_punct.max_run 从 15 改成 25：保存后立即生效、回显一致。"""
        resp = self._save_item({"id": "mid_punct", "params": {"max_run": 25}})
        self.assertEqual(resp["item"]["params"]["max_run"], 25)
        # 生效口径：25 字内不算缺标点，16 字的句子不再被打回
        event = FakeEvent()
        event.set_reply("今天天气真不错我们去公园走走吧")      # 16 字，旧阈值会命中
        run(self.drive(event))
        self.assertEqual(self.reply_text(event), "今天天气真不错我们去公园走走吧")
        self.assertEqual(self.plugin.log.stats()["total"], 0)
        # 超过 25 字仍然命中（换一句与历史毫无重合的话，只考 mid_punct）
        event2 = FakeEvent()
        event2.set_reply("他背着书包一路小跑赶去教室路上还顺便帮同学捡了掉在地上的作业本")
        run(self.drive(event2))
        hit_ids = {v["id"] for v in self.plugin.log.recent(1)[0]["violations"]}
        self.assertIn("mid_punct", hit_ids)

    def test_param_save_accepts_numeric_strings_and_clamps(self):
        """面板/桥可能把数字传成字符串：照样认；越界钳到声明范围，不再回退默认 15。"""
        resp = self._save_item({"id": "mid_punct", "params": {"max_run": "20"}})
        self.assertEqual(resp["item"]["params"]["max_run"], 20)      # "20" 也认
        resp = self._save_item({"id": "mid_punct", "params": {"max_run": 9999}})
        self.assertEqual(resp["item"]["params"]["max_run"], 300)     # 钳到上限
        resp = self._save_item({"id": "mid_punct", "params": {"max_run": 1}})
        self.assertEqual(resp["item"]["params"]["max_run"], 3)       # 钳到下限
        # 空值 → 回 schema 默认，不再落成 0
        resp = self._save_item({"id": "mid_punct", "params": {"max_run": ""}})
        self.assertEqual(resp["item"]["params"]["max_run"], 15)

    def test_param_save_is_incremental_not_wholesale(self):
        """只发改动字段（勾选时只带 id+enabled）：其他参数必须原样保留。"""
        self._save_item({"id": "mid_punct", "params": {"max_run": 25}})
        # 用户随后只切了一下勾选（请求里根本没有 params 字段）
        self._save_item({"id": "mid_punct", "enabled": False})
        item = self.plugin.items.get("mid_punct")
        self.assertEqual(item.params["max_run"], 25)                 # 没被冲掉
        self.assertFalse(item.enabled)
        # 再勾回来
        self._save_item({"id": "mid_punct", "enabled": True})
        self.assertTrue(self.plugin.items.get("mid_punct").enabled)

    def test_param_save_rejects_garbage_gracefully(self):
        """乱填的值（对象/None）不落盘，接口不 500，其余字段照常保存。"""
        resp = self._save_item({
            "id": "mid_punct",
            "params": {"max_run": {"bad": 1}},
            "prompt": "随便改一句注入词",
        })
        self.assertEqual(resp["item"]["params"]["max_run"], 15)      # 默认值兜底
        self.assertIn("随便改一句注入词", resp["item"]["prompt"])

    def test_lines_param_save_keeps_list_form(self):
        """lines 型参数（套话清单）：字符串按行拆成数组，数组原样保留。"""
        resp = self._save_item({"id": "boilerplate", "params": {
            "phrases": "新的套话甲\n新的套话乙",
        }})
        self.assertIn("新的套话甲", resp["item"]["params"]["phrases"])
        self.assertIn("新的套话乙", resp["item"]["params"]["phrases"])
        self.assertTrue(all(isinstance(x, str) for x in resp["item"]["params"]["phrases"]))

    def test_param_save_persists_to_disk(self):
        self._save_item({"id": "mid_punct", "params": {"max_run": 25}})
        # 断言前先确认保存动作真的写盘了（ItemStore.update 内部会 save()）
        self.assertTrue(self.plugin.items.path.exists(),
                        f"items.json 没有落盘：{self.plugin.items.path}")
        raw = json.loads(self.plugin.items.path.read_text(encoding="utf-8"))
        self.assertEqual(raw["items"]["mid_punct"]["params"]["max_run"], 25)
        # 重启（重新加载注册表）后仍然生效
        reloaded = S_ItemStore(self.plugin.items.path)
        reloaded.load()
        self.assertEqual(reloaded.get("mid_punct").params["max_run"], 25)

    def test_fake_splitter_end_to_end(self):
        """端到端模拟对话分段Pro：劫持 Context.send_message → 按换行切段 → 逐段直发。

        期望：整条先被本地公式修好（0 次 AI 调用），随后分段发出去的碎片不再被质检第二遍。
        """
        plat = FakePlatform()
        self.attach_platform(plat)
        self.plugin._wrap_platform_sends()
        self.plugin._assert_context_send()

        async def fake_splitter(session: Any, chain: Any) -> bool:
            text = "".join(c.text for c in chain.chain if isinstance(c, Plain))
            for piece in [x for x in text.split("\n") if x.strip()]:
                await plat.send_by_session(session, types.SimpleNamespace(chain=[Plain(piece)]))
            return True

        # 分段插件后包一层（它就是这么劫持的），人味守卫下次请求再把自己补到最外层
        self.context.send_message = fake_splitter
        self.plugin._assert_context_send()
        self.assertTrue(getattr(self.context.send_message, "_humanize_guard_wrapper", False))

        chain = types.SimpleNamespace(chain=[Plain("今天有点累。\n明天再说吧。")])
        run(self.context.send_message(UMO, chain))

        sent = ["".join(c.text for c in ch.chain if isinstance(c, Plain)) for _, ch in plat.sent]
        self.assertEqual(sent, ["今天有点累。", "明天再说吧"])       # 分段照旧，内容已被修好
        self.assertEqual(self.provider.calls, [])                    # 全程零 AI 调用
        self.assertEqual(self.plugin.log.stats()["total"], 1)        # 只记一条（整条那次）
        entry = self.plugin.log.recent(1)[0]
        self.assertEqual(entry["action"], "local")
        self.assertEqual(self.plugin.history.context(UMO).prev, "今天有点累。\n明天再说吧")


if __name__ == "__main__":
    unittest.main()
