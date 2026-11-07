"""规则层单元测试：每个检测项正反两用例 + 上下文依赖项。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.items import ItemStore  # noqa: E402
from core.model import RuleContext  # noqa: E402
from core.rules import (  # noqa: E402
    RULE_FUNCS,
    longest_common,
    normalize,
    run_rules,
)

_STORE = ItemStore(Path(tempfile.mkdtemp()) / "items.json")
_STORE.load()


def store() -> ItemStore:
    return _STORE


def check(item_id: str, text: str, ctx: RuleContext | None = None) -> bool:
    s = store()
    item = s.get(item_id)
    assert item is not None, item_id
    out = run_rules(text, [item], ctx or RuleContext())
    return any(v.id == item_id for v in out)


class TestHelpers(unittest.TestCase):
    def test_normalize_keeps_alnum_cjk(self):
        self.assertEqual(normalize("哈 哈，2024！ABC-"), "哈哈2024abc")

    def test_longest_common(self):
        length, frag = longest_common("今天天气真不错", "今天天气真好")
        self.assertGreaterEqual(length, 5)
        self.assertTrue("今天天气" in frag)


class TestPunctuationRules(unittest.TestCase):
    def test_end_period_hits_only_final_period(self):
        self.assertTrue(check("end_period", "今天有点累。"))
        self.assertTrue(check("end_period", "今天有点累."))
        self.assertFalse(check("end_period", "今天有点累"))
        self.assertFalse(check("end_period", "今天有点累！"))
        self.assertFalse(check("end_period", "今天有点累……"))

    def test_mid_punct_run_on(self):
        self.assertTrue(check("mid_punct", "今天天气真不错我们一起去公园走走吧顺便买点喝的回家"))
        self.assertFalse(check("mid_punct", "今天天气真不错，我们去公园吧"))
        self.assertFalse(check("mid_punct", "好"))            # 短句豁免
        self.assertFalse(check("mid_punct", "https://example.com/some/very/long/path/without/punctuation"))

    def test_punct_oddity(self):
        self.assertTrue(check("punct_oddity", "太棒了！！！！"))
        self.assertTrue(check("punct_oddity", "啊。。。。。。"))
        self.assertFalse(check("punct_oddity", "太棒了！！"))
        self.assertFalse(check("punct_oddity", "真的吗？？"))


class TestOverlapRules(unittest.TestCase):
    def test_overlap_prev(self):
        ctx = RuleContext(prev="今天天气真不错我们去公园走走吧", recent=["今天天气真不错我们去公园走走吧"])
        self.assertTrue(check("overlap_prev", "今天天气真不错我们去公园走走吧，你说呢", ctx))
        self.assertFalse(check("overlap_prev", "我刚吃完饭，准备看会儿剧", ctx))

    def test_overlap_prev_empty_history_passes(self):
        self.assertFalse(check("overlap_prev", "今天天气真不错我们去公园走走吧"))

    def test_overlap_self(self):
        self.assertTrue(check("overlap_self", "我们一起去公园吧，我们一起去公园吧"))
        self.assertFalse(check("overlap_self", "我们一起去公园吧，顺便买点喝的"))

    def test_same_head_needs_two_matches(self):
        ctx = RuleContext(recent=["哈哈那你明天来找我玩", "哈哈那你记得早点过来", "嗯嗯我先去忙了"])
        self.assertTrue(check("same_head", "哈哈那你明天有空吗", ctx))
        single = RuleContext(recent=["哈哈那你明天来找我玩"])
        self.assertFalse(check("same_head", "哈哈那你明天有空吗", single))

    def test_same_tail_particle(self):
        ctx = RuleContext(recent=["哈哈那你早点休息呢", "哈哈那我先去休息呢", "行吧那你先睡呢"])
        self.assertTrue(check("same_tail", "哈哈那他今晚早点休息呢", ctx))   # 尾三字命中2次
        self.assertTrue(check("same_tail", "那我也不啰嗦了呢", ctx))          # 语气词命中3次
        fresh = RuleContext(recent=["明天见"])
        self.assertFalse(check("same_tail", "早点休息呢", fresh))


class TestAITellRules(unittest.TestCase):
    def test_boilerplate(self):
        self.assertTrue(check("boilerplate", "希望这能帮到你"))
        self.assertFalse(check("boilerplate", "这事儿包在我身上"))

    def test_structure(self):
        self.assertTrue(check("structure", "1. 吃饭\n2. 睡觉"))
        self.assertTrue(check("structure", "这是**重点**内容"))
        self.assertFalse(check("structure", "1. 就是一个编号而已别紧张"))

    def test_identity_leak(self):
        self.assertTrue(check("identity_leak", "作为一个AI我无法回答这个问题"))
        self.assertTrue(check("identity_leak", "我是语言模型，需要澄清一下"))
        self.assertFalse(check("identity_leak", "我就是这么想的"))

    def test_written_style(self):
        self.assertTrue(check("written_style", "值得注意的是，这个方案可行"))
        self.assertFalse(check("written_style", "对了，这个方案可行"))

    def test_residue(self):
        self.assertTrue(check("residue", "你好<quoted_message>他说的</quoted_message>"))
        self.assertTrue(check("residue", "你好\u3000世界"))
        self.assertFalse(check("residue", "你好世界"))

    def test_echo_user(self):
        ctx = RuleContext(user_prev="你今天下午要不要一起去图书馆自习")
        self.assertTrue(check("echo_user", "我今天下午要不要一起去图书馆自习", ctx))
        self.assertFalse(check("echo_user", "去不了，晚上还有课", ctx))

    def test_filler_dup(self):
        self.assertTrue(check("filler_dup", "嘿嘿真的太好笑了嘿嘿真的服了嘿嘿"))
        self.assertFalse(check("filler_dup", "我觉得这个什么什么的就算了"))

    def test_too_long(self):
        long_text = "今天天气不错" * 100
        self.assertTrue(check("too_long", long_text))
        self.assertFalse(check("too_long", "短短的一句话"))


class TestDispatch(unittest.TestCase):
    def test_every_seed_has_rule_impl_or_is_judge(self):
        s = store()
        for item in s.list():
            if item.engine == "rule":
                self.assertIn(item.id, RULE_FUNCS, f"规则项缺实现：{item.id}")

    def test_rule_exception_does_not_propagate(self):
        s = store()
        item = s.get("end_period")

        def boom(*_a, **_k):
            raise RuntimeError("炸了")

        original = RULE_FUNCS["end_period"]
        RULE_FUNCS["end_period"] = boom
        try:
            out = run_rules("随便", [item], RuleContext())
            self.assertEqual(out, [])
        finally:
            RULE_FUNCS["end_period"] = original


if __name__ == "__main__":
    unittest.main()
