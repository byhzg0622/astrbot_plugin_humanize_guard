"""本地公式修复层的单元测试（不需要假 astrbot 运行时）。

覆盖：各条"公式"能修什么、修成什么样；修不掉的原样留给 AI；
以及最要紧的安全阀 —— 只采纳"违规变少且不新增违规"的补丁。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
WORKSPACE = PLUGIN_DIR.parent
for _path in (str(WORKSPACE), str(PLUGIN_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from astrbot_plugin_humanize_guard.core.fix import (
    AI_ONLY_IDS,
    FIXABLE_IDS,
    can_fix,
    local_repair,
)
from astrbot_plugin_humanize_guard.core.items import ItemStore
from astrbot_plugin_humanize_guard.core.model import RuleContext
from astrbot_plugin_humanize_guard.core.rules import run_rules


def make_items(**enabled: bool):
    store = ItemStore(Path(tempfile.mkdtemp()) / "items.json")
    store.load()
    for item_id, on in enabled.items():
        store.update(item_id, enabled=on)
    return store.enabled_rule_items()


class LocalRepairCase(unittest.TestCase):
    def setUp(self):
        self.items = make_items()

    # -- 能修的 -------------------------------------------------------------
    def test_end_period(self):
        ctx = RuleContext()
        result = local_repair("今天有点累。", run_rules("今天有点累。", self.items, ctx), self.items, ctx)
        self.assertEqual(result.text, "今天有点累")
        self.assertIn("end_period", result.fixed)
        self.assertFalse(result.remaining)

    def test_end_period_keeps_exclamation(self):
        ctx = RuleContext()
        result = local_repair("今天有点累！", run_rules("今天有点累！", self.items, ctx), self.items, ctx)
        self.assertEqual(result.text, "今天有点累！")      # 规则本就通过 → 不动
        self.assertFalse(result.changed)

    def test_punct_runs_collapse(self):
        ctx = RuleContext()
        text = "太好了！！！！"
        result = local_repair(text, run_rules(text, self.items, ctx), self.items, ctx)
        self.assertEqual(result.text, "太好了！")
        self.assertIn("punct_oddity", result.fixed)

    def test_dots_collapse_to_ellipsis(self):
        # 句中连发句点：收成省略号
        ctx = RuleContext()
        text = "好。。。。真的吗"
        result = local_repair(text, run_rules(text, self.items, ctx), self.items, ctx)
        self.assertEqual(result.text, "好…真的吗")

    def test_trailing_dots_dropped_with_end_period(self):
        # 句尾连发句点：句末句号公式直接清掉整串
        ctx = RuleContext()
        text = "嗯。。。。"
        result = local_repair(text, run_rules(text, self.items, ctx), self.items, ctx)
        self.assertEqual(result.text, "嗯")
        self.assertIn("end_period", result.fixed)

    def test_residue_cleaned(self):
        ctx = RuleContext()
        text = "你好\u3000呀\u200b"
        result = local_repair(text, run_rules(text, self.items, ctx), self.items, ctx)
        self.assertEqual(result.text, "你好 呀")

    def test_structure_markdown_removed(self):
        ctx = RuleContext()
        text = "**重点**在这：\n- 第一点\n- 第二点"
        result = local_repair(text, run_rules(text, self.items, ctx), self.items, ctx)
        self.assertNotIn("**", result.text)
        self.assertNotIn("- ", result.text)
        self.assertFalse(result.remaining)

    def test_mid_punct_inserts_comma(self):
        ctx = RuleContext()
        text = "今天天气真不错我们去公园走走吧顺便买点吃的一起回家好吗"
        result = local_repair(text, run_rules(text, self.items, ctx), self.items, ctx)
        self.assertIn("，", result.text)
        self.assertGreater(len(result.text), len(text))
        self.assertFalse(result.remaining)

    def test_long_clause_gets_several_commas(self):
        ctx = RuleContext()
        text = "其实吧我觉得今天这个事吧也不是什么大事你也别太往心里去反正就这样了"
        result = local_repair(text, run_rules(text, self.items, ctx), self.items, ctx)
        self.assertFalse(result.remaining)
        for clause in [c for c in result.text.replace("，", "\n").splitlines() if c]:
            self.assertLessEqual(len(clause), 15)

    # -- 断句质量（"鸡肋"反馈后重写的规则） --------------------------------
    def test_comma_never_cuts_inside_word(self):
        """绝不允许"也不是，什么大事"这类把词切开的硬断。"""
        ctx = RuleContext()
        text = "其实吧我觉得今天这个事吧也不是什么大事你也别太往心里去"
        result = local_repair(text, run_rules(text, self.items, ctx), self.items, ctx)
        # 断点只可能落在完整词块边界
        self.assertNotIn("也不是，", result.text)
        self.assertNotIn("，什么大事", result.text)
        # 每个分句仍然合规
        self.assertFalse(result.remaining)

    def test_comma_refuses_when_no_good_boundary(self):
        """找不到自然断点就整句放行，交给 AI —— 宁可不修，不硬修。"""
        ctx = RuleContext()
        text = "我今天早上起来的时候头有点疼就没去上班在家里躺了一天"
        result = local_repair(text, run_rules(text, self.items, ctx), self.items, ctx)
        self.assertFalse(result.changed)
        self.assertEqual([v.id for v in result.remaining], ["mid_punct"])
        self.assertEqual(result.text, text)          # 原文一字不动

    def test_comma_breaks_before_connectives_and_subjects(self):
        ctx = RuleContext()
        text = "晚上一起吃个饭然后去看电影怎么样顺便把那件事说清楚"
        result = local_repair(text, run_rules(text, self.items, ctx), self.items, ctx)
        self.assertIn("，然后", result.text)
        self.assertIn("，顺便", result.text)
        self.assertFalse(result.remaining)

    def test_comma_both_sides_within_threshold(self):
        """每一刀切下去两边都不能还是超长，避免"补了个寂寞"。"""
        ctx = RuleContext()
        text = "我昨天去超市买了好多东西回来以后发现忘记买盐了然后又跑了一趟回来的时候顺便取了快递累死我了"
        result = local_repair(text, run_rules(text, self.items, ctx), self.items, ctx)
        # 这句的合格断点都不在合法区间 → 拒绝硬修（AI 接手）
        self.assertFalse(result.changed)
        self.assertEqual(result.text, text)

    def test_same_tail_particle_dropped(self):
        ctx = RuleContext(prev="晚安哦", recent=["晚安哦", "早点睡哦", "你也早点哦"])
        text = "那就这样吧哦"
        before = run_rules(text, self.items, ctx)
        self.assertIn("same_tail", [v.id for v in before])
        result = local_repair(text, before, self.items, ctx)
        self.assertEqual(result.text, "那就这样吧")
        self.assertFalse(result.remaining)

    def test_boilerplate_phrase_deleted(self):
        ctx = RuleContext()
        text = "希望这能帮到你，今天就聊到这吧"
        result = local_repair(text, run_rules(text, self.items, ctx), self.items, ctx)
        self.assertNotIn("希望这能帮到你", result.text)
        self.assertIn("今天就聊到这吧", result.text)

    def test_overlap_self_duplicate_clause_dropped(self):
        ctx = RuleContext()
        text = "今天真的好累啊，今天真的好累啊"
        before = run_rules(text, self.items, ctx)
        self.assertIn("overlap_self", [v.id for v in before])
        result = local_repair(text, before, self.items, ctx)
        self.assertEqual(result.text, "今天真的好累啊")
        self.assertFalse(result.remaining)

    # -- 修不掉的 -----------------------------------------------------------
    def test_overlap_prev_left_to_ai(self):
        ctx = RuleContext(prev="今天天气真不错我们去公园走走吧")
        text = "今天天气真不错我们去公园走走吧。"
        before = run_rules(text, self.items, ctx)
        result = local_repair(text, before, self.items, ctx)
        self.assertEqual(result.text, "今天天气真不错我们去公园走走吧")   # 句号本地修掉了
        self.assertIn("end_period", result.fixed)
        self.assertEqual([v.id for v in result.remaining], ["overlap_prev", "overlap_prev"])

    def test_ai_only_ids_reported(self):
        for item_id in AI_ONLY_IDS:
            self.assertFalse(can_fix(item_id), item_id)
        for item_id in FIXABLE_IDS:
            self.assertTrue(can_fix(item_id), item_id)

    def test_cannot_fix_does_not_touch_text(self):
        ctx = RuleContext()
        text = "普通的一句话，没什么问题"
        before = run_rules(text, self.items, ctx)
        self.assertFalse(before)
        result = local_repair(text, before, self.items, ctx)
        self.assertEqual(result.text, text)
        self.assertFalse(result.changed)

    # -- 安全阀 -------------------------------------------------------------
    def test_never_adds_new_violation(self):
        """每个落地的补丁都必须让违规变少且不新增 —— 逐条核对。"""
        ctx = RuleContext(prev="晚安哦", recent=["晚安哦", "早点睡哦", "你也早点哦"])
        samples = [
            "今天有点累。",
            "太好了！！！！",
            "你好\u3000呀",
            "**重点**在这\n- 第一点\n- 第二点",
            "今天天气真不错我们去公园走走吧顺便买点吃的一起回家好吗",
            "那就这样吧哦",
            "希望这能帮到你，今天就聊到这吧",
            "今天真的好累啊，今天真的好累啊",
        ]
        for text in samples:
            before = run_rules(text, self.items, ctx)
            result = local_repair(text, before, self.items, ctx)
            after = run_rules(result.text, self.items, ctx)
            before_ids = [v.id for v in before]
            after_ids = [v.id for v in after]
            self.assertFalse(set(after_ids) - set(before_ids), f"新增了违规：{text} -> {result.text}")
            self.assertLessEqual(len(after), len(before), f"违规没变少：{text} -> {result.text}")
            # 结果必须与"复检口径"一致
            self.assertEqual([v.id for v in result.remaining], after_ids)
            # 没修好时原文不能变（要么真修好一项，要么原样）
            if not result.changed:
                self.assertEqual(result.text, text)

    def test_repair_is_idempotent(self):
        ctx = RuleContext()
        text = "太好了！！！！今天有点累。"
        once = local_repair(text, run_rules(text, self.items, ctx), self.items, ctx)
        twice = local_repair(once.text, run_rules(once.text, self.items, ctx), self.items, ctx)
        self.assertEqual(twice.text, once.text)
        self.assertFalse(twice.remaining)

    def test_empty_input_safe(self):
        ctx = RuleContext()
        self.assertEqual(local_repair("", [], self.items, ctx).text, "")
        self.assertEqual(local_repair("没事", [], self.items, ctx).text, "没事")


if __name__ == "__main__":
    unittest.main()
