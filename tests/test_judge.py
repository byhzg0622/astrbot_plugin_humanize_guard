"""AI 检测层：注入词拼装（随勾选变化）与裁决解析。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.items import ItemStore  # noqa: E402
from core.judge import build_judge_prompt, parse_verdict  # noqa: E402
from core.model import RuleContext  # noqa: E402

FENCE = chr(96) * 3


def make_store() -> ItemStore:
    store = ItemStore(Path(tempfile.mkdtemp()) / "items.json")
    store.load()
    return store


class PromptAssembly(unittest.TestCase):
    def setUp(self):
        self.store = make_store()
        self.ctx = RuleContext(
            prev="上一条已发送的消息",
            recent=["上一条已发送的消息"],
            user_prev="用户的上一条消息",
        )

    def test_system_contains_enabled_items_only(self):
        enabled = self.store.enabled_judge_items()
        built = build_judge_prompt(enabled, "测试消息", self.ctx)
        self.assertIn("句中缺少标点", built["system"])
        self.assertIn("整体 AI 味", built["system"])
        self.assertIn("测试消息", built["prompt"])
        self.assertIn("上一条已发送的消息", built["system"])
        self.assertIn("用户的上一条消息", built["system"])
        self.assertIn("end_period", built["ids"])

    def test_prompt_changes_with_checkbox(self):
        before = build_judge_prompt(self.store.enabled_judge_items(), "测试", self.ctx)
        self.store.update("ai_flavor", enabled=False)
        self.store.update("mid_punct", enabled=False)
        after = build_judge_prompt(self.store.enabled_judge_items(), "测试", self.ctx)
        self.assertNotEqual(before["system"], after["system"])
        self.assertNotIn("整体 AI 味", after["system"])
        self.assertNotIn("句中缺少标点", after["system"])
        self.assertNotIn("ai_flavor", after["ids"])
        self.assertNotIn("mid_punct", after["ids"])
        self.assertIn("end_period", after["ids"])

    def test_prompt_changes_with_custom_item(self):
        before = build_judge_prompt(self.store.enabled_judge_items(), "测试", self.ctx)
        custom = self.store.add_custom("新检测项甲", "每句话都要带一个错别字")
        after = build_judge_prompt(self.store.enabled_judge_items(), "测试", self.ctx)
        self.assertIn(custom.id, after["ids"])
        self.assertIn("新检测项甲", after["system"])
        self.assertIn("每句话都要带一个错别字", after["system"])
        self.assertNotEqual(before["system"], after["system"])

    def test_json_contract_lists_ids(self):
        import json
        built = build_judge_prompt(self.store.enabled_judge_items(), "测试", self.ctx)
        marker = '"id" 只能取自：'
        self.assertIn(marker, built["system"])
        line = built["system"].split(marker, 1)[1].splitlines()[0]
        ids = json.loads(line)
        self.assertEqual(ids, built["ids"])


class VerdictParsing(unittest.TestCase):
    def setUp(self):
        self.ids = ["end_period", "ai_flavor"]

    def test_plain_json(self):
        out, err = parse_verdict(
            '{"pass": false, "violations": [{"id": "end_period", "reason": "末尾句号"}]}',
            self.ids,
        )
        self.assertEqual(err, "")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].id, "end_period")
        self.assertEqual(out[0].engine, "judge")

    def test_fenced_json_and_surrounding_text(self):
        raw = FENCE + 'json\n前置说明 {"pass": false, "violations": [{"id": "ai_flavor", "reason": "客服腔"}]} 后缀'
        out, err = parse_verdict(raw, self.ids)
        self.assertEqual(err, "")
        self.assertEqual(out[0].id, "ai_flavor")

    def test_unknown_id_filtered(self):
        raw = '{"pass": false, "violations": [{"id": "没注册的", "reason": "x"}]}'
        out, _ = parse_verdict(raw, self.ids)
        # 未注册 id 必须被过滤；模型明确判 fail 时兜底成通用打回，不许放行
        self.assertNotIn("没注册的", [v.id for v in out])
        self.assertTrue(all(v.id in self.ids or v.id == "ai_reject" for v in out))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].id, "ai_reject")

    def test_pass_false_without_items_becomes_generic(self):
        out, err = parse_verdict('{"pass": false}', self.ids)
        self.assertEqual(err, "")
        self.assertEqual(out[0].id, "ai_reject")

    def test_pass_true_empty(self):
        out, err = parse_verdict('{"pass": true, "violations": []}', self.ids)
        self.assertEqual(err, "")
        self.assertEqual(out, [])

    def test_garbage_is_error(self):
        out, err = parse_verdict("我觉得这条消息挺好的呀", self.ids)
        self.assertEqual(out, [])
        self.assertNotEqual(err, "")

    def test_empty_output_is_error(self):
        out, err = parse_verdict("", self.ids)
        self.assertEqual(out, [])
        self.assertNotEqual(err, "")


if __name__ == "__main__":
    unittest.main()
