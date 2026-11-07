"""检测项注册表：种子、自定义自动注入词、持久化、恢复默认。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.items import ItemStore, gen_prompt, gen_rewrite  # noqa: E402


class ItemStoreCase(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / "items.json"
        self.store = ItemStore(self.path)
        self.store.load()

    def test_seeds_loaded_in_order(self):
        ids = [i.id for i in self.store.list()]
        self.assertIn("mid_punct", ids)
        self.assertIn("end_period", ids)
        self.assertIn("ai_flavor", ids)
        self.assertEqual(ids[0], "mid_punct")
        self.assertTrue(all(i.builtin for i in self.store.list()))

    def test_custom_item_autogens_prompts(self):
        item = self.store.add_custom("句尾带英文逗号", "消息最后一个字符是英文逗号")
        self.assertTrue(item.id.startswith("custom_"))
        self.assertEqual(item.engine, "judge")
        self.assertIn("句尾带英文逗号", item.prompt)
        self.assertIn("消息最后一个字符是英文逗号", item.prompt)
        self.assertTrue(item.rewrite.startswith("修正"))
        self.assertIn("句尾带英文逗号", item.rewrite)

    def test_custom_item_explicit_prompts_kept(self):
        item = self.store.add_custom("项A", "说明A", prompt="自定义注入词", rewrite="自定义改写词")
        self.assertEqual(item.prompt, "自定义注入词")
        self.assertEqual(item.rewrite, "自定义改写词")

    def test_persistence_roundtrip(self):
        item = self.store.add_custom("临时项", "临时说明")
        self.store.update("ai_flavor", enabled=False, prompt="改过的注入词")
        self.store.update("mid_punct", params={"max_run": 20})

        reloaded = ItemStore(self.path)
        reloaded.load()
        self.assertIsNotNone(reloaded.get(item.id))
        self.assertEqual(reloaded.get(item.id).desc, "临时说明")
        ai = reloaded.get("ai_flavor")
        self.assertFalse(ai.enabled)
        self.assertEqual(ai.prompt, "改过的注入词")
        self.assertEqual(reloaded.get("mid_punct").params.get("max_run"), 20)

    def test_delete_rules(self):
        item = self.store.add_custom("待删", "说明")
        with self.assertRaises(ValueError):
            self.store.delete("end_period")     # 内置不可删
        self.store.delete(item.id)
        self.assertIsNone(self.store.get(item.id))

    def test_reset_builtin_keeps_custom(self):
        custom = self.store.add_custom("保留我", "说明")
        self.store.update("ai_flavor", enabled=False)
        self.store.update("mid_punct", params={"max_run": 99})
        restored = self.store.reset_builtin()
        self.assertGreater(restored, 0)
        self.assertTrue(self.store.get("ai_flavor").enabled)
        self.assertEqual(self.store.get("mid_punct").params.get("max_run"), 15)
        self.assertIsNotNone(self.store.get(custom.id))

    def test_enabled_judge_items_follow_checkbox(self):
        before = [i.id for i in self.store.enabled_judge_items()]
        self.assertIn("ai_flavor", before)
        self.store.update("ai_flavor", enabled=False)
        after = [i.id for i in self.store.enabled_judge_items()]
        self.assertNotIn("ai_flavor", after)
        self.assertLess(len(after), len(before))

    def test_rewrite_hints_only_requested(self):
        hints = self.store.rewrite_hints(["end_period", "不存在的项"])
        self.assertIn("end_period", hints)
        self.assertEqual(len(hints), 1)

    def test_gen_helpers(self):
        self.assertIn("名字", gen_prompt("名字", "描述"))
        self.assertIn("描述", gen_prompt("名字", "描述"))
        self.assertIn("名字", gen_rewrite("名字", ""))

    def test_saved_file_is_json(self):
        self.store.save()
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(raw["version"], 1)
        self.assertIn("items", raw)


if __name__ == "__main__":
    unittest.main()
