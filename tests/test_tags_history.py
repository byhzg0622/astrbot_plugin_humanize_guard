"""标签剥离/还原 + 会话历史 + 打回日志。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.history import SessionHistory  # noqa: E402
from core.log import CheckLog  # noqa: E402
from core.tags import rejoin_tags, strip_message_markup, strip_tags  # noqa: E402


class TagStripping(unittest.TestCase):
    def test_affection_labels_stripped_and_collected(self):
        bare, tags = strip_tags("你真讨厌【好感度+5】")
        self.assertEqual(bare, "你真讨厌")
        self.assertEqual(tags, ["【好感度+5】"])

    def test_multiple_tag_styles(self):
        bare, tags = strip_tags("[好感度:80]早上好[ mood ]开心[ /mood ]")
        self.assertNotIn("好感度", bare)
        self.assertNotIn("mood", bare)
        self.assertGreaterEqual(len(tags), 2)

    def test_plain_text_untouched(self):
        bare, tags = strip_tags("普通的聊天消息")
        self.assertEqual(bare, "普通的聊天消息")
        self.assertEqual(tags, [])

    def test_extra_patterns_from_config(self):
        bare, tags = strip_tags("标记{好感}在这", extra_patterns=[r"\{[^}]{0,10}\}"])
        self.assertEqual(bare, "标记在这")
        self.assertTrue(any("好感" in t for t in tags))

    def test_bad_extra_pattern_ignored(self):
        bare, tags = strip_tags("原文在这", extra_patterns=["("])
        self.assertEqual(bare, "原文在这")
        self.assertEqual(tags, [])

    def test_rejoin_puts_tags_back(self):
        bare, tags = strip_tags("你真棒。【好感度+3】")
        self.assertEqual(bare, "你真棒。")
        self.assertEqual(rejoin_tags(bare, tags), "你真棒。【好感度+3】")

    def test_quoted_markup_stripped(self):
        text = strip_message_markup("他说啥<quoted_message>原文</quoted_message>？")
        self.assertNotIn("quoted_message", text)
        self.assertIn("？", text)


class HistoryCase(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.hist = SessionHistory(self.dir / "history.json", keep_k=3)
        self.hist.load()

    def test_record_and_context(self):
        self.hist.record("umo1", "第一条", user_prev="你好")
        self.hist.record("umo1", "第二条", user_prev="在吗")
        ctx = self.hist.context("umo1")
        self.assertEqual(ctx.prev, "第二条")
        self.assertEqual(ctx.recent[0], "第二条")
        self.assertEqual(ctx.user_prev, "在吗")

    def test_keep_k_trims(self):
        for i in range(6):
            self.hist.record("umo1", f"消息{i}")
        entry = self.hist.get("umo1")
        self.assertEqual(len(entry["recent"]), 3)
        self.assertEqual(entry["prev"], "消息5")

    def test_sessions_isolated(self):
        self.hist.record("a", "A的上一句")
        self.hist.record("b", "B的上一句")
        self.assertEqual(self.hist.context("a").prev, "A的上一句")
        self.assertEqual(self.hist.context("b").prev, "B的上一句")

    def test_persist_roundtrip(self):
        self.hist.record("umo1", "落盘这一条", user_prev="hi")
        again = SessionHistory(self.dir / "history.json", keep_k=3)
        again.load()
        self.assertEqual(again.context("umo1").prev, "落盘这一条")

    def test_empty_session_safe(self):
        ctx = self.hist.context("不存在的会话")
        self.assertEqual(ctx.prev, "")
        self.assertEqual(ctx.recent, [])

    def test_clear(self):
        self.hist.record("umo1", "x")
        removed = self.hist.clear()
        self.assertEqual(removed, 1)
        self.assertEqual(self.hist.stats()["sessions"], 0)


class LogCase(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.log = CheckLog(self.dir / "log.json")
        self.log.load()

    def test_add_and_recent(self):
        self.log.add(
            umo="umo1", action="rewrite",
            violations=[{"id": "end_period", "name": "句末有句号", "reason": "末尾是句号"}],
            original="你真棒。", final="你真棒", rounds=1,
        )
        entries = self.log.recent(10)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["action"], "rewrite")
        self.assertEqual(entries[0]["violations"][0]["name"], "句末有句号")
        stats = self.log.stats()
        self.assertEqual(stats["rejections"], 1)

    def test_ring_buffer_and_persist(self):
        for i in range(400):
            self.log.add(umo="u", action="dry", original=str(i))
        self.assertLessEqual(len(self.log.recent(1000)), 300)
        again = CheckLog(self.dir / "log.json")
        again.load()
        self.assertEqual(len(again.recent(1000)), len(self.log.recent(1000)))

    def test_clear(self):
        self.log.add(umo="u", action="pass")
        self.assertEqual(self.log.clear(), 1)
        self.assertEqual(self.log.recent(), [])


if __name__ == "__main__":
    unittest.main()
