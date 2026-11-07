"""面板后端：检测项管理、注入词预览、设置读写、日志与试跑。

挂载方式与工作区 ai_mind 插件同构（那套在 AstrBot v4.28 上实测可用）：
GET+POST 双注册、双命名空间（目录名与 dsh/插件名）、兜底路由收尾。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from astrbot.api import logger

PLUGIN_NAME = "astrbot_plugin_humanize_guard"
PLUGIN_ID = "dsh/" + PLUGIN_NAME
PLUGIN_VERSION = "v1.0.0"

try:
    from astrbot.api.web import error_response, json_response, request as web_request
except ImportError:  # 老版本没有这套助手：直接回 dict，FastAPI 自己序列化
    web_request = None  # type: ignore[assignment]

    def json_response(data: Any = None, **_kwargs: Any) -> Any:  # type: ignore[misc]
        return {} if data is None else data

    def error_response(message: str, **_kwargs: Any) -> Any:  # type: ignore[misc]
        return {"status": "error", "message": message, "data": None}


class WebApiMixin:
    """面板接口。要求宿主插件提供 .config/.context/.items/.history/.log 等。"""

    # ------------------------------------------------------------------
    # 路由注册
    # ------------------------------------------------------------------
    def _route_defs(self) -> list[tuple[str, Any, str]]:
        return [
            ("status", self._api_status, "总览状态"),
            ("items", self._api_items, "检测项列表"),
            ("items/save", self._api_items_save, "保存检测项（勾选/注入词/改写词/参数）"),
            ("items/add", self._api_items_add, "新建自定义检测项"),
            ("items/delete", self._api_items_delete, "删除自定义检测项"),
            ("items/reset", self._api_items_reset, "内置检测项恢复默认"),
            ("prompt/preview", self._api_prompt_preview, "预览拼装后的检测注入词"),
            ("settings", self._api_settings, "读取设置"),
            ("settings/save", self._api_settings_save, "保存设置"),
            ("logs", self._api_logs, "打回日志"),
            ("logs/clear", self._api_logs_clear, "清空日志"),
            ("test", self._api_test, "试跑检测"),
            ("rewrite_test", self._api_rewrite_test, "试跑检测 + 改写"),
            ("history", self._api_history, "会话历史（上一句）"),
        ]

    def _register_web_apis(self) -> None:
        context = getattr(self, "context", None)
        register = getattr(context, "register_web_api", None)
        if not callable(register):
            logger.warning(
                "[人味守卫] 这个 AstrBot 版本没有 context.register_web_api，"
                "面板后端不可用（聊天质检不受影响）"
            )
            return
        routes = self._route_defs()
        self._routes = {name: handler for name, handler, _ in routes}
        methods = ["GET", "POST"]
        namespaces = (PLUGIN_NAME, PLUGIN_ID)
        ok = 0
        for namespace in namespaces:
            for name, handler, desc in routes:
                try:
                    register(f"/{namespace}/{name}", handler, methods, f"质检面板：{desc}")
                    ok += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"[人味守卫] 注册面板接口 {namespace}/{name} 失败：{exc}")
            try:
                register(f"/{namespace}/<path:rest>", self._api_fallback, methods, "质检面板：兜底")
                ok += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[人味守卫] 注册兜底路由失败：{exc}")
        logger.info(
            f"[人味守卫] 面板后端已注册 {ok} 个路由（命名空间 {'、'.join(namespaces)}）"
        )

    async def _api_fallback(self, rest: str = "") -> Any:
        logger.debug(f"[人味守卫] 面板兜底路由收到未知路径：{rest}")
        return error_response(f"未知的面板接口：{rest}", status_code=404)

    # ------------------------------------------------------------------
    # 请求参数
    # ------------------------------------------------------------------
    async def _params(self) -> dict[str, Any]:
        """把查询串 + JSON body 拍平成一层参数表。

        注意：**不再**把 body 里的 "params" 键整体拍平上来。检测项的阈值参数
        就放在这个键里（{"id": "mid_punct", "params": {"max_run": 25}}），
        拍平会把 params 整个吞掉 —— 表现就是"面板保存了阈值，后端却回
        '没有要保存的字段'，参数永远停在默认值 15"。
        settings 页的扁平 dotted-path 不受影响（那边本来就没有嵌套 params）。
        """
        merged: dict[str, Any] = {}
        if web_request is None:
            return merged
        try:
            for key in list(web_request.query.keys()):
                merged[key] = web_request.query.get(key)
        except Exception:  # noqa: BLE001
            pass
        body = await self._read_body()
        if isinstance(body, dict):
            for wrapper in ("data", "query", "body"):
                inner = body.get(wrapper)
                if isinstance(inner, dict):
                    merged.update(inner)
            for key, value in body.items():
                # "params" 是检测项阈值参数的专用键，必须原样透传（值可以是 dict）；
                # 其余保留键跳过，避免重复展开
                if key in merged:
                    continue
                if key in ("data", "query", "body"):
                    continue
                merged[key] = value
        return merged

    async def _read_body(self) -> Any:
        if web_request is None:
            return None
        try:
            data = await web_request.json()
            if isinstance(data, dict) and data:
                return data
        except Exception:  # noqa: BLE001
            pass
        try:
            raw = await web_request.body()
        except Exception:  # noqa: BLE001
            return None
        if not raw:
            return None
        try:
            parsed = json.loads(raw.decode("utf-8"))
            if isinstance(parsed, dict):
                return parsed
        except Exception:  # noqa: BLE001
            return None
        return None

    @staticmethod
    def _want_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        return None

    @staticmethod
    def _coerce_param(raw: Any, spec: dict[str, Any]) -> Any:
        """按面板声明的类型把一个参数值洗成“存进去即生效”的形态。

        - int/float：数字字符串、浮点都认（"25"、25.0 → 25）；空值/乱填 → schema 默认；
          越界钳到声明范围（没有声明就不钳）。
        - lines：字符串按行拆；列表原样保留。
        - text：一律字符串。
        返回 None 表示“这个值没救，别写进 params”（免得把垃圾值存盘后一直生效不了）。
        """
        kind = str(spec.get("type") or "text").strip().lower()
        default = spec.get("default")
        if kind in {"int", "float"}:
            from ..core.config import as_int, as_float
            low = spec.get("low") if kind == "int" else spec.get("low")
            high = spec.get("high") if kind == "int" else spec.get("high")
            if isinstance(raw, str) and not raw.strip():
                raw = default
            if kind == "int":
                coerced = as_int(raw, as_int(default, 0, low, high), low, high)
            else:
                coerced = as_float(raw, as_float(default, 0.0, low, high), low, high)
            return coerced
        if kind == "lines":
            if isinstance(raw, list):
                return [str(x).strip() for x in raw if str(x).strip()]
            if isinstance(raw, str):
                return [line.strip() for line in raw.splitlines() if line.strip()]
            if raw in (None, ""):
                return []
            return None
        if isinstance(raw, (dict,)):
            return raw
        if raw is None:
            return None
        return str(raw)

    # ------------------------------------------------------------------
    # 总览
    # ------------------------------------------------------------------
    async def _api_status(self) -> Any:
        return json_response({
            "enabled": bool(self.enabled),
            "detect_rules": bool(self.enable_rules),
            "detect_judge": bool(self.enable_judge),
            "dry_run": bool(self.dry_run),
            "max_rounds": int(self.max_rounds),
            "model_mode": self.model_mode,
            "provider_id": self.provider_id,
            "timeout": float(self.model_timeout),
            "feedback": bool(self.feedback_enabled),
            "items": self.items.stats(),
            "history": self.history.stats(),
            "log": self.log.stats(),
            "blacklist": len(self.blacklist),
            "whitelist": len(self.whitelist),
            "whitelist_mode": bool(self.whitelist_mode),
            "feedback_pending": len(self._feedback),
            "fast_fix": bool(self.fast_fix),
            "recheck_judge": bool(self.recheck_judge),
            "segment_dedup": bool(self.dedup_fragments),
            "segment_wrap": bool(self._ctx_send_wrapped),
            "version": PLUGIN_VERSION,
        })

    # ------------------------------------------------------------------
    # 检测项
    # ------------------------------------------------------------------
    async def _api_items(self) -> Any:
        return json_response({
            "items": [item.to_dict() for item in self.items.list()],
            "stats": self.items.stats(),
        })

    @staticmethod
    def _param_schema_map(item_id: str) -> dict[str, Any]:
        """该检测项在面板上声明的参数表（key -> spec）。运行时兜底导入，避免循环依赖。"""
        try:
            from ..core.items import PARAM_SCHEMA
        except Exception:  # noqa: BLE001 - 老结构没有 PARAM_SCHEMA：按自由参数处理
            return {}
        return {str(spec.get("key")): spec for spec in PARAM_SCHEMA.get(str(item_id), [])}

    async def _api_items_save(self) -> Any:
        """保存检测项（勾选 / 注入词 / 改写词 / 阈值参数）。

        参数这里踩过一个真实的坑：面板是“只发改动过的字段”的增量提交
        （勾选只发 id+enabled），而后端如果**整包覆盖** params，
        任何一条没随请求回来的字段都会被旧值/默认值冲掉 ——
        表现就是“阈值在面板上改了，保存后一刷新又是 15”。
        所以现在的规矩：
        1) 没随请求携带的字段一律不覆盖（增量合并，保留注册表里的现值）；
        2) 随请求携带的值按 PARAM_SCHEMA 逐字段**类型清洗 + 范围钳制**，
           存进去的值 = 实际生效的值 = 下次回显的值，不再出现“存了 25 跑的是 15”。
        """
        params = await self._params()
        item_id = str(params.get("id") or "").strip()
        if not item_id:
            return error_response("缺少检测项 id", status_code=400)
        item = self.items.get(item_id)
        if item is None:
            return error_response(f"检测项不存在：{item_id}", status_code=404)
        kwargs: dict[str, Any] = {}
        enabled = self._want_bool(params.get("enabled"))
        if enabled is not None:
            kwargs["enabled"] = enabled
        for field in ("prompt", "rewrite", "name", "desc"):
            if field in params and params.get(field) is not None:
                kwargs[field] = str(params.get(field))
        incoming = params.get("params")
        incoming_dict: dict[str, Any] | None = None
        if isinstance(incoming, dict):
            incoming_dict = incoming
        elif isinstance(incoming, str) and incoming.strip():
            try:
                parsed = json.loads(incoming)
                if isinstance(parsed, dict):
                    incoming_dict = parsed
            except Exception:  # noqa: BLE001
                return error_response("params 需要是 JSON 对象", status_code=400)
        if incoming_dict is not None:
            merged: dict[str, Any] = dict(item.params or {})
            schema_map = self._param_schema_map(item_id)
            rejected: list[str] = []
            for key, value in incoming_dict.items():
                spec = schema_map.get(str(key))
                if spec is None:
                    # 面板没声明的自由参数：字符串值原样收，其他类型也保留
                    merged[str(key)] = value
                    continue
                coerced = self._coerce_param(value, spec)
                if coerced is None:
                    rejected.append(str(key))
                    continue
                merged[str(key)] = coerced
            kwargs["params"] = merged
            if rejected:
                logger.warning(
                    f"[人味守卫] 检测项 {item_id} 的参数 {rejected} 值非法，已忽略"
                )
        if not kwargs:
            return error_response("没有要保存的字段", status_code=400)
        try:
            item = self.items.update(item_id, **kwargs)
        except Exception as exc:  # noqa: BLE001
            return error_response(str(exc), status_code=400)
        logger.info(
            f"[人味守卫] 检测项已更新：{item.id}（{item.name}）"
            + (f"｜params={item.params}" if kwargs.get("params") is not None else "")
        )
        return json_response({"item": item.to_dict(), "stats": self.items.stats()})

    async def _api_items_add(self) -> Any:
        params = await self._params()
        name = str(params.get("name") or "").strip()
        desc = str(params.get("desc") or "").strip()
        if not name:
            return error_response("检测项名称不能为空", status_code=400)
        try:
            item = self.items.add_custom(
                name, desc,
                prompt=str(params.get("prompt") or ""),
                rewrite=str(params.get("rewrite") or ""),
            )
        except Exception as exc:  # noqa: BLE001
            return error_response(str(exc), status_code=400)
        logger.info(f"[人味守卫] 新建检测项：{item.id}（{item.name}）")
        return json_response({
            "item": item.to_dict(),
            "auto_prompt": not str(params.get("prompt") or "").strip(),
            "stats": self.items.stats(),
        })

    async def _api_items_delete(self) -> Any:
        params = await self._params()
        item_id = str(params.get("id") or "").strip()
        try:
            self.items.delete(item_id)
        except Exception as exc:  # noqa: BLE001
            return error_response(str(exc), status_code=400)
        return json_response({"ok": True, "stats": self.items.stats()})

    async def _api_items_reset(self) -> Any:
        restored = self.items.reset_builtin()
        return json_response({"ok": True, "restored": restored, "stats": self.items.stats()})

    async def _api_prompt_preview(self) -> Any:
        """把当前勾选检测项的注入词拼成完整提示词 —— 随勾选实时变化。"""
        params = await self._params()
        umo = str(params.get("umo") or "")
        sample = str(params.get("text") or "") or "今天天气真不错，我们一起去公园吧"
        ctx = self.history.context(umo) if umo else None
        if ctx is None:
            from ..core.model import RuleContext
            ctx = RuleContext()
        from ..core.judge import build_judge_prompt
        items = self.items.enabled_judge_items()
        built = build_judge_prompt(items, sample, ctx)
        return json_response({
            "system": built["system"],
            "prompt": built["prompt"],
            "ids": built["ids"],
            "count": len(items),
            "total": self.items.stats()["total"],
        })

    # ------------------------------------------------------------------
    # 设置（schema 驱动，取自 ai_mind 的成熟做法）
    # ------------------------------------------------------------------
    def _settings_schema(self) -> dict[str, Any]:
        if getattr(self, "_schema_cache", None) is not None:
            return self._schema_cache
        path = Path(__file__).resolve().parent.parent / "_conf_schema.json"
        try:
            self._schema_cache = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[人味守卫] 读不到 _conf_schema.json：{exc}")
            self._schema_cache = {}
        return self._schema_cache

    def _settings_leaf(self, path: str, child: dict[str, Any]) -> dict[str, Any]:
        from ..core.config import cfg_get
        return {
            "path": path,
            "key": path.rsplit(".", 1)[-1],
            "label": str(child.get("description") or path),
            "hint": str(child.get("hint") or ""),
            "type": str(child.get("type") or "string").strip().lower(),
            "options": [str(x) for x in (child.get("options") or [])],
            "value": cfg_get(self.config, path, child.get("default")),
            "default": child.get("default"),
        }

    def _settings_fields(self, spec: Any, prefix: str = "") -> list[dict[str, Any]]:
        items = spec.get("items") if isinstance(spec, dict) else None
        if not isinstance(items, dict):
            return []
        out: list[dict[str, Any]] = []
        for key, child in items.items():
            if not isinstance(child, dict):
                continue
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(child.get("type") or "").strip().lower() == "object":
                children = self._settings_fields(child, path)
                if not children:
                    continue
                out.append({
                    "path": path,
                    "key": str(key),
                    "label": str(child.get("description") or key),
                    "hint": str(child.get("hint") or ""),
                    "type": "object",
                    "options": [],
                    "children": children,
                })
                continue
            out.append(self._settings_leaf(path, child))
        return out

    def _settings_payload(self) -> dict[str, Any]:
        groups: list[dict[str, Any]] = []
        for key, spec in self._settings_schema().items():
            if not isinstance(spec, dict):
                continue
            if str(spec.get("type") or "").strip().lower() == "object":
                fields = self._settings_fields(spec, str(key))
            else:
                fields = [self._settings_leaf(str(key), spec)]
            if not fields:
                continue
            groups.append({
                "key": str(key),
                "label": spec.get("description") or key,
                "hint": spec.get("hint") or "",
                "fields": fields,
            })
        save = getattr(self.config, "save_config", None)
        return {
            "groups": groups,
            "writable": callable(save),
            "saved_at": float(getattr(self, "_settings_saved_at", 0.0) or 0.0),
        }

    def _set_config_value(self, path: str, value: Any) -> bool:
        from ..core.config import cfg_get  # noqa: F401 - 读回校验用
        parts = [p for p in str(path).split(".") if p]
        if not parts:
            return False
        nodes: list[Any] = [self.config]
        for part in parts[:-1]:
            child = nodes[-1].get(part) if hasattr(nodes[-1], "get") else None
            if not isinstance(child, dict):
                child = {}
            nodes.append(child)
        nodes[-1][parts[-1]] = value
        for index in range(len(parts) - 2, -1, -1):
            nodes[index][parts[index]] = nodes[index + 1]
        return True

    async def _api_settings(self) -> Any:
        return json_response(self._settings_payload())

    async def _api_settings_save(self) -> Any:
        from ..core.config import cfg_get
        params = await self._params()
        body = await self._read_body()
        payload: dict[str, Any] = body if isinstance(body, dict) else {}
        if not payload:
            payload = params
        values = payload.get("values")
        if not isinstance(values, dict) or not values:
            return error_response("没有收到任何要保存的设置", status_code=400)
        save = getattr(self.config, "save_config", None)
        if not callable(save):
            return error_response(
                "这个 AstrBot 版本的插件配置不支持回写，请到「配置」页修改", status_code=400
            )
        updates: dict[str, Any] = {}
        for path, value in values.items():
            self._set_config_value(str(path), value)
            updates[str(path)] = value
        try:
            save()
        except TypeError:
            try:
                save(dict(self.config))
            except Exception as exc:  # noqa: BLE001
                return error_response(f"写入配置文件失败：{exc}", status_code=500)
        except Exception as exc:  # noqa: BLE001
            return error_response(f"写入配置文件失败：{exc}", status_code=500)
        self._settings_saved_at = time.time()
        try:
            self._reload_config()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[人味守卫] 设置热更新失败：{exc}")
        # 逐项读回，写不进去的要报出来（防“改完自己弹回”）
        bad = []
        for path, value in updates.items():
            got = cfg_get(self.config, path, None)
            if isinstance(value, bool) or isinstance(got, bool):
                same = bool(value) == bool(got) and isinstance(value, bool) == isinstance(got, bool)
            elif isinstance(value, (int, float)) and isinstance(got, (int, float)):
                same = abs(float(value) - float(got)) < 1e-6
            else:
                same = str(value) == str(got)
            if not same:
                bad.append({"path": path, "wanted": value, "got": got})
        logger.info(f"[人味守卫] 面板保存了 {len(updates)} 项设置")
        result = self._settings_payload()
        result["saved"] = sorted(updates)
        result["not_saved"] = bad
        return json_response(result)

    # ------------------------------------------------------------------
    # 日志 / 试跑 / 历史
    # ------------------------------------------------------------------
    async def _api_logs(self) -> Any:
        params = await self._params()
        try:
            limit = int(float(params.get("limit") or 50))
        except (TypeError, ValueError):
            limit = 50
        return json_response({"entries": self.log.recent(limit), "stats": self.log.stats()})

    async def _api_logs_clear(self) -> Any:
        removed = self.log.clear()
        return json_response({"ok": True, "removed": removed})

    async def _api_test(self) -> Any:
        params = await self._params()
        text = str(params.get("text") or "").strip()
        if not text:
            return error_response("没有要检测的文本", status_code=400)
        umo = str(params.get("umo") or "")
        try:
            result = await self.qc_detect(text, umo)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[人味守卫] 试跑检测失败：{exc}", exc_info=True)
            return error_response(f"试跑失败：{exc}", status_code=500)
        return json_response(result)

    async def _api_rewrite_test(self) -> Any:
        params = await self._params()
        text = str(params.get("text") or "").strip()
        if not text:
            return error_response("没有要改写的文本", status_code=400)
        umo = str(params.get("umo") or "")
        try:
            result = await self.qc_rewrite_once(text, umo)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[人味守卫] 试跑改写失败：{exc}", exc_info=True)
            return error_response(f"试跑失败：{exc}", status_code=500)
        return json_response(result)

    async def _api_history(self) -> Any:
        params = await self._params()
        umo = str(params.get("umo") or "").strip()
        if not umo:
            return error_response("缺少 umo", status_code=400)
        entry = self.history.get(umo)
        return json_response({
            "umo": umo,
            "prev": entry.get("prev", ""),
            "recent": entry.get("recent", []),
            "user_prev": entry.get("user_prev", ""),
        })
