"""配置读取助手（零第三方依赖，与工作区其他插件同构）。"""

from __future__ import annotations

from typing import Any, Mapping


def cfg_get(config: Mapping[str, Any] | None, path: str, default: Any = None) -> Any:
    """从（可能嵌套的）插件配置里安全地取一个值。"""
    current: Any = config
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return default
            current = current[part]
        else:
            return default
    return default if current is None else current


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    if value is None:
        return default
    return bool(value)


def as_int(value: Any, default: int = 0, low: int | None = None, high: int | None = None) -> int:
    """尽量宽容：数字字符串 / 浮点都认；**越界钳到边界**而不是回退默认。

    面板改参数（比如 mid_punct 的“连续多少字算缺标点”）最怕两件事：
    ① 明明填了 25 却因为类型问题被当成默认 15 —— 所以这里对 "25"、25.0 全部认；
    ② 填 9999 或 1 直接被扔回默认 —— 所以越界一律钳到边界，让“存进去的=生效的=回显的”。
    只有完全不是数字（空串 / None / 乱填）才回退默认。
    """
    if isinstance(value, bool):
        return int(value)
    try:
        result = int(float(value))
    except (TypeError, ValueError):
        return _clamp_int(default, low, high)
    return _clamp_int(result, low, high)


def _clamp_int(value: int, low: int | None, high: int | None) -> int:
    if low is not None and value < low:
        value = low
    if high is not None and value > high:
        value = high
    return value


def as_float(value: Any, default: float = 0.0, low: float | None = None, high: float | None = None) -> float:
    """与 as_int 同一套宽容语义：越界钳到边界，只有非法值才回退默认。"""
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    if result != result:  # NaN
        result = default
    if low is not None and result < low:
        result = low
    if high is not None and result > high:
        result = high
    return result


def as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def as_lines(value: Any) -> list[str]:
    """把配置里的"每行一条"文本 / 列表洗成去空行的字符串列表。"""
    if value is None:
        return []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []
