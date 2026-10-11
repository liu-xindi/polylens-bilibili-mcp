"""弹幕数：dm/web/view 接口（protobuf）→ 每段弹幕数。

官方播放器打开视频时请求这个接口，响应里字段 8 是该段弹幕数，平台按分段时长封顶。
只为取这一个数，手写最小的 protobuf 读取，不引依赖。
"""

from __future__ import annotations

from collections.abc import Iterator

from ..errors import BilibiliError, RateLimitedError
from ._constants import ENDPOINTS
from ._http import HttpClient, _RateLimited
from ._signing import fetch_nav, sign_params

_COUNT_FIELD = 8


def fetch_danmaku_count(client: HttpClient, aid: int, cid: int, duration: float | None) -> int:
    nav = fetch_nav(client)
    params = sign_params(
        {
            "type": 1,
            "oid": cid,
            "pid": aid,
            "duration": int(duration or 0),
            "without_subtitle": "true",
            "web_location": 1315873,
        },
        nav.img_key,
        nav.sub_key,
    )
    try:
        raw = client.get_api_bytes(ENDPOINTS["danmaku_view"], params)
    except _RateLimited as e:
        raise RateLimitedError(e.describe("取弹幕数")) from None
    return parse_danmaku_count(raw)


def parse_danmaku_count(raw: bytes) -> int:
    try:
        for number, value in _fields(raw):
            if number == _COUNT_FIELD and isinstance(value, int):
                return value
    except IndexError:
        raise BilibiliError("弹幕信息响应无法解析") from None
    return 0  # 没有弹幕时平台省略该字段（protobuf 默认值不编码）


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = buf[i]
        i += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return value, i


def _fields(buf: bytes) -> Iterator[tuple[int, int | bytes]]:
    """逐个给出顶层字段 (编号, 值)：varint 为整数，其余为原始字节。"""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
            yield number, value
        elif wire == 2:
            size, i = _varint(buf, i)
            yield number, buf[i : i + size]
            i += size
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            yield number, buf[i : i + size]
            i += size
        else:
            raise BilibiliError(f"弹幕信息响应无法解析（wire type {wire}）")
