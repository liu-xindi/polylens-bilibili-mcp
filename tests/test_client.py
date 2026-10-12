"""输入解析、参数归一与能力入口。不联网。"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import patch

import pytest

from polylens_bilibili import client as client_mod
from polylens_bilibili.client import (
    BilibiliClient,
    _extract_page,
    _id_params,
    resolve_up,
    resolve_video,
    resolve_video_parts,
)
from polylens_bilibili.errors import BilibiliError, RateLimitedError
from polylens_bilibili.models import Page, UpInfo

BV_URL = "https://www.bilibili.com/video/BV1xx411c7mD/"


# ── 视频号解析 ──────────────────────────────────────────────────────────────


def test_resolve_video_extracts_bv_from_url() -> None:
    assert resolve_video(BV_URL)[0] == "BV1xx411c7mD"


def test_resolve_video_accepts_bare_id() -> None:
    """裸号直接可用，不必写成链接。"""
    assert resolve_video("BV1xx411c7mD")[0] == "BV1xx411c7mD"
    assert resolve_video("av170001")[0] == "av170001"


def test_resolve_video_extracts_from_share_text() -> None:
    assert resolve_video("看这个 av170001 真不错")[0] == "av170001"


def test_resolve_video_rejects_unparsable_input() -> None:
    with pytest.raises(BilibiliError):
        resolve_video("https://example.com/whatever")


def test_resolve_video_expands_short_link(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        def geturl(self) -> str:
            return BV_URL + "?p=2"

        def __enter__(self):
            return self

        def __exit__(self, *a: Any) -> None:
            return None

    monkeypatch.setattr(client_mod, "urlopen", lambda req, timeout=20: _Resp())
    video_id, page = resolve_video("https://b23.tv/abcdef")
    assert video_id == "BV1xx411c7mD"
    assert page == 2  # 展开后的链接里的 ?p= 同样生效


# ── 分段序号 ────────────────────────────────────────────────────────────────


def test_extract_page_from_query() -> None:
    assert _extract_page(BV_URL + "?p=2") == 2
    assert _extract_page("https://www.bilibili.com/video/BV1xx?spm_id=x&p=3&vd=y") == 3
    assert _extract_page("BV1xx?p=4") == 4


def test_extract_page_absent_or_non_positive() -> None:
    assert _extract_page(BV_URL) is None
    assert _extract_page("BV1xx?p=0") is None


def test_resolve_video_page_precedence() -> None:
    """显式传入 > 链接里的 ?p= > 第 1 段。"""
    assert resolve_video(BV_URL + "?p=3")[1] == 3
    assert resolve_video(BV_URL + "?p=3", 5)[1] == 5
    assert resolve_video(BV_URL)[1] == 1


@pytest.mark.parametrize("bad", [0, -3])
def test_resolve_video_rejects_non_positive_page(bad: int) -> None:
    with pytest.raises(BilibiliError):
        resolve_video(BV_URL, bad)


def test_link_with_zero_page_falls_back_instead_of_failing() -> None:
    """?p=0 是链接自带的，不是模型传的；这样的链接在网页上照样能打开，不该为它报错。"""
    assert resolve_video(BV_URL + "?p=0")[1] == 1


# ── 视频号 → 接口参数 ───────────────────────────────────────────────────────


def test_id_params_by_kind() -> None:
    assert _id_params("BV1xx411c7mD") == {"bvid": "BV1xx411c7mD"}
    assert _id_params("av170001") == {"aid": 170001}


def test_id_params_rejects_unknown_shape() -> None:
    with pytest.raises(BilibiliError):
        _id_params("xyz123")


# ── get_login_status ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("is_login", [True, False])
def test_get_login_status_reports_platform_answer(is_login: bool) -> None:
    with patch.object(client_mod, "fetch_nav", return_value=(("i", "s", is_login))) as nav:
        nav.return_value = type("N", (), {"is_login": is_login})()
        assert BilibiliClient().get_login_status() is is_login


@pytest.mark.parametrize(
    "failure",
    [OSError("network down"), BilibiliError("接口返回失败: -400"), KeyError("wbi_img")],
)
def test_get_login_status_returns_none_when_unverifiable(
    failure: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    """网络不可达、平台报错、响应改形状都归为 null，与"确定未登录"区分开；原因只进日志。"""
    with patch.object(client_mod, "fetch_nav", side_effect=failure), caplog.at_level("WARNING"):
        assert BilibiliClient().get_login_status() is None
    assert repr(failure) in caplog.records[0].getMessage()


def test_get_login_status_lets_unexpected_errors_surface() -> None:
    """代码缺陷不该被伪装成"平台不给答案"。"""
    with patch.object(client_mod, "fetch_nav", side_effect=TypeError("bug")):
        with pytest.raises(TypeError):
            BilibiliClient().get_login_status()


# ── 能力入口的参数处理 ──────────────────────────────────────────────────────


def test_comment_replies_with_empty_ids_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """没给 id 就没有可取的回复，返回空列表，不报错也不发请求。"""
    called: list[int] = []
    monkeypatch.setattr(
        BilibiliClient, "_view", lambda self, vid: called.append(1) or {}  # type: ignore[func-returns-value]
    )
    assert BilibiliClient().get_comment_replies("BV1xx", comment_ids=[], pages=1).threads == []
    assert not called


def _view_stub(duration: float | None = 100.0) -> dict[str, Any]:
    return {
        "aid": 1, "bvid": "BV1xx", "cid": 200, "title": "t",
        "stat": {}, "duration": duration,
    }


# ── 批量 ────────────────────────────────────────────────────────────────────


def _multi_view(bvid: str) -> dict[str, Any]:
    return {
        "aid": 1, "bvid": bvid, "title": bvid, "stat": {}, "duration": 90,
        "pages": [{"cid": 10 + p, "page": p, "part": f"P{p}", "duration": 30} for p in (1, 2, 3)],
    }


def test_video_infos_fetch_each_video_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """同一视频的多个分段共用一次 view；结果按输入顺序，重复的输入照样逐项返回。"""
    fetched: list[str] = []
    monkeypatch.setattr(
        BilibiliClient, "_view", lambda self, vid: fetched.append(vid) or _multi_view(vid)
    )
    inputs = ["BV1a?p=1", "BV1b?p=2", "BV1a?p=3", "BV1a?p=1"]
    results = BilibiliClient().get_video_infos(inputs)
    assert fetched == ["BV1a", "BV1b"]
    assert [r.input for r in results] == inputs
    assert [(r.value.id, r.value.current_page) for r in results if r.value] == [
        ("BV1a", 1), ("BV1b", 2), ("BV1a", 3), ("BV1a", 1),
    ]


def test_batch_keeps_going_after_an_item_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def view(self: Any, vid: str) -> dict[str, Any]:
        if vid == "BV1gone":
            raise BilibiliError("视频不存在：BV1gone")
        return _multi_view(vid)

    monkeypatch.setattr(BilibiliClient, "_view", view)
    results = BilibiliClient().get_video_infos(["BV1gone", "不是视频", "BV1a"])
    assert [r.error for r in results] == [
        "视频不存在：BV1gone", "无法从输入解析 BV/av 号：不是视频", None,
    ]
    assert results[2].value is not None


def _fake_counts(monkeypatch: pytest.MonkeyPatch, fail: dict[int, Exception]) -> list[int]:
    asked: list[int] = []

    def count(
        http: Any, aid: int, cid: int, duration: float | None, *, refresh: bool = False
    ) -> tuple[int, str | None]:
        asked.append(cid)
        if cid in fail:
            raise fail[cid]
        return cid * 10, None

    monkeypatch.setattr(BilibiliClient, "_view", lambda self, vid: _multi_view(vid))
    monkeypatch.setattr(client_mod, "fetch_danmaku_count", count)
    return asked


def test_danmaku_counts_whole_video_or_one_part(monkeypatch: pytest.MonkeyPatch) -> None:
    """不带 ?p= 取全部分段；带 ?p=N 只取那一段，各项独立成组。"""
    _fake_counts(monkeypatch, {})
    whole, one = BilibiliClient().get_danmaku_counts(["BV1a", "BV1a?p=2"])
    assert [(r.page, r.danmaku_count) for r in whole.value or []] == [(1, 110), (2, 120), (3, 130)]
    assert [(r.page, r.part, r.danmaku_count) for r in one.value or []] == [(2, "P2", 120)]


def test_danmaku_count_failure_stays_in_its_row(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_counts(monkeypatch, {12: BilibiliError("弹幕信息响应无法解析")})
    (result,) = BilibiliClient().get_danmaku_counts(["BV1a"])
    assert [(r.danmaku_count, r.error) for r in result.value or []] == [
        (110, None), (None, "弹幕信息响应无法解析"), (130, None),
    ]


def test_danmaku_counts_stop_requesting_after_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    asked = _fake_counts(monkeypatch, {11: RateLimitedError("取弹幕数触发风控（412），稍后重试。")})
    first, second = BilibiliClient().get_danmaku_counts(["BV1a", "BV1b"])
    assert asked == [11]
    rows = [*(first.value or []), *(second.value or [])]
    assert {r.error for r in rows} == {"取弹幕数触发风控（412），稍后重试。"}


@pytest.mark.parametrize(("expanded", "page"), [
    ("https://www.bilibili.com/video/BV1a?p=1", None),
    ("https://www.bilibili.com/video/BV1a?p=4", [(4, 4)]),
])
def test_resolve_video_part_from_short_link(
    monkeypatch: pytest.MonkeyPatch, expanded: str, page: list[tuple[int, int]] | None
) -> None:
    """短链展开后总带 p=1，分不清是否指定了第 1 段，按没指定算。"""
    monkeypatch.setattr(client_mod, "_expand_short_link", lambda text, cookie="": expanded)
    assert resolve_video_parts("https://b23.tv/abc") == ("BV1a", page)


def test_resolve_video_parts_page_only_when_written() -> None:
    assert resolve_video_parts("BV1a") == ("BV1a", None)
    assert resolve_video_parts("https://www.bilibili.com/video/BV1a/?p=3") == ("BV1a", [(3, 3)])


def test_resolve_video_parts_accepts_ranges() -> None:
    assert resolve_video_parts("BV1a?p=1-12") == ("BV1a", [(1, 12)])
    assert resolve_video_parts("BV1a?p=1-3,7") == ("BV1a", [(1, 3), (7, 7)])
    assert resolve_video_parts("看这个 BV1a?p=2-4，好看") == ("BV1a", [(2, 4)])


@pytest.mark.parametrize("spec", ["3-1", "0-2", "1-", "1--3", "1,,3"])
def test_resolve_video_parts_rejects_bad_range(spec: str) -> None:
    with pytest.raises(BilibiliError, match="无法识别的分段范围"):
        resolve_video_parts(f"BV1a?p={spec}")


def test_single_part_tools_reject_range() -> None:
    """范围以前会被读成第一个数字，静默只取那一段。"""
    with pytest.raises(BilibiliError, match="不支持范围"):
        resolve_video("BV1a?p=1-12")


def test_trailing_comma_after_page_is_punctuation() -> None:
    assert _extract_page("BV1a?p=3,") == 3


def test_danmaku_counts_by_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """按写的顺序取，重复的段只取一次。"""
    asked = _fake_counts(monkeypatch, {})
    (result,) = BilibiliClient().get_danmaku_counts(["BV1a?p=3,1-2,2"])
    assert [(r.page, r.part) for r in result.value or []] == [(3, "P3"), (1, "P1"), (2, "P2")]
    assert asked == [13, 11, 12]


def test_danmaku_range_past_last_part_fails_the_video(monkeypatch: pytest.MonkeyPatch) -> None:
    asked = _fake_counts(monkeypatch, {})
    (result,) = BilibiliClient().get_danmaku_counts(["BV1a?p=2-1000000"])
    assert result.error == "该视频共 3 段，没有第 1000000 段"
    assert asked == []


def test_batch_stops_requesting_after_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """触发风控后继续请求只会延长封禁，其余项记同一原因。"""
    fetched: list[int] = []

    def fetch(http: Any, mid: int) -> Any:
        fetched.append(mid)
        if mid == 2:
            raise RateLimitedError("被拦了")
        return UpInfo(mid=mid)

    monkeypatch.setattr(client_mod, "fetch_up_info", fetch)
    results = BilibiliClient().get_up_infos(["1", "2", "3", "4"])
    assert fetched == [1, 2]
    assert [r.error for r in results] == [None, "被拦了", "被拦了", "被拦了"]


def test_batch_reports_network_error_per_item(monkeypatch: pytest.MonkeyPatch) -> None:
    def fetch(http: Any, mid: int) -> Any:
        if mid == 1:
            raise ConnectionResetError("reset")
        return UpInfo(mid=mid)

    monkeypatch.setattr(client_mod, "fetch_up_info", fetch)
    results = BilibiliClient().get_up_infos(["1", "2"])
    assert results[0].error == "网络错误：reset"
    assert results[1].value == UpInfo(mid=2)


def test_up_infos_fetch_each_mid_once(monkeypatch: pytest.MonkeyPatch) -> None:
    fetched: list[int] = []
    monkeypatch.setattr(
        client_mod, "fetch_up_info", lambda http, mid: fetched.append(mid) or UpInfo(mid=mid)
    )
    results = BilibiliClient().get_up_infos(["42", "https://space.bilibili.com/42"])
    assert fetched == [42]
    assert [r.value.mid for r in results if r.value] == [42, 42]


# ── 短链展开 ────────────────────────────────────────────────────────────────


class _Resp:
    def __init__(self, url: str) -> None:
        self._url = url

    def geturl(self) -> str:
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *a: Any) -> None:
        return None


@pytest.mark.parametrize(
    "text",
    [
        "https://b23.tv/abcdef",
        "【标题】 https://b23.tv/abcdef",                       # App 分享按钮的默认格式
        "【标题】 https://b23.tv/abcdef?share_source=copy_web",
        "看这个 https://b23.tv/abcdef。后面还有中文",
        "【a】https://bili2233.cn/abcdef 附言",
    ],
)
def test_short_link_found_inside_share_text(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    """分享文案里链接前面有标题，不能把整段丢给 urlparse。"""
    seen: dict[str, Any] = {}

    def _fake(req, timeout=20):
        seen["url"] = req.full_url
        return _Resp(BV_URL)

    monkeypatch.setattr(client_mod, "urlopen", _fake)
    assert resolve_video(text) == ("BV1xx411c7mD", 1)
    assert "。" not in seen["url"] and "附言" not in seen["url"]  # 中文没被吃进链接


def test_short_link_keeps_the_page_written_by_the_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """短链跳转不转发 query，跳转目标还自带 p=1，只看展开后的链接会把段号盖成 1。"""
    monkeypatch.setattr(
        client_mod, "urlopen",
        lambda req, timeout=20: _Resp(BV_URL + "?share_source=copy&p=1&spmid=x"),
    )
    assert resolve_video("https://b23.tv/abcdef?p=150") == ("BV1xx411c7mD", 150)
    assert resolve_video("【标题】 https://b23.tv/abcdef?p=150") == ("BV1xx411c7mD", 150)


def test_short_link_falls_back_to_page_in_expanded_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """分段分享出来的短链自己不带 ?p=，段号在展开后的链接里。"""
    monkeypatch.setattr(
        client_mod, "urlopen", lambda req, timeout=20: _Resp(BV_URL + "?p=7&share_source=copy")
    )
    assert resolve_video("https://b23.tv/abcdef") == ("BV1xx411c7mD", 7)


def test_explicit_page_still_wins_over_both(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        client_mod, "urlopen", lambda req, timeout=20: _Resp(BV_URL + "?p=1")
    )
    assert resolve_video("https://b23.tv/abcdef?p=150", page=3) == ("BV1xx411c7mD", 3)


def test_short_link_carries_cookie(monkeypatch: pytest.MonkeyPatch) -> None:
    """短链走网页域名，平台对部分出口地址有风控，带登录 cookie 才放行。"""
    seen: dict[str, Any] = {}

    def _fake(req, timeout=20):
        seen["headers"] = dict(req.headers)
        return _Resp(BV_URL)

    monkeypatch.setattr(client_mod, "urlopen", _fake)
    resolve_video("https://b23.tv/abcdef", cookie="SESSDATA=x")
    assert seen["headers"].get("Cookie") == "SESSDATA=x"


def test_short_link_without_cookie_sends_no_cookie_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def _fake(req, timeout=20):
        seen["headers"] = dict(req.headers)
        return _Resp(BV_URL)

    monkeypatch.setattr(client_mod, "urlopen", _fake)
    resolve_video("https://b23.tv/abcdef")
    assert "Cookie" not in seen["headers"]


@pytest.mark.parametrize(
    "text", ["b23.tv/abcdef", "【标题】 b23.tv/abcdef 附言", "bili2233.cn/abcdef"]
)
def test_short_link_without_scheme(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    seen: dict[str, Any] = {}

    def _fake(req, timeout=20):
        seen["url"] = req.full_url
        return _Resp(BV_URL)

    monkeypatch.setattr(client_mod, "urlopen", _fake)
    assert resolve_video(text) == ("BV1xx411c7mD", 1)
    assert seen["url"].startswith("https://") and seen["url"].endswith("/abcdef")


def test_bare_host_inside_longer_domain_is_not_a_short_link() -> None:
    with pytest.raises(BilibiliError, match="无法从输入解析"):
        resolve_video("notb23.tv/abcdef")


def test_dead_short_link_is_reported_as_such(monkeypatch: pytest.MonkeyPatch) -> None:
    """失效短链不跳转，平台直接回 200，展开后还是短链本身。"""
    monkeypatch.setattr(
        client_mod, "urlopen", lambda req, timeout=20: _Resp("https://b23.tv/zzzzzzz")
    )
    with pytest.raises(BilibiliError, match="短链无效或已失效"):
        resolve_video("https://b23.tv/zzzzzzz")


def test_non_short_link_never_goes_online(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(req, timeout=20):
        raise AssertionError("不该联网")

    monkeypatch.setattr(client_mod, "urlopen", _boom)
    assert resolve_video(BV_URL)[0] == "BV1xx411c7mD"


# ── UP 主定位 ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "https://space.bilibili.com/3690981465524933",
        "https://space.bilibili.com/3690981465524933/upload/video",
        "space.bilibili.com/3690981465524933?spm_id_from=333.1007",
        " 3690981465524933 ",
    ],
)
def test_resolve_up_accepts_space_link_or_mid(text: str) -> None:
    assert resolve_up(text) == 3690981465524933


def test_resolve_up_rejects_other_input() -> None:
    with pytest.raises(BilibiliError):
        resolve_up("高中物理老何")


# ── 评论请求串行 ────────────────────────────────────────────────────────────


def test_comment_requests_run_one_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """并发时同一热度序会话会被交错推进，评论被拆散到几个请求里；故评论请求整个串行。"""
    active = peak = 0
    lock = threading.Lock()

    def fake_fetch(*args: Any, **kwargs: Any) -> Any:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return Page(items=[]) if "count" in kwargs else []

    monkeypatch.setattr(client_mod, "fetch_comments", fake_fetch)
    monkeypatch.setattr(client_mod, "fetch_replies", fake_fetch)
    monkeypatch.setattr(BilibiliClient, "_view", lambda self, vid: _view_stub())
    calls = [
        lambda: BilibiliClient().get_comments("BV1", count=1, sort="newest"),
        lambda: BilibiliClient().get_comments("BV2", count=1, sort="hot", batch_id="b"),
        lambda: BilibiliClient().get_comment_replies("BV3", comment_ids=["1"], pages=1),
    ] * 2
    threads = [threading.Thread(target=c) for c in calls]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == 1


def test_danmaku_requests_run_one_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """超时后立即重取的调用排在前一次后面，取完即命中缓存，不重复请求。"""
    active = peak = 0
    lock = threading.Lock()

    def count(*args: Any, **kwargs: Any) -> tuple[int, str | None]:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return 1, None

    monkeypatch.setattr(BilibiliClient, "_view", lambda self, vid: _multi_view(vid))
    monkeypatch.setattr(client_mod, "fetch_danmaku_count", count)
    threads = [
        threading.Thread(target=lambda: BilibiliClient().get_danmaku_counts(["BV1a"]))
        for _ in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == 1
