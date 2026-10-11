"""MCP 工具层：内存客户端断言工具清单、参数校验与返回结构。不联网。"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from polylens_bilibili import __version__
from polylens_bilibili import server as server_mod
from polylens_bilibili.api._subtitles import SubtitleTrack
from polylens_bilibili.errors import AuthRequiredError
from polylens_bilibili.models import (
    Batched,
    Comment,
    FeedItem,
    LoginCheckResult,
    Page,
    QrLoginSession,
    QrStatus,
    ReplyBatch,
    ReplyThread,
    SearchItem,
    SubtitleEntry,
    UpInfo,
    UpVideoItem,
    VideoInfo,
    VideoPart,
)

BV_URL = "https://www.bilibili.com/video/BV1xx411c7mD/"
_TOOL_NAMES = {
    "get_video_info", "get_parts", "get_comments", "get_comment_replies", "get_subtitles",
    "search_videos", "suggest_keywords", "list_up_videos", "get_up_info", "get_feed",
    "get_login_status", "logout", "start_qr_login", "complete_qr_login", "get_version",
}


def _run(coro_fn) -> Any:
    return anyio.run(coro_fn)


def _call(tool: str, args: dict[str, Any] | None = None) -> Any:
    """在内存会话里调一次工具，返回原始 CallToolResult。"""

    async def scenario() -> Any:
        server = create_server_for_test()
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            await client.initialize()
            return await client.call_tool(tool, args or {})

    return _run(scenario)


def create_server_for_test():
    return server_mod.create_server()


def _payload(tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    return _call(tool, args).structuredContent


class _StubClient:
    """替身：工具层只经 _client() 拿它，这里按用例需要覆写方法。"""

    def __init__(self, **behaviour: Any) -> None:
        self._behaviour = {"require_login": None, **behaviour}

    def __getattr__(self, name: str) -> Any:
        if name in self._behaviour:
            value = self._behaviour[name]
            return value if callable(value) else (lambda *a, **kw: value)
        raise AttributeError(name)


def _with_client(**behaviour: Any):
    return patch.object(server_mod, "_client", lambda: _StubClient(**behaviour))


# ── 工具清单与服务说明 ──────────────────────────────────────────────────────


def test_tool_list_is_exactly_the_declared_set() -> None:
    async def scenario() -> set[str]:
        server = create_server_for_test()
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            await client.initialize()
            return {t.name for t in (await client.list_tools()).tools}

    assert _run(scenario) == _TOOL_NAMES


def test_tools_declare_read_and_world_hints() -> None:
    """取数据的工具标只读；写本地凭据的不标只读；只碰本地的不标 openWorld。"""

    async def scenario() -> dict[str, Any]:
        server = create_server_for_test()
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            await client.initialize()
            return {t.name: t.annotations for t in (await client.list_tools()).tools}

    ann = _run(scenario)
    assert ann["get_comments"].readOnlyHint is True
    assert ann["get_comments"].openWorldHint is True
    assert ann["logout"].openWorldHint is False
    assert ann["complete_qr_login"].readOnlyHint is False


def test_content_tools_declare_url_and_page() -> None:
    async def scenario() -> dict[str, Any]:
        server = create_server_for_test()
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            await client.initialize()
            return {t.name: t.inputSchema for t in (await client.list_tools()).tools}

    schemas = _run(scenario)
    assert "page" in schemas["get_subtitles"]["properties"]
    # 评论按整片取，与分段无关；批量工具的分段写在每项的 ?p= 里
    for name in ("get_comments", "get_comment_replies", "get_video_info", "get_parts"):
        assert "page" not in schemas[name]["properties"], name


# ── 内容类工具的返回结构 ────────────────────────────────────────────────────


def test_get_video_info_one_row_per_input() -> None:
    results = [
        Batched("BV1a", VideoInfo(id="BV1a", title="标题", view_count=0)),
        Batched("BV1bad", error="视频不存在：BV1bad"),
    ]
    with _with_client(get_video_infos=results):
        payload = _payload(
            "get_video_info",
            {"videos": ["BV1a", "BV1bad"], "jq": "map({input, id, view_count, error})"},
        )
    assert payload["count"] == 2 and payload["failed"] == 1
    assert payload["videos"] == (
        "videos[2]{input,id,view_count,error}:\n"
        "  BV1a,BV1a,0,\n"
        "  BV1bad,,,视频不存在：BV1bad"
    )
    assert payload["jq_count"] == 2
    assert payload["elapsed_s"] >= 0


def test_get_video_info_columns_wrap_fields_with_input_and_error() -> None:
    with _with_client(get_video_infos=[Batched("BV1a", VideoInfo(id="BV1a", title="t"))]):
        payload = _payload("get_video_info", {"videos": ["BV1a"], "jq": "."})
    header = payload["videos"].splitlines()[0]
    assert header.startswith("videos[1]{input,id,title,")
    assert header.endswith(",current_page,current_part,error}:")


def _comment(cid: str, content: str, **kw: Any) -> Comment:
    base = Comment(
        id=cid, author="u", author_url=None, author_level=None, is_up=False,
        ip_location=None, content=content, like_count=0, reply_count=0,
        parent_id=None, created_at=None, is_top=False, up_liked=False,
        image_urls=None, link_titles=None,
    )
    return replace(base, **kw)


def test_get_comments_returns_toon_and_paging() -> None:
    page = Page(items=[_comment("1", "a"), _comment("2", "b")], has_more=True, next_cursor="tok")
    with _with_client(get_comments=page):
        payload = _payload("get_comments", {"jq": ".", "url": BV_URL, "batch_id": "b1", "count": 2})
    assert payload["video_id"] == "BV1xx411c7mD"
    assert payload["count"] == 2
    assert payload["has_more"] is True
    assert payload["next_cursor"] == "tok"
    assert payload["comments"].startswith(
        "comments[2]{id,author,author_url,author_level,is_up,ip_location,content,like_count,"
        "reply_count,parent_id,created_at,is_top,up_liked,image_urls,link_titles}:"
    )


def test_get_comments_rate_limited_partial_carries_message() -> None:
    page = Page(items=[], has_more=True, next_cursor="SESSION", rate_limited="只取到部分")
    with _with_client(get_comments=page):
        payload = _payload(
            "get_comments", {"jq": ".", "url": BV_URL, "batch_id": "b1", "count": 40}
        )
    assert payload["next_cursor"] == "SESSION"
    assert payload["message"] == "只取到部分"


def test_get_comments_omits_next_cursor_at_end() -> None:
    with _with_client(get_comments=Page(items=[], has_more=False)):
        payload = _payload("get_comments", {"jq": ".", "url": BV_URL, "batch_id": "b1", "count": 5})
    assert payload["has_more"] is False
    assert payload["next_cursor"] is None
    assert payload["message"] is None
    assert payload["comments"].startswith("comments[0]{")


def test_get_comment_replies_groups_by_thread() -> None:
    threads = [
        ReplyThread("1", Page(items=[_comment("11", "x")], has_more=True)),
        ReplyThread("2", Page(items=[], has_more=False)),
    ]
    with _with_client(get_comment_replies=ReplyBatch(threads)):
        payload = _payload(
            "get_comment_replies", {"jq": ".", "url": BV_URL, "comment_ids": ["1", "2"], "pages": 1}
        )
    results = payload["results"]
    assert [r["comment_id"] for r in results] == ["1", "2"]
    assert results[0]["has_more"] is True
    assert "next_cursor" not in results[0]
    assert results[1]["has_more"] is False
    assert results[0]["replies"].startswith("replies[1]{")
    assert "reply_count" not in results[0]["replies"]


def test_get_comment_replies_surfaces_thread_error() -> None:
    threads = [
        ReplyThread("1", Page(items=[_comment("11", "x")])),
        ReplyThread("2", Page(items=[]), error="接口返回失败: 12006 没有该评论"),
    ]
    with _with_client(get_comment_replies=ReplyBatch(threads)):
        payload = _payload(
            "get_comment_replies", {"jq": ".", "url": BV_URL, "comment_ids": ["1", "2"], "pages": 1}
        )
    assert [r["error"] for r in payload["results"]] == [None, "接口返回失败: 12006 没有该评论"]


def test_get_comment_replies_surfaces_withheld() -> None:
    """平台扣下的回复条数按主评论逐条给出，让调用方知道引用链可能断在哪。"""
    threads = [
        ReplyThread("1", Page(items=[_comment("11", "x")], has_more=False), withheld=6, total=34),
        ReplyThread("2", Page(items=[], has_more=False)),
    ]
    with _with_client(get_comment_replies=ReplyBatch(threads)):
        payload = _payload(
            "get_comment_replies", {"jq": ".", "url": BV_URL, "comment_ids": ["1", "2"], "pages": 1}
        )
    assert [r["withheld"] for r in payload["results"]] == [6, 0]
    assert [r["total"] for r in payload["results"]] == [34, None]


def test_get_subtitles_returns_toon_with_lang_info() -> None:
    track = SubtitleTrack([SubtitleEntry(start=1.0, end=2.0, content="一句")], "ai-zh",
                          ["en-US", "ai-zh"])
    with _with_client(get_subtitles=track):
        payload = _payload("get_subtitles", {"jq": ".", "url": BV_URL})
    assert payload["count"] == 1
    assert payload["lang"] == "ai-zh"  # 默认取首条时也告知实际语种
    assert payload["available_langs"] == ["en-US", "ai-zh"]
    assert payload["subtitles"] == "subtitles[1]{start,end,content}:\n  1,2,一句"


def test_get_subtitles_passes_lang_through() -> None:
    seen: dict[str, object] = {}

    def _capture(video_id, page=1, lang=None):
        seen.update(video_id=video_id, page=page, lang=lang)
        return SubtitleTrack([], "en-US", ["en-US"])

    with _with_client(get_subtitles=_capture):
        _payload("get_subtitles", {"jq": ".", "url": BV_URL + "?p=3", "lang": "en-US"})
    assert seen == {"video_id": "BV1xx411c7mD", "page": 3, "lang": "en-US"}


def test_get_parts_groups_by_video() -> None:
    parts = [VideoPart(page=1, part="片头", duration=60.0),
             VideoPart(page=2, part="正片", duration=600.0)]
    results = [Batched("BV1a", parts), Batched("BV1bad", error="视频不存在：BV1bad")]
    with _with_client(get_parts=results):
        payload = _payload("get_parts", {"jq": ".", "videos": ["BV1a", "BV1bad"]})
    ok, bad = payload["results"]
    assert ok["input"] == "BV1a" and ok["count"] == 2
    assert ok["parts"] == "parts[2]{page,part,duration}:\n  1,片头,60\n  2,正片,600"
    assert ok["error"] is None
    assert bad == {"input": "BV1bad", "count": 0, "parts": None, "jq_count": None,
                   "error": "视频不存在：BV1bad"}


def test_get_parts_jq_runs_per_video() -> None:
    results = [
        Batched("BV1a", [VideoPart(page=1, part="甲", duration=1.0)]),
        Batched("BV1b", [VideoPart(page=1, part="乙", duration=1.0),
                         VideoPart(page=2, part="丙", duration=1.0)]),
    ]
    with _with_client(get_parts=results):
        payload = _payload("get_parts", {"jq": "length", "videos": ["BV1a", "BV1b"]})
    assert [r["parts"] for r in payload["results"]] == ["1", "2"]


def test_search_returns_toon_and_paging() -> None:
    page = Page(
        items=[
            SearchItem(title="t1", url="u1", author=None, author_url=None, published_at=None,
                       duration_sec=None, category=None, tags=None, summary=None,
                       view_count=9, danmaku_count=None, comment_count=None, like_count=None,
                       favorite_count=None)
        ],
        has_more=True,
        next_cursor="1",
    )
    with _with_client(search=page):
        payload = _payload("search_videos", {"jq": ".", "query": "py"})
    assert payload["count"] == 1
    assert payload["has_more"] is True and payload["next_cursor"] == "1"
    assert payload["results"].startswith(
        "results[1]{title,url,author,author_url,published_at,duration_sec,category,tags,summary,"
        "view_count,danmaku_count,comment_count,like_count,favorite_count}:"
    )


def test_search_passes_order_through() -> None:
    seen: dict[str, Any] = {}

    def search(**kw: Any) -> Page[SearchItem]:
        seen.update(kw)
        return Page(items=[])

    with _with_client(search=search):
        _payload("search_videos", {"jq": ".", "query": "py", "order": "most_danmaku"})
    assert seen["order"] == "most_danmaku"


# ── 内联图片 ────────────────────────────────────────────────────────────────


def test_start_qr_login_returns_inline_qr_image() -> None:
    """二维码作为图片直接返回，不依赖运行服务的机器有图形界面。"""
    session = QrLoginSession(key="k1", url="https://passport.bilibili.com/qr/k1")
    with patch.object(server_mod.BilibiliClient, "start_qr_login", lambda self: session):
        result = _call("start_qr_login")
    assert result.content[0].type == "image"
    assert result.content[0].mimeType == "image/png"
    assert result.content[0].data  # base64 PNG
    meta = json.loads(result.content[-1].text)
    assert meta["key"] == "k1"
    assert meta["url"] == session.url


# ── 版本 ────────────────────────────────────────────────────────────────────


def test_get_version_description_carries_startup_version() -> None:
    async def scenario() -> str:
        server = create_server_for_test()
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            await client.initialize()
            tools = {t.name: t for t in (await client.list_tools()).tools}
            return tools["get_version"].description or ""

    assert f"本说明对应版本 {__version__}。" in _run(scenario)


def test_get_version_reports_running_and_latest() -> None:
    with patch.object(server_mod, "_latest_release", lambda: "9.9.9"):
        assert _payload("get_version") == {"running": __version__, "latest": "9.9.9"}


def test_latest_release_is_none_when_pypi_unreachable() -> None:
    def fail(*_a: Any, **_k: Any) -> Any:
        raise OSError("offline")

    with patch.object(server_mod, "urlopen", fail):
        assert server_mod._latest_release() is None


# ── 登录类工具 ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("answer", [True, False, None])
def test_get_login_status_passes_through_three_states(answer: bool | None) -> None:
    with _with_client(get_login_status=answer):
        assert _payload("get_login_status")["is_login"] is answer


def test_logout_reports_whether_credential_existed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "cookie"
    monkeypatch.setattr("polylens_bilibili.credentials.cookie_file_path", lambda: path)
    assert _payload("logout")["deleted"] is False
    path.write_text("x")
    assert _payload("logout")["deleted"] is True
    assert not path.exists()


def test_complete_qr_login_success_saves_cookie(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "cookie"
    monkeypatch.setattr("polylens_bilibili.credentials.cookie_file_path", lambda: path)
    result = LoginCheckResult(status=QrStatus.SUCCESS, cookie="SESSDATA=ok")
    with patch.object(server_mod.BilibiliClient, "check_qr_login", lambda self, key: result):
        payload = _payload("complete_qr_login", {"key": "k1"})
    assert payload["status"] == "success"
    assert path.read_text() == "SESSDATA=ok"


@pytest.mark.parametrize("status", [QrStatus.WAITING, QrStatus.SCANNED, QrStatus.EXPIRED])
def test_complete_qr_login_pending_states(status: QrStatus) -> None:
    result = LoginCheckResult(status=status)
    with patch.object(server_mod.BilibiliClient, "check_qr_login", lambda self, key: result):
        payload = _payload("complete_qr_login", {"key": "k1"})
    assert payload["status"] == status.value


# ── 登录 ────────────────────────────────────────────────────────────────────

_PLATFORM_CALLS = [
    ("get_video_info", {"videos": ["BV1a"], "jq": "."}),
    ("get_parts", {"videos": ["BV1a"], "jq": "."}),
    ("get_up_info", {"authors": ["2"], "jq": "."}),
    ("get_comments", {"url": BV_URL, "count": 1, "batch_id": "b1", "jq": "."}),
    ("get_comment_replies", {"url": BV_URL, "comment_ids": ["1"], "pages": 1, "jq": "."}),
    ("get_subtitles", {"url": BV_URL, "jq": "."}),
    ("search_videos", {"query": "x", "jq": "."}),
    ("suggest_keywords", {"term": "x"}),
    ("list_up_videos", {"author_url": "2", "jq": "."}),
    ("get_feed", {"jq": "."}),
]


class _LoggedOut:
    """require_login 报未登录；其余方法一旦被调用就记下来。"""

    def __init__(self) -> None:
        self.used: list[str] = []

    def require_login(self) -> None:
        raise AuthRequiredError("all")

    def __getattr__(self, name: str) -> Any:
        self.used.append(name)
        raise AssertionError(f"未登录时不应调用 {name}")


def test_platform_tools_cover_every_login_tool() -> None:
    exempt = {"get_login_status", "logout", "start_qr_login", "complete_qr_login", "get_version"}
    assert {name for name, _ in _PLATFORM_CALLS} == _TOOL_NAMES - exempt


@pytest.mark.parametrize(("tool", "args"), _PLATFORM_CALLS)
def test_platform_tools_refuse_when_logged_out(tool: str, args: dict[str, Any]) -> None:
    stub = _LoggedOut()
    with patch.object(server_mod, "_client", lambda: stub):
        result = _call(tool, args)
    assert result.isError
    assert "需要登录" in result.content[0].text
    assert stub.used == []


def test_login_free_tools_skip_the_check() -> None:
    stub = _LoggedOut()
    stub.get_login_status = lambda: False  # type: ignore[method-assign]
    with patch.object(server_mod, "_client", lambda: stub), \
            patch.object(server_mod, "_latest_release", lambda: None):
        assert _payload("get_login_status") == {"is_login": False}
        assert _payload("get_version")["latest"] is None


def test_elapsed_s_includes_login_check() -> None:
    def slow_check() -> None:
        time.sleep(0.3)

    with _with_client(require_login=slow_check, suggest=["x"]):
        payload = _payload("suggest_keywords", {"term": "x"})
    assert payload["elapsed_s"] >= 0.3


def test_tool_reuses_the_client_that_checked_login() -> None:
    """核验与取数据用同一个 client，nav 的结果才能在本次调用内复用。"""
    made: list[_StubClient] = []

    def factory(cookie: str = "") -> _StubClient:
        made.append(_StubClient(suggest=["x"]))
        return made[-1]

    with patch.object(server_mod, "BilibiliClient", factory), \
            patch.object(server_mod, "load_cookie", lambda: ""):
        _payload("suggest_keywords", {"term": "x"})
        _payload("suggest_keywords", {"term": "x"})
    assert len(made) == 2  # 每次调用一个，调用之间不复用


# ── 错误上浮 ────────────────────────────────────────────────────────────────


def test_unparsable_url_is_tool_error() -> None:
    result = _call("get_video_info", {"url": "https://example.com/x"})
    assert result.isError


def test_tool_outcomes_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    """mcp 库只把异常转成错误结果，不记日志；服务端的记录全靠工具层。"""

    def broken(*_a: Any, **_kw: Any) -> None:
        raise KeyError("stat")

    with caplog.at_level("INFO", logger="polylens_bilibili"):
        with _with_client(get_login_status=True):
            _call("get_login_status")
        with _with_client():
            _call("get_subtitles", {"url": "https://example.com/x", "jq": "."})
        with _with_client(get_subtitles=broken):
            assert _call("get_subtitles", {"url": BV_URL, "jq": "."}).isError
    ok, failed, crashed = [r for r in caplog.records if r.name == "polylens_bilibili.server"]
    assert ok.levelname == "INFO" and "get_login_status" in ok.getMessage()
    assert failed.levelname == "WARNING" and "无法从输入解析" in failed.getMessage()
    assert failed.exc_info is None
    assert crashed.levelname == "ERROR" and crashed.exc_info is not None


# ── jq ──────────────────────────────────────────────────────────────────────


def test_list_tools_accept_jq() -> None:
    async def scenario() -> dict[str, Any]:
        server = create_server_for_test()
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            await client.initialize()
            return {t.name: t.inputSchema for t in (await client.list_tools()).tools}

    schemas = _run(scenario)
    with_jq = {name for name, schema in schemas.items() if "jq" in schema["properties"]}
    assert with_jq == {
        "search_videos", "list_up_videos", "get_feed", "get_comments", "get_subtitles",
        "get_comment_replies", "get_parts", "get_video_info", "get_up_info",
    }
    subtitles_desc = schemas["get_subtitles"]["properties"]["jq"]["description"]
    assert "start,end,content" in subtitles_desc
    assert "用不到的字段：start（需要定位时间时保留）、end；" in subtitles_desc
    assert "用不到的字段" not in schemas["get_parts"]["properties"]["jq"]["description"]


def test_get_comments_jq_keeps_count_and_paging() -> None:
    page = Page(items=[_comment("1", "a"), _comment("2", "b")], has_more=True, next_cursor="tok")
    with _with_client(get_comments=page):
        payload = _payload(
            "get_comments",
            {
                "url": BV_URL, "count": 2, "batch_id": "b1",
                "jq": '[.[] | select(.content == "b") | {id, content}]',
            },
        )
    assert payload["count"] == 2
    assert payload["jq_count"] == 1
    assert payload["comments"] == "comments[1]{id,content}:\n  \"2\",b"
    assert payload["has_more"] is True
    assert payload["next_cursor"] == "tok"


def test_get_comment_replies_jq_runs_per_thread() -> None:
    threads = [
        ReplyThread("1", Page(items=[_comment("11", "x"), _comment("12", "y")])),
        ReplyThread("2", Page(items=[_comment("21", "z")])),
    ]
    with _with_client(get_comment_replies=ReplyBatch(threads)):
        payload = _payload(
            "get_comment_replies",
            {"url": BV_URL, "comment_ids": ["1", "2"], "pages": 1, "jq": "[.[] | .content]"},
        )
    assert [r["replies"] for r in payload["results"]] == ['["x","y"]', '["z"]']
    assert [r["jq_count"] for r in payload["results"]] == [2, 1]


def test_get_subtitles_jq_text_only() -> None:
    entries = [SubtitleEntry(start=1.0, end=2.0, content="一句"),
               SubtitleEntry(start=3.0, end=4.0, content="二句")]
    with _with_client(get_subtitles=SubtitleTrack(entries, "ai-zh", ["ai-zh"])):
        payload = _payload(
            "get_subtitles", {"url": BV_URL, "jq": 'map(.content) | join("\\n")'}
        )
    assert payload["subtitles"] == "一句\n二句"
    assert payload["count"] == 2
    assert payload["jq_count"] is None


def test_jq_is_required() -> None:
    with _with_client(get_feed=[]):
        result = _call("get_feed")
    assert result.isError
    assert "jq" in result.content[0].text


def test_bad_jq_is_tool_error_with_fields() -> None:
    parts = [VideoPart(page=1, part="片头", duration=60.0)]
    with _with_client(get_parts=[Batched("BV1a", parts)]):
        result = _call("get_parts", {"videos": ["BV1a"], "jq": ".["})
    assert result.isError
    assert "page,part,duration" in result.content[0].text


def test_bad_jq_on_batch_lists_input_and_error_columns() -> None:
    with _with_client(get_up_infos=[Batched("42", UpInfo(mid=42))]):
        result = _call("get_up_info", {"authors": ["42"], "jq": ".["})
    assert result.isError
    assert "可用字段：input,mid," in result.content[0].text
    assert ",face_url,error。" in result.content[0].text


# ── 并发 ────────────────────────────────────────────────────────────────────


def test_slow_tool_does_not_block_other_tools() -> None:
    """评论限速排队这类慢调用不能卡住整个服务（10-08 实测：快工具陪等两分钟后被客户端判超时）。"""
    release = threading.Event()

    def slow_comments(*args: Any, **kwargs: Any) -> Page[Comment]:
        release.wait(5)
        return Page(items=[])

    async def scenario() -> list[str]:
        order: list[str] = []
        server = create_server_for_test()
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            await client.initialize()

            async def slow() -> None:
                await client.call_tool(
                    "get_comments", {"url": BV_URL, "batch_id": "b1", "count": 1, "jq": "."}
                )
                order.append("slow")

            async with anyio.create_task_group() as tg:
                tg.start_soon(slow)
                await anyio.sleep(0.2)
                await client.call_tool("suggest_keywords", {"term": "x"})
                order.append("fast")
                release.set()
        return order

    with _with_client(get_comments=slow_comments, suggest=["x"]):
        assert _run(scenario) == ["fast", "slow"]


# ── elapsed_s ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("tool", "args", "behaviour"),
    [
        ("get_video_info", {"videos": [BV_URL], "jq": "."}, {"get_video_infos": []}),
        (
            "get_comments", {"jq": ".", "url": BV_URL, "batch_id": "b1", "count": 1},
            {"get_comments": Page(items=[])},
        ),
        (
            "get_subtitles", {"jq": ".", "url": BV_URL},
            {"get_subtitles": SubtitleTrack([], None, [])},
        ),
        ("search_videos", {"jq": ".", "query": "x"}, {"search": Page(items=[])}),
    ],
)
def test_content_tools_attach_elapsed_s(
    tool: str, args: dict[str, Any], behaviour: dict[str, Any]
) -> None:
    with _with_client(**behaviour):
        payload = _payload(tool, args)
    assert payload["elapsed_s"] >= 0
    assert next(iter(payload)) == "elapsed_s"


def test_login_tools_have_no_elapsed_s() -> None:
    with _with_client(get_login_status=True):
        assert "elapsed_s" not in _payload("get_login_status")


# ── 排序方式与整片时长 ──────────────────────────────────────────────────────


def test_get_comments_exposes_sort_modes() -> None:
    """对外是语义名，平台的 mode 编码不出现在 schema 里。"""

    async def scenario() -> dict[str, Any]:
        server = create_server_for_test()
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            await client.initialize()
            tools = {t.name: t for t in (await client.list_tools()).tools}
            return tools["get_comments"].inputSchema["properties"]["mode"]

    spec = _run(scenario)
    assert set(spec.get("enum") or []) == {"hot", "newest"}
    assert spec.get("default") == "hot"


def test_hot_comments_require_batch_id() -> None:
    called: list[int] = []
    with _with_client(get_comments=lambda *a, **kw: called.append(1) or Page(items=[])):
        result = _call("get_comments", {"url": BV_URL, "count": 1, "jq": "."})
    assert result.isError
    assert "batch_id" in result.content[0].text
    assert not called


def test_get_comments_passes_sort_through() -> None:
    seen: dict[str, object] = {}

    def _capture(video_id, *, count, cursor=None, sort="hot", batch_id=None, refresh=False):
        seen.update(video_id=video_id, count=count, sort=sort)
        return Page(items=[])

    with _with_client(get_comments=_capture):
        _payload("get_comments", {"jq": ".", "url": BV_URL, "count": 5, "mode": "newest"})
    assert seen["sort"] == "newest"


def test_get_comments_passes_batch_id_and_refresh() -> None:
    seen: dict[str, object] = {}

    def _capture(video_id, **kwargs):
        seen.update(kwargs)
        return Page(items=[], from_cache=True, cached_at="2026-10-09 12:00")

    with _with_client(get_comments=_capture):
        payload = _payload(
            "get_comments",
            {"jq": ".", "url": BV_URL, "count": 5, "mode": "newest", "refresh": True},
        )
    assert seen["refresh"] is True and seen["batch_id"] is None
    assert payload["from_cache"] is True and payload["cached_at"] == "2026-10-09 12:00"


def test_get_comment_replies_reports_cache_once_per_call() -> None:
    seen: dict[str, object] = {}

    def _capture(video_id, **kwargs):
        seen.update(kwargs)
        threads = [ReplyThread("1", Page(items=[_comment("11", "x")]))]
        return ReplyBatch(threads, from_cache=True, cached_at="2026-10-09 12:00")

    with _with_client(get_comment_replies=_capture):
        payload = _payload(
            "get_comment_replies",
            {"jq": ".", "url": BV_URL, "comment_ids": ["1"], "pages": 1, "refresh": True},
        )
    assert seen["refresh"] is True
    assert payload["from_cache"] is True and payload["cached_at"] == "2026-10-09 12:00"
    assert "cached_at" not in payload["results"][0]


def test_get_video_info_reports_total_duration() -> None:
    info = VideoInfo(id="BV1", title="t", duration_sec=79.0, total_duration_sec=81976.0)
    with _with_client(get_video_infos=[Batched("BV1", info)]):
        payload = _payload(
            "get_video_info",
            {"videos": ["BV1"], "jq": ".[0] | [.duration_sec, .total_duration_sec]"},
        )
    assert payload["videos"] == "[79,81976]"


# ── 搜索联想 ────────────────────────────────────────────────────────────────


def test_suggest_keywords_returns_list() -> None:
    with _with_client(suggest=["python", "pycharm"]):
        payload = _payload("suggest_keywords", {"term": "py"})
    assert payload["count"] == 2
    assert payload["suggestions"] == ["python", "pycharm"]
    assert payload["elapsed_s"] >= 0


# ── UP 主投稿 ───────────────────────────────────────────────────────────────


def test_list_up_videos_resolves_link_and_returns_toon() -> None:
    seen: dict[str, Any] = {}

    def get_up_videos(mid: int, **kw: Any) -> tuple[str, int, Page[UpVideoItem]]:
        seen.update(mid=mid, **kw)
        return "老何", 497, Page(
            items=[UpVideoItem(title="t1", url="u1", published_at=None, duration_sec=None,
                               view_count=9, danmaku_count=None, comment_count=None)],
            has_more=True,
            next_cursor="1",
        )

    with _with_client(get_up_videos=get_up_videos):
        payload = _payload("list_up_videos", {
            "author_url": "https://space.bilibili.com/42/upload/video", "order": "most_viewed",
            "jq": ".",
        })
    assert seen["mid"] == 42 and seen["order"] == "most_viewed"
    assert payload["author"] == "老何"
    assert payload["author_url"] == "https://space.bilibili.com/42"
    assert payload["total"] == 497
    assert payload["videos"].startswith(
        "videos[1]{title,url,published_at,duration_sec,view_count,danmaku_count,"
        "comment_count}:"
    )
    assert payload["has_more"] is True and payload["next_cursor"] == "1"


def test_get_up_info_one_row_per_input() -> None:
    seen: list[list[str]] = []

    def get_up_infos(inputs: list[str]) -> list[Batched[UpInfo]]:
        seen.append(inputs)
        return [Batched(inputs[0], UpInfo(mid=42, author="老何", follower_count=0)),
                Batched(inputs[1], error="UP 主不存在：mid 9")]

    with _with_client(get_up_infos=get_up_infos):
        payload = _payload("get_up_info", {
            "authors": ["https://space.bilibili.com/42", "9"],
            "jq": "map({input, author, follower_count, vip, error})",
        })
    assert seen == [["https://space.bilibili.com/42", "9"]]
    assert payload["count"] == 2 and payload["failed"] == 1
    assert payload["authors"] == (
        "authors[2]{input,author,follower_count,vip,error}:\n"
        "  \"https://space.bilibili.com/42\",老何,0,,\n"
        "  \"9\",,,,UP 主不存在：mid 9"
    )
    assert payload["elapsed_s"] >= 0


# ── 首页推荐 ────────────────────────────────────────────────────────────────


def test_get_feed_returns_toon() -> None:
    items = [
        FeedItem(title="t1", url="u1", author="甲", author_url="s1",
                 published_at="2026-08-17 10:00",
                 duration_sec=225.0, view_count=1234, danmaku_count=5, like_count=67,
                 rcmd_reason="1万点赞"),
        FeedItem(title="t2", url="u2", author=None, author_url=None, published_at=None,
                 duration_sec=None, view_count=None, danmaku_count=None, like_count=None,
                 rcmd_reason=None),
    ]
    with _with_client(get_feed=items):
        payload = _payload("get_feed", {"jq": "."})
    assert payload["count"] == 2
    assert payload["feed"].startswith(
        "feed[2]{title,url,author,author_url,published_at,duration_sec,view_count,"
        "danmaku_count,like_count,rcmd_reason}:"
    )
    assert "1万点赞" in payload["feed"]
    assert payload["elapsed_s"] >= 0


def test_get_feed_takes_no_url() -> None:
    """它是入口而不是围绕某条内容的工具。"""

    async def scenario() -> dict[str, Any]:
        server = create_server_for_test()
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            await client.initialize()
            tools = {t.name: t for t in (await client.list_tools()).tools}
            return tools["get_feed"].inputSchema

    schema = _run(scenario)
    assert set(schema.get("properties", {})) == {"jq"}


def test_get_feed_empty_batch() -> None:
    with _with_client(get_feed=[]):
        payload = _payload("get_feed", {"jq": "."})
    assert payload["count"] == 0
    assert payload["feed"].startswith("feed[0]{")
