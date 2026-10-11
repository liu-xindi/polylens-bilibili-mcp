"""纯解析逻辑测试：不联网。"""

from __future__ import annotations

import os
import time

import pytest

from polylens_bilibili.api._comments import _normalize_reply
from polylens_bilibili.api._subtitles import _normalize_url
from polylens_bilibili.api._video import (
    build_video_info,
    list_parts,
)
from polylens_bilibili.errors import BilibiliError


@pytest.fixture(autouse=True)
def _pin_timezone():
    """钉死时区：时间戳出参按本机时区渲染，测试需固定时区才有确定字符串。"""
    old = os.environ.get("TZ")
    os.environ["TZ"] = "Asia/Shanghai"
    time.tzset()
    yield
    if old is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = old
    time.tzset()


# ── build_video_info ──────────────────────────────────────────────────────


def _view(**kw):
    base = {
        "aid": 100, "bvid": "BV1xx", "cid": 200,
        "title": "test title", "owner": {"name": "up主", "mid": 42},
        "pubdate": 1700000000, "stat": {}, "desc": "",
    }
    base.update(kw)
    return base


def test_build_video_info_basic_fields():
    view = _view(stat={"view": 1000, "like": 50, "danmaku": 10, "reply": 7, "coin": 20})
    info, aid, cid = build_video_info(view)
    assert info.title == "test title"
    assert info.author == "up主"
    assert info.author_url == "https://space.bilibili.com/42"
    assert info.id == "BV1xx"
    assert aid == 100
    assert cid == 200
    assert info.view_count == 1000
    assert info.like_count == 50
    assert info.danmaku_count_total == 10
    assert info.comment_count == 7
    assert info.coin_count == 20
    assert info.published_at == "2023-11-15 06:13"  # 本机时区可读时间


def test_build_video_info_absent_stats_are_none():
    """平台没给的统计项为 None；给了 0 则保留 0。"""
    info, _, _ = build_video_info(_view(stat={"view": 0}))
    assert info.view_count == 0
    assert info.like_count is None
    assert info.coin_count is None


def test_build_video_info_desc_v2_preferred_over_desc():
    info, _, _ = build_video_info(_view(desc_v2=[{"raw_text": "v2 text"}], desc="old text"))
    assert info.summary == "v2 text"


def test_build_video_info_desc_fallback_when_no_desc_v2():
    info, _, _ = build_video_info(_view(desc="fallback desc"))
    assert info.summary == "fallback desc"


def test_build_video_info_empty_desc_gives_none():
    info, _, _ = build_video_info(_view(desc=""))
    assert info.summary is None


def test_build_video_info_cid_from_pages_when_missing():
    view = _view()
    del view["cid"]
    view["pages"] = [{"cid": 300}, {"cid": 301}]
    _, _, cid = build_video_info(view)
    assert cid == 300


def test_build_video_info_raises_when_no_cid():
    view = _view()
    del view["cid"]
    with pytest.raises(BilibiliError):
        build_video_info(view)


def test_build_video_info_url_shape():
    info, _, _ = build_video_info(_view(bvid="BV1abc"))
    assert info.url == "https://www.bilibili.com/video/BV1abc/"


def test_build_video_info_category_and_duration():
    info, _, _ = build_video_info(_view(duration=120, videos=3, tid=17))
    assert info.duration_sec == 120
    assert info.part_count == 3
    assert info.category_id == 17


def test_build_video_info_copyright():
    assert build_video_info(_view(copyright=1))[0].copyright == "自制"
    assert build_video_info(_view(copyright=2))[0].copyright == "转载"
    assert build_video_info(_view())[0].copyright is None


def test_build_video_info_staff_for_union_video():
    view = _view(staff=[
        {"mid": 42, "title": "UP主", "name": "甲"},
        {"mid": 43, "title": "剪辑", "name": "乙"},
    ])
    staff = build_video_info(view)[0].staff
    assert staff is not None
    assert [(m.name, m.role, m.author_url) for m in staff] == [
        ("甲", "UP主", "https://space.bilibili.com/42"),
        ("乙", "剪辑", "https://space.bilibili.com/43"),
    ]


def test_build_video_info_staff_absent_is_none():
    assert build_video_info(_view())[0].staff is None


def test_build_video_info_single_part_has_no_part_fields():
    """单段视频不给分段字段。"""
    info, _, _ = build_video_info(_view())
    assert info.current_page is None
    assert info.current_part is None


# ── 多段视频（page → cid）────────────────────────────────────────────────────


def _multi_view(**kw):
    return _view(
        cid=200,  # 顶层 cid = 第 1 段，与 pages[0] 一致
        videos=3,
        pages=[
            {"cid": 200, "page": 1, "part": "片头", "duration": 60},
            {"cid": 201, "page": 2, "part": "正片", "duration": 600},
            {"cid": 202, "page": 3, "part": "片尾", "duration": 30},
        ],
        **kw,
    )


def test_build_video_info_multi_part_selects_cid_and_exposes_parts():
    info, _aid, cid = build_video_info(_multi_view(), page=2)
    assert cid == 201  # 用第 2 段的 cid（内部返回值，供后续能力）
    assert info.current_page == 2
    assert info.current_part == "正片"
    assert info.duration_sec == 600  # 当前段时长，非整片
    assert info.part_count == 3


def test_build_video_info_multi_part_default_page_one():
    info, _aid, cid = build_video_info(_multi_view())
    assert cid == 200
    assert info.current_page == 1


# ── _normalize_reply ────────────────────────────────────────────────────────


def _reply(**kw):
    base = {
        "rpid": 1, "member": {"uname": "用户A", "mid": 123},
        "content": {"message": "hello"}, "like": 5, "count": 2,
        "ctime": 1700000000, "parent": 0,
    }
    base.update(kw)
    return base


def test_normalize_reply_basic():
    c = _normalize_reply(_reply())
    assert c.id == "1"
    assert c.author == "用户A"
    assert c.content == "hello"
    assert c.like_count == 5
    assert c.reply_count == 2
    assert c.created_at == "2023-11-15 06:13"


def test_normalize_reply_parent_id_when_nonzero():
    assert _normalize_reply(_reply(parent=999)).parent_id == "999"


def test_normalize_reply_parent_id_none_when_zero():
    assert _normalize_reply(_reply(parent=0)).parent_id is None


def test_normalize_reply_omits_parent_when_equals_root():
    """parent 指向所属主评论时置空，嵌套关系已经表达了这层。"""
    assert _normalize_reply(_reply(parent=555), root_id=555).parent_id is None


def test_normalize_reply_keeps_parent_on_cross_reply():
    """parent 指向另一条二级评论时保留，这是嵌套推不出的"谁回复谁"。"""
    assert _normalize_reply(_reply(parent=888), root_id=555).parent_id == "888"


def test_normalize_reply_missing_member_fields():
    c = _normalize_reply({"rpid": 2, "member": {}, "content": {}, "like": 0, "count": 0})
    assert c.author == ""
    assert c.content == ""


def test_normalize_reply_no_timestamp_gives_none():
    assert _normalize_reply(_reply(ctime=0)).created_at is None


# ── _normalize_url ──────────────────────────────────────────────────────────


def test_normalize_url_protocol_relative():
    assert _normalize_url("//api.bilibili.com/sub.json") == "https://api.bilibili.com/sub.json"


def test_normalize_url_absolute_unchanged():
    assert _normalize_url("https://example.com/sub.json") == "https://example.com/sub.json"
    assert _normalize_url("http://example.com/sub.json") == "http://example.com/sub.json"


# ── list_parts ──────────────────────────────────────────────────────────────


def test_list_parts_enumerates_multi_part_video():
    parts = list_parts(_multi_view())
    assert [(p.page, p.part, p.duration) for p in parts] == [
        (1, "片头", 60.0),
        (2, "正片", 600.0),
        (3, "片尾", 30.0),
    ]


def test_list_parts_single_video_gives_one_entry():
    parts = list_parts(_view(duration=42))
    assert len(parts) == 1
    assert parts[0].page == 1 and parts[0].duration == 42.0


# ── 关键标识缺失时显式失败 ──────────────────────────────────────────────────


@pytest.mark.parametrize("missing", ["aid", "bvid"])
def test_build_video_info_requires_identifiers(missing: str):
    """缺了给默认值会让评论查 oid=0、链接拼成 /video//，错误推迟到更难定位处。"""
    view = _view()
    del view[missing]
    with pytest.raises(BilibiliError):
        build_video_info(view)


def test_normalize_reply_without_rpid_raises():
    """这个 id 是 get_comment_replies 的入参，给空串会让钻取失败在更远的地方。"""
    with pytest.raises(BilibiliError):
        _normalize_reply({"member": {}, "content": {}})


# ── 评论的非文本内容 ────────────────────────────────────────────────────────


def test_normalize_reply_flags_default_to_false():
    c = _normalize_reply(_reply())
    assert c.is_top is False and c.up_liked is False
    assert c.image_urls is None and c.link_titles is None


def test_normalize_reply_reads_author_level():
    c = _normalize_reply(_reply(member={"uname": "u", "level_info": {"current_level": 6}}))
    assert c.author_level == 6


def test_normalize_reply_author_level_missing_is_none():
    assert _normalize_reply(_reply()).author_level is None


def test_normalize_reply_author_url_from_string_mid():
    c = _normalize_reply(_reply(member={"uname": "u", "mid": "320773657"}))
    assert c.author_url == "https://space.bilibili.com/320773657"


def test_normalize_reply_is_up_compares_string_mid_with_upper():
    c = _normalize_reply(_reply(member={"uname": "u", "mid": "42"}), upper_mid=42)
    assert c.is_up is True
    assert _normalize_reply(_reply(), upper_mid=42).is_up is False
    assert _normalize_reply(_reply()).is_up is False


def test_normalize_reply_ip_location_strips_prefix():
    c = _normalize_reply(_reply(reply_control={"location": "IP属地：湖北"}))
    assert c.ip_location == "湖北"


def test_normalize_reply_ip_location_missing_is_none():
    assert _normalize_reply(_reply(reply_control={})).ip_location is None


def test_normalize_reply_reads_top_and_up_liked():
    c = _normalize_reply(
        _reply(reply_control={"is_up_top": True}, up_action={"like": True, "reply": False})
    )
    assert c.is_top is True and c.up_liked is True


def test_normalize_reply_collects_image_urls():
    """配图只取地址，宽高与体积不带；缺 img_src 的条目跳过。"""
    c = _normalize_reply(_reply(content={
        "message": "看图",
        "pictures": [
            {"img_src": "http://a.jpg", "img_width": 392, "img_height": 388},
            {"img_src": "http://b.jpg"},
            {"img_width": 1},
        ],
    }))
    assert c.image_urls == "http://a.jpg\nhttp://b.jpg"


def test_normalize_reply_takes_link_titles_from_jump_url():
    """标题在同一份响应的 jump_url 里，不必另发请求；content 原文保持裸链接。"""
    c = _normalize_reply(_reply(content={
        "message": "https://b23.tv/x",
        "jump_url": {"https://b23.tv/x": {"title": "教你开启英文视频的中文字幕", "state": 0}},
    }))
    assert c.content == "https://b23.tv/x"
    assert c.link_titles == "教你开启英文视频的中文字幕"


def test_normalize_reply_skips_keyword_entries_in_jump_url():
    """jump_url 还装着平台自动识别的可点击搜索词（键是词本身），那不是评论里的链接。"""
    c = _normalize_reply(_reply(content={
        "message": "应该叫 aarch64 吧，装 binutils 就行",
        "jump_url": {
            "aarch64": {"title": "aarch64", "state": 0, "extra": {}},
            "binutils": {"title": "binutils", "state": 0, "extra": {}},
        },
    }))
    assert c.link_titles is None


def test_normalize_reply_keeps_only_real_links_when_mixed():
    c = _normalize_reply(_reply(content={
        "message": "看 https://b23.tv/x 讲的 aarch64",
        "jump_url": {
            "https://b23.tv/x": {"title": "某个视频", "state": 0},
            "aarch64": {"title": "aarch64", "state": 0, "extra": {}},
        },
    }))
    assert c.link_titles == "某个视频"


# ── 整片时长 ────────────────────────────────────────────────────────────────


def test_total_duration_is_whole_video_on_multi_part():
    """顶层 duration 是整片时长；多段时 duration_sec 换成了当前段，总时长另立字段。"""
    info, _, _ = build_video_info(_multi_view(duration=690), page=2)
    assert info.duration_sec == 600  # 第 2 段
    assert info.total_duration_sec == 690  # 整片


def test_total_duration_equals_duration_on_single_part():
    info, _, _ = build_video_info(_view(duration=42))
    assert info.duration_sec == 42
    assert info.total_duration_sec == 42
