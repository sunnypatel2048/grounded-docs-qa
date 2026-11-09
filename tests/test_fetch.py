"""Downloader tests against a fake site: no network, no real sleeping."""

from __future__ import annotations

import json

import pytest

from docs_qa import fetch
from docs_qa.fetch import BASE_URL, CrawlAborted, Crawler, Response

DOCS = f"{BASE_URL}/docs/16/"


def page(*hrefs: str) -> bytes:
    links = "".join(f'<a href="{h}">link</a>' for h in hrefs)
    return f"<html><body>{links}</body></html>".encode()


class FakeSite:
    """URL -> list of responses. The last response repeats once the list runs out."""

    def __init__(self, pages: dict[str, bytes], robots: Response | None = None):
        self.routes: dict[str, list[Response]] = {
            DOCS + name: [Response(200, body, {})] for name, body in pages.items()
        }
        self.routes[f"{BASE_URL}/robots.txt"] = [robots or Response(404, b"", {})]
        self.requests: list[str] = []

    def respond(self, url: str, *responses: Response) -> None:
        self.routes[url] = list(responses)

    def __call__(self, url: str) -> Response:
        self.requests.append(url)
        responses = self.routes.get(url, [Response(404, b"not found", {})])
        return responses.pop(0) if len(responses) > 1 else responses[0]

    def page_requests(self) -> list[str]:
        return [u for u in self.requests if not u.endswith("/robots.txt")]


class FakeTime:
    """A clock that only moves when the crawler sleeps (or a test advances it)."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


SITE = {
    "index.html": page("intro.html", "sql.html#anchor", "sql.html?x=1", "/docs/17/intro.html"),
    "intro.html": page("index.html", "sql.html", "https://example.com/x.html"),
    "sql.html": page("intro.html", "img/diagram.png", "../17/sql.html"),
}


def make_crawler(site, tmp_path, time=None, **kwargs) -> Crawler:
    time = time or FakeTime()
    return Crawler(
        fetch=site,
        sleep=time.sleep,
        clock=time.clock,
        raw_dir=tmp_path,
        log=lambda _msg: None,
        **kwargs,
    )


# --- links --------------------------------------------------------------------------------


def test_extract_links_keeps_only_this_versions_pages():
    html = page(
        "intro.html",
        "sql.html#anchor",
        "sql.html?x=1",
        "/docs/16/abs.html",
        "/docs/17/intro.html",
        "../17/other.html",
        "https://example.com/x.html",
        "img/diagram.png",
        "sub/dir.html",
        "intro.html",  # duplicate
    ).decode()
    links = fetch.extract_links(html, DOCS + "index.html", "/docs/16/")
    assert links == [DOCS + "intro.html", DOCS + "sql.html", DOCS + "abs.html"]


@pytest.mark.parametrize(
    "url",
    [
        f"{BASE_URL}/docs/17/intro.html",
        "https://example.com/docs/16/intro.html",
        f"{DOCS}diagram.png",
        f"{DOCS}sub/page.html",
        f"{DOCS}.html",
    ],
)
def test_page_name_rejects_urls_outside_the_version(url):
    assert fetch.page_name(url, "/docs/16/") is None


# --- crawling -----------------------------------------------------------------------------


def test_crawl_downloads_every_linked_page_once(tmp_path):
    site = FakeSite(SITE)
    report = make_crawler(site, tmp_path).crawl("16")

    assert sorted(site.page_requests()) == sorted(DOCS + name for name in SITE)
    assert (report.pages, report.downloaded, report.cached, report.failed) == (3, 3, 0, [])
    for name, body in SITE.items():
        assert (tmp_path / "16" / name).read_bytes() == body
    assert not list((tmp_path / "16").glob("*.part"))


def test_manifest_records_each_page(tmp_path):
    make_crawler(FakeSite(SITE), tmp_path).crawl("16")
    lines = (tmp_path / "16" / "manifest.jsonl").read_text().splitlines()
    entries = {e["page"]: e for e in map(json.loads, lines)}

    assert set(entries) == set(SITE)
    sql = entries["sql.html"]
    assert sql["url"] == DOCS + "sql.html"
    assert sql["bytes"] == len(SITE["sql.html"])
    assert len(sql["sha256"]) == 64 and sql["fetched_at"]


def test_second_run_requests_no_pages(tmp_path):
    make_crawler(FakeSite(SITE), tmp_path).crawl("16")
    site = FakeSite(SITE)
    report = make_crawler(site, tmp_path).crawl("16")

    assert site.page_requests() == []
    assert (report.pages, report.downloaded, report.cached) == (3, 0, 3)
    lines = (tmp_path / "16" / "manifest.jsonl").read_text().splitlines()
    assert len(lines) == 3  # nothing appended for pages already logged


def test_interrupted_run_resumes_where_it_stopped(tmp_path):
    make_crawler(FakeSite(SITE), tmp_path).crawl("16", limit=1)
    assert [p.name for p in (tmp_path / "16").glob("*.html")] == ["index.html"]

    site = FakeSite(SITE)
    report = make_crawler(site, tmp_path).crawl("16")
    assert sorted(site.page_requests()) == [DOCS + "intro.html", DOCS + "sql.html"]
    assert (report.downloaded, report.cached) == (2, 1)


def test_page_on_disk_without_manifest_line_is_recovered(tmp_path):
    (tmp_path / "16").mkdir()
    (tmp_path / "16" / "index.html").write_bytes(SITE["index.html"])  # crashed before logging

    site = FakeSite(SITE)
    make_crawler(site, tmp_path).crawl("16")

    assert DOCS + "index.html" not in site.page_requests()
    entries = fetch.Manifest(tmp_path / "16").entries
    assert entries["index.html"]["fetched_at"] is None
    assert entries["intro.html"]["fetched_at"] is not None


def test_limit_caps_pages_per_version(tmp_path):
    report = make_crawler(FakeSite(SITE), tmp_path).crawl("16", limit=2)
    assert report.pages == 2
    assert len(list((tmp_path / "16").glob("*.html"))) == 2


def test_waits_between_requests(tmp_path):
    time = FakeTime()
    make_crawler(FakeSite(SITE), tmp_path, time=time, delay=1.5).crawl("16")
    # robots.txt is fetched before the crawler's clock starts; 3 pages -> 2 waits
    assert time.sleeps == [1.5, 1.5]


# --- robots.txt ---------------------------------------------------------------------------


def test_robots_disallowed_pages_are_skipped(tmp_path):
    robots = Response(200, b"User-agent: *\nDisallow: /docs/16/sql.html\n", {})
    site = FakeSite(SITE, robots=robots)
    report = make_crawler(site, tmp_path).crawl("16")

    assert DOCS + "sql.html" not in site.requests
    assert report.skipped_by_robots == 1
    assert report.downloaded == 2


def test_robots_crawl_delay_raises_our_delay(tmp_path):
    robots = Response(200, b"User-agent: *\nCrawl-delay: 5\n", {})
    assert make_crawler(FakeSite(SITE, robots=robots), tmp_path, delay=1.0).delay == 5.0
    assert make_crawler(FakeSite(SITE, robots=robots), tmp_path, delay=8.0).delay == 8.0


def test_missing_robots_txt_allows_everything(tmp_path):
    report = make_crawler(FakeSite(SITE, robots=Response(404, b"", {})), tmp_path).crawl("16")
    assert report.downloaded == 3


@pytest.mark.parametrize("status", [500, 503, 0])
def test_unreadable_robots_txt_aborts(tmp_path, status):
    with pytest.raises(CrawlAborted, match="robots.txt"):
        make_crawler(FakeSite(SITE, robots=Response(status, b"", {})), tmp_path)


# --- errors -------------------------------------------------------------------------------


def test_404_is_not_retried(tmp_path):
    site = FakeSite(SITE)
    site.respond(DOCS + "sql.html", Response(404, b"", {}))
    report = make_crawler(site, tmp_path).crawl("16")

    assert site.requests.count(DOCS + "sql.html") == 1
    assert report.failed == [DOCS + "sql.html"]
    assert not (tmp_path / "16" / "sql.html").exists()


def test_server_error_retries_with_backoff_then_succeeds(tmp_path):
    time = FakeTime()
    site = FakeSite(SITE)
    site.respond(DOCS + "sql.html", Response(503, b"", {}), Response(200, SITE["sql.html"], {}))
    report = make_crawler(site, tmp_path, time=time, delay=1.0).crawl("16")

    assert site.requests.count(DOCS + "sql.html") == 2
    assert report.failed == []
    assert 2.0 in time.sleeps  # delay * 2**1


def test_server_error_gives_up_after_max_attempts(tmp_path):
    site = FakeSite(SITE)
    site.respond(DOCS + "sql.html", Response(500, b"", {}))
    report = make_crawler(site, tmp_path, max_attempts=3).crawl("16")

    assert site.requests.count(DOCS + "sql.html") == 3
    assert report.failed == [DOCS + "sql.html"]


def test_429_honours_retry_after(tmp_path):
    time = FakeTime()
    site = FakeSite(SITE)
    site.respond(
        DOCS + "sql.html",
        Response(429, b"", {"retry-after": "30"}),
        Response(200, SITE["sql.html"], {}),
    )
    make_crawler(site, tmp_path, time=time, delay=1.0).crawl("16")
    assert 30.0 in time.sleeps


def test_backoff_is_capped(tmp_path):
    time = FakeTime()
    site = FakeSite(SITE)
    site.respond(
        DOCS + "sql.html",
        Response(429, b"", {"retry-after": "3600"}),
        Response(200, SITE["sql.html"], {}),
    )
    make_crawler(site, tmp_path, time=time).crawl("16")
    assert max(time.sleeps) == fetch.MAX_BACKOFF_S


def test_many_failures_in_a_row_abort_the_crawl(tmp_path):
    names = [f"p{i}.html" for i in range(5)]
    site = FakeSite({"index.html": page(*names)})  # every linked page is a 404
    with pytest.raises(CrawlAborted, match="in a row"):
        make_crawler(site, tmp_path, max_consecutive_failures=3).crawl("16")
    assert len(site.page_requests()) == 1 + 3


@pytest.mark.parametrize(
    ("value", "expected"),
    [("30", 30.0), ("0", 0.0), (None, None), ("-5", None), ("Wed, 21 Oct 2026 07:28:00 GMT", None)],
)
def test_retry_after_parsing(value, expected):
    assert fetch._retry_after_seconds(value) == expected
