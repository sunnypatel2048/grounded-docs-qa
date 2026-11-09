"""Polite, resumable downloader for the PostgreSQL documentation.

For each version it starts at /docs/<version>/index.html and follows links that stay inside
/docs/<version>/ and end in .html, saving each page under data/raw/<version>/.

Politeness: robots.txt is read first (disallowed paths are skipped, any crawl delay it asks for
is honoured), requests go one at a time with a delay, the crawler identifies itself, and
rate-limit or server errors are retried with backoff instead of hammered.

Resumability: a page already on disk is never requested again. Its links are read from the
saved file, so an interrupted run continues where it stopped and a complete re-run makes no
page requests at all. Each page is written atomically, then logged as one line in
manifest.jsonl, so a crash never leaves a half-written page that looks complete.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
import urllib.robotparser
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlsplit

BASE_URL = "https://www.postgresql.org"
USER_AGENT = "grounded-docs-qa/0.1 (+https://github.com/sunnypatel2048/grounded-docs-qa)"
DEFAULT_RAW_DIR = Path("data/raw")
MAX_BACKOFF_S = 60.0


@dataclass(frozen=True)
class Response:
    status: int  # 0 = no HTTP response at all (DNS failure, timeout, connection reset)
    body: bytes
    headers: dict[str, str]  # lower-cased names


Fetch = Callable[[str], Response]


def http_get(url: str, timeout: float = 30.0) -> Response:
    """One GET request. Never raises: every outcome becomes a Response."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as r:
            headers = {k.lower(): v for k, v in r.headers.items()}
            return Response(r.status, r.read(), headers)
    except urllib.error.HTTPError as e:
        headers = {k.lower(): v for k, v in (e.headers or {}).items()}
        return Response(e.code, e.read() if e.fp else b"", headers)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return Response(0, str(e).encode(), {})


class CrawlAborted(Exception):
    """Too many failures in a row, or robots.txt couldn't be read: stop rather than hammer."""


# --- links --------------------------------------------------------------------------------


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self.hrefs.extend(value for name, value in attrs if name == "href" and value)


def page_name(url: str, prefix: str) -> str | None:
    """'https://www.postgresql.org/docs/16/sql-select.html' -> 'sql-select.html', or None if the
    URL isn't a page of this version (other versions, other sites, non-HTML, odd paths)."""
    parts = urlsplit(url)
    if parts.netloc != urlsplit(BASE_URL).netloc or not parts.path.startswith(prefix):
        return None
    name = parts.path[len(prefix) :]
    if not name.endswith(".html") or "/" in name or ".." in name or not name[:-5]:
        return None
    return name


def extract_links(html: str, page_url: str, prefix: str) -> list[str]:
    """Same-version page URLs linked from this page, without #anchors or ?queries, in order."""
    parser = _LinkParser()
    parser.feed(html)
    links: list[str] = []
    for href in parser.hrefs:
        absolute, _fragment = urldefrag(urljoin(page_url, href))
        name = page_name(absolute, prefix)
        if name is not None:
            url = f"{BASE_URL}{prefix}{name}"
            if url not in links:
                links.append(url)
    return links


# --- robots.txt ---------------------------------------------------------------------------


def load_robots(fetch: Fetch) -> urllib.robotparser.RobotFileParser:
    robots = urllib.robotparser.RobotFileParser()
    response = fetch(f"{BASE_URL}/robots.txt")
    if response.status == 200:
        robots.parse(response.body.decode("utf-8", "replace").splitlines())
    elif 400 <= response.status < 500:
        robots.allow_all = True  # no robots.txt means no restrictions (RFC 9309)
        robots.modified()  # mark as read; an unread parser refuses everything
    else:
        # Server error or no response: RFC 9309 says assume everything is disallowed.
        raise CrawlAborted(
            f"couldn't read robots.txt (status {response.status or 'no response'}); "
            "not crawling without it"
        )
    return robots


def polite_delay(robots: urllib.robotparser.RobotFileParser, requested: float) -> float:
    """The larger of our own delay and whatever robots.txt asks of us."""
    delay = requested
    crawl_delay = robots.crawl_delay(USER_AGENT)
    if crawl_delay:
        delay = max(delay, float(crawl_delay))
    rate = robots.request_rate(USER_AGENT)
    if rate and rate.requests:
        delay = max(delay, rate.seconds / rate.requests)
    return delay


# --- manifest -----------------------------------------------------------------------------


class Manifest:
    """One JSON line per downloaded page: url, size, sha256, when. Append-only, so a crash can
    lose at most the line being written; pages on disk without a line are recovered on load."""

    def __init__(self, directory: Path):
        self.path = directory / "manifest.jsonl"
        self.entries: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self.entries[entry["page"]] = entry

    def record(self, page: str, url: str, body: bytes, fetched_at: str | None) -> None:
        entry = {
            "page": page,
            "url": url,
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "fetched_at": fetched_at,  # None = recovered from a file already on disk
        }
        self.entries[page] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)  # atomic: the page appears complete or not at all


# --- crawling -----------------------------------------------------------------------------


@dataclass
class VersionReport:
    version: str
    pages: int = 0  # pages handled this run (downloaded + already on disk)
    downloaded: int = 0
    cached: int = 0
    skipped_by_robots: int = 0
    failed: list[str] = field(default_factory=list)
    bytes_downloaded: int = 0
    seconds: float = 0.0


class Crawler:
    def __init__(
        self,
        *,
        fetch: Fetch = http_get,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        raw_dir: Path = DEFAULT_RAW_DIR,
        delay: float = 1.0,
        max_attempts: int = 3,
        max_consecutive_failures: int = 10,
        log: Callable[[str], None] = print,
    ):
        self.fetch, self.sleep, self.clock = fetch, sleep, clock
        self.raw_dir = Path(raw_dir)
        self.max_attempts = max_attempts
        self.max_consecutive_failures = max_consecutive_failures
        self.log = log
        self.robots = load_robots(fetch)
        self.delay = polite_delay(self.robots, delay)
        self._last_request: float | None = None

    def _wait_turn(self) -> None:
        if self._last_request is not None:
            remaining = self.delay - (self.clock() - self._last_request)
            if remaining > 0:
                self.sleep(remaining)
        self._last_request = self.clock()

    def _download(self, url: str) -> bytes | None:
        """The page body, or None after a permanent error or max_attempts transient ones."""
        for attempt in range(1, self.max_attempts + 1):
            self._wait_turn()
            response = self.fetch(url)
            if response.status == 200:
                return response.body
            transient = response.status in (0, 429) or response.status >= 500
            if not transient:  # 404, 403, 410 ...: asking again won't help
                self.log(f"  {response.status} {url}")
                return None
            if attempt < self.max_attempts:
                backoff = self.delay * 2**attempt
                retry_after = _retry_after_seconds(response.headers.get("retry-after"))
                wait = min(max(backoff, retry_after or 0.0), MAX_BACKOFF_S)
                self.log(f"  {response.status or 'no response'} {url}; retrying in {wait:.0f}s")
                self.sleep(wait)
        self.log(f"  gave up on {url} after {self.max_attempts} attempts")
        return None

    def crawl(self, version: str, limit: int | None = None) -> VersionReport:
        started = self.clock()
        prefix = f"/docs/{version}/"
        out_dir = self.raw_dir / version
        manifest = Manifest(out_dir)
        report = VersionReport(version)
        start = f"{BASE_URL}{prefix}index.html"
        queue, seen = deque([start]), {start}
        consecutive_failures = 0

        while queue and (limit is None or report.pages < limit):
            url = queue.popleft()
            name = page_name(url, prefix)
            path = out_dir / name

            if path.exists():
                body = path.read_bytes()
                if name not in manifest.entries:  # written, then crashed before logging
                    manifest.record(name, url, body, fetched_at=None)
                report.cached += 1
            elif not self.robots.can_fetch(USER_AGENT, url):
                report.skipped_by_robots += 1
                continue
            else:
                body = self._download(url)
                if body is None:
                    report.failed.append(url)
                    consecutive_failures += 1
                    if consecutive_failures >= self.max_consecutive_failures:
                        raise CrawlAborted(
                            f"{consecutive_failures} pages failed in a row; stopping. "
                            "Run again later to resume."
                        )
                    continue
                consecutive_failures = 0
                _write_atomic(path, body)
                manifest.record(name, url, body, fetched_at=datetime.now(UTC).isoformat())
                report.downloaded += 1
                report.bytes_downloaded += len(body)
                if report.downloaded % 50 == 0:
                    self.log(f"  {version}: {report.pages + 1} pages ({report.downloaded} new)")

            report.pages += 1
            for link in extract_links(body.decode("utf-8", "replace"), url, prefix):
                if link not in seen:
                    seen.add(link)
                    queue.append(link)

        report.seconds = self.clock() - started
        return report


def _retry_after_seconds(value: str | None) -> float | None:
    try:
        seconds = float(value) if value is not None else None
    except ValueError:
        return None  # HTTP-date form: fall back to our own backoff
    return seconds if seconds is not None and seconds >= 0 else None
