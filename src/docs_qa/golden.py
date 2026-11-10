"""The golden set: questions with known answers and known evidence, for evaluating retrieval
and answers. Stored in eval/golden.yaml.

Evidence is a section plus a short quote copied from it, not a chunk ID: chunk IDs depend on
the chunker, and Task 11 tries other chunkings. A retrieved chunk counts as finding a piece of
evidence when it comes from that section and contains the quote (see `chunk_has_evidence`).
That is stricter than "right section" (a long section spans several chunks) and survives any
re-chunking.

Each question's dev/test split comes from a hash of its ID, so nobody picks which questions are
held out: tune on dev, report on test.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

FORMAT_VERSION = 1
DEFAULT_PATH = Path("eval/golden.yaml")

TYPES = ("single", "multi_hop", "version_diff", "unanswerable")
STYLES = ("keyword", "paraphrase")
AREAS = (
    "settings",
    "sql",
    "functions",
    "catalogs",
    "admin",
    "internals",
    "clients",
    "extensions",
    "release_notes",
)
SOURCES = ("human", "llm")
STATUSES = ("draft", "reviewed")
TEST_SHARE = 0.3

# Share of the set each group should have (the roadmap's 60 / 25 / 15).
TARGET_MIX = {
    "single": 0.60,
    "multi_hop + version_diff": 0.25,
    "unanswerable": 0.15,
}

_ID = re.compile(r"^q\d{3,}$")
_REF = re.compile(r"^(?P<version>\d+)/(?P<page>[a-z0-9_.-]+\.html)#(?P<key>.+)$")
_REQUIRED = ("id", "question", "versions", "type", "style", "area", "source", "status", "answer")


class GoldenError(Exception):
    """The file can't be read as a golden set at all (not YAML, wrong top-level shape)."""


@dataclass(frozen=True)
class Evidence:
    ref: str  # "16/runtime-config-autovacuum.html#GUC-AUTOVACUUM-NAPTIME"
    quote: str  # copied word for word from that section

    @property
    def parts(self) -> tuple[str, str, str]:
        """(version, page, section key)."""
        m = _REF.match(self.ref)
        if not m:
            raise ValueError(f"bad evidence ref {self.ref!r}")
        return m["version"], m["page"], m["key"]


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    versions: tuple[str, ...]
    type: str
    style: str
    area: str
    source: str
    status: str
    answer: str
    evidence: tuple[Evidence, ...] = ()
    notes: str = ""

    @property
    def split(self) -> str:
        bucket = int(hashlib.sha256(self.id.encode()).hexdigest(), 16) % 1000
        return "test" if bucket < TEST_SHARE * 1000 else "dev"


@dataclass
class GoldenSet:
    questions: list[Question]
    problems: list[str] = field(default_factory=list)  # "q012: ..." for each rule broken

    def reviewed(self) -> list[Question]:
        return [q for q in self.questions if q.status == "reviewed"]


# --- loading and offline validation ---------------------------------------------------------


def load(path: Path = DEFAULT_PATH) -> GoldenSet:
    import yaml

    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise GoldenError(f"{path} isn't valid YAML: {e}") from e
    return parse_data(data)


def parse_data(data) -> GoldenSet:
    if not isinstance(data, dict) or "questions" not in data:
        raise GoldenError("expected a mapping with `format` and `questions`")
    if data.get("format") != FORMAT_VERSION:
        raise GoldenError(f"unsupported format {data.get('format')!r}; expected {FORMAT_VERSION}")
    raw = data["questions"] or []
    if not isinstance(raw, list):
        raise GoldenError("`questions` must be a list")

    golden = GoldenSet([])
    seen: set[str] = set()
    for n, item in enumerate(raw, 1):
        label = item.get("id", f"entry {n}") if isinstance(item, dict) else f"entry {n}"
        errors: list[str] = []
        question = _parse_question(item, errors)
        if question is not None:
            if question.id in seen:
                errors.append("duplicate id")
            seen.add(question.id)
            errors += _rule_problems(question)
        golden.problems += [f"{label}: {e}" for e in errors]
        if question is not None:
            golden.questions.append(question)
    return golden


def _parse_question(item, errors: list[str]) -> Question | None:
    if not isinstance(item, dict):
        errors.append("each question must be a mapping")
        return None
    missing = [k for k in _REQUIRED if item.get(k) in (None, "", [])]
    if missing:
        errors.append(f"missing {', '.join(missing)}")
        return None
    unknown = set(item) - {*_REQUIRED, "evidence", "notes"}
    if unknown:
        errors.append(f"unknown fields {', '.join(sorted(unknown))}")

    versions = item["versions"] if isinstance(item["versions"], list) else [item["versions"]]
    evidence = []
    for e in item.get("evidence") or []:
        if not isinstance(e, dict) or not e.get("ref") or not e.get("quote"):
            errors.append("each evidence item needs a ref and a quote")
            continue
        evidence.append(Evidence(str(e["ref"]).strip(), str(e["quote"]).strip()))
    return Question(
        id=str(item["id"]),
        question=str(item["question"]).strip(),
        versions=tuple(str(v) for v in versions),  # YAML reads 16 as a number
        type=str(item["type"]),
        style=str(item["style"]),
        area=str(item["area"]),
        source=str(item["source"]),
        status=str(item["status"]),
        answer=str(item["answer"]).strip(),
        evidence=tuple(evidence),
        notes=str(item.get("notes") or "").strip(),
    )


def _rule_problems(q: Question) -> list[str]:
    errors = []
    if not _ID.match(q.id):
        errors.append("id must look like q001")
    for name, value, allowed in (
        ("type", q.type, TYPES),
        ("style", q.style, STYLES),
        ("area", q.area, AREAS),
        ("source", q.source, SOURCES),
        ("status", q.status, STATUSES),
    ):
        if value not in allowed:
            errors.append(f"{name} must be one of {', '.join(allowed)} (got {value!r})")
    if not all(v.isdigit() for v in q.versions):
        errors.append("versions must be major versions like 16")
    if len(set(q.versions)) != len(q.versions):
        errors.append("versions are repeated")

    for e in q.evidence:
        m = _REF.match(e.ref)
        if not m:
            errors.append(f"evidence ref {e.ref!r} must look like 16/page.html#SECTION-KEY")
        elif m["version"] not in q.versions:
            errors.append(f"evidence {e.ref} is from a version the question isn't about")
        if not 3 <= len(e.quote) <= 300:
            errors.append(f"quote for {e.ref} should be a short phrase (3-300 characters)")

    evidence_versions = {_REF.match(e.ref)["version"] for e in q.evidence if _REF.match(e.ref)}
    if q.type == "unanswerable":
        if q.evidence:
            errors.append("an unanswerable question has no evidence")
    elif not q.evidence:
        errors.append(f"a {q.type} question needs evidence")
    if q.type == "single":
        if len(q.versions) != 1:
            errors.append("a single question is about exactly one version")
    elif q.type == "multi_hop":
        if len({e.ref for e in q.evidence}) < 2:
            errors.append("a multi_hop question needs evidence from at least 2 sections")
    elif q.type == "version_diff":
        if len(q.versions) < 2 or len(evidence_versions) < 2:
            errors.append("a version_diff question needs evidence from at least 2 versions")
    return errors


# --- matching evidence in retrieved text ----------------------------------------------------

_QUOTE_CHARS = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "\xa0": " "})


def normalize(text: str) -> str:
    """Case, curly quotes and whitespace don't matter when matching a quote."""
    return re.sub(r"\s+", " ", text.translate(_QUOTE_CHARS)).strip().lower()


def chunk_has_evidence(
    version: str, page: str, section_key: str, text: str, evidence: Evidence
) -> bool:
    """Does this retrieved chunk count as finding this piece of evidence?"""
    return (version, page, section_key) == evidence.parts and normalize(
        evidence.quote
    ) in normalize(text)


# --- checking evidence against the stored docs ----------------------------------------------


def evidence_problems(conn, golden: GoldenSet) -> list[str]:
    """Each evidence ref must name a stored section, and its quote must appear in that section."""
    problems = []
    texts: dict[tuple[str, str, str], str | None] = {}
    for q in golden.questions:
        for e in q.evidence:
            if not _REF.match(e.ref):
                continue  # already reported by the offline rules
            parts = e.parts
            if parts not in texts:
                rows = conn.execute(
                    "SELECT content FROM chunks WHERE version = %s AND page_path = %s "
                    "AND section_key = %s ORDER BY ordinal",
                    parts,
                ).fetchall()
                # Drop each chunk's location line; the quote must come from the section's text.
                texts[parts] = " ".join(c.split("\n\n", 1)[-1] for (c,) in rows) if rows else None
            text = texts[parts]
            if text is None:
                problems.append(f"{q.id}: no stored section {e.ref}")
            elif normalize(e.quote) not in normalize(text):
                problems.append(f"{q.id}: quote not found in {e.ref}: {e.quote!r}")
    conn.rollback()
    return problems


# --- the mix ----------------------------------------------------------------------------------


def mix_report(golden: GoldenSet) -> list[str]:
    """Lines describing the set's make-up against the targets."""
    qs = golden.questions
    if not qs:
        return ["no questions yet"]
    total = len(qs)
    types = Counter(q.type for q in qs)
    groups = {
        "single": types["single"],
        "multi_hop + version_diff": types["multi_hop"] + types["version_diff"],
        "unanswerable": types["unanswerable"],
    }
    reviewed = len(golden.reviewed())
    lines = [f"{total} questions, {reviewed} reviewed, {total - reviewed} drafts"]
    for name, count in groups.items():
        lines.append(
            f"  {name:<26}{count:>4}  {count / total:>4.0%}  (target {TARGET_MIX[name]:.0%})"
        )
    for label, values in (
        ("type", [q.type for q in qs]),
        ("style", [q.style for q in qs]),
        ("area", [q.area for q in qs]),
        ("source", [q.source for q in qs]),
        ("split", [q.split for q in qs]),
        ("version", [v for q in qs for v in q.versions]),
    ):
        counts = Counter(values)
        lines.append(f"{label}: " + ", ".join(f"{k} {counts[k]}" for k in sorted(counts)))
    return lines
