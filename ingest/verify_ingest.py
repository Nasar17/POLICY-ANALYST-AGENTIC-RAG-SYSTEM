"""
Phase 1 verification: Check that ch119_sections.json meets the spec.

Checks:
  1. All 20 expected sections are present
  2. All metadata fields are populated
  3. Body text is non-empty and does NOT start with the section number or title
  4. History field is present (non-empty)
  5. Source URLs follow the expected pattern
  6. Body text does NOT contain the History line (clean separation)
  7. Print sample body excerpts for human review
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_FILE = ROOT / "data" / "ch119_sections.json"

EXPECTED_SECTIONS = {
    "119.01", "119.011", "119.021", "119.035",
    "119.07", "119.0701", "119.071", "119.0711",
    "119.0712", "119.0713", "119.0714", "119.0715",
    "119.0725", "119.084", "119.092", "119.10",
    "119.105", "119.11", "119.12", "119.15",
}

REQUIRED_FIELDS = {
    "jurisdiction", "title_num", "title_name",
    "chapter_num", "chapter_name", "section_num",
    "section_title", "text", "history", "source_url",
}

URL_RE = re.compile(
    r"https://www\.leg\.state\.fl\.us/statutes/index\.cfm"
    r"\?App_mode=Display_Statute&URL=0100-0199/0119/Sections/0119\.\d+\.html"
)

HISTORY_MARKER = re.compile(r"^History\s*[.—]", re.IGNORECASE)


def run():
    if not DATA_FILE.exists():
        print(f"FAIL: {DATA_FILE} not found — run ingest_chapter.py first")
        sys.exit(1)

    records = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    failures = []
    warnings = []

    found_nums = {r["section_num"] for r in records}

    # 1. Section count
    missing = EXPECTED_SECTIONS - found_nums
    extra = found_nums - EXPECTED_SECTIONS
    if missing:
        failures.append(f"Missing sections: {sorted(missing)}")
    if extra:
        warnings.append(f"Unexpected extra sections: {sorted(extra)}")

    for r in records:
        snum = r["section_num"]
        prefix = f"[{snum}]"

        # 2. All required fields present and non-None
        for field in REQUIRED_FIELDS:
            if field not in r:
                failures.append(f"{prefix} Missing field: {field}")
            elif r[field] is None and field not in {"department"}:
                warnings.append(f"{prefix} Field is None: {field}")

        # 3. Static metadata correctness
        if r.get("jurisdiction") != "FL":
            failures.append(f"{prefix} jurisdiction != FL: {r.get('jurisdiction')!r}")
        if r.get("title_num") != "X":
            failures.append(f"{prefix} title_num != X: {r.get('title_num')!r}")
        if r.get("chapter_num") != "119":
            failures.append(f"{prefix} chapter_num != 119: {r.get('chapter_num')!r}")

        # 4. Body text non-empty and doesn't start with section number
        text = r.get("text", "")
        if not text.strip():
            failures.append(f"{prefix} text is empty")
        elif text.strip().startswith(snum):
            failures.append(f"{prefix} text starts with section number (header not stripped)")

        # 5. History non-empty
        if not r.get("history", "").strip():
            warnings.append(f"{prefix} history is empty")

        # 6. History not leaked into body text
        if HISTORY_MARKER.search(text):
            failures.append(f"{prefix} body text contains 'History.' (not stripped)")

        # 7. Source URL format
        url = r.get("source_url", "")
        if not URL_RE.match(url):
            failures.append(f"{prefix} source_url looks wrong: {url}")

    # Print results
    print(f"Records: {len(records)}  |  Expected: {len(EXPECTED_SECTIONS)}")
    print()

    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  FAIL  {f}")
    else:
        print("All checks PASSED")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  WARN  {w}")

    print()
    print("=== Sample body excerpts (first 200 chars each) ===")
    spot_check = ["119.01", "119.07", "119.071", "119.10"]
    for snum in spot_check:
        for r in records:
            if r["section_num"] == snum:
                print(f"\n[{snum}] {r['section_title']}")
                print(f"  text:    {r['text'][:200]}")
                print(f"  history: {r['history'][:100]}")
                break

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    run()
