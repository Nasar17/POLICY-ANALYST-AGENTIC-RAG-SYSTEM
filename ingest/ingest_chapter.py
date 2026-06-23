"""
Phase 1 ingestion: Fetch and parse Chapter 119 (Public Records) of Florida Statutes.

HTML structure (confirmed from live site):
  div.Section           one per statute section
    span.SectionNumber  "119.01"
    span.Catchline > span.CatchlineText  "General state policy..."
    span.SectionBody    (inline body text, sometimes present)
    div.Subsection      "(1) It is the policy..."
    div.Paragraph       "(a) Automation..."
    div.SubParagraph    "1. ..."
    div.History > span.HistoryText  "s. 1, ch. ..."
  div.IndexItem         TOC entries — skip entirely

Body text strategy: extract the Section div's full text, then strip the
section-number header and history tail. This sidesteps nested-div
double-counting that comes from walking Subsection > Paragraph > SubParagraph.

Output:
  data/raw/ch119_raw.html   raw page (cached after first fetch)
  data/ch119_sections.json  one record per section with full metadata schema
"""

import json
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DATA_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)

CHAPTER_META = {
    "jurisdiction": "FL",
    "title_num": "X",
    "title_name": "Public Officers, Employees, and Records",
    "chapter_num": "119",
    "chapter_name": "Public Records",
    "department": None,
}

CHAPTER_URL = (
    "https://www.leg.state.fl.us/statutes/index.cfm"
    "?App_mode=Display_Statute&URL=0100-0199/0119/0119.html"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def fetch(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def section_url(section_num: str) -> str:
    padded = "0" + section_num  # 119.x -> 0119.x
    return (
        "https://www.leg.state.fl.us/statutes/index.cfm"
        f"?App_mode=Display_Statute&URL=0100-0199/0119/Sections/{padded}.html"
    )


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_section_div(div: Tag) -> dict | None:
    """Parse a single div.Section into a clean record dict."""
    # --- Section number ---
    num_span = div.find("span", class_="SectionNumber")
    if not num_span:
        return None
    section_num = clean(num_span.get_text())

    # --- Section title ---
    catchline_text = div.find("span", class_="CatchlineText")
    if catchline_text:
        section_title = clean(catchline_text.get_text()).rstrip(".")
    else:
        cl = div.find("span", class_="Catchline")
        section_title = clean(cl.get_text()).rstrip(".") if cl else ""

    # --- History ---
    history_div = div.find("div", class_="History")
    history_text = ""
    if history_div:
        ht = history_div.find("span", class_="HistoryText")
        history_text = clean(ht.get_text()) if ht else clean(history_div.get_text())
        history_div.decompose()  # remove from tree before extracting body

    # --- Body ---
    # Strategy: take the full section text, then strip off the leading
    # "SECTION_NUM CATCHLINE.—" prefix. What remains is the body.
    # This avoids double-counting from nested Subsection > Paragraph > SubParagraph.
    full_text = clean(div.get_text())

    # Build the header prefix we expect at the start of full_text
    # e.g. "119.01 General state policy on public records.—"
    escaped_num = re.escape(section_num)
    escaped_title = re.escape(section_title)
    header_re = re.compile(
        rf"^{escaped_num}\s+{escaped_title}\.?\s*[——\-]?\s*",
        re.IGNORECASE,
    )
    body = header_re.sub("", full_text).strip()

    # Fallback: if the regex didn't strip anything meaningful, try a simpler cut
    if body == full_text:
        # Find the first em-dash after the section number and cut there
        em_pos = full_text.find("—")
        if em_pos != -1 and em_pos < 300:
            body = full_text[em_pos + 1:].strip()

    return {
        **CHAPTER_META,
        "section_num": section_num,
        "section_title": section_title,
        "text": body,
        "history": history_text,
        "source_url": section_url(section_num),
    }


def parse_chapter(html: str) -> list[dict]:
    """Parse whole-chapter HTML into clean section records."""
    soup = BeautifulSoup(html, "lxml")
    records = []
    for div in soup.find_all("div", class_="Section"):
        record = parse_section_div(div)
        if record:
            records.append(record)
    return records


def main() -> list[dict]:
    print("Phase 1: Ingesting Chapter 119 — Florida Public Records Act")
    print(f"  Source: {CHAPTER_URL}\n")

    raw_path = RAW_DIR / "ch119_raw.html"
    if raw_path.exists():
        print("  Using cached raw HTML (delete data/raw/ch119_raw.html to re-fetch)")
        html = raw_path.read_text(encoding="utf-8")
    else:
        print("  Fetching page...")
        html = fetch(CHAPTER_URL)
        raw_path.write_text(html, encoding="utf-8")
        print(f"  Saved raw HTML ({len(html):,} bytes) -> {raw_path}")

    print("  Parsing sections...")
    records = parse_chapter(html)

    if not records:
        print("\n  ERROR: Zero sections parsed.")
        print("  Inspect data/raw/ch119_raw.html to debug.")
        sys.exit(1)

    out_path = DATA_DIR / "ch119_sections.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"  Parsed {len(records)} sections -> {out_path}")
    print()
    print("  Sections:")
    for r in records:
        tlen = len(r["text"])
        print(f"    [{r['section_num']}] {r['section_title'][:55]:<55}  "
              f"text={tlen:>6} chars  history={'yes' if r['history'] else 'NO'}")

    return records


if __name__ == "__main__":
    main()
