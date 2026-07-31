"""Extract the "True Labels - Group" causes per case from a case-dossier HTML.

Walks the document in order, tracking the current case from `<h2>Case N ...</h2>`
headings, and for each "True Labels - Group" block collects the cause(s) under
each cause-type label (e.g. "Underlying Cause", "Immediate Cause of Death").

Output: JSON mapping case number -> {cause_type: [causes]}.
"""
import argparse
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[1]
DEFAULT_HTML = HERE / "real_cases" / "dry_run_case_dossiers-corrected2.html"
DEFAULT_OUT = HERE / "real_cases" / "true_labels_group.json"
GROUP_CAUSES_FILE = REPO_ROOT / "src" / "coda" / "resources" / "champs" / "group_causes.txt"

CASE_RE = re.compile(r"Case\s+(\d+)")


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def load_canonical():
    return [norm(l) for l in GROUP_CAUSES_FILE.read_text().splitlines() if l.strip()]


def split_causes(value, canonical):
    """A cell may hold several comma-separated causes, but cause names also
    contain commas ('Other endocrine, metabolic, blood, and immune disorders').
    Segment by greedily matching against the canonical group-cause list; fall
    back to the whole cell if a fragment doesn't resolve."""
    value = norm(value)
    canon = {c.lower(): c for c in canonical}
    tokens = [t.strip() for t in value.split(",")]
    out = []
    acc = []
    for tok in tokens:
        acc.append(tok)
        joined = ", ".join(acc)
        if joined.lower() in canon:
            out.append(canon[joined.lower()])
            acc = []
    if acc:
        out.append(", ".join(acc))
    return out or ([value] if value else [])


def extract(html_path):
    soup = BeautifulSoup(Path(html_path).read_text(encoding="utf-8"), "html.parser")
    canonical = load_canonical()
    results = {}
    current = None
    for el in soup.find_all(["h2", "h3"]):
        text = el.get_text(" ", strip=True)
        m = CASE_RE.search(text)
        if el.name == "h2" and m:
            current = int(m.group(1))
            results.setdefault(current, {})
            continue
        if el.name == "h3" and "True Labels" in text and "Group" in text and current:
            block = el.find_parent("div") or el.parent
            for p in block.find_all("p"):
                strong = p.find("strong")
                if not strong:
                    continue
                label = strong.get_text(strip=True).rstrip(":").strip()
                value = p.get_text(" ", strip=True)
                value = value.replace(strong.get_text(strip=True), "", 1).lstrip(": ").strip()
                for cause in split_causes(value, canonical):
                    bucket = results[current].setdefault(label, [])
                    if cause not in bucket:
                        bucket.append(cause)
    return {str(k): results[k] for k in sorted(results)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--html", default=str(DEFAULT_HTML))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    data = extract(args.html)
    Path(args.out).write_text(json.dumps(data, indent=2))
    print(f"Wrote {len(data)} cases to {args.out}")
    for case, causes in data.items():
        print(f"  case {case}: {causes}")


if __name__ == "__main__":
    main()
