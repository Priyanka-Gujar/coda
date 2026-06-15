"""Cause-of-death accuracy for the CODA time study (not certainty-based).

For each case, the agent's *final* inference (end of narrative) is compared to the
expected underlying cause (cases.json `expected_cause`, a CHAMPS group cause taken
from the CHAMPS resource page the case is adapted from). Exact group-cause-name match.

Reports the 2x2 the protocol cares about, per model:
    {Top-1 match, Top-3 contains} x {VA only (Phase 1), VA + clinical (Phase 2)}

Usage
-----
    python benchmarks/timing_eval/evaluate_accuracy.py
"""
import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
PHASES = {"phase1_va": "VA only", "phase2_va_clinical": "VA + clinical"}


def final_ranked_causes(phase_dir: Path):
    """Names of the final chunk's causes, highest score first (top of list = top-1)."""
    lines = (phase_dir / "chunks.jsonl").read_text().splitlines()
    if not lines:
        return []
    causes = json.loads(lines[-1]).get("causes", {})
    return [c["name"] for c in sorted(causes.values(), key=lambda c: c["score"], reverse=True)]


def collect(results_root: Path, expected: dict) -> pd.DataFrame:
    rows = []
    for cell_dir in sorted(p for p in results_root.iterdir()
                           if p.is_dir() and p.name != "analysis"):
        for case_dir in sorted(p for p in cell_dir.iterdir() if p.is_dir()):
            gold = expected.get(case_dir.name)
            for phase_key, phase_label in PHASES.items():
                phase_dir = case_dir / phase_key
                if not (phase_dir / "chunks.jsonl").exists():
                    continue
                ranked = final_ranked_causes(phase_dir)
                top1 = ranked[0] if ranked else None
                top3 = ranked[:3]
                rows.append({
                    "cell": cell_dir.name, "case": case_dir.name, "phase": phase_label,
                    "expected": gold, "top1": top1, "top3": "; ".join(top3),
                    "hit_top1": gold == top1,
                    "hit_top3": gold in top3,
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(HERE / "results"))
    parser.add_argument("--cases", default=str(HERE / "cases.json"))
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text())
    expected = {c["case_id"]: c.get("expected_cause") for c in cases}
    if any(v is None for v in expected.values()):
        raise SystemExit("Some cases lack 'expected_cause' in cases.json.")

    results_root = Path(args.results)
    df = collect(results_root, expected)
    if df.empty:
        raise SystemExit(f"No results under {results_root}.")

    out_dir = results_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "accuracy_per_case.csv", index=False)

    # Aggregate: accuracy = fraction of cases hit, per (cell, phase).
    agg = (df.groupby(["cell", "phase"])
             .agg(n=("case", "count"),
                  top1_acc=("hit_top1", "mean"),
                  top3_acc=("hit_top3", "mean"))
             .reset_index())
    agg["top1_acc"] = (agg["top1_acc"] * 100).round(1)
    agg["top3_acc"] = (agg["top3_acc"] * 100).round(1)

    # Wide 2x2 table: rows = model, columns = phase x {Top-1, Top-3}.
    wide = agg.pivot(index="cell", columns="phase", values=["top1_acc", "top3_acc"])
    wide = wide.reorder_levels([1, 0], axis=1).sort_index(axis=1)
    agg.to_csv(out_dir / "accuracy_summary.csv", index=False)

    print("\nExpected underlying cause (from CHAMPS resource pages):")
    for c in cases:
        print(f"  {c['case_id']}: {c['expected_cause']}")

    print(f"\n=== Accuracy (% of {df['case'].nunique()} cases), exact group-cause match "
          "on the final inference ===\n")
    print(wide.to_string())
    print("\nPer-case detail:\n")
    for cell, g in df.groupby("cell"):
        print(cell)
        for _, r in g.iterrows():
            m1 = "T1" if r["hit_top1"] else ("T3" if r["hit_top3"] else "  ")
            print(f"  [{m1:>2}] {r['case']} {r['phase']:<14} exp={r['expected']:<28} "
                  f"top1={str(r['top1'])}")
        print()
    print(f"Wrote accuracy_per_case.csv and accuracy_summary.csv to {out_dir}/")


if __name__ == "__main__":
    main()
