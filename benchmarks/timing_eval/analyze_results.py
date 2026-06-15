"""Summarize and plot the CODA time-study results produced by run_timing_eval.py.

Walks the results tree (results/<cell>/<case>/<phase>/chunks.jsonl), computes the
spoken-time-to-threshold per case, aggregates per (cell, phase), and renders:

  - aggregate_stats.csv  : n cases, ascertainment rate, mean/median/IQR time-to-threshold
  - per_case.csv         : one row per (cell, case, phase)
  - trajectories_<cell>.png : confidence-vs-spoken-time curves, one panel per case,
                              with the threshold line and the VA->clinical boundary
  - time_to_threshold.png   : distribution of time-to-threshold by phase, per cell

Works on partial results (e.g. only some cases finished).

Usage
-----
    python benchmarks/timing_eval/analyze_results.py
    python benchmarks/timing_eval/analyze_results.py --threshold 0.75 --results <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_timing_eval import trajectory, time_to_threshold, final_call

HERE = Path(__file__).parent
PHASES = {"phase1_va": "Phase 1 (VA only)",
          "phase2_va_clinical": "Phase 2 (VA + clinical)"}


def va_boundary_s(case_dir: Path):
    """Spoken seconds at which clinical data begins (= VA audio length), or None."""
    inf = case_dir / "phase1_va" / "inference.json"
    if inf.exists():
        return json.loads(inf.read_text()).get("audio_duration_s")
    return None


def collect(results_root: Path, threshold: float):
    """Return (per_case_df, points_df) across every cell/case/phase found."""
    per_case, points = [], []
    for cell_dir in sorted(p for p in results_root.iterdir() if p.is_dir()
                           and p.name != "analysis"):
        for case_dir in sorted(p for p in cell_dir.iterdir() if p.is_dir()):
            boundary = va_boundary_s(case_dir)
            for phase_key, phase_label in PHASES.items():
                chunks = case_dir / phase_key / "chunks.jsonl"
                if not chunks.exists():
                    continue
                traj = trajectory(case_dir / phase_key)
                t_thr, cause_thr = time_to_threshold(traj, threshold)
                final_cause, final_score = final_call(traj)
                per_case.append({
                    "cell": cell_dir.name, "case": case_dir.name,
                    "phase": phase_label,
                    "time_to_threshold_s": t_thr,
                    "reached_threshold": t_thr is not None,
                    "cause_at_threshold": cause_thr,
                    "final_cause": final_cause, "final_score": final_score,
                    "va_boundary_s": boundary,
                })
                for elapsed, cause, score in traj:
                    points.append({"cell": cell_dir.name, "case": case_dir.name,
                                   "phase": phase_label, "elapsed_s": elapsed,
                                   "top_cause": cause, "top_score": score,
                                   "va_boundary_s": boundary})
    return pd.DataFrame(per_case), pd.DataFrame(points)


def aggregate(per_case: pd.DataFrame) -> pd.DataFrame:
    """Per (cell, phase): n cases, ascertainment rate, time-to-threshold stats."""
    out = []
    for (cell, phase), g in per_case.groupby(["cell", "phase"], sort=False):
        reached = g[g["reached_threshold"]]
        times = reached["time_to_threshold_s"].to_numpy(dtype=float)
        out.append({
            "cell": cell, "phase": phase,
            "n_cases": len(g),
            "n_reached": len(reached),
            "ascertainment_rate": round(len(reached) / len(g), 3) if len(g) else None,
            "mean_time_s": round(float(np.mean(times)), 1) if times.size else None,
            "median_time_s": round(float(np.median(times)), 1) if times.size else None,
            "iqr_lo_s": round(float(np.percentile(times, 25)), 1) if times.size else None,
            "iqr_hi_s": round(float(np.percentile(times, 75)), 1) if times.size else None,
            "mean_final_score": round(float(g["final_score"].mean()), 3),
        })
    return pd.DataFrame(out)


def plot_trajectories(points: pd.DataFrame, threshold: float, out_dir: Path):
    """One figure per cell: confidence-vs-spoken-time, a panel per case."""
    import matplotlib.pyplot as plt

    for cell, cg in points.groupby("cell"):
        cases = sorted(cg["case"].unique())
        ncol = min(3, len(cases))
        nrow = int(np.ceil(len(cases) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow),
                                 squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        for i, case in enumerate(cases):
            ax = axes.flat[i]
            ax.set_visible(True)
            g = cg[cg["case"] == case]
            for phase, pg in g.groupby("phase"):
                pg = pg.sort_values("elapsed_s")
                ax.plot(pg["elapsed_s"], pg["top_score"], marker="o", ms=4,
                        drawstyle="steps-post", label=phase)
            boundary = g["va_boundary_s"].dropna()
            if not boundary.empty:
                ax.axvline(float(boundary.iloc[0]), ls=":", color="gray", lw=1,
                           label="clinical data starts")
            ax.axhline(threshold, ls="--", color="red", lw=1)
            ax.set_title(case, fontsize=10)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("spoken time (s)")
            ax.set_ylabel("top-cause confidence")
        # one shared legend
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   fontsize=9, frameon=False)
        fig.suptitle(f"Confidence trajectory - {cell} (threshold {threshold})",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        path = out_dir / f"trajectories_{cell}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"  wrote {path}")


def plot_time_distribution(per_case: pd.DataFrame, out_dir: Path):
    """Strip plot of time-to-threshold by phase, one panel per cell."""
    import matplotlib.pyplot as plt

    cells = sorted(per_case["cell"].unique())
    fig, axes = plt.subplots(1, len(cells), figsize=(5 * len(cells), 4),
                             squeeze=False)
    phases = list(PHASES.values())
    for j, cell in enumerate(cells):
        ax = axes[0][j]
        g = per_case[per_case["cell"] == cell]
        for x, phase in enumerate(phases):
            pg = g[g["phase"] == phase]
            reached = pg[pg["reached_threshold"]]
            y = reached["time_to_threshold_s"].astype(float)
            jitter = np.random.uniform(-0.08, 0.08, size=len(y))
            ax.scatter(np.full(len(y), x) + jitter, y, s=40, alpha=0.7, zorder=3)
            if len(y):
                ax.hlines(np.median(y), x - 0.2, x + 0.2, color="black", lw=2)
            n_never = len(pg) - len(reached)
            ax.annotate(f"{len(reached)}/{len(pg)} reached"
                        + (f"\n{n_never} never" if n_never else ""),
                        (x, ax.get_ylim()[1] if len(y) else 1), ha="center",
                        va="top", fontsize=8, color="gray")
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels([p.replace(" (", "\n(") for p in phases], fontsize=9)
        ax.set_ylabel("spoken time to threshold (s)")
        ax.set_title(cell, fontsize=10)
        ax.set_ylim(bottom=0)
    fig.suptitle("Spoken time to confident ascertainment", fontsize=12)
    fig.tight_layout()
    path = out_dir / "time_to_threshold.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(HERE / "results"),
                        help="Results root produced by run_timing_eval.py")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Confidence threshold for ascertainment (default: 0.75)")
    parser.add_argument("--no-plots", action="store_true", help="Stats only, skip plots")
    args = parser.parse_args()

    results_root = Path(args.results)
    if not results_root.exists():
        raise SystemExit(f"No results at {results_root}. Run run_timing_eval.py first.")

    per_case, points = collect(results_root, args.threshold)
    if per_case.empty:
        raise SystemExit(f"No chunks.jsonl found under {results_root}.")

    out_dir = results_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    agg = aggregate(per_case)
    per_case.to_csv(out_dir / "per_case.csv", index=False)
    agg.to_csv(out_dir / "aggregate_stats.csv", index=False)

    print("\n=== Aggregate stats (threshold "
          f"{args.threshold}) ===\n")
    print(agg.to_string(index=False))
    print(f"\nWrote per_case.csv and aggregate_stats.csv to {out_dir}/")

    if not args.no_plots:
        print("\nPlots:")
        plot_trajectories(points, args.threshold, out_dir)
        plot_time_distribution(per_case, out_dir)

    print(f"\nDone. Analysis in {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
