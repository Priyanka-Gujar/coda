"""Aggregate accuracy vs spoken time (chunked, qwen).

For each phase, steps every case's chunked prediction onto a 1s grid, holding
the final prediction after a case ends (frozen), so n stays constant. Plots
top-1 and top-3 accuracy (reference = underlying cause) over spoken seconds.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE / "real_cases_results"
RESULTS_GPTOSS = HERE / "real_cases_results_gptoss"
REPORT = HERE / "real_cases_report"
MODELS = [("qwen2.5:7b", RESULTS, "-"), ("gpt-oss:20b", RESULTS_GPTOSS, "--")]
TRUTH = json.loads((HERE / "real_cases" / "true_labels_group.json").read_text())
CASES = sorted(TRUTH, key=int)

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e2"
C_TOP1, C_TOP3 = "#2a78d6", "#eb6834"
PHASES = [("phase1_va", "VA narrative only"),
          ("phase2_va_clinical", "VA + clinical")]


def ref_cause(k):
    return TRUTH[k].get("Underlying Cause", ["?"])[0]


def chunk_points(root, k, phase):
    """[(elapsed_s, top1_correct, top3_correct)] over a case's chunks."""
    p = root / f"case{k}" / phase / "chunked" / "chunks.jsonl"
    ref = ref_cause(k)
    pts = []
    for line in p.open():
        r = json.loads(line)
        ranked = [v["name"] for v in sorted(r.get("causes", {}).values(),
                                            key=lambda x: -x["score"])]
        pts.append((r["audio_elapsed_s"], ranked[:1] == [ref], ref in ranked[:3]))
    return pts


def va_duration(k):
    p = RESULTS / f"case{k}" / "phase1_va" / "whole" / "inference.json"
    return json.loads(p.read_text()).get("audio_duration_s")


def step_onto(grid, pts, idx):
    """Forward-fill correctness (component idx of each point) onto grid; 0 before
    the first chunk, frozen at the last value after the final chunk."""
    out = np.zeros(len(grid))
    ci, cur = 0, 0
    for i, t in enumerate(grid):
        while ci < len(pts) and pts[ci][0] <= t:
            cur = int(pts[ci][idx])
            ci += 1
        out[i] = cur
    return out


def phase_curves(root, phase, grid):
    per_case = {k: chunk_points(root, k, phase) for k in CASES}
    top1 = np.mean([step_onto(grid, per_case[k], 1) for k in CASES], axis=0)
    top3 = np.mean([step_onto(grid, per_case[k], 2) for k in CASES], axis=0)
    return top1, top3


def phase_grid(phase):
    tmax = max(chunk_points(RESULTS, k, phase)[-1][0] for k in CASES)
    return np.arange(0, tmax + 1, 1.0)


def main():
    REPORT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (phase, label) in zip(axes, PHASES):
        ax.set_facecolor(SURFACE)
        grid = phase_grid(phase)
        if phase == "phase2_va_clinical":
            vas = [va_duration(k) for k in CASES]
            ax.axvspan(min(vas), max(vas), color=MUTED, alpha=0.08, zorder=0)
            ax.text(np.median(vas), 1.02, "clinical onset (range across cases)",
                    color=MUTED, fontsize=8, ha="center")
        for name, root, ls in MODELS:
            if not root.exists():
                continue
            top1, top3 = phase_curves(root, phase, grid)
            ax.plot(grid, top3, color=C_TOP3, lw=2.2, ls=ls, label=f"{name} top-3")
            ax.plot(grid, top1, color=C_TOP1, lw=2.2, ls=ls, label=f"{name} top-1")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, grid[-1])
        ax.set_title(label, fontsize=12, color=INK)
        ax.set_xlabel("spoken audio time (s)", fontsize=10, color=MUTED)
        ax.tick_params(labelsize=9, colors=MUTED, labelleft=True)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.grid(True, color=GRID, lw=0.6)
        ax.legend(fontsize=9, loc="upper right", facecolor=SURFACE, edgecolor=GRID)
    axes[0].set_ylabel("accuracy (fraction of 20 cases)", fontsize=10, color=MUTED)
    fig.tight_layout()
    out = REPORT / "accuracy_over_time.png"
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
