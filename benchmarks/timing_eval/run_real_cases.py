"""Run CODA over the real case recordings (real_cases/), all-local backends.

Fixed pipeline (nothing leaves the machine):
  transcription : faster-whisper, medium model
  grounding     : gilda (local resources)
  inference     : champs agent via Ollama gpt-oss:20b (localhost:11434)

Two phases mirror the study protocol:
  phase 1 - VA narrative only        (input = the case's VA recording)
  phase 2 - VA + clinical            (input = real_cases/combined/case<N>_combined.wav,
            one run so the agent carries VA context into the clinical segment)

Each phase runs in two modes:
  whole   - one-pass whole-file transcription + a single inference (final call)
  chunked - 20s chunks, inference per chunk (confidence trajectory / time-to-stabilization)

Phase 1 runs across all cases before phase 2, as the protocol asks.

Usage
-----
    python benchmarks/timing_eval/run_real_cases.py                 # everything
    python benchmarks/timing_eval/run_real_cases.py --only 10       # one case (smoke test)
    python benchmarks/timing_eval/run_real_cases.py --modes whole   # skip chunked
    python benchmarks/timing_eval/run_real_cases.py --dry-run
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from make_combined import concat_va_clinical

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[1]
SRC = REPO_ROOT / "src"
REAL_CASES = HERE / "real_cases"
COMBINED_DIR = REAL_CASES / "combined"
RESULTS_DIR = HERE / "real_cases_results"

# Fixed, all-local pipeline.
TRANSCRIBER = "faster-whisper"
WHISPER_MODEL = "medium"
GROUNDER = "gilda"
AGENT = "champs"
PROVIDER = "ollama"
MODEL = "qwen2.5:7b-instruct"
LANGUAGE = "en"
TASK = "transcribe"

CHUNK_SECONDS = 20.0
NO_SPEECH_THRESHOLD = 1.0


# --------------------------------------------------------------------------- setup

def load_cases():
    """case_id -> {"va": Path, "clinical": Path} from cases.csv."""
    cases = defaultdict(dict)
    with (REAL_CASES / "cases.csv").open() as fh:
        for row in csv.DictReader(fh):
            cases[row["case"]][row["type"]] = REAL_CASES / row["fname"]
    return cases


def audio_duration_s(path):
    """Spoken length in seconds via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    return round(float(out.stdout.strip()), 3)


def child_env():
    """coda is not pip-installed, so append (never clobber) its src to PYTHONPATH."""
    env = os.environ.copy()
    env["PYTHONPATH"] = (env.get("PYTHONPATH", "") + os.pathsep + str(SRC)).strip(os.pathsep)
    return env


# ----------------------------------------------------------------------------- run

def run_cli(input_path, out_dir, chunked, dry_run):
    """Invoke `python -m coda.cli` on one audio file; stream output to run.log."""
    cmd = [sys.executable, "-m", "coda.cli",
           "--input", str(input_path), "--output", str(out_dir),
           "--transcriber", TRANSCRIBER, "--whisper-model", WHISPER_MODEL,
           "--no-speech-threshold", str(NO_SPEECH_THRESHOLD),
           "--grounder", GROUNDER, "--agent", AGENT,
           "--provider", PROVIDER, "--model", MODEL,
           "--language", LANGUAGE, "--task", TASK, "--verbose"]
    if chunked:
        cmd += ["--chunking", str(CHUNK_SECONDS)]

    print("    " + " ".join(cmd))
    if dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "run.log").open("w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=child_env(), cwd=str(REPO_ROOT))
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


# ------------------------------------------------------------------------- analysis

def trajectory(out_dir):
    """Read chunks.jsonl into [(audio_elapsed_s, top_cause, top_score), ...]."""
    traj = []
    path = out_dir / "chunks.jsonl"
    if not path.exists():
        return traj
    with path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            causes = rec.get("causes", {})
            if causes:
                top = max(causes.values(), key=lambda c: c["score"])
                top_cause, top_score = top["name"], round(top["score"], 3)
            else:
                top_cause, top_score = None, 0.0
            traj.append((rec.get("audio_elapsed_s"), top_cause, top_score))
    return traj


def time_to_stabilization(traj):
    """First (audio_elapsed_s, cause) after which the top cause never changes and
    equals the final call. Model-agnostic 'time until CODA settles'. (None, None)
    if the trajectory is empty."""
    if not traj:
        return None, None
    final_cause = traj[-1][1]
    for i, (elapsed, cause, _) in enumerate(traj):
        if cause == final_cause and all(t[1] == final_cause for t in traj[i:]):
            return elapsed, cause
    return None, None


def final_call(traj):
    """(cause, score) at the last chunk."""
    if not traj:
        return None, 0.0
    _, cause, score = traj[-1]
    return cause, score


def whole_final(out_dir):
    """(cause, score) from a whole-file run's inference.json."""
    path = out_dir / "inference.json"
    if not path.exists():
        return None, 0.0
    causes = json.loads(path.read_text()).get("causes", {})
    if not causes:
        return None, 0.0
    top = max(causes.values(), key=lambda c: c["score"])
    return top["name"], round(top["score"], 3)


# ------------------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", default=None,
                        help="Case ids to run (default: all)")
    parser.add_argument("--phases", default="va,combined",
                        help="Comma-separated phases to run (default: va,combined)")
    parser.add_argument("--modes", default="whole,chunked",
                        help="Comma-separated modes to run (default: whole,chunked)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if a completed inference.json already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running them")
    args = parser.parse_args()

    cases = load_cases()
    case_ids = args.only if args.only else sorted(cases, key=int)
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    chunked_flags = {"whole": False, "chunked": True}

    # Resolve each case's audio inputs up front (build combined on demand).
    runnable = []
    for cid in case_ids:
        parts = cases.get(cid, {})
        if "va" not in parts or not parts["va"].exists():
            print(f"!! case {cid}: no VA recording, skipping", file=sys.stderr)
            continue
        entry = {"cid": cid, "va": parts["va"], "combined": None}
        if "clinical" in parts and parts["clinical"].exists():
            combined = COMBINED_DIR / f"case{cid}_combined.wav"
            if not combined.exists() and not args.dry_run:
                COMBINED_DIR.mkdir(parents=True, exist_ok=True)
                concat_va_clinical(parts["va"], parts["clinical"], combined)
            entry["combined"] = combined
        runnable.append(entry)

    total = 0
    failed = 0
    # Phase 1 (all cases) before phase 2, per protocol.
    for phase in phases:
        for entry in runnable:
            cid = entry["cid"]
            if phase == "va":
                input_path, phase_dir = entry["va"], "phase1_va"
            elif phase == "combined":
                if entry["combined"] is None:
                    print(f"   case {cid}: no clinical recording, skipping phase 2")
                    continue
                input_path, phase_dir = entry["combined"], "phase2_va_clinical"
            else:
                raise SystemExit(f"unknown phase '{phase}'")

            for mode in modes:
                out_dir = RESULTS_DIR / f"case{cid}" / phase_dir / mode
                total += 1
                print(f"=== [{total}] case{cid} {phase_dir} {mode}")
                # inference.json is written only when a CLI run finishes, so its
                # presence marks a completed run: skip it to resume after a stop.
                if (out_dir / "inference.json").exists() and not args.force:
                    print(f"    skip (already done): {out_dir}")
                    continue
                print(f"    input: {input_path}")
                code = run_cli(input_path, out_dir, chunked_flags[mode], args.dry_run)
                if code != 0:
                    failed += 1
                    print(f"!! FAILED ({code}): case{cid} {phase_dir} {mode} "
                          f"(see {out_dir / 'run.log'})", file=sys.stderr)

    if not args.dry_run:
        write_summary(runnable)

    print(f"\nDone: {total} run(s), {failed} failure(s).")
    return 1 if failed else 0


def write_summary(runnable):
    """Roll up per-case final calls and chunked time-to-stabilization into summary.{csv,json}."""
    rows = []
    for entry in runnable:
        cid = entry["cid"]
        case_dir = RESULTS_DIR / f"case{cid}"
        p1 = case_dir / "phase1_va"
        p2 = case_dir / "phase2_va_clinical"

        p1_traj = trajectory(p1 / "chunked")
        p2_traj = trajectory(p2 / "chunked")
        p1_t, p1_cause_t = time_to_stabilization(p1_traj)
        p2_t, p2_cause_t = time_to_stabilization(p2_traj)
        p1_wc, p1_ws = whole_final(p1 / "whole")
        p2_wc, p2_ws = whole_final(p2 / "whole")
        p1_cc, p1_cs = final_call(p1_traj)
        p2_cc, p2_cs = final_call(p2_traj)

        rows.append({
            "case_id": cid,
            "va_duration_s": audio_duration_s(entry["va"]),
            "combined_duration_s": audio_duration_s(entry["combined"]) if entry["combined"] else None,
            "p1_whole_cause": p1_wc, "p1_whole_score": p1_ws,
            "p1_chunk_final_cause": p1_cc, "p1_chunk_final_score": p1_cs,
            "p1_time_to_stable_s": p1_t, "p1_stable_cause": p1_cause_t,
            "p2_whole_cause": p2_wc, "p2_whole_score": p2_ws,
            "p2_chunk_final_cause": p2_cc, "p2_chunk_final_score": p2_cs,
            "p2_time_to_stable_s": p2_t, "p2_stable_cause": p2_cause_t,
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(rows, indent=2))
    if rows:
        with (RESULTS_DIR / "summary.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nSummary written to {(RESULTS_DIR / 'summary.csv').resolve()}")


if __name__ == "__main__":
    main()
