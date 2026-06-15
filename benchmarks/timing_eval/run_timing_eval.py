"""CODA prototype time study: how long must someone talk before CODA is confident?

For each synthetic case (cases.json) this synthesizes spoken audio, feeds it to the
CODA CLI in live-style 20s chunks, and records the top cause-of-death confidence as
the spoken "wall clock" advances. The headline number per case is the spoken time at
which the top cause first crosses the confidence threshold (default 0.75).

Two phases mirror the study protocol:
  Phase 1 - VA narrative only          (audio = <case>_va.wav)
  Phase 2 - VA narrative + clinical data (audio = <case>_combined.wav, one run so the
            agent's accumulated dialogue carries VA context into the clinical segment)

The evaluation runs over a matrix of (Whisper model x inference LLM). LLM specs are
"provider:model" (e.g. "ollama:gpt-oss:20b", "openai:gpt-4o-mini"); the bare spec
"toy" uses CODA's rule-based toy agent (handy for plumbing checks without an LLM).

Usage
-----
    # default matrix: gpt-oss:20b (Ollama) x Whisper small
    python benchmarks/timing_eval/run_timing_eval.py

    # custom matrix
    python benchmarks/timing_eval/run_timing_eval.py \
        --whisper-models small,medium \
        --llm-models "ollama:gpt-oss:20b,openai:gpt-4o-mini"

    # fast plumbing check, no LLM/Ollama needed
    python benchmarks/timing_eval/run_timing_eval.py --whisper-models tiny --llm-models toy

Requires macOS `say` and `ffmpeg` on PATH for speech synthesis.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[1]
SRC = REPO_ROOT / "src"

SAY_VOICE = "Samantha"   # single consistent narrator
CHUNK_SECONDS = 20.0
THRESHOLD = 0.75


# --------------------------------------------------------------------------- audio

def _say_to_wav(text: str, out_path: Path):
    """Render text to a 16kHz mono WAV via macOS `say` + ffmpeg (Whisper's format)."""
    aiff = out_path.with_suffix(".aiff")
    subprocess.run(["say", "-v", SAY_VOICE, "-o", str(aiff), text], check=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(aiff), "-ar", "16000", "-ac", "1", str(out_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    aiff.unlink(missing_ok=True)


def _concat_wavs(a: Path, b: Path, out_path: Path):
    """Concatenate two same-format WAVs (VA then clinical) into one."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(a), "-i", str(b),
         "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1", str(out_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _wav_duration_s(path: Path) -> float:
    with wave.open(str(path)) as w:
        return round(w.getnframes() / w.getframerate(), 3)


def ensure_audio(case: dict, audio_dir: Path, regenerate: bool) -> dict:
    """Synthesize va/clinical/combined WAVs for a case (cached). Returns durations."""
    cid = case["case_id"]
    va_wav = audio_dir / f"{cid}_va.wav"
    clin_wav = audio_dir / f"{cid}_clinical.wav"
    combined_wav = audio_dir / f"{cid}_combined.wav"

    if regenerate or not va_wav.exists():
        _say_to_wav(case["va_narrative"], va_wav)
    if regenerate or not clin_wav.exists():
        _say_to_wav(case["clinical_narrative"], clin_wav)
    if regenerate or not combined_wav.exists():
        _concat_wavs(va_wav, clin_wav, combined_wav)

    return {
        "va_wav": va_wav, "combined_wav": combined_wav,
        "va_duration_s": _wav_duration_s(va_wav),
        "clinical_duration_s": _wav_duration_s(clin_wav),
        "combined_duration_s": _wav_duration_s(combined_wav),
    }


# ----------------------------------------------------------------------------- run

def parse_llm_spec(spec: str):
    """('toy', None, None) for the toy agent, else ('champs', provider, model)."""
    spec = spec.strip()
    if spec == "toy":
        return "toy", None, None
    if ":" not in spec:
        raise ValueError(f"LLM spec '{spec}' must be 'provider:model' or 'toy'")
    provider, model = spec.split(":", 1)
    return "champs", provider, model


def cell_dirname(whisper_model: str, llm_spec: str) -> str:
    return f"whisper-{whisper_model}__{llm_spec}".replace(":", "-").replace("/", "-")


def run_cli(audio_path: Path, out_dir: Path, whisper_model: str, llm_spec: str):
    """Invoke `python -m coda.cli` on one audio file in 20s chunked mode."""
    agent, provider, model = parse_llm_spec(llm_spec)
    cmd = [sys.executable, "-m", "coda.cli",
           "--input", str(audio_path), "--output", str(out_dir),
           "--chunking", str(CHUNK_SECONDS),
           "--agent", agent, "--grounder", "gilda",
           "--whisper-model", whisper_model, "--language", "en"]
    if provider:
        cmd += ["--provider", provider, "--model", model]

    env = os.environ.copy()
    env["PYTHONPATH"] = (env.get("PYTHONPATH", "") + os.pathsep + str(SRC)).strip(os.pathsep)
    subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), check=True)


# ------------------------------------------------------------------------- analysis

def trajectory(out_dir: Path):
    """Read chunks.jsonl into [(audio_elapsed_s, top_cause, top_score), ...]."""
    traj = []
    with (out_dir / "chunks.jsonl").open() as fh:
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


def time_to_threshold(traj, threshold):
    """First (audio_elapsed_s, cause) where top score >= threshold, else (None, None)."""
    for elapsed, cause, score in traj:
        if score >= threshold:
            return elapsed, cause
    return None, None


def final_call(traj):
    """(cause, score) at the last chunk."""
    if not traj:
        return None, 0.0
    _, cause, score = traj[-1]
    return cause, score


# ------------------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--whisper-models", default="small",
                        help="Comma-separated Whisper model sizes (default: small)")
    parser.add_argument("--llm-models", default="ollama:gpt-oss:20b",
                        help="Comma-separated 'provider:model' specs, or 'toy' "
                             "(default: ollama:gpt-oss:20b)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"Confidence threshold for ascertainment (default: {THRESHOLD})")
    parser.add_argument("--cases", default=str(HERE / "cases.json"),
                        help="Path to cases.json")
    parser.add_argument("--output", default=str(HERE / "results"),
                        help="Output root for per-run results and summary")
    parser.add_argument("--regenerate-audio", action="store_true",
                        help="Re-synthesize audio even if cached WAVs exist")
    args = parser.parse_args()

    if shutil.which("say") is None or shutil.which("ffmpeg") is None:
        raise SystemExit("This script requires macOS `say` and `ffmpeg` on PATH.")

    cases = json.loads(Path(args.cases).read_text())
    whisper_models = [m.strip() for m in args.whisper_models.split(",") if m.strip()]
    llm_specs = [m.strip() for m in args.llm_models.split(",") if m.strip()]

    out_root = Path(args.output)
    audio_dir = HERE / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print("Synthesizing audio...")
    audio = {c["case_id"]: ensure_audio(c, audio_dir, args.regenerate_audio) for c in cases}

    rows = []
    for whisper_model in whisper_models:
        for llm_spec in llm_specs:
            cell = cell_dirname(whisper_model, llm_spec)
            print(f"\n=== Cell: Whisper={whisper_model}  LLM={llm_spec} ===")
            for case in cases:
                cid = case["case_id"]
                a = audio[cid]
                p1_dir = out_root / cell / cid / "phase1_va"
                p2_dir = out_root / cell / cid / "phase2_va_clinical"

                print(f"  {cid} phase 1 (VA, {a['va_duration_s']}s)...")
                run_cli(a["va_wav"], p1_dir, whisper_model, llm_spec)
                print(f"  {cid} phase 2 (VA+clinical, {a['combined_duration_s']}s)...")
                run_cli(a["combined_wav"], p2_dir, whisper_model, llm_spec)

                p1_traj, p2_traj = trajectory(p1_dir), trajectory(p2_dir)
                p1_t75, p1_cause75 = time_to_threshold(p1_traj, args.threshold)
                p2_t75, p2_cause75 = time_to_threshold(p2_traj, args.threshold)
                p1_fc, p1_fs = final_call(p1_traj)
                p2_fc, p2_fs = final_call(p2_traj)

                rows.append({
                    "whisper_model": whisper_model, "llm": llm_spec,
                    "case_id": cid, "topic": case["topic"],
                    "va_duration_s": a["va_duration_s"],
                    "clinical_duration_s": a["clinical_duration_s"],
                    "combined_duration_s": a["combined_duration_s"],
                    "p1_time_to_thresh_s": p1_t75, "p1_cause_at_thresh": p1_cause75,
                    "p1_final_cause": p1_fc, "p1_final_score": p1_fs,
                    "p2_time_to_thresh_s": p2_t75, "p2_cause_at_thresh": p2_cause75,
                    "p2_final_cause": p2_fc, "p2_final_score": p2_fs,
                })

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(rows, indent=2))
    if rows:
        with (out_root / "summary.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nThreshold = {args.threshold}.  'time_to_thresh_s' = spoken seconds until the "
          f"top cause first reaches it ('-' = never within the narrative).\n")
    hdr = f"{'cell':<32} {'case':<9} {'P1 t75':>7} {'P1 final':<28} {'P2 t75':>7} {'P2 final':<28}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cell = f"{r['whisper_model']}/{r['llm']}"
        p1 = f"{r['p1_time_to_thresh_s']}s" if r["p1_time_to_thresh_s"] is not None else "-"
        p2 = f"{r['p2_time_to_thresh_s']}s" if r["p2_time_to_thresh_s"] is not None else "-"
        p1f = f"{r['p1_final_cause']} ({r['p1_final_score']})"
        p2f = f"{r['p2_final_cause']} ({r['p2_final_score']})"
        print(f"{cell:<32} {r['case_id']:<9} {p1:>7} {p1f:<28} {p2:>7} {p2f:<28}")
    print(f"\nResults written to {out_root.resolve()}/")


if __name__ == "__main__":
    main()
