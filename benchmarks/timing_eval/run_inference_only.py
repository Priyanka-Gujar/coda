"""Re-run inference over already-transcribed cases with a different LLM.

Reuses the saved transcripts from a prior run (default: the qwen results) and
replays them through the champs agent with a new model, so no re-transcription
happens. Whole-file runs infer once on the full transcript; chunked runs feed
each saved chunk text through the agent in order (accumulating), reproducing the
trajectory with the new model. Output mirrors the CLI's layout in a separate
directory, so the source results are never overwritten.

    python run_inference_only.py                       # gpt-oss:20b -> real_cases_results_gptoss/
    python run_inference_only.py --only 1 2 --modes whole
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from coda.cli import build_agent, write_outputs

PROVIDER = "ollama"
MODEL = "gpt-oss:20b"
SRC_ROOT = HERE / "real_cases_results"
OUT_ROOT = HERE / "real_cases_results_gptoss"
PHASE_DIRS = ["phase1_va", "phase2_va_clinical"]
MODES = ["whole", "chunked"]


async def replay(src_dir, mode, provider, model):
    """Return (full_text, per_chunk) by replaying src_dir's transcript(s)."""
    agent = build_agent("champs", provider=provider, model=model)
    per_chunk = []
    if mode == "whole":
        text = (src_dir / "transcript.txt").read_text().strip()
        audio_s = json.loads((src_dir / "inference.json").read_text()).get("audio_duration_s", 0) or 0
        t = time.perf_counter()
        inf = await agent.process_chunk("chunk-0", text, [], time.time())
        per_chunk.append(("chunk-0", text, [], inf,
                          {"audio_s": audio_s, "grounding_s": 0.0,
                           "inference_s": round(time.perf_counter() - t, 3)}))
        return text, per_chunk
    for r in (json.loads(l) for l in (src_dir / "chunks.jsonl").open()):
        text = r.get("text", "")
        t = time.perf_counter()
        inf = await agent.process_chunk(r["chunk_id"], text, [], r.get("timestamp") or time.time())
        per_chunk.append((r["chunk_id"], text, [], inf,
                          {"audio_s": r.get("timing", {}).get("audio_s", 0.0),
                           "grounding_s": 0.0,
                           "inference_s": round(time.perf_counter() - t, 3)}))
    return agent.all_text.strip(), per_chunk


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", default=PROVIDER)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--src", default=str(SRC_ROOT), help="Results dir to reuse transcripts from")
    ap.add_argument("--out", default=str(OUT_ROOT), help="Output dir (kept separate from src)")
    ap.add_argument("--only", nargs="*", default=None, help="Case ids (default: all found in src)")
    ap.add_argument("--phases", default=",".join(PHASE_DIRS))
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--force", action="store_true", help="Re-run even if output exists")
    args = ap.parse_args()

    src_root, out_root = Path(args.src), Path(args.out)
    cases = args.only or sorted((d.name[4:] for d in src_root.glob("case*")), key=int)
    phases = [p for p in args.phases.split(",") if p]
    modes = [m for m in args.modes.split(",") if m]

    total = failed = 0
    for cid in cases:
        for phase in phases:
            for mode in modes:
                src_dir = src_root / f"case{cid}" / phase / mode
                out_dir = out_root / f"case{cid}" / phase / mode
                if not (src_dir / "inference.json").exists():
                    continue
                total += 1
                print(f"=== [{total}] case{cid} {phase} {mode}  ({args.model})")
                if (out_dir / "inference.json").exists() and not args.force:
                    print("    skip (already done)")
                    continue
                try:
                    full_text, per_chunk = asyncio.run(
                        replay(src_dir, mode, args.provider, args.model))
                except Exception as e:
                    failed += 1
                    print(f"!! FAILED: {e}", file=sys.stderr)
                    continue
                src_meta = json.loads((src_dir / "inference.json").read_text())
                meta = {"input": src_meta.get("input"), "input_type": "audio",
                        "mode": mode, "agent": "champs", "grounder": "none (reused transcript)",
                        "provider": args.provider, "model": args.model,
                        "transcriber": src_meta.get("transcriber"),
                        "whisper_model": src_meta.get("whisper_model"),
                        "reused_transcript_from": str(src_dir),
                        "audio_duration_s": src_meta.get("audio_duration_s")}
                write_outputs(out_dir, full_text, per_chunk, meta)
                top = per_chunk[-1][3].get("causes", {})
                best = max(top.values(), key=lambda c: c["score"])["name"] if top else "-"
                print(f"    -> top={best}")

    print(f"\nDone: {total} run(s), {failed} failure(s). Output in {out_root}")


if __name__ == "__main__":
    main()
