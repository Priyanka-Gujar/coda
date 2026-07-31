"""Per-case presentation of the real-cases timing eval.

Outputs into real_cases_report/:
  final_predictions.csv   one-shot top-1/top-3 vs all reference causes, plus timing
  index.html              per-case reference causes + final chunked predictions,
                          reasoning, and follow-up questions (VA vs VA+clinical)
"""
import csv
import html
import json
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "real_cases_results"
RESULTS_GPTOSS = HERE / "real_cases_results_gptoss"
REPORT = HERE / "real_cases_report"
TRUTH = json.loads((HERE / "real_cases" / "true_labels_group.json").read_text())
CASES = sorted(TRUTH, key=int)


def ref_types(k):
    """[(type, [causes])] in COD order for case k."""
    order = ["Underlying Cause", "Immediate Cause of Death", "Morbid Conditions"]
    t = TRUTH[k]
    return [(name, t[name]) for name in order if name in t]


def whole_top(path, n=3):
    if not path.exists():
        return []
    c = json.loads(path.read_text()).get("causes", {})
    return [v["name"] for v in sorted(c.values(), key=lambda x: -x["score"])][:n]


def whole_timing(path):
    """(audio_duration_s, total_processing_s) from a whole-file run, or (None, None)."""
    if not path.exists():
        return None, None
    d = json.loads(path.read_text())
    return d.get("audio_duration_s"), d.get("timing", {}).get("total_s")


def chunk_final(chunked_dir):
    """Final prediction plus timing: top-3, reasoning, questions from the last
    chunk with causes; audio length and the final chunk's inference time (that
    last inference runs over the whole accumulated transcript)."""
    empty = {"top3": [], "reasoning": "", "questions": [], "audio_s": None, "infer_s": None}
    p = chunked_dir / "chunks.jsonl"
    if not p.exists():
        return empty
    rows = [json.loads(l) for l in p.open()]
    if not rows:
        return empty
    last = rows[-1]
    with_causes = next((r for r in reversed(rows) if r.get("causes")), None)
    if with_causes is None:
        return empty
    top3 = [(v["name"], v["score"]) for v in
            sorted(with_causes["causes"].values(), key=lambda x: -x["score"])[:3]]
    return {"top3": top3, "reasoning": with_causes.get("reasoning") or "",
            "questions": with_causes.get("questions") or [],
            "audio_s": last.get("audio_elapsed_s"),
            "infer_s": last.get("timing", {}).get("inference_s")}


def write_table():
    out = REPORT / "final_predictions.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "underlying", "immediate", "morbid_conditions",
                    "va_top1", "va_top3", "combined_top1", "combined_top3",
                    "va_audio_s", "va_processing_s",
                    "combined_audio_s", "combined_processing_s"])
        for k in CASES:
            t = TRUTH[k]
            va_p = RESULTS / f"case{k}" / "phase1_va" / "whole" / "inference.json"
            cm_p = RESULTS / f"case{k}" / "phase2_va_clinical" / "whole" / "inference.json"
            va, cm = whole_top(va_p), whole_top(cm_p)
            va_audio, va_proc = whole_timing(va_p)
            cm_audio, cm_proc = whole_timing(cm_p)
            w.writerow([k,
                        "; ".join(t.get("Underlying Cause", [])),
                        "; ".join(t.get("Immediate Cause of Death", [])),
                        "; ".join(t.get("Morbid Conditions", [])),
                        va[0] if va else "-", " | ".join(va),
                        cm[0] if cm else "-", " | ".join(cm),
                        va_audio, va_proc, cm_audio, cm_proc])
    print(f"wrote {out}")


def whole_transcription(path):
    """(audio_duration_s, transcription_s) from a whole-file run."""
    if not path.exists():
        return None, None
    d = json.loads(path.read_text())
    return d.get("audio_duration_s"), d.get("timing", {}).get("transcription_s")


def phase_block(title, k, phase):
    d = chunk_final(RESULTS_GPTOSS / f"case{k}" / phase / "chunked")
    audio_s, trans_s = whole_transcription(RESULTS / f"case{k}" / phase / "whole" / "inference.json")
    top3 = ", ".join(f"{html.escape(n)} ({s:.2f})" for n, s in d["top3"]) or "-"
    qs = "".join(f"<li>{html.escape(q)}</li>" for q in d["questions"])
    audio = f"{audio_s:.0f}s" if audio_s is not None else "-"
    trans = f"{trans_s:.0f}s" if trans_s is not None else "-"
    infer = f"{d['infer_s']:.1f}s" if d["infer_s"] is not None else "-"
    return (f"<div class='phase'><h3>{title}</h3>"
            f"<p class='timing'>Audio duration: {audio}, CODA processing time: "
            f"{trans} transcription / {infer} inference</p>"
            f"<p class='top'><b>Final top-3:</b> {top3}</p>"
            f"<p class='reason'>{html.escape(d['reasoning'])}</p>"
            f"<b style='font-size:13px'>Follow-up questions</b><ul>{qs}</ul></div>")


def write_index():
    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             "<title>CODA timing evaluation on CHAMPS cases</title>",
             "<style>",
             "body{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;",
             "margin:0;background:#fcfcfb;color:#0b0b0b}",
             "header{position:sticky;top:0;background:#fcfcfb;border-bottom:1px solid #e6e5e2;",
             "padding:12px 24px}",
             "nav a{margin-right:10px;color:#2a78d6;text-decoration:none;font-size:13px}",
             "main{padding:0 24px 60px}",
             ".case{border-bottom:1px solid #e6e5e2;padding:22px 0}",
             ".case h2{margin:0 0 4px}",
             ".refs{color:#52514e;font-size:14px;margin:0 0 4px}",
             ".refs b{color:#0b0b0b}",
             ".detail{display:flex;gap:24px;margin-top:12px}",
             ".phase{flex:1;background:#f7f6f3;border:1px solid #e6e5e2;",
             "border-radius:6px;padding:12px 16px}",
             ".phase h3{margin:0 0 6px;font-size:14px}",
             ".phase .timing{font-size:12px;color:#52514e;margin:0 0 8px}",
             ".phase .top{font-size:13px;color:#0b0b0b;margin:0 0 8px}",
             ".phase .reason{font-size:13px;color:#52514e;margin:0 0 8px}",
             ".phase ul{margin:0;padding-left:18px;font-size:13px;color:#52514e}",
             ".overview{padding:20px 0;border-bottom:1px solid #e6e5e2}",
             ".overview img{max-width:100%;height:auto}",
             ".overview p{max-width:1100px;color:#52514e;font-size:14px;margin:0 0 10px}",
             "</style></head><body>",
             "<header><b>CODA timing evaluation on CHAMPS cases</b> — reference causes vs final "
             "prediction (gpt-oss:20b inference, faster-whisper medium transcription, gilda).",
             "<nav>" + " ".join(f"<a href='#c{k}'>#{k}</a>" for k in CASES) + "</nav></header>",
             "<main>",
             "<div class='overview'><h2>Accuracy vs spoken time (n=20)</h2>"
             "<p>Plots showing whether the CHAMPS reference group cause matches the top CODA-inferred "
             "cause (blue) or is found in the top 3 CODA-inferred causes (orange) based on "
             "voice input containing only a VA narrative (left) or the same VA narrative "
             "followed by a clinical narrative (right).</p>"
             "<p>Since each case's voice recording length differs, the range of time at which "
             "there is a VA narrative to clinical narrative switch over is shown as a gray "
             "shaded area. Once a case's recording ends, the final inferred causes are used "
             "when calculating the average accuracy across cases.</p>"
             "<img src='accuracy_over_time.png' alt='accuracy over time'></div>"]
    for k in CASES:
        refs = "  ".join(f"<b>{name}:</b> {', '.join(c)}" for name, c in ref_types(k))
        parts.append(f"<div class='case' id='c{k}'><h2>Case {k}</h2>"
                     f"<p class='refs'>{refs}</p>"
                     f"<div class='detail'>{phase_block('VA only', k, 'phase1_va')}"
                     f"{phase_block('VA + clinical', k, 'phase2_va_clinical')}</div></div>")
    parts.append("</main></body></html>")
    out = REPORT / "index.html"
    out.write_text("\n".join(parts))
    print(f"wrote {out}")


def main():
    REPORT.mkdir(exist_ok=True)
    write_table()
    write_index()


if __name__ == "__main__":
    main()
