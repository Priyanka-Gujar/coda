"""Turn a synthetic case into a mock VA interview and render it as two-voice audio.

An LLM writes a realistic interviewer<->caregiver dialogue whose every factual
statement is grounded in the case (VA narrative + clinical info). Each turn is
synthesized with macOS `say` (a distinct voice per speaker) and concatenated into
one 16kHz WAV - the same format CODA's CLI ingests.

Usage
-----
    python benchmarks/timing_eval/generate_interview.py --case case_01
    python benchmarks/timing_eval/generate_interview.py --case case_03 \
        --provider openai --model gpt-5.4-mini
"""
import argparse
import json
import subprocess
from pathlib import Path

from coda.llm_api import create_llm_client

HERE = Path(__file__).parent
INTERVIEWER_VOICE = "Daniel"    # trained VA interviewer
INTERVIEWEE_VOICE = "Samantha"  # family caregiver

DIALOGUE_SCHEMA = {
    "type": "object",
    "properties": {
        "dialogue": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string",
                                "enum": ["interviewer", "interviewee"]},
                    "text": {"type": "string"},
                },
                "required": ["speaker", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["dialogue"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You write realistic verbal autopsy (VA) mock interviews for testing a \
cause-of-death tool. Produce a natural spoken dialogue between two people:

- interviewer: a trained VA fieldworker.
- interviewee: a close family caregiver of the deceased.

Hard rules:
- EVERY factual claim the caregiver makes must be grounded in the provided CASE FACTS \
(symptoms, timeline, care-seeking, ages, test results). Do NOT invent any symptom, \
diagnosis, date, medication, or lab result that is not in the facts.
- The caregiver speaks in plain lay language, recounting what they observed and what \
clinicians told them. Clinical/laboratory findings must be voiced as things the family \
was TOLD by health workers ("the doctors later said..."), never as if the caregiver ran \
tests themselves.
- The interviewer asks open then follow-up questions in a natural VA style: greeting and \
brief purpose, illness onset and progression, specific symptoms, timeline, care-seeking, \
final hours, and anything the clinicians reported. It is fine for the caregiver to express \
uncertainty ("I'm not sure", "I don't remember exactly").
- Keep it conversational and realistic: roughly 14-22 turns total, short turns. Do not \
summarize at the end; just end naturally.
"""


def generate_dialogue(case: dict, provider: str, model: str):
    client = create_llm_client(provider=provider, model=model)
    user_prompt = (
        f"CASE FACTS for case {case['case_id']} ({case['topic']}):\n\n"
        f"VA NARRATIVE:\n{case['va_narrative']}\n\n"
        f"CLINICAL INFO:\n{case['clinical_narrative']}\n\n"
        "Write the mock VA interview dialogue grounded strictly in these facts."
    )
    resp = client.call_with_schema(
        system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt,
        schema=DIALOGUE_SCHEMA, schema_name="va_mock_interview", temperature=0.4,
    )
    if resp.get("api_failed"):
        raise SystemExit("LLM call failed.")
    return resp["dialogue"]


def say_to_wav(text: str, voice: str, out_path: Path):
    aiff = out_path.with_suffix(".aiff")
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(aiff), "-ar", "16000", "-ac", "1",
                    str(out_path)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    aiff.unlink(missing_ok=True)


def render_audio(dialogue, out_dir: Path) -> Path:
    """Synthesize each turn with its speaker's voice, concatenate to one WAV."""
    turns_dir = out_dir / "turns"
    turns_dir.mkdir(parents=True, exist_ok=True)
    seg_paths = []
    for i, turn in enumerate(dialogue):
        voice = INTERVIEWER_VOICE if turn["speaker"] == "interviewer" else INTERVIEWEE_VOICE
        seg = turns_dir / f"{i:03d}_{turn['speaker']}.wav"
        say_to_wav(turn["text"], voice, seg)
        seg_paths.append(seg)

    combined = out_dir / "interview.wav"
    inputs = []
    for p in seg_paths:
        inputs += ["-i", str(p)]
    filt = "".join(f"[{i}:a]" for i in range(len(seg_paths)))
    filt += f"concat=n={len(seg_paths)}:v=0:a=1[out]"
    subprocess.run(["ffmpeg", "-y", *inputs, "-filter_complex", filt,
                    "-map", "[out]", str(combined)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default="case_01", help="case_id from cases.json")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--cases", default=str(HERE / "cases.json"))
    parser.add_argument("--no-audio", action="store_true", help="dialogue only")
    args = parser.parse_args()

    cases = {c["case_id"]: c for c in json.loads(Path(args.cases).read_text())}
    if args.case not in cases:
        raise SystemExit(f"Unknown case {args.case}. Have: {', '.join(cases)}")
    case = cases[args.case]

    print(f"Generating mock interview for {args.case} via {args.provider}:{args.model}...")
    dialogue = generate_dialogue(case, args.provider, args.model)

    out_dir = HERE / "interviews" / args.case
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "interview.json").write_text(json.dumps(dialogue, indent=2, ensure_ascii=False))
    transcript = "\n".join(
        f"{'INTERVIEWER' if t['speaker'] == 'interviewer' else 'CAREGIVER  '}: {t['text']}"
        for t in dialogue)
    (out_dir / "interview.txt").write_text(transcript + "\n")

    print(f"\n{len(dialogue)} turns:\n")
    print(transcript)

    if not args.no_audio:
        print("\nRendering two-voice audio...")
        wav = render_audio(dialogue, out_dir)
        import wave
        with wave.open(str(wav)) as w:
            dur = w.getnframes() / w.getframerate()
        print(f"Wrote {wav} ({dur:.1f}s)")
    print(f"\nDone. Output in {out_dir}/")


if __name__ == "__main__":
    main()
