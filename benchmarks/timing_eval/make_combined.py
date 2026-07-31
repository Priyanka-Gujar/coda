"""Build combined VA+clinical audio per real case.

Reads real_cases/cases.csv, and for every case that has both a "va" and a
"clinical" recording, concatenates them (VA first, then clinical) into
real_cases/combined/<case>_combined.wav (16kHz mono, Whisper's format) so a
single CODA run accumulates VA context before hearing the clinical segment.
"""
import csv
import subprocess
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
REAL_CASES = HERE / "real_cases"
COMBINED_DIR = REAL_CASES / "combined"


def load_cases():
    """case_id -> {"va": Path, "clinical": Path} from cases.csv."""
    cases = defaultdict(dict)
    with (REAL_CASES / "cases.csv").open() as fh:
        for row in csv.DictReader(fh):
            cases[row["case"]][row["type"]] = REAL_CASES / row["fname"]
    return cases


def concat_va_clinical(va, clinical, out_path):
    """Concatenate two audio files into one 16kHz mono WAV (VA then clinical)."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(va), "-i", str(clinical),
         "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1",
         "-ar", "16000", "-ac", "1", str(out_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def main():
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    for cid in sorted(cases, key=int):
        parts = cases[cid]
        if "va" not in parts or "clinical" not in parts:
            print(f"case {cid}: skipped (missing {'va' if 'va' not in parts else 'clinical'})")
            continue
        va, clinical = parts["va"], parts["clinical"]
        missing = [p for p in (va, clinical) if not p.exists()]
        if missing:
            print(f"case {cid}: skipped (file not found: {missing[0].name})")
            continue
        out_path = COMBINED_DIR / f"case{cid}_combined.wav"
        concat_va_clinical(va, clinical, out_path)
        print(f"case {cid}: {va.name} + {clinical.name} -> {out_path.name}")


if __name__ == "__main__":
    main()
