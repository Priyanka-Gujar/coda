# CODA timing evaluation

How long must someone talk before CODA reaches a confident cause-of-death call, and
how does accuracy build as a verbal-autopsy (VA) narrative and then clinical data are
heard? This folder holds two evaluations:

- **Real CHAMPS recordings** (`run_real_cases.py` and friends) - the main study.
- **Synthetic dry run** (`run_synthetic_dry_run.py`) - a no-audio plumbing check that
  synthesizes speech from `synthetic_dry_run_cases.json` via macOS `say`.

Only scripts and small metadata are version-controlled. Audio recordings and all
generated outputs are gitignored (see `.gitignore`).

## Pipeline components (all local)

- Transcription: faster-whisper, `medium` model.
- Grounding: gilda.
- Inference: CHAMPS LLM agent via Ollama. Default models are `qwen2.5:7b-instruct`
  (`run_real_cases.py`) and `gpt-oss:20b` (`run_inference_only.py`).

Requirements: `ffmpeg` on PATH, a running Ollama with the two models pulled, and the
project installed (or its `src/` on `PYTHONPATH`; the run scripts add it themselves).
Report scripts also need `matplotlib`, and `extract_true_labels.py` needs
`beautifulsoup4`.

## Data layout

`real_cases/` holds the inputs. Committed:

- `cases.csv` - index mapping each recording filename to a case number and type
  (`va` or `clinical`).
- `true_labels_group.json` - the CHAMPS reference group causes per case (underlying,
  immediate, morbid conditions), derived from the case-dossier HTML.

Not committed (obtain separately, then drop into `real_cases/`): the `*.m4a`
recordings, and the case-dossier HTML that `extract_true_labels.py` parses.

## Reproduce the real evaluation

From this folder, with the recordings and `cases.csv` in `real_cases/`:

```bash
# 1. Build combined VA+clinical audio per case (VA then clinical)
python make_combined.py

# 2. Full pipeline with qwen2.5:7b (transcribe, ground, infer), VA then combined,
#    whole-file and 20s-chunked modes -> real_cases_results/
python run_real_cases.py

# 3. Replay the saved transcripts through gpt-oss:20b inference only (no
#    re-transcription) -> real_cases_results_gptoss/
python run_inference_only.py

# 4. Ground-truth labels: true_labels_group.json is committed. To regenerate it
#    (needs the dossier HTML in real_cases/):
python extract_true_labels.py

# 5. Report -> real_cases_report/
python make_accuracy_curve.py     # accuracy_over_time.png
python make_final_report.py       # index.html + final_predictions.csv
```

Both run scripts are resumable: a completed run writes `inference.json`, and re-running
skips it unless `--force` is given. Useful flags: `--only <ids>`, `--phases`, `--modes`.

Open `real_cases_report/index.html` for the per-case results and the accuracy-over-time
figure.

## Synthetic dry run (no recordings needed)

```bash
# fast plumbing check, no LLM/Ollama needed
python run_synthetic_dry_run.py --whisper-models tiny --llm-models toy
```
