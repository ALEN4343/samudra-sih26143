# Running SAMUDRA on another PC

Everything needed to run the dashboard and the full pipeline is in this folder.
The pre-built incidents are included, so nothing has to be regenerated and
nothing downloads at run time.

## Requirements

- Windows, macOS or Linux
- Python 3.11 or newer (3.13 is what this was built on)

Check with:

```bash
python --version
```

## Setup (once, ~3 minutes)

From inside this folder:

**Windows (PowerShell)**

```bash
python -m venv .venv; .venv\Scripts\python.exe -m pip install -r requirements-demo.txt; .venv\Scripts\python.exe -m pip install -e . --no-deps
```

**macOS / Linux**

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements-demo.txt && .venv/bin/python -m pip install -e . --no-deps
```

`requirements-demo.txt` deliberately leaves out `torch` and `torchvision`. They
are only used for training the segmenter, which is not part of the demo — the
pipeline falls back to the synthetic slick when no checkpoint is present, which
is the documented behaviour (CLAUDE.md build order step 11). Skipping them saves
a ~2.5 GB download.

To train the model as well, install the full set instead:

```bash
.venv\Scripts\python.exe -m pip install -e .
```

## Run the dashboard

**Windows**

```bash
.venv\Scripts\python.exe -m samudra.api --port 8077
```

**macOS / Linux**

```bash
.venv/bin/python -m samudra.api --port 8077
```

Then open <http://localhost:8077>. Pick an incident from the dropdown:

| Incident | Traffic | What it demonstrates |
|---|---|---|
| `demo-001` | synthetic, 24 vessels | the core attribution result — culprit ranked #1 |
| `demo-002` | synthetic, 46 vessels | a scenario never used during development |
| `houston-001` | **real** NOAA AIS, 426 vessels | behaviour against genuine traffic density |

## Re-run the pipeline from scratch (optional)

The incidents are already built. To regenerate one:

```bash
.venv\Scripts\python.exe -m samudra.synth.generate --scenario demo-001 --vessels 40 --decoys 7
```

```bash
.venv\Scripts\python.exe -m samudra.attribution --incident demo-001
```

```bash
.venv\Scripts\python.exe -m samudra.impact.forecast --incident demo-001
```

```bash
.venv\Scripts\python.exe -m samudra.evidence.report --incident demo-001
```

`scripts/run_demo.sh` chains all of these, but it needs bash — on Windows that
means Git Bash, not PowerShell.

## Run the tests

```bash
.venv\Scripts\python.exe -m pytest -q
```

All tests should pass. The attribution tests are the ones that matter: they
assert the planted culprit comes out ranked first.

## Notes

- `artifacts/audit.log` is a SHA-256 hash chain. It verifies *within* a run, not
  across a copy, so `--verify` may report a break on a fresh machine until the
  pipeline is re-run. That is expected, not corruption.
- No network calls happen during the demo. Everything replays from `artifacts/`.
- The Kaggle SAR training datasets (~1.5 GB) are **not** included — they are only
  needed to retrain the segmenter. Sources are documented in CLAUDE.md section 7.
