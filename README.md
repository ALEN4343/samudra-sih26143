# SAMUDRA

**S**AR **A**ttribution of **M**arine **U**nauthorised **D**ischarge, **R**anking and **A**ssessment.

Detecting an oil slick from radar is a solved problem. Naming the ship that spilled it,
with a number attached to the claim, is not. SAMUDRA does the second thing.

Given a SAR scene and AIS traffic, it reverse-drifts the observed slick to an origin
envelope, prunes the vessel field to those that could physically have been there,
forward-simulates a release hypothesis for every survivor, and ranks them into a
hash-chained evidence dossier.

---

## Run the demo

```bash
python -m venv .venv && .venv/Scripts/pip install -e .
bash scripts/run_demo.sh demo-001
```

That generates the scenario, runs the full pipeline, writes the dossier, verifies the
audit chain, and serves the dashboard at <http://127.0.0.1:8000>.

For the second scenario — a different seed, culprit, traffic density and wind band that
the engine was never developed against:

```bash
bash scripts/run_demo.sh demo-002 --regen --seed 77341 --vessels 80 --decoys 9 --wind 7.0 9.0
```

---

## What it produces

```
[1/6] generate scenario          47  (incl. 7 decoys near origin) tracks, slick 89.77 km2
[2/6] attribute                  in scene 24  ->  in envelope 3  ->  scored 3  ->  ranked 3
[3/6] trust and behaviour        7 of 47 vessels flagged;  identity mismatch: [419254301]
[4/6] forecast and impact        none within 72 h; closest approach 295.2 km at +72h
[5/6] evidence dossier           artifacts/demo-001/dossier_demo-001.pdf
[6/6] chain of custody           INTACT

#  MMSI        NAME                  POST  SCORE   IoU  dORI   AGE  TRUST  BEHAV  CLASS
1  461279535   PACIFIC PIONEER      0.998  0.896 0.798   0.6   8.5   1.00   1.00  MATCHED
2  412848373   GOLDEN ENDEAVOUR     0.002  0.576 0.240   1.0  12.2   1.00   1.00  MATCHED
3  371494057   CRIMSON ENDEAVOUR    0.000  0.283 0.122  86.3  10.5   0.90   1.35  MATCHED
```

Against planted ground truth, across both scenarios:

| | demo-001 | demo-002 |
|---|---|---|
| Culprit ranked #1 | yes | yes |
| Release time error | 0.02 h | 0.12 h |
| Release position error | 0.98 km | 2.34 km |
| Slick age error | 0.02 h | 0.12 h |
| Planted anomalies found | 7 / 7 | 7 / 7 |
| False positives | 0 / 47 | 0 / 89 |

---

## How it decides

Each candidate's own AIS track is used to simulate what a discharge would have looked
like, sampled every 30 minutes across the plausible window. The simulated slick is
scored against the observed one:

```
score = 0.40 * IoU
      + 0.25 * centroid proximity
      + 0.25 * orientation agreement
      + 0.10 * area ratio
```

Weights live in `config/weights.yaml` and are **never** tuned to make a scenario produce
the expected answer. Posteriors are a softmax over score multiplied by trust, behaviour
and proximity priors.

A vessel discharging while under way is a *line* source, which is why a real slick is
elongated along the ship's course. That is what makes the orientation term carry real
information: in the table above it is the term that separates rank 2 from rank 3.

The **funnel** is the argument, not a log line. 24 vessels in the scene became 3 in the
origin envelope became 1 named ship.

---

## Verifying it rather than trusting it

The synthetic generator plants a culprit, a release time and a release position, and
writes them to `ground_truth.json`. **No pipeline module may read that file** — a test
enforces it structurally by scanning the source for reads.

```bash
.venv/Scripts/python -m pytest -q      # 58 tests
```

The tests that matter:

- `test_drift.py` — round-trips the planted release through an *independently
  implemented* drift model and requires IoU > 0.5 against the observed slick. Currently
  0.86. Near-zero means a coordinate bug, which nothing downstream would notice.
- `test_attribution.py` — the planted culprit must rank first, and the funnel must
  actually narrow. If only one candidate survives pruning, ranking never had to
  discriminate and a correct answer proves nothing.
- `test_scenarios.py` — the same assertions over *every* scenario. Passing on demo-001
  alone is weak evidence; demo-002 is the generalisation test.
- `test_evidence.py` — altering an artifact, rewriting the audit log, or deleting a file
  must all be detected.

---

## What is real and what is synthetic

The dashboard and every page of the dossier are labelled `SYNTHETIC` when the scenario
is generated. The problem statement permits synthetic data to demonstrate the algorithm,
and the claim here is specific: **the attribution engine works and is verifiable against
planted ground truth**. That claim is stronger when the boundary is stated plainly.

Two honest caveats worth stating out loud rather than burying:

- **The Fay age cross-check is not independent.** It needs a reference-area constant
  encoding an assumed discharge volume and oil type. It is a consistency check against
  an assumed spreading rate, and the dossier says so.
- **The bundled coastline is simplified.** This build environment sits behind TLS
  interception, so the Natural Earth download fails certificate verification. A
  simplified Indian west coast ships with the code and the source actually used is
  reported in every result. Drop `ne_50m_coastline.geojson` into `data/raw/` and it is
  picked up with no code change.

---

## Architecture

Eleven layers, specified in [CLAUDE.md](CLAUDE.md) — read that for the contracts,
algorithms and data flow. In brief:

| # | Layer | Module |
|---|---|---|
| 1 | Ingestion & normalisation | `ingest/` |
| 2 | Detection — slick segmentation | `detection/` |
| 3 | Baseline & anomaly scoring | `baseline/` |
| 4 | Vessel detection — CFAR | `vessels/` |
| 5 | Trust, identity & behaviour | `trust/` |
| 6 | Candidate pruning | `attribution/prune.py` |
| 7 | **Drift & attribution** | `attribution/` |
| 8 | Impact & forecast | `impact/` |
| 9 | Evidence & chain of custody | `evidence/` |
| 10 | API & dashboard | `api.py`, `web/` |
| 11 | Orchestration & reproducibility | `scripts/`, `synth/` |

Layer 7 is the differentiator. Layers 2–4 currently run on the synthetic scene; the
real-data path for them is described in CLAUDE.md sections 4.1–4.3.

Every module has a CLI entrypoint and writes to `artifacts/<incident_id>/`:

```bash
python -m samudra.synth.generate --scenario demo-001
python -m samudra.attribution     --incident demo-001
python -m samudra.trust.score     --incident demo-001
python -m samudra.impact.forecast --incident demo-001
python -m samudra.evidence.report --incident demo-001
python -m samudra.evidence.integrity --verify
python -m samudra.api --port 8000
```

---

## Conventions

- All timestamps UTC. Convert to epoch seconds **only** through
  `timeutil.epoch_seconds` — pandas 3.0 stores `datetime64[us]`, so the common
  `astype("int64") / 1e9` idiom is 1000× wrong and fails silently.
- All geometry GeoJSON, EPSG:4326, longitude first.
- Metric work reprojects to a local azimuthal equidistant CRS via `pyproj`. Never a
  fixed degrees-per-km constant.
- Exceptions fail loudly.
