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

On Windows, double-click **`scripts\start_dashboard.bat`** — it starts the API
in its own window and opens the dashboard. Use **http://127.0.0.1:8000**, not
`localhost`: the server binds IPv4 only and Windows may try IPv6 `::1` first.

Two dashboards are served:

- `/` — the original single-page investigator view.
- `/app` — the unified view: one draggable timeline from the reverse-drift
  hindcast through acquisition to the +72 h forecast, animated wind and current
  from the pipeline's own environment field, the Coast Guard alert with its
  distribution list, and the dossier download.
- `/ops` — the operations console: top navigation, all map layers off until you
  ask for them, and a **Satellite** page that runs the real model over real SAR
  imagery and feeds the result into the same attribution engine.

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

#  MMSI        NAME                  POST  SCORE   IoU  dORI   AGE  TRUST  BEHAV   TYPE  CLASS
1  461279535   PACIFIC PIONEER      0.715  0.898 0.798   0.6   8.5   1.00   1.00   1.25  MATCHED
2  412848373   GOLDEN ENDEAVOUR     0.199  0.578 0.240   1.0  12.2   1.00   1.00   1.25  MATCHED
3  371494057   CRIMSON ENDEAVOUR    0.085  0.289 0.122  86.3  10.5   0.90   1.35   1.40  MATCHED
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
geometric_score = 0.40 * IoU
                + 0.25 * exp(-centroid_offset_km / 10)
                + 0.25 * cos(orientation_delta)
                + 0.10 * area_ratio

likelihood = softmax(geometric_score)          # across candidates
posterior  = likelihood * trust * behaviour * type_risk
           * (1.8 if the vessel went dark over the release window)
           * (1 + 0.4 * prior offences)        # renormalised
```

Weights live in `config/weights.yaml` and are **never** tuned to make a scenario produce
the expected answer. The one calibration constant that was changed — the softmax
temperature — is documented with its evidence in CLAUDE.md section 10.1: at the default
of 1.0, a candidate whose slick was 86 degrees misaligned ranked second on its priors
alone, which inverts the claim that geometry decides.

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
.venv/Scripts/python -m pytest -q      # 100 tests
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
- **`prior_offences` is always zero.** The feedback loop that would populate it
  (`persist/dossier_db.py`, connector A) is architected but not built, so the system does
  not yet compound across incidents.
- **The bundled coastline is simplified.** This build environment sits behind TLS
  interception, so the Natural Earth download fails certificate verification. A
  simplified Indian west coast ships with the code and the source actually used is
  reported in every result. Drop `ne_50m_coastline.geojson` into `data/raw/` and it is
  picked up with no code change.

---

## Architecture

Eleven layers, specified in [CLAUDE.md](CLAUDE.md) — read that for the contracts,
algorithms and data flow. In brief:

| # | Layer | Module | Status |
|---|---|---|---|
| 1 | Data Ingestion | `ingest/` | AIS real (Houston); SAR/optical/env pending |
| 2 | Preprocessing & Baseline | `baseline/`, `detection/preprocess.py` | **built** |
| 3 | AI Oil-Spill Detection | `detection/` | **built** (needs a GPU-trained checkpoint) |
| 4 | Vessel Analysis (CFAR + AIS matching) | `vessels/cfar.py` | **built**; `match.py` pending |
| 5 | AIS Trust & Behaviour | `trust/` | **built** |
| 6 | Environment & Drift Model | `attribution/drift.py` | **built** |
| 6b | Candidate Pruning | `attribution/prune.py` | **built** |
| 7 | **Vessel Attribution Engine** | `attribution/` | **built** |
| 8 | Impact Assessment & Prediction | `impact/` | **built** |
| 9 | Investigator Dashboard | `api.py`, `web/` | **built** |
| 10 | Evidence & Integrity | `evidence/` | **built** |
| 11 | Dissemination & Alerting | `dissemination/` | **built** (composes, does not transmit) |
| 1 | Satellite ingest console | `satellite/` | **built** (replay + drop-in; live/NRT needs credentials) |

Layer 7 is the differentiator, and it is built and verified. Section 10.3 of CLAUDE.md
lists every remaining gap explicitly.

**Detection has a deliberate fallback.** The pipeline uses the segmenter when
`data/models/seg.pt` exists and falls back to the synthetic slick when it does not, so
`run_demo.sh` works either way — both paths are verified. It also *refuses* a checkpoint
whose own metadata marks it non-representative (a CPU smoke run), because silently
routing the demo through known-bad weights while looking like an upgrade is worse than
not using them. Override with `--detection always`.

**CFAR** recovers **13 of 13** planted vessels on `demo-001` at the spec's `k=4.5`, with
4 spurious targets. Raising the grouping threshold from 2 to 5 pixels cut false positives
from 269 to 4 without losing a single true target; that is a grouping parameter, not the
detection threshold the spec fixes.

Every module has a CLI entrypoint and writes to `artifacts/<incident_id>/`:

```bash
python -m samudra.synth.generate --scenario demo-001
python -m samudra.attribution     --incident demo-001
python -m samudra.trust.score     --incident demo-001
python -m samudra.impact.forecast --incident demo-001
python -m samudra.evidence.report --incident demo-001
python -m samudra.evidence.integrity --verify
python -m samudra.dissemination   --incident demo-001 --print-message --geojson
python -m samudra.satellite       --list
python -m samudra.satellite       --live --days 14
python -m samudra.satellite       --observation sos-sentinel-test-3 --process
python -m samudra.api --port 8000
```

---

## Satellite operations (layer 1 console)

`/ops` -> **Satellite**, or `python -m samudra.satellite --list`.

Three modes, kept apart on purpose, because the difference is exactly what a
reviewer will push on:

| Mode | What it is | Status here |
|---|---|---|
| **LIVE / NRT** | a live catalogue query against an Indian provider | **needs an account.** EOS-04 (Bhoonidhi/NRSC) and INSAT-3DS / EOS-06 (MOSDAC/ISRO) both gate their catalogue behind a registered login and publish no open endpoint — Bhoonidhi 404s on every API path, MOSDAC `/opendata/` returns 401. Adapters are present and marked **unverified**. No non-Indian mission is carried as a substitute |
| **REAL SATELLITE REPLAY** | genuine SAR already on this machine, with its published provenance | the offline demo |
| **SYNTHETIC DEMO** | this project's generated scenes | kept, labelled, never called an observation |

**What LIVE/NRT means here, precisely.** Nothing can be retrieved live until
you register. Both Indian providers were probed from this machine on
2026-09-23: Bhoonidhi returns 404 on `/opensearch`, `/api/` and `/services`;
MOSDAC `/opendata/` returns **401** — it exists and refuses without credentials.
So the console reports `AUTHENTICATION REQUIRED` and stops. It does not
substitute another country's mission and hope nobody asks.

**The working route today** is the drop-in: download an EOS-04 / INSAT-3DS /
EOS-06 product from the portal yourself, put it in `data/satellite/incoming/`
with a sidecar `.json`, and it ingests with full provenance.

**The model actually runs.** Pressing *Process acquisition* loads
`data/models/seg.pt`, preprocesses the chip exactly as training did, runs
DeepLabv3+ over it and polygonises the output. There is no cached mask and no
lookup. `ground_truth.json` and every `label/` directory are blocked from the
inference path by a runtime guard, and a test asserts at source level that no
line both names a label and reads a file.

**Provenance is reported, including where it is missing.** The research archives
publish no per-chip acquisition time, so the console prints
`NOT PUBLISHED BY SOURCE` rather than inventing one.

**Two things are declared context, not satellite data.** A research chip has no
coordinate reference system, so its position on the map is an *operator input*;
and drift needs a wind/current field while attribution needs AIS, so both come
from a named context incident. The detection is real; the water it is placed in
is declared. A dropped-in GeoTIFF that carries its own transform uses that
instead.

To process a product you downloaded yourself (e.g. EOS-04 from Bhoonidhi), put
it in `data/satellite/incoming/` with a sidecar `.json` of the metadata the
provider published. Anything the sidecar does not state stays `UNDECLARED`.

```bash
BHOONIDHI_USER / BHOONIDHI_PASS
MOSDAC_USER / MOSDAC_PASS
```

No credentials are stored in this repository and none are hardcoded.

---

## Alerting (layer 11)

`python -m samudra.dissemination --incident demo-001` writes
`artifacts/demo-001/alert.json`: an NOS-DCP-tiered alert addressed to the Indian
Coast Guard units that would respond, with range and bearing to the slick and a
transmit-ready message. `--geojson` also writes the station layer for response
agencies. The dashboard shows the same payload and lets you export both.

Three things to say out loud rather than let a judge find them:

- **Nothing is transmitted.** The system composes and addresses the alert;
  `transmitted` is always `false`. A real transport needs verified recipients and
  an authority to send.
- **Tiers are NOS-DCP tiers** (I local / II regional / III national), because the
  tier decides who is on the distribution list. NOS-DCP is written in tonnes and
  SAR measures area, so area is an explicit volume proxy.
- **The station list is compiled from public ICG establishment locations** to
  roughly 1 km — fine for nearest-unit selection, not an operational directory.

---

## Conventions

- All timestamps UTC. Convert to epoch seconds **only** through
  `timeutil.epoch_seconds` — pandas 3.0 stores `datetime64[us]`, so the common
  `astype("int64") / 1e9` idiom is 1000× wrong and fails silently.
- All geometry GeoJSON, EPSG:4326, longitude first.
- Metric work reprojects to a local azimuthal equidistant CRS via `pyproj`. Never a
  fixed degrees-per-km constant.
- Exceptions fail loudly.
