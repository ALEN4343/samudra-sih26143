# SAMUDRA

**S**AR **A**ttribution of **M**arine **U**nauthorised **D**ischarge, **R**anking and **A**ssessment.

*Smart India Hackathon — SIH26143 (NTRO): satellite imagery + AIS correlation for oil-spill attribution.*

Detecting an oil slick from radar is a solved problem. Naming the ship that spilled it,
with a number attached to the claim, is not. SAMUDRA does the second thing.

Given a SAR scene and AIS traffic, it reverse-drifts the observed slick to an origin
envelope, prunes the vessel field to those that could physically have been there,
forward-simulates a release hypothesis for every survivor, ranks them, forecasts where
the slick goes next, and writes a hash-chained evidence dossier.

**Full project report:** [SAMUDRA_Project_Report.pdf](SAMUDRA_Project_Report.pdf) — aim,
architecture, algorithms, data, results, limitations and how to run it.

---

## Quick start (Windows)

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps
.venv\Scripts\python.exe -m samudra.api --port 8000
```

Then open **<http://127.0.0.1:8000/investigate>**. Use `127.0.0.1`, not `localhost` — the
server binds IPv4 only and Windows may try IPv6 first. Or double-click
`scripts\start_dashboard.bat`, which starts the server and opens a browser.

The pre-built incidents ship with the repo, so nothing is regenerated or downloaded at run
time. `requirements-demo.txt` leaves out `torch`; install the full set with
`pip install -e .` to run detection or training. See
[RUN_ON_ANOTHER_PC.md](RUN_ON_ANOTHER_PC.md) for macOS / Linux.

### The four views

| Route | What it is |
|---|---|
| **`/investigate`** | **Investigator view — start here.** Result first: the potential source vessel and why, then the evidence, then the technical detail. |
| `/` | The original single-page investigator view. |
| `/app` | Unified timeline: reverse-drift hindcast → acquisition → +72 h forecast, animated wind and current, Coast Guard alert. |
| `/ops` | Operations console: runs the real model over real SAR (Satellite tab) and feeds it into the same attribution engine. |

Open a specific case with `?case=`, e.g. `/investigate?case=kutch-exercise-001`.

---

## The investigator view (`/investigate`)

Built as a redesigned copy of `/` — the original page is untouched. It computes no score,
drift or forecast of its own: every number comes from the API, which serves the pipeline's
artifacts.

- **Case summary and pipeline row** — slick area, age, candidates, conditions and data
  availability, then *Detect → Characterise → Trace → Filter → Attribute → Forecast*. Each
  step's state is derived from real outputs, and clicking a step jumps to that part of the page.
- **Potential source vessel** — the attribution score as a percentage with a plain-language
  explanation, the margin over the next candidate, and key evidence with strength marks.
- **Map** — orange slick, yellow origin envelope, the selected vessel's track in red (bold only
  from two hours before release to acquisition, with direction arrows and a labelled release
  point), faint filtered traffic, green forecast. The on-map legend doubles as layer toggles;
  basemap switcher, lat/lon grid and an offline coastline.
- **Simulate discharge** — re-runs the existing drift model from the candidate's estimated
  release and overlays the simulated slick scored during attribution on the observed one.
- **Timeline** — hindcast → acquisition → +24/48/72 h. Forecasts snap to the computed
  horizons and are never interpolated.
- **Compare candidates**, **Why were they filtered?** (only the rules `prune.py` applies), and
  collapsible technical details down to the full scoring formula with real values.
- **Satellite evidence** — the stored scene with every detected polygon; where no scene
  exists it says so and substitutes nothing.
- Responsive: desktop sidebar, tablet drawer, stacked phone layout.

Language is deliberate throughout: *potential source vessel* and *attribution score*, never
"culprit". Model output is decision-support evidence and does not establish legal
responsibility.

---

## What it produces

`demo-001`, a synthetic scenario with a planted culprit, run end to end:

```
[1/7] generate scenario          47 (incl. 7 decoys near origin) tracks
[2/7] detect slick               3 polygon(s) from the segmenter; checkpoint: yes
[3/7] attribute                  in scene 24  ->  in envelope 3  ->  scored 3  ->  ranked 3
[4/7] trust and behaviour        7 of 47 vessels flagged;  identity mismatch: [419254301]
[5/7] forecast and impact        none within 72 h
[6/7] evidence dossier           artifacts/demo-001/dossier_demo-001.pdf
[7/7] chain of custody           INTACT
```

| # | MMSI | Vessel | Score | Match | IoU | Course Δ | Age | Trust | Behav | Type |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 461279535 | PACIFIC PIONEER | **70.6%** | 0.892 | 0.802 | 0.4° | 8.5 h | 1.00 | 1.00 | 1.25 |
| 2 | 412848373 | GOLDEN ENDEAVOUR | 20.6% | 0.584 | 0.239 | 0.8° | 12.3 h | 1.00 | 1.00 | 1.25 |
| 3 | 371494057 | CRIMSON ENDEAVOUR | 8.7% | 0.292 | 0.122 | 86.5° | 10.5 h | 0.90 | 1.35 | 1.40 |

Crimson Endeavour went silent on AIS for 69 minutes inside the area, which raises its
behaviour prior — but its simulated slick would have been 86° off the observed one, so it
still ranks last. Priors modulate; geometry decides.

**Against planted ground truth** (measured when each scenario was built):

| | demo-001 | demo-002 (never developed against) | kutch-exercise-001 |
|---|---|---|---|
| Culprit ranked #1 | yes | yes | yes |
| Release time error | 0.02 h | 0.12 h | — |
| Release position error | 0.98 km | 2.34 km | — |
| Vessels narrowed | 24 → 3 | 46 → 3 | 164 → 3 |
| Planted anomalies found | 7 / 7 | 7 / 7 | — |

On real Houston AIS (560 ships) the culprit ranks 9th of 12: in a single-lane channel,
several tugs run the identical line within the hour, so their slicks are geometrically
indistinguishable. The engine still narrows 426 vessels to a shortlist of 12 that contains
the culprit. That is reported, not hidden.

---

## How it decides

Each candidate's own AIS track is used to simulate what a discharge would have looked like,
sampled every 30 minutes across the plausible window, and the simulated slick is scored
against the observed one:

```
geometric_score = 0.40 * IoU
                + 0.25 * exp(-centroid_offset_km / 10)
                + 0.25 * cos(orientation_delta)
                + 0.10 * area_ratio

likelihood = softmax(geometric_score / 0.25)   # across candidates
posterior  = likelihood * trust * behaviour * proximity * type_risk
           * (1.8 if the vessel went dark over the release window)
           * (1 + 0.4 * prior offences)        # renormalised
```

Weights live in `config/weights.yaml` (and are served to the UI by `/api/config/scoring`, so
the explanation can never drift from the model). They are **never** tuned to make a scenario
produce the expected answer. The one calibration constant that was changed — the softmax
temperature — is documented with its evidence in CLAUDE.md section 10.1.

A vessel discharging while under way is a *line* source, which is why a real slick is
elongated along the ship's course. That is what makes the orientation term carry real
information.

---

## Verifying it rather than trusting it

The synthetic generator plants a culprit, a release time and a release position, and writes
them to `ground_truth.json`. **No pipeline module may read that file** — a test enforces it
structurally by scanning the source for reads.

```bash
.venv\Scripts\python.exe -m pytest -q -m "not live"
```

The tests in `tests/test_satellite.py` that exercise the research archives need the Deep-SAR
SOS and CSIRO training chips under `data/raw/oilspill/`; they fail on a checkout without them.

The tests that matter:

- `test_drift.py` — round-trips the planted release through an *independently implemented*
  drift model and requires IoU > 0.5 against the observed slick (currently 0.86).
- `test_attribution.py` — the planted culprit must rank first, and the funnel must actually
  narrow.
- `test_scenarios.py` — the same assertions over *every* scenario.
- `test_evidence.py` — altering an artifact, rewriting the audit log, or deleting a file must
  all be detected.
- `test_investigate_ui.py` — the investigator view never hardcodes scoring weights, never
  uses guilt language, never interpolates forecasts, never invents imagery; and its three API
  endpoints behave as documented.

---

## What is real and what is synthetic

| Data | Source | Status |
|---|---|---|
| Operational SAR | ISRO **EOS-04** L2B ARD (NRSC Bhoonidhi), Gulf of Kutch, 2026-07-05 01:17:53 UTC | **Real** — calibration and timezone verified against the product itself |
| Environmental context | MOSDAC Data Download API — INSAT-3DS SST, EOS-06 wind and ocean colour | **Real** — catalogue search verified live |
| AIS | NOAA MarineCadastre, Houston Ship Channel | **Real** |
| Detector training data | Deep-SAR SOS + CSIRO Sentinel-1 (DOI 10.25919/4v55-dn16) | **Real, foreign** — no labelled Indian SAR oil-spill set exists |
| Demo scenes, wind/current fields | `samudra.synth` | **Synthetic**, labelled everywhere |
| Ground truth for verification | planted culprit, release, spoofing, loitering, gaps | **Synthetic by design** |

The problem statement permits synthetic data to demonstrate the algorithm, and the claim here
is specific: **the attribution engine works and is verifiable against planted ground truth.**
Synthetic cases are labelled *Simulation mode* in the UI and *SYNTHETIC* in the dossier.

Honest caveats, stated rather than buried:

- **SAR cannot confirm oil.** Every detection is an oil-like surface anomaly.
- **The Fay age cross-check is not independent** — it needs an assumed discharge volume, so
  it is a consistency check, and the UI and dossier say so.
- **`prior_offences` is always zero.** The feedback loop that would populate it (connector A)
  is architected but not built.
- **The real EOS-04 scene is not attributed** — no AIS for it exists on this machine, and
  widening the envelope until some vessel falls inside is how innocent ships get accused.
- **The optical (EO) branch is designed, not built.** Detection is SAR-only.

---

## Architecture

Eleven layers, specified in [CLAUDE.md](CLAUDE.md) — read that for the contracts, algorithms
and data flow.

| # | Layer | Module | Status |
|---|---|---|---|
| 1 | Data ingestion | `ingest/`, `satellite/`, `monitor/` | **built** — real AIS, real EOS-04, MOSDAC search; drop-in folder for downloaded products |
| 2 | Preprocessing & baseline | `detection/preprocess.py`, `baseline/` | **built** — baseline runs in global fallback until 15+ scenes per cell |
| 3 | AI oil-spill detection | `detection/` | **built** — DeepLabv3+ ResNet-50, oil IoU 0.742 on the held-out test split |
| 4 | Vessel analysis (CFAR) | `vessels/cfar.py` | **built**; SAR↔AIS matching pending |
| 5 | AIS trust & behaviour | `trust/` | **built** |
| 6 | Environment & drift model | `attribution/drift.py` | **built** |
| 6b | Candidate pruning | `attribution/prune.py` | **built** |
| 7 | **Vessel attribution engine** | `attribution/` | **built and verified** |
| 8 | Impact & forecast | `impact/` | **built** — Natural Earth 1:10 m coastline, shoreline ETA |
| 9 | Investigator dashboards | `api.py`, `web/` | **built** — four views |
| 10 | Evidence & integrity | `evidence/` | **built** |
| 11 | Dissemination & alerting | `dissemination/` | **built** — composes, never transmits |

**Detection has a deliberate fallback.** The pipeline uses the segmenter when
`data/models/seg.pt` exists and falls back to the synthetic slick when it does not, so
`run_demo.sh` works either way. It also *refuses* a checkpoint whose own metadata marks it
non-representative. Override with `--detection always`.

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
python -m samudra.api --port 8000
```

Rebuild a whole case in one command (Git Bash):

```bash
bash scripts/run_demo.sh demo-001 --regen --no-serve
bash scripts/run_demo.sh demo-002 --regen --seed 77341 --vessels 80 --decoys 9 --wind 7.0 9.0
```

---

## Satellite operations (layer 1)

`/ops` → **Satellite**, or `python -m samudra.satellite --list`. Three modes, kept apart on
purpose:

| Mode | What it is | Status here |
|---|---|---|
| **LIVE / NRT** | live catalogue queries against Indian providers only | **MOSDAC search works without an account** (INSAT-3DS SST, EOS-06 wind and ocean colour — verified over the Gulf of Kutch); **download needs a registered account**. Bhoonidhi (EOS-04) publishes no API — 404 on `/opensearch`, `/api/`, `/services`. No foreign mission is carried as a substitute. |
| **REAL SATELLITE REPLAY** | genuine SAR already on this machine, with published provenance | the offline demo |
| **SYNTHETIC DEMO** | this project's generated scenes | kept, labelled, never called an observation |

The working route for new EOS-04 data is the **drop-in folder**: download a product from the
portal, put it in `data/satellite/incoming/` with a sidecar `.json`, and it ingests with full
provenance. `acquired_at` is `None` whenever a source does not publish an acquisition time —
never invented — and no label source is ever on the inference path.

```bash
BHOONIDHI_USER / BHOONIDHI_PASS
MOSDAC_USER / MOSDAC_PASS
```

No credentials are stored in this repository and none are hardcoded
(`python scripts/check_credentials.py` checks them without printing anything secret).

---

## Alerting (layer 11)

`python -m samudra.dissemination --incident demo-001` writes an NOS-DCP-tiered alert
(I local / II regional / III national) addressed to the Indian Coast Guard units that would
respond, with range and bearing to the slick. The dashboards show the same payload and export
it as JSON and GeoJSON.

- **Nothing is transmitted.** `transmitted` is always `false`.
- NOS-DCP is written in tonnes and SAR measures area, so area is an explicit volume proxy.
- The station list is compiled from public ICG establishment locations to roughly 1 km.

---

## API

All endpoints read `artifacts/<incident_id>/`; none runs analysis of its own except `/replay`,
which re-runs the existing forward drift for display.

| Endpoint | Returns |
|---|---|
| `GET /api/incidents` | analysed incidents |
| `GET /api/incident/{id}` | the incident record plus forecast polygons |
| `GET /api/incident/{id}/tracks` | AIS tracks as GeoJSON with per-vertex timestamps and funnel role |
| `GET /api/incident/{id}/replay?mmsi=` | forward drift replay of a ranked candidate's best hypothesis (default: rank 1) |
| `GET /api/incident/{id}/env` | the wind and current field, decimated |
| `GET /api/incident/{id}/scene` · `/scene.png` | stored EPSG:4326 scene raster and every detected polygon |
| `GET /api/incident/{id}/alert` | composed Coast Guard alert |
| `GET /api/incident/{id}/dossier` | evidence dossier PDF, regenerated on request |
| `GET /api/config/scoring` | scoring weights from `config/weights.yaml` |
| `GET /api/audit` | hash-chain verification |
| `GET /api/satellite/…` | satellite sources, model status, inference, processing |

---

## Conventions

- All timestamps UTC. Convert to epoch seconds **only** through `timeutil.epoch_seconds` —
  pandas 3.0 stores `datetime64[us]`, so the common `astype("int64") / 1e9` idiom is 1000×
  wrong and fails silently.
- All geometry GeoJSON, EPSG:4326, longitude first. Metric work reprojects to a local
  azimuthal equidistant CRS via `pyproj`, never a fixed degrees-per-km constant.
- Wind and current directions in `env_summary` are the direction the flow goes **toward**.
- Exceptions fail loudly.
