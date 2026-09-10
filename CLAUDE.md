# SAMUDRA — Architecture

**S**AR **A**ttribution of **M**arine **U**nauthorised **D**ischarge, **R**anking and **A**ssessment.

A maritime oil-spill attribution system. It takes a SAR scene and AIS traffic, detects
the slick, reverse-drifts it to an origin envelope, prunes the vessel field to those that
could physically have been there, forward-simulates a release hypothesis for each
survivor, and ranks them into a defensible, hash-chained evidence dossier.

The differentiator is **layer 7**. Detecting oil is a solved problem. Naming the ship
that spilled it, with a number attached to the claim, is not.

---

## 1. The 11 layers

| # | Layer | Section | Module |
|---|---|---|---|
| 1 | Ingestion & Normalisation | — | `ingest/` |
| 2 | Detection — Slick Segmentation | §4.1 | `detection/` |
| 3 | Baseline & Anomaly Scoring | §4.2 | `baseline/` |
| 4 | Vessel Detection — CFAR | §4.3 | `vessels/` |
| 5 | Trust, Identity & Behaviour | §4.4 | `trust/` |
| 6 | Candidate Pruning | §4.5.3 | `attribution/prune.py` |
| 7 | Drift & Attribution | §4.5 | `attribution/` |
| 8 | Impact & Forecast | — | `impact/` |
| 9 | Evidence & Chain of Custody | §4.6 | `evidence/` |
| 10 | API & Dashboard | — | `api.py`, `web/` |
| 11 | Orchestration & Reproducibility | — | `scripts/`, `synth/` |

One line each, in order:

1. **Ingestion & Normalisation** — read SAR GeoTIFFs and AIS point feeds, normalise to the contracts in §3, write `artifacts/<incident_id>/`.
2. **Detection** — land mask, wind gate, tiled CNN segmentation, polygonisation, confidence fusion.
3. **Baseline & Anomaly** — per-cell H3 backscatter statistics conditioned on wind, z-scored against history.
4. **Vessel Detection** — two-parameter CFAR over the scene to find radar targets, including ones with no AIS.
5. **Trust, Identity & Behaviour** — AIS integrity checks and behavioural flags; classify each vessel MATCHED / DARK / PHANTOM / IDENTITY_MISMATCH.
6. **Candidate Pruning** — reverse-advect the slick to an origin envelope, keep only vessels intersecting it in space *and* time.
7. **Drift & Attribution** — sample release hypotheses along each surviving track, forward-simulate, score against the observed slick, rank into posteriors.
8. **Impact & Forecast** — forward-forecast the slick to +24/48/72h with an uncertainty cone; compute coastline intersection and ETA.
9. **Evidence & Chain of Custody** — PDF dossier plus a SHA-256 hash-chained audit log over all inputs and outputs.
10. **API & Dashboard** — FastAPI over the artifacts directory; single-file Leaflet frontend with the funnel, ranked suspects and drift replay.
11. **Orchestration & Reproducibility** — synthetic ground-truth generator and one-command end-to-end runner.

### Data flow

```
SAR scene ──┐
            ├─► [2] detect ──► slick polygon ──┐
AIS feed ───┤        ▲                         │
            │        │                         ▼
            │   [3] anomaly              [6] reverse-drift
            │                                  │
            ├─► [4] CFAR ──► radar targets     ▼
            │        │                   origin envelope
            │        ▼                         │
            └─► [5] trust/behaviour ──────► [7] prune + hypothesise + rank
                                               │
                                    ┌──────────┼──────────┐
                                    ▼          ▼          ▼
                                 [8] forecast [9] dossier [10] dashboard
```

---

## 2. Project structure

```
samudra/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── config/
│   └── weights.yaml
├── data/
│   ├── raw/
│   │   ├── ais/
│   │   └── oilspill/
│   │       ├── sos/
│   │       └── binary/
│   └── models/
│       └── seg.pt
├── artifacts/
│   ├── audit.log
│   └── <incident_id>/
├── notebooks/
├── scripts/
│   ├── run_demo.sh
│   └── train_segmenter.py
├── src/samudra/
│   ├── __init__.py
│   ├── contracts.py
│   ├── api.py
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── ais.py
│   │   └── scene.py
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── preprocess.py
│   │   ├── segmenter.py
│   │   ├── polygonize.py
│   │   └── fuse.py
│   ├── baseline/
│   │   ├── __init__.py
│   │   └── anomaly.py
│   ├── vessels/
│   │   ├── __init__.py
│   │   └── cfar.py
│   ├── trust/
│   │   ├── __init__.py
│   │   ├── score.py
│   │   └── behaviour.py
│   ├── attribution/
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── drift.py
│   │   ├── prune.py
│   │   ├── hypothesis.py
│   │   └── rank.py
│   ├── impact/
│   │   ├── __init__.py
│   │   ├── forecast.py
│   │   └── coastline.py
│   ├── evidence/
│   │   ├── __init__.py
│   │   ├── report.py
│   │   └── integrity.py
│   └── synth/
│       ├── __init__.py
│       └── generate.py
├── tests/
└── web/
    └── index.html
```

### Conventions

- All timestamps UTC, timezone-aware.
- All geometry GeoJSON, EPSG:4326, **longitude first**.
- Every module has a CLI entrypoint and writes to `artifacts/<incident_id>/`.
- Metric computation reprojects to a local azimuthal equidistant CRS via `pyproj`.
  Never approximate with a fixed degrees-per-km constant.
- Exceptions fail loudly. Never silently swallowed.
- `contracts.py` is the single source of truth and changes only by explicit request.

---

## 3. Contracts

All models are pydantic v2. `src/samudra/contracts.py` is authoritative; this section is
the specification it implements.

### Vessel and AIS

```
AisTrack
  mmsi: int
  points: list[AisPoint]              # see note
  vessel_name: str | None
  imo: int | None
  vessel_type: str | None
  length_m: float | None
  flag: str | None
  gaps: list[AisGap]
```

> `AisPoint` is referenced here but defined in `contracts.py` as:
> `mmsi: int`, `t: datetime`, `lat: float`, `lon: float`, `sog: float`,
> `cog: float | None`, `heading: float | None`.

```
AisGap
  start_t: datetime
  end_t: datetime
  duration_min: float
  start_lat, start_lon, end_lat, end_lon: float

VesselDetection                        # radar target from CFAR, layer 4
  det_id: str
  lat, lon: float
  length_m: float | None
  peak_intensity: float
  matched_mmsi: int | None
```

### Detection

```
SlickPolygon
  slick_id: str
  geometry: dict                       # GeoJSON Polygon, EPSG:4326
  area_km2: float
  perimeter_km: float
  centroid_lat, centroid_lon: float
  major_axis_deg: float                # orientation, 0-180, degrees from north
  minor_axis_m: float
  major_axis_m: float
  eccentricity: float
  shape_complexity: float              # perimeter / (2 * sqrt(pi * area))
  mean_sigma0_db: float
  cnn_oil_prob: float
  anomaly_z: float
  wind_gate_pass: bool
  confidence: float                    # §4.1 fusion
```

### Environment

```
EnvField
  lats: list[float]
  lons: list[float]
  times: list[datetime]
  wind_u, wind_v: nested list[float]   # m/s
  curr_u, curr_v: nested list[float]   # m/s

EnvSummary
  mean_wind_speed_ms: float
  mean_wind_dir_deg: float
  mean_current_speed_ms: float
  mean_current_dir_deg: float
  wind_gate_pass: bool
```

### Trust and behaviour

```
TrustScore
  mmsi: int
  score: float                         # 0-1, 1 = fully trusted
  classification: Literal["MATCHED", "DARK", "PHANTOM", "IDENTITY_MISMATCH"]
  trust_flags: list[Flag]
  behaviour_flags: list[Flag]

Flag
  code: str
  severity: Literal["INFO", "WARN", "CRITICAL"]
  detail: str
  at: datetime | None
  lat, lon: float | None
```

### Attribution

```
Hypothesis
  hypothesis_id: str
  mmsi: int
  release_at: datetime
  release_lat, release_lon: float
  simulated_geometry: dict             # GeoJSON Polygon
  iou: float
  centroid_offset_km: float
  orientation_delta_deg: float
  area_ratio: float
  score: float                         # §4.5 weighted sum, 0-1

Suspect
  mmsi: int
  vessel_name: str | None
  rank: int
  posterior: float                     # softmax across candidates, sums to 1
  best_hypothesis: Hypothesis
  top_hypotheses: list[Hypothesis]
  trust_prior: float
  behaviour_prior: float
  proximity_prior: float
  rationale: list[str]                 # human-readable bullets

FunnelCounts
  total_in_scene: int
  in_envelope: int
  scored: int
  ranked: int

SlickAge
  best_hours: float
  low_hours: float                     # across top 3 hypotheses
  high_hours: float
  fay_estimate_hours: float
  agrees_with_fay: bool
```

### Incident and forecast

```
Incident
  incident_id: str
  acquisition_at: datetime
  aoi_bounds: tuple[float, float, float, float]   # min_lon, min_lat, max_lon, max_lat
  observed_slick: SlickPolygon
  origin_envelope: dict                # GeoJSON Polygon
  env_summary: EnvSummary
  funnel: FunnelCounts
  suspects: list[Suspect]
  slick_age: SlickAge

ForecastPolygon
  horizon_hours: int                   # 24, 48, 72
  geometry: dict
  uncertainty_cone: dict
  area_km2: float
  coastline_intersects: bool
  coastline_eta: datetime | None
  affected_shoreline_km: float
```

### Synthetic ground truth

```
Scenario
  scenario_id: str
  culprit_mmsi: int
  true_release_at: datetime
  true_release_lat, true_release_lon: float
  acquisition_at: datetime
  aoi_bounds: tuple[float, float, float, float]
```

**`ground_truth.json` is read by the test suite only.** No pipeline module may open it.

---

## 4. Algorithms

### 4.1 Slick segmentation

- **Preprocess** — land mask; wind gate accepting only 3–10 m/s (below, the sea is too
  flat to see a slick against; above, wind mixes it away); tile 512×512 with 64 px overlap.
- **Segment** — DeepLabv3+ / ResNet-50 over tiles, stitched with overlap-averaged logits.
- **Polygonize** — mask to polygons carrying every geometric feature in §3, including
  `major_axis_deg`, `eccentricity` and `shape_complexity`.
- **Fuse** — confidence combines the learned and statistical signals:

  ```
  confidence = cnn_oil_prob * sigmoid(anomaly_z - 1.5) * wind_gate_pass
  ```

  A dark patch is only oil if the CNN says so *and* it is anomalous against that cell's
  own history *and* the wind was in the detectable band. Look-alikes fail at least one.

### 4.2 Baseline and anomaly

H3 resolution-7 grid over the AOI. Per cell, accumulate sigma-0 statistics conditioned on
wind bin. Anomaly is the z-score of the observed cell mean against that conditional
distribution. **If fewer than 15 scenes are available for a cell, fall back to a global
background estimate and log that clearly** — the fallback must be visible, not silent.

### 4.3 Vessel detection (CFAR)

Two-parameter CFAR with guard cells, threshold `k = 4.5`. Morphological grouping of
above-threshold pixels into targets. Vessel length from the connected component's major
axis. Targets are matched to AIS positions at acquisition time; unmatched radar targets
are **DARK** vessels, and AIS reports with no radar target are **PHANTOM**.

### 4.4 Trust, identity and behaviour

**Trust checks** — reported SOG vs speed derived from consecutive positions (flag
disagreement > 5 kn); position teleports; impossible acceleration; positions on land;
MMSI MID prefix inconsistent with declared flag; duplicate MMSI broadcasting from two
places at once; AIS gaps over 15 minutes inside the AOI.

**Behaviour checks** — loitering (low speed with small net displacement over a window);
unexplained speed reduction; course deviation from the vessel's own prevailing heading;
night-time manoeuvring.

**Classification**

| Class | Meaning |
|---|---|
| `MATCHED` | AIS report and radar target agree |
| `DARK` | Radar target, no AIS |
| `PHANTOM` | AIS report, no radar target |
| `IDENTITY_MISMATCH` | AIS present but internally inconsistent (MID/flag, duplicate MMSI, teleport) |

### 4.5 Drift and attribution

#### 4.5.1 Drift physics

```
v = current + 0.03 * wind_rotated
```

where `wind_rotated` is the wind vector turned **15° clockwise** (Coriolis/Ekman
deflection in the northern hemisphere). Integration is 5-minute Euler steps. Each
particle takes an independent random-walk diffusion step with `K = 5 m²/s`.

Positions convert to metres via `pyproj`, never a fixed constant. This is the single most
common source of a silent catastrophic bug in this system, which is why the round-trip
test in `tests/test_drift.py` exists.

- `advect_forward(release_lat, release_lon, release_time, target_time, env_field, n_particles=500)`
  → particle positions plus a polygon (alpha shape, falling back to convex hull).
- `advect_reverse(slick_polygon, acquisition_time, hours_back, env_field)`
  → origin envelope polygon per time step.

#### 4.5.2 Hypothesis scoring

For each surviving vessel, sample release hypotheses every **30 minutes** along its own
track within the window. Forward-simulate each and score against the observed slick:

```
score = 0.40 * iou
      + 0.25 * centroid_term
      + 0.25 * orientation_term
      + 0.10 * area_term
```

with

```
centroid_term    = exp(-centroid_offset_km / 10.0)
orientation_term = 1 - (angular_delta_deg / 90.0)      # clamped to [0, 1]
area_term        = min(a, b) / max(a, b)               # a, b = simulated, observed area
```

Weights load from `config/weights.yaml`. **They are never tuned to make a scenario
produce the right answer.** If ranking is wrong, print the term breakdown and find the
broken term.

#### 4.5.3 Candidate pruning

Given the reverse-advected origin envelope and all AIS tracks, keep only vessels
intersecting the envelope in **both space and time**. Report the funnel:
`total_in_scene → in_envelope → scored → ranked`.

The funnel is a first-class output, not a log line. Showing that 40 vessels became 6
became 1 is the argument.

#### 4.5.4 Ranking

```
weighted_i  = best_score_i * trust_prior_i * behaviour_prior_i * proximity_prior_i
posterior_i = softmax(weighted / T)_i
```

Prior multipliers come from `config/weights.yaml`. Before layer 5 exists, all priors
pass as `1.0`. Each `Suspect` carries `rationale` — plain-English bullets a human
investigator can read without knowing the formula.

#### 4.5.5 Slick age

`age = acquisition_time - best_hypothesis.release_at`, reported as a range across the
top 3 hypotheses. Cross-check against inverse Fay spreading (area grows roughly as
`t^0.75`) and report whether the two independent estimates agree. Disagreement is
reported, not hidden.

### 4.6 Evidence and chain of custody

**`report.py`** produces a PDF via reportlab containing, in order: incident header; the
SAR scene chip with the slick polygon overlaid (rendered with matplotlib, embedded as an
image); environmental conditions table; candidate funnel; ranked suspect table with
posteriors and score breakdowns; winning hypothesis map; trust and behaviour flags
raised; estimated slick age; forecast summary; MARPOL Annex I reference section.

**`integrity.py`** computes SHA-256 over the canonical JSON of all pipeline inputs and
outputs, appends it with a timestamp to `artifacts/audit.log`, **hash-chained to the
previous entry**. The chain hash prints on the dossier's final page. An investigator can
verify no artifact was altered after the fact.

---

## 5. Scope

**In scope:** the layers above, running on synthetic ground truth and on real open data.

**Explicitly out of scope** — decline these if proposed: Docker, Kubernetes, CI. Login,
JWT, user accounts. Postgres or PostGIS (DuckDB needs no server). React or Next.js (a
single Leaflet HTML file will not break on stage). Refactoring attribution once its test
passes. Merging the two SAR datasets into one label scheme without inspecting their
actual masks first.

## 6. Honesty

Synthetic data is labelled synthetic, everywhere — in the dashboard, in the dossier, and
out loud. The problem statement permits synthetic data to demonstrate the algorithm. The
system's claim is that the *attribution engine* works and is verifiable against planted
ground truth; that claim is stronger when the boundary between real and synthetic is
stated plainly rather than blurred.
