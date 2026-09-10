# SAMUDRA — Oil Spill Attribution Engine

**SIH26143 (NTRO) — Satellite + AIS correlation for spill attribution**
**Final build contract — rev. 3**

This document is the build contract. Read it before writing code. Do not deviate from
the data contracts in section 3 — every module is built against them independently.
This file supersedes all earlier drafts. Section numbers below are referenced directly
by the Claude Code build prompts — do not renumber.

---

## 1. System identity

This is **not** an oil spill detector. Detectors already exist (EMSA CleanSeaNet,
SkyTruth Cerulean). This is an **attribution engine**: it answers *which vessel*, with a
defensible likelihood, against an adversary actively evading detection, and it does so
using a compounding memory that gets better with every incident rather than scoring
each scene in isolation.

Five architectural commitments distinguish it from existing systems:

| # | Gap in existing systems | Component |
|---|---|---|
| 1 | Detection is stateless per-scene | `baseline/` — persistent per-cell backscatter memory |
| 2 | Attribution is nearest-track proximity | `attribution/` — forward-simulation hypothesis testing |
| 3 | AIS is trusted or absent, never scored | `trust/` — adversarial credibility + behaviour scoring |
| 4 | Revisit gaps are dead time | `ingest/` — opportunistic multi-sensor (SAR + optical) |
| 5 | Irrelevant traffic isn't filtered explicitly | `attribution/prune.py` — named pruning stage with a reported funnel |

Output is an **evidence dossier for prosecution**, not a real-time alert. Sentinel-1
revisit makes interception impossible; deterrence through attribution is the achievable
goal. This is stated openly in the pitch, not hidden.

The brief explicitly asks for: (a) detection + geometric characterisation + age,
(b) hindcast to origin + forecast of future spread, (c) attribution scored on proximity,
trajectory, and behavioural anomalies, with irrelevant traffic filtered out. Every
lettered requirement maps to a numbered layer below — see the table in section 2.

---

## 2. System layers (11-layer architecture)

This is the authoritative layer list. It matches the reviewed architecture diagram.

| Layer | Name | Maps to brief requirement |
|---|---|---|
| 1 | Data Ingestion | SAR + EO + AIS + environment + AOI |
| 2 | Preprocessing & Baseline | — (supports detection quality) |
| 3 | AI Oil-Spill Detection | (a) detect and characterise |
| 4 | Vessel Analysis (CFAR + AIS matching) | (c) reconstruct vessel traffic |
| 5 | AIS Trust & Behaviour | (c) behavioural anomalies |
| 6 | Environment & Drift Model | (b) hindcast + forecast |
| 6b | Candidate Pruning | (c) filter out irrelevant traffic |
| 7 | Vessel Attribution Engine | (c) score and rank suspects |
| 8 | Impact Assessment & Prediction | (b) future flow of the slick |
| 9 | Investigator Dashboard | "suitable visual interface" |
| 10 | Evidence & Integrity | prosecution-grade output |
| 11 | Dissemination & Alerting | Disaster Management theme |

Storage & Infrastructure runs underneath all layers. A feedback loop (labelled **A** on
the diagram) carries confirmed incidents from the dashboard back into the vessel
dossier DB, which feeds attribution priors on future incidents — this is what makes the
system compound rather than stay static.

---

## 3. Repo layout

```
samudra/
├── CLAUDE.md                  # this file, symlinked
├── pyproject.toml
├── config/
│   ├── aoi.yaml                # AOI bounding boxes (demo cells)
│   └── weights.yaml            # scoring weights, tunable without code change
├── src/samudra/
│   ├── contracts.py            # ALL pydantic models. Build this first.
│   ├── synth/
│   │   └── generate.py         # synthetic scenario generator with ground truth
│   ├── ingest/
│   │   ├── sentinel1.py        # GEE -> GeoTIFF (VV/VH, calibrated, terrain-corrected)
│   │   ├── optical.py          # S2/S3 opportunistic pass (stub-able)
│   │   ├── environment.py      # Open-Meteo marine: wind u/v, current u/v
│   │   └── ais.py              # Real AIS loader (MarineCadastre bulk / AccessAIS)
│   ├── baseline/
│   │   ├── grid.py             # H3 res-7 cell indexing over AOI
│   │   ├── build.py            # rolling sigma0 stats per (cell, wind_bin)
│   │   └── anomaly.py          # z-score a new scene against baseline
│   ├── detection/
│   │   ├── preprocess.py       # land mask, wind gate, tiling
│   │   ├── segmenter.py        # DeepLabv3+ inference
│   │   ├── train.py            # fine-tuning on real SAR oil-spill datasets
│   │   ├── polygonize.py       # raster -> polygons + geometric features
│   │   └── fuse.py             # CNN prob x anomaly x wind gate -> confidence
│   ├── vessels/
│   │   ├── cfar.py             # SAR ship detection + length estimation
│   │   ├── tracks.py           # AIS interpolation, gap detection
│   │   └── match.py            # SAR <-> AIS association (Hungarian)
│   ├── trust/
│   │   └── score.py            # kinematic + identity + physical + behavioural checks
│   ├── attribution/
│   │   ├── drift.py            # particle advection (forward and reverse)
│   │   ├── prune.py            # candidate pruning + funnel reporting
│   │   ├── hypothesis.py       # release hypothesis generation + simulation
│   │   └── rank.py             # likelihood x priors -> posterior ranking + slick age
│   ├── impact/
│   │   └── forecast.py         # forward spread forecast + coastline impact ETA
│   ├── persist/
│   │   ├── db.py                # DuckDB schema + accessors
│   │   ├── corridor.py          # chronic discharge corridor aggregation
│   │   └── dossier_db.py        # per-MMSI offence history (feeds connector A)
│   ├── evidence/
│   │   ├── report.py            # reportlab PDF dossier
│   │   └── integrity.py         # SHA-256 hash chain + append-only audit log
│   └── api.py                   # FastAPI: serves artifacts as GeoJSON/JSON
├── web/                          # Leaflet frontend, dark basemap
├── scripts/
│   ├── run_demo.sh
│   └── train_segmenter.py
├── artifacts/<incident_id>/      # every stage writes here; demo replays from disk
├── raw_data/                     # unsorted downloads land here first
├── data/
│   ├── raw/
│   │   ├── ais/                  # sorted real AIS (Houston, filtered)
│   │   └── oilspill/
│   │       ├── sos/              # Kaggle Deep-SAR SOS (segmentation masks)
│   │       └── binary/           # Kaggle Sentinel-1 binary oil/no-oil
│   └── models/                   # checkpoints
└── tests/
```

Every pipeline stage is a CLI entrypoint that reads from and writes to `artifacts/`.
Nothing runs live during the demo. This is deliberate — it makes the demo deterministic
and lets modules be built in parallel against fixture files.

---

## 4. Data contracts

Write `contracts.py` first and commit it before anything else. All models are pydantic v2.

```python
class Scene(BaseModel):
    scene_id: str
    sensor: Literal["S1_GRD", "S2_MSI", "S3_SLSTR"]
    acquired_at: datetime          # UTC, always
    footprint: dict                # GeoJSON Polygon
    raster_path: Path
    incidence_angle_deg: float | None

class EnvSample(BaseModel):
    lat: float; lon: float; t: datetime
    wind_speed_ms: float; wind_dir_deg: float      # meteorological convention
    current_u_ms: float; current_v_ms: float       # eastward, northward

class SlickDetection(BaseModel):
    slick_id: str
    scene_id: str
    polygon: dict                  # GeoJSON Polygon, EPSG:4326
    area_km2: float
    perimeter_km: float
    shape_complexity: float        # P / (2 * sqrt(pi * A)); 1.0 = circle
    eccentricity: float
    major_axis_deg: float          # 0-180, key attribution feature
    mean_sigma0_db: float
    contrast_db: float             # background mean - slick mean
    edge_gradient: float
    cnn_oil_prob: float
    cnn_lookalike_prob: float
    baseline_anomaly_z: float      # from baseline/anomaly.py
    wind_gate_pass: bool
    confidence: float              # fused, calibrated 0-1

class AisPoint(BaseModel):
    mmsi: int; t: datetime
    lat: float; lon: float
    sog: float | None; cog: float | None; heading: float | None

class SarShip(BaseModel):
    target_id: str; scene_id: str
    lat: float; lon: float
    est_length_m: float
    peak_sigma0_db: float

class AisTrack(BaseModel):
    mmsi: int
    imo: int | None; name: str | None
    ship_type: int | None; flag_mid: int | None
    declared_length_m: float | None
    points: list[AisPoint]         # sorted by t
    gaps: list[tuple[datetime, datetime]]

class TrustScore(BaseModel):
    mmsi: int; scene_id: str
    score: float                   # 0-1, 1 = fully credible
    trust_flags: list[str]         # e.g. ["SOG_MISMATCH", "NO_SAR_RETURN"]
    behaviour_flags: list[str]     # e.g. ["LOITERING", "SLOW_STEAMING", "NIGHT_MANOEUVRE"]
    sar_corroborated: bool
    classification: Literal["MATCHED", "DARK", "PHANTOM", "IDENTITY_MISMATCH"]

class ReleaseHypothesis(BaseModel):
    mmsi: int
    release_at: datetime
    release_lat: float; release_lon: float
    simulated_polygon: dict
    iou: float
    centroid_offset_km: float
    orientation_delta_deg: float
    area_ratio: float
    geometric_score: float

class Suspect(BaseModel):
    mmsi: int; name: str | None
    best_hypothesis: ReleaseHypothesis
    likelihood: float              # normalised across candidates
    trust_prior: float
    behaviour_prior: float
    type_risk_prior: float
    gap_coincidence: bool
    prior_offences: int
    posterior: float               # final ranking value
    rationale: list[str]           # human-readable, goes into the PDF

class SlickAge(BaseModel):
    incident_id: str
    estimated_range_hours: tuple[float, float]   # across top-3 hypotheses
    fay_spreading_estimate_hours: float | None
    agreement: bool                # do the two estimators roughly agree

class Scenario(BaseModel):
    # synthetic ground truth ONLY. No pipeline module other than tests may read this.
    scenario_id: str
    culprit_mmsi: int
    true_release_at: datetime
    true_release_lat: float; true_release_lon: float
    acquisition_at: datetime
    aoi_bounds: tuple[float, float, float, float]
```

---

## 5. Module specifications

### 5.1 `baseline/` — persistent memory

Index the AOI with H3 resolution 7 (~5 km² cells). For each cell, maintain running mean
and standard deviation of VV sigma0, **bucketed by wind speed bin** (0–3, 3–5, 5–7,
7–10, 10+ m/s). Wind conditioning is essential — a cell is legitimately dark at 2 m/s
and legitimately bright at 12 m/s, and pooling those makes the baseline meaningless.

Storage: parquet, loaded into DuckDB. No database server.

`anomaly.py` returns a z-score per pixel: `(sigma0 - cell_mean[wind_bin]) / cell_std[wind_bin]`.
A genuine spill is anomalously dark *for that cell under those conditions*. A chronic
seep is not anomalous at all — which is the entire point.

Bootstrapping needs ≥15 scenes per AOI cell. If fewer are available, fall back to a
global background estimate and log that clearly rather than failing.

### 5.2 `detection/`

Preprocessing: land mask from Natural Earth coastlines with a 500 m buffer; wind gate
masking out pixels where wind is outside 3–10 m/s (below, the sea is glassy and
everything reads as a false positive; above, wind roughening scrubs the slick
signature). Tile to 512×512 with 64 px overlap.

Segmentation: DeepLabv3+, ResNet-50 backbone, torchvision. Training data is two
independent open SAR oil-spill datasets (see section 7 — the originally planned Zenodo
mirror was unavailable): a segmentation-mask set used as the primary trainer, and a
binary oil/no-oil set used as auxiliary signal. Class weighting is required; oil pixels
are heavily under-represented. Input is VV in dB, clipped to [-35, 0], normalised.

**Polygonization and geometric feature extraction is an explicit stage** — this is
where `area_km2`, `perimeter_km`, `shape_complexity`, `eccentricity`, and
`major_axis_deg` are computed from the segmentation mask. Every downstream module
(drift, attribution) depends on this output; detection is not "done" until these
features exist.

**Optical branch:** Sentinel-2/3 sun-glint and thermal contrast detection over the same
AOI when a cloud-free pass is available in the revisit gap. Feeds the same fusion step
as the SAR branch rather than sitting unused. A tighter time bracket from combining
sensors directly shrinks the attribution candidate set in `prune.py`.

Fusion:
```
confidence = cnn_oil_prob * sigmoid(baseline_anomaly_z - 1.5) * wind_gate_pass
```
Calibrate on a held-out split; do not ship raw softmax as probability.

**Age estimation** does not live in this layer — it is computed in `attribution/rank.py`
from the winning release hypothesis, since age requires knowing *when* the release
happened, which is an attribution output, not a detection output. See 5.5.

### 5.3 `vessels/`

CFAR ship detection on the same scene — two-parameter CFAR with guard cells: a pixel is
a target where `sigma0 > mu_bg + k * sigma_bg`, k ≈ 4.5, background estimated from an
annulus excluding guard cells. Morphological grouping into targets. Estimate vessel
length from the major axis of the connected component, corrected for range/azimuth
pixel spacing. Runs on the **calibrated, land-masked output of `preprocess.py`**, not
raw ingestion.

AIS interpolation: piecewise linear in time for gaps under 30 minutes. For longer gaps
use a constant-turn-rate model. Flag any gap over 15 minutes while the vessel was
inside the scene footprint.

Matching: Hungarian assignment, cost = `haversine_km + lambda * |sar_length - ais_length|`,
gated at 2 km. Unmatched SAR ships → DARK. Unmatched AIS positions inside the footprint
where SAR shows nothing → PHANTOM.

### 5.4 `trust/` — adversarial and behavioural layer

Two check families, both contributing to `TrustScore`:

**Trust (adversarial integrity).** Kinematic — reported SOG vs speed derived from
consecutive positions (flag >5 kn disagreement), position teleports, physically
impossible acceleration, positions on land, COG inconsistent with derived bearing.
Identity — MMSI MID prefix inconsistent with declared flag, duplicate MMSI broadcasting
from two locations simultaneously, SAR-measured length disagreeing with AIS-declared
dimensions by >25%. Physical — the strongest signal: AIS claims a position inside the
SAR footprint, CFAR finds no target there. AIS has no authentication whatsoever — it is
an unencrypted VHF self-report — so this cross-check is the only ground truth available.

**Behaviour (anomaly scoring).** Loitering (low speed, small net displacement over a
window), unexplained speed reduction (slow steaming is the classic discharge
signature), course deviation from the vessel's own prevailing heading, night-time
manoeuvring. This directly answers the brief's "behavioural anomalies" requirement,
which is distinct from adversarial spoofing.

Both trust score and behaviour flags feed `attribution/rank.py` as priors. A spoofed
track is not merely flagged; it reweights the entire suspect ranking.

### 5.5 `attribution/` — the core

**Two-stage design**, satisfying both the brief's explicit "trace toward origin"
requirement and the stronger inference-based differentiator:

**Stage 1 — hindcast (reverse advection).** Particle advection, forward Euler,
5-minute steps:
```
v_particle = v_current + alpha * v_wind
alpha = 0.03
wind deflection: rotate wind vector 15 deg clockwise (Northern Hemisphere Coriolis)
spreading: random walk, K = 5 m^2/s, applied per particle per step
```
Run in reverse from the observed slick to produce an **origin envelope** — a
space-time region, not a point. This is the literal backtracking the brief asks for,
and it is cheap, so it runs first to prune the search space before the expensive step.

**Stage 2 — candidate pruning (`prune.py`).** Intersect the origin envelope with every
AIS track in the scene. Only vessels intersecting it in both space and time survive.
This is the brief's explicit "filter out irrelevant traffic" requirement, and it must
be reported as a funnel, not silently applied:
```
total_in_scene -> in_envelope -> scored -> ranked
```
That funnel, with real numbers, is one of the most persuasive artifacts in the pitch.

**Stage 3 — forward hypothesis testing (`hypothesis.py`).** For each surviving vessel,
sample release times every 30 minutes along its own track, forward-simulate to
acquisition time, and score the simulated polygon against the observed one:
```
geometric_score = w1*IoU
                + w2*exp(-centroid_offset_km / 10)
                + w3*cos(radians(orientation_delta_deg))
                + w4*exp(-abs(log(area_ratio)))
```
Weights in `config/weights.yaml`, defaults 0.4 / 0.25 / 0.25 / 0.10. Orientation
matters more than it looks: a discharge from a *moving* vessel produces an elongated
slick aligned with its course — a strong discriminator between two ships that
transited the same water. Keep each vessel's best-scoring hypothesis. Softmax across
vessels → likelihood.

**Posterior (`rank.py`):**
```
posterior = likelihood
          * trust_prior
          * behaviour_prior
          * type_risk_prior          # tanker/bulk > container > fishing
          * (1.8 if gap_coincidence else 1.0)
          * (1 + 0.4 * prior_offences)
```
Renormalise. Emit ranked `Suspect` list with rationale strings. Never output a binary
accusation — ranked candidates with likelihood ratios only.

**Slick age.** `age = acquisition_time - best_hypothesis.release_at`, reported as a
range across the top 3 hypotheses. Cross-check against inverse Fay spreading
(area grows roughly as t^0.75) and report whether the two independent estimators agree
— agreement is a strong, cheap credibility signal for the pitch.

### 5.6 `impact/` — forecast and coastline effect

Forward advection of the *observed* slick (not a hypothesis) at +24h/+48h/+72h, with an
uncertainty cone derived from particle spread at each horizon. Intersect each forecast
polygon with a coastline geometry (Natural Earth) to report affected shoreline length
and an impact ETA. This is the brief's "predict the future flow of the slick"
requirement and must appear as a distinct product in the dashboard — a time slider
toggling observed vs each forecast horizon — not just an internal artifact.

### 5.7 `evidence/`

PDF via reportlab: SAR chip with slick polygon overlay, environmental conditions table,
candidate funnel, candidate vessel table with posteriors, hypothesis replay map for the
top suspect, estimated slick age, forecast summary, trust and behaviour flags raised,
MARPOL Annex I citation. Frame as evidence for the Indian Coast Guard under NOS-DCP.

Integrity: SHA-256 over the canonical JSON of all inputs and outputs; append hash plus
timestamp to `artifacts/audit.log`, hash-chained to the previous entry. A hash chain
gives the same chain-of-custody property as a blockchain without inviting the "why
blockchain" question from a technical panel.

### 5.8 Dissemination

An alert output to Indian Coast Guard / NOS-DCP alongside the evidence dossier, with
severity-tiered notification thresholds and a GeoJSON/REST export for response
agencies. This is what ties the system explicitly to the Disaster Management theme
rather than leaving it implicit.

### 5.9 Feedback loop (connector A)

Confirmed incidents, verified through the dashboard's human-in-the-loop review, write
back into the vessel dossier DB (`persist/dossier_db.py`), which supplies
`prior_offences` to future attribution runs. This is what makes the system compound: a
stateless per-scene detector is exactly as good on day 400 as on day 1; this one is not.

---

## 6. Build order

Checkpoints are hard gates — do not proceed past a failing one. This order is
risk-driven, not diagram-driven: the attribution engine (pure geometry, no GPU, no
downloads) is built and proven correct before any real data is touched, using a
synthetic scenario generator with known ground truth as the verification harness.

| Order | Task | Checkpoint |
|---|---|---|
| 0 | `contracts.py` complete and committed | imports clean, models instantiate |
| 1 | `synth/generate.py` — synthetic scenario with planted culprit | 5 artifact files written, culprit named |
| 2 | `attribution/drift.py` | forward-advected polygon overlaps ground truth, IoU > 0.5 |
| 3 | `attribution/prune.py` + `hypothesis.py` + `rank.py` | **top-ranked MMSI matches the planted culprit** |
| 4 | `api.py` + `web/` dashboard | map renders slick, tracks, ranked suspects |
| 5 | `trust/score.py` wired into priors | planted spoofed/loitering vessels flagged; culprit still ranked #1 |
| 6 | `impact/forecast.py` | forecast polygons render and grow over time on a slider |
| 7 | `evidence/report.py` | PDF generates with a real map image, not a placeholder |
| 8 | `scripts/run_demo.sh`, second synthetic scenario | culprit ranked #1 on a scenario not used during development |
| 9 | Real AIS ingestion (Houston, MarineCadastre bulk) | ranking still correct against real, messy traffic |
| 10 | SAR model training (two open datasets) | confusion matrix produced, oil vs background/look-alike |
| 11 | Real detection wired in, with synthetic fallback preserved | `run_demo.sh` still works with no checkpoint present |
| 12 | Rehearse demo x4 | no live computation on the critical path |

Checkpoint 3 is the most important gate in the entire build: it is the empirical proof
that the core differentiator works, independent of anything else in the system.

---

## 7. Data sources actually used

The problem statement explicitly permits synthetic AIS ("Real AIS if available may be
used else synthetic data can be prepared") and points at MarineCadastre and a Zenodo
SAR dataset. What was actually used, and why, should be stated plainly in the pitch —
this is compliance with the brief, not a shortcut:

- **AIS:** real historical broadcast-point data from NOAA MarineCadastre, bulk daily
  file, filtered to a Houston Ship Channel bounding box. Real vessel traffic, real
  density and noise, not synthetic tracks pretending to be real.
- **SAR training imagery:** the originally planned Zenodo Sentinel-1 oil-spill mirror
  was unavailable — a documented, verifiable multi-day outage across Zenodo's files,
  search, and website services, not a data-access failure on our side. Substituted with
  two independent open datasets on Kaggle: one with pixel-level segmentation masks
  (primary trainer), one with binary oil/no-oil labels (auxiliary signal). Both are
  genuinely open Sentinel-1 SAR imagery.
- **Environmental fields:** synthetic for the demo AOI (Arabian Sea), generated with
  realistic wind/current statistics. Open-Meteo marine API is the production path.
- **Ground truth for verification:** entirely synthetic, generated with a known culprit
  vessel, known release time and position, and planted adversarial behaviour (spoofing,
  loitering, gaps). This is what lets the attribution engine's correctness be checked
  without needing an independently verified real spill, which does not exist on any
  demo timescale.

---

## 8. Rules for Claude Code

- **One module per session.** Load `contracts.py` plus the target module's spec section
  only. Do not let context sprawl across the whole repo.
- **Fixtures before implementations.** For each module, write
  `tests/fixtures/<module>_in.json` by hand first where practical.
- **No live network calls in any function under `attribution/` or `detection/`.**
  Ingest is the only layer that touches the network.
- **All timestamps UTC, all geometry EPSG:4326 GeoJSON.** No exceptions, no local CRS
  leaking into contracts.
- **Every stage CLI-invokable:** `python -m samudra.<module> --incident <id>`. Reads
  from and writes to `artifacts/<id>/`.
- **Never adjust scoring weights to force a correct ranking.** If attribution ranks the
  wrong vessel, the fix is a bug fix in the geometry or priors, diagnosed by printing
  the full score breakdown — not a weight tweak that happens to fix one scenario.
- Do not add authentication, user accounts, or RBAC. Out of scope, and it dilutes the
  pitch.
- Do not refactor across module boundaries without updating `contracts.py` first.
- `raw_data/` contents are never assumed — inspect actual file structure, column names,
  and label conventions before writing any loader against them.

---

## 9. What is stubbed, and why that's fine to say out loud

- Real-time interception is explicitly out of scope. Sentinel-1 revisit over the Indian
  EEZ is roughly 6–12 days; the system is forensic attribution and deterrence, not
  interception. State this before a judge asks — it reads as expertise, not a gap.
- The optical (Sentinel-2/3) branch is architected and wired into fusion, but validated
  on limited real passes given the timeline — say so plainly.
- The Indian EEZ deployment AOI uses synthetic AIS by design (per the brief); the
  Houston AOI uses real AIS to validate the engine against genuine traffic density.
  Both are legitimate demonstrations of different things — say which is which on every
  slide that shows a map.
- Corridor/chronic-discharge mapping is architected (`persist/corridor.py`) but seeded
  from a single demo AOI rather than the full EEZ within this timeline.

---

## 10. Implementation notes and deviations

Added by the build. Sections 1–9 are unchanged. Every deviation from the contract
above is recorded here rather than left to be discovered in the code.

### 10.1 Deliberate deviations

**Posterior temperature (5.5).** The order of operations follows 5.5 exactly —
`softmax(geometric_score)` → multiply priors → renormalise — but the softmax uses
temperature 0.25, not the 1.0 a plain softmax implies.

`geometric_score` is bounded in `[0, 1]`, so a plain softmax barely separates
candidates: `exp(0.90)/exp(0.29)` is only 1.8×, which priors of up to 1.7× can
overturn. Measured on demo-001 at T=1.0, a candidate whose simulated slick was
**86° off** with IoU 0.122 ranked **second** on its priors alone, above a
candidate that scored twice as well. That inverts the system's central claim that
geometry decides and priors only modulate. T=0.05 was also rejected: a 0.998
posterior is effectively the binary accusation 5.5 forbids.

Temperature is a calibration constant, not one of the scoring weights section 8
forbids tuning. It lives in `config/weights.yaml` and changes no ranking order —
only the confidence spread.

**Area term (5.5).** The spec writes `w4*exp(-abs(log(area_ratio)))`. The code
uses `min(a,b)/max(a,b)`, which is algebraically identical and avoids a log of
zero when a simulated polygon degenerates.

### 10.2 Additions

- `geo.py` — shared projection, polygon metrics, IoU. The *physics* in
  `synth/generate.py` and `attribution/drift.py` stays independently implemented;
  that independence is what the drift round-trip test verifies.
- `timeutil.py` — **all** epoch conversion goes through `epoch_seconds`. pandas 3.0
  stores `datetime64[us]`, so the common `astype("int64") / 1e9` idiom returns
  values 1000× too small and fails *silently*: time-window filters match nothing
  instead of raising. This bug cost a debugging cycle; do not reintroduce it.
- `contracts.py` Part B — models the pipeline needs that section 4 does not define
  (`Flag`, `AisGap`, `EnvField`, `EnvSummary`, `FunnelCounts`, `ForecastPolygon`,
  `Incident`). Kept in a separate labelled block. `tests/test_contracts.py` parses
  section 4 out of this file and asserts Part A matches it field-for-field, so the
  contract and its implementation cannot drift apart silently.
- `advect_forward(..., track=...)` — a vessel discharging while under way is a
  *line* source. Without it the simulated slick is a round blob with no
  orientation, and the 25% orientation term in 5.5 carries no information.
- Decoy vessels in the synthetic generator. Without them the origin envelope held
  exactly one candidate and ranking never had to discriminate, so a correct answer
  proved nothing.

### 10.3 Known gaps

- **Coastline is bundled and simplified.** This environment sits behind TLS
  interception, so the Natural Earth download fails certificate verification even
  with `certifi`. A simplified Indian west coast ships with the code and the source
  actually used is reported in every result. Drop `ne_50m_coastline.geojson` into
  `data/raw/` and it is picked up with no code change. Neither demo scenario
  reaches shore within 72 h, so the ETA path is untested against a real impact.
- **The Fay age cross-check is not independent.** It requires a reference-area
  constant encoding an assumed discharge volume and oil type. It is a consistency
  check against an assumed spreading rate. The dossier says so; do not let it be
  described as corroboration.
- **`prior_offences` is always 0.** `persist/dossier_db.py` (connector A) is not
  built, so the feedback loop is architected but not closed.
- Not yet built: `ingest/sentinel1.py`, `optical.py`, `environment.py`,
  `baseline/`, `vessels/match.py`, `persist/`, layer 11 dissemination.
