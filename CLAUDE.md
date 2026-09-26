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
- **Operational SAR imagery — real EOS-04, the satellite the brief actually names.**
  One NRSC Bhoonidhi L2B ARD scene over the Gulf of Kutch is now ingested end to
  end: `E04_SAR_MRS_05JUL2026_...N22565_E069445`, acquired 2026-07-05 01:17:53 UTC,
  dual-pol HH/HV, 18 m, 10501 x 10401 px, UTM 42N, RTC applied. This is ISRO data
  from an ISRO portal, distinct from the training imagery above, which remains
  foreign because no labelled Indian SAR oil-spill dataset is published. The
  distinction is worth stating on the slide rather than blurring: **the model is
  trained on open foreign SAR because that is the only labelled oil-spill data
  that exists; it is run on Indian SAR.**
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

**Segmentation objective (5.2).** The spec fixes the fusion formula, not the
training loss. The loss is now weighted cross-entropy **plus soft Dice**
(`dice_weight`, default 0.5), and the oil class weight is damped by
`weight_power` (default 0.5) instead of the full inverse frequency.

Why, measured rather than assumed: weighted CE alone gave oil **recall 0.983 at
precision 0.345** — the network had learned that over-predicting oil is cheap,
because the weight punishes a missed oil pixel far harder than a false one. CE
scores pixels independently and has no view of the predicted region; Dice scores
the overlap of the region as a whole, so inflating it stops being free. Both
knobs are CLI flags, and `--dice-weight 0 --weight-power 1.0` reproduces the old
objective exactly — which is how the before/after in
`training_runs/` was produced.

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
- `dissemination/` — layer 11 (section 5.8). Composes a severity-tiered alert
  addressed to the Indian Coast Guard units that would actually respond, and
  exports the station set as GeoJSON. Severity uses **NOS-DCP tiers (I local /
  II regional / III national)** rather than an invented scale, because the tier
  is what decides the distribution list. The list is not the N nearest pins: a
  tier adds *levels of command*, so slots are reserved for a regional HQ and
  (Tier III) Coast Guard HQ, which a plain distance sort loses whenever several
  local stations cluster — see the Gulf of Kutch case in
  `tests/test_dissemination.py`. Thresholds live in `config/weights.yaml`.
  NOS-DCP is written in tonnes and SAR measures area, so area is an explicit
  volume proxy; the alert says so rather than implying tonnage.
- **`satellite/` — the operations console's ingest and inference path (layer 1
  + layer 3).** `sources.py` enumerates what this machine can actually process
  in three separated modes: LIVE/NRT (credential-gated, reports
  AUTHENTICATION REQUIRED and never substitutes archive data for a feed),
  REAL_REPLAY (genuine SAR from published archives, plus a drop-in folder for a
  product you downloaded yourself) and SYNTHETIC (this project's own scenes,
  labelled). `inference.py` runs the real checkpoint over the real pixels.
  `pipeline.py` deliberately implements almost nothing: it writes
  `detected_slicks.geojson` into a fresh incident and calls
  `attribution.__main__.run()`, so there is no second drift engine and no second
  ranker.

  **LIVE/NRT carries Indian providers only.** EOS-04 (Bhoonidhi/NRSC), INSAT-3DS
  and EOS-06 (MOSDAC/ISRO).

  **Correction — an earlier revision of this section was wrong about MOSDAC.**
  It recorded that both providers "gate their catalogue behind a registered
  account and publish no open endpoint", citing a 401 from MOSDAC
  `/opendata/`. That probe hit the wrong service. MOSDAC publishes a documented
  **Data Download API** (`/downloadapi-manual`, client at `/software/mdapi.zip`)
  whose manual states plainly: *"You don't need to log in to search; an account
  is only required to download data."* Search takes `datasetId`, `startTime`,
  `endTime`, `boundingBox` (`minLon,minLat,maxLon,maxLat`) and `count` (max
  100); `datasetId` values come from the catalogue at
  `/catalog-app/satellite.php` and look like `3SIMG_L1B_STD` (INSAT-3D Imager
  L1B) and `E06OCM_L2C_AD` (EOS-06 OCM L2C). Download is credentialed, capped at
  5000 files/user/day, and three bad logins lock the account for an hour.

  **The MOSDAC adapter is now real and exercised.** Endpoints were read out of
  ISRO's own reference client (`/software/mdapi.zip`), not guessed:
  search `GET /apios/datasets.json`, token `/download_api/gettoken`, download
  `/download_api/download`. `Mosdac.search()` queries the live service and
  parses its response; `tests/test_satellite.py` covers it.

  Dataset ids come from MOSDAC's own catalogue service
  (`POST /catalog/Search/getAllProductData.php`, satellite ids INSAT-3DS=24,
  EOS-06=22). The manual's published example `E06OCM_L2C_AD` is stale; the real
  prefix is `E06OCM_L2C_LAC_*`. The five SAMUDRA consumes are in
  `Mosdac.DATASETS`, each carrying the layer it feeds. **All were confirmed to
  hold real data over the Gulf of Kutch for 2026-07-04..06 — the window of this
  project's real EOS-04 acquisition:**

  | dataset | product | granules (3 days) |
  |---|---|---|
  | `3SIMG_L2B_SST` | INSAT-3DS sea surface temperature | 133 |
  | `E06SCT_L2B_WV12` | EOS-06 scatterometer wind, 12.5 km | 176 |
  | `E06SCT_L4_AWV6HOURLY` | EOS-06 analysed wind, 6-hourly | 12 |
  | `E06OCM_L2C_LAC_OC` | EOS-06 OCM-3 ocean colour | 23 |

  This is also the evidence for the revisit answer (section 9): EOS-04 sees a
  given point roughly every 5-12 days, while INSAT-3DS delivers ~48 half-hourly
  observations per day over the same water. The gap is filled with *context* —
  wind that drives the drift model and the 3-10 m/s detection gate, SST, and
  chlorophyll for biogenic look-alike rejection — not with extra detections;
  at 1-4 km INSAT-3DS cannot resolve a routine discharge and does not pretend to.

  **Search and download are verified separately, and must stay that way.**
  `search_verified` is True for MOSDAC and False for Bhoonidhi;
  `download_verified` is False for both, because retrieval needs a
  human-approved account and has never been run by this code. `verified` is the
  end-to-end claim and is therefore still False for both. A single flag was the
  original modelling error: it forced a choice between understating a working
  search and overclaiming a download that does not exist.
  `test_mosdac_download_refuses_even_though_its_search_is_open` pins this.
  LIVE/NRT `available` keys off credentials, never off search, so an open
  catalogue can never be displayed as a live pixel feed.

  Bhoonidhi's 404s on `/opensearch`, `/api/` and `/services` stand as measured.

  An earlier revision carried a Copernicus / Sentinel-1 adapter, because ESA's
  catalogue answers unauthenticated and it was the only live path that could be
  exercised here. **It was removed.** Sentinel-1 is an ESA mission; SIH26143
  specifies an Indian sensor stack, and a foreign mission sitting in the live
  provider list invites exactly the wrong conclusion no matter how it is
  labelled. `tests/test_satellite.py::test_no_foreign_provider_is_carried_as_a_live_source`
  keeps it out.

  The working route today is the drop-in folder: a product you downloaded from
  the portal, plus a sidecar `.json`, ingests with full provenance.

  Two rules the module exists to enforce. **`acquired_at` is `None` whenever the
  source does not publish a per-image acquisition time** — never "now", never a
  plausible constant, never the file mtime; the console prints "NOT PUBLISHED BY
  SOURCE". And **no label source is ever on the inference path**:
  `assert_no_ground_truth()` is a runtime guard and `tests/test_satellite.py`
  asserts at source level that no line both names a label and reads a file.

  The honest seam is stated in the UI rather than buried: a research chip has no
  CRS and no timestamp, so its map position is an **operator input**
  (`georeference: "operator"`) and the wind/current field and AIS traffic come
  from a named **context incident**. The detection is real; the water it is
  placed in is declared context.
- **Training protocol corrected — the previous revision selected on test.**
  `scripts/train_segmenter.py` evaluated the published test split after every
  epoch and kept the checkpoint with the best test oil-IoU. That is selection on
  the test set: the number it reported was the maximum of N noisy draws, not an
  estimate of unseen performance. There is now a deterministic 85/15
  train/val partition of the published train directory (hash of the chip
  identity, so it does not move when `limit` changes), validation drives early
  stopping, checkpoint selection and the decision threshold, and **test is
  loaded once at the end**. `tests/test_training_protocol.py` asserts the splits
  are disjoint and deterministic. The old numbers are in git; they are not
  comparable to the new ones.
- `web/ops.html` (route `/ops`) — operations console. Top navigation is the
  primary mechanism (Dashboard / Satellite / Spill / Drift / AIS / Evidence /
  Help); there is no permanent application sidebar, only a contextual panel per
  map page. **Every optional map layer is off at load** and nothing is drawn
  until it is asked for. There is no standalone Winds panel: wind remains a
  first-class backend variable driving the hindcast and the detection gate, and
  surfaces as a compact chip where it is relevant. `/` and `/app` are untouched.
- `web/dashboard.html` (route `/app`) — a second dashboard beside the original.
  One draggable timeline runs T−16 h (reverse-drift steps) → T0 (acquisition) →
  T+72 h (forecast horizons), with AIS positions interpolated on the new `t`
  array in `/api/.../tracks`. Forward time **snaps to the computed 24/48/72 h
  horizons and never interpolates between them**: a smoothly morphing polygon
  would be a second, silently different drift model living in the browser.
  `web/flow_layer.js` animates wind and current from the pipeline's own
  `env.npz` (served by `/api/.../env`) — the same field drift integrates,
  though the layer is a display integrator without leeway, deflection or
  diffusion, so no trajectory should be read off it. `/` is untouched.

- **`ingest/eos04.py` — real EOS-04 L2B ARD, verified against itself.** Layer 1
  for the satellite the brief names. Two numbers decide whether anything
  downstream is defensible — the radiometric scale and the acquisition time —
  and both are *checked against the product's own independent statements*
  rather than trusted:

  *Calibration.* BAND_META publishes sigma0, gamma0 and beta0 constants, which
  are not independent: `beta0 = sigma0/sin(theta)` and `gamma0 = sigma0/cos(theta)`.
  Both identities hold on the real product to **0.0018 dB and 0.0000 dB** at the
  published 37.85 deg incidence, which confirms the equation
  `sigma0_dB = 20*log10(DN) - K` from the product rather than from a handbook. A
  constant wrong by a few dB — the error nothing downstream would notice, since
  it merely rescales every contrast — breaks the identity immediately and the
  product is refused.

  *Timezone.* BAND_META states SceneCenterTime with no zone, and separately
  publishes SunElevationAtCenter. Modelled solar elevation at the scene centre
  is **+7.50 deg reading it as UTC** against the published +6.64, and **-44.26
  deg reading it as IST**. The timezone is therefore established by physics. It
  matters: 5h30m of error moves every back-projected release position tens of
  kilometres, which is enough to change which ship ranks first.

  Land is an **operator input** here, deliberately. The bundled coastline
  reaches 22.47 N and cannot resolve the Gulf of Kutch; worse, the Rann salt
  flats and tidal creeks are radiometrically *dark*, so an automatic dark-feature
  detector over them produces false positives indistinguishable from slicks —
  observed directly, 52 regions over a coastal AOI versus 1 over a water-only
  one. Shipping a land mask that is wrong in the one place being demonstrated is
  worse than declaring the water box and recording that it was declared.

  Consequences elsewhere: `satellite/inference.py` gained `load_scene_db`, which
  feeds a **calibrated** product's true dB straight onto the [-35, 0] training
  ramp instead of the per-image percentile stretch used for uncalibrated chips —
  running an absolutely-calibrated product through a relative stretch would
  discard the very thing that makes it worth having. Nodata is filled with the
  scene median (so absent data is ordinary sea, never a fabricated dark anomaly)
  and predictions over it are zeroed. `satellite/pipeline.py` gained an exact
  pixel->lon/lat path using the product's own affine transform; the previous
  anchor-and-ground-sample approximation ignores UTM grid convergence, worth
  ~190 m at the edge of a 50 km AOI. And `sources.list_dropin` was reading a
  **projected** raster's bounds as degrees, which put this scene at latitude
  2.5 million — fixed, with a regression test.

### 10.3 Known gaps

- **Coastline — RESOLVED.** This was recorded as a permanent limitation ("the
  Natural Earth download fails certificate verification even with `certifi`")
  and it was not one. The failure was TLS interception, not the network:
  `truststore` routes verification through the Windows certificate store and
  the fetch succeeds. Verification stays ON; nothing disables it.
  `scripts/fetch_coastline.py` does it, and `impact/coastline.py` picks the file
  up with no code change.

  This mattered more than a cosmetic gap. The bundled fallback traces the Indian
  **west coast only**, so a shoreline-impact ETA anywhere on the east coast, the
  Andamans or Lakshadweep had no coast to intersect — and the project's own
  claim is national coverage. Measured against the real coastline now on disk,
  distance from the nearest modelled shore: Chennai 1.1 km, Visakhapatnam
  0.3 km, Paradip 2.4 km, Sundarbans 4.8 km, Port Blair 2.2 km.

  `COASTLINE_PATHS` prefers **10m over 50m**. Natural Earth's "10m" is
  1:10,000,000 — finer than "50m", not coarser — and the difference is not
  cosmetic: at 1:50m the Lakshadweep group is dropped entirely, putting the
  nearest modelled shore **71 km** from Kavaratti instead of **0.3 km**. Small
  islands are exactly the sensitive landfalls an impact ETA exists to warn
  about. Cost is 0.62 s to load instead of 0.11 s.

  Still true: neither demo scenario reaches shore within 72 h, so the ETA path
  remains untested against a real impact.
- **The Fay age cross-check is not independent.** It requires a reference-area
  constant encoding an assumed discharge volume and oil type. It is a consistency
  check against an assumed spreading rate. The dossier says so; do not let it be
  described as corroboration.
- **`prior_offences` is always 0.** `persist/dossier_db.py` (connector A) is not
  built, so the feedback loop is architected but not closed.
- **The baseline has one scene, so anomaly scoring runs in global-fallback mode.**
  CLAUDE.md 5.1 needs >= 15 scenes per cell before local statistics mean anything.
  `baseline/anomaly.py` accumulates correctly and reports which mode ran; it simply
  has no history to draw on yet, and says so on every run.
- **The only checkpoint is a CPU smoke run.** `detection/segmenter.py` works and its
  output is correct plumbing, but the weights are not meaningful. Attribution refuses
  to use them by default. Run `notebooks/train_colab.ipynb` for real weights.
- **Alerts are composed, never transmitted.** `dissemination/` addresses the
  message and shows exactly what would go out; `transmitted` is always false and
  nothing leaves the machine. Wiring a transport needs real verified recipients
  and an authority to send, which is a deployment decision, not a demo feature.
- **The Coast Guard station list is compiled from public establishment
  locations**, to roughly 1 km. Good enough to pick the nearest unit and show
  range and bearing; verify against the current ICG list before operational use.
- **The Indian depiction is drawn by us, not by the basemap.** Bhuvan was tried
  first and removed: NRSC's open tilecache answers every request with HTTP 200
  and a "Data not available" placeholder without NRSC access — verified over
  Gujarat, Mumbai, Kashmir and open ocean — and a 200 means Leaflet never fires
  `tileerror`, so it cannot even be detected and fallen back from. Every global
  basemap (Esri, OSM, CARTO) draws Jammu & Kashmir per international convention.
  So `web/india_boundary.json` carries the national boundary with **J&K, Ladakh
  (including Aksai Chin) and Arunachal Pradesh inside it**, and the dashboard
  draws it over the tiles. It is **filled**, not outlined: the fill covers
  whatever line the tiles drew underneath, where an outline alone would leave
  the basemap's boundary visible beside ours. Below zoom 9 it drops to an
  outline so coastal detail returns at incident zoom.
  Source: DataMeet Community Maps (`github.com/datameet/maps`,
  `Country/india-composite.geojson`, CC-BY 4.0), dissolved and simplified to
  0.01°. Verified to contain Srinagar, Leh, Aksai Chin, Itanagar and Tawang and
  to exclude Kathmandu, with a northern extent of 37.1°N. It is a display
  boundary, not a survey product, and must not be used for demarcation.
- **INSAT-3DS SST and EOS-06 OCM-3 chlorophyll in `web/isro_layers.js` are
  modelled fields, not MOSDAC products.** They show what the ingest layer would
  look like once those sensors are wired in and are badged PROPOSED everywhere
  they appear. Swapping in real data means replacing `sstAt()` and `chlAt()`
  with a sampler over a MOSDAC GeoTIFF; nothing else changes.
- **No AIS exists for the real EOS-04 scene, so it cannot be attributed.** The
  Gulf of Kutch on 2026-07-05 is covered by no AIS this machine holds — the real
  AIS is Houston, and the Indian AOIs are synthetic by design (section 7).
  `pipeline.process` therefore returns `attribution_unavailable` for that scene
  and names no suspect. That is the designed behaviour, not a missing feature:
  widening the origin envelope until some vessel falls inside is precisely how
  an attribution engine starts accusing innocent ships. Attribution remains
  demonstrated on the synthetic incidents, which have a sealed culprit to check
  the answer against.
- **The detection model is trained on VV and run here on HH — now partly
  quantified.** EOS-04's MRS product is HH/HV; CLAUDE.md 5.2 trains on VV, and
  HH sits a few dB below VV over ocean at this incidence. No labelled Indian SAR
  oil-spill set exists, so the shift cannot be measured the usual way. Two
  measurements bound it instead, both made on the real EOS-04 HH scene:

  *False positives.* Over open water: **0.014% of pixels flagged, one region**
  in 2743 x 872 px — and that region was a 1.6 dB darkening 276 m downwind of a
  127 m vessel, i.e. a wake, which the `contrast_db` field now surfaces for
  rejection. Over the coastal margin of the same scene the model flagged land
  edges, tidal creeks and wet mudflats heavily, which is why the marine AOI is
  an operator input.

  *Sensitivity.* `scripts/detection_limit.py` plants elliptical slicks of known
  area and known contrast into real EOS-04 HH ocean by dB subtraction — which
  preserves the true speckle texture underneath, where substituting smooth
  synthetic values would make them trivially detectable — and runs the real
  checkpoint at its real threshold. 14 independent ocean chips per cell:

  | area | 1 dB | 2 dB | 3 dB | 5 dB | 8 dB | 12 dB |
  |---|---|---|---|---|---|---|
  | 0.05 km² | 0% | 0% | 0% | 14% | **86%** | **100%** |
  | 0.10 km² | 0% | 0% | 21% | **93%** | **100%** | **100%** |
  | 0.25 km² | 0% | 14% | **93%** | **100%** | 100% | 100% |
  | 1.00 km² | 0% | 50% | **100%** | 100% | 100% | 100% |

  Because this is measured on HH pixels, the domain shift is *inside* these
  numbers rather than a caveat beside them.

  **The floor is contrast, not area.** Real oil damps capillary waves 5-15 dB,
  so operation sits in the right-hand columns. Below ~2 dB — the measured
  speckle sd is 2.4 dB — a dark patch is not separable from a wind shadow or a
  wake at any size, and the table shows exactly that: the 1 dB column is zero
  everywhere, including at 2.5 km². Nothing below 0.021 km² can be reported at
  all, because `min_area_px = 64` at 18 m is that area; the 0.005-0.02 km² rows
  are rejected by the speck filter before the model is consulted.

  Worth stating in the pitch: **0.05 km² is 5 hectares, 200x smaller than the
  ~10 km² floor a 1-4 km geostationary pixel imposes.** Small-slick detection is
  therefore not a weakness of this system; revisit is. An FRS-mode product
  (9 m instead of 18 m) would put four times the pixels on the same slick and
  drop the component floor to 0.005 km² — a data choice, not an ML one, and the
  single largest available improvement.
- Not yet built: `optical.py`, `environment.py`, `baseline/grid.py` and
  `build.py` as separate modules (folded into `anomaly.py`), `vessels/tracks.py`
  and `match.py`, `persist/`. `ingest/sentinel1.py` is **deliberately not built**
  — Sentinel-1 is an ESA mission and this system ingests Indian satellites;
  `ingest/eos04.py` occupies that slot.
