"""Data contracts — CLAUDE.md section 4.

Part A reproduces section 4 of the build contract exactly: same model names,
same field names, same types. Every module is built against these independently,
so they change only by explicit request.

Part B holds models the pipeline needs that section 4 does not define. They are
kept separate and labelled so the boundary between "the contract" and "what this
build added" stays visible.

All models are pydantic v2. All timestamps UTC. All geometry GeoJSON EPSG:4326,
longitude first.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ==========================================================================
# PART A — CLAUDE.md section 4, verbatim
# ==========================================================================


class Scene(_Base):
    scene_id: str
    sensor: Literal["S1_GRD", "S2_MSI", "S3_SLSTR"]
    acquired_at: datetime  # UTC, always
    footprint: dict[str, Any]  # GeoJSON Polygon
    raster_path: Path
    incidence_angle_deg: float | None = None


class EnvSample(_Base):
    lat: float
    lon: float
    t: datetime
    wind_speed_ms: float
    wind_dir_deg: float  # meteorological convention
    current_u_ms: float  # eastward
    current_v_ms: float  # northward


class SlickDetection(_Base):
    slick_id: str
    scene_id: str
    polygon: dict[str, Any]  # GeoJSON Polygon, EPSG:4326
    area_km2: float
    perimeter_km: float
    shape_complexity: float  # P / (2 * sqrt(pi * A)); 1.0 = circle
    eccentricity: float
    major_axis_deg: float = Field(ge=0, le=180)  # key attribution feature
    mean_sigma0_db: float
    contrast_db: float  # background mean - slick mean
    edge_gradient: float
    cnn_oil_prob: float = Field(ge=0, le=1)
    cnn_lookalike_prob: float = Field(ge=0, le=1)
    baseline_anomaly_z: float
    wind_gate_pass: bool
    confidence: float = Field(ge=0, le=1)  # fused, calibrated


class AisPoint(_Base):
    mmsi: int
    t: datetime
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    sog: float | None = None
    cog: float | None = None
    heading: float | None = None


class SarShip(_Base):
    target_id: str
    scene_id: str
    lat: float
    lon: float
    est_length_m: float
    peak_sigma0_db: float


class AisTrack(_Base):
    mmsi: int
    imo: int | None = None
    name: str | None = None
    ship_type: int | None = None  # numeric AIS code
    flag_mid: int | None = None  # MMSI MID prefix
    declared_length_m: float | None = None
    points: list[AisPoint] = Field(default_factory=list)  # sorted by t
    gaps: list[tuple[datetime, datetime]] = Field(default_factory=list)


class TrustScore(_Base):
    mmsi: int
    scene_id: str
    score: float = Field(ge=0, le=1)  # 1 = fully credible
    trust_flags: list[str] = Field(default_factory=list)
    behaviour_flags: list[str] = Field(default_factory=list)
    sar_corroborated: bool = False
    classification: Literal["MATCHED", "DARK", "PHANTOM", "IDENTITY_MISMATCH"]


class ReleaseHypothesis(_Base):
    mmsi: int
    release_at: datetime
    release_lat: float
    release_lon: float
    simulated_polygon: dict[str, Any]
    iou: float
    centroid_offset_km: float
    orientation_delta_deg: float
    area_ratio: float
    geometric_score: float


class Suspect(_Base):
    mmsi: int
    name: str | None = None
    best_hypothesis: ReleaseHypothesis
    likelihood: float  # normalised across candidates
    trust_prior: float
    behaviour_prior: float
    type_risk_prior: float
    gap_coincidence: bool
    prior_offences: int
    posterior: float  # final ranking value
    rationale: list[str] = Field(default_factory=list)


class SlickAge(_Base):
    incident_id: str
    estimated_range_hours: tuple[float, float]  # across top-3 hypotheses
    fay_spreading_estimate_hours: float | None = None
    agreement: bool


class Scenario(_Base):
    """Synthetic ground truth ONLY.

    No pipeline module other than tests may read this. tests/test_attribution.py
    enforces it by scanning src/ for reads of ground_truth.json.
    """

    scenario_id: str
    culprit_mmsi: int
    true_release_at: datetime
    true_release_lat: float
    true_release_lon: float
    acquisition_at: datetime
    aoi_bounds: tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat


# ==========================================================================
# PART B — additions this build needed
#
# Not in section 4. Kept separate so the contract boundary stays visible. If any
# of these should be promoted into the contract, that is a decision to make
# explicitly rather than by drift.
# ==========================================================================


class Flag(_Base):
    """Structured form of the trust/behaviour flag strings in TrustScore.

    Section 4 types those as list[str]. Carrying severity and location as well
    lets the dossier group flags by seriousness and put them on a map; `code`
    alone is what serialises back into the contract's list[str].
    """

    code: str
    severity: Literal["INFO", "WARN", "CRITICAL"]
    detail: str
    at: datetime | None = None
    lat: float | None = None
    lon: float | None = None


class AisGap(_Base):
    """Expanded form of the (start, end) tuple in AisTrack.gaps.

    Position at both ends is needed to answer 'where did it go dark', which the
    bare timestamp pair cannot.
    """

    start_t: datetime
    end_t: datetime
    duration_min: float
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float


class EnvField(_Base):
    """Gridded environment. EnvSample is a point; drift needs a field.

    Vectors are 'flowing toward', not the meteorological 'coming from' used by
    EnvSample.wind_dir_deg.
    """

    lats: list[float]
    lons: list[float]
    times: list[datetime]
    wind_u: list[list[float]]
    wind_v: list[list[float]]
    curr_u: list[list[float]]
    curr_v: list[list[float]]


class EnvSummary(_Base):
    """AOI-mean conditions at acquisition, for the dashboard and dossier."""

    mean_wind_speed_ms: float
    mean_wind_dir_deg: float
    mean_current_speed_ms: float
    mean_current_dir_deg: float
    wind_gate_pass: bool


class FunnelCounts(_Base):
    """CLAUDE.md 5.5 requires the funnel be reported, but does not type it."""

    total_in_scene: int
    in_envelope: int
    scored: int
    ranked: int


class ForecastPolygon(_Base):
    """Layer 8 output. Section 5.6 specifies the behaviour, not the shape."""

    horizon_hours: int  # 24, 48, 72
    geometry: dict[str, Any]
    uncertainty_cone: dict[str, Any]
    area_km2: float
    coastline_intersects: bool
    coastline_eta: datetime | None = None
    affected_shoreline_km: float = 0.0


class Incident(_Base):
    """The artifacts/<incident_id>/incident.json envelope."""

    incident_id: str
    acquisition_at: datetime
    aoi_bounds: tuple[float, float, float, float]
    synthetic: bool = False
    observed_slick: SlickDetection
    origin_envelope: dict[str, Any]
    env_summary: EnvSummary
    funnel: FunnelCounts
    suspects: list[Suspect] = Field(default_factory=list)
    slick_age: SlickAge


class CoastguardStation(_Base):
    """An Indian Coast Guard establishment on the alert distribution list.

    `nosdcp_tier_min` is the lowest NOS-DCP tier at which this unit is addressed:
    a district station is on every alert, Coast Guard HQ only on Tier III.
    """

    id: str
    name: str
    role: Literal["NATIONAL_HQ", "REGION_HQ", "DISTRICT_HQ", "STATION"]
    region: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    mrcc: bool = False
    nosdcp_tier_min: int = Field(ge=1, le=3, default=1)


class AlertRecipient(CoastguardStation):
    """A station once it has been addressed: range, bearing and what it is asked to do."""

    distance_km: float
    bearing_deg: float = Field(ge=0, lt=360)
    action: str


class Alert(_Base):
    """Layer 11 output — the artifacts/<incident_id>/alert.json envelope.

    Severity follows NOS-DCP tiering (I local / II regional / III national)
    because the tier is what determines the distribution list. `transmitted` is
    always False: this build composes and addresses alerts, it does not send them.
    """

    alert_id: str
    incident_id: str
    issued_at: datetime
    acquisition_at: datetime
    severity: Literal["ADVISORY", "ALERT", "EMERGENCY"]
    nosdcp_tier: int = Field(ge=1, le=3)
    tier_label: str
    tier_reasons: list[str]
    position: dict[str, Any]
    slick: dict[str, Any]
    shore: dict[str, Any]
    top_suspect: dict[str, Any] | None = None
    recipients: list[AlertRecipient]
    message: str
    attachment: str
    legal_basis: str
    transmitted: bool = False
    transmission_note: str
    station_source: str
