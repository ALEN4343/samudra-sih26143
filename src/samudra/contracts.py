"""Single source of truth for every data structure crossing a module boundary.

Implements CLAUDE.md section 3. Do not change this file without asking first.

Conventions:
  - All timestamps are timezone-aware UTC.
  - All geometry is GeoJSON, EPSG:4326, longitude first.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _as_utc(v: datetime) -> datetime:
    """Attach UTC to a naive datetime, convert an aware one. Never guess local time."""
    if v.tzinfo is None:
        return v.replace(tzinfo=timezone.utc)
    return v.astimezone(timezone.utc)


# --------------------------------------------------------------------------
# Vessel and AIS
# --------------------------------------------------------------------------


class AisPoint(_Base):
    """A single AIS broadcast. Referenced by section 3, defined here."""

    mmsi: int
    t: datetime
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    sog: float
    cog: float | None = None
    heading: float | None = None

    _utc = field_validator("t")(_as_utc)


class AisGap(_Base):
    start_t: datetime
    end_t: datetime
    duration_min: float
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float

    _utc = field_validator("start_t", "end_t")(_as_utc)


class AisTrack(_Base):
    mmsi: int
    points: list[AisPoint]
    vessel_name: str | None = None
    imo: int | None = None
    vessel_type: str | None = None
    length_m: float | None = None
    flag: str | None = None
    gaps: list[AisGap] = Field(default_factory=list)


class VesselDetection(_Base):
    """A radar target from CFAR (layer 4), independent of any AIS report."""

    det_id: str
    lat: float
    lon: float
    length_m: float | None = None
    peak_intensity: float
    matched_mmsi: int | None = None


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


class SlickPolygon(_Base):
    slick_id: str
    geometry: dict[str, Any]
    area_km2: float
    perimeter_km: float
    centroid_lat: float
    centroid_lon: float
    major_axis_deg: float = Field(ge=0, le=180)
    minor_axis_m: float
    major_axis_m: float
    eccentricity: float
    shape_complexity: float
    mean_sigma0_db: float
    cnn_oil_prob: float = Field(ge=0, le=1)
    anomaly_z: float
    wind_gate_pass: bool
    confidence: float = Field(ge=0, le=1)


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


class EnvField(_Base):
    lats: list[float]
    lons: list[float]
    times: list[datetime]
    wind_u: list[list[float]]
    wind_v: list[list[float]]
    curr_u: list[list[float]]
    curr_v: list[list[float]]


class EnvSummary(_Base):
    mean_wind_speed_ms: float
    mean_wind_dir_deg: float
    mean_current_speed_ms: float
    mean_current_dir_deg: float
    wind_gate_pass: bool


# --------------------------------------------------------------------------
# Trust and behaviour
# --------------------------------------------------------------------------


class Flag(_Base):
    code: str
    severity: Literal["INFO", "WARN", "CRITICAL"]
    detail: str
    at: datetime | None = None
    lat: float | None = None
    lon: float | None = None


class TrustScore(_Base):
    mmsi: int
    score: float = Field(ge=0, le=1)
    classification: Literal["MATCHED", "DARK", "PHANTOM", "IDENTITY_MISMATCH"]
    trust_flags: list[Flag] = Field(default_factory=list)
    behaviour_flags: list[Flag] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------


class Hypothesis(_Base):
    hypothesis_id: str
    mmsi: int
    release_at: datetime
    release_lat: float
    release_lon: float
    simulated_geometry: dict[str, Any]
    iou: float
    centroid_offset_km: float
    orientation_delta_deg: float
    area_ratio: float
    score: float

    _utc = field_validator("release_at")(_as_utc)


class Suspect(_Base):
    mmsi: int
    vessel_name: str | None = None
    rank: int
    posterior: float = Field(ge=0, le=1)
    best_hypothesis: Hypothesis
    top_hypotheses: list[Hypothesis] = Field(default_factory=list)
    trust_prior: float = 1.0
    behaviour_prior: float = 1.0
    proximity_prior: float = 1.0
    rationale: list[str] = Field(default_factory=list)


class FunnelCounts(_Base):
    total_in_scene: int
    in_envelope: int
    scored: int
    ranked: int


class SlickAge(_Base):
    best_hours: float
    low_hours: float
    high_hours: float
    fay_estimate_hours: float
    agrees_with_fay: bool


# --------------------------------------------------------------------------
# Incident and forecast
# --------------------------------------------------------------------------


class Incident(_Base):
    incident_id: str
    acquisition_at: datetime
    aoi_bounds: tuple[float, float, float, float]
    observed_slick: SlickPolygon
    origin_envelope: dict[str, Any]
    env_summary: EnvSummary
    funnel: FunnelCounts
    suspects: list[Suspect] = Field(default_factory=list)
    slick_age: SlickAge | None = None

    _utc = field_validator("acquisition_at")(_as_utc)


class ForecastPolygon(_Base):
    horizon_hours: int
    geometry: dict[str, Any]
    uncertainty_cone: dict[str, Any]
    area_km2: float
    coastline_intersects: bool
    coastline_eta: datetime | None = None
    affected_shoreline_km: float = 0.0


# --------------------------------------------------------------------------
# Synthetic ground truth
# --------------------------------------------------------------------------


class Scenario(_Base):
    """Planted ground truth. Only the test suite may read this back."""

    scenario_id: str
    culprit_mmsi: int
    true_release_at: datetime
    true_release_lat: float
    true_release_lon: float
    acquisition_at: datetime
    aoi_bounds: tuple[float, float, float, float]

    _utc = field_validator("true_release_at", "acquisition_at")(_as_utc)
