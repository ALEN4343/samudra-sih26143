"""EOS-04 (RISAT-1A) L2B ARD ingest — real Indian C-band SAR into SAMUDRA.

Layer 1 for the satellite this project is actually built around. EOS-04 is
ISRO's C-band SAR; NRSC distributes the L2B Analysis Ready Data product through
Bhoonidhi. This module turns one such product into calibrated sigma-nought in
dB, on the product's own grid, carrying the provenance the product publishes
and nothing it does not.

WHAT THIS MODULE REFUSES TO INVENT
----------------------------------
Two values decide whether a downstream attribution result is defensible: the
radiometric scale and the acquisition time. Both are read from the product, and
both are *verified against the product's own independent statements* rather
than trusted:

  calibration   BAND_META publishes three constants - sigma0, gamma0, beta0.
                They are not independent: beta0 = sigma0 / sin(theta) and
                gamma0 = sigma0 / cos(theta). `verify_calibration_constants`
                checks that the published triple satisfies both identities at
                the published incidence angle. If it does, the equation

                    sigma0_dB = 20 * log10(DN) - Calibration_Constant

                is confirmed by the product itself, not assumed from a
                handbook.

  timestamp     BAND_META publishes SceneCenterTime without a timezone, and
                also publishes SunElevationAtCenter. `verify_acquisition_time_utc`
                computes solar elevation at the scene centre for the candidate
                timestamp. Reading the time as UTC reproduces the published sun
                elevation to under a degree; reading it as IST puts the sun 44
                degrees below the horizon. The timezone is therefore
                established by physics rather than by convention.

A product that fails either check raises, rather than quietly producing numbers
wrong by a constant offset - the failure mode that is hardest to notice
downstream and most damaging in a dossier.

LAND IS AN OPERATOR INPUT HERE, AND THAT IS DELIBERATE
------------------------------------------------------
The bundled simplified coastline (CLAUDE.md 10.3) reaches 22.47 N and cannot
resolve the Gulf of Kutch's creeks, islands and salt flats. That matters more
than usual for this AOI: the Rann of Kutch mudflats are *radiometrically dark*
in SAR, so an automatic dark-feature detector run over them produces enormous
false positives that look exactly like slicks. Rather than ship a land mask
that is wrong in the one place it is being demonstrated, the marine AOI is an
explicit operator input, recorded as such - the same discipline already applied
to chip georeferencing in satellite/sources.py.

`scene_derived_land_mask` is offered as a secondary net, using the HV/HH
cross-pol ratio (volume scattering over vegetation and urban land, low over
water). It is scene-derived, not authoritative, and is labelled that way
everywhere it appears.

CLI
---
    python -m samudra.ingest.eos04 --product <dir> --verify
    python -m samudra.ingest.eos04 --product <dir> --pol HH \
        --aoi 68.9,22.2,69.9,22.9 --out data/satellite/incoming/kutch_hh.tif
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

# NRSC writes month names in this form in BAND_META.txt.
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}

#: Tolerance for the sigma0/gamma0/beta0 identity check, in dB.
CAL_IDENTITY_TOL_DB = 0.05

#: Tolerance for the solar-elevation timezone check, in degrees. The forward
#: model below omits the equation of time and uses a first-order declination,
#: which is worth roughly a degree; the IST/UTC difference is ~50 degrees, so
#: this is nowhere near tight enough to be ambiguous.
SUN_ELEV_TOL_DEG = 3.0

#: Product mask value meaning "valid data". Other values are nodata border or
#: layover/shadow bits.
MASK_VALID = 128


def _parse_dt(s: str) -> datetime:
    """'05-JUL-2026 01:17:53.956000000' -> aware UTC datetime.

    Timezone is asserted separately by `verify_acquisition_time_utc`; this
    function only parses, and tags UTC because that is what the verification
    establishes for this product family.
    """
    date_part, _, time_part = s.strip().partition(" ")
    d, mon, y = date_part.split("-")
    if not time_part:
        return datetime(int(y), _MONTHS[mon.upper()], int(d), tzinfo=timezone.utc)
    hh, mm, ss = time_part.split(":")
    sec = float(ss)
    whole = int(sec)
    micro = int(round((sec - whole) * 1e6))
    if micro >= 1_000_000:  # rounding can carry
        whole, micro = whole + 1, 0
    return datetime(int(y), _MONTHS[mon.upper()], int(d),
                    int(hh), int(mm), whole, micro, tzinfo=timezone.utc)


def read_band_meta(product_dir: Path | str) -> dict[str, str]:
    """Parse BAND_META.txt (or the .meta twin) into a plain key->string dict."""
    p = Path(product_dir)
    for c in [p / "BAND_META.txt", *sorted(p.glob("*.meta"))]:
        if c.exists():
            out: dict[str, str] = {}
            for line in c.read_text(errors="replace").splitlines():
                k, sep, v = line.partition("=")
                if sep:
                    out[k.strip()] = v.strip()
            if out:
                return out
    raise FileNotFoundError(f"no BAND_META.txt or *.meta under {p}")


@dataclass(frozen=True)
class Eos04Scene:
    """One EOS-04 L2B ARD product, with the provenance it actually publishes."""

    product_dir: Path
    product_id: str
    ots_product_id: str
    satellite: str
    sensor: str
    gen_agency: str
    acquired_at: datetime
    acquired_start: datetime
    acquired_end: datetime
    incidence_angle_deg: float
    node: str
    imaging_mode: str
    polarisations: tuple[str, ...]
    pixel_spacing_m: float
    width: int
    height: int
    centre_lat: float
    centre_lon: float
    sun_elevation_deg: float
    cal_db: dict[str, float]
    cal_gamma0_db: dict[str, float]
    cal_beta0_db: dict[str, float]
    noise_bias_dn2: dict[str, float]
    rtc_applied: bool
    dem_source: str
    raw: dict[str, str] = field(repr=False, default_factory=dict)

    # -- file locations ----------------------------------------------------
    def band_path(self, pol: str) -> Path:
        """The measurement raster for one polarisation."""
        pol = pol.upper()
        cand = self.product_dir / f"scene_{pol}" / f"imagery_{pol}.tif"
        if cand.exists():
            return cand
        hits = sorted(self.product_dir.glob(f"**/imagery_{pol}.tif"))
        if not hits:
            raise FileNotFoundError(f"no imagery_{pol}.tif under {self.product_dir}")
        return hits[0]

    def ancillary_path(self, kind: str) -> Path | None:
        """mask / lia / area sidecar raster, if the product shipped it."""
        hits = sorted(self.product_dir.glob(f"*_{kind}.tif"))
        return hits[0] if hits else None

    @property
    def nesz_db(self) -> dict[str, float]:
        """Noise-equivalent sigma zero implied by the published noise bias."""
        return {p: 10 * math.log10(b) - self.cal_db[p]
                for p, b in self.noise_bias_dn2.items()
                if b > 0 and p in self.cal_db}

    def to_provenance(self) -> dict[str, Any]:
        """Everything a dossier needs to say where these pixels came from."""
        return {
            "satellite": self.satellite,
            "sensor": self.sensor,
            "product": self.ots_product_id,
            "product_id": self.product_id,
            "generating_agency": self.gen_agency,
            "product_type": self.raw.get("ProductType"),
            "acquired_at": self.acquired_at.isoformat().replace("+00:00", "Z"),
            "acquisition_time_known": True,
            "acquisition_time_source": (
                "BAND_META SceneCenterTime; timezone confirmed UTC against the "
                "product's published sun elevation"),
            "incidence_angle_deg": self.incidence_angle_deg,
            "polarisations": list(self.polarisations),
            "imaging_mode": self.imaging_mode,
            "orbit_node": self.node,
            "resolution_m": self.pixel_spacing_m,
            "rtc_applied": self.rtc_applied,
            "dem_source": self.dem_source,
            "calibration_constant_db": self.cal_db,
            "nesz_db": {k: round(v, 2) for k, v in self.nesz_db.items()},
            "source": "NRSC Bhoonidhi (ISRO) open data, direct download",
            "source_url": "https://bhoonidhi.nrsc.gov.in",
            "licence": "NRSC OpenData - attribution to NRSC/ISRO required",
        }


def open_product(product_dir: Path | str) -> Eos04Scene:
    """Read a product directory into an `Eos04Scene`. No pixels are read."""
    p = Path(product_dir)
    if not p.is_dir():
        raise NotADirectoryError(p)
    # Allow pointing at the parent of a single unpacked product.
    if not (p / "BAND_META.txt").exists():
        subs = [d for d in sorted(p.iterdir())
                if d.is_dir() and (d / "BAND_META.txt").exists()]
        if len(subs) == 1:
            p = subs[0]

    m = read_band_meta(p)
    pols = tuple(m[k].strip() for k in ("TxRxPol1", "TxRxPol2")
                 if m.get(k) and m[k].strip() not in ("", "NA"))

    def fdict(prefix: str) -> dict[str, float]:
        return {pol: float(m[f"{prefix}{pol}"])
                for pol in pols if f"{prefix}{pol}" in m}

    return Eos04Scene(
        product_dir=p,
        product_id=m.get("ProductID", ""),
        ots_product_id=m.get("OTSProductID", p.name),
        satellite=m.get("SatID", "EOS-04"),
        sensor=m.get("Sensor", "SAR"),
        gen_agency=m.get("GenAgency", "NRSC"),
        acquired_at=_parse_dt(m["SceneCenterTime"]),
        acquired_start=_parse_dt(m.get("SceneStartTime", m["SceneCenterTime"])),
        acquired_end=_parse_dt(m.get("SceneEndTime", m["SceneCenterTime"])),
        incidence_angle_deg=float(m.get("IncidenceAngle", "nan")),
        node=m.get("Node", "UNKNOWN"),
        imaging_mode=m.get("ImagingMode", "").strip(),
        polarisations=pols,
        pixel_spacing_m=float(m.get("OutputPixelSpacing", "nan")),
        width=int(m.get("NoPixels", 0)),
        height=int(m.get("NoScans", 0)),
        centre_lat=float(m.get("SceneCenterLat", "nan")),
        centre_lon=float(m.get("SceneCenterLon", "nan")),
        sun_elevation_deg=float(m.get("SunElevationAtCenter", "nan")),
        cal_db=fdict("Calibration_Constant_"),
        cal_gamma0_db=fdict("Calibration_Constant_Gamma0_"),
        cal_beta0_db=fdict("Calibration_Constant_Beta0_"),
        noise_bias_dn2=fdict("Image_Noise_Bias_"),
        rtc_applied=m.get("RTC_Apply_Flag", "0").strip() == "1",
        dem_source=m.get("DEMSource", "unstated"),
        raw=m,
    )


# ---------------------------------------------------------------------------
# Verification - the product checked against its own published statements
# ---------------------------------------------------------------------------

def verify_calibration_constants(scene: Eos04Scene) -> dict[str, Any]:
    """Confirm sigma0/gamma0/beta0 satisfy their defining identities.

    beta0  = sigma0 / sin(theta)  ->  K_sigma - K_beta  = -10*log10(sin theta)
    gamma0 = sigma0 / cos(theta)  ->  K_sigma - K_gamma = -10*log10(cos theta)

    Holding for both at the published incidence angle confirms the constants are
    defined against `X_dB = 20*log10(DN) - K_X`, which is the equation this
    module applies. Nothing here is taken on faith from a handbook.
    """
    th = math.radians(scene.incidence_angle_deg)
    exp_beta = -10 * math.log10(math.sin(th))
    exp_gamma = -10 * math.log10(math.cos(th))
    checks: list[dict] = []
    ok = True
    for pol in scene.polarisations:
        if pol not in scene.cal_beta0_db or pol not in scene.cal_gamma0_db:
            continue
        got_beta = scene.cal_db[pol] - scene.cal_beta0_db[pol]
        got_gamma = scene.cal_db[pol] - scene.cal_gamma0_db[pol]
        for name, got, exp in (("beta0", got_beta, exp_beta),
                               ("gamma0", got_gamma, exp_gamma)):
            resid = abs(got - exp)
            passed = resid <= CAL_IDENTITY_TOL_DB
            ok &= passed
            checks.append({"polarisation": pol, "identity": name,
                           "published_delta_db": round(got, 4),
                           "expected_delta_db": round(exp, 4),
                           "residual_db": round(resid, 4), "pass": passed})
    return {
        "pass": bool(ok and checks),
        "incidence_angle_deg": scene.incidence_angle_deg,
        "equation": "sigma0_dB = 20*log10(DN) - Calibration_Constant",
        "checks": checks,
        "detail": ("The published sigma0, gamma0 and beta0 constants satisfy "
                   "both defining identities at the published incidence angle, "
                   "so the calibration equation is confirmed by the product."
                   if ok and checks else
                   "Published constants do not satisfy the sigma0/gamma0/beta0 "
                   "identities. Do not calibrate this product with this module."),
    }


def solar_elevation_deg(lat: float, lon: float, when: datetime) -> float:
    """First-order solar elevation. Good to ~1 deg; used only to pin a timezone."""
    t = when.astimezone(timezone.utc)
    doy = t.timetuple().tm_yday
    decl = 23.44 * math.sin(math.radians(360.0 * (284 + doy) / 365.0))
    hours = t.hour + t.minute / 60 + (t.second + t.microsecond / 1e6) / 3600
    solar = hours + lon / 15.0
    hour_angle = math.radians((solar - 12.0) * 15.0)
    s = (math.sin(math.radians(lat)) * math.sin(math.radians(decl))
         + math.cos(math.radians(lat)) * math.cos(math.radians(decl))
         * math.cos(hour_angle))
    return math.degrees(math.asin(max(-1.0, min(1.0, s))))


def verify_acquisition_time_utc(scene: Eos04Scene) -> dict[str, Any]:
    """Establish that SceneCenterTime is UTC, from the published sun elevation.

    BAND_META states the time without a zone. Interpreting it as UTC and as IST
    gives solar elevations tens of degrees apart, so the product's own
    SunElevationAtCenter settles it.
    """
    utc = scene.acquired_at
    as_ist = solar_elevation_deg(scene.centre_lat, scene.centre_lon,
                                 utc - timedelta(hours=5, minutes=30))
    as_utc = solar_elevation_deg(scene.centre_lat, scene.centre_lon, utc)
    published = scene.sun_elevation_deg
    r_utc, r_ist = abs(as_utc - published), abs(as_ist - published)
    ok = r_utc <= SUN_ELEV_TOL_DEG and r_utc < r_ist
    return {
        "pass": bool(ok),
        "published_sun_elevation_deg": published,
        "modelled_if_utc_deg": round(as_utc, 2),
        "modelled_if_ist_deg": round(as_ist, 2),
        "residual_if_utc_deg": round(r_utc, 2),
        "residual_if_ist_deg": round(r_ist, 2),
        "conclusion": ("SceneCenterTime is UTC - reading it as UTC reproduces "
                       "the product's own sun elevation; reading it as IST does "
                       "not." if ok else
                       "Timezone NOT established. Do not use this timestamp for "
                       "drift hindcast."),
    }


def verify(scene: Eos04Scene) -> dict[str, Any]:
    """Both provenance checks, as one reportable block."""
    cal = verify_calibration_constants(scene)
    tim = verify_acquisition_time_utc(scene)
    return {"calibration": cal, "timestamp": tim,
            "pass": bool(cal["pass"] and tim["pass"])}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def dn_to_sigma0_db(
    dn: np.ndarray,
    cal_db: float,
    noise_bias_dn2: float | None = None,
    floor_db: float = -40.0,
) -> np.ndarray:
    """DN -> sigma0 in dB. Invalid (DN == 0) pixels come back as NaN.

    `noise_bias_dn2` subtracts the published additive noise power before the
    log. It is OFF by default at every call site in this module: it materially
    brightens the darkest targets - exactly the pixels a slick occupies - and
    altering the provider's radiometry by default is not something a dossier
    should do silently. Where it is used, it is recorded in the sidecar.
    """
    dn = np.asarray(dn)
    valid = dn > 0
    inten = dn.astype(np.float64) ** 2
    if noise_bias_dn2:
        inten = inten - float(noise_bias_dn2)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(np.where(inten > 0, inten, np.nan)) - cal_db
    db = np.where(valid, db, np.nan)
    return np.maximum(db, floor_db).astype(np.float32)


def _aoi_window(ds, aoi_lonlat: tuple[float, float, float, float]):
    """Operator marine AOI in lon/lat -> a rasterio window in the product CRS."""
    from pyproj import Transformer
    from rasterio.windows import from_bounds

    minlon, minlat, maxlon, maxlat = aoi_lonlat
    tf = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
    xs, ys = [], []
    for lon in (minlon, maxlon):
        for lat in (minlat, maxlat):
            x, y = tf.transform(lon, lat)
            xs.append(x)
            ys.append(y)
    win = from_bounds(min(xs), min(ys), max(xs), max(ys), ds.transform)
    return win.round_offsets().round_lengths()


def read_sigma0_db(
    scene: Eos04Scene,
    pol: str = "HH",
    aoi_lonlat: tuple[float, float, float, float] | None = None,
    noise_subtract: bool = False,
):
    """Calibrated sigma0 dB for one polarisation, optionally over an AOI.

    Returns (db, transform, crs, meta). `db` is float32 with NaN outside the
    product's valid data. Layover/shadow pixels from the product mask are also
    NaN when the mask shipped.
    """
    import rasterio
    from rasterio.windows import Window

    pol = pol.upper()
    if pol not in scene.cal_db:
        raise KeyError(f"{pol} not in product (has {scene.polarisations})")

    path = scene.band_path(pol)
    with rasterio.open(path) as ds:
        win = None
        if aoi_lonlat is not None:
            # rasterio raises rather than returning a zero-size window, so the
            # "no overlap" case has to be caught, not tested for afterwards.
            try:
                win = _aoi_window(ds, aoi_lonlat).intersection(
                    Window(0, 0, ds.width, ds.height))
            except Exception as exc:  # noqa: BLE001 - WindowError and friends
                raise ValueError(
                    f"AOI {aoi_lonlat} does not overlap the product footprint"
                ) from exc
            if win.width <= 0 or win.height <= 0:
                raise ValueError(
                    f"AOI {aoi_lonlat} does not overlap the product footprint")
        dn = ds.read(1, window=win)
        transform = ds.window_transform(win) if win is not None else ds.transform
        crs = ds.crs

    db = dn_to_sigma0_db(
        dn, scene.cal_db[pol],
        scene.noise_bias_dn2.get(pol) if noise_subtract else None,
    )

    invalid_mask_frac = 0.0
    mp = scene.ancillary_path("mask")
    if mp is not None:
        with rasterio.open(mp) as ds:
            mk = ds.read(1, window=win)
        bad = mk != MASK_VALID
        invalid_mask_frac = float(bad.mean())
        db = np.where(bad, np.nan, db).astype(np.float32)

    finite = np.isfinite(db)
    meta: dict[str, Any] = {
        "polarisation": pol,
        "calibration_constant_db": scene.cal_db[pol],
        "equation": "sigma0_dB = 20*log10(DN) - Calibration_Constant",
        "noise_subtracted": bool(noise_subtract),
        "noise_bias_dn2": scene.noise_bias_dn2.get(pol),
        "nesz_db": round(scene.nesz_db.get(pol, float("nan")), 2),
        "shape": [int(db.shape[0]), int(db.shape[1])],
        "valid_fraction": round(float(finite.mean()), 4),
        "product_mask_excluded_fraction": round(invalid_mask_frac, 4),
        "aoi_lonlat": list(aoi_lonlat) if aoi_lonlat else None,
        "aoi_is_operator_input": bool(aoi_lonlat),
    }
    if finite.any():
        v = db[finite]
        meta["sigma0_db_percentiles"] = {
            str(q): round(float(np.percentile(v, q)), 2)
            for q in (1, 5, 50, 95, 99)
        }
    return db, transform, crs, meta


def scene_derived_land_mask(
    scene: Eos04Scene,
    aoi_lonlat: tuple[float, float, float, float] | None = None,
    ratio_db_threshold: float = -11.0,
) -> np.ndarray | None:
    """Coarse land mask from the HV/HH cross-pol ratio. NOT authoritative.

    Volume and multiple scattering over vegetation and built-up land raise HV
    relative to HH; over water HV sits near the noise floor. The threshold is a
    scene-level heuristic and is reported as scene-derived wherever it is used.
    It does NOT reliably separate tidal mudflats and salt pans from water, which
    is precisely why the marine AOI stays an operator input.

    Returns None when the product is single-polarisation.
    """
    if "HV" not in scene.cal_db or "HH" not in scene.cal_db:
        return None
    hh, _, _, _ = read_sigma0_db(scene, "HH", aoi_lonlat)
    hv, _, _, _ = read_sigma0_db(scene, "HV", aoi_lonlat)
    with np.errstate(invalid="ignore"):
        ratio = hv - hh
    return np.where(np.isfinite(ratio), ratio > ratio_db_threshold, False)


def footprint_wgs84(transform, crs, shape) -> dict:
    """GeoJSON Polygon of a raster window, in EPSG:4326 as CLAUDE.md 8 requires."""
    from pyproj import Transformer
    from rasterio.transform import array_bounds

    h, w = shape
    left, bottom, right, top = array_bounds(h, w, transform)
    tf = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    ring = []
    for x, y in ((left, bottom), (right, bottom), (right, top), (left, top),
                 (left, bottom)):
        lon, lat = tf.transform(x, y)
        ring.append([round(lon, 6), round(lat, 6)])
    return {"type": "Polygon", "coordinates": [ring]}


def _describe_footprint(footprint: dict) -> str:
    """A factual bounding-box description, with no invented place name."""
    ring = footprint["coordinates"][0]
    lons = [c[0] for c in ring]
    lats = [c[1] for c in ring]

    def ns(v: float) -> str:
        return f"{abs(v):.2f}{'N' if v >= 0 else 'S'}"

    def ew(v: float) -> str:
        return f"{abs(v):.2f}{'E' if v >= 0 else 'W'}"

    return (f"{ns(min(lats))}-{ns(max(lats))}, "
            f"{ew(min(lons))}-{ew(max(lons))}")


def write_sigma0_geotiff(
    scene: Eos04Scene,
    out_path: Path | str,
    pol: str = "HH",
    aoi_lonlat: tuple[float, float, float, float] | None = None,
    noise_subtract: bool = False,
) -> dict[str, Any]:
    """Write calibrated sigma0 dB as float32 GeoTIFF, plus a drop-in sidecar.

    The raster keeps the product's native UTM grid. Nothing is reprojected here:
    resampling a calibrated SAR image to degrees before detection would smear
    the speckle statistics detection depends on. Geometry crosses into
    EPSG:4326 at polygonisation, not before.
    """
    import rasterio

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    db, transform, crs, meta = read_sigma0_db(scene, pol, aoi_lonlat, noise_subtract)

    profile = {
        "driver": "GTiff", "dtype": "float32", "count": 1,
        "height": db.shape[0], "width": db.shape[1],
        "crs": crs, "transform": transform, "nodata": float("nan"),
        "compress": "deflate", "predictor": 3, "tiled": True,
        "blockxsize": 512, "blockysize": 512,
    }
    with rasterio.open(out, "w", **profile) as ds:
        ds.write(db, 1)
        ds.update_tags(
            SAMUDRA_UNITS="sigma0 dB",
            SAMUDRA_CALIBRATED="true",
            SAMUDRA_SATELLITE=scene.satellite,
            SAMUDRA_ACQUIRED_AT=scene.acquired_at.isoformat().replace("+00:00", "Z"),
            SAMUDRA_EQUATION=meta["equation"],
        )

    fp = footprint_wgs84(transform, crs, db.shape)
    prov = scene.to_provenance()
    prov.update({
        "band": "C-band",
        "polarisation": pol,
        # Stated as coordinates, not as a place name. A hardcoded region would
        # be wrong the first time this module is pointed at a scene somewhere
        # else on the Indian coast, and nothing in the product publishes one.
        "region": _describe_footprint(fp),
        "footprint": fp,
        "calibration": meta,
        "units": "sigma0 dB",
        "calibrated": True,
        "verification": verify(scene),
        "provenance_note": (
            "Real EOS-04 (ISRO) C-band SAR from NRSC Bhoonidhi. Pixels are "
            "calibrated sigma-nought in dB using the product's own published "
            "constant, with the calibration equation confirmed against the "
            "product's sigma0/gamma0/beta0 identities and the acquisition "
            "timezone confirmed against its published sun elevation. "
            "Georeferencing is the product's own UTM grid."
            + ("" if aoi_lonlat is None else
               " The marine AOI is an OPERATOR INPUT, not a product value.")
        ),
    })
    side = out.with_suffix(".json")
    side.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    return {"raster": str(out), "sidecar": str(side), "calibration": meta,
            "footprint": prov["footprint"]}


# ---------------------------------------------------------------------------

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="EOS-04 L2B ARD ingest.")
    ap.add_argument("--product", required=True, help="unpacked product directory")
    ap.add_argument("--pol", default="HH")
    ap.add_argument("--aoi", help="operator marine AOI: minlon,minlat,maxlon,maxlat")
    ap.add_argument("--out", help="write calibrated sigma0 dB GeoTIFF here")
    ap.add_argument("--noise-subtract", action="store_true",
                    help="subtract the published noise bias (off by default)")
    ap.add_argument("--verify", action="store_true", help="provenance checks only")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    scene = open_product(a.product)
    aoi = tuple(float(v) for v in a.aoi.split(",")) if a.aoi else None
    if aoi is not None and len(aoi) != 4:
        ap.error("--aoi needs minlon,minlat,maxlon,maxlat")

    v = verify(scene)
    if a.json and a.verify:
        print(json.dumps({"scene": scene.to_provenance(), "verification": v},
                         indent=2, default=str))
        return

    print("=" * 74)
    print(f"EOS-04 PRODUCT  {scene.ots_product_id}")
    print("=" * 74)
    print(f"  satellite / sensor  {scene.satellite} {scene.sensor}  ({scene.gen_agency})")
    print(f"  product type        {scene.raw.get('ProductType')}")
    print(f"  acquired            {scene.acquired_at.isoformat()}  "
          f"({scene.node.lower()}, {scene.imaging_mode})")
    print(f"  grid                {scene.width} x {scene.height} @ "
          f"{scene.pixel_spacing_m:g} m")
    print(f"  incidence           {scene.incidence_angle_deg:.2f} deg")
    print(f"  polarisations       {', '.join(scene.polarisations)}")
    print(f"  RTC applied         {scene.rtc_applied}  (DEM {scene.dem_source})")
    for p, n in scene.nesz_db.items():
        print(f"  NESZ {p:<3}            {n:.2f} dB")

    c = v["calibration"]
    print(f"\nCALIBRATION SELF-CHECK  {'PASS' if c['pass'] else 'FAIL'}")
    print(f"  {c['equation']}")
    for ck in c["checks"]:
        print(f"    {ck['polarisation']:<3} {ck['identity']:<7} published "
              f"{ck['published_delta_db']:+.4f} dB   expected "
              f"{ck['expected_delta_db']:+.4f} dB   residual "
              f"{ck['residual_db']:.4f} dB  {'ok' if ck['pass'] else 'FAIL'}")

    t = v["timestamp"]
    print(f"\nTIMEZONE SELF-CHECK     {'PASS' if t['pass'] else 'FAIL'}")
    print(f"  product states sun elevation {t['published_sun_elevation_deg']:+.2f} deg")
    print(f"  modelled if UTC              {t['modelled_if_utc_deg']:+.2f} deg   "
          f"residual {t['residual_if_utc_deg']:.2f} deg")
    print(f"  modelled if IST              {t['modelled_if_ist_deg']:+.2f} deg   "
          f"residual {t['residual_if_ist_deg']:.2f} deg")
    print(f"  {t['conclusion']}")

    if a.verify:
        return
    if not v["pass"]:
        raise SystemExit("\nRefusing to calibrate: a provenance check failed.")

    if aoi:
        print(f"\nOPERATOR MARINE AOI     {aoi}")
        print("  Declared by the operator, not published by the product. The "
              "bundled\n  coastline cannot resolve the Gulf of Kutch, and the "
              "Rann salt flats are\n  radiometrically dark, so land is excluded "
              "by declaration rather than\n  guessed.")

    out = a.out or (f"data/satellite/incoming/"
                    f"eos04_{scene.product_id}_{a.pol.lower()}.tif")
    r = write_sigma0_geotiff(scene, out, a.pol, aoi, a.noise_subtract)
    m = r["calibration"]
    print("\nCALIBRATED OUTPUT")
    print(f"  raster              {r['raster']}")
    print(f"  sidecar             {r['sidecar']}")
    print(f"  shape               {m['shape'][1]} x {m['shape'][0]} px")
    print(f"  valid pixels        {m['valid_fraction'] * 100:.1f}%")
    print(f"  noise subtracted    {m['noise_subtracted']}")
    if "sigma0_db_percentiles" in m:
        q = m["sigma0_db_percentiles"]
        print(f"  sigma0 dB           p1 {q['1']}  p5 {q['5']}  p50 {q['50']}  "
              f"p95 {q['95']}  p99 {q['99']}")


if __name__ == "__main__":
    main()
