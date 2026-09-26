"""EOS-04 L2B ARD ingest — the two things that must never be wrong.

A calibration constant off by a few dB, or a timestamp off by 5h30m, produces
output that looks entirely reasonable and is entirely wrong. Neither would be
caught by eye in a dossier. So both are verified against the product's own
independent statements, and those verifications are tested here.

Most tests build a synthetic mini-product so the suite runs anywhere. The tests
marked `real_product` additionally run against the genuine NRSC scene when it
is present under data/raw/eos04 (1.1 GB, gitignored, see CLAUDE.md 7).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from samudra.ingest import eos04

REAL_PRODUCT = Path("data/raw/eos04")

# Values copied verbatim from the real product's BAND_META.txt so the synthetic
# fixture is radiometrically identical to a genuine one.
CAL_HH = 69.259
INCIDENCE = 37.85365


def _band_meta(**overrides) -> str:
    base = {
        "ProductID": "2651503181",
        "OTSProductID": "E04_SAR_MRS_05JUL2026_TEST",
        "SatID": "EOS-04",
        "Sensor": "SAR",
        "GenAgency": "NRSC",
        "ProductType": "L2B-ARD-PRODUCT",
        "SceneStartTime": "05-JUL-2026 01:17:42.641000000",
        "SceneCenterTime": "05-JUL-2026 01:17:53.956000000",
        "SceneEndTime": "05-JUL-2026 01:18:05.271000000",
        "SceneCenterLat": "22.445728",
        "SceneCenterLon": "69.362416",
        "SunElevationAtCenter": "6.63773",
        "IncidenceAngle": str(INCIDENCE),
        "OutputPixelSpacing": "18.00",
        "NoPixels": "64",
        "NoScans": "48",
        "Node": "DESCENDING",
        "ImagingMode": "MRS ",
        "TxRxPol1": "HH",
        "TxRxPol2": "HV",
        "Calibration_Constant_HH": str(CAL_HH),
        "Calibration_Constant_HV": str(CAL_HH),
        "Calibration_Constant_Gamma0_HH": "68.233",
        "Calibration_Constant_Gamma0_HV": "68.233",
        "Calibration_Constant_Beta0_HH": "67.140",
        "Calibration_Constant_Beta0_HV": "67.140",
        "Image_Noise_Bias_HH": "14768.912",
        "Image_Noise_Bias_HV": "10883.416",
        "RTC_Apply_Flag": "1",
        "DEMSource": "COPERNICUS30",
    }
    base.update({k: str(v) for k, v in overrides.items()})
    return "\n".join(f"{k}={v}" for k, v in base.items())


@pytest.fixture
def mini_product(tmp_path: Path) -> Path:
    """A 64x48 EOS-04 product on the real product's UTM grid."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    d = tmp_path / "E04_SAR_MRS_TEST"
    (d / "scene_HH").mkdir(parents=True)
    (d / "scene_HV").mkdir(parents=True)
    (d / "BAND_META.txt").write_text(_band_meta(), encoding="utf-8")

    transform = from_origin(442800.0, 2575800.0, 18.0, 18.0)
    profile = dict(driver="GTiff", dtype="uint16", count=1, width=64, height=48,
                   crs="EPSG:32642", transform=transform, nodata=0)

    rng = np.random.default_rng(0)
    for pol, level in (("HH", 231.0), ("HV", 70.0)):
        dn = rng.gamma(4.0, level / 4.0, size=(48, 64)).astype(np.uint16)
        dn[:, :4] = 0  # a nodata margin, as real products carry
        with rasterio.open(d / f"scene_{pol}" / f"imagery_{pol}.tif", "w",
                           **profile) as ds:
            ds.write(dn, 1)

    mask = np.full((48, 64), eos04.MASK_VALID, dtype=np.uint16)
    mask[:, :4] = 0
    with rasterio.open(d / "E04_TEST_mask.tif", "w", **profile) as ds:
        ds.write(mask, 1)
    return d


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_band_meta_parses_into_a_scene(mini_product: Path):
    s = eos04.open_product(mini_product)
    assert s.satellite == "EOS-04"
    assert s.sensor == "SAR"
    assert s.polarisations == ("HH", "HV")
    assert s.pixel_spacing_m == 18.0
    assert s.rtc_applied is True
    assert s.cal_db["HH"] == pytest.approx(CAL_HH)


def test_open_product_accepts_the_parent_of_one_unpacked_product(mini_product: Path):
    """Users unzip into a folder; pointing at that folder must work."""
    s = eos04.open_product(mini_product.parent)
    assert s.product_dir == mini_product


def test_scene_centre_time_is_parsed_as_utc(mini_product: Path):
    s = eos04.open_product(mini_product)
    assert s.acquired_at == datetime(2026, 7, 5, 1, 17, 53, 956000,
                                     tzinfo=timezone.utc)
    assert s.acquired_start < s.acquired_at < s.acquired_end


# ---------------------------------------------------------------------------
# Calibration: the equation is confirmed by the product, not assumed
# ---------------------------------------------------------------------------

def test_calibration_identities_hold_for_a_genuine_constant_triple(mini_product: Path):
    s = eos04.open_product(mini_product)
    v = eos04.verify_calibration_constants(s)
    assert v["pass"], v
    assert v["equation"] == "sigma0_dB = 20*log10(DN) - Calibration_Constant"
    # Both identities, both polarisations.
    assert len(v["checks"]) == 4
    assert all(c["residual_db"] < eos04.CAL_IDENTITY_TOL_DB for c in v["checks"])


def test_a_corrupted_calibration_constant_is_caught(tmp_path: Path):
    """The failure this check exists for: a constant wrong by a few dB.

    Nothing downstream would notice; every sigma0 would simply be offset, and a
    slick contrast of 8 dB would read as 5 dB. The gamma0/beta0 identities break
    immediately, so the product is refused instead.
    """
    d = tmp_path / "bad"
    d.mkdir()
    (d / "BAND_META.txt").write_text(
        _band_meta(Calibration_Constant_HH=CAL_HH + 3.0), encoding="utf-8")
    s = eos04.open_product(d)
    v = eos04.verify_calibration_constants(s)
    assert not v["pass"]
    failed = [c for c in v["checks"] if not c["pass"]]
    assert {c["polarisation"] for c in failed} == {"HH"}
    assert "do not calibrate" in v["detail"].lower()


def test_dn_to_sigma0_db_matches_the_published_equation():
    db = eos04.dn_to_sigma0_db(np.array([[231.0]]), CAL_HH)
    assert db[0, 0] == pytest.approx(20 * math.log10(231.0) - CAL_HH, abs=1e-4)


def test_zero_dn_is_nodata_not_minus_infinity():
    """DN 0 means "not observed". Log of it is -inf, which would poison stats."""
    db = eos04.dn_to_sigma0_db(np.array([[0.0, 231.0]]), CAL_HH)
    assert np.isnan(db[0, 0])
    assert np.isfinite(db[0, 1])


def test_noise_subtraction_is_off_by_default_and_only_brightens(mini_product: Path):
    s = eos04.open_product(mini_product)
    plain, _, _, m_plain = eos04.read_sigma0_db(s, "HH")
    subbed, _, _, m_sub = eos04.read_sigma0_db(s, "HH", noise_subtract=True)
    assert m_plain["noise_subtracted"] is False
    assert m_sub["noise_subtracted"] is True
    both = np.isfinite(plain) & np.isfinite(subbed)
    # Removing an additive noise power can only lower the intensity, so the
    # subtracted product is never brighter than the plain one.
    assert np.all(subbed[both] <= plain[both] + 1e-4)


def test_nesz_is_derived_from_the_published_noise_bias(mini_product: Path):
    s = eos04.open_product(mini_product)
    assert s.nesz_db["HH"] == pytest.approx(
        10 * math.log10(14768.912) - CAL_HH, abs=1e-6)


# ---------------------------------------------------------------------------
# Timestamp: the timezone is established by physics
# ---------------------------------------------------------------------------

def test_timezone_is_confirmed_utc_against_the_published_sun_elevation(
        mini_product: Path):
    s = eos04.open_product(mini_product)
    v = eos04.verify_acquisition_time_utc(s)
    assert v["pass"], v
    assert v["residual_if_utc_deg"] < eos04.SUN_ELEV_TOL_DEG
    # IST would put this dawn acquisition in the middle of the night.
    assert v["modelled_if_ist_deg"] < -30
    assert v["residual_if_ist_deg"] > 40


def test_an_ist_timestamp_would_be_rejected(tmp_path: Path):
    """If NRSC ever shipped local time, the drift hindcast must not run.

    A 5h30m error in acquisition time shifts every back-projected release
    position by tens of kilometres, which is enough to put a different ship at
    the top of the ranking.
    """
    d = tmp_path / "ist"
    d.mkdir()
    # The same instant expressed in IST, with the sun elevation left as published.
    (d / "BAND_META.txt").write_text(
        _band_meta(SceneCenterTime="05-JUL-2026 06:47:53.956000000"),
        encoding="utf-8")
    s = eos04.open_product(d)
    v = eos04.verify_acquisition_time_utc(s)
    assert not v["pass"]
    assert "do not use this timestamp" in v["conclusion"].lower()


def test_solar_elevation_model_is_sane_at_known_points():
    # Local solar noon at the equator on the equinox: sun overhead.
    e = eos04.solar_elevation_deg(0.0, 0.0, datetime(2026, 3, 21, 12, 0,
                                                    tzinfo=timezone.utc))
    assert e > 85
    # Local midnight, same place: sun far below the horizon.
    e = eos04.solar_elevation_deg(0.0, 0.0, datetime(2026, 3, 21, 0, 0,
                                                    tzinfo=timezone.utc))
    assert e < -85


# ---------------------------------------------------------------------------
# Geometry and output
# ---------------------------------------------------------------------------

def test_footprint_is_wgs84_degrees_not_utm_metres(mini_product: Path):
    """CLAUDE.md 8: all geometry leaves in EPSG:4326.

    The product grid is UTM, so its bounds are metres in the hundreds of
    thousands. Emitting those as a GeoJSON polygon is the bug this guards.
    """
    import rasterio

    s = eos04.open_product(mini_product)
    with rasterio.open(s.band_path("HH")) as ds:
        fp = eos04.footprint_wgs84(ds.transform, ds.crs, (ds.height, ds.width))
    lons = [c[0] for c in fp["coordinates"][0]]
    lats = [c[1] for c in fp["coordinates"][0]]
    assert all(-180 <= v <= 180 for v in lons)
    assert all(-90 <= v <= 90 for v in lats)
    # And it lands in the Gulf of Kutch, where the fixture says it is.
    assert 68.0 < min(lons) < 71.0
    assert 21.0 < min(lats) < 24.0


def test_product_mask_excludes_nodata_from_the_calibrated_raster(mini_product: Path):
    s = eos04.open_product(mini_product)
    db, _, _, meta = eos04.read_sigma0_db(s, "HH")
    # The fixture zeroes the first four columns in both the image and the mask.
    assert np.all(np.isnan(db[:, :4]))
    assert meta["valid_fraction"] == pytest.approx(60 / 64, abs=0.02)


def test_write_sigma0_geotiff_round_trips_and_is_tagged_calibrated(
        mini_product: Path, tmp_path: Path):
    import rasterio

    s = eos04.open_product(mini_product)
    out = tmp_path / "cal.tif"
    r = eos04.write_sigma0_geotiff(s, out, "HH")

    with rasterio.open(out) as ds:
        assert ds.dtypes[0] == "float32"
        assert ds.crs.to_epsg() == 32642, "native UTM grid must be preserved"
        assert ds.tags()["SAMUDRA_CALIBRATED"] == "true"
        assert ds.tags()["SAMUDRA_ACQUIRED_AT"].startswith("2026-07-05T01:17")
        band = ds.read(1)

    expected, _, _, _ = eos04.read_sigma0_db(s, "HH")
    both = np.isfinite(band) & np.isfinite(expected)
    assert both.any()
    assert np.allclose(band[both], expected[both], atol=1e-4)

    side = json.loads(Path(r["sidecar"]).read_text())
    assert side["acquisition_time_known"] is True
    assert side["verification"]["pass"] is True
    assert side["calibrated"] is True
    assert "bhoonidhi" in side["source_url"]


def test_operator_aoi_is_recorded_as_an_operator_input(mini_product: Path,
                                                       tmp_path: Path):
    """Land exclusion here is a declaration, and must be labelled as one."""
    import rasterio

    s = eos04.open_product(mini_product)
    with rasterio.open(s.band_path("HH")) as ds:
        fp = eos04.footprint_wgs84(ds.transform, ds.crs, (ds.height, ds.width))
    lons = [c[0] for c in fp["coordinates"][0]]
    lats = [c[1] for c in fp["coordinates"][0]]
    pad_x = (max(lons) - min(lons)) / 4
    pad_y = (max(lats) - min(lats)) / 4
    inner = (min(lons) + pad_x, min(lats) + pad_y,
             max(lons) - pad_x, max(lats) - pad_y)

    out = tmp_path / "aoi.tif"
    r = eos04.write_sigma0_geotiff(s, out, "HH", aoi_lonlat=inner)
    assert r["calibration"]["aoi_is_operator_input"] is True
    assert r["calibration"]["shape"][0] < s.height
    side = json.loads(Path(r["sidecar"]).read_text())
    assert "OPERATOR INPUT" in side["provenance_note"]


def test_aoi_outside_the_footprint_raises_rather_than_returning_empty(
        mini_product: Path):
    s = eos04.open_product(mini_product)
    with pytest.raises(ValueError, match="does not overlap"):
        eos04.read_sigma0_db(s, "HH", aoi_lonlat=(0.0, 0.0, 1.0, 1.0))


def test_scene_derived_land_mask_is_offered_but_labelled_non_authoritative():
    """It exists as a secondary net; the docstring must keep saying so."""
    doc = eos04.scene_derived_land_mask.__doc__ or ""
    assert "NOT authoritative" in doc
    assert "operator input" in doc


# ---------------------------------------------------------------------------
# Inference reads calibrated products as absolute, chips as relative
# ---------------------------------------------------------------------------

def test_inference_uses_absolute_radiometry_for_a_calibrated_product(
        mini_product: Path, tmp_path: Path):
    from samudra.satellite import inference

    s = eos04.open_product(mini_product)
    out = tmp_path / "cal.tif"
    eos04.write_sigma0_geotiff(s, out, "HH")

    db, valid, radio = inference.load_scene_db(out)
    assert radio["radiometry"] == "absolute"
    assert radio["units"] == "sigma0 dB"
    assert not valid[:, :4].any(), "nodata margin must not be marked valid"
    assert np.isfinite(db).all(), "NaN must be filled before the network"
    # Filled pixels take the scene median, so they are ordinary sea, not a
    # fabricated dark anomaly.
    assert db[:, :4] == pytest.approx(np.median(db[valid]))


def test_inference_falls_back_to_relative_stretch_for_an_uncalibrated_chip(
        tmp_path: Path):
    from PIL import Image

    from samudra.satellite import inference

    p = tmp_path / "chip.png"
    Image.fromarray(np.full((32, 32), 120, dtype=np.uint8)).save(p)
    db, valid, radio = inference.load_scene_db(p)
    assert radio["radiometry"] == "relative"
    assert "not physical" in radio["detail"]
    assert valid.all()


# ---------------------------------------------------------------------------
# Regression: a projected drop-in must not be read as degrees
# ---------------------------------------------------------------------------

def test_projected_dropin_bounds_are_converted_to_degrees(mini_product: Path,
                                                          tmp_path: Path):
    """UTM bounds are metres. Read as degrees they put the scene at lat 2.5e6.

    Every downstream distance — envelope radius, centroid offset, coastline ETA
    — would then be computed against a position off the planet, silently.
    """
    from samudra.satellite import sources

    s = eos04.open_product(mini_product)
    incoming = tmp_path / "incoming"
    eos04.write_sigma0_geotiff(s, incoming / "eos04_test_hh.tif", "HH")

    obs = sources.list_dropin(incoming)
    assert len(obs) == 1
    o = obs[0]
    assert o.georeference == "product"
    assert 21.0 < o.anchor_lat < 24.0, f"anchor_lat was {o.anchor_lat}"
    assert 68.0 < o.anchor_lon < 71.0, f"anchor_lon was {o.anchor_lon}"
    assert o.ground_sample_m == pytest.approx(18.0)
    assert o.acquisition_time_known is True


# ---------------------------------------------------------------------------
# The genuine NRSC product, when it is on this machine
# ---------------------------------------------------------------------------

real_product = pytest.mark.skipif(
    not REAL_PRODUCT.is_dir(),
    reason="real EOS-04 product not present (1.1 GB, gitignored)")


@real_product
def test_real_product_passes_both_provenance_checks():
    s = eos04.open_product(REAL_PRODUCT)
    v = eos04.verify(s)
    assert v["pass"], v
    assert s.satellite == "EOS-04"
    assert s.raw["ProductType"] == "L2B-ARD-PRODUCT", (
        "a 17-day INDIA-MOSAIC-ARD composite cannot date a transient slick")


@real_product
def test_real_product_ocean_backscatter_is_physically_plausible():
    """A calibration error of a few dB would show up here and nowhere else.

    Open ocean C-band HH at ~38 degrees incidence sits between roughly -25 dB
    (near-calm) and -10 dB (gale). A constant offset large enough to matter
    pushes the median outside that band.
    """
    s = eos04.open_product(REAL_PRODUCT)
    db, _, _, meta = eos04.read_sigma0_db(s, "HH", aoi_lonlat=(68.55, 21.80,
                                                               69.05, 22.20))
    med = meta["sigma0_db_percentiles"]["50"]
    assert -25.0 < med < -10.0, f"ocean median {med} dB is not a sea surface"


# ---------------------------------------------------------------------------
# Contrast: the number that lets an investigator overrule a confident model
# ---------------------------------------------------------------------------

def test_local_background_ring_excludes_every_detection_not_just_its_own():
    """A neighbouring slick must not be allowed to darken the background.

    If the ring included other detected regions, two nearby dark features would
    each measure the other as "background", and both would report a contrast far
    smaller than they really have — the failure mode that hides a real spill
    sitting next to a second one.
    """
    from samudra.satellite.pipeline import _local_background

    detected = np.zeros((40, 40), dtype=bool)
    mine = np.zeros((40, 40), dtype=bool)
    mine[18:22, 8:12] = True          # the region being measured
    neighbour = np.zeros((40, 40), dtype=bool)
    neighbour[18:22, 20:24] = True    # a second detection, inside the pad
    detected |= mine | neighbour

    ring = _local_background(mine, detected, pad=15)
    assert ring.any()
    assert not (ring & mine).any(), "ring must exclude the region itself"
    assert not (ring & neighbour).any(), "ring must exclude other detections"
    # It is genuinely local: nothing far outside the padded box.
    assert not ring[:2, :].any()


def test_local_background_is_empty_for_an_empty_region():
    from samudra.satellite.pipeline import _local_background

    empty = np.zeros((10, 10), dtype=bool)
    assert not _local_background(empty, empty).any()
