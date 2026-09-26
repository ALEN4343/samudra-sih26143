"""Satellite detection -> the existing SAMUDRA pipeline. CLAUDE.md layers 3-7.

This module deliberately implements almost nothing. Its whole job is to turn a
model prediction on a satellite image into the artifacts the EXISTING pipeline
already consumes, and then call that pipeline. There is no second drift engine,
no second pruning stage and no second ranker here; `attribution.__main__.run()`
does all of it, exactly as it does for a synthetic incident.

What it writes into a fresh incident directory:

    detected_slicks.geojson   the model's polygons, with their measured
                              geometry — this is what attribution consumes
    observed_slick.geojson    the same polygon, because run() reads the
                              acquisition time from it
    detection_meta.json       which checkpoint, which threshold, representative
    satellite_observation.json  full provenance of the input image
    env.npz, ais.parquet      COPIED from a base incident (see below)

THE TWO THINGS THAT ARE NOT SATELLITE-DERIVED, stated here and surfaced in the
UI rather than buried:

1. POSITION. A research SAR chip has no CRS. To hindcast a slick you need to
   know where it is, so the operator supplies an anchor and the geometry is
   built around it at a stated ground sample distance. Recorded as
   `georeference: "operator"`. A dropped-in GeoTIFF with a real transform is
   used instead and recorded as `"product"`.
2. TIME, ENVIRONMENT AND TRAFFIC. Drift needs a wind/current field and
   attribution needs AIS, at the time of acquisition. A chip with no timestamp
   and no position has neither, so the context of a base incident is reused and
   labelled `context_source`. The DETECTION is real; the water it is placed in
   is the demo AOI's.

That distinction is the honest answer to "is this a real satellite demo?" — the
image and the prediction are real, the georeferencing and the environment are
declared context. Claiming otherwise would be the fabrication this build exists
to avoid.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon, mapping

from samudra.geo import Projector, polygon_metrics
from samudra.satellite import inference as sat_inference
from samudra.satellite.sources import SatelliteObservation

ARTIFACTS = Path("artifacts")


def _contour_polygon(labelled: np.ndarray, region_id: int) -> np.ndarray:
    """Outline of one labelled region as an ordered pixel ring.

    A convex hull of the region's pixels: the downstream geometry the attribution
    layer uses is second-moment based (major axis, eccentricity, area), and a
    hull preserves all three while producing a clean simple polygon. A marching
    -squares contour would add vertices without adding information those terms
    can use.
    """
    ys, xs = np.nonzero(labelled == region_id)
    pts = np.column_stack([xs, ys]).astype(float)
    if len(pts) < 3:
        return pts
    from shapely.geometry import MultiPoint

    hull = MultiPoint([tuple(p) for p in pts]).convex_hull
    if hull.geom_type != "Polygon":
        return pts
    return np.asarray(hull.exterior.coords)


def _local_background(inside: np.ndarray, detected: np.ndarray,
                      pad: int = 24) -> np.ndarray:
    """Pixels in a ring around a region, for a local background estimate.

    Local rather than scene-wide on purpose: sea backscatter varies across a
    scene with wind and current, so comparing a slick against the whole-scene
    mean would read that gradient as contrast. `detected` is every flagged
    region, so a neighbouring detection cannot darken the background its
    neighbour is being measured against.
    """
    ys, xs = np.nonzero(inside)
    if ys.size == 0:
        return np.zeros_like(inside)
    h, w = inside.shape
    r0, r1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    c0, c1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    ring = np.zeros_like(inside)
    ring[r0:r1, c0:c1] = True
    return ring & ~detected


def _product_pixel_to_lonlat(raster_path):
    """Exact pixel -> (lon, lat) using the product's own affine transform.

    Returns None when the raster carries no CRS, in which case the caller falls
    back to the anchor-and-ground-sample approximation.

    Worth the extra path: the approximation treats the image as axis-aligned
    about its centre, which ignores UTM grid convergence. At 22.6 N, half a
    degree off the zone's central meridian, that is about 0.2 degrees of
    rotation — roughly 190 m of positional error at the edge of a 50 km AOI.
    Attribution scores a centroid offset against a 10 km scale, so 190 m does
    not change a ranking, but a dossier that reports coordinates to five
    decimals should not be wrong in the third.
    """
    try:
        import rasterio
        from rasterio.transform import xy as _xy

        with rasterio.open(raster_path) as ds:
            if ds.crs is None:
                return None
            transform, crs = ds.transform, ds.crs
    except Exception:  # noqa: BLE001 - not a raster, or rasterio unavailable
        return None

    tr = None
    if crs.to_epsg() != 4326:
        from pyproj import Transformer

        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    def to_lonlat(cols: np.ndarray, rows: np.ndarray):
        xs, ys = _xy(transform, list(np.asarray(rows)), list(np.asarray(cols)),
                     offset="center")
        xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        return (xs, ys) if tr is None else tr.transform(xs, ys)

    return to_lonlat


def detection_to_geojson(
    result: dict,
    anchor_lat: float,
    anchor_lon: float,
    ground_sample_m: float,
    observation: SatelliteObservation,
    threshold: float,
) -> tuple[dict, list[dict]]:
    """Model mask -> GeoJSON polygons + SlickDetection-shaped records."""
    proj = Projector(anchor_lat, anchor_lon)
    labelled = result["labelled"]
    h, w = labelled.shape
    prob = result["prob"]
    exact = (_product_pixel_to_lonlat(observation.raster_path)
             if observation.georeference == "product" else None)

    feats: list[dict] = []
    records: list[dict] = []
    ids = [i for i in range(1, int(labelled.max()) + 1)]
    # Largest first: attribution takes the most confident polygon.
    ids.sort(key=lambda i: int((labelled == i).sum()), reverse=True)

    for n, rid in enumerate(ids):
        ring_px = _contour_polygon(labelled, rid)
        if len(ring_px) < 4:
            continue
        if exact is not None:
            lon, lat = exact(ring_px[:, 0], ring_px[:, 1])
        else:
            # pixel -> metres about the anchor (image y grows downward)
            x_m = (ring_px[:, 0] - w / 2.0) * ground_sample_m
            y_m = -(ring_px[:, 1] - h / 2.0) * ground_sample_m
            lon, lat = proj.to_deg(x_m, y_m)
        poly = Polygon(np.column_stack([lon, lat]))
        if not poly.is_valid or poly.area <= 0:
            continue

        m = polygon_metrics(proj, poly)
        inside = labelled == rid
        cnn_prob = float(prob[inside].mean())

        # Contrast against local background, in real dB where the product is
        # calibrated. This is the single most diagnostic number for whether a
        # dark feature is oil: a genuine slick damps capillary waves by 5-15 dB,
        # while a ship wake, a rain cell edge or a speckle fluctuation is worth
        # a fraction of one. The model does not see it — it scores texture — so
        # reporting it alongside the model's probability is what lets an
        # investigator throw out a confident-looking sub-dB detection.
        contrast_db = mean_db = None
        db_img, radiometry = result.get("db"), result.get("radiometry") or {}
        if db_img is not None and radiometry.get("radiometry") == "absolute":
            ring_i = _local_background(inside, labelled > 0, pad=24)
            if ring_i.any():
                mean_db = float(np.nanmean(db_img[inside]))
                contrast_db = float(np.nanmean(db_img[ring_i]) - mean_db)
        # Confidence is the model's own mean probability inside the region,
        # tempered by how large the region is relative to a speck. Nothing here
        # is a constant chosen to look good.
        size_term = float(min(1.0, inside.sum() / (0.002 * h * w)))
        conf = float(cnn_prob * (0.6 + 0.4 * size_term))

        rec = {
            "slick_id": f"sat-{n:03d}",
            "scene_id": observation.observation_id,
            "polygon": mapping(poly),
            "area_km2": m["area_km2"],
            "perimeter_km": m["perimeter_km"],
            "shape_complexity": m["shape_complexity"],
            "eccentricity": m["eccentricity"],
            "major_axis_deg": m["major_axis_deg"],
            "centroid_lat": m["centroid_lat"],
            "centroid_lon": m["centroid_lon"],
            "area_px": int(inside.sum()),
            "mean_sigma0_db": mean_db,
            "contrast_db": contrast_db,
            "contrast_note": (
                None if contrast_db is None else
                "Oil damps capillary waves by roughly 5-15 dB. Below about "
                "2 dB a dark feature is more consistent with a wake, a wind "
                "shadow or speckle than with a slick."),
            "cnn_oil_prob": cnn_prob,
            "confidence": conf,
            "threshold": threshold,
            "class_label": "OIL-LIKE SURFACE ANOMALY",
            "checkpoint_representative": bool(
                result["checkpoint_meta"].get("representative", False)
            ),
        }
        records.append(rec)
        feats.append({
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {k: v for k, v in rec.items() if k != "polygon"},
        })

    return {"type": "FeatureCollection", "features": feats}, records


def build_incident(
    observation: SatelliteObservation,
    result: dict,
    base_incident: str,
    anchor_lat: float,
    anchor_lon: float,
    ground_sample_m: float,
    incident_id: str | None = None,
    root: Path = ARTIFACTS,
    reference_time: str | None = None,
) -> dict[str, Any]:
    """Materialise the detection as an incident the existing pipeline can run."""
    base = root / base_incident
    if not base.is_dir():
        raise FileNotFoundError(f"base incident {base} not found")

    inc_id = incident_id or f"sat-{observation.observation_id}"
    d = root / inc_id
    d.mkdir(parents=True, exist_ok=True)

    for name in ("env.npz", "ais.parquet"):
        src = base / name
        if not src.exists():
            raise FileNotFoundError(f"{src} missing — cannot supply drift/AIS context")
        shutil.copy2(src, d / name)

    base_inc = json.loads((base / "incident.json").read_text())
    acq = reference_time or (
        observation.acquired_at if observation.acquisition_time_known
        else base_inc["acquisition_at"]
    )

    fc, records = detection_to_geojson(
        result, anchor_lat, anchor_lon, ground_sample_m, observation, result["threshold"]
    )
    if not records:
        raise RuntimeError(
            "the model found no anomaly above threshold in this observation — "
            "there is nothing to attribute. Try another acquisition or lower the "
            "threshold; do not invent a detection."
        )

    for f in fc["features"]:
        f["properties"]["acquisition_at"] = acq
    (d / "detected_slicks.geojson").write_text(json.dumps(fc, indent=2))

    # run() reads the acquisition time out of observed_slick.geojson.
    best = max(fc["features"], key=lambda f: f["properties"]["confidence"])
    (d / "observed_slick.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": best["geometry"],
            "properties": {**best["properties"], "acquisition_at": acq},
        }],
    }, indent=2))

    (d / "detection_meta.json").write_text(json.dumps({
        "source": "satellite_console",
        "representative": bool(result["checkpoint_meta"].get("representative", False)),
        "checkpoint": str(result["checkpoint_meta"].get("arch", "unknown")),
        "checkpoint_epoch": result["checkpoint_meta"].get("epoch"),
        "threshold": result["threshold"],
        "device": result["device"],
        "n_regions": len(records),
        "class_label": "OIL-LIKE SURFACE ANOMALY",
        "class_caveat": result["class_caveat"],
        "ground_truth_used": False,
        "ground_truth_note": (
            "No label source is on the inference path. The mask came from the "
            "model's output on the image's pixels."
        ),
    }, indent=2))

    (d / "satellite_observation.json").write_text(json.dumps({
        "observation": observation.to_dict(),
        "georeference": {
            "mode": observation.georeference if observation.georeference == "product"
                    else "operator",
            "anchor_lat": anchor_lat,
            "anchor_lon": anchor_lon,
            "ground_sample_m": ground_sample_m,
            "note": (
                "Taken from the product's own transform."
                if observation.georeference == "product" else
                "OPERATOR INPUT. This chip carries no CRS, so its position on "
                "the map was chosen by the operator to demonstrate downstream "
                "integration. It is not satellite-derived and must not be "
                "presented as a real spill location."
            ),
        },
        "context_source": {
            "base_incident": base_incident,
            "reference_time": acq,
            "reference_time_is_satellite_derived": bool(observation.acquisition_time_known),
            "supplies": ["env.npz (wind + current field)", "ais.parquet (vessel traffic)"],
            "note": (
                "Drift needs an environment field and attribution needs AIS, both "
                "at the time of acquisition. A chip with no timestamp and no "
                "position has neither, so the demo AOI's context is reused. The "
                "detection is real; the water it is placed in is declared context."
            ),
        },
        "inference": {
            "threshold": result["threshold"],
            "device": result["device"],
            "anomaly_pixel_fraction": result["anomaly_pixel_fraction"],
            "mean_confidence_in_mask": result["mean_confidence_in_mask"],
            "max_probability": result["max_probability"],
            "stages": result["stages"],
            "checkpoint_meta": {
                k: v for k, v in result["checkpoint_meta"].items()
                if isinstance(v, (str, int, float, bool, type(None), list))
            },
        },
        "detections": [{k: v for k, v in r.items() if k != "polygon"} for r in records],
    }, indent=2, default=str))

    return {"incident_id": inc_id, "dir": str(d), "detections": records,
            "acquisition_at": acq}


def attribute(incident_id: str, root: Path = ARTIFACTS, quiet: bool = True) -> dict:
    """Run the EXISTING attribution chain over the satellite incident."""
    from samudra.attribution.__main__ import run

    # detection="always": the polygon came from the model on a real image, which
    # is the whole point. The usual "auto" guard would swap in a synthetic slick
    # whenever the checkpoint is not marked representative, and silently
    # replacing the thing being demonstrated is worse than showing a weak result.
    return run(incident_id, root=root, detection="always", write_outputs=True, quiet=quiet)


def process(
    observation: SatelliteObservation,
    base_incident: str = "demo-001",
    anchor_lat: float | None = None,
    anchor_lon: float | None = None,
    ground_sample_m: float = 30.0,
    threshold: float | None = None,
    root: Path = ARTIFACTS,
    run_attribution: bool = True,
) -> dict[str, Any]:
    """Whole chain: image -> model -> geometry -> drift -> AIS -> ranked suspects."""
    # A LIVE catalogue entry is real metadata for a product still sitting on the
    # provider's servers. Running the model needs pixels, and there are none, so
    # this refuses with the exact reason rather than quietly falling back to an
    # archive chip and calling the result "live".
    if not observation.pixels_local or not observation.raster_path:
        size = (observation.size_bytes or 0) / 1e9
        raise RuntimeError(
            f"{observation.observation_id} is a LIVE CATALOGUE ENTRY — its "
            f"metadata is real but the image ({size:.2f} GB) has not been "
            f"downloaded, so there is nothing to run the model over. Download "
            f"the product from the provider portal and drop it into "
            f"data/satellite/incoming/ with a sidecar .json, or switch to REAL "
            f"SATELLITE REPLAY."
        )
    result = sat_inference.run(observation.raster_path, threshold=threshold)

    if observation.georeference == "product" and observation.anchor_lat is not None:
        lat, lon = observation.anchor_lat, observation.anchor_lon
        gsd = observation.ground_sample_m or ground_sample_m
    else:
        base_inc = json.loads((root / base_incident / "incident.json").read_text())
        lat = anchor_lat if anchor_lat is not None else base_inc["observed_slick"]["centroid_lat"]
        lon = anchor_lon if anchor_lon is not None else base_inc["observed_slick"]["centroid_lon"]
        gsd = ground_sample_m

    built = build_incident(
        observation, result, base_incident, lat, lon, gsd, root=root
    )
    out = {
        "observation": observation.to_dict(),
        "inference": {k: v for k, v in result.items()
                      if k not in ("prob", "mask", "grey", "labelled",
                                   "db", "valid")},
        "incident": built,
    }
    if run_attribution:
        try:
            out["attribution"] = attribute(built["incident_id"], root=root)
        except RuntimeError as exc:
            # A detection with no vessel traffic in its origin envelope is a
            # legitimate outcome, not a bug. It happens whenever a real
            # observation is processed over water and time for which no AIS has
            # been ingested — which is the normal case for a scene downloaded
            # today. Report it as an unattributed detection; the alternative,
            # widening the envelope until something falls in, is how an
            # attribution engine starts naming innocent ships.
            out["attribution"] = None
            out["attribution_unavailable"] = {
                "reason": str(exc),
                "detail": (
                    "Detection stands on its own. Attribution needs AIS "
                    "covering this footprint and acquisition time; the context "
                    f"incident '{base_incident}' does not provide it."
                ),
                "acquisition_at": observation.acquired_at,
                "context_incident": base_incident,
            }
    return out
