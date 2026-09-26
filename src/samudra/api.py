"""Investigator dashboard API — CLAUDE.md layer 10.

Reads artifacts/<incident_id>/ from disk. Runs no analysis of its own: everything
it serves was produced by the pipeline and written to the artifacts directory, so
what the dashboard shows and what the dossier records cannot diverge.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from samudra.attribution import drift
from samudra.timeutil import epoch_seconds

ROOT = Path("artifacts")
WEB = Path("web")

app = FastAPI(title="SAMUDRA", description="Maritime oil spill attribution")


def _incident_dir(incident_id: str) -> Path:
    # Never let a path component escape the artifacts directory.
    if "/" in incident_id or "\\" in incident_id or incident_id.startswith("."):
        raise HTTPException(400, f"invalid incident id: {incident_id!r}")
    d = ROOT / incident_id
    if not d.is_dir():
        raise HTTPException(404, f"no such incident: {incident_id}")
    return d


def _load_incident(incident_id: str) -> dict:
    d = _incident_dir(incident_id)
    f = d / "incident.json"
    if not f.exists():
        raise HTTPException(
            409,
            f"{incident_id} has no incident.json — run "
            f"'python -m samudra.attribution --incident {incident_id}' first",
        )
    return json.loads(f.read_text())


@app.get("/api/monitor/status")
def monitor_status() -> dict:
    """What the continuous monitor last did. Reads state; never starts one.

    `last_check_at` and `last_new_data_at` are separate fields and the UI must
    keep them separate: a service that checked 30 seconds ago and last saw a
    satellite six days ago is the NORMAL state for 5-11 day revisit, and
    collapsing the two into one "last updated" would read as a live feed.
    """
    from samudra.monitor import read_status

    return read_status()


@app.post("/api/monitor/check")
def monitor_check(process: bool = True) -> dict:
    """Run one monitoring cycle now, on demand.

    Exists so the console has a button and a judge can see the loop execute
    rather than take a status file's word for it. It is the same `scan_once`
    the daemon calls — not a second, friendlier implementation.
    """
    from samudra.monitor import scan_once

    return scan_once(process=process)


@app.get("/api/basemap/coastline")
def basemap_coastline(
    min_lon: float = 60.0, min_lat: float = 0.0,
    max_lon: float = 100.0, max_lat: float = 40.0,
    simplify_deg: float = 0.004,
) -> dict:
    """Real coastline as GeoJSON, so the map draws land without tile servers.

    The dashboards use Esri raster tiles for their basemap, which means a
    machine with no route to arcgisonline gets polygons floating on an empty
    black rectangle — measured here, the tile request returns nothing at all.
    That reads as a broken map rather than a deliberate one, and it is the state
    a demo machine behind a restrictive network will be in.

    Natural Earth is already on disk for the shoreline-impact ETA, so serving it
    as vectors costs one endpoint and removes the external dependency for land.
    It is a coastline, not a basemap: no labels, no bathymetry, no roads.
    Returns an empty FeatureCollection when no Natural Earth file is present,
    rather than substituting the west-coast-only fallback, which would draw a
    confidently wrong shore on the east coast.
    """
    from samudra.impact.coastline import COASTLINE_PATHS, load_coastline

    if not any(p.exists() for p in COASTLINE_PATHS):
        return {"type": "FeatureCollection", "features": [],
                "source": "none on disk — run scripts/fetch_coastline.py"}

    from shapely.geometry import box, mapping

    coast = load_coastline()
    clipped = coast["geometry"].intersection(
        box(min_lon, min_lat, max_lon, max_lat))
    if clipped.is_empty:
        return {"type": "FeatureCollection", "features": [],
                "source": coast["source"]}
    if simplify_deg > 0:
        clipped = clipped.simplify(simplify_deg, preserve_topology=False)
    return {
        "type": "FeatureCollection",
        "source": coast["source"],
        "features": [{"type": "Feature", "properties": {},
                      "geometry": mapping(clipped)}],
    }


@app.get("/api/incidents")
def list_incidents() -> list[dict]:
    if not ROOT.is_dir():
        return []
    out = []
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir():
            continue
        f = d / "incident.json"
        if not f.exists():
            out.append({"incident_id": d.name, "ready": False})
            continue
        inc = json.loads(f.read_text())
        out.append(
            {
                "incident_id": inc["incident_id"],
                "ready": True,
                "acquisition_at": inc["acquisition_at"],
                "synthetic": inc.get("synthetic", False),
                "area_km2": inc["observed_slick"]["area_km2"],
                "suspects": len(inc["suspects"]),
                "top_mmsi": inc["suspects"][0]["mmsi"] if inc["suspects"] else None,
            }
        )
    return out


@app.get("/api/incident/{incident_id}")
def get_incident(incident_id: str) -> dict:
    inc = _load_incident(incident_id)
    d = _incident_dir(incident_id)
    extra = {}
    for name, key in (
        ("forecast_24h.geojson", "forecast_24h"),
        ("forecast_48h.geojson", "forecast_48h"),
        ("forecast_72h.geojson", "forecast_72h"),
    ):
        p = d / name
        if p.exists():
            extra[key] = json.loads(p.read_text())
    return {**inc, "forecasts": extra}


@app.get("/api/incident/{incident_id}/tracks")
def get_tracks(incident_id: str, max_points: int = 240) -> dict:
    """AIS tracks as GeoJSON LineStrings, tagged by role for map colouring."""
    inc = _load_incident(incident_id)
    d = _incident_dir(incident_id)

    ranked = {s["mmsi"]: s for s in inc["suspects"]}
    envelope = set(inc["funnel"].get("in_envelope_mmsis", []))
    in_scene = set(inc["funnel"].get("in_scene_mmsis", []))

    df = pd.read_parquet(d / "ais.parquet").sort_values(["mmsi", "t"])
    ts = epoch_seconds(df.t)

    features = []
    for mmsi, idx in df.groupby("mmsi", sort=False).indices.items():
        g = df.iloc[idx]
        mmsi = int(mmsi)
        if mmsi in ranked:
            role, rank = "suspect", ranked[mmsi]["rank"]
        elif mmsi in envelope:
            role, rank = "envelope", None
        elif mmsi in in_scene:
            role, rank = "filtered", None
        else:
            role, rank = "outside", None

        # Decimate for the browser; a full track is a few thousand points.
        step = max(1, len(g) // max_points)
        sub = g.iloc[::step]
        coords = [[float(a), float(b)] for a, b in zip(sub.lon, sub.lat)]
        if len(coords) < 2:
            continue
        # One epoch-second per retained vertex. The dashboard timeline needs to
        # place each vessel at an arbitrary instant, which a bare LineString
        # cannot answer. Additive: consumers that ignore `t` are unaffected.
        t_s = [float(v) for v in np.asarray(ts)[idx][::step]]

        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "mmsi": mmsi,
                    "role": role,
                    "rank": rank,
                    "t": t_s,
                    "vessel_name": (
                        None if pd.isna(g.vessel_name.iloc[0]) else str(g.vessel_name.iloc[0])
                    ),
                    "vessel_type": (
                        None if pd.isna(g.vessel_type.iloc[0]) else str(g.vessel_type.iloc[0])
                    ),
                    "posterior": ranked[mmsi]["posterior"] if mmsi in ranked else None,
                },
            }
        )

    order = {"filtered": 0, "outside": 0, "envelope": 1, "suspect": 2}
    features.sort(key=lambda f: order.get(f["properties"]["role"], 0))
    return {"type": "FeatureCollection", "features": features}


@app.get("/api/incident/{incident_id}/replay")
def get_replay(incident_id: str, n_frames: int = 40) -> dict:
    """Particle positions per timestep for the winning hypothesis."""
    inc = _load_incident(incident_id)
    d = _incident_dir(incident_id)
    if not inc["suspects"]:
        raise HTTPException(409, "no suspects to replay")

    top = inc["suspects"][0]
    b = top["best_hypothesis"]
    env = drift.EnvField.load(d / "env.npz")

    df = pd.read_parquet(d / "ais.parquet")
    g = df[df.mmsi == top["mmsi"]].sort_values("t")
    track = np.column_stack([epoch_seconds(g.t), g.lat.to_numpy(), g.lon.to_numpy()])

    frames = drift.replay_forward(
        b["release_lat"], b["release_lon"],
        datetime.fromisoformat(b["release_at"]),
        datetime.fromisoformat(inc["acquisition_at"]),
        env, track=track, n_frames=n_frames, seed=5,
    )
    return {
        "incident_id": incident_id,
        "mmsi": top["mmsi"],
        "vessel_name": top["vessel_name"],
        "release_at": b["release_at"],
        "acquisition_at": inc["acquisition_at"],
        "frames": frames,
    }


@app.get("/api/incident/{incident_id}/dossier")
def get_dossier(incident_id: str):
    """Generate the evidence dossier and return it as a download.

    Regenerated on request rather than served from cache, so the audit entry and
    chain hash printed inside always describe the artifacts as they are now.
    """
    from samudra.evidence.report import build

    _load_incident(incident_id)  # 404/409 before doing the work
    pdf = build(incident_id, ROOT)
    return FileResponse(
        pdf, media_type="application/pdf", filename=pdf.name,
        headers={"Content-Disposition": f'attachment; filename="{pdf.name}"'},
    )


@app.get("/api/incident/{incident_id}/env")
def get_env(incident_id: str, grid: int = 24, frames: int = 25) -> dict:
    """The gridded wind and current field, decimated for the browser.

    The dashboard animates drift particles on this, so it needs the field over
    time, not just at acquisition. Full resolution is 153 x 48 x 48 x 4 floats —
    about 5 MB of JSON — so both axes are decimated and values are rounded to
    centimetres per second, which is well below the field's own accuracy.
    """
    d = _incident_dir(incident_id)
    f = d / "env.npz"
    if not f.exists():
        raise HTTPException(404, f"{incident_id} has no env.npz")
    z = np.load(f)

    lats, lons, times = z["lats"], z["lons"], z["times"]
    si = max(1, len(lats) // max(2, grid))
    sj = max(1, len(lons) // max(2, grid))
    st = max(1, len(times) // max(2, frames))

    def pack(name: str) -> list[list[list[float]]]:
        a = z[name][::st, ::si, ::sj]
        return np.round(a.astype(float), 3).tolist()

    return {
        "incident_id": incident_id,
        "lats": np.round(lats[::si], 4).tolist(),
        "lons": np.round(lons[::sj], 4).tolist(),
        "times": [
            datetime.fromtimestamp(t, tz=UTC).isoformat() for t in times[::st]
        ],
        "wind_u": pack("wind_u"),
        "wind_v": pack("wind_v"),
        "curr_u": pack("curr_u"),
        "curr_v": pack("curr_v"),
        "units": "m/s, vectors flowing TOWARD (not meteorological 'from')",
    }


@app.get("/api/incident/{incident_id}/alert")
def get_alert(incident_id: str) -> dict:
    """Layer 11 — compose the coastguard alert. Composes only; sends nothing."""
    from samudra.dissemination import alert as alert_mod

    _load_incident(incident_id)  # 404/409 before doing the work
    a = alert_mod.build(incident_id, ROOT)
    return {**a, "stations": alert_mod.stations_geojson(a)}


@app.get("/api/coastguard")
def get_coastguard() -> dict:
    """Indian Coast Guard establishments as GeoJSON, for the map and for export."""
    from samudra.dissemination import alert as alert_mod

    return alert_mod.stations_geojson()


# --------------------------------------------------------------------------
# Satellite operations — layer 1 ingest + layer 3 inference, with provenance.
#
# These endpoints never fabricate an acquisition time, never return imagery for
# a mode that has no data, and never read a label source. See
# samudra/satellite/sources.py and inference.py for the rules.
# --------------------------------------------------------------------------


@app.get("/api/satellite/sources")
def satellite_sources(replay_limit: int = 40) -> dict:
    from samudra.satellite import sources

    return sources.list_all(replay_limit=replay_limit)


@app.get("/api/satellite/live")
def satellite_live() -> dict:
    from samudra.satellite import sources

    return sources.live_status()


@app.post("/api/satellite/check")
@app.get("/api/satellite/check")
def satellite_check() -> dict:
    """AUTO MONITOR's one honest action: ask the source, report what it said."""
    from samudra.satellite import sources

    return sources.check_for_new_data()


@app.get("/api/satellite/model")
def satellite_model() -> dict:
    from samudra.satellite import inference

    return inference.checkpoint_status()


@app.get("/api/satellite/observation/{observation_id}")
def satellite_observation(observation_id: str) -> dict:
    from samudra.satellite import sources

    o = sources.get(observation_id)
    if o is None:
        raise HTTPException(404, f"no such observation: {observation_id}")
    return o.to_dict()


@app.get("/api/satellite/observation/{observation_id}/image")
def satellite_image(observation_id: str, kind: str = "raw"):
    """The actual pixels: the input chip, the predicted mask, or the overlay.

    `raw` is served so a reviewer can see the image the model was given, and
    `mask`/`overlay` so they can see what it produced. Nothing is cached between
    calls — `mask` re-runs inference.
    """
    import io

    from PIL import Image

    from samudra.satellite import inference, sources

    o = sources.get(observation_id)
    if o is None:
        raise HTTPException(404, f"no such observation: {observation_id}")

    grey = inference.load_grey(o.raster_path)
    if kind == "raw":
        img = Image.fromarray(grey).convert("RGB")
    else:
        r = inference.run(o.raster_path)
        mask = r["mask"]
        if kind == "mask":
            img = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
        elif kind == "overlay":
            rgb = np.stack([grey] * 3, axis=-1).astype(np.float32)
            rgb[mask] = 0.45 * rgb[mask] + 0.55 * np.array([255.0, 107.0, 74.0])
            img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
        else:
            raise HTTPException(400, "kind must be raw, mask or overlay")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    from fastapi.responses import StreamingResponse

    return StreamingResponse(buf, media_type="image/png",
                             headers={"Cache-Control": "no-store"})


@app.get("/api/satellite/observation/{observation_id}/infer")
def satellite_infer(observation_id: str, threshold: float | None = None) -> dict:
    """Run the model. Statistics only — no attribution, nothing written."""
    from samudra.satellite import inference, sources

    o = sources.get(observation_id)
    if o is None:
        raise HTTPException(404, f"no such observation: {observation_id}")
    r = inference.run(o.raster_path, threshold=threshold)
    return {
        "observation": o.to_dict(),
        "threshold": r["threshold"],
        "device": r["device"],
        "anomaly_pixel_fraction": r["anomaly_pixel_fraction"],
        "mean_confidence_in_mask": r["mean_confidence_in_mask"],
        "max_probability": r["max_probability"],
        "regions": r["regions"],
        "stages": r["stages"],
        "class_label": r["class_label"],
        "class_caveat": r["class_caveat"],
        "checkpoint": {k: v for k, v in r["checkpoint_meta"].items()
                       if isinstance(v, (str, int, float, bool, type(None)))},
        "ground_truth_used": False,
    }


@app.post("/api/satellite/observation/{observation_id}/process")
@app.get("/api/satellite/observation/{observation_id}/process")
def satellite_process(
    observation_id: str,
    base_incident: str = "demo-001",
    anchor_lat: float | None = None,
    anchor_lon: float | None = None,
    gsd: float = 30.0,
    threshold: float | None = None,
) -> dict:
    """Full chain: image -> model -> geometry -> drift -> AIS -> ranked candidates."""
    from samudra.satellite import pipeline, sources

    o = sources.get(observation_id)
    if o is None:
        raise HTTPException(404, f"no such observation: {observation_id}")
    try:
        out = pipeline.process(
            o, base_incident=base_incident, anchor_lat=anchor_lat,
            anchor_lon=anchor_lon, ground_sample_m=gsd, threshold=threshold,
            root=ROOT,
        )
    except RuntimeError as exc:           # model found nothing — a real outcome
        raise HTTPException(422, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    return json.loads(json.dumps(out, default=str))


@app.get("/api/audit")
def get_audit() -> dict:
    """Verify the hash chain over all recorded artifacts."""
    from samudra.evidence import integrity

    return integrity.verify(root=ROOT)


@app.get("/")
def index():
    f = WEB / "index.html"
    if not f.exists():
        raise HTTPException(404, "web/index.html not found")
    return FileResponse(f)


@app.get("/ops")
def ops_console():
    """The operations console: top-nav pages, satellite ingest, layers off by default.

    A third page rather than a rewrite of the other two. /  and /app are the
    demos that already work; breaking a working demo to reach a better one is a
    bad trade the week before a review.
    """
    f = WEB / "ops.html"
    if not f.exists():
        raise HTTPException(404, "web/ops.html not found")
    return FileResponse(f)


@app.get("/app")
def dashboard():
    """The unified dashboard: ISRO sensor stack, drift timeline, alerting.

    Served on its own route so the original single-page dashboard at / keeps
    working untouched while this one is being rehearsed against.
    """
    f = WEB / "dashboard.html"
    if not f.exists():
        raise HTTPException(404, "web/dashboard.html not found")
    return FileResponse(f)


if WEB.is_dir():
    app.mount("/static", StaticFiles(directory=WEB), name="static")


def main() -> None:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="Serve the SAMUDRA dashboard.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
