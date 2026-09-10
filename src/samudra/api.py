"""Investigator dashboard API — CLAUDE.md layer 10.

Reads artifacts/<incident_id>/ from disk. Runs no analysis of its own: everything
it serves was produced by the pipeline and written to the artifacts directory, so
what the dashboard shows and what the dossier records cannot diverge.
"""

from __future__ import annotations

import json
from datetime import datetime
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

        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "mmsi": mmsi,
                    "role": role,
                    "rank": rank,
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


@app.get("/")
def index():
    f = WEB / "index.html"
    if not f.exists():
        raise HTTPException(404, "web/index.html not found")
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
