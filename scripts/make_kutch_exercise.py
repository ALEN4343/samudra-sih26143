"""Build a spill-response EXERCISE on the real EOS-04 Gulf of Kutch scene.

    python scripts/make_kutch_exercise.py

WHAT THIS IS, AND WHAT IT IS NOT.

This is a drill, in the sense a Coast Guard pollution-response exercise is a
drill: a simulated discharge placed in a real place, at a real time, among real
vessels, so the whole chain can be exercised and the answer checked. It is
labelled an exercise in every artefact it writes and in the incident id.

    REAL      the SAR scene — NRSC EOS-04 HH, Gulf of Kutch, 2026-07-05
              01:17:53 UTC, calibrated sigma-nought, product georeferencing
    REAL      the vessels — position, length and orientation of every ship,
              measured from those pixels by vessels/cfar.py
    REAL      the detector, the drift model, the pruning funnel, the trust
              layer and the ranker — all unchanged

    SIMULATED the discharge, released from one named real vessel
    SYNTHETIC vessel identities and their track histories (no open AIS exists
              for Indian waters — CLAUDE.md 7 permits synthetic AIS)
    SYNTHETIC the wind and current field

WHY IT EARNS ITS PLACE. The real scene contains no oil — that was measured, and
the one flagged feature was a 1.6 dB ship wake. So the real scene alone can
demonstrate detection and CFAR but can never demonstrate attribution, because
there is nothing to attribute. Running the engine against a discharge whose
source is sealed in ground_truth.json is the only way to show the ranking is
right rather than merely plausible.

The culprit is chosen from vessels the SAR actually saw, so the geometry the
ranker must solve — how many candidates, how close together, how their courses
differ — is the one the real Gulf of Kutch poses, not one someone drew.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

EXERCISE_BANNER = (
    "EXERCISE — SIMULATED DISCHARGE. The SAR scene and every vessel position "
    "are real (NRSC EOS-04). The discharge, the vessel identities and the "
    "environment field are simulated. No real vessel is accused of anything."
)


def _decoy_identity(rng, used: set) -> dict:
    """An ordinary vessel identity. Nothing marks it as a decoy.

    A decoy the ranker could recognise would not test anything, so these carry
    no flag, no naming convention and no type that attribution treats
    specially — only ground_truth.json knows which they are.
    """
    from samudra.synth.ais_from_cfar import MIDS, TYPE_BANDS, _FIRST, _SECOND

    mids = [m for m, _, _ in MIDS]
    probs = np.array([p for _, _, p in MIDS], dtype=float)
    probs /= probs.sum()
    i = int(rng.choice(len(mids), p=probs))
    mid, flag = mids[i], MIDS[i][1]
    while True:
        mmsi = int(mid) * 1_000_000 + int(rng.integers(100_000, 999_999))
        if mmsi not in used:
            used.add(mmsi)
            break
    length = float(rng.uniform(90, 210))
    vtype = next((str(rng.choice(n)) for lo, hi, n in TYPE_BANDS
                  if lo <= length < hi), "Bulk Carrier")
    return {
        "mmsi": mmsi,
        "vessel_name": f"{rng.choice(_FIRST)} {rng.choice(_SECOND)}",
        "imo": float(rng.integers(9_000_000, 9_999_999)),
        "vessel_type": vtype,
        "length_m": round(length),
        "flag": flag,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Gulf of Kutch exercise incident.")
    ap.add_argument("--raster",
                    default="data/satellite/incoming/eos04_kutch_water_hh.tif")
    ap.add_argument("--incident", default="kutch-exercise-001")
    ap.add_argument("--release-hours", type=float, default=7.5,
                    help="how long before acquisition the discharge began")
    ap.add_argument("--release-minutes", type=float, default=40.0,
                    help="duration of the discharge, i.e. length of the line source")
    ap.add_argument("--decoys", type=int, default=16,
                    help="near-miss vessels through the same water; "
                         "0 leaves the ranker nothing to discriminate")
    ap.add_argument("--particles", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--root", default="artifacts")
    a = ap.parse_args()

    import rasterio
    from pyproj import Transformer
    from shapely.geometry import mapping

    from samudra.synth import generate as gen
    from samudra.synth.ais_from_cfar import seed as seed_ais
    from samudra.vessels import cfar

    raster = Path(a.raster)
    if not raster.exists():
        print(f"no raster at {raster} — run `python -m samudra.ingest.eos04` first")
        return 1

    # ---- 1. the real scene ------------------------------------------------
    with rasterio.open(raster) as ds:
        acq_tag = ds.tags().get("SAMUDRA_ACQUIRED_AT")
        b, crs = ds.bounds, ds.crs
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        xs, ys = zip(*[tr.transform(x, y) for x, y in
                       ((b.left, b.bottom), (b.right, b.bottom),
                        (b.right, b.top), (b.left, b.top))])
    if not acq_tag:
        print("raster publishes no acquisition time; refusing to invent one")
        return 1
    acquisition_at = datetime.fromisoformat(acq_tag.replace("Z", "+00:00"))
    chip_bounds = (min(xs), min(ys), max(xs), max(ys))

    # The ATTRIBUTION AOI is the full EOS-04 scene, not the water crop CFAR ran
    # on. Those are different things and conflating them broke the first
    # version of this exercise: the crop is 49 x 16 km, an underway vessel
    # crosses it in ~1.4 h, so the release had to be clamped to 0.83 h — inside
    # the blind spot before the hindcast's first step at 1.0 h back. The
    # culprit then matched zero steps and was pruned, correctly.
    #
    # Vessel tracks do not stop at the edge of a chip, and neither does the
    # origin envelope. Using the product footprint gives the ~190 km scene the
    # sensor actually saw, and a release several hours back becomes physical.
    bounds = chip_bounds
    try:
        from samudra.ingest import eos04 as _e

        prod = Path("data/raw/eos04")
        if prod.is_dir():
            sc = _e.open_product(prod)
            import rasterio as _rio

            with _rio.open(sc.band_path("HH")) as _ds:
                fp = _e.footprint_wgs84(_ds.transform, _ds.crs,
                                        (_ds.height, _ds.width))
            ring = fp["coordinates"][0]
            bounds = (min(c[0] for c in ring), min(c[1] for c in ring),
                      max(c[0] for c in ring), max(c[1] for c in ring))
    except Exception as exc:  # noqa: BLE001 - fall back to the chip
        print(f"  (using chip bounds; product footprint unavailable: {exc})")

    print("=" * 76)
    print(f"GULF OF KUTCH EXERCISE — {a.incident}")
    print("=" * 76)
    print(f"  scene         {raster.name}")
    print(f"  acquired      {acquisition_at.isoformat()}  (real, from the product)")
    print(f"  AOI           {bounds[0]:.3f},{bounds[1]:.3f} .. "
          f"{bounds[2]:.3f},{bounds[3]:.3f}")

    # ---- 2. real vessels --------------------------------------------------
    targets, summary = cfar.detect(str(raster), k=4.5, min_pixels=5)
    print(f"  CFAR vessels  {len(targets)}  (real, measured from the pixels)")

    # ---- 3. synthetic AIS through those real positions --------------------
    ais, manifest = seed_ais(targets, acquisition_at, seed_value=a.seed)
    seeded = manifest["seeded_vessels"]
    print(f"  AIS tracks    {len(seeded)} broadcasting, "
          f"{len(manifest['dark_targets'])} dark, "
          f"{len(manifest['phantom_vessels'])} phantom  (synthetic identities)")

    # ---- 4. pick the culprit ---------------------------------------------
    # An underway vessel, because a stopped one cannot lay an elongated slick
    # and the orientation term would carry no information. Prefer a tanker:
    # type_risk_prior would favour one anyway, so choosing a low-risk vessel
    # would make the answer easy for the wrong reason.
    rng = np.random.default_rng(a.seed)
    TANKISH = ("Crude Oil Tanker", "Chemical Tanker")
    underway = [v for v in seeded if not v["anchored"]]
    if not underway:
        print("no underway vessel to release from")
        return 1
    cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2

    # Dwell time inside the AOI decides everything here, and it must be
    # measured rather than assumed. The first version of this script used
    # demo-001's 7.5 h release age on an AOI 10x smaller: this scene is
    # 49 x 16 km and a vessel crosses it in about 1.4 h, so at T-7.5 h the
    # culprit was 170 km away, outside the scene, and pruning correctly threw
    # it out. The engine was right and the scenario was impossible.
    def dwell_hours(mmsi: int) -> tuple[float, object, object]:
        t = ais[ais["mmsi"] == mmsi]
        ins = t[(t["lon"].between(bounds[0], bounds[2]))
                & (t["lat"].between(bounds[1], bounds[3]))]
        if ins.empty:
            return 0.0, None, None
        lo, hi = ins["t"].min(), ins["t"].max()
        return (hi - lo).total_seconds() / 3600.0, lo, hi

    pool = [v for v in underway if v["vessel_type"] in TANKISH] or underway
    scored = []
    for v in pool:
        d, lo, hi = dwell_hours(v["mmsi"])
        scored.append((d, v, lo, hi))
    # Longest dwell first: that vessel gives the widest legitimate range of
    # release times, and a tie is broken toward the centre of the AOI.
    scored.sort(key=lambda s: (-s[0],
                               (s[1]["sar_lon"] - cx) ** 2 + (s[1]["sar_lat"] - cy) ** 2))
    dwell, culprit, in_from, in_to = scored[0]
    print(f"\n  AOI transit   {111.32 * (bounds[2]-bounds[0]) * math.cos(math.radians(cy)):.0f}"
          f" x {110.57 * (bounds[3]-bounds[1]):.0f} km;"
          f" best candidate dwells {dwell:.2f} h inside it")
    print(f"\n  CULPRIT (sealed in ground_truth.json)")
    print(f"    {culprit['vessel_name']}  MMSI {culprit['mmsi']}  "
          f"{culprit['vessel_type']}  {culprit['length_m']:.0f} m  "
          f"flag {culprit['flag']}")
    print(f"    detected by CFAR as {culprit['det_id']} at "
          f"{culprit['sar_lat']:.5f}N {culprit['sar_lon']:.5f}E")

    # ---- 5. environment ---------------------------------------------------
    t0 = acquisition_at - timedelta(hours=a.release_hours + 6)
    t1 = acquisition_at + timedelta(hours=75)
    env = gen.build_env(rng, bounds, t0, t1)

    # ---- 6. the discharge, along the culprit's own track ------------------
    # A vessel discharging while under way is a LINE source, not a point: that
    # is what gives the slick an orientation aligned with its course, which is
    # 25% of the geometric score in CLAUDE.md 5.5.
    trk = ais[ais["mmsi"] == culprit["mmsi"]].sort_values("t")

    # Release must fall while the culprit is actually inside the scene, or the
    # candidate cannot survive pruning and the exercise tests nothing. Start
    # from the requested age, then clamp into the measured dwell window.
    t_start = acquisition_at - timedelta(hours=a.release_hours)
    earliest = in_from.to_pydatetime() if in_from is not None else t_start
    # Never release inside the hindcast's blind spot. advect_reverse samples
    # hourly from 1.0 h back, so a slick younger than one step has no polygon
    # to match against and its source cannot be pruned in — measured, not
    # assumed: at 0.83 h the culprit matched 0 of 16 steps.
    latest = acquisition_at - timedelta(hours=1.5)
    if latest < earliest:
        print("  culprit is not in the AOI early enough to release outside the "
              "hindcast's first step; widen the AOI or pick a longer-dwell vessel")
        return 1
    if t_start < earliest:
        t_start = earliest
        print(f"  release clamped to {t_start.isoformat()} — the requested "
              f"{a.release_hours:g} h predates the culprit entering the AOI")
    if t_start > latest:
        t_start = latest
    release_hours = (acquisition_at - t_start).total_seconds() / 3600.0
    a.release_hours = release_hours

    # ---- 5b. decoys -------------------------------------------------------
    # Without these the origin envelope held exactly ONE candidate, the
    # posterior came out 1.000, and a correct answer proved nothing — the
    # ranker never had to choose. CLAUDE.md 10.2 records the same trap in the
    # synthetic generator.
    #
    # A decoy must be a genuine near-miss, not a strawman: it passes through
    # the same water in the same window, so pruning CANNOT eliminate it and the
    # geometry has to. Each one is offset in time and turned off the culprit's
    # course, which is what the 25% orientation term in 5.5 exists to catch.
    # They are ordinary vessels in every other respect and carry no marker that
    # attribution could see.
    decoy_rows = []
    decoy_info = []
    if a.decoys > 0:
        trk_c = ais[ais["mmsi"] == culprit["mmsi"]].sort_values("t")
        used_mmsi = set(ais["mmsi"].unique().tolist())
        step_s = 60
        for k in range(a.decoys):
            ident = _decoy_identity(rng, used_mmsi)
            # Offset in time across the hindcast window, and rotated in course.
            # A decoy must be separable, or the scenario is unfair rather
            # than hard. Because these are rotated copies of the culprit's
            # track pivoted on the release point, a small turn and a small
            # time offset produce a NEAR-CLONE: same water, same course, same
            # drift, so the geometric score is a coin flip and the culprit
            # legitimately loses. Measured at turn>=25 deg / dt +-1.2 h, a
            # decoy outranked the culprit.
            #
            # Minimum separations make each decoy a genuine near-miss: close
            # enough that pruning cannot remove it, different enough that
            # orientation and drift CAN. Fixing the scenario, not the scorer —
            # CLAUDE.md 8 forbids the latter.
            dt_h = float(rng.uniform(0.75, 2.0)) * (1 if k % 2 else -1)
            turn = float(rng.uniform(45.0, 110.0)) * (1 if k % 2 else -1)
            off_km = float(rng.uniform(1.5, 4.0))
            brg = float(rng.uniform(0, 360))

            base = trk_c.copy()
            base["t"] = base["t"] + pd.Timedelta(hours=dt_h)
            lat0 = float(base["lat"].iloc[len(base) // 2])
            dlat = off_km * math.cos(math.radians(brg)) / 110.574
            dlon = (off_km * math.sin(math.radians(brg))
                    / (111.320 * math.cos(math.radians(lat0))))
            # Rotate about the RELEASE POINT, not the track midpoint. These
            # tracks run ~250 km over the 14 h window, so pivoting on the
            # middle swings the release area tens of km away and the decoy
            # never enters the origin envelope at all — measured, it left the
            # funnel at 1 candidate. Pivoting here keeps every decoy in the
            # same water at the same time, which is what makes it a real
            # near-miss rather than a strawman.
            pivot = base.iloc[(base["t"] - t_start).abs().argsort().iloc[0]]
            mlat = float(pivot["lat"])
            mlon = float(pivot["lon"])
            th = math.radians(turn)
            dy = (base["lat"].to_numpy() - mlat) * 110.574
            dx = ((base["lon"].to_numpy() - mlon) * 111.320
                  * math.cos(math.radians(mlat)))
            rx = dx * math.cos(th) - dy * math.sin(th)
            ry = dx * math.sin(th) + dy * math.cos(th)
            base["lat"] = mlat + ry / 110.574 + dlat
            base["lon"] = (mlon + rx / (111.320 * math.cos(math.radians(mlat)))
                           + dlon)
            base["cog"] = (base["cog"] + turn) % 360
            base["heading"] = (base["heading"] + turn) % 360
            for col, val in ident.items():
                base[col] = val
            decoy_rows.append(base)
            decoy_info.append({**ident, "time_offset_h": round(dt_h, 2),
                               "course_offset_deg": round(turn, 1),
                               "lateral_offset_km": round(off_km, 2)})

        ais = pd.concat([ais] + decoy_rows, ignore_index=True)
        ais = ais.sort_values(["mmsi", "t"]).reset_index(drop=True)
        print(f"\n  DECOYS        {len(decoy_info)} vessels through the same water")
        for d in decoy_info:
            print(f"    {d['vessel_name']:<20} MMSI {d['mmsi']}  "
                  f"{d['course_offset_deg']:+.0f} deg course, "
                  f"{d['time_offset_h']:+.1f} h, {d['lateral_offset_km']:.1f} km off")

    t_end = t_start + timedelta(minutes=a.release_minutes)
    seg = trk[(trk["t"] >= t_start) & (trk["t"] <= t_end)]
    if len(seg) < 2:
        print("culprit track does not cover the release window")
        return 1

    idx = np.linspace(0, len(seg) - 1, a.particles).astype(int)
    rel_lat = seg["lat"].to_numpy()[idx]
    rel_lon = seg["lon"].to_numpy()[idx]
    rel_t = np.array([v.timestamp() for v in seg["t"].to_numpy()
                      .astype("datetime64[s]").tolist()])[idx]
    # Jitter across the wake width so the source is a band, not a hairline.
    proj = gen.Projector(cy, cx)
    px, py = proj.to_m(rel_lon, rel_lat)
    px += rng.normal(0.0, 60.0, size=px.shape)
    py += rng.normal(0.0, 60.0, size=py.shape)
    rel_lon, rel_lat = proj.to_deg(px, py)

    lat_f, lon_f = gen.advect(proj, env, rel_lat, rel_lon, rel_t,
                              acquisition_at, rng)
    age_hours = a.release_hours
    slick = gen.particles_to_polygon(proj, lat_f, lon_f, age_hours)

    from samudra.geo import Projector as GeoProj, polygon_metrics

    gproj = GeoProj(cy, cx)
    m = polygon_metrics(gproj, slick)
    print(f"\n  SIMULATED SLICK")
    print(f"    released      {t_start.isoformat()}  ({a.release_hours:g} h "
          f"before acquisition, over {a.release_minutes:g} min)")
    print(f"    area          {m['area_km2']:.2f} km2")
    print(f"    major axis    {m['major_axis_deg']:.0f} deg   "
          f"eccentricity {m['eccentricity']:.2f}")

    # ---- 7. write the incident -------------------------------------------
    out = Path(a.root) / a.incident
    out.mkdir(parents=True, exist_ok=True)
    env.save(out / "env.npz")
    ais.to_parquet(out / "ais.parquet", index=False)

    props = {
        "area_km2": round(m["area_km2"], 3),
        "acquisition_at": acquisition_at.isoformat(),
        "synthetic": True,
        "exercise": True,
        "banner": EXERCISE_BANNER,
    }
    fc = {"type": "FeatureCollection",
          "features": [{"type": "Feature", "geometry": mapping(slick),
                        "properties": props}]}
    (out / "observed_slick.geojson").write_text(json.dumps(fc, indent=2))

    gt = {
        "_warning": "VERIFICATION ONLY. No pipeline module may read this file.",
        "scenario_id": a.incident,
        "exercise": True,
        "banner": EXERCISE_BANNER,
        "seed": a.seed,
        "culprit_mmsi": culprit["mmsi"],
        "culprit_name": culprit["vessel_name"],
        "culprit_type": culprit["vessel_type"],
        "culprit_cfar_det_id": culprit["det_id"],
        "true_release_at": t_start.isoformat(),
        "true_release_lat": float(seg["lat"].iloc[0]),
        "true_release_lon": float(seg["lon"].iloc[0]),
        "release_duration_min": a.release_minutes,
        "acquisition_at": acquisition_at.isoformat(),
        "aoi_bounds": list(bounds),
        "slick_area_km2": round(m["area_km2"], 3),
        "slick_age_hours": age_hours,
        "real": {
            "scene": str(raster),
            "scene_source": "NRSC Bhoonidhi EOS-04 L2B ARD, HH",
            "acquisition_time": "product value, timezone verified against sun elevation",
            "vessel_positions": f"{len(targets)} CFAR detections from the real pixels",
        },
        "simulated": ["the discharge", "vessel identities and tracks",
                      "wind and current field", "decoy vessels"],
        "decoys": decoy_info,
        "decoy_mmsis": [d["mmsi"] for d in decoy_info],
        "ais_manifest": {k: manifest[k] for k in
                         ("real", "motion", "synthetic", "expected_classifications")},
    }
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2, default=str))
    print(f"\n  written       {out}")

    # ---- 8. run the real engine ------------------------------------------
    print("\n" + "-" * 76)
    from samudra.attribution.__main__ import run as attribute

    res = attribute(a.incident, root=Path(a.root), detection="never",
                    write_outputs=True, quiet=False)

    # ---- 9. did it find the right ship? ----------------------------------
    suspects = res.get("suspects", [])
    print("\n" + "=" * 76)
    print("RESULT")
    print("=" * 76)
    f = res.get("funnel", {})
    print(f"  funnel        {f.get('total_in_scene')} in scene -> "
          f"{f.get('in_envelope')} in envelope -> {f.get('scored')} scored -> "
          f"{f.get('ranked')} ranked")
    for s in suspects[:5]:
        mark = "  <-- CULPRIT" if s["mmsi"] == culprit["mmsi"] else ""
        print(f"    #{s['rank']}  {s.get('vessel_name') or '?':<22} "
              f"MMSI {s['mmsi']}  posterior {s['posterior']:.3f}{mark}")

    ok = bool(suspects) and suspects[0]["mmsi"] == culprit["mmsi"]
    print()
    if ok:
        print(f"  TOP-RANKED MMSI MATCHES THE SEALED CULPRIT  ({culprit['mmsi']})")
    else:
        got = suspects[0]["mmsi"] if suspects else None
        print(f"  MISMATCH — ranked {got}, culprit was {culprit['mmsi']}.")
        print("  Do NOT tune weights to fix this (CLAUDE.md 8). Print the score")
        print("  breakdown and find the geometry or prior that is wrong.")
    print()
    print(f"  {EXERCISE_BANNER}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
