"""Export the investigator view as a static site that needs no Python server.

    python scripts/export_static.py            # writes site/
    python scripts/export_static.py --out dist

Every file is produced by calling the same functions the API serves
(samudra.api), so the hosted page shows exactly what /investigate shows live.
Nothing is computed here that the API does not compute. The page switches to
these files when window.SAMUDRA_STATIC is set (see web/investigate/app.js).

What changes in a snapshot: the evidence dossier is generated once, at export
time, rather than on each click, and the page says so.

The output is plain files, so it deploys to Vercel, a Hugging Face Static
Space, GitHub Pages or Netlify unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _dump(path: Path, obj) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, default=str, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")
    return len(text)


async def _body(resp) -> bytes:
    return b"".join([c async for c in resp.body_iterator])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="site")
    a = ap.parse_args()

    os.chdir(REPO)                     # the API resolves artifacts/ and web/ relative to here
    from samudra import api

    out = (REPO / a.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    data = out / "data"
    data.mkdir(parents=True)

    incidents = [i for i in api.list_incidents() if i.get("ready")]
    if not incidents:
        raise SystemExit("no analysed incidents under artifacts/ — nothing to export")
    _dump(data / "incidents.json", incidents)
    _dump(data / "scoring.json", api.get_scoring_config())

    bounds = []
    for inc in incidents:
        iid = inc["incident_id"]
        d = data / iid
        full = api.get_incident(iid)
        _dump(d / "incident.json", full)
        _dump(d / "tracks.json", api.get_tracks(iid))
        _dump(d / "alert.json", api.get_alert(iid))
        for s in full["suspects"]:
            _dump(d / f"replay_{s['mmsi']}.json", api.get_replay(iid, mmsi=s["mmsi"]))

        scene = api.get_scene_meta(iid)
        if scene.get("available"):
            (d / "scene.png").write_bytes(asyncio.run(_body(api.get_scene_png(iid))))
            scene["image_url"] = f"data/{iid}/scene.png"
        _dump(d / "scene.json", scene)

        pdf = Path(api.get_dossier(iid).path)
        shutil.copyfile(pdf, d / f"dossier_{iid}.pdf")

        if full.get("aoi_bounds"):
            bounds.append(full["aoi_bounds"])
        print(f"  {iid}: {len(full['suspects'])} candidates, scene={'yes' if scene.get('available') else 'no'}")

    # One coastline covering every case; the page asks for its case's AOI + 8 deg.
    w = min(b[0] for b in bounds) - 8; s = min(b[1] for b in bounds) - 8
    e = max(b[2] for b in bounds) + 8; n = max(b[3] for b in bounds) + 8
    _dump(data / "coastline.json", api.basemap_coastline(min_lon=w, min_lat=s, max_lon=e, max_lat=n))

    # Page, styles, script, Leaflet and the India boundary, with paths made relative.
    web = REPO / "web"
    for f in ("app.js", "style.css"):
        shutil.copyfile(web / "investigate" / f, out / f)
    shutil.copytree(web / "vendor", out / "vendor")
    shutil.copyfile(web / "india_boundary.json", out / "india_boundary.json")

    stamp = datetime.now(UTC).strftime("%d %b %Y %H:%M UTC")
    html = (web / "investigate" / "index.html").read_text(encoding="utf-8")
    html = html.replace("/static/vendor/", "vendor/")
    html = re.sub(r"/static/investigate/", "", html)
    flag = (f'<script>window.SAMUDRA_STATIC=true;'
            f'window.SAMUDRA_EXPORTED_AT={json.dumps(stamp)};</script>\n')
    html = html.replace('<script src="app.js', flag + '<script src="app.js', 1)
    if "SAMUDRA_STATIC" not in html:
        raise SystemExit("could not insert the static flag — index.html changed shape")
    (out / "index.html").write_text(html, encoding="utf-8")

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"wrote {out}  ({total / 1e6:.1f} MB, {sum(1 for p in out.rglob('*') if p.is_file())} files)")


if __name__ == "__main__":
    main()
