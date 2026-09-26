"""Satellite observation sources — layer 1 ingest for the operations console.

THE RULE THIS MODULE EXISTS TO ENFORCE: an observation carries the provenance it
actually has, and says so where it has none.

Concretely, `acquired_at` is `None` whenever the source does not publish a
per-image acquisition time, and `acquisition_time_known` is False. It is never
filled in with "now", never with a plausible-looking constant, and never with
the file's mtime. A judge asking "is that timestamp real?" gets either a real
timestamp with a source, or "not published by this source".

Three modes, kept separate on purpose:

  LIVE_NRT      a configured, credentialed data source. Absent credentials it
                reports AUTH_REQUIRED and refuses to invent anything.
  REAL_REPLAY   genuine satellite imagery already on disk: published research
                archives, or a product you downloaded yourself and dropped in.
  SYNTHETIC     the project's own generated scenes. Labelled, never dressed up.

GEOREFERENCING is the other honesty trap and is handled the same way. A 256x256
research chip has no CRS. Placing it at a convenient spot in the Arabian Sea so
the drift model has something to chew on would be fabricating georeferencing, so
an anchor is an explicit OPERATOR INPUT, recorded as `georeference: "operator"`.
A dropped-in GeoTIFF that carries a real transform is recorded as
`georeference: "product"` and uses it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

Mode = Literal["LIVE_NRT", "REAL_REPLAY", "SYNTHETIC"]

SOS_ROOT = Path("data/raw/oilspill/sos")
BINARY_ROOT = Path("data/raw/oilspill/binary/data")
DROPIN_ROOT = Path("data/satellite/incoming")

# Environment variables that gate LIVE/NRT. Nothing is hardcoded; absence is
# reported, not worked around.
LIVE_ENV = {
    "bhoonidhi": ("BHOONIDHI_USER", "BHOONIDHI_PASS"),
    "mosdac": ("MOSDAC_USER", "MOSDAC_PASS"),
}


@dataclass
class SatelliteObservation:
    """One satellite image SAMUDRA can process, with everything known about it."""

    observation_id: str
    mode: Mode
    satellite: str
    sensor: str
    product: str
    raster_path: str

    # Provenance. `None` means "the source does not publish this", not "unknown
    # so we guessed".
    acquired_at: str | None = None
    acquisition_time_known: bool = False
    processed_at: str | None = None
    received_at: str = ""
    footprint: dict[str, Any] | None = None
    region: str | None = None
    resolution_m: float | None = None
    pixel_size: tuple[int, int] | None = None
    polarisation: str | None = None
    band: str | None = None

    source: str = ""
    source_url: str | None = None
    licence: str | None = None
    citation: str | None = None
    doi: str | None = None

    georeference: Literal["product", "operator", "none"] = "none"
    anchor_lat: float | None = None
    anchor_lon: float | None = None
    ground_sample_m: float | None = None

    checksum_sha256: str | None = None
    provenance_note: str = ""
    caveats: list[str] = field(default_factory=list)

    # A live catalogue entry is real metadata for a product whose pixels are
    # still on the provider's servers. It must never be mistaken for an image
    # this machine can run a model over, so it says which it is.
    pixels_local: bool = True
    size_bytes: int | None = None
    download_url: str | None = None
    provider: str | None = None
    online: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _size(p: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(p) as im:
            return im.size
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- REAL REPLAY

SOS_CITATION = (
    "Deep-SAR Oil Spill (SOS) dataset — Sentinel-1 (C-band) and ALOS PALSAR "
    "(L-band) chips with pixel-level oil masks, as shipped in "
    "data/raw/oilspill/sos."
)
SOS_CAVEATS = [
    "Research chips, 256x256 px, no coordinate reference system and no "
    "per-chip acquisition timestamp published with the archive.",
    "A geographic anchor, if used, is an operator input and is labelled as such.",
]

BINARY_CITATION = (
    "Blondeau-Patissier, D., Schroeder, T., Diakogiannis, F., Li, Z. (2022) "
    "CSIRO Sentinel-1 SAR image dataset of oil- and non-oil features for "
    "machine learning (Deep Learning). CSIRO Data Access Portal."
)
BINARY_CAVEATS = [
    "Per-IMAGE label only (Class_0 = no oil features incl. look-alikes, "
    "Class_1 = contains oil features) — there is no ground-truth mask, so a "
    "prediction on these chips cannot be scored pixel-wise.",
    "Collection footprint is 101.4E-154.8E, 26.1S-21.1N (Southeast Asia / "
    "Australia). These chips are NOT Indian waters.",
    "Collection window 2015-05-01 to 2022-08-31; individual acquisition times "
    "are not published per chip.",
]


def _sos_observation(path: Path, sensor_dir: str, split: str) -> SatelliteObservation:
    sat = {"sentinel": "Sentinel-1", "palsar": "ALOS PALSAR"}[sensor_dir]
    band = {"sentinel": "C-band", "palsar": "L-band"}[sensor_dir]
    return SatelliteObservation(
        observation_id=f"sos-{sensor_dir}-{split}-{path.stem}",
        mode="REAL_REPLAY",
        satellite=sat,
        sensor=f"SAR {band}",
        band=band,
        product=f"SOS/{split}/{sensor_dir}/{path.name}",
        raster_path=str(path),
        acquired_at=None,
        acquisition_time_known=False,
        received_at=_now(),
        region="not published per chip",
        pixel_size=_size(path),
        source="Deep-SAR SOS archive (local copy)",
        licence="see the dataset's own licence",
        citation=SOS_CITATION,
        georeference="none",
        checksum_sha256=_sha256(path),
        provenance_note=(
            "Real spaceborne SAR imagery from a published research archive. The "
            "archive does not publish per-chip acquisition times or geolocation, "
            "so neither is claimed here."
        ),
        caveats=list(SOS_CAVEATS),
    )


def _binary_observation(path: Path, cls: int) -> SatelliteObservation:
    return SatelliteObservation(
        observation_id=f"csiro-s1-class{cls}-{path.stem}",
        mode="REAL_REPLAY",
        satellite="Sentinel-1",
        sensor="SAR C-band",
        band="C-band",
        product=f"CSIRO-S1/Class_{cls}/{path.name}",
        raster_path=str(path),
        acquired_at=None,
        acquisition_time_known=False,
        received_at=_now(),
        region="101.4E-154.8E, 26.1S-21.1N (collection footprint)",
        pixel_size=_size(path),
        source="CSIRO Data Access Portal (local copy)",
        source_url="https://doi.org/10.25919/4v55-dn16",
        doi="10.25919/4v55-dn16",
        licence="see metadata/license.html in the dataset",
        citation=BINARY_CITATION,
        georeference="none",
        checksum_sha256=_sha256(path),
        provenance_note=(
            "Real Sentinel-1 SAR chip. Published image-level label for this chip "
            f"is Class_{cls} = {'contains oil features' if cls else 'no oil features (clean sea or look-alike)'}. "
            "That label is NOT shown to the model and is not used to produce the "
            "prediction; it is carried so a reviewer can check the answer afterwards."
        ),
        caveats=list(BINARY_CAVEATS),
    )


#: Research chips (Deep-SAR SOS, CSIRO) are OFF in the operator-facing list.
#:
#: They are Sentinel-1 (ESA) and ALOS PALSAR (JAXA) — the datasets this project
#: TRAINS on, because no labelled Indian SAR oil-spill set is published. They
#: are not observations of anything this system monitors, and listing 40 of them
#: beside one real EOS-04 scene made the acquisition list read as a foreign
#: satellite feed, which is the opposite of what SIH26143 specifies and the
#: opposite of what this build does.
#:
#: They remain reachable for model work — `list_replay(include_research=True)`,
#: or SAMUDRA_RESEARCH_CHIPS=1 — because evaluating the checkpoint on held-out
#: chips is a legitimate thing to do. It is just not an acquisition.
RESEARCH_CHIPS_ENV = "SAMUDRA_RESEARCH_CHIPS"


def research_chips_enabled() -> bool:
    return os.environ.get(RESEARCH_CHIPS_ENV, "").strip().lower() in ("1", "true", "yes")


def list_replay(limit: int = 40, kind: str = "all",
                include_research: bool | None = None) -> list[SatelliteObservation]:
    """Real SAR imagery available on this machine.

    By default this is operator-supplied products only — the drop-in folder,
    i.e. genuine Indian acquisitions you downloaded from Bhoonidhi. Training
    chips are excluded; see RESEARCH_CHIPS_ENV above for why and how to get
    them back.

    When research chips ARE enabled they deliberately draw from the SOS *test*
    split: those were held out of training, so a prediction on one is a
    prediction on an image the model has never seen. Using a training chip would
    make the demo a memory test.
    """
    out: list[SatelliteObservation] = []
    if include_research is None:
        include_research = research_chips_enabled()
    if not include_research:
        kind = "none"

    if kind in ("all", "sos"):
        for sensor_dir in ("sentinel", "palsar"):
            d = SOS_ROOT / "test" / sensor_dir / "image"
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)[
                : max(1, limit // 4)
            ]:
                out.append(_sos_observation(f, sensor_dir, "test"))

    if kind in ("all", "csiro"):
        for cls in (1, 0):
            d = BINARY_ROOT / f"Class_{cls}"
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.jpg"))[: max(1, limit // 4)]:
                out.append(_binary_observation(f, cls))

    return out[:limit]


# ------------------------------------------------------------------- DROP-IN


def list_dropin(root: Path = DROPIN_ROOT) -> list[SatelliteObservation]:
    """Products you downloaded yourself, e.g. an EOS-04 scene from Bhoonidhi.

    Put the raster in data/satellite/incoming/ with a sidecar JSON of the same
    stem carrying the metadata the provider published. Anything the sidecar does
    not state stays unknown — the loader does not invent it.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    out = []
    for f in sorted(root.iterdir()):
        if f.suffix.lower() not in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
            continue
        side = f.with_suffix(".json")
        meta: dict[str, Any] = {}
        if side.exists():
            try:
                meta = json.loads(side.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}

        geo, glat, glon, gsd = "none", None, None, None
        fp = meta.get("footprint")
        try:  # a real GeoTIFF brings its own georeferencing; use it if present
            import rasterio

            with rasterio.open(f) as ds:
                if ds.crs is not None:
                    geo = "product"
                    b = ds.bounds
                    # A projected product (EOS-04 ships UTM) has bounds in
                    # metres. Treating them as degrees put the scene at
                    # lat 2.5 million, and every downstream distance silently
                    # became meaningless. Convert, always.
                    corners = [(b.left, b.bottom), (b.right, b.bottom),
                               (b.right, b.top), (b.left, b.top)]
                    if ds.crs.to_epsg() != 4326:
                        from pyproj import Transformer

                        tr = Transformer.from_crs(ds.crs, "EPSG:4326",
                                                  always_xy=True)
                        corners = [tr.transform(x, y) for x, y in corners]
                        # Ground sample is in the product's own units (metres
                        # for a projected CRS), so it needs no conversion.
                        gsd = abs(ds.transform.a)
                    else:
                        gsd = abs(ds.transform.a) * 111320
                    lons = [c[0] for c in corners]
                    lats = [c[1] for c in corners]
                    glat = (min(lats) + max(lats)) / 2
                    glon = (min(lons) + max(lons)) / 2
                    if fp is None:
                        fp = {"type": "Polygon",
                              "coordinates": [[list(c) for c in corners]
                                              + [list(corners[0])]]}
        except Exception:  # noqa: BLE001 - not a GeoTIFF, or rasterio unavailable
            pass

        acq = meta.get("acquired_at")
        out.append(SatelliteObservation(
            observation_id=f"dropin-{f.stem}",
            mode="REAL_REPLAY",
            satellite=meta.get("satellite", "UNDECLARED — set it in the sidecar JSON"),
            sensor=meta.get("sensor", "UNDECLARED"),
            band=meta.get("band"),
            polarisation=meta.get("polarisation"),
            product=meta.get("product", f.name),
            raster_path=str(f),
            acquired_at=acq,
            acquisition_time_known=bool(acq),
            processed_at=meta.get("processed_at"),
            received_at=_now(),
            footprint=fp,
            region=meta.get("region"),
            resolution_m=meta.get("resolution_m"),
            pixel_size=_size(f),
            source=meta.get("source", "operator drop-in, source not declared"),
            source_url=meta.get("source_url"),
            licence=meta.get("licence"),
            citation=meta.get("citation"),
            doi=meta.get("doi"),
            georeference=geo,
            anchor_lat=glat,
            anchor_lon=glon,
            ground_sample_m=gsd or meta.get("resolution_m"),
            checksum_sha256=_sha256(f),
            provenance_note=meta.get(
                "provenance_note",
                "Operator-supplied product. SAMUDRA reports exactly what the "
                "sidecar JSON declares and marks the rest unknown.",
            ),
            caveats=(["No sidecar JSON found — satellite, sensor and acquisition "
                      "time are undeclared for this file."] if not side.exists() else []),
        ))
    return out


# --------------------------------------------------------------- SYNTHETIC


def list_synthetic(root: Path = Path("artifacts")) -> list[SatelliteObservation]:
    out = []
    for d in sorted(Path(root).iterdir()) if Path(root).is_dir() else []:
        scene = d / "scene.tif"
        if not d.is_dir() or not scene.exists():
            continue
        inc = d / "incident.json"
        acq = None
        if inc.exists():
            try:
                acq = json.loads(inc.read_text()).get("acquisition_at")
            except json.JSONDecodeError:
                pass
        out.append(SatelliteObservation(
            observation_id=f"synthetic-{d.name}",
            mode="SYNTHETIC",
            satellite="SYNTHETIC — no satellite",
            sensor="simulated SAR",
            product=f"artifacts/{d.name}/scene.tif",
            raster_path=str(scene),
            acquired_at=acq,
            acquisition_time_known=False,
            received_at=_now(),
            region="Arabian Sea demo AOI",
            pixel_size=None,
            source="samudra.synth.generate — generated by this project",
            georeference="product",
            provenance_note=(
                "Generated scene, not an observation. The timestamp is the "
                "scenario's own simulated acquisition time, not a satellite's."
            ),
            caveats=["SYNTHETIC. Never present this as a satellite observation."],
        ))
    return out


# --------------------------------------------------------------- LIVE / NRT


AOI_ARABIAN_SEA = (68.0, 15.0, 73.0, 20.0)


def _live_observation(p: dict[str, Any]) -> SatelliteObservation:
    """A live catalogue hit -> an observation whose pixels are NOT here yet."""
    return SatelliteObservation(
        observation_id=f"live-{p['provider']}-{p['product_id']}",
        mode="LIVE_NRT",
        satellite=p["satellite"],
        sensor=p["sensor"],
        band="C-band",
        polarisation=p.get("polarisation"),
        product=p["name"],
        raster_path="",                      # nothing downloaded
        pixels_local=False,
        acquired_at=p["acquired_at"],
        acquisition_time_known=True,         # published by the provider
        processed_at=p.get("published_at"),
        received_at=_now(),
        footprint=p.get("footprint"),
        region="query AOI: Arabian Sea 68-73E, 15-20N",
        resolution_m=10.0 if (p.get("product_type") or "").startswith("IW_GRDH") else None,
        source=p["source"],
        source_url=p.get("source_url"),
        licence=p.get("licence"),
        checksum_sha256=p.get("checksum"),
        size_bytes=p.get("size_bytes"),
        download_url=p.get("download_url"),
        provider=p["provider"],
        online=p.get("online"),
        georeference="product",
        provenance_note=(
            f"LIVE catalogue entry, retrieved from the provider at "
            f"{_now()[:19]}Z. Satellite, sensor, mode, orbit and sensing time "
            f"are the provider's own values, not this system's. The IMAGE ITSELF "
            f"has not been downloaded — it is "
            f"{(p.get('size_bytes') or 0) / 1e9:.2f} GB on the provider's servers."
        ),
        caveats=[
            "Pixels are NOT on this machine. Nothing can be inferred from this "
            "entry until the product is downloaded.",
            "Downloading requires the provider account that produced this "
            "listing.",
        ],
    )


def list_live(days: int = 14, limit: int = 12,
              bbox=AOI_ARABIAN_SEA) -> tuple[list[SatelliteObservation], dict]:
    """Query every provider that can be queried. Returns (observations, report)."""
    from samudra.satellite import live as live_mod

    obs: list[SatelliteObservation] = []
    report: list[dict] = []
    for name in ("bhoonidhi", "mosdac"):
        prov = live_mod.get_provider(name)
        try:
            rows = prov.search(bbox=bbox, days=days, limit=limit)
            obs += [_live_observation(r) for r in rows]
            report.append({"provider": name, "result": "OK",
                           "products": len(rows)})
        except live_mod.ProviderError as e:
            report.append({"provider": name, "result": "UNAVAILABLE",
                           "products": 0, "detail": str(e),
                           "reachable": getattr(prov, "probe", lambda: {})().get("reachable")})
        except Exception as e:  # noqa: BLE001
            report.append({"provider": name, "result": "ERROR",
                           "products": 0, "detail": f"{type(e).__name__}: {e}"})
    return obs, {"checked_at": _now(), "aoi": list(bbox), "days": days,
                 "providers": report}


def download_live(observation_id: str, dest: Path = DROPIN_ROOT,
                  progress=None) -> dict[str, Any]:
    """Fetch a live product's pixels and register it as a local observation.

    The live catalogue entry and the downloaded file are deliberately the same
    provenance record: the sidecar JSON written here carries the provider's own
    satellite, sensor, product id and sensing time, so once it is on disk the
    console still reports where it came from rather than calling it an anonymous
    drop-in. That is the whole point of routing the download through this path
    instead of telling someone to copy a file in by hand.
    """
    from samudra.satellite import live as live_mod

    obs = None
    for o in list_live(limit=60)[0]:
        if o.observation_id == observation_id:
            obs = o
            break
    if obs is None:
        raise ValueError(f"{observation_id} is not in the current live catalogue")

    prov = live_mod.get_provider(obs.provider or "bhoonidhi")
    if not prov.has_credentials:
        raise live_mod.ProviderError(
            f"Downloading needs credentials. Set "
            f"{prov.env_user} / {prov.env_pass} "
            f"and try again. Catalogue search does not need them, which is why "
            f"this product is listed at all."
        )

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    # observation_id is "live-<provider>-<product uuid>"; the uuid keeps its dashes
    product_id = observation_id.split("-", 2)[2]
    zip_path = prov.download(product_id, dest, progress=progress)
    tif = live_mod.extract_measurement(zip_path, dest)

    side = tif.with_suffix(".json")
    side.write_text(json.dumps({
        "satellite": obs.satellite,
        "sensor": obs.sensor,
        "band": obs.band,
        "polarisation": obs.polarisation,
        "product": obs.product,
        "acquired_at": obs.acquired_at,
        "processed_at": obs.processed_at,
        "resolution_m": obs.resolution_m,
        "region": obs.region,
        "source": obs.source,
        "source_url": obs.source_url,
        "licence": obs.licence,
        "footprint": obs.footprint,
        "provenance_note": (
            f"Downloaded by SAMUDRA from {obs.provider} on {_now()[:19]}Z. "
            f"Catalogue id {product_id}. Satellite, sensor and sensing time are "
            f"the provider's published values."
        ),
    }, indent=2), encoding="utf-8")

    try:
        zip_path.unlink()          # the measurement raster is what we need
    except OSError:
        pass
    return {"observation_id": f"dropin-{tif.stem}", "raster": str(tif),
            "sidecar": str(side), "from_live": observation_id}


def live_status() -> dict[str, Any]:
    """What LIVE/NRT can actually do right now, established by asking.

    Never returns imagery, and never presents archive data as a feed. What it
    Both Indian providers gate search behind an account, so this normally
    reports LIVE/NRT UNAVAILABLE. That is the honest state, not a bug.
    """
    from samudra.satellite import live as live_mod

    st = live_mod.status()
    providers = []
    for p in st["providers"]:
        name = p["provider"]
        reach = live_mod.get_provider(name).probe()
        providers.append({
            "provider": name,
            "satellite": p["satellite"],
            "configured": p["has_credentials"],
            "env_vars": p["env_vars"],
            "search_needs_credentials": p["search_needs_credentials"],
            # Search and download are verified separately. LIVE/NRT means
            # pixels, so `available` below keys off credentials regardless of
            # how good the catalogue query is — an open search must never be
            # displayed as a live feed.
            "search_verified": p["search_verified"],
            "download_verified": p["download_verified"],
            "adapter_verified": p["adapter_verified"],
            "status": p["capability"],
            "host_reachable": None if reach is None else reach["reachable"],
            "register_url": p.get("register_url"),
            "note": p["note"],
        })

    searchable = [p for p in providers if p["configured"]]
    return {
        "available": bool(searchable),
        "status": "LIVE/NRT READY" if searchable else "LIVE/NRT UNAVAILABLE",
        "checked_at": _now(),
        "truststore": st["truststore"],
        "providers": providers,
        "explanation": (
            "Indian sources only. EOS-04 (SAR) comes from NRSC Bhoonidhi; "
            "INSAT-3DS (SST) and EOS-06 (ocean colour, scatterometer wind) come "
            "from MOSDAC. Both gate their catalogue behind a registered account "
            "and publish no open endpoint — Bhoonidhi returns 404 on every API "
            "path, MOSDAC /opendata/ returns 401 — so live retrieval is "
            "unavailable until one is configured. No non-Indian mission is "
            "carried as a substitute. The working route today is to download a "
            "product from the portal and drop it into data/satellite/incoming/ "
            "with a sidecar .json."
        ),
    }


def check_for_new_data(days: int = 14, limit: int = 12) -> dict[str, Any]:
    """AUTO MONITOR's action: ask the providers, report exactly what they said."""
    obs, report = list_live(days=days, limit=limit)
    ok = [r for r in report["providers"] if r["result"] == "OK"]
    if not ok:
        return {
            "checked_at": report["checked_at"],
            "result": "AUTHENTICATION REQUIRED",
            "new_products": [],
            "providers": report["providers"],
            "detail": "No provider could be queried without credentials.",
        }
    return {
        "checked_at": report["checked_at"],
        "result": "NEW DATA FOUND" if obs else "NO NEW DATA",
        "new_products": [o.to_dict() for o in obs],
        "providers": report["providers"],
        "detail": (
            f"{len(obs)} product(s) over {report['aoi']} in the last "
            f"{days} days. Metadata is live; pixels are not downloaded."
        ),
    }


# ------------------------------------------------------------------ registry


def list_all(replay_limit: int = 40, include_live: bool = True) -> dict[str, Any]:
    dropin = list_dropin()
    replay = dropin + list_replay(limit=replay_limit)
    live_obs: list[SatelliteObservation] = []
    live_report: dict = {}
    if include_live:
        try:
            live_obs, live_report = list_live()
        except Exception as e:  # noqa: BLE001 - the console must survive a dead network
            live_report = {"checked_at": _now(), "error": f"{type(e).__name__}: {e}"}
    return {
        "live": live_status(),
        "live_products": [o.to_dict() for o in live_obs],
        "live_report": live_report,
        "replay": [o.to_dict() for o in replay],
        "synthetic": [o.to_dict() for o in list_synthetic()],
        "dropin_dir": str(DROPIN_ROOT),
    }


def get(observation_id: str) -> SatelliteObservation | None:
    for o in list_dropin() + list_replay(limit=10_000) + list_synthetic():
        if o.observation_id == observation_id:
            return o
    if observation_id.startswith("live-"):
        # Re-query rather than cache: a live entry must reflect the catalogue
        # now, not whatever it said when the page was last loaded.
        for o in list_live(limit=40)[0]:
            if o.observation_id == observation_id:
                return o
    return None
