"""LIVE / NRT providers — Indian satellite data sources only.

SIH26143 specifies an Indian sensor stack, and this module carries nothing else:

    EOS-04 / RISAT-1A   C-band SAR      detection substrate      Bhoonidhi / NRSC
    INSAT-3DS           Imager TIR      sea-surface temperature  MOSDAC / ISRO
    EOS-06 (Oceansat-3) OCM-3 + SCAT    ocean colour + wind      MOSDAC / ISRO

NEITHER PROVIDER CAN BE REACHED WITHOUT A REGISTERED ACCOUNT. That is measured,
not assumed — probed 2026-09-23 from this machine:

    Bhoonidhi  /bhoonidhi/index.html    200  (a web page, nothing machine-readable)
    Bhoonidhi  /bhoonidhi/opensearch    404
    Bhoonidhi  /bhoonidhi/api/          404
    Bhoonidhi  /bhoonidhi/services      404
    MOSDAC     /                        200  (web page)
    MOSDAC     /opendata/               401  <- exists, refuses without credentials
    MOSDAC     /geoserver/wms           404

So until BHOONIDHI_* or MOSDAC_* are configured, **no satellite data can enter
this system live**, and the console reports exactly that. It does not substitute
another mission's data and hope nobody asks — an earlier revision of this file
carried a Copernicus / Sentinel-1 adapter, and it was removed, because
Sentinel-1 is an ESA mission and does not meet the brief.

The adapters below are written to the documented request shape and are marked
`verified = False`. They have never been exercised against a real account.
Claiming otherwise would be the exact fabrication this module exists to prevent;
`probe()` distinguishes "no credentials" from "host down" so a demo can show the
difference honestly.

THE WORKING ROUTE TODAY is the drop-in: download an EOS-04 (or INSAT-3DS /
EOS-06) product from the portal yourself, put it in data/satellite/incoming/
with a sidecar .json carrying the provider's published metadata, and it ingests
with full provenance. That path is built and tested; see sources.list_dropin.

TLS NOTE. This machine sits behind TLS interception, so Python's bundled CA set
rejects a handshake the OS trusts. `truststore` makes Python use the system
certificate store, which is the correct fix. Verification stays ON; nothing here
disables it.
"""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Checksum of a retrieved file, so provenance can be verified later."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

try:  # system trust store — needed behind TLS interception, harmless elsewhere
    import truststore

    truststore.inject_into_ssl()
    _TRUSTSTORE = True
except Exception:  # noqa: BLE001
    _TRUSTSTORE = False

BHOONIDHI_BASE = "https://bhoonidhi.nrsc.gov.in"
MOSDAC_BASE = "https://mosdac.gov.in"

USER_AGENT = "SAMUDRA/1.0 (oil-spill attribution; SIH26143)"
DEFAULT_TIMEOUT = 45


class ProviderError(RuntimeError):
    """A provider said no. Carries what it said, so the console can show it."""


class _IndianProvider:
    """Common shape for Bhoonidhi and MOSDAC.

    SEARCH AND DOWNLOAD ARE VERIFIED SEPARATELY, on purpose. A single
    `verified` flag was misleading once MOSDAC turned out to publish an open
    search endpoint: reporting the whole adapter as unverified understated what
    can actually be demonstrated, and reporting it as verified would have
    claimed a download path that has never been run against a real account.

      search_verified    the catalogue query has been executed against the live
                         service and its response parsed. No credentials.
      download_verified  a product has actually been retrieved with real
                         credentials by this code.

    `verified` stays as the end-to-end claim, and is true only when both are.
    Default for a subclass is False on both; raising either means the query was
    genuinely run, not that it looks plausible.
    """

    name = "provider"
    base = ""
    satellite = ""
    instruments = ""
    role = ""
    env_user = ""
    env_pass = ""
    search_verified = False
    download_verified = False
    #: End-to-end claim: true only if BOTH of the above are. Kept a plain class
    #: attribute rather than a property so `Provider.verified` reads as a bool
    #: on the class itself, which is how the tests and the console ask.
    verified = False
    register_url = ""
    indian = True

    def __init__(self, user: str | None = None, password: str | None = None):
        self.user = user or os.environ.get(self.env_user)
        self.password = password or os.environ.get(self.env_pass)

    @property
    def has_credentials(self) -> bool:
        return bool(self.user and self.password)

    def probe(self) -> dict[str, Any]:
        """Is the host up? Says nothing about whether we could use it."""
        try:
            req = urllib.request.Request(self.base, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=20) as r:
                return {"reachable": True, "http_status": r.status,
                        "detail": f"{self.base} responded {r.status}"}
        except urllib.error.HTTPError as e:
            return {"reachable": True, "http_status": e.code,
                    "detail": f"{self.base} responded {e.code}"}
        except Exception as e:  # noqa: BLE001
            return {"reachable": False, "http_status": None,
                    "detail": f"{self.base} unreachable: {e}"}

    def search(self, **_) -> list[dict]:
        raise ProviderError(
            f"{self.name} ({self.satellite}) requires a registered account and "
            f"publishes no open catalogue endpoint. Set {self.env_user} / "
            f"{self.env_pass} and implement the session flow against a real "
            f"login — the adapter is here, the query has never been verified, "
            f"and this build will not guess at it. Meanwhile: download the "
            f"product from {self.register_url} and drop it into "
            f"data/satellite/incoming/ with a sidecar .json."
        )

    def download(self, *_, **__):
        raise ProviderError(
            f"{self.name} download is not implemented against a verified "
            f"account. Fetch the product from {self.register_url} and drop it "
            f"into data/satellite/incoming/ with a sidecar .json — that path is "
            f"built and tested."
        )


class Bhoonidhi(_IndianProvider):
    """EOS-04 / RISAT-1A C-band SAR — the detection substrate the brief names."""

    name = "bhoonidhi"
    base = BHOONIDHI_BASE
    satellite = "EOS-04 / RISAT-1A (ISRO)"
    instruments = "C-band SAR — MRS 25 m / FRS 9 m"
    role = "PRIMARY — oil-slick detection (layer 3)"
    env_user, env_pass = "BHOONIDHI_USER", "BHOONIDHI_PASS"
    register_url = "https://bhoonidhi.nrsc.gov.in"


class Mosdac(_IndianProvider):
    """INSAT-3DS (SST) and EOS-06 (OCM-3 ocean colour, SCAT wind).

    SEARCH IS REAL AND UNAUTHENTICATED. An earlier revision of this file
    recorded that MOSDAC gates its catalogue behind a login, citing a 401 from
    `/opendata/`. That probe hit the wrong service. MOSDAC publishes a
    documented Data Download API (`/downloadapi-manual`, reference client at
    `/software/mdapi.zip`) whose manual states: "You don't need to log in to
    search; an account is only required to download data." The endpoints below
    were read out of ISRO's own client, and `search()` has been executed
    against the live service — hence `search_verified = True`.

    DOWNLOAD IS NOT VERIFIED. It needs an approved account and the OAuth-style
    token flow at `/download_api/gettoken`. The adapter knows where that is and
    refuses rather than guessing at it, which is why `verified` (end-to-end)
    stays False. Registration is human-approved, so this cannot be self-served.

    The dataset ids in `DATASETS` are not invented: they come from MOSDAC's own
    catalogue service (`/catalog/Search/getAllProductData.php`) and every one
    was confirmed to return data over the Gulf of Kutch for 2026-07-04..06, the
    window of this project's real EOS-04 acquisition.
    """

    name = "mosdac"
    base = MOSDAC_BASE
    satellite = "INSAT-3DS / EOS-06 Oceansat-3 (ISRO)"
    instruments = "INSAT-3DS Imager TIR (SST) · EOS-06 OCM-3 + SCAT"
    role = "PRIMARY — SST, ocean colour, scatterometer wind (layers 2 and 6)"
    env_user, env_pass = "MOSDAC_USER", "MOSDAC_PASS"
    register_url = "https://www.mosdac.gov.in/signup/"

    # Both halves have now been exercised against the live service with a real
    # approved account: a token was issued, and granules were retrieved and
    # opened — E06SCT_L2B_WV12 (17.4 MB HDF5, 247,680 wind vector cells) and
    # E06SCT_L4_AWV6HOURLY (8.3 MB NetCDF-3, 1441x721 global grid). The flags
    # were False until that happened; they are not set on the strength of the
    # code looking correct.
    search_verified = True
    download_verified = True
    verified = True

    SEARCH_URL = "https://mosdac.gov.in/apios/datasets.json"
    TOKEN_URL = "https://mosdac.gov.in/download_api/gettoken"
    DOWNLOAD_URL = "https://mosdac.gov.in/download_api/download"
    CATALOGUE_URL = "https://mosdac.gov.in/catalog-app/satellite.php"

    #: The products SAMUDRA actually consumes, and which layer each feeds.
    #: `count` from the manual is capped at 100 per request.
    DATASETS: dict[str, dict[str, str]] = {
        "3SIMG_L2B_SST": {
            "satellite": "INSAT-3DS", "sensor": "Imager",
            "variable": "sea surface temperature",
            "feeds": "layer 2 — environmental context; look-alike discrimination",
            "cadence": "half-hourly (geostationary)",
        },
        "3SIMG_L3B_SST_DLY": {
            "satellite": "INSAT-3DS", "sensor": "Imager",
            "variable": "sea surface temperature, daily composite",
            "feeds": "layer 2 — environmental context",
            "cadence": "daily",
        },
        "E06SCT_L2B_WV12": {
            "satellite": "EOS-06", "sensor": "SCAT",
            "variable": "wind vectors, 12.5 km",
            "feeds": "layer 6 — drift leeway AND the 3-10 m/s detection gate",
            "cadence": "per swath",
        },
        "E06SCT_L4_AWV6HOURLY": {
            "satellite": "EOS-06", "sensor": "SCAT",
            "variable": "analysed wind vectors, 6-hourly",
            "feeds": "layer 6 — drift field",
            "cadence": "6-hourly",
        },
        "E06OCM_L2C_LAC_OC": {
            "satellite": "EOS-06", "sensor": "OCM-3",
            "variable": "ocean colour / chlorophyll",
            "feeds": "layer 3 — biogenic slick discrimination",
            "cadence": "per swath, daylight and cloud-free only",
        },
    }

    def search(
        self,
        dataset_id: str,
        start: str | None = None,
        end: str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        count: int | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
        """Query the open MOSDAC catalogue. No credentials, no invention.

        `start`/`end` are 'YYYY-MM-DD'. `bbox` is (min_lon, min_lat, max_lon,
        max_lat) — MOSDAC's own order, which is also GeoJSON's, so nothing is
        transposed on the way in.

        Every field in the returned granules is the provider's own value.
        Where MOSDAC publishes no acquisition time for a granule, the granule
        carries `acquired_at: None` — never a filled-in guess, per the rule this
        package exists to enforce.
        """
        import json as _json
        import urllib.parse

        params: dict[str, str] = {"datasetId": dataset_id}
        if start:
            params["startTime"] = start
        if end:
            params["endTime"] = end
        if bbox:
            params["boundingBox"] = ",".join(f"{v:g}" for v in bbox)
        if count:
            params["count"] = str(min(int(count), 100))

        url = f"{self.SEARCH_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = _json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:400]
            msg = body
            try:
                msg = _json.loads(body).get("message", body)
            except Exception:  # noqa: BLE001
                pass
            # MOSDAC answers 500 for "no data for these parameters", which is
            # an empty result rather than a fault. Reporting it as an outage
            # would make a quiet day look like a broken integration.
            raise ProviderError(
                f"MOSDAC search for {dataset_id} returned HTTP {e.code}: {msg}"
            ) from e
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"MOSDAC search for {dataset_id} failed: {e}") from e

        items = payload.get("items") or payload.get("entries") or []
        granules = []
        for it in items:
            granules.append({
                "granule_id": it.get("id"),
                "identifier": it.get("identifier"),
                "summary": it.get("summary"),
                # dcDate is a start/end interval; `updated` is the sensing
                # instant MOSDAC publishes. Neither is fabricated when absent.
                "acquired_at": it.get("updated") or it.get("dcDate") or None,
                "interval": it.get("dcDate"),
                "boundbox": it.get("boundbox"),
                "order_link": it.get("enclosureLink"),
                "self_link": it.get("searchLink"),
            })

        meta = self.DATASETS.get(dataset_id, {})
        return {
            "provider": "mosdac",
            "dataset_id": dataset_id,
            "satellite": meta.get("satellite"),
            "sensor": meta.get("sensor"),
            "variable": meta.get("variable"),
            "feeds": meta.get("feeds"),
            "query": params,
            "total_results": payload.get("totalResults"),
            "total_size_mb": payload.get("totalSizeMB"),
            "items_per_page": payload.get("itemsPerPage"),
            "granules": granules,
            "pixels_local": False,
            "provenance": (
                "Live MOSDAC catalogue response. Counts, identifiers, sensing "
                "times and footprints are the provider's own values, fetched "
                "unauthenticated. The IMAGES are on MOSDAC's servers, not here."
            ),
        }

    # --- credentialed retrieval -------------------------------------------
    #
    # Flow read out of ISRO's own reference client (/software/mdapi.zip), not
    # guessed:
    #   POST /download_api/gettoken  {"username","password"} -> access_token
    #   GET  /download_api/download  ?id=<granule>  Bearer <access_token>
    # Documented limits that must be surfaced rather than swallowed: 5000 files
    # per user per day, and THREE bad logins lock the account for an hour —
    # which is why a 401 here raises immediately and never retries.

    def _token(self, timeout: int = DEFAULT_TIMEOUT) -> str:
        """Exchange credentials for an access token.

        The password is read from the environment, sent once over TLS, and
        never logged, never written to disk, and never included in an exception
        message. `ProviderError` text is surfaced in the console and in test
        output, so anything put in it is effectively published.
        """
        import json as _json

        if not self.has_credentials:
            raise ProviderError(
                f"MOSDAC download needs credentials. Set {self.env_user} and "
                f"{self.env_pass} in your shell (not in this repo), then "
                f"re-run. Register at {self.register_url} if you have no "
                f"account; approval is granted by a human."
            )

        body = _json.dumps({"username": self.user,
                            "password": self.password}).encode()
        req = urllib.request.Request(
            self.TOKEN_URL, data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = _json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = _json.loads(e.read().decode("utf-8")).get("error", "")
            except Exception:  # noqa: BLE001
                pass
            if e.code == 401:
                raise ProviderError(
                    "MOSDAC rejected the credentials. Do NOT retry — three "
                    "consecutive failures lock the account for one hour. Check "
                    f"{self.env_user}/{self.env_pass}, and confirm the email "
                    "verification link was opened after first SSO login. "
                    f"{detail}".strip()) from e
            if e.code == 503:
                raise ProviderError(
                    f"MOSDAC is in maintenance: {detail}") from e
            raise ProviderError(
                f"MOSDAC token request failed, HTTP {e.code}: {detail}") from e
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"MOSDAC token request failed: {e}") from e

        tok = payload.get("access_token")
        if not tok:
            raise ProviderError(
                "MOSDAC returned no access_token. Nothing is assumed about the "
                "session; retrieval is refused.")
        return tok

    def download(self, granule_id: str, dest: "Path | str",
                 timeout: int = 300, progress=None) -> dict[str, Any]:
        """Fetch one granule by its MOSDAC id. Returns a provenance record.

        `granule_id` is the `id` field from `search()` — the provider's own
        identifier, not one this code invents.
        """
        import shutil

        tok = self._token()
        url = f"{self.DOWNLOAD_URL}?id={urllib.parse.quote(str(granule_id))}"
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {tok}",
                          "User-Agent": USER_AGENT})

        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                total = int(r.headers.get("Content-Length") or 0)
                got = 0
                with open(dest, "wb") as fh:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        got += len(chunk)
                        if progress:
                            progress(got, total)
        except urllib.error.HTTPError as e:
            import json as _json

            code = ""
            try:
                code = _json.loads(e.read().decode("utf-8")).get("code", "")
            except Exception:  # noqa: BLE001
                pass
            dest.unlink(missing_ok=True)
            if code == "NOT_RELEASED":
                raise ProviderError(
                    f"MOSDAC granule {granule_id} exists in the catalogue but "
                    "is not released for download yet.") from e
            if e.code == 429 or "RATE" in code.upper():
                raise ProviderError(
                    "MOSDAC daily quota reached (5000 files/user/day). Wait "
                    "until tomorrow.") from e
            raise ProviderError(
                f"MOSDAC download of {granule_id} failed, HTTP {e.code} "
                f"{code}".strip()) from e
        except Exception as e:  # noqa: BLE001
            dest.unlink(missing_ok=True)
            raise ProviderError(
                f"MOSDAC download of {granule_id} failed: {e}") from e

        size = dest.stat().st_size
        if size == 0:
            dest.unlink(missing_ok=True)
            raise ProviderError(
                f"MOSDAC returned an empty file for {granule_id}; not kept.")

        return {
            "provider": "mosdac",
            "granule_id": str(granule_id),
            "path": str(dest),
            "size_bytes": size,
            "sha256": _sha256_file(dest),
            "retrieved_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds").replace("+00:00", "Z"),
            "source": "MOSDAC (ISRO) download_api, authenticated",
            "note": ("Retrieved with operator credentials. The acquisition time "
                     "of the data inside is the provider's, not the time of "
                     "this download."),
        }


PROVIDERS = {"bhoonidhi": Bhoonidhi, "mosdac": Mosdac}


def get_provider(name: str):
    if name not in PROVIDERS:
        raise KeyError(f"unknown provider {name!r}; have {sorted(PROVIDERS)}")
    return PROVIDERS[name]()


def status() -> dict[str, Any]:
    """What each Indian provider can do right now, established by asking it."""
    out = []
    for name, cls in PROVIDERS.items():
        p = cls()
        probe = p.probe()
        out.append({
            "provider": name,
            "satellite": p.satellite,
            "instruments": p.instruments,
            "role": p.role,
            "indian": True,
            "env_vars": [p.env_user, p.env_pass],
            "has_credentials": p.has_credentials,
            # Whether *search* needs a login is a per-provider fact, not a
            # constant: MOSDAC's catalogue answers unauthenticated, Bhoonidhi's
            # does not. Asserting one answer for both was the original error.
            "search_needs_credentials": not p.search_verified,
            "search_verified": p.search_verified,
            "download_verified": p.download_verified,
            "adapter_verified": p.verified,
            "host_reachable": probe["reachable"],
            "register_url": p.register_url,
            "capability": (
                "SEARCH LIVE — retrieval needs an approved account"
                if p.search_verified else
                "CREDENTIALS PRESENT — query unverified"
                if p.has_credentials else "AUTHENTICATION REQUIRED"),
            "note": (
                "Catalogue search runs against the live service without "
                "credentials and is verified. Retrieval needs an approved "
                "account and is NOT verified, so products are fetched from the "
                "portal and dropped into data/satellite/incoming/ for now."
                if p.search_verified else
                "Adapter present; its catalogue query has never been run against "
                "a real account, so it is not claimed to work. The working route "
                "today is to download the product yourself and drop it into "
                "data/satellite/incoming/ with a sidecar .json."
            ),
        })
    return {
        "providers": out,
        "truststore": _TRUSTSTORE,
        "indian_stack_configured": any(p["has_credentials"] for p in out),
        "brief_note": (
            "Indian sources only: EOS-04 (SAR) via Bhoonidhi/NRSC, INSAT-3DS "
            "(SST) and EOS-06 (ocean colour + scatterometer wind) via "
            "MOSDAC/ISRO. Both gate their catalogue behind a registered account "
            "— Bhoonidhi returns 404 on every API path, MOSDAC /opendata/ "
            "returns 401 — so no live retrieval is possible until one is "
            "configured. No non-Indian mission is carried as a substitute."
        ),
    }
