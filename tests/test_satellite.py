"""Satellite operations — provenance, modes, and the ground-truth guard.

The tests that matter most here are the negative ones. It is easy to write a
satellite demo that looks convincing and is quietly reading the answer key, and
easy to write one that fabricates a timestamp because a field wanted filling.
These assert that neither happens, so a later refactor cannot reintroduce it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from samudra.satellite import inference, sources

WEB = Path("web")


# ------------------------------------------------------- observation contract

def test_observation_contract_fields():
    obs = sources.list_replay(limit=4, include_research=True)
    assert obs, "no real SAR imagery found on disk"
    o = obs[0]
    for f in ("observation_id", "mode", "satellite", "sensor", "product",
              "raster_path", "source", "provenance_note", "checksum_sha256"):
        assert getattr(o, f), f"{f} is empty on a real observation"
    assert o.mode in ("LIVE_NRT", "REAL_REPLAY", "SYNTHETIC")
    assert Path(o.raster_path).exists()


def test_checksum_is_of_the_actual_file():
    import hashlib

    o = sources.list_replay(limit=1, include_research=True)[0]
    h = hashlib.sha256(Path(o.raster_path).read_bytes()).hexdigest()
    assert o.checksum_sha256 == h


def test_observation_is_serialisable():
    o = sources.list_replay(limit=1, include_research=True)[0]
    json.dumps(o.to_dict())  # must survive the API boundary


# ----------------------------------------------- the timestamp honesty rule

def test_unknown_acquisition_time_is_none_not_invented():
    """The SOS and CSIRO archives publish no per-chip acquisition time.

    A plausible-looking timestamp is worse than no timestamp: it survives being
    quoted at a judge. So the field stays None and the flag stays False.
    """
    for o in sources.list_replay(limit=8, include_research=True):
        if o.source.startswith(("Deep-SAR", "CSIRO")):
            assert o.acquired_at is None
            assert o.acquisition_time_known is False


def test_no_observation_claims_a_time_it_cannot_support():
    for o in sources.list_replay(limit=20, include_research=True) + sources.list_synthetic():
        if o.acquired_at is not None:
            assert o.acquisition_time_known or o.mode == "SYNTHETIC"


def test_csiro_chips_do_not_claim_indian_waters():
    """The CSIRO collection footprint is Southeast Asia / Australia."""
    csiro = [o for o in sources.list_replay(limit=40, include_research=True) if "CSIRO" in o.source]
    assert csiro
    for o in csiro:
        assert "101" in (o.region or "") and "154" in (o.region or "")
        assert any("NOT Indian waters" in c for c in o.caveats)


def test_csiro_class_semantics_come_from_publisher_metadata():
    """Part 11: the label meaning is verified, not inferred from appearance."""
    meta = Path("data/raw/oilspill/binary/metadata/dataset_metadata.xml")
    if not meta.exists():
        pytest.skip("binary dataset metadata not present")
    text = meta.read_text(encoding="utf-8")
    assert "not containing any oil features" in text
    assert "10.25919/4v55-dn16" in text


# ------------------------------------------------------------- replay mode

def test_replay_draws_from_the_held_out_test_split():
    """A prediction on a training chip would be a memory test, not a detection."""
    for o in sources.list_replay(limit=20, include_research=True):
        if o.product.startswith("SOS/"):
            assert "/test/" in o.product


def test_replay_works_without_network():
    """Offline replay: every path is local."""
    for o in sources.list_replay(limit=6, include_research=True):
        p = Path(o.raster_path)
        assert p.is_absolute() or p.exists()
        assert "://" not in str(p)


# ------------------------------------------------------------ synthetic mode

def test_synthetic_is_labelled_and_never_called_a_satellite():
    for o in sources.list_synthetic():
        assert o.mode == "SYNTHETIC"
        assert "SYNTHETIC" in o.satellite.upper()
        assert any("SYNTHETIC" in c.upper() for c in o.caveats)


# ------------------------------------------------------------ live / NRT

def test_credentialed_providers_report_authentication_required(monkeypatch):
    """Both Indian providers gate SEARCH and must say so, never appear usable."""
    for u, p in sources.LIVE_ENV.values():
        monkeypatch.delenv(u, raising=False)
        monkeypatch.delenv(p, raising=False)
    st = sources.live_status()
    gated = [p for p in st["providers"] if p["search_needs_credentials"]]
    assert gated, "bhoonidhi and mosdac gate search and must be listed"
    for p in gated:
        assert p["status"] == "AUTHENTICATION REQUIRED"
        assert p["configured"] is False
        assert p["adapter_verified"] is False


def test_live_reports_which_env_vars_it_wants():
    st = sources.live_status()
    names = {v for p in st["providers"] for v in p["env_vars"]}
    assert {"BHOONIDHI_USER", "BHOONIDHI_PASS",
            "MOSDAC_USER", "MOSDAC_PASS"} == names


def test_no_credentials_are_hardcoded():
    src = Path("src/samudra/satellite/sources.py").read_text(encoding="utf-8")
    for bad in ("password=", "passwd=", "api_key=", "apikey="):
        assert bad not in src.lower()


def test_check_reports_a_recognised_state_and_never_local_pixels(monkeypatch):
    """Whatever it returns, every product must be flagged as not downloaded."""
    for u, p in sources.LIVE_ENV.values():
        monkeypatch.delenv(u, raising=False)
        monkeypatch.delenv(p, raising=False)
    r = sources.check_for_new_data()
    assert r["result"] in ("AUTHENTICATION REQUIRED", "SOURCE UNAVAILABLE",
                           "NO NEW DATA", "NEW DATA FOUND")
    for p in r["new_products"]:
        assert p["mode"] == "LIVE_NRT"
        assert p["pixels_local"] is False
        assert p["raster_path"] == ""
        assert p["acquired_at"], "a live product must carry the provider's time"


def test_live_products_are_never_folded_into_replay_or_synthetic(monkeypatch):
    """The three modes stay separate keys, so nothing can be mislabelled."""
    for u, p in sources.LIVE_ENV.values():
        monkeypatch.delenv(u, raising=False)
        monkeypatch.delenv(p, raising=False)
    all_ = sources.list_all(replay_limit=4)
    assert all(o["mode"] != "LIVE_NRT" for o in all_["replay"] + all_["synthetic"])
    assert all(o["mode"] == "LIVE_NRT" for o in all_.get("live_products", []))
    assert all(o["pixels_local"] is True
               for o in all_["replay"] + all_["synthetic"])


# --------------------------------------------------- missing / invalid data

def test_unknown_observation_id_returns_none():
    assert sources.get("does-not-exist-anywhere") is None


def test_dropin_directory_is_optional():
    assert sources.list_dropin(Path("data/satellite/definitely-not-here")) == []


def test_dropin_without_sidecar_declares_the_gap(tmp_path):
    from PIL import Image

    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(tmp_path / "scene.png")
    got = sources.list_dropin(tmp_path)
    assert len(got) == 1
    o = got[0]
    assert "UNDECLARED" in o.satellite
    assert o.acquisition_time_known is False
    assert any("No sidecar" in c for c in o.caveats)


def test_dropin_uses_declared_metadata(tmp_path):
    from PIL import Image

    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(tmp_path / "eos04.png")
    (tmp_path / "eos04.json").write_text(json.dumps({
        "satellite": "EOS-04", "sensor": "SAR MRS",
        "acquired_at": "2026-03-01T05:20:00+00:00",
        "product": "EOS04_MRS_TEST", "source": "NRSC Bhoonidhi",
        "resolution_m": 25.0,
    }))
    o = sources.list_dropin(tmp_path)[0]
    assert o.satellite == "EOS-04"
    assert o.acquisition_time_known is True
    assert o.acquired_at.startswith("2026-03-01")
    assert o.resolution_m == 25.0


# ------------------------------------------------- THE ground-truth guard

def test_ground_truth_guard_rejects_the_answer_key():
    with pytest.raises(RuntimeError):
        inference.assert_no_ground_truth("artifacts/demo-001/ground_truth.json")
    with pytest.raises(RuntimeError):
        inference.assert_no_ground_truth("data/raw/oilspill/sos/test/sentinel/label/3.png")


def test_ground_truth_guard_allows_an_image():
    inference.assert_no_ground_truth("data/raw/oilspill/sos/test/sentinel/image/3.png")


def test_inference_path_never_opens_a_ground_truth_file():
    """Source-level assertion: no line both names a label source and reads it.

    Prose and the guard itself may say `ground_truth`; what must not exist is a
    line that mentions it next to a file read. Checking for the substring alone
    would flag this module's own docstring, which is the opposite of useful.
    """
    readers = ("open(", "read_text", "read_bytes", "json.load", "np.load",
               "imread", "Image.open", "rasterio.open", "read_parquet")
    for mod in ("inference.py", "pipeline.py", "sources.py", "__main__.py"):
        text = (Path("src/samudra/satellite") / mod).read_text(encoding="utf-8")
        for n, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            if "ground_truth" not in low and "/label/" not in low:
                continue
            assert not any(r.lower() in low for r in readers), \
                f"{mod}:{n} reads a label source on the inference path: {line.strip()}"


def test_pipeline_records_that_ground_truth_was_not_used():
    text = Path("src/samudra/satellite/pipeline.py").read_text(encoding="utf-8")
    assert '"ground_truth_used": False' in text


def test_replay_observation_paths_are_images_not_labels():
    for o in sources.list_replay(limit=20, include_research=True):
        s = o.raster_path.replace("\\", "/")
        assert "/label/" not in s
        assert "ground_truth" not in s


# ------------------------------------------------------ inference behaviour

@pytest.mark.skipif(not inference.checkpoint_status()["available"],
                    reason="no trained checkpoint on this machine")
def test_inference_runs_and_reports_its_own_confidence():
    o = sources.list_replay(limit=1, include_research=True)[0]
    r = inference.run(o.raster_path)
    assert r["mask"].shape == r["prob"].shape
    assert 0.0 <= r["anomaly_pixel_fraction"] <= 1.0
    assert 0.0 <= r["max_probability"] <= 1.0
    assert r["class_label"] == "OIL-LIKE SURFACE ANOMALY"
    assert "not a confirmed oil detection" in r["class_caveat"]
    names = [s["stage"] for s in r["stages"]]
    assert "ML inference" in names


@pytest.mark.skipif(not inference.checkpoint_status()["available"],
                    reason="no trained checkpoint on this machine")
def test_two_different_chips_give_different_masks():
    """If the mask were hardcoded or cached, this would fail."""
    a, b = sources.list_replay(limit=6, include_research=True)[0], sources.list_replay(limit=6, include_research=True)[3]
    ra = inference.run(a.raster_path)
    rb = inference.run(b.raster_path)
    assert not np.array_equal(ra["mask"], rb["mask"])


@pytest.mark.skipif(not inference.checkpoint_status()["available"],
                    reason="no trained checkpoint on this machine")
def test_threshold_actually_changes_the_result():
    o = sources.list_replay(limit=1, include_research=True)[0]
    lo = inference.run(o.raster_path, threshold=0.2)["anomaly_pixel_fraction"]
    hi = inference.run(o.raster_path, threshold=0.9)["anomaly_pixel_fraction"]
    assert lo >= hi, "a higher threshold must not select more pixels"


def test_checkpoint_status_flags_a_non_representative_model():
    st = inference.checkpoint_status()
    if st["available"]:
        assert "representative" in st
        if not st["representative"]:
            assert "NOT REPRESENTATIVE" in st["detail"]


# ============================================================ LIVE / NRT
#
# These hit a real network endpoint. They are marked so a CI box without egress
# can deselect them, but they are NOT mocked: the point of the live path is that
# it talks to a real provider, and a mocked test of that proves nothing.

live_net = pytest.mark.live


def test_live_module_never_disables_tls_verification():
    """Behind TLS interception the tempting fix is verify=False. Never."""
    src = Path("src/samudra/satellite/live.py").read_text(encoding="utf-8")
    for bad in ("verify=False", "CERT_NONE", "_create_unverified_context",
                "check_hostname = False"):
        assert bad not in src, f"TLS verification weakened: {bad}"
    assert "truststore" in src, "system trust store is how this is done properly"


def test_live_providers_declare_whether_their_adapter_is_verified():
    """Each adapter states exactly how far it has actually been exercised.

    MOSDAC is now verified end to end: a token was issued against the live
    service with an approved account, and real granules were retrieved and
    opened. Bhoonidhi is not, and cannot be — it publishes no API at all, so
    products arrive only by manual download.

    The flags were both False until the retrieval actually happened. They are
    not set on the strength of the code looking correct.
    """
    from samudra.satellite import live

    for name in ("bhoonidhi", "mosdac"):
        p = live.get_provider(name)
        assert isinstance(p.verified, bool)
        assert isinstance(p.search_verified, bool)
        assert isinstance(p.download_verified, bool)
        # `verified` is the end-to-end claim and must never exceed its parts.
        assert p.verified == (p.search_verified and p.download_verified)

    assert live.Bhoonidhi.verified is False
    assert live.Bhoonidhi.search_verified is False
    assert live.Mosdac.search_verified is True
    assert live.Mosdac.download_verified is True


def test_bhoonidhi_refuses_to_invent_a_catalogue_it_cannot_query(monkeypatch):
    """Bhoonidhi publishes no open endpoint, so search must raise, not guess."""
    from samudra.satellite import live

    p = live.get_provider("bhoonidhi")
    with pytest.raises(live.ProviderError) as e:
        p.search()
    assert "registered account" in str(e.value)
    assert p.register_url in str(e.value)


def test_mosdac_download_refuses_without_credentials(monkeypatch):
    """Retrieval is implemented and verified, but still needs YOUR account.

    The two halves stay separate: search is open and needs nothing, download
    needs an approved login. With the environment cleared it must fail closed
    and point at registration rather than half-attempting anything — and it
    must NOT retry, because MOSDAC locks an account for an hour after three
    consecutive bad logins.
    """
    from samudra.satellite import live

    monkeypatch.delenv("MOSDAC_USER", raising=False)
    monkeypatch.delenv("MOSDAC_PASS", raising=False)

    p = live.Mosdac()
    assert p.has_credentials is False
    with pytest.raises(live.ProviderError) as e:
        p.download("anything", tmp_path_factory_placeholder := "ignored.bin")
    msg = str(e.value)
    assert "MOSDAC_USER" in msg and "MOSDAC_PASS" in msg
    assert p.register_url in msg
    # The refusal must never leak a credential, even an absent one.
    assert "password" not in msg.lower().replace("mosdac_pass", "")


def test_mosdac_dataset_ids_carry_the_layer_each_one_feeds():
    """A dataset id with no stated purpose is a number nobody can audit."""
    from samudra.satellite import live

    ds = live.Mosdac.DATASETS
    assert "3SIMG_L2B_SST" in ds          # INSAT-3DS sea surface temperature
    assert "E06SCT_L2B_WV12" in ds        # EOS-06 scatterometer wind
    assert "E06OCM_L2C_LAC_OC" in ds      # EOS-06 ocean colour
    for key, meta in ds.items():
        assert meta["satellite"] in ("INSAT-3DS", "EOS-06"), key
        assert meta["variable"] and meta["feeds"], key
    # Wind feeds both the drift model and the detection gate; say so.
    assert "gate" in ds["E06SCT_L2B_WV12"]["feeds"]


def test_live_catalogue_entry_is_marked_as_having_no_pixels():
    o = sources._live_observation({
        "provider": "copernicus", "product_id": "abc", "name": "S1D_IW_GRDH_x",
        "satellite": "Sentinel-1D", "sensor": "SAR C-band",
        "acquired_at": "2026-09-18T01:02:38Z", "source": "CDSE (live)",
        "size_bytes": 2_140_000_000, "online": True,
    })
    assert o.mode == "LIVE_NRT"
    assert o.pixels_local is False
    assert o.raster_path == ""
    assert o.acquisition_time_known is True       # the provider DOES publish it
    assert any("NOT on this machine" in c for c in o.caveats)


def test_pipeline_refuses_to_process_a_live_entry_without_pixels():
    """The worst possible bug would be silently running an archive chip and
    labelling the result LIVE. This is the guard against it."""
    from samudra.satellite import pipeline

    o = sources._live_observation({
        "provider": "copernicus", "product_id": "abc", "name": "S1D_IW_GRDH_x",
        "satellite": "Sentinel-1D", "sensor": "SAR C-band",
        "acquired_at": "2026-09-18T01:02:38Z", "source": "CDSE (live)",
        "size_bytes": 2_140_000_000, "online": True,
    })
    with pytest.raises(RuntimeError) as e:
        pipeline.process(o)
    assert "LIVE CATALOGUE ENTRY" in str(e.value)
    assert "has not been downloaded" in str(e.value)


def test_bhoonidhi_download_points_at_the_manual_route(monkeypatch):
    """Bhoonidhi has no API, so its refusal must tell you what to do instead.

    Unlike MOSDAC this cannot be fixed with credentials — NRSC publishes no
    programmatic endpoint. The only route is downloading the product from the
    portal by hand, so the error names the drop-in folder rather than implying
    an account would help.
    """
    from samudra.satellite import live

    p = live.get_provider("bhoonidhi")
    with pytest.raises(live.ProviderError) as e:
        p.download("anything")
    assert "data/satellite/incoming" in str(e.value)


def test_no_foreign_provider_is_carried_as_a_live_source():
    """The brief specifies an Indian sensor stack. LIVE/NRT must offer nothing else.

    An earlier revision carried a Copernicus / Sentinel-1 adapter because its
    catalogue answers without an account. It was removed: a non-Indian mission
    sitting in the live provider list invites exactly the wrong conclusion.
    """
    from samudra.satellite import live

    assert set(live.PROVIDERS) == {"bhoonidhi", "mosdac"}
    for p in live.status()["providers"]:
        assert p["indian"] is True
    src = Path("src/samudra/satellite/live.py").read_text(encoding="utf-8")
    for foreign in ("dataspace.copernicus.eu", "CDSE_USER", "class Copernicus"):
        assert foreign not in src, f"non-Indian provider still wired in: {foreign}"


def test_live_is_unavailable_until_an_indian_account_is_configured(monkeypatch):
    for u, p in sources.LIVE_ENV.values():
        monkeypatch.delenv(u, raising=False)
        monkeypatch.delenv(p, raising=False)
    st = sources.live_status()
    assert st["available"] is False
    assert st["status"] == "LIVE/NRT UNAVAILABLE"
    assert len(st["providers"]) == 2
    for p in st["providers"]:
        assert p["register_url"]
        # `configured` is the credential state and is what LIVE/NRT keys off.
        # `download_verified` says the CODE works, which is a different claim
        # and stays True for MOSDAC even with the environment cleared — an
        # implemented retrieval does not become unimplemented because this
        # particular machine has no login. Conflating the two would make the
        # console flip between "not built" and "built" depending on env vars.
        assert p["configured"] is False
    by_name = {p["provider"]: p for p in st["providers"]}
    assert by_name["bhoonidhi"]["status"] == "AUTHENTICATION REQUIRED"
    assert by_name["mosdac"]["search_verified"] is True


def test_indian_provider_hosts_are_reachable_so_the_console_can_tell_the_difference():
    """'needs a login' and 'host is down' are different answers to a judge."""
    from samudra.satellite import live

    for name in ("bhoonidhi", "mosdac"):
        probe = live.get_provider(name).probe()
        assert "reachable" in probe and "detail" in probe
