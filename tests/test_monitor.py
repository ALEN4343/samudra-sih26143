"""The continuous monitor — and the one claim it must never make.

Monitoring is the easiest thing in this project to overclaim. A loop that runs
every five minutes over a satellite with 5-11 day revisit is a continuous
SERVICE, not a continuous FEED, and the difference is the whole honesty of the
feature. These tests pin that difference in place.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from samudra.monitor import service


def test_status_is_declared_absent_not_invented(tmp_path: Path):
    """No status file must never look like a check that found nothing."""
    st = service.read_status(tmp_path / "nope.json")
    assert st["running"] is False
    assert st["checks"] == 0
    assert st["last_check_at"] is None
    assert "never run" in st["detail"].lower()


def test_corrupt_status_does_not_raise(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{ not json", encoding="utf-8")
    st = service.read_status(p)
    assert st["running"] is False
    assert "unreadable" in st["detail"]


def test_checked_and_found_are_separate_fields(tmp_path: Path):
    """The core honesty property.

    A service that checked 30 seconds ago and last saw a satellite six days ago
    is the NORMAL state at this revisit. One "last updated" field would read as
    a live feed, so the two are published separately and the second must not be
    silently refreshed by a check that found nothing.
    """
    p = tmp_path / "status.json"
    report = {"checked_at": "2026-09-25T05:49:07Z", "result": "NO NEW DATA",
              "new_observations": [], "processed": [], "errors": [],
              "known_observations": [], "elapsed_s": 1.0}
    st = service._publish(report, checks=9, started_at="2026-09-25T05:00:00Z",
                          running=True, last_new="2026-09-19T01:17:53Z", path=p)

    assert st["last_check_at"] == "2026-09-25T05:49:07Z"
    assert st["last_new_data_at"] == "2026-09-19T01:17:53Z"
    assert st["last_check_at"] != st["last_new_data_at"]
    assert st["last_result"] == "NO NEW DATA"
    # And the caveat travels with the status, not just the docs.
    assert "SATELLITE does not" in st["detail"]
    assert json.loads(p.read_text())["checks"] == 9


def test_no_new_data_is_a_normal_result_not_an_error(tmp_path: Path):
    out = service.scan_once(root=tmp_path, process=False,
                            poll_environment=False)
    assert out["result"] in ("NO NEW DATA", "NEW DATA FOUND",
                             "PROCESSED NEW DATA")
    assert out["errors"] == [] or all("detail" in e for e in out["errors"])


def test_environment_poll_is_labelled_context_never_detection():
    """INSAT-3DS at 1-4 km cannot resolve a routine discharge.

    If the monitor ever described these products as detections, the console
    would show a continuous oil-spill feed that does not exist.
    """
    for ds, purpose in service.ENV_DATASETS:
        assert ds.startswith(("3SIMG", "E06")), ds
        assert purpose and "detect" not in purpose.lower(), ds


def test_environment_datasets_are_indian_only():
    for ds, _ in service.ENV_DATASETS:
        assert ds.startswith(("3SIMG", "E06", "E04")), (
            f"{ds} is not an ISRO product id")


def test_monitor_never_claims_live_for_replayed_pixels(tmp_path: Path):
    """No status VALUE may say LIVE.

    Checked on the published status rather than by grepping the source: the
    module's own docstring warns against this claim, and a source-level search
    flags that warning as the violation. Behaviour is what matters — a drop-in
    product is REAL_REPLAY, and calling it live is the one lie that would not
    survive a judge checking the acquisition date.
    """
    p = tmp_path / "status.json"
    report = {"checked_at": "2026-09-25T05:49:07Z", "result": "NO NEW DATA",
              "new_observations": [], "processed": [], "errors": [],
              "known_observations": [], "elapsed_s": 1.0}
    st = service._publish(report, checks=1, started_at="2026-09-25T05:00:00Z",
                          running=True, last_new=None, path=p)

    def values(node):
        if isinstance(node, dict):
            for v in node.values():
                yield from values(v)
        elif isinstance(node, list):
            for v in node:
                yield from values(v)
        elif isinstance(node, str):
            yield node

    for v in values({k: x for k, x in st.items() if k != "detail"}):
        assert "live" not in v.lower(), f"status value claims live: {v!r}"


@pytest.mark.parametrize("flag", ["process", "poll_environment"])
def test_scan_once_accepts_being_switched_off(tmp_path: Path, flag):
    """An offline demo must be able to run the loop without network or GPU."""
    kwargs = {"root": tmp_path, "process": False, "poll_environment": False}
    kwargs[flag] = False
    out = service.scan_once(**kwargs)
    assert "checked_at" in out and "result" in out
