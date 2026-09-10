"""Chain of custody — CLAUDE.md section 4.6.

SHA-256 over the canonical JSON of every pipeline input and output, appended to
artifacts/audit.log and hash-chained to the previous entry. An investigator can
recompute the digests and confirm no artifact was altered after the fact; a
tampered entry breaks every link after it, not just its own.

Canonicalisation matters. Dict ordering, float formatting and whitespace all
change the bytes without changing the meaning, so the same artifact would
otherwise hash differently on different runs and the chain would be worthless.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

AUDIT_LOG = Path("artifacts/audit.log")
GENESIS = "0" * 64

# Excluded from hashing:
#   ground_truth.json  - verification only, not a pipeline artifact
#   audit.log          - cannot contain its own digest
#   dossier_*.pdf      - the dossier ATTESTS to the artifacts; it is not one of
#                        them. Including it is circular: the dossier prints the
#                        chain hash, so writing it changes the directory it just
#                        hashed and every later verify reports BROKEN. The
#                        dossier's own integrity comes from the chain hash
#                        printed inside it, checked against audit.log.
EXCLUDE = {"ground_truth.json", "audit.log"}
EXCLUDE_PREFIXES = ("dossier_",)


def _excluded(name: str) -> bool:
    return name in EXCLUDE or name.startswith(EXCLUDE_PREFIXES)


def canonical_json(obj) -> bytes:
    """Byte-stable JSON: sorted keys, no incidental whitespace, UTF-8."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")


def hash_file(path: Path) -> str:
    """SHA-256 of a file. JSON and GeoJSON are canonicalised first."""
    if path.suffix in (".json", ".geojson"):
        try:
            return hashlib.sha256(canonical_json(json.loads(path.read_text()))).hexdigest()
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass  # fall through and hash the raw bytes
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(incident_dir: Path) -> dict[str, dict]:
    """Digest every artifact in the incident directory."""
    out: dict[str, dict] = {}
    for p in sorted(incident_dir.iterdir()):
        if not p.is_file() or _excluded(p.name):
            continue
        out[p.name] = {
            "sha256": hash_file(p),
            "bytes": p.stat().st_size,
        }
    return out


def last_entry(log: Path = AUDIT_LOG) -> dict | None:
    if not log.exists():
        return None
    lines = [l for l in log.read_text().splitlines() if l.strip()]
    return json.loads(lines[-1]) if lines else None


def append(incident_id: str, incident_dir: Path, log: Path = AUDIT_LOG) -> dict:
    """Append a hash-chained entry and return it."""
    prev = last_entry(log)
    prev_hash = prev["chain_hash"] if prev else GENESIS

    files = manifest(incident_dir)
    entry = {
        "incident_id": incident_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "prev_hash": prev_hash,
        "files": files,
    }
    # The chain hash covers this entry *and* the previous chain hash, so altering
    # any earlier entry invalidates everything that follows.
    entry["chain_hash"] = hashlib.sha256(canonical_json(entry)).hexdigest()

    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def verify(log: Path = AUDIT_LOG, root: Path = Path("artifacts")) -> dict:
    """Re-walk the chain and re-hash on-disk files. Reports every break."""
    if not log.exists():
        return {"ok": False, "entries": 0, "errors": ["audit.log does not exist"]}

    entries = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    errors: list[str] = []
    prev_hash = GENESIS

    for i, e in enumerate(entries):
        if e["prev_hash"] != prev_hash:
            errors.append(
                f"entry {i} ({e['incident_id']}): prev_hash does not match entry {i - 1}"
            )
        body = {k: v for k, v in e.items() if k != "chain_hash"}
        recomputed = hashlib.sha256(canonical_json(body)).hexdigest()
        if recomputed != e["chain_hash"]:
            errors.append(f"entry {i} ({e['incident_id']}): chain_hash does not match its body")
        prev_hash = e["chain_hash"]

    # Only the newest entry per incident is expected to match what is on disk;
    # earlier entries describe superseded runs.
    newest: dict[str, dict] = {}
    for e in entries:
        newest[e["incident_id"]] = e
    for inc, e in newest.items():
        d = root / inc
        for name, rec in e["files"].items():
            p = d / name
            if not p.exists():
                errors.append(f"{inc}/{name}: recorded but missing from disk")
            elif hash_file(p) != rec["sha256"]:
                errors.append(f"{inc}/{name}: ON-DISK CONTENT DIFFERS from the recorded digest")

    return {
        "ok": not errors,
        "entries": len(entries),
        "head": entries[-1]["chain_hash"] if entries else GENESIS,
        "errors": errors,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Hash-chained artifact audit log.")
    ap.add_argument("--incident")
    ap.add_argument("--root", default="artifacts")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    if a.verify or not a.incident:
        r = verify(root=Path(a.root))
        print(f"entries : {r['entries']}")
        print(f"head    : {r.get('head', '-')}")
        print(f"status  : {'INTACT' if r['ok'] else 'BROKEN'}")
        for e in r["errors"]:
            print(f"  ! {e}")
        raise SystemExit(0 if r["ok"] else 1)

    e = append(a.incident, Path(a.root) / a.incident)
    print(f"incident   : {e['incident_id']}")
    print(f"files      : {len(e['files'])}")
    for n, rec in e["files"].items():
        print(f"  {rec['sha256'][:16]}...  {rec['bytes']:>10,}  {n}")
    print(f"prev hash  : {e['prev_hash']}")
    print(f"chain hash : {e['chain_hash']}")


if __name__ == "__main__":
    main()
