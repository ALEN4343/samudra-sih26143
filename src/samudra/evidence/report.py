"""Evidence dossier — CLAUDE.md section 4.6.

A PDF an investigator can act on and a court could scrutinise. Every number in
it comes from artifacts/<incident_id>/, and the final page carries the audit
chain hash so the document can be tied to the exact artifacts that produced it.

The synthetic banner is not decoration. If the scenario is synthetic the dossier
says so on page one, in the footer of every page, and in the findings.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Polygon as MplPolygon
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from samudra.evidence import integrity
from samudra.geo import load_polygon
from samudra.timeutil import epoch_seconds

INK = colors.HexColor("#111418")
MUTED = colors.HexColor("#5b6570")
RULE = colors.HexColor("#c8ced6")
ACCENT = colors.HexColor("#b3261e")
BAND = colors.HexColor("#eef1f5")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def scene_chip(incident_dir: Path, inc: dict, out_png: Path) -> Path | None:
    """SAR scene cropped to the slick, with the detected polygon overlaid."""
    tif = incident_dir / "scene.tif"
    if not tif.exists():
        return None

    poly = load_polygon(inc["observed_slick"]["geometry"])
    minx, miny, maxx, maxy = poly.bounds
    padx = max((maxx - minx) * 1.6, 0.10)
    pady = max((maxy - miny) * 1.6, 0.08)

    with rasterio.open(tif) as src:
        b = src.bounds
        w, h = src.width, src.height

        def px(lon, lat):
            return (
                (lon - b.left) / (b.right - b.left) * w,
                (b.top - lat) / (b.top - b.bottom) * h,
            )

        x0, y1 = px(minx - padx, miny - pady)
        x1, y0 = px(maxx + padx, maxy + pady)
        x0, x1 = max(int(min(x0, x1)), 0), min(int(max(x0, x1)), w)
        y0, y1 = max(int(min(y0, y1)), 0), min(int(max(y0, y1)), h)
        if x1 - x0 < 10 or y1 - y0 < 10:
            return None
        arr = src.read(1, window=((y0, y1), (x0, x1)))

    fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=170)
    ax.imshow(
        arr,
        cmap="gray",
        vmin=int(np.percentile(arr, 1)),
        vmax=int(np.percentile(arr, 99)),
        interpolation="nearest",
    )

    ring = np.asarray(poly.exterior.coords)
    pts = np.array([px(lon, lat) for lon, lat in ring])
    pts[:, 0] -= x0
    pts[:, 1] -= y0
    ax.add_patch(MplPolygon(pts, closed=True, fill=False, edgecolor="#ff3b30", lw=1.9))

    ax.set_title("SAR scene with detected slick boundary", fontsize=9, color="#111418")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#c8ced6")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png


def hypothesis_map(incident_dir: Path, inc: dict, out_png: Path) -> Path | None:
    """Winning hypothesis: observed vs simulated slick, envelope, culprit track."""
    if not inc["suspects"]:
        return None
    top = inc["suspects"][0]

    fig, ax = plt.subplots(figsize=(7.0, 5.6), dpi=170)

    for st in inc.get("envelope_steps", []):
        r = np.asarray(load_polygon(st["geometry"]).exterior.coords)
        ax.plot(r[:, 0], r[:, 1], color="#e0a800", lw=0.45, alpha=0.30)

    env = np.asarray(load_polygon(inc["origin_envelope"]).exterior.coords)
    ax.plot(env[:, 0], env[:, 1], color="#e0a800", lw=1.5, ls="--", label="Origin envelope")

    try:
        import pandas as pd

        df = pd.read_parquet(incident_dir / "ais.parquet")
        g = df[df.mmsi == top["mmsi"]].sort_values("t")
        ts = epoch_seconds(g.t)
        rel = datetime.fromisoformat(top["best_hypothesis"]["release_at"]).timestamp()
        acq = datetime.fromisoformat(inc["acquisition_at"]).timestamp()
        m = (ts >= rel - 6 * 3600) & (ts <= acq)
        ax.plot(
            g.lon.to_numpy()[m],
            g.lat.to_numpy()[m],
            color="#1f4e8c",
            lw=1.2,
            label=f"MMSI {top['mmsi']} track",
        )
    except Exception:
        pass  # the map is still useful without the track

    sim = np.asarray(
        load_polygon(top["best_hypothesis"]["simulated_geometry"]).exterior.coords
    )
    ax.fill(sim[:, 0], sim[:, 1], color="#1f4e8c", alpha=0.22)
    ax.plot(sim[:, 0], sim[:, 1], color="#1f4e8c", lw=1.4, label="Simulated slick")

    obs = np.asarray(load_polygon(inc["observed_slick"]["geometry"]).exterior.coords)
    ax.fill(obs[:, 0], obs[:, 1], color="#ff3b30", alpha=0.32)
    ax.plot(obs[:, 0], obs[:, 1], color="#b3261e", lw=1.6, label="Observed slick")

    b = top["best_hypothesis"]
    ax.plot(
        [b["release_lon"]],
        [b["release_lat"]],
        marker="*",
        ms=15,
        color="#111418",
        label="Estimated release point",
    )

    # Frame the slick and envelope, not the vessel's whole transit. The track
    # crosses the entire AOI, so autoscaling shrinks the evidence to a speck.
    xs = list(obs[:, 0]) + list(sim[:, 0]) + list(env[:, 0])
    ys = list(obs[:, 1]) + list(sim[:, 1]) + list(env[:, 1])
    padx = (max(xs) - min(xs)) * 0.28 + 0.02
    pady = (max(ys) - min(ys)) * 0.28 + 0.02
    ax.set_xlim(min(xs) - padx, max(xs) + padx)
    ax.set_ylim(min(ys) - pady, max(ys) + pady)

    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("Winning release hypothesis", fontsize=9)
    ax.legend(fontsize=7, loc="best", framealpha=0.92)
    ax.grid(alpha=0.18, lw=0.5)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------


def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "t", parent=ss["Title"], fontSize=19, leading=23, textColor=INK, spaceAfter=2
        ),
        "sub": ParagraphStyle(
            "s", parent=ss["Normal"], fontSize=9.5, textColor=MUTED, spaceAfter=10
        ),
        "h": ParagraphStyle(
            "h", parent=ss["Heading2"], fontSize=11.5, textColor=INK,
            spaceBefore=13, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "b", parent=ss["Normal"], fontSize=9, leading=13, textColor=INK,
            alignment=TA_JUSTIFY,
        ),
        "small": ParagraphStyle(
            "sm", parent=ss["Normal"], fontSize=7.6, leading=10.4, textColor=MUTED
        ),
        "mono": ParagraphStyle(
            "m", parent=ss["Normal"], fontSize=7.4, leading=10,
            fontName="Courier", textColor=INK,
        ),
        "warn": ParagraphStyle(
            "w", parent=ss["Normal"], fontSize=9, leading=12.5, textColor=ACCENT
        ),
    }


def _table(rows, widths, header=True, align_right=None):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), BAND),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
        ]
    for c in align_right or []:
        style.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


def build(incident_id: str, root: Path = Path("artifacts")) -> Path:
    d = root / incident_id
    inc = json.loads((d / "incident.json").read_text())
    S = _styles()
    synthetic = inc.get("synthetic", False)

    # Record the artifacts first so the dossier can print its own chain hash.
    entry = integrity.append(incident_id, d)

    out_pdf = d / f"dossier_{incident_id}.pdf"
    tmp = d / "_fig"
    tmp.mkdir(exist_ok=True)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 6.8)
        canvas.setFillColor(MUTED)
        tag = "SYNTHETIC SCENARIO - NOT A REAL INCIDENT" if synthetic else "OFFICIAL"
        canvas.drawString(18 * mm, 12 * mm, f"SAMUDRA  |  {incident_id}  |  {tag}")
        canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, f"Page {doc.page}")
        canvas.setStrokeColor(RULE)
        canvas.line(18 * mm, 15 * mm, A4[0] - 18 * mm, 15 * mm)
        canvas.restoreState()

    doc = BaseDocTemplate(
        str(out_pdf),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=20 * mm,
        title=f"SAMUDRA dossier {incident_id}",
    )
    doc.addPageTemplates(
        [
            PageTemplate(
                id="all",
                frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")],
                onPage=footer,
            )
        ]
    )

    F = []
    A = F.append

    # ---- header ----------------------------------------------------------
    A(Paragraph("Oil Discharge Attribution Dossier", S["title"]))
    A(Paragraph(f"Incident {incident_id}", S["sub"]))
    if synthetic:
        A(
            Paragraph(
                "<b>SYNTHETIC SCENARIO.</b> This incident was generated with a known "
                "planted culprit to verify the attribution engine end to end. The vessel "
                "identities, AIS traffic and SAR scene are simulated. Nothing here "
                "describes a real vessel or a real discharge.",
                S["warn"],
            )
        )
        A(Spacer(1, 7))

    acq = datetime.fromisoformat(inc["acquisition_at"])
    aoi = inc["aoi_bounds"]
    sl = inc["observed_slick"]
    A(
        _table(
            [
                ["Acquisition (UTC)", acq.strftime("%Y-%m-%d %H:%M:%S")],
                [
                    "Area of interest",
                    f"{aoi[0]:.2f}E {aoi[1]:.2f}N to {aoi[2]:.2f}E {aoi[3]:.2f}N",
                ],
                ["Slick area", f"{sl['area_km2']:.1f} km2"],
                ["Slick centroid", f"{sl['centroid_lat']:.4f}N, {sl['centroid_lon']:.4f}E"],
                ["Long axis bearing", f"{sl['major_axis_deg']:.0f} deg from north"],
                ["Eccentricity", f"{sl['eccentricity']:.3f}"],
                ["Shape complexity", f"{sl['shape_complexity']:.2f}"],
                ["Report generated", datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")],
            ],
            [46 * mm, 118 * mm],
            header=False,
        )
    )

    # ---- finding ---------------------------------------------------------
    A(Paragraph("1. Finding", S["h"]))
    if inc["suspects"]:
        t = inc["suspects"][0]
        bh = t["best_hypothesis"]
        rel = datetime.fromisoformat(bh["release_at"])
        A(
            Paragraph(
                f"The observed slick is most consistent with a discharge from "
                f"<b>{t['vessel_name'] or 'unknown vessel'}</b> (MMSI {t['mmsi']}"
                f"{', ' + t['vessel_type'] if t.get('vessel_type') else ''}"
                f"{', flag ' + t['flag'] if t.get('flag') else ''}) beginning at "
                f"approximately <b>{rel.strftime('%Y-%m-%d %H:%M UTC')}</b> near "
                f"{bh['release_lat']:.4f}N, {bh['release_lon']:.4f}E. "
                f"Posterior probability <b>{t['posterior']:.3f}</b> across "
                f"{inc['funnel']['ranked']} ranked candidates.",
                S["body"],
            )
        )
        A(Spacer(1, 5))
        A(
            Paragraph(
                "This is a physics-based consistency ranking, not proof. It states which "
                "vessel's own reported track best explains the observed slick under the "
                "modelled wind and current field. It should be corroborated with port "
                "state inspection, oil record book review and, where available, oil "
                "fingerprinting.",
                S["small"],
            )
        )

    # ---- scene chip ------------------------------------------------------
    A(Paragraph("2. Detected slick", S["h"]))
    chip = scene_chip(d, inc, tmp / "chip.png")
    if chip:
        A(Image(str(chip), width=doc.width * 0.86, height=doc.width * 0.86 * 0.74))
    else:
        A(Paragraph("SAR scene unavailable for this incident.", S["small"]))

    # ---- environment -----------------------------------------------------
    A(Paragraph("3. Environmental conditions at acquisition", S["h"]))
    e = inc["env_summary"]
    A(
        _table(
            [
                ["Parameter", "Value", "Note"],
                ["Mean wind speed", f"{e['mean_wind_speed_ms']:.1f} m/s",
                 "Detection window 3-10 m/s"],
                ["Mean wind direction", f"{e['mean_wind_dir_deg']:.0f} deg", "Toward"],
                ["Mean current speed", f"{e['mean_current_speed_ms']:.2f} m/s", ""],
                ["Mean current direction", f"{e['mean_current_dir_deg']:.0f} deg", "Toward"],
                [
                    "Wind gate",
                    "PASS" if e["wind_gate_pass"] else "FAIL",
                    "Below 3 m/s the sea is too flat; above 10 m/s oil is mixed away",
                ],
            ],
            [42 * mm, 30 * mm, 92 * mm],
            align_right=[1],
        )
    )

    # ---- funnel ----------------------------------------------------------
    A(Paragraph("4. Candidate funnel", S["h"]))
    f = inc["funnel"]
    A(
        _table(
            [
                ["Stage", "Vessels", "Basis"],
                ["In scene", str(f["total_in_scene"]),
                 "Reporting inside the AOI during the search window"],
                ["In origin envelope", str(f["in_envelope"]),
                 "Intersects the reverse-drift envelope in space and time"],
                ["Scored", str(f["scored"]),
                 "Produced at least one testable release hypothesis"],
                ["Ranked", str(f["ranked"]), "Assigned a posterior"],
            ],
            [42 * mm, 22 * mm, 100 * mm],
            align_right=[1],
        )
    )

    # ---- age -------------------------------------------------------------
    A(Paragraph("5. Estimated slick age", S["h"]))
    a = inc["slick_age"]
    A(
        _table(
            [
                ["Estimate", "Hours", "Method"],
                ["Best hypothesis", f"{a['best_hours']:.1f}",
                 "Acquisition minus estimated release"],
                ["Range across top 3", f"{a['low_hours']:.1f} - {a['high_hours']:.1f}",
                 "Hypothesis spread"],
                ["Fay cross-check", f"{a['fay_estimate_hours']:.1f}",
                 "Inverse gravity-viscous spreading"],
                ["Agreement", "YES" if a["agrees_with_fay"] else "NO",
                 "Within the configured tolerance"],
            ],
            [42 * mm, 30 * mm, 92 * mm],
            align_right=[1],
        )
    )
    A(Spacer(1, 4))
    A(
        Paragraph(
            "The Fay figure assumes a reference discharge volume and oil type set in "
            "config/weights.yaml. It is a consistency check against an assumed spreading "
            "rate, not an independent measurement, and should not be presented as "
            "corroboration.",
            S["small"],
        )
    )

    # ---- suspects --------------------------------------------------------
    A(PageBreak())
    A(Paragraph("6. Ranked candidates", S["h"]))
    rows = [["#", "MMSI", "Vessel", "Post.", "Score", "IoU", "dCentre", "dOrient", "Area"]]
    for s in inc["suspects"]:
        b = s["best_hypothesis"]
        rows.append(
            [
                str(s["rank"]),
                str(s["mmsi"]),
                (s["vessel_name"] or "-")[:20],
                f"{s['posterior']:.3f}",
                f"{b['score']:.3f}",
                f"{b['iou']:.3f}",
                f"{b['centroid_offset_km']:.1f} km",
                f"{b['orientation_delta_deg']:.0f} deg",
                f"{b['area_ratio']:.2f}",
            ]
        )
    A(
        _table(
            rows,
            [8 * mm, 24 * mm, 38 * mm, 16 * mm, 16 * mm, 15 * mm, 18 * mm, 17 * mm, 12 * mm],
            align_right=[3, 4, 5, 6, 7, 8],
        )
    )
    A(Spacer(1, 4))
    A(
        Paragraph(
            "Score = 0.40 IoU + 0.25 centroid + 0.25 orientation + 0.10 area, weights "
            "from config/weights.yaml. Posterior is a softmax over score multiplied by "
            "trust, behaviour and proximity priors.",
            S["small"],
        )
    )

    if inc["suspects"]:
        A(Paragraph("6.1 Rationale for the leading candidate", S["h"]))
        for line in inc["suspects"][0]["rationale"]:
            A(Paragraph(f"&bull; {line}", S["body"]))
            A(Spacer(1, 1.5))

    # ---- hypothesis map --------------------------------------------------
    A(Paragraph("7. Winning hypothesis", S["h"]))
    hm = hypothesis_map(d, inc, tmp / "hyp.png")
    if hm:
        A(Image(str(hm), width=doc.width * 0.90, height=doc.width * 0.90 * 0.80))

    # ---- flags -----------------------------------------------------------
    A(PageBreak())
    A(Paragraph("8. Trust and behaviour flags", S["h"]))
    frows = [["MMSI", "Class", "Type", "Sev", "Detail"]]
    for s in inc["suspects"]:
        for fl in s.get("trust_flags", []) + s.get("behaviour_flags", []):
            frows.append(
                [
                    str(s["mmsi"]),
                    s.get("classification", ""),
                    fl["code"],
                    fl["severity"][:4],
                    Paragraph(fl["detail"], S["small"]),
                ]
            )
    if len(frows) == 1:
        A(
            Paragraph(
                "No trust or behaviour flags were raised against any ranked candidate.",
                S["body"],
            )
        )
    else:
        A(_table(frows, [24 * mm, 26 * mm, 34 * mm, 12 * mm, 68 * mm]))

    ts = inc.get("trust_summary")
    if ts:
        A(Spacer(1, 4))
        A(
            Paragraph(
                f"{ts['vessels_flagged']} of {ts['vessels_scored']} vessels in the scene "
                f"carry at least one flag. Vessels classified IDENTITY_MISMATCH: "
                f"{ts['identity_mismatch'] or 'none'}.",
                S["small"],
            )
        )

    # ---- forecast --------------------------------------------------------
    A(Paragraph("9. Forecast and impact", S["h"]))
    frows = [["Horizon", "Area", "Cone", "To shore", "Shore contact"]]
    any_fc = False
    for h in (24, 48, 72):
        p = d / f"forecast_{h}h.geojson"
        if not p.exists():
            continue
        any_fc = True
        feats = json.loads(p.read_text())["features"]
        props = next(x["properties"] for x in feats if x["properties"]["kind"] == "forecast")
        cone = next(
            x["properties"] for x in feats if x["properties"]["kind"] == "uncertainty_cone"
        )
        contact = "none"
        if props.get("coastline_intersects"):
            eta = (props.get("coastline_eta") or "")[:16]
            contact = f"{props['affected_shoreline_km']:.1f} km, ETA {eta}"
        frows.append(
            [
                f"+{h} h",
                f"{props['area_km2']:.1f} km2",
                f"{cone['area_km2']:.1f} km2",
                f"{props.get('distance_to_coast_km', 0):.0f} km",
                contact,
            ]
        )
    if any_fc:
        A(_table(frows, [20 * mm, 28 * mm, 28 * mm, 24 * mm, 64 * mm], align_right=[1, 2, 3]))
    else:
        A(Paragraph("Forecast not computed for this incident.", S["small"]))

    # ---- MARPOL ----------------------------------------------------------
    A(Paragraph("10. Regulatory reference - MARPOL Annex I", S["h"]))
    for para in [
        "Annex I of the International Convention for the Prevention of Pollution from "
        "Ships (MARPOL 73/78) governs the prevention of pollution by oil and entered "
        "into force on 2 October 1983.",
        "<b>Regulation 15</b> controls the discharge of oil from machinery space bilges. "
        "Outside special areas, discharge is permitted only where the ship is proceeding "
        "en route, the effluent passes through approved oil filtering equipment, and the "
        "oil content of the effluent without dilution does not exceed 15 parts per "
        "million.",
        "<b>Regulation 34</b> controls discharge from the cargo area of oil tankers. "
        "Outside special areas, conditions include that the tanker is more than 50 "
        "nautical miles from the nearest land, is proceeding en route, and discharges at "
        "an instantaneous rate not exceeding 30 litres per nautical mile, subject to "
        "total quantity limits and operation of an approved oil discharge monitoring and "
        "control system.",
        "<b>Special areas.</b> Annex I designates special areas in which discharge is far "
        "more tightly restricted, and in which any discharge of oil or oily mixtures from "
        "oil tankers and from ships of 400 gross tonnage and above is prohibited. The "
        "designated special areas include the Oman area of the Arabian Sea, the Gulfs "
        "area, the Gulf of Aden and the Red Sea. Whether a given position falls inside a "
        "special area must be confirmed against the coordinates in the Annex before any "
        "enforcement step is taken.",
        "<b>Evidentiary status.</b> This dossier is decision support for targeting "
        "inspection. It does not establish that a discharge was unlawful. Establishing an "
        "offence requires confirmation of the substance discharged, the quantity, the "
        "ship's position relative to special area boundaries, and the state of its "
        "pollution prevention equipment, ordinarily through port state control inspection "
        "and examination of the Oil Record Book.",
    ]:
        A(Paragraph(para, S["body"]))
        A(Spacer(1, 4))

    # ---- chain of custody ------------------------------------------------
    A(PageBreak())
    A(Paragraph("11. Chain of custody", S["h"]))
    A(
        Paragraph(
            "SHA-256 over the canonical JSON of every pipeline artifact, appended to "
            "artifacts/audit.log and hash-chained to the previous entry. Recomputing "
            "these digests verifies that no artifact was altered after this dossier was "
            "produced. Altering any earlier entry invalidates every chain hash that "
            "follows it.",
            S["body"],
        )
    )
    A(Spacer(1, 6))
    hrows = [["Artifact", "Bytes", "SHA-256"]]
    for name, rec in entry["files"].items():
        hrows.append([name, f"{rec['bytes']:,}", Paragraph(rec["sha256"], S["mono"])])
    A(_table(hrows, [42 * mm, 22 * mm, 100 * mm], align_right=[1]))
    A(Spacer(1, 8))
    A(
        _table(
            [
                ["Recorded at (UTC)", Paragraph(entry["recorded_at"], S["mono"])],
                ["Previous chain hash", Paragraph(entry["prev_hash"], S["mono"])],
                ["Chain hash", Paragraph(f"<b>{entry['chain_hash']}</b>", S["mono"])],
            ],
            [42 * mm, 122 * mm],
            header=False,
        )
    )

    doc.build(F)

    for p in tmp.glob("*.png"):
        p.unlink()
    tmp.rmdir()
    return out_pdf


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the evidence dossier PDF.")
    ap.add_argument("--incident", required=True)
    ap.add_argument("--root", default="artifacts")
    a = ap.parse_args()
    p = build(a.incident, Path(a.root))
    print(f"written: {p}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
