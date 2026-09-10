#!/usr/bin/env bash
# SAMUDRA end-to-end demo runner.
#
#   scripts/run_demo.sh [scenario-id] [options]
#
#   --regen        regenerate the scenario even if artifacts already exist
#   --no-serve     run the pipeline but do not start the API server
#   --port N       serve on N (default 8000)
#
# Every stage prints one line. Any stage failing stops the run.

set -euo pipefail

SCENARIO="${1:-demo-001}"
[[ "${SCENARIO}" == --* ]] && SCENARIO="demo-001" || { [[ $# -gt 0 ]] && shift || true; }

REGEN=0
SERVE=1
PORT=8000
VESSELS=40
DECOYS=7
SEED=""
WIND_MIN=4.0
WIND_MAX=9.0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --regen)    REGEN=1; shift ;;
    --no-serve) SERVE=0; shift ;;
    --port)     PORT="$2"; shift 2 ;;
    --vessels)  VESSELS="$2"; shift 2 ;;
    --decoys)   DECOYS="$2"; shift 2 ;;
    --seed)     SEED="$2"; shift 2 ;;
    --wind)     WIND_MIN="$2"; WIND_MAX="$3"; shift 3 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."

# Prefer the project virtualenv; fall back to whatever python is on PATH.
if [[ -x ".venv/Scripts/python.exe" ]]; then
  PY=".venv/Scripts/python.exe"
elif [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python"
fi

ART="artifacts/${SCENARIO}"
step=0
total=6

say() { step=$((step+1)); printf '[%d/%d] %-26s %s\n' "$step" "$total" "$1" "$2"; }

t_start=$(date +%s)
echo "=============================================================="
echo " SAMUDRA  |  scenario ${SCENARIO}"
echo "=============================================================="

# ---- 1. scenario ----------------------------------------------------------
if [[ $REGEN -eq 1 || ! -f "${ART}/ground_truth.json" ]]; then
  ARGS=(--scenario "${SCENARIO}" --vessels "${VESSELS}" --decoys "${DECOYS}"
        --wind-min "${WIND_MIN}" --wind-max "${WIND_MAX}")
  [[ -n "${SEED}" ]] && ARGS+=(--seed "${SEED}")
  rm -rf "${ART}"
  OUT=$("$PY" -m samudra.synth.generate "${ARGS[@]}")
  TRACKS=$(sed -n 's/^tracks generated *: *//p' <<<"$OUT" | head -1)
  AREA=$(sed -n 's/^slick area *: *//p' <<<"$OUT" | head -1)
  say "generate scenario" "${TRACKS} tracks, slick ${AREA}"
else
  say "generate scenario" "reusing existing ${ART} (--regen to rebuild)"
fi

# ---- 2. attribution -------------------------------------------------------
OUT=$("$PY" -m samudra.attribution --incident "${SCENARIO}")
FUNNEL=$(sed -n 's/^FUNNEL *//p' <<<"$OUT" | head -1)
say "attribute" "${FUNNEL}"

# ---- 3. trust -------------------------------------------------------------
TRUST=$(sed -n 's/^TRUST *//p' <<<"$OUT" | head -1)
say "trust and behaviour" "${TRUST}"

# ---- 4. forecast ----------------------------------------------------------
FOUT=$("$PY" -m samudra.impact.forecast --incident "${SCENARIO}")
IMPACT=$(grep -m1 'COASTLINE IMPACT' <<<"$FOUT" | sed 's/COASTLINE IMPACT: *//')
say "forecast and impact" "${IMPACT}"

# ---- 5. dossier -----------------------------------------------------------
DOUT=$("$PY" -m samudra.evidence.report --incident "${SCENARIO}")
say "evidence dossier" "$(sed 's/^written: *//' <<<"$DOUT")"

# ---- 6. audit -------------------------------------------------------------
AOUT=$("$PY" -m samudra.evidence.integrity --verify)
say "chain of custody" "$(grep -m1 'status' <<<"$AOUT" | sed 's/status *: *//')"

echo "--------------------------------------------------------------"
grep -A5 '^#  MMSI' <<<"$OUT" || true
echo "--------------------------------------------------------------"
printf 'completed in %ss\n' "$(( $(date +%s) - t_start ))"

if [[ $SERVE -eq 1 ]]; then
  echo
  echo "Dashboard: http://127.0.0.1:${PORT}    (ctrl-c to stop)"
  exec "$PY" -m samudra.api --port "${PORT}"
fi
