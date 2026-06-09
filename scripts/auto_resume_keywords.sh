#!/usr/bin/env bash
# Auto-resume the 0.2 sequential keyword run until all 401 sources succeed (or caps hit).
#
# Each run of run_keywords_sequential.py attempts every not-yet-done source once, with
# fail-fast retries on transient 503/429 overload (deferring the rest). This loop just
# re-invokes it between sleeps to mop up the deferred ones — and because it keeps running
# for hours, it naturally sweeps through the low-traffic window (~05:00 IDT) where the
# success rate is high and the backlog clears fast. Nothing is lost between passes: the
# runner resumes from its vocab snapshot, so suggested-keyword ids stay consistent.
#
# Run it yourself, detached, so it survives closing the editor:
#     nohup bash scripts/auto_resume_keywords.sh > /tmp/seq_0_2_auto.log 2>&1 &
#
# Tune via env: MAX_PASSES (default 40), SLEEP seconds between passes (default 900).
set -u
cd "$(dirname "$0")/.."

RESULTS=results/keywords_sequential_0_2/gemini_KEYWORDS_0_2_SEQ.json
TOTAL=401
MAX_PASSES=${MAX_PASSES:-40}
SLEEP=${SLEEP:-900}

done_count() {
  python3 -c "import json;d=json.load(open('$RESULTS'));print(sum(1 for r in d if r.get('response') is not None))" 2>/dev/null || echo 0
}

prev=-1
stall=0
for pass in $(seq 1 "$MAX_PASSES"); do
  echo "===== PASS $pass / $MAX_PASSES   $(date '+%F %T %Z') ====="
  python3 scripts/run_keywords_sequential.py
  d=$(done_count)
  echo "===== after pass $pass: ${d}/${TOTAL} succeeded ====="

  if [ "$d" -ge "$TOTAL" ]; then echo "ALL ${TOTAL} DONE — stopping."; break; fi
  if [ "$d" -le "$prev" ]; then stall=$((stall + 1)); else stall=0; fi
  prev=$d
  if [ "$stall" -ge 5 ]; then
    echo "No progress in 5 consecutive passes — stopping (sustained outage or persistent errors)."
    break
  fi
  echo "sleeping ${SLEEP}s before next pass ..."
  sleep "$SLEEP"
done
echo "auto-resume finished at $(date '+%F %T %Z'): $(done_count)/${TOTAL} succeeded"
