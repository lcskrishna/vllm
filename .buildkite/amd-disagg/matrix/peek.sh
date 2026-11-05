#!/bin/bash
# peek.sh <job-id> [lines]: bring-up milestones and recent errors for one matrix job.
source "$(dirname "$0")/crusoe.env"
L="${LOG_ROOT}"; j=$1; n=${2:-12}; d=$L/$j
MILESTONES='Loading weights took|Model loading took|Available KV cache|GPU KV cache size|Maximum concurrency|Graph capturing finished|init engine|Application startup complete|HSA_STATUS|OutOfMemory|out of memory|Traceback|(Error|Exception): |MORI|[Mm]ori[A-Za-z]*(Error|Fail|fail)|[Hh]eap|remote blocks'
ls -l "$d" 2>/dev/null | tail -n +2 | awk '{print $5, $9}'
for f in "$d"/prefill_*.log "$d"/decode_*.log "$d"/vllm_router.log; do
    [[ -f "$f" ]] || continue
    echo "== $(basename "$f") $(stat -c %s "$f")B, last write $(( $(date +%s) - $(stat -c %Y "$f") ))s ago"
    grep -aE "$MILESTONES" "$f" | grep -avE '(GET|POST) /' | tail -n "$n" | cut -c1-320
    echo "-- tail:"; tail -n 2 "$f" | cut -c1-320
done
echo "== job log"
grep -aE 'healthy|waiting on|PASS|FAIL|TIMEOUT|exited|ERROR' "$L/vllm-disagg-pd-$j.log" | tail -n 8 | cut -c1-320
