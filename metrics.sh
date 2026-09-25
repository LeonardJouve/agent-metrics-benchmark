#!/usr/bin/env bash
# 1-second cgroup v2 sampler: one JSON object per line until DURATION elapses or the
# process is killed (the runner never stops it; sandbox delete does).
#
# Stays inside the measured cgroup, so overhead must be tiny: no subprocess per sample.
# Timestamps come from EPOCHREALTIME and printf %(...)T, values from bash read loops.
# ponytail: one `sleep` fork per tick; replace with a long-lived reader if overhead matters.
set -u
export TZ=UTC

CG_ROOT="${CG_ROOT:-/sys/fs/cgroup}"
OUT="${OUT:-/tmp/metrics.jsonl}"
INTERVAL="${INTERVAL:-1}"
DURATION="${DURATION:-0}" # 0 = run until killed
ALLOC_CPU="${ALLOC_CPU:-0}"
ALLOC_MEM_GIB="${ALLOC_MEM_GIB:-0}"

# A sandbox normally sees its own cgroup as the root. Resolve the real path anyway so a
# non-namespaced layout still samples the right cgroup. One fork at startup is affordable.
if [[ ! -r "$CG_ROOT/cpu.stat" && -r /proc/self/cgroup ]]; then
  relative=$(sed -n 's/^0:://p' /proc/self/cgroup)
  if [[ -n "$relative" && -r "$CG_ROOT$relative/cpu.stat" ]]; then
    CG_ROOT="$CG_ROOT$relative"
  fi
fi

if [[ ! -r "$CG_ROOT/cpu.stat" ]]; then
  printf '{"error":"cgroup v2 cpu.stat not readable","cgroup_root":"%s"}\n' "$CG_ROOT" >"$OUT"
  exit 1
fi

to_us() { # decimal seconds -> integer microseconds, no subprocess
  local value=$1 seconds fraction
  seconds=${value%%.*}
  fraction=${value#*.}
  [[ "$value" == *.* ]] || fraction=0
  fraction="${fraction}000000"
  printf '%s' $((10#${seconds:-0} * 1000000 + 10#${fraction:0:6}))
}

NOW_US=0
clock_us() { # sets NOW_US from EPOCHREALTIME
  local stamp=$EPOCHREALTIME seconds fraction
  seconds=${stamp%.*}
  fraction=${stamp#*.}
  fraction="${fraction}000000"
  NOW_US=$((10#$seconds * 1000000 + 10#${fraction:0:6}))
}

INTERVAL_US=$(to_us "$INTERVAL")
DURATION_US=$(to_us "$DURATION")
clock_us
START_US=$NOW_US
tick=0

exec 3>>"$OUT"
while :; do
  usage=""; user=""; system=""; periods=""; throttled=""; throttled_usec=""
  while read -r key value; do
    case $key in
      usage_usec) usage=$value ;;
      user_usec) user=$value ;;
      system_usec) system=$value ;;
      nr_periods) periods=$value ;;
      nr_throttled) throttled=$value ;;
      throttled_usec) throttled_usec=$value ;;
    esac
  done <"$CG_ROOT/cpu.stat"

  high=""; max_events=""; oom=""; oom_kill=""
  while read -r key value; do
    case $key in
      high) high=$value ;;
      max) max_events=$value ;;
      oom) oom=$value ;;
      oom_kill) oom_kill=$value ;;
    esac
  done <"$CG_ROOT/memory.events"

  read -r cpu_max <"$CG_ROOT/cpu.max"
  read -r memory_current <"$CG_ROOT/memory.current"
  read -r memory_max <"$CG_ROOT/memory.max"
  peak=""
  if [[ -r "$CG_ROOT/memory.peak" ]]; then
    read -r peak <"$CG_ROOT/memory.peak"
  fi

  clock_us
  printf -v iso '%(%Y-%m-%dT%H:%M:%S)T' $((NOW_US / 1000000))
  printf -v epoch '%d.%06d' $((NOW_US / 1000000)) $((NOW_US % 1000000))
  printf '{"ts":"%sZ","epoch":%s,"usage_usec":%s,"user_usec":%s,"system_usec":%s,"nr_periods":%s,"nr_throttled":%s,"throttled_usec":%s,"cpu_max":"%s","memory_current":%s,"memory_max":"%s","memory_peak":%s,"mem_high":%s,"mem_max_events":%s,"oom":%s,"oom_kill":%s,"alloc_cpu":%s,"alloc_mem_gib":%s,"cgroup_root":"%s","cgroup_version":"v2"}\n' \
    "$iso" \
    "$epoch" \
    "${usage:-null}" "${user:-null}" "${system:-null}" \
    "${periods:-null}" "${throttled:-null}" "${throttled_usec:-null}" \
    "$cpu_max" "${memory_current:-null}" "$memory_max" "${peak:-null}" \
    "${high:-null}" "${max_events:-null}" "${oom:-null}" "${oom_kill:-null}" \
    "$ALLOC_CPU" "$ALLOC_MEM_GIB" "$CG_ROOT" >&3

  tick=$((tick + 1))
  if ((DURATION_US > 0)); then
    clock_us
    ((NOW_US - START_US >= DURATION_US)) && break
  fi

  clock_us
  target=$((START_US + tick * INTERVAL_US))
  if ((target > NOW_US)); then
    wait_us=$((target - NOW_US))
    printf -v nap '%d.%06d' $((wait_us / 1000000)) $((wait_us % 1000000))
    sleep "$nap"
  fi
done
