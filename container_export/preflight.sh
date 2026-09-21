#!/usr/bin/env bash
# =============================================================================
# preflight.sh — one-command PASS/FAIL gate for a CHEAP Vast.ai GPU
# =============================================================================
#
# Purpose
# -------
# Prove the whole ForAINet chain works on a cheap, supported GPU (T4 / RTX 3090,
# ~$0.10-0.30/hr) BEFORE renting the expensive A100. It exercises exactly the two
# things that are costly to discover on a paid machine:
#
#   1. the LAZ->PLY converter + the build-time smoke test, and
#   2. training STARTUP for both backbones (MinkowskiEngine + TorchSparse),
#
# and it stops at the first failure with a clear message. A green run here means
# the A100 run will not fall over on a broken data path, a missing LAZ backend,
# an unapplied 4-class patch, or an unsupported GPU.
#
# It is the scripted form of docs/preflight_cheap_gpu.md — read that for the why
# behind each step; this file is the what.
#
# How the instance is wired (env vars set at Vast instance creation)
# ------------------------------------------------------------------
#   DATASET_TRAIN_URL  archive of the modified SegmentedForest dataset (14 .laz + offsets)
#   DATASET_PATCH      archive with the 4-class/TorchSparse patch AND the converter code
#   WANDB_API_KEY      (optional) only used by step F, the wandb validation
#
# The dataset is fetched ONCE. After a successful extraction step D records a sha256 for
# every extracted file (plus the URL it came from) in $DATA/.preflight_manifest.sha256, and
# later runs re-verify those checksums instead of pulling 4.3 GB again -- which matters
# whenever /data survives the run: a mounted volume, or just re-running the gate on the same
# box. A changed URL, a missing file or a single bad checksum re-downloads everything.
# DATASET_PATCH is deliberately NOT cached: it is ~50 KB and it is the thing you iterate on,
# so a stale copy of it would defeat the point of re-running the gate.
#
# What it repairs by itself (a rented box is disposable; do not hand-fix it)
# -------------------------------------------------------------------------
# Most rented images are ENV-ONLY: the conda/CUDA stack is there, the ForAINet source is
# not (locally it arrived through the ./ForAINet:/workspace bind mount, which does not
# exist on a rented host). Rather than stop, step B:
#
#   * finds the converter wherever the archive unpacked it (with or without a nested
#     container_export/ folder),
#   * clones ForAINet at the pinned commit if no train.py is on the box, and re-derives
#     every downstream path from where it landed,
#   * applies patches/forainet-local.patch and verifies 4 classes + the TorchSparse model,
#   * deletes the patch's debug-only `import debugpy` line when debugpy is not installed,
#   * installs the pinned laspy (2.5.3) + the lazrs LAZ backend the base image lacks.
#
# Each of those has an opt-out below.
#
# Tunables (all optional, sane defaults)
# --------------------------------------
#   PREFLIGHT_DRYRUN=1        print every command without running it (works with no GPU/Docker)
#   PREFLIGHT_PLOTS="a b c"   convert only these stems (faster/cheaper); default = all 14
#   PREFLIGHT_EPOCHS=2        epochs per training run (2 is enough to prove startup)
#   PREFLIGHT_WANDB=1         also run step F (a short wandb.log=True run to validate the key)
#   PREFLIGHT_SKIP_TRAIN=1    stop after the converter+smoke checks (scope #1 only)
#   PREFLIGHT_ALLOW_NOGPU=1   don't hard-fail when nvidia-smi is absent (for dry inspection)
#   PREFLIGHT_PY=python3.8    interpreter to use inside the image
#   PREFLIGHT_FORAINET_DIR    where ForAINet is (or should be cloned to) — the git root that
#                             CONTAINS PointCloudSegmentation/. Default: found, else
#                             /workspace/ForAINet
#   PREFLIGHT_FORAINET_REF    commit to check out (default: the SHA this repo pins; the patch
#                             does not apply to upstream main)
#   PREFLIGHT_FORAINET_URL    clone source (default: the upstream GitHub repo)
#   PREFLIGHT_NO_CLONE=1      never clone — fail instead if the image has no ForAINet
#   PREFLIGHT_NO_PIP=1        never pip install — an old laspy becomes a WARN, not a repair
#   PREFLIGHT_LASPY_SPEC      what to pip install for laspy (default: laspy[lazrs]==2.5.3).
#                             Drop the ==pin to stop checking the version at all
#   PREFLIGHT_FORCE_FETCH=1   ignore the cached dataset and re-download $DATASET_TRAIN_URL
#   PREFLIGHT_ADOPT_EXISTING=0  do not trust a /data that has clouds but no manifest
#                             (default 1: adopt them once, with a WARN, then checksum them)
#   PREFLIGHT_DATA_DIR        where the dataset lands (default /data)
#   PREFLIGHT_PREP_DIR        where DATASET_PATCH unpacks (default /opt/prep)
#
# Usage
# -----
#   bash preflight.sh                       # full gate, all 14 plots, both backbones
#   PREFLIGHT_PLOTS="plot_02 plot_11 plot_01" bash preflight.sh   # 3-plot fast gate
#   PREFLIGHT_DRYRUN=1 bash preflight.sh    # show the plan, run nothing
#
# Exit code 0 = every hard check passed -> cleared to rent the A100.
# =============================================================================

set -uo pipefail

# --- configuration -----------------------------------------------------------
PY="${PREFLIGHT_PY:-python3.8}"
DRYRUN="${PREFLIGHT_DRYRUN:-0}"
EPOCHS="${PREFLIGHT_EPOCHS:-2}"
DO_WANDB="${PREFLIGHT_WANDB:-0}"
SKIP_TRAIN="${PREFLIGHT_SKIP_TRAIN:-0}"
ALLOW_NOGPU="${PREFLIGHT_ALLOW_NOGPU:-0}"
PLOTS="${PREFLIGHT_PLOTS:-}"          # empty = all plots
NO_CLONE="${PREFLIGHT_NO_CLONE:-0}"
NO_PIP="${PREFLIGHT_NO_PIP:-0}"
FORCE_FETCH="${PREFLIGHT_FORCE_FETCH:-0}"
ADOPT="${PREFLIGHT_ADOPT_EXISTING:-1}"

# laspy, pinned. The base image ships 2.0.3 with no LAZ backend at all; the previous
# `pip install --upgrade "laspy[lazrs]"` floated to whatever pip resolved that day, so two
# runs of the same gate could install two different readers. 2.5.3 is a version that
# installs under the image's Python 3.8 and has both the lazrs backend and
# header.parse_crs. If the pin cannot be satisfied the install below retries unpinned
# rather than failing the gate — a working newer laspy beats no laspy.
LASPY_SPEC="${PREFLIGHT_LASPY_SPEC:-laspy[lazrs]==2.5.3}"
case "$LASPY_SPEC" in
  *==*) LASPY_WANT="${LASPY_SPEC##*==}" ;;   # the version step B2 asserts afterwards
  *)    LASPY_WANT="" ;;                     # unpinned spec: capability checks only
esac

# Where ForAINet comes from when the image does not carry it. The SHA is the commit this
# repo's submodule pins; forainet-local.patch is a diff against exactly that tree.
FORAINET_URL="${PREFLIGHT_FORAINET_URL:-https://github.com/prs-eth/ForAINet.git}"
FORAINET_REF="${PREFLIGHT_FORAINET_REF:-5fe600ae8f2fe913ae8740f475f0261a702f2a72}"

# Paths inside the image. Overridable so the gate can also run somewhere that is not the
# container's root filesystem (a mounted volume, a laptop check-out).
PREP="${PREFLIGHT_PREP_DIR:-/opt/prep}"                      # converter + patch land here
PREP_ROOT="$PREP"                                            # resolved in step B (nested archive?)
DATA="${PREFLIGHT_DATA_DIR:-/data}"                          # dataset archive extracts here

# ForAINet paths. These are only DEFAULTS — the bind-mounted layout of the local image.
# locate_forainet() overwrites FORAINET_ROOT/PCS from what is actually on the box, then
# set_paths() re-derives the four that hang off $PCS. Nothing below may hardcode them:
# a cloned tree lives at /workspace/ForAINet, one level deeper than the bind mount.
FORAINET_ROOT="/workspace"                                   # git root; where `git apply` runs
PCS="/workspace/PointCloudSegmentation"                      # train.py cwd + data root
set_paths() {
  RAW="${PCS}/data_set1_5classes/treeinsfused/raw/SegmentedForests"   # ForAINet globs raw/**/*.ply
  CACHE="${PCS}/data_set1_5classes/treeinsfused/processed_0.2"        # stale-cache trap
  SEG_DS="${PCS}/torch_points3d/datasets/segmentation/treeins_set1.py"
  TS_MODEL="${PCS}/torch_points3d/models/panoptic/PointGroup3heads_ts.py"
}
set_paths

# Compute capabilities the image was built for (TORCH_CUDA_ARCH_LIST, no +PTX).
SUPPORTED_CC="6.0 7.0 7.5 8.0 8.6"

# --- pretty output + failure tracking ---------------------------------------
BOLD=$(printf '\033[1m'); RED=$(printf '\033[31m'); GRN=$(printf '\033[32m')
YLW=$(printf '\033[33m'); DIM=$(printf '\033[2m'); RST=$(printf '\033[0m')
# Disable colour if not a terminal.
if [ ! -t 1 ]; then BOLD=; RED=; GRN=; YLW=; DIM=; RST=; fi

FAILED=0
declare -a RESULTS=()

section() { printf '\n%s========== %s ==========%s\n' "$BOLD" "$1" "$RST"; }
info()    { printf '%s..%s %s\n' "$DIM" "$RST" "$1"; }
ok()      { printf '%s[OK]%s   %s\n' "$GRN" "$RST" "$1"; RESULTS+=("OK   $1"); }
warn()    { printf '%s[WARN]%s %s\n' "$YLW" "$RST" "$1"; RESULTS+=("WARN $1"); }
fail()    { printf '%s[FAIL]%s %s\n' "$RED" "$RST" "$1"; RESULTS+=("FAIL $1"); FAILED=1; }

# run CMD... — echo it, then execute (unless dry run).
run() {
  printf '%s$ %s%s\n' "$DIM" "$*" "$RST"
  if [ "$DRYRUN" = "1" ]; then return 0; fi
  "$@"
}
# run_sh 'shell string' — same, for pipelines/redirection.
#
# `-o pipefail` is NOT optional and NOT inherited: `set -o pipefail` at the top of this
# file belongs to THIS shell, and `bash -c` starts a fresh one. Without it every
# `train.py ... | tee log` reports tee's exit status, so a training run that died on its
# first line ("can't open file 'train.py'") was reported as a pass — the one failure mode
# that makes this whole gate worthless.
run_sh() {
  printf '%s$ %s%s\n' "$DIM" "$1" "$RST"
  if [ "$DRYRUN" = "1" ]; then return 0; fi
  bash -o pipefail -c "$1"
}

# --- download + extract any archive type ------------------------------------
download() {   # download URL DEST
  local url="$1" dest="$2"
  if command -v wget >/dev/null 2>&1; then
    run wget -q --show-progress -O "$dest" "$url"
  elif command -v curl >/dev/null 2>&1; then
    run curl -fL --progress-bar -o "$dest" "$url"
  else
    fail "neither wget nor curl is available to fetch $url"; return 1
  fi
}

extract() {   # extract ARCHIVE DESTDIR — detects .zip/.tar.gz/.tar/.7z, sniffs if unsure
  local arc="$1" dest="$2"
  [ "$DRYRUN" = "1" ] || mkdir -p "$dest"
  case "$arc" in
    *.zip)              run_sh "unzip -o -q '$arc' -d '$dest'" ;;
    *.tar.gz|*.tgz)     run_sh "tar -xzf '$arc' -C '$dest'" ;;
    *.tar)              run_sh "tar -xf '$arc' -C '$dest'" ;;
    *.7z)               run_sh "7z x -y -o'$dest' '$arc'" ;;
    *)  # unknown extension (presigned URLs often have none): sniff by content
        if [ "$DRYRUN" = "1" ]; then info "would sniff archive type of $arc"; return 0; fi
        if   tar -tzf "$arc" >/dev/null 2>&1; then run_sh "tar -xzf '$arc' -C '$dest'"
        elif tar -tf  "$arc" >/dev/null 2>&1; then run_sh "tar -xf  '$arc' -C '$dest'"
        elif unzip -t "$arc" >/dev/null 2>&1; then run_sh "unzip -o -q '$arc' -d '$dest'"
        elif command -v 7z >/dev/null 2>&1 && 7z l "$arc" >/dev/null 2>&1; then run_sh "7z x -y -o'$dest' '$arc'"
        else fail "cannot determine archive type of $arc"; return 1
        fi ;;
  esac
}

fetch_and_extract() {   # fetch_and_extract URL DESTDIR LABEL
  local url="$1" dest="$2" label="$3"
  if [ -z "$url" ]; then fail "$label: env var is empty/unset"; return 1; fi
  local fname; fname="$(basename "${url%%\?*}")"       # strip any ?query
  [ -n "$fname" ] || fname="${label}.archive"
  local tmp="/tmp/${fname}"
  info "$label -> $tmp"
  download "$url" "$tmp" || return 1
  extract "$tmp" "$dest" || return 1
}

# count files matching a glob under a dir (recursive); prints an integer
count() { if [ "$DRYRUN" = "1" ]; then echo "?"; else find "$1" -type f -name "$2" 2>/dev/null | wc -l | tr -d ' '; fi; }

# --- fetch once, then verify: the dataset cache ------------------------------
# The dataset is 4.3 GB. Downloading it again on every run of the gate is the single
# slowest thing here, and it is pure waste whenever the previous copy is still on disk.
# So after a good extraction we write, INSIDE the destination directory:
#
#   .preflight_manifest.sha256   sha256 of every extracted file, paths relative to $DATA
#   .preflight_source            the URL those files came from
#
# and a later run re-verifies that manifest instead of re-fetching. Checksums, not
# timestamps or sizes: a half-written .laz from an interrupted extraction has a plausible
# size and would otherwise sail through into the converter.
MANIFEST=".preflight_manifest.sha256"
SRCMARK=".preflight_source"

manifest_write() {   # manifest_write DEST URL
  local dest="$1" url="$2"
  ( cd "$dest" 2>/dev/null || exit 1
    find . -type f ! -name "$MANIFEST" ! -name "$SRCMARK" -print0 \
      | xargs -0 -r sha256sum > "$MANIFEST" ) \
    || { warn "could not write $dest/$MANIFEST — the next run will re-download"; return 1; }
  printf '%s\n' "$url" > "$dest/$SRCMARK"
  info "recorded $(wc -l < "$dest/$MANIFEST" | tr -d ' ') checksum(s) in $dest/$MANIFEST"
}

# 0 = there is already a good copy in DEST; non-zero = fetch it.
dataset_cached() {   # dataset_cached DEST URL LABEL GLOB
  local dest="$1" url="$2" label="$3" glob="$4" n bad found
  [ -d "$dest" ] || return 1

  if [ -s "$dest/$MANIFEST" ]; then
    if [ ! -r "$dest/$SRCMARK" ] || [ "$(cat "$dest/$SRCMARK" 2>/dev/null)" != "$url" ]; then
      warn "$label: the cached copy in $dest came from a different URL — re-downloading"
      return 1
    fi
    n="$(wc -l < "$dest/$MANIFEST" | tr -d ' ')"
    info "$label: verifying $n cached file(s) in $dest against $MANIFEST"
    # Judge by the checksum lines themselves, not by exit status: `sha256sum -c` also
    # exits non-zero on unreadable entries, and we want to SHOW which file went bad.
    bad="$( cd "$dest" && sha256sum -c "$MANIFEST" 2>&1 | grep -v ': OK$' | head -n 5 )"
    if [ -z "$bad" ]; then
      ok "$label: $n file(s) already present and checksum-verified — download skipped"
      return 0
    fi
    printf '%s\n' "$bad"
    warn "$label: cached copy failed verification — re-downloading"
    return 1
  fi

  # No manifest. Either this is the first run since the cache existed, or someone staged
  # the data by hand. Adopt what is there once (nothing to compare it against yet) and
  # write the manifest, so every run after this one is verified.
  found="$(find "$dest" -type f -name "$glob" 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$ADOPT" = "1" ] && [ "${found:-0}" -gt 0 ]; then
    warn "$label: $found $glob already in $dest but no $MANIFEST — adopting them UNVERIFIED (trust on first use); PREFLIGHT_FORCE_FETCH=1 re-downloads"
    manifest_write "$dest" "$url"
    return 0
  fi
  return 1
}

fetch_cached() {   # fetch_cached URL DESTDIR LABEL GLOB — fetch_and_extract, but once
  local url="$1" dest="$2" label="$3" glob="$4"
  if [ -z "$url" ]; then fail "$label: env var is empty/unset"; return 1; fi
  if [ "$DRYRUN" = "1" ]; then
    info "would verify $dest/$MANIFEST and fetch $label only if that fails"
    fetch_and_extract "$url" "$dest" "$label"
    return 0
  fi
  if [ "$FORCE_FETCH" = "1" ]; then
    info "$label: PREFLIGHT_FORCE_FETCH=1 — ignoring any cached copy"
  elif dataset_cached "$dest" "$url" "$label" "$glob"; then
    return 0
  fi
  fetch_and_extract "$url" "$dest" "$label" || return 1
  manifest_write "$dest" "$url"
}

# --- where did the DATASET_PATCH archive actually put the payload? -----------
# Two layouts are in circulation: `tar -czf x.tgz container_export` nests everything under
# container_export/, `tar -czf x.tgz -C container_export .` does not. Assuming one of them
# is how a good run reported three FAILs (converter, patch, smoke test) and still converted
# 14 clouds — the operator happened to cd into the nested folder, so cwd saved the import.
resolve_prep_root() {   # resolve_prep_root PREPDIR — prints the dir that holds pipeline/
  local prep="$1" cand
  if [ -f "$prep/pipeline/convert.py" ]; then printf '%s\n' "$prep"; return 0; fi
  cand="$(find "$prep" -maxdepth 3 -type f -path '*/pipeline/convert.py' 2>/dev/null | head -n1)"
  if [ -n "$cand" ]; then printf '%s\n' "$(dirname "$(dirname "$cand")")"; return 0; fi
  printf '%s\n' "$prep"; return 1
}

# --- find (or clone) the ForAINet source tree --------------------------------
# Sets FORAINET_ROOT — the git root that CONTAINS PointCloudSegmentation/, because that is
# where forainet-local.patch's a/PointCloudSegmentation/... paths resolve — plus PCS and,
# through set_paths, everything hanging off it. Returns non-zero if there is no tree.
#
# Why cloning is the right answer and not a workaround: the image only ever had the code
# because docker-compose bind-mounted ./ForAINet onto /workspace. On a rented host there is
# nothing to mount, and ForAINet is a public repo — so fetch the same commit the submodule
# pins. A tree at any other commit would make `git apply` fail, not silently misbehave.
locate_forainet() {
  local root hit
  for root in "${PREFLIGHT_FORAINET_DIR:-}" /workspace /workspace/ForAINet \
              /opt/ForAINet /root/ForAINet /ForAINet; do
    [ -n "$root" ] || continue
    if [ -f "$root/PointCloudSegmentation/train.py" ]; then
      FORAINET_ROOT="$root"; PCS="$root/PointCloudSegmentation"; set_paths
      ok "ForAINet source found at $PCS"
      return 0
    fi
  done

  # Some templates unpack it somewhere else entirely; look before giving up. Bounded depth,
  # and /proc /sys /dev /data pruned so this stays a second, not a filesystem walk.
  hit="$(find / -maxdepth 5 \( -path /proc -o -path /sys -o -path /dev -o -path "$DATA" \) -prune -o \
              -type f -path '*/PointCloudSegmentation/train.py' -print 2>/dev/null | head -n1)"
  if [ -n "$hit" ]; then
    PCS="$(dirname "$hit")"; FORAINET_ROOT="$(dirname "$PCS")"; set_paths
    ok "ForAINet source found at $PCS (by search)"
    return 0
  fi

  if [ "$NO_CLONE" = "1" ]; then
    fail "no ForAINet on this box and PREFLIGHT_NO_CLONE=1 — set PREFLIGHT_FORAINET_DIR or bake it into the image"
    return 1
  fi
  if ! command -v git >/dev/null 2>&1; then
    fail "no ForAINet on this box and no git to clone it — apt-get update && apt-get install -y git"
    return 1
  fi

  # Clone into /workspace itself when it is empty (that is the layout the runbook and the
  # image's WORKDIR assume); otherwise one level down, and let set_paths follow.
  local dest="${PREFLIGHT_FORAINET_DIR:-}"
  if [ -z "$dest" ]; then
    if [ -d /workspace ] && [ -z "$(ls -A /workspace 2>/dev/null)" ]; then dest=/workspace
    else dest=/workspace/ForAINet; fi
  fi
  warn "no ForAINet in this image (env-only) — cloning ${FORAINET_URL##*/} @ ${FORAINET_REF:0:7} into $dest"

  # Shallow fetch of the exact commit is ~16 MB and a couple of seconds; the full clone is
  # the fallback for servers that refuse a by-SHA fetch.
  mkdir -p "$(dirname "$dest")" 2>/dev/null
  if ! run_sh "git init -q '$dest' && git -C '$dest' remote add origin '$FORAINET_URL' 2>/dev/null; \
               git -C '$dest' fetch -q --depth 1 origin '$FORAINET_REF' && git -C '$dest' checkout -q FETCH_HEAD"; then
    info "shallow fetch by SHA refused — falling back to a full clone"
    run_sh "rm -rf '$dest' && git clone -q '$FORAINET_URL' '$dest' && git -C '$dest' checkout -q '$FORAINET_REF'" || {
      fail "could not clone ForAINet from $FORAINET_URL"; return 1; }
  fi

  if [ -f "$dest/PointCloudSegmentation/train.py" ]; then
    FORAINET_ROOT="$dest"; PCS="$dest/PointCloudSegmentation"; set_paths
    ok "ForAINet cloned to $PCS @ $(git -C "$dest" rev-parse --short HEAD 2>/dev/null)"
    return 0
  fi
  fail "clone finished but $dest/PointCloudSegmentation/train.py is missing"
  return 1
}

# --- delete the patch's debug-only import ------------------------------------
# forainet-local.patch adds `import debugpy` to train.py for local VSCode debugging (the
# breakpoint calls themselves are already commented out). On a rented GPU nothing attaches
# a debugger and debugpy is usually not installed, so that one line turns a working
# train.py into an ImportError at startup. Idempotent.
strip_debugpy() {
  [ -f "$PCS/train.py" ] || return 0
  grep -q '^import debugpy' "$PCS/train.py" 2>/dev/null || return 0
  if "$PY" -c 'import debugpy' >/dev/null 2>&1; then
    info "train.py imports debugpy and debugpy is installed — leaving it alone"
    return 0
  fi
  run_sh "sed -i '/^import debugpy\$/d' '$PCS/train.py'" \
    && warn "stripped 'import debugpy' from train.py (debug-only line from the patch; not installed here)" \
    || fail "could not strip 'import debugpy' from train.py — training will fail at import"
}

# =============================================================================
section "PRE-FLIGHT START"
printf 'python           : %s\n' "$PY"
printf 'dry run          : %s\n' "$DRYRUN"
printf 'epochs / run     : %s\n' "$EPOCHS"
printf 'plots            : %s\n' "${PLOTS:-<all 14>}"
printf 'ForAINet dir     : %s\n' "${PREFLIGHT_FORAINET_DIR:-<auto-detect; clone @ ${FORAINET_REF:0:7} if absent>}"
printf 'self-heal        : clone=%s pip=%s\n' "$([ "$NO_CLONE" = 1 ] && echo off || echo on)" "$([ "$NO_PIP" = 1 ] && echo off || echo on)"
printf 'laspy spec       : %s\n' "$LASPY_SPEC"
printf 'dataset cache    : %s (%s)\n' "$DATA" "$([ "$FORCE_FETCH" = 1 ] && echo 'forced re-download' || echo 'reuse if checksums verify')"
printf 'DATASET_TRAIN_URL: %s\n' "${DATASET_TRAIN_URL:+<set>}"
printf 'DATASET_PATCH    : %s\n' "${DATASET_PATCH:+<set>}"
printf 'WANDB_API_KEY    : %s\n' "${WANDB_API_KEY:+<set>}"

if ! command -v "$PY" >/dev/null 2>&1 && [ "$DRYRUN" != "1" ]; then
  for alt in python3 python; do command -v "$alt" >/dev/null 2>&1 && PY="$alt" && break; done
  info "python3.8 not found; falling back to $PY"
fi

# --- A. environment sanity ---------------------------------------------------
section "A. Environment sanity"
if command -v nvidia-smi >/dev/null 2>&1; then
  run nvidia-smi -L && ok "nvidia-smi sees a GPU"
elif [ "$ALLOW_NOGPU" = "1" ] || [ "$DRYRUN" = "1" ]; then
  warn "nvidia-smi not found (continuing: dry run / ALLOW_NOGPU)"
else
  fail "nvidia-smi not found — training needs a GPU. Set PREFLIGHT_ALLOW_NOGPU=1 to inspect only."
fi

# torch CUDA build + the running GPU's compute capability vs the image's arch list.
if [ "$DRYRUN" = "1" ]; then
  info "would check torch.version.cuda and torch.cuda.get_device_capability() against: $SUPPORTED_CC"
else
  CC_OUT="$("$PY" - "$SUPPORTED_CC" <<'PYEOF' 2>&1
import sys
supported = set(sys.argv[1].split())
try:
    import torch
except Exception as e:
    print("NOTORCH", e); sys.exit(3)
print("cuda_build", torch.version.cuda)
if not torch.cuda.is_available():
    print("NOCUDA"); sys.exit(4)
cc = torch.cuda.get_device_capability()
tag = "%d.%d" % cc
print("device", torch.cuda.get_device_name(0), tag)
print("SUPPORTED" if tag in supported else "UNSUPPORTED", tag)
PYEOF
)"
  echo "$CC_OUT"
  if   echo "$CC_OUT" | grep -q "^SUPPORTED";   then ok "GPU arch is in the image's kernel set"
  elif echo "$CC_OUT" | grep -q "^UNSUPPORTED"; then fail "GPU arch NOT built into the image (no +PTX fallback) — you will get 'no kernel image available'. Rent T4/3090/A100/A40/A10/V100/P100, NOT 4090/L40S/H100."
  elif echo "$CC_OUT" | grep -q "NOTORCH";      then fail "torch not importable — is this the for-ai-net image?"
  elif echo "$CC_OUT" | grep -q "NOCUDA" && [ "$ALLOW_NOGPU" != "1" ]; then fail "torch sees no CUDA device"
  else warn "could not classify GPU arch; check output above"
  fi
fi

# SPARSE_BACKEND silently overrides the backbone choice — never let it survive into training.
if [ -n "${SPARSE_BACKEND:-}" ]; then
  warn "SPARSE_BACKEND=$SPARSE_BACKEND was set — unsetting it (it overrides model_name)"
  unset SPARSE_BACKEND
fi
export -n SPARSE_BACKEND 2>/dev/null || true
ok "SPARSE_BACKEND is clear"

# --- B. converter, ForAINet tree, patch (from DATASET_PATCH) ----------------
section "B. Converter + ForAINet tree + patch (\$DATASET_PATCH)"
fetch_and_extract "${DATASET_PATCH:-}" "$PREP" "DATASET_PATCH"

[ "$DRYRUN" = "1" ] || PREP_ROOT="$(resolve_prep_root "$PREP")"
PATCH_FILE="$PREP_ROOT/patches/forainet-local.patch"
SMOKE="$PREP_ROOT/smoke_test.py"
export PYTHONPATH="${PREP_ROOT}:${PYTHONPATH:-}"
info "PYTHONPATH=$PYTHONPATH"

if [ "$DRYRUN" = "1" ]; then
  info "would verify the converter under $PREP, locate (or clone @ ${FORAINET_REF:0:7}) ForAINet, and apply the 4-class patch"
else
  [ -f "$PREP_ROOT/pipeline/convert.py" ] && ok "converter present at $PREP_ROOT/pipeline/convert.py" \
    || fail "converter missing under $PREP (searched for */pipeline/convert.py — is this the container_export archive?)"

  # Where is the training code — and if nowhere, get it.
  locate_forainet

  if [ -f "$PCS/train.py" ]; then
    # Patch state: 4 classes + the TorchSparse model file.
    NUMCLS="$(grep -hoE 'Treeins_NUM_CLASSES\s*=\s*[0-9]+' "$SEG_DS" 2>/dev/null | grep -oE '[0-9]+' | head -n1)"
    if [ "$NUMCLS" = "4" ] && [ -f "$TS_MODEL" ]; then
      ok "already patched (Treeins_NUM_CLASSES=4, PointGroup3heads_ts.py present)"
    elif [ ! -f "$PATCH_FILE" ]; then
      fail "patch file not found at $PATCH_FILE — an unpatched tree trains a 5-class head on 4-class data"
    else
      warn "not patched (NUM_CLASSES=${NUMCLS:-?}, ts_model=$([ -f "$TS_MODEL" ] && echo yes || echo no)) — applying patch"
      if ( cd "$FORAINET_ROOT" && git apply --check "$PATCH_FILE" ) 2>/dev/null; then
        run_sh "cd '$FORAINET_ROOT' && git apply '$PATCH_FILE'"
      elif ( cd "$FORAINET_ROOT" && git apply --check --reverse "$PATCH_FILE" ) 2>/dev/null; then
        info "patch reverses cleanly — this tree already carries it; not applying twice"
      else
        fail "git apply --check failed — tree is at $(git -C "$FORAINET_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?'), the patch is a diff against ${FORAINET_REF:0:7}"
      fi
      NUMCLS="$(grep -hoE 'Treeins_NUM_CLASSES\s*=\s*[0-9]+' "$SEG_DS" 2>/dev/null | grep -oE '[0-9]+' | head -n1)"
      [ "$NUMCLS" = "4" ] && [ -f "$TS_MODEL" ] && ok "patch applied (NUM_CLASSES=4, ts model present)" \
                                                || fail "patch did not take (NUM_CLASSES=${NUMCLS:-?})"
    fi
    strip_debugpy

    # The configs the patch touches must PARSE under THIS image's omegaconf, or step E
    # dies four minutes from now -- after the download and the conversion -- with an error
    # that names no model and no backbone. hydra-core 1.0.7 pins omegaconf 2.0.x, whose
    # loader cannot construct a YAML merge key (`<<: *anchor`); omegaconf 2.3 on a dev box
    # expands it happily, so a file that is fine on your laptop can still kill the run
    # here. That exact asymmetry cost one run. One second, checked before any data moves.
    CFG_BAD=0
    for cfg in "$PCS/conf/models/panoptic/FORpartseg_3heads.yaml" \
               "$PCS/conf/training/treeins_set1.yaml" \
               "$PCS/conf/training/default.yaml"; do
      [ -f "$cfg" ] || continue
      "$PY" -c 'import sys; from omegaconf import OmegaConf; OmegaConf.load(sys.argv[1])' "$cfg" \
        || { fail "$(basename "$cfg") does not parse under this image's omegaconf — a '<<:' merge key is the usual cause; both training runs would fail before reading model_name"
             CFG_BAD=1; }
    done
    [ "$CFG_BAD" = "0" ] && ok "patched configs parse under this image's omegaconf"

    # numpy 1.24 REMOVED the np.float / np.int / np.bool / np.object / np.str aliases
    # (deprecated since 1.20). ForAINet still uses them in the panoptic metrics, and that
    # code first executes at EPOCH 31 -- `models: prepare_epoch: 30` gates every instance
    # metric -- so a single `astype(np.float)` survives 30 epochs of training and then
    # kills the run. That is exactly what happened on the A100: ~7 h of paid GPU time for
    # an AttributeError. This grep costs milliseconds and runs before anything is rented.
    #
    # Only the files on THIS config's code path are checked; the tree carries ~90 more
    # occurrences in datasets and trackers we never load (npm3d, s3dis, stpls3d, set2/3).
    if "$PY" -c 'import numpy,sys; v=tuple(int(x) for x in numpy.__version__.split(".")[:2]); sys.exit(0 if v>=(1,24) else 1)' 2>/dev/null; then
      NPY_HITS="$(grep -rnE 'np\.(float|int|bool|object|str|long|unicode|complex)([^a-zA-Z0-9_]|$)' \
            "$PCS/torch_points3d/metrics/panoptic_tracker_pointgroup_treeins_partseg.py" \
            "$PCS/torch_points3d/datasets/panoptic/treeins_set1.py" \
            "$PCS/torch_points3d/datasets/segmentation/treeins_set1.py" 2>/dev/null \
          | grep -vE 'np\.(float|int)(8|16|32|64|128|_)' \
          | grep -vE '^[^:]*:[0-9]+: *#')"
      if [ -n "$NPY_HITS" ]; then
        printf '%s\n' "$NPY_HITS" | head -n 5
        fail "removed numpy aliases (np.float/np.int/...) on the training code path, with numpy $("$PY" -c 'import numpy;print(numpy.__version__)') — these raise AttributeError at epoch $(grep -oE 'prepare_epoch: *[0-9]+' "$PCS/conf/models/panoptic/FORpartseg_3heads.yaml" | grep -oE '[0-9]+' | head -n1)+1, not at startup"
      else
        ok "no removed numpy aliases on the training code path"
      fi
    fi
  fi
fi

# --- B2. laspy: pinned, with a LAZ backend and header.parse_crs -------------
# The base image ships laspy 2.0.3 with NO LAZ backend: it cannot open the export at all,
# and it predates header.parse_crs, which the reader calls. Dockerfile.train fixes both at
# build time — but a rented template is often the plain base image, so fix it here too.
# $LASPY_SPEC pins the version (see the top of this file); a laspy that is merely "new
# enough" is not the same reader twice, and this gate exists to be reproducible.
section "B2. laspy / LAZ backend ($LASPY_SPEC)"

laspy_probe() {   # three lines: "laspy <v>", "backends <a,b|NONE>", "parse_crs <True|False>"
  "$PY" - <<'PYEOF' 2>&1
try:
    import laspy
except Exception as exc:
    print("laspy NOT-INSTALLED (%s)" % exc); print("backends NONE"); print("parse_crs False")
else:
    print("laspy %s" % laspy.__version__)
    print("backends %s" % (",".join(b.name for b in laspy.LazBackend if b.is_available()) or "NONE"))
    print("parse_crs %s" % hasattr(laspy.LasHeader, "parse_crs"))
PYEOF
}

PARSE_CRS="True"
if [ "$DRYRUN" = "1" ]; then
  info "would probe laspy for version ${LASPY_WANT:-<unpinned>} + a LAZ backend + header.parse_crs, installing '$LASPY_SPEC' if any of those is missing"
else
  LASPY_OUT="$(laspy_probe)"; echo "$LASPY_OUT"
  LASPY_VER="$(echo "$LASPY_OUT"  | awk '/^laspy/{print $2}')"
  BACKENDS="$(echo "$LASPY_OUT"  | awk '/^backends/{print $2}')"
  PARSE_CRS="$(echo "$LASPY_OUT" | awk '/^parse_crs/{print $2}')"

  # Install when a capability is missing OR the version is not the pinned one — that last
  # clause is what makes 2.0.3-with-lazrs-bolted-on and a floating 2.6 converge here.
  NEED_LASPY=0
  { [ -z "$BACKENDS" ] || [ "$BACKENDS" = "NONE" ]; } && NEED_LASPY=1
  [ "$PARSE_CRS" != "True" ] && NEED_LASPY=1
  [ -n "$LASPY_WANT" ] && [ "$LASPY_VER" != "$LASPY_WANT" ] && NEED_LASPY=1

  if [ "$NEED_LASPY" = "1" ]; then
    if [ "$NO_PIP" = "1" ]; then
      warn "laspy needs work (version=$LASPY_VER want=${LASPY_WANT:-any}, backends=$BACKENDS, parse_crs=$PARSE_CRS) but PREFLIGHT_NO_PIP=1 — not touching pip"
    else
      # No numpy pin here on purpose: Dockerfile.train pins numpy==1.24.4 because it knows
      # its base image. On an arbitrary rented image that pin is as likely to break torch
      # as to help, so install laspy alone and verify torch afterwards.
      if ! run "$PY" -m pip install --no-cache-dir "$LASPY_SPEC"; then
        # An unknown Python (3.7, or something newer than the wheels) can make the pin
        # unsatisfiable. A working laspy still beats none, so fall back — and say so.
        warn "pip install '$LASPY_SPEC' failed — retrying without the version pin"
        run "$PY" -m pip install --no-cache-dir --upgrade "laspy[lazrs]" \
          || warn "pip install 'laspy[lazrs]' failed too — continuing with what is here"
      fi
      LASPY_OUT="$(laspy_probe)"; echo "$LASPY_OUT"
      LASPY_VER="$(echo "$LASPY_OUT"  | awk '/^laspy/{print $2}')"
      BACKENDS="$(echo "$LASPY_OUT"  | awk '/^backends/{print $2}')"
      PARSE_CRS="$(echo "$LASPY_OUT" | awk '/^parse_crs/{print $2}')"
    fi
  fi

  # The version is a WARN, never a FAIL: the gate is about capabilities, and a laspy that
  # reads LAZ and exposes parse_crs does the job whatever its number says.
  if [ -z "$LASPY_WANT" ]; then
    info "laspy version not pinned (spec: $LASPY_SPEC) — installed: $LASPY_VER"
  elif [ "$LASPY_VER" = "$LASPY_WANT" ]; then
    ok "laspy $LASPY_VER (pinned)"
  else
    warn "laspy is $LASPY_VER, wanted $LASPY_WANT — the pin did not take (Python here is $("$PY" -V 2>&1))"
  fi

  # An EMPTY $BACKENDS means the probe itself produced nothing (no interpreter, import
  # crash) — that is a failure too, not a pass. Only a real backend list counts.
  { [ -n "$BACKENDS" ] && [ "$BACKENDS" != "NONE" ]; } && ok "LAZ backend available ($BACKENDS)" \
    || fail "no LAZ backend — laspy cannot open a .laz at all (pip install '$LASPY_SPEC')"
  [ "$PARSE_CRS" = "True" ] && ok "laspy exposes header.parse_crs" \
    || warn "laspy predates header.parse_crs — every plot logs 'could not parse CRS' and stores null; harmless, the exported clouds carry no CRS"

  # pip resolving laspy can quietly move numpy out from under torch.
  if "$PY" - <<'PYEOF'
import sys
import numpy
print("numpy %s" % numpy.__version__)
try:
    import torch
    print("torch %s (cuda build %s)" % (torch.__version__, torch.version.cuda))
except ModuleNotFoundError:
    print("torch not installed")
except Exception as exc:
    print("torch BROKEN: %s" % exc); sys.exit(1)
PYEOF
  then ok "numpy/torch still import after the laspy step"
  else fail "torch no longer imports — the laspy install moved numpy out from under it; pin numpy back to the image's version"
  fi
fi

# --- C. converter + smoke test (scope #1) -----------------------------------
section "C. Converter smoke test"
if [ "$DRYRUN" = "1" ]; then
  run "$PY" "$SMOKE"
elif [ -f "$SMOKE" ]; then
  SMOKE_OUT="$("$PY" "$SMOKE" 2>&1)"; SMOKE_RC=$?
  echo "$SMOKE_OUT"
  if [ "$SMOKE_RC" = "0" ]; then
    ok "smoke test passed (LAZ backend, extra dims, no Hydra, torch)"
  elif [ "$PARSE_CRS" != "True" ] && echo "$SMOKE_OUT" | grep -q "parse_crs"; then
    # The only assertion in there that training does not depend on (see B2).
    warn "smoke test failed only on header.parse_crs — accepted: training never reads the CRS"
  else
    fail "smoke test failed — see output above"
  fi
else
  fail "smoke_test.py not found under $PREP"
fi

# --- D. data + real conversion (\$DATASET_TRAIN_URL) ------------------------
section "D. Data + LAZ->PLY conversion (\$DATASET_TRAIN_URL)"
fetch_cached "${DATASET_TRAIN_URL:-}" "$DATA" "DATASET_TRAIN_URL" '*.laz'

# The archive may extract into a subfolder — find where the .laz actually landed.
if [ "$DRYRUN" = "1" ]; then
  LAZ_DIR="$DATA"; info "would locate .laz under $DATA and count them (expect 14)"
else
  LAZ_DIR="$(dirname "$(find "$DATA" -type f -name '*.laz' 2>/dev/null | head -n1)" 2>/dev/null)"
  [ -n "$LAZ_DIR" ] && [ -d "$LAZ_DIR" ] || LAZ_DIR="$DATA"
  NLAZ="$(count "$LAZ_DIR" '*.laz')"
  NYML="$(count "$LAZ_DIR" '*_offsets.yml')"
  info "found $NLAZ .laz and $NYML _offsets.yml in $LAZ_DIR"
  [ "$NLAZ" -ge 1 ] && ok "$NLAZ .laz present$([ "$NLAZ" = 14 ] || echo ' (expected 14)')" \
                    || fail "no .laz files found under $DATA — check DATASET_TRAIN_URL archive"
  [ "$NYML" = 14 ] && ok "14 _offsets.yml present" || warn "$NYML _offsets.yml (14 expected; only needed for restore, not training)"
fi

# Convert to the exact path ForAINet globs. --plots is SPACE-separated (argparse), not comma.
CONV_ARGS=(--to ply "$LAZ_DIR" "$RAW")
[ -n "$PLOTS" ] && CONV_ARGS+=(--plots $PLOTS)
run "$PY" -m pipeline.convert "${CONV_ARGS[@]}" && ok "conversion command ran" || fail "conversion failed"

if [ "$DRYRUN" != "1" ]; then
  EXPECT_PLY=14; [ -n "$PLOTS" ] && EXPECT_PLY="$(echo $PLOTS | wc -w)"
  NPLY="$(count "$RAW" '*.ply')"
  [ "$NPLY" -ge "$EXPECT_PLY" ] && ok "$NPLY .ply in raw dir (expected >= $EXPECT_PLY)" \
                                || fail "only $NPLY .ply in $RAW (expected $EXPECT_PLY)"
fi

# Clear any stale preprocessed-tensor cache so ForAINet rebuilds from THIS raw/.
run_sh "rm -rf '$CACHE'" && ok "cleared processed_0.2 cache"

# --- E. training startup, both backbones (scope #2) -------------------------
if [ "$SKIP_TRAIN" = "1" ]; then
  section "E. Training startup — SKIPPED (PREFLIGHT_SKIP_TRAIN=1)"
elif [ "$DRYRUN" != "1" ] && [ ! -f "$PCS/train.py" ]; then
  section "E. Training startup — NOT RUN"
  fail "no train.py at $PCS — step B could not get a ForAINet tree, so there is nothing to train"
else
  section "E. Training startup — both backbones ($EPOCHS epochs each)"
  COMMON="task=panoptic data=panoptic/treeins_set1 models=panoptic/FORpartseg_3heads training=treeins_set1 training.epochs=${EPOCHS} training.wandb.log=False"

  # An exit status alone is not proof: require the trainer's own "EPOCH n / m" line, so a
  # process that exits 0 without ever reaching the loop cannot pass either.
  trained() {   # trained LOGFILE
    [ "$DRYRUN" = "1" ] && return 0
    grep -qE "EPOCH [0-9]+ / [0-9]+" "$1" 2>/dev/null
  }

  info "Run 1: MinkowskiEngine (PointGroup-PAPER)"
  if run_sh "cd '$PCS' && '$PY' train.py $COMMON model_name=PointGroup-PAPER job_name=preflight_mink 2>&1 | tee /tmp/preflight_mink.log" \
     && trained /tmp/preflight_mink.log; then
    ok "MinkowskiEngine run finished $EPOCHS epoch(s)"
  else
    fail "MinkowskiEngine run failed or never entered the epoch loop — see /tmp/preflight_mink.log"
  fi

  info "Run 2: TorchSparse 1.4 (PointGroup-PAPER-TS)"
  if run_sh "cd '$PCS' && '$PY' train.py $COMMON model_name=PointGroup-PAPER-TS job_name=preflight_ts 2>&1 | tee /tmp/preflight_ts.log" \
     && trained /tmp/preflight_ts.log; then
    ok "TorchSparse run finished $EPOCHS epoch(s)"
  else
    fail "TorchSparse run failed or never entered the epoch loop — see /tmp/preflight_ts.log"
  fi

  # Best-effort sanity greps on the captured logs (informational; the real gate is exit 0).
  #
  # Each block is gated on that run having actually TRAINED. A log from a run that died at
  # startup contains no deprecation warning either, so the old unconditional check reported
  # "[OK] no Minkowski deprecation warning on run 2 (TorchSparse active)" for two runs that
  # never built a model — an absence-of-evidence pass, the same trap as the tee/pipefail bug.
  if [ "$DRYRUN" != "1" ]; then
    if trained /tmp/preflight_mink.log; then
      grep -q "11872126" /tmp/preflight_mink.log && ok "Model size == 11872126" \
        || warn "did not see 'Model size 11872126' in the log (architecture changed?)"
      if grep -q "iou_per_class" /tmp/preflight_mink.log; then
        NIOU="$(grep -oE "iou_per_class[^}]*}" /tmp/preflight_mink.log | tail -n1 | grep -oE "[0-9]+\.[0-9]+|[0-9]+:" | wc -l)"
        info "iou_per_class entries seen (approx): $NIOU — expect 4; 5 means the patch is not applied"
      fi
      grep -qi "deprecat" /tmp/preflight_mink.log && info "Minkowski deprecation warning present on run 1 (expected)"
    else
      info "run 1 never entered the epoch loop — skipping its log sanity checks"
    fi

    if trained /tmp/preflight_ts.log; then
      grep -qi "deprecat" /tmp/preflight_ts.log && warn "deprecation warning on run 2 — TorchSparse may NOT be active (SPARSE_BACKEND?)" \
        || ok "no Minkowski deprecation warning on run 2 (TorchSparse active)"
    else
      info "run 2 never entered the epoch loop — skipping its log sanity checks"
    fi
  fi
fi

# --- F. optional wandb validation -------------------------------------------
if [ "$DO_WANDB" = "1" ]; then
  section "F. W&B validation (1 epoch, wandb.log=True)"
  if [ -z "${WANDB_API_KEY:-}" ]; then
    fail "PREFLIGHT_WANDB=1 but WANDB_API_KEY is not set"
  else
    COMMONW="task=panoptic data=panoptic/treeins_set1 models=panoptic/FORpartseg_3heads training=treeins_set1 training.epochs=1"
    if run_sh "cd '$PCS' && '$PY' train.py $COMMONW model_name=PointGroup-PAPER training.wandb.log=True job_name=preflight_wandb 2>&1 | tee /tmp/preflight_wandb.log"; then
      ok "wandb run reached training (key + entity resolve)"
    else
      fail "wandb run failed — likely key format (upgrade client) or entity not found; see /tmp/preflight_wandb.log"
    fi
  fi
fi

# --- summary -----------------------------------------------------------------
section "SUMMARY"
printf '%sForAINet : %s%s\n' "$DIM" "$PCS" "$RST"
printf '%sraw dir  : %s%s\n' "$DIM" "$RAW" "$RST"
printf '%sconverter: %s%s\n\n' "$DIM" "$PREP_ROOT" "$RST"
for line in "${RESULTS[@]}"; do
  case "$line" in
    OK*)   printf '%s  %s%s\n' "$GRN" "$line" "$RST" ;;
    WARN*) printf '%s  %s%s\n' "$YLW" "$line" "$RST" ;;
    FAIL*) printf '%s  %s%s\n' "$RED" "$line" "$RST" ;;
  esac
done
echo
if [ "$FAILED" = "0" ]; then
  printf '%s%sPRE-FLIGHT PASSED%s — cleared to rent the A100.\n' "$BOLD" "$GRN" "$RST"
  [ "$DRYRUN" = "1" ] && printf '%s(dry run: nothing was executed)%s\n' "$DIM" "$RST"
  exit 0
else
  printf '%s%sPRE-FLIGHT FAILED%s — fix the [FAIL] items above before renting anything.\n' "$BOLD" "$RED" "$RST"
  exit 1
fi
