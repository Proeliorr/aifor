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
# Tunables (all optional, sane defaults)
# --------------------------------------
#   PREFLIGHT_DRYRUN=1        print every command without running it (works with no GPU/Docker)
#   PREFLIGHT_PLOTS="a b c"   convert only these stems (faster/cheaper); default = all 14
#   PREFLIGHT_EPOCHS=2        epochs per training run (2 is enough to prove startup)
#   PREFLIGHT_WANDB=1         also run step F (a short wandb.log=True run to validate the key)
#   PREFLIGHT_SKIP_TRAIN=1    stop after the converter+smoke checks (scope #1 only)
#   PREFLIGHT_ALLOW_NOGPU=1   don't hard-fail when nvidia-smi is absent (for dry inspection)
#   PREFLIGHT_PY=python3.8    interpreter to use inside the image
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

# Paths inside the for-ai-net image.
FORAINET_ROOT="/workspace"                                   # git root; where `git apply` runs
PCS="/workspace/PointCloudSegmentation"                      # train.py cwd + data root
PREP="/opt/prep"                                             # converter + patch land here
DATA="/data"                                                 # dataset archive extracts here
RAW="${PCS}/data_set1_5classes/treeinsfused/raw/SegmentedForests"   # ForAINet globs raw/**/*.ply
CACHE="${PCS}/data_set1_5classes/treeinsfused/processed_0.2"        # stale-cache trap
SEG_DS="${PCS}/torch_points3d/datasets/segmentation/treeins_set1.py"
TS_MODEL="${PCS}/torch_points3d/models/panoptic/PointGroup3heads_ts.py"

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
run_sh() {
  printf '%s$ %s%s\n' "$DIM" "$1" "$RST"
  if [ "$DRYRUN" = "1" ]; then return 0; fi
  bash -c "$1"
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

# =============================================================================
section "PRE-FLIGHT START"
printf 'python           : %s\n' "$PY"
printf 'dry run          : %s\n' "$DRYRUN"
printf 'epochs / run     : %s\n' "$EPOCHS"
printf 'plots            : %s\n' "${PLOTS:-<all 14>}"
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

# --- B. patch + converter from DATASET_PATCH --------------------------------
section "B. Patch + converter (\$DATASET_PATCH)"
fetch_and_extract "${DATASET_PATCH:-}" "$PREP" "DATASET_PATCH"
export PYTHONPATH="${PREP}:${PYTHONPATH:-}"
info "PYTHONPATH=$PYTHONPATH"

if [ "$DRYRUN" = "1" ]; then
  info "would verify converter at $PREP/pipeline/convert.py and the 4-class patch"
else
  [ -f "$PREP/pipeline/convert.py" ] && ok "converter present at $PREP/pipeline/convert.py" \
                                     || fail "converter missing at $PREP/pipeline/convert.py (DATASET_PATCH archive layout?)"
  # Is the ForAINet training code even in the image?
  if [ -f "$PCS/train.py" ]; then
    ok "ForAINet code present at $PCS/train.py"
  else
    fail "no train.py at $PCS — image is env-only; it must include ForAINet (COPY ForAINet/ /workspace/) or be bind-mounted"
  fi
  # Patch state: 4 classes + the TorchSparse model file.
  NUMCLS="$(grep -hoE 'Treeins_NUM_CLASSES\s*=\s*[0-9]+' "$SEG_DS" 2>/dev/null | grep -oE '[0-9]+' | head -n1)"
  if [ "$NUMCLS" = "4" ] && [ -f "$TS_MODEL" ]; then
    ok "already patched (Treeins_NUM_CLASSES=4, PointGroup3heads_ts.py present)"
  else
    warn "not patched (NUM_CLASSES=${NUMCLS:-?}, ts_model=$([ -f "$TS_MODEL" ] && echo yes || echo no)) — applying patch"
    if [ -f "$PREP/patches/forainet-local.patch" ]; then
      if ( cd "$FORAINET_ROOT" && git apply --check "$PREP/patches/forainet-local.patch" ) 2>/dev/null; then
        run_sh "cd '$FORAINET_ROOT' && git apply '$PREP/patches/forainet-local.patch'"
      else
        fail "git apply --check failed — the ForAINet tree does not match the patch (wrong commit?)"
      fi
      NUMCLS="$(grep -hoE 'Treeins_NUM_CLASSES\s*=\s*[0-9]+' "$SEG_DS" 2>/dev/null | grep -oE '[0-9]+' | head -n1)"
      [ "$NUMCLS" = "4" ] && [ -f "$TS_MODEL" ] && ok "patch applied (NUM_CLASSES=4, ts model present)" \
                                                || fail "patch did not take (NUM_CLASSES=${NUMCLS:-?})"
    else
      fail "patch file not found at $PREP/patches/forainet-local.patch"
    fi
  fi
fi

# --- C. converter + smoke test (scope #1) -----------------------------------
section "C. Converter smoke test"
if [ -f "$PREP/smoke_test.py" ] || [ "$DRYRUN" = "1" ]; then
  if run "$PY" "$PREP/smoke_test.py"; then ok "smoke test passed (LAZ backend, extra dims, no Hydra, torch)"; \
                                      else fail "smoke test failed — see output above"; fi
else
  fail "smoke_test.py not found at $PREP/smoke_test.py"
fi

# --- D. data + real conversion (\$DATASET_TRAIN_URL) ------------------------
section "D. Data + LAZ->PLY conversion (\$DATASET_TRAIN_URL)"
fetch_and_extract "${DATASET_TRAIN_URL:-}" "$DATA" "DATASET_TRAIN_URL"

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
else
  section "E. Training startup — both backbones ($EPOCHS epochs each)"
  COMMON="task=panoptic data=panoptic/treeins_set1 models=panoptic/FORpartseg_3heads training=treeins_set1 training.epochs=${EPOCHS} training.wandb.log=False"
  cd "$PCS" 2>/dev/null || { [ "$DRYRUN" = "1" ] || fail "cannot cd to $PCS"; }

  info "Run 1: MinkowskiEngine (PointGroup-PAPER)"
  if run_sh "cd '$PCS' && '$PY' train.py $COMMON model_name=PointGroup-PAPER job_name=preflight_mink 2>&1 | tee /tmp/preflight_mink.log"; then
    ok "MinkowskiEngine run finished $EPOCHS epoch(s)"
  else
    fail "MinkowskiEngine run failed — see /tmp/preflight_mink.log"
  fi

  info "Run 2: TorchSparse 1.4 (PointGroup-PAPER-TS)"
  if run_sh "cd '$PCS' && '$PY' train.py $COMMON model_name=PointGroup-PAPER-TS job_name=preflight_ts 2>&1 | tee /tmp/preflight_ts.log"; then
    ok "TorchSparse run finished $EPOCHS epoch(s)"
  else
    fail "TorchSparse run failed — see /tmp/preflight_ts.log"
  fi

  # Best-effort sanity greps on the captured logs (informational; the real gate is exit 0).
  if [ "$DRYRUN" != "1" ] && [ -f /tmp/preflight_mink.log ]; then
    grep -q "11872126" /tmp/preflight_mink.log && ok "Model size == 11872126" \
      || warn "did not see 'Model size 11872126' in the log (architecture changed?)"
    if grep -q "iou_per_class" /tmp/preflight_mink.log; then
      NIOU="$(grep -oE "iou_per_class[^}]*}" /tmp/preflight_mink.log | tail -n1 | grep -oE "[0-9]+\.[0-9]+|[0-9]+:" | wc -l)"
      info "iou_per_class entries seen (approx): $NIOU — expect 4; 5 means the patch is not applied"
    fi
    grep -qi "deprecat" /tmp/preflight_mink.log && info "Minkowski deprecation warning present on run 1 (expected)"
    if [ -f /tmp/preflight_ts.log ]; then
      grep -qi "deprecat" /tmp/preflight_ts.log && warn "deprecation warning on run 2 — TorchSparse may NOT be active (SPARSE_BACKEND?)" \
        || ok "no Minkowski deprecation warning on run 2 (TorchSparse active)"
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
