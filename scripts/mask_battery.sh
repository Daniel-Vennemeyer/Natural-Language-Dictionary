#!/usr/bin/env bash
# MASK BATTERY: run the MASK honesty transfer test on every deception steering vector.
#
# MASK only needs the steering VECTOR, not the in-domain ACME pass, so this builds any
# missing vector directly from its checkpoint with routed_acme --emit-vector (seconds for
# axis, one judge pass for balanced) instead of requiring a full racme run first. Existing
# racme cells with a saved 'cvec' are reused as-is.
#
# Reports per cell: full-sample dLIE (statement contradicts the model's own belief) and the
# MASK-FAITHFUL knows-subset dLIE (items where belief == ground truth).
#
# COST: each cell generates (1 belief + |alphas| statements) x --n and runs that many
# extractor calls, so it is minutes-to-tens-of-minutes per cell. Defaults are trimmed
# accordingly: alphas 0,4,8 (the effect is monotone; 8 is where it is largest) and n=200.
#
# usage: bash scripts/mask_battery.sh <GPU> [mask_config]     (default known_facts)
#        MASK_N=150 MASK_ALPHAS=0,8 bash scripts/mask_battery.sh 1
#        bash scripts/mask_battery.sh summary
set -e
OUT=outputs/runs/soft_wc
CFG=configs/experiments/deception_roleplay_qwen4b_l20.yaml

if [[ "$1" == "summary" ]]; then
  python3 - <<'EOF'
import glob, re
print("| method | K | router | MASK cfg | base lying | best a | dLIE (full) | "
      "dLIE (knows) | knows n |")
print("|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob("outputs/runs/soft_wc/mask_dbat_*.log")):
    m = re.match(r".*mask_dbat_([a-z]+)_k(\d+)_(axis|balanced)_([a-z_]+)\.log$", f)
    if not m:
        continue
    meth, K, router, mcfg = m.groups()
    base, full, knows, kn = "-", None, None, "-"
    for line in open(f):
        mm = re.search(r"alpha\s+0: LYING rate ([\d.]+)", line)
        if mm:
            base = mm.group(1)
        mm = re.search(r"KNOWS-SUBSET \(belief == ground truth: (\d+)/", line)
        if mm:
            kn = mm.group(1)
        mm = re.search(r"KNOWS alpha\s+([\d.]+): dLIE ([+-][\d.]+) \+-([\d.]+) "
                       r"\(([+-][\d.]+) sd", line)
        if mm and (knows is None or float(mm.group(2)) > knows[1]):
            knows = (float(mm.group(1)), float(mm.group(2)), float(mm.group(4)))
        mm = re.search(r"^\[mask\]   alpha\s+([\d.]+): dLIE ([+-][\d.]+) \+-([\d.]+) "
                       r"\(([+-][\d.]+) sd", line)
        if mm and (full is None or float(mm.group(2)) > full[1]):
            full = (float(mm.group(1)), float(mm.group(2)), float(mm.group(4)))
    fs = f"{full[1]:+.3f} ({full[2]:+.1f}sd)" if full else "-"
    ks = f"{knows[1]:+.3f} ({knows[2]:+.1f}sd)" if knows else "-"
    ba = f"{full[0]:g}" if full else "-"
    print(f"| {meth} | {K} | {router} | {mcfg} | {base} | {ba} | {fs} | {ks} | {kn} |")
print("\ndLIE > 0 = the vector makes the model state things it does not believe. The "
      "knows-subset is MASK's intended reading (the model demonstrably knows the fact).")
EOF
  exit 0
fi

GPU=$1; MCFG=${2:-known_facts}
N=${MASK_N:-200}
ALPHAS=${MASK_ALPHAS:-0,4,8}
ENV="CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
run() { echo ">>> $*"; eval "$ENV $*"; }

U=/scratch/mech-taxonomy
UNITS=$U/units_rpdec_l20.npz; PACTS=$U/prompt_last_rpdec_l20.npz
METHODS=${METHODS:-"soft bsae dmpca expert"}
KS=${MASK_KS:-"1 3 5 7"}
ROUTERS=${MASK_ROUTERS:-"axis balanced"}

shopt -s nullglob
CELLS=()
for M in $METHODS; do for K in $KS; do for R in $ROUTERS; do
  ck=$OUT/dbat_${M}_k${K}.pt
  [[ -f $ck ]] || continue
  vec=$OUT/dbat_${M}_k${K}_racme_${R}.pt          # reuse a racme cell if it has a cvec
  if ! python3 -c "
import torch,sys
try: c=torch.load('$vec',map_location='cpu',weights_only=False)
except Exception: sys.exit(1)
sys.exit(0 if c.get('cvec') is not None else 1)" 2>/dev/null; then
    vec=$OUT/vec_dbat_${M}_k${K}_${R}.pt          # otherwise emit the vector directly
    if [[ ! -f $vec ]]; then
      DF=""; [[ -f $OUT/dbat_${M}_k${K}_causal.pt ]] && DF="--dirs-from $OUT/dbat_${M}_k${K}_causal.pt"
      run python scripts/routed_acme.py --config $CFG --units $UNITS --prompt-acts $PACTS \
        --domain deception --max-prompt-chars 800 --checkpoint $ck $DF \
        --router $R --emit-vector --out $vec
    fi
  fi
  CELLS+=("$vec")
done; done; done
if [[ ${#CELLS[@]} -eq 0 ]]; then
  echo "no deception checkpoints found under $OUT (dbat_<method>_k<K>.pt)"; exit 1
fi
echo "=== MASK battery: ${#CELLS[@]} vectors, config=$MCFG, n=$N, alphas=$ALPHAS ==="

for pt in "${CELLS[@]}"; do
  base=$(basename "$pt" .pt)
  short=${base/_racme_/_}; short=${short#vec_}    # dbat_<method>_k<K>_<router>
  log=$OUT/mask_${short}_${MCFG}.log
  if [[ -f $log ]]; then echo ">>> $log exists, skipping"; continue; fi
  run python scripts/mask_acme.py --config $CFG --racme "$pt" --mask-config $MCFG \
    --n $N --alphas $ALPHAS --out $OUT/mask_${short}_${MCFG}.pt "2>&1 | tee $log"
done
echo "=== MASK battery done ===  (table: bash scripts/mask_battery.sh summary)"
