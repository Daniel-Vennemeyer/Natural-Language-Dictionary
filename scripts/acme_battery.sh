#!/usr/bin/env bash
# ACME-ONLY battery: no training, no causal legs. Runs routed_acme (axis vs balanced) on
# EXISTING checkpoints, then the paired racme_compare, for one domain.
#
#   axis     = signed least-squares reconstruction of the behavior axis in the atom span
#   balanced = MAXIMIN over the domain's judged categories (lifts the WORST one)
# balanced prints 'WORST category cos: axis -> balanced' BEFORE generating: a negative
# worst-cosine means the axis MUST trade that category away, which is where balanced wins
# (sycophancy k5: -0.109 -> +0.325, and ACME went +.030 -> +.107 with all 3 categories up).
#
# Cells whose checkpoint is missing are skipped, so this is safe to run over a partially
# built battery. Resumable: a leg whose .log exists is skipped (delete it to rerun).
#
# usage: bash scripts/acme_battery.sh <syco|dec|emo> <GPU> [K...]      (default K=1 3 5 7)
#        METHODS="soft dmpca" bash scripts/acme_battery.sh syco 1 5
#        bash scripts/acme_battery.sh summary
set -e
U=/scratch/mech-taxonomy
OUT=outputs/runs/soft_wc

if [[ "$1" == "summary" ]]; then
  python3 - <<'EOF'
import glob, re
print("| domain | method | K | best a | balanced ACME | axis ACME | bal-axis (paired) | "
      "worst cat cos (axis->bal) |")
print("|---|---|---|---|---|---|---|---|")

def acme(path):
    """-> (best_alpha, acme, sd) using the run's own operating point line."""
    best = None
    try:
        for line in open(path):
            m = re.search(r"alpha \+([\d.]+): ACME ([+-][\d.]+) \+-([\d.]+) \(paired SE, "
                          r"([+-][\d.]+) sd", line)
            if m and (best is None or float(m.group(2)) > best[1]):
                best = (float(m.group(1)), float(m.group(2)), float(m.group(4)))
    except FileNotFoundError:
        return None
    return best

def worstcos(path):
    try:
        for line in open(path):
            m = re.search(r"WORST category cos ([+-][\d.]+) -> ([+-][\d.]+)", line)
            if m:
                return f"{float(m.group(1)):+.3f} -> {float(m.group(2)):+.3f}"
    except FileNotFoundError:
        pass
    return "-"

def paired(path):
    rows, cur = {}, None
    try:
        for line in open(path):
            m = re.search(r"=== alpha ([\d.]+) ===", line)
            if m:
                cur = float(m.group(1)); continue
            m = re.search(r"\[cmp\]\s+target\s+\S+ - \S+ = ([+-][\d.]+) \+-([\d.]+) "
                          r"\(([+-][\d.]+) sd\)", line)
            if m and cur is not None:
                rows[cur] = (float(m.group(1)), float(m.group(3)))
    except FileNotFoundError:
        return "-"
    if not rows:
        return "-"
    a = max(rows, key=lambda k: rows[k][0])
    d, sd = rows[a]
    return f"{d:+.3f} ({sd:+.1f}sd){'*' if abs(sd) > 2 else ''}"

for dom, pfx, cmp_ in (("syco", "bat_", "syco"), ("dec", "dbat_", "dec"),
                       ("emo", "ebat_", "emo")):
    for f in sorted(glob.glob(f"outputs/runs/soft_wc/{pfx}*_racme_balanced.log")):
        m = re.match(rf".*{pfx}([a-z]+)_k(\d+)_racme_balanced\.log$", f)
        if not m:
            continue
        meth, K = m.group(1), m.group(2)
        b = acme(f); a_ = acme(f.replace("_balanced.log", "_axis.log"))
        bs = f"{b[1]:+.3f} ({b[2]:+.1f}sd)" if b else "INCOMPLETE"
        as_ = f"{a_[1]:+.3f} ({a_[2]:+.1f}sd)" if a_ else "INCOMPLETE"
        besta = f"{b[0]:g}" if b else "-"
        print(f"| {dom} | {meth} | {K} | {besta} | {bs} | {as_} | "
              f"{paired(f'outputs/runs/soft_wc/cmp_{cmp_}_{meth}_k{K}.log')} | "
              f"{worstcos(f)} |")
print("\n* = |paired difference| > 2 SE.  A negative axis worst-cosine is the signature "
      "that the axis must trade a category away.")
print("INCOMPLETE = the .log exists but has no alpha line: the leg is still running or died "
      "before generating.\nNote: 'best a' here is the max-ACME alpha IGNORING the length "
      "guard, so where the guard binds this\nreports a larger number at a different alpha "
      "than routed_acme's own operating point (and than\nacme_report.py). Do not mix cells "
      "from the two tables.")
EOF
  exit 0
fi

DOM=$1; GPU=$2; shift 2
KS=${*:-"1 3 5 7"}
METHODS=${METHODS:-"soft bsae dmpca"}
case $DOM in
  syco) CFG=configs/experiments/sycophancy_grounded_sparse_decomposition_l20_resp.yaml
        UNITS=$U/units_l20.npz; PACTS=$U/prompt_last_l20.npz; PFX=bat_
        DOMAIN=sycophancy; MPC=${BAT_MAX_PROMPT_CHARS:-400}; GPF="" ;;
  dec)  CFG=configs/experiments/deception_roleplay_qwen4b_l20.yaml
        UNITS=$U/units_rpdec_l20.npz; PACTS=$U/prompt_last_rpdec_l20.npz; PFX=dbat_
        DOMAIN=deception; MPC=${BAT_MAX_PROMPT_CHARS:-800}; GPF="" ;;
  emo)  CFG=configs/experiments/emotion_emobank_qwen4b_l20.yaml
        UNITS=$U/units_emo_l20.npz; PACTS=$U/prompt_last_emo_l20.npz; PFX=ebat_
        DOMAIN=emotion; MPC=${BAT_MAX_PROMPT_CHARS:-400}
        GPF="--gen-prompts $U/emo_gen_prompts.txt" ;;   # constant dataset frame otherwise
  *) echo "usage: bash scripts/acme_battery.sh <syco|dec|emo> <GPU> [K...]"; exit 1 ;;
esac
# emotion defaults to VADER: its behavior IS sentiment, so a deterministic lexicon
# removes the LLM judge from the headline number entirely (and skips loading the 30B).
if [[ "$DOM" == "emo" ]]; then RP=${BAT_RACME_PROVIDER:-vader}; else RP=${BAT_RACME_PROVIDER:-local}; fi
BASE="--config $CFG --units $UNITS --prompt-acts $PACTS --domain $DOMAIN $GPF \
 --max-prompt-chars $MPC --judge-provider $RP --alpha-eval 2,4,8 --interv-prompts 96"
ENV="CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src"
run() { echo ">>> $*"; eval "$ENV $*"; }

for M in $METHODS; do
  for K in $KS; do
    tag="${PFX}${M}_k${K}"
    if [[ ! -f $OUT/${tag}.pt ]]; then
      echo ">>> no checkpoint $OUT/${tag}.pt -- skipping (train it with the domain battery)"
      continue
    fi
    DF=""; [[ -f $OUT/${tag}_causal.pt ]] && DF="--dirs-from $OUT/${tag}_causal.pt"
    for R in axis balanced; do
      rlog=$OUT/${tag}_racme_${R}.log
      if [[ -f $rlog ]]; then echo ">>> $rlog exists, skipping"; else
        run python scripts/routed_acme.py $BASE --checkpoint $OUT/${tag}.pt $DF \
          --router $R --out $OUT/${tag}_racme_${R}.pt "2>&1 | tee $rlog"
      fi
    done
    run python scripts/racme_compare.py --a $OUT/${tag}_racme_balanced.pt \
      --b $OUT/${tag}_racme_axis.pt --label-a balanced --label-b axis \
      "2>&1 | tee $OUT/cmp_${DOM}_${M}_k${K}.log"
  done
done
echo "=== ACME done: $DOM / $METHODS / K=$KS ===  (table: bash scripts/acme_battery.sh summary)"
