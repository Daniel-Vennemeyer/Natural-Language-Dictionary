#!/usr/bin/env bash
# ROLEPLAY-DECEPTION METHOD BATTERY: fidelity / AIE / selectivity at K, mirroring battery_syco.
#
# Differences from the sycophancy battery:
#   - config/units = deception_roleplay (740 paired examples, shared-prompt contrastive design)
#   - soft stage-2 cycle runs with --probe-n 256 (default 512 would swallow TRAIN and leave the
#     CYCV acceptance split EMPTY at this n)
#   - racme ACME leg IS available now: routed_acme --domain deception scores three banded
#     facets (falsehood / concealment / misleading_framing). Runs axis + balanced routers.
#   - reft = LoReFT trained end-to-end (no rl artifact needed since the subspace is learned)
#   - held-out judge = gemma-2-9b-it (cross-family)
#   - --max-prompt-chars 800 EVERYWHERE (training, causal legs, ACME): roleplay prompts are
#     493 chars median (max 753) and the 400 default cut the TAIL off 85% of them -- the
#     question itself -- so neither the model nor the judge saw what was being asked
#     (baseline deceptiveness 0.016). Every deception number produced before this is void.
#
# usage: bash scripts/battery_dec.sh <GPU> <method> <K...>
#   bash scripts/battery_dec.sh 0 soft 1 3 5 7
#   bash scripts/battery_dec.sh 1 dmpca 1 3 5 7
#   bash scripts/battery_dec.sh 1 bsae 1 3 5 7
# summarize: python scripts/battery_summary.py --prefix ${PFX}
set -e -o pipefail
GPU=$1; METHOD=$2; shift 2
U=/scratch/mech-taxonomy
# SUBJECT-MODEL OVERRIDES: see battery_syco.sh header. BAT_EVAL_JUDGE sets the held-out
# judge (default gemma-2-9b-it; use Qwen/Qwen2.5-7B-Instruct when the SUBJECT is a gemma).
CFG=${BAT_CFG:-configs/experiments/deception_roleplay_qwen4b_l20.yaml}
UNITS=${BAT_UNITS:-$U/units_rpdec_l20.npz}
PACTS=${BAT_PACTS:-$U/prompt_last_rpdec_l20.npz}
PFX=${BAT_PREFIX:-dbat_}
BT=${BAT_BATCH_TEXTS:-256}
JT=${BAT_JUDGE_TEMP:-1.5}                        # raise (3-4) for judges that rate soft prompts a hard 0 (saturated digits = no gradient)                      # drop to 96-128 for a ~35B subject (judge backprop VRAM)
EVJ=${BAT_EVAL_JUDGE:-google/gemma-2-9b-it}
# REGULARIZER OVERRIDES: BAT_LAMBDA_DIV=0 removes the behavior-corr distinctness
# penalty (the ablation that collapses effective K), _FID the interpretation-fidelity
# term, _LANG the sayability pull. Use a distinct BAT_PREFIX so the ablated run does not
# overwrite the standard one.
LD=${BAT_LAMBDA_DIV:-2.0}
LF=${BAT_LAMBDA_FID:-1.0}
LL=${BAT_LAMBDA_LANG:-0.4}
# SAE CONFIG (bsae leg). Defaults are sae_baseline's own; the configuration sweep
# (scripts/sae_sweep.sh) found denser + smaller better in all three domains, so the
# stronger-baseline setting is BAT_SAE_DICT=2048 BAT_SAE_TOPK=64.
SD=${BAT_SAE_DICT:-4096}; STK=${BAT_SAE_TOPK:-32}
# RUNTIME TRIMS. Defaults reproduce the full protocol; override to shorten a rerun.
#   BAT_CAUSAL_ONLY=distill      drop the PROMPTING CEILING leg (an oracle upper bound, a
#                                full generation pass per atom, not part of the comparison)
#   BAT_SWEEP_ALPHAS=...         fewer alphas in the selectivity sweep. NOTE this changes
#                                what per-atom best-alpha selects over, so honestSel stops
#                                being comparable to rows built on the full grid.
#   BAT_ROUTERS="balanced"       skip the axis ACME arm (racme_compare is skipped with it)
CONLY=${BAT_CAUSAL_ONLY:-ceiling,distill}
SWA=${BAT_SWEEP_ALPHAS:-2,4,6,8,12,16}
ROUTERS=${BAT_ROUTERS:-"axis balanced"}
OUT=outputs/runs/soft_wc
MPC=${BAT_MAX_PROMPT_CHARS:-800}                 # 0% of roleplay prompts truncated at 800
# SEED: vary for cross-seed robustness (use a distinct BAT_PREFIX per seed so runs
# do not overwrite each other). causal_diag must see the SAME seed as training.
SEED=${BAT_SEED:-0}
DATA="--config $CFG --units $UNITS --prompt-acts $PACTS --max-prompt-chars $MPC --seed $SEED"
CD="--config $CFG --units $UNITS --acts-key sent_acts --prompt-acts $PACTS --max-prompt-chars $MPC --seed $SEED"
RP=${BAT_RACME_PROVIDER:-local}                  # ACME judge: local 30B (free) or openai
RACME="--config $CFG --units $UNITS --prompt-acts $PACTS --domain deception \
 --max-prompt-chars $MPC --judge-provider $RP --alpha-eval 2,4,8 --interv-prompts 96"
ENV="CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src"

run() {
  if [[ -n "$BAT_SKIP" ]]; then
    local o; o=$(echo "$*" | grep -o '\-\-out [^ ]*\.pt' | head -1 | awk '{print $2}')
    if [[ -n "$o" && -f "$o" ]]; then echo ">>> skip (exists): $o"; return 0; fi
  fi
  echo ">>> $*"; eval "$ENV $*"
}

for K in "$@"; do
  tag="${PFX}${METHOD}_k${K}"
  case $METHOD in
    soft)
      if [[ -f $OUT/${tag}_s1.pt ]]; then
        echo ">>> stage-1 exists ($OUT/${tag}_s1.pt), skipping retrain"
      else
        run python scripts/soft_concepts.py $DATA --acts-key sent_acts --num-concepts "$K" \
          --target acts_wc_axisw --lambda-fid $LF --lambda-div $LD --lambda-lang $LL \
          --steps 300 --batch-texts $BT --judge-temp $JT --fit-ridge 0.2 \
          --out $OUT/${tag}_s1.pt "2>&1 | tee $OUT/${tag}_s1.log"
      fi
      run python scripts/soft_concepts.py $DATA --acts-key sent_acts --num-concepts "$K" \
        --target acts_wc_axisw --lambda-fid $LF --lambda-div $LD --lambda-lang $LL \
        --lambda-cyc 0.9 --cyc-warmup 0 --cyc-every 1 --probe-n 256 \
        --resume-from $OUT/${tag}_s1.pt --steps 100 --batch-texts $BT --judge-temp $JT --fit-ridge 0.2 \
        --eval-judge $EVJ \
        --out $OUT/${tag}.pt "2>&1 | tee $OUT/${tag}.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${tag}.pt \
        --names-from $OUT/${tag}.pt --only $CONLY --alpha 8 --interv-prompts 48 \
        --out $OUT/${tag}_causal.pt "2>&1 | tee $OUT/${tag}_causal.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${tag}.pt \
        --names-from $OUT/${tag}.pt --only sweep --dirs-from $OUT/${tag}_causal.pt \
        --alphas $SWA --interv-prompts 96 \
        --out $OUT/${tag}_sweep.pt "2>&1 | tee $OUT/${tag}_sweep.log"
      ;;
    bsae)
      run python scripts/sae_baseline.py $DATA --num-concepts "$K" --support-min 0.05 \
        --dict $SD --topk $STK \
        --out $OUT/${tag}.pt "2>&1 | tee $OUT/${tag}.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${tag}.pt \
        --only $CONLY --alpha 8 --interv-prompts 48 \
        --out $OUT/${tag}_causal.pt "2>&1 | tee $OUT/${tag}_causal.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${tag}.pt \
        --dirs causal --only sweep --alphas $SWA --interv-prompts 96 \
        --out $OUT/${tag}_sweep.pt "2>&1 | tee $OUT/${tag}_sweep.log"
      ;;
    dmpca)
      run python scripts/dir_baseline.py $DATA --method diffmean_pca --num-concepts "$K" \
        --out $OUT/${tag}.pt "2>&1 | tee $OUT/${tag}.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${tag}.pt \
        --only $CONLY --alpha 8 --interv-prompts 48 \
        --out $OUT/${tag}_causal.pt "2>&1 | tee $OUT/${tag}_causal.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${tag}.pt \
        --dirs causal --only sweep --alphas $SWA --interv-prompts 96 \
        --out $OUT/${tag}_sweep.pt "2>&1 | tee $OUT/${tag}_sweep.log"
      ;;
    expert)
      # EXPERT TAXONOMY: hand-defined categories as named directions (judge-scored
      # cov(acts, category score)). No training stage -- the taxonomy is given. Runs the
      # SAME causal legs as the learned methods so the columns are comparable.
      run python scripts/expert_taxonomy.py $DATA --domain deception \
        --out $OUT/${tag}_dirs.pt "2>&1 | tee $OUT/${tag}_dirs.log"
      # LADDER through dir_baseline so structR2 / namedR2 / Fidelity(named) are computed by
      # the SAME code as the dmpca/bsae rows; --names-from keeps the expert's own labels
      # (re-captioning them would measure our captioner, not the expert taxonomy).
      run python scripts/dir_baseline.py $DATA --method external \
        --dirs-npy $OUT/${tag}_dirs.npy --names-from $OUT/${tag}_dirs.pt \
        --num-concepts "$K" --out $OUT/${tag}.pt "2>&1 | tee $OUT/${tag}.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${tag}.pt \
        --only $CONLY --alpha 8 --interv-prompts 48 \
        --out $OUT/${tag}_causal.pt "2>&1 | tee $OUT/${tag}_causal.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${tag}.pt \
        --dirs causal --only sweep --alphas $SWA --interv-prompts 96 \
        --out $OUT/${tag}_sweep.pt "2>&1 | tee $OUT/${tag}_sweep.log"
      ;;
    reft)
      # LoReFT (Wu et al. 2024) trained END-TO-END on this domain: frozen model, trainables
      # are a rank-K orthonormal projection + an affine map inside it, CE on positive-class
      # responses. The learned subspace rows are the atoms, captioned and run through the
      # SAME causal legs as every other method.
      run python scripts/reft_baseline.py $DATA --rank "$K" \
        --steps ${BAT_REFT_STEPS:-400} --out $U/${PFX}reft_basis_k${K}.npy \
        "2>&1 | tee $OUT/${tag}_basis.log"
      run python scripts/dir_baseline.py $DATA --dirs-npy $U/${PFX}reft_basis_k${K}.npy \
        --num-concepts "$K" --out $OUT/${tag}.pt "2>&1 | tee $OUT/${tag}.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${tag}.pt \
        --only $CONLY --alpha 8 --interv-prompts 48 \
        --out $OUT/${tag}_causal.pt "2>&1 | tee $OUT/${tag}_causal.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${tag}.pt \
        --dirs causal --only sweep --alphas $SWA --interv-prompts 96 \
        --out $OUT/${tag}_sweep.pt "2>&1 | tee $OUT/${tag}_sweep.log"
      ;;
    bank)
      # NO-METHOD ABLATION: top-K off the domain concept bank (set BANK_FILE to the
      # gen_bank.py deception bank -- there is no extraction cache in this domain and the
      # built-in fallback list is sycophancy-flavored).
      [[ -n "$BANK_FILE" ]] || { echo "bank leg needs BANK_FILE=<gen_bank output>"; exit 1; }
      run python scripts/bank_topk.py $DATA --num-concepts "$K" \
        --select ${BAT_BANK_SELECT:-top} --max-prompt-chars $MPC \
        --out $OUT/${tag}.pt "2>&1 | tee $OUT/${tag}.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${tag}.pt \
        --only $CONLY --alpha 8 --interv-prompts 48 \
        --out $OUT/${tag}_causal.pt "2>&1 | tee $OUT/${tag}_causal.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${tag}.pt \
        --dirs causal --only sweep --alphas $SWA --interv-prompts 96 \
        --out $OUT/${tag}_sweep.pt "2>&1 | tee $OUT/${tag}_sweep.log"
      ;;
    *) echo "unknown method: $METHOD (soft|bsae|dmpca|expert|reft|bank; psae not available for deception)"; exit 1 ;;
  esac
  # ---- ACME: axis (signed axis reconstruction) vs balanced (maximin over the three
  # deception facets). balanced prints 'WORST category cos: axis -> balanced' BEFORE
  # generating: if the worst is already positive there is no trade in this domain and the
  # two arms should agree -- that null SUPPORTS the mechanism story rather than denting it.
  for R in $ROUTERS; do
    rlog=$OUT/${tag}_racme_${R}.log
    if [[ -f $rlog ]]; then echo ">>> $rlog exists, skipping"; else
      DF=""; [[ -f $OUT/${tag}_causal.pt ]] && DF="--dirs-from $OUT/${tag}_causal.pt"
      run python scripts/routed_acme.py $RACME --checkpoint $OUT/${tag}.pt $DF \
        --router $R --out $OUT/${tag}_racme_${R}.pt "2>&1 | tee $rlog"
    fi
  done
  if [[ -f $OUT/${tag}_racme_balanced.pt && -f $OUT/${tag}_racme_axis.pt ]]; then
    run python scripts/racme_compare.py --a $OUT/${tag}_racme_balanced.pt \
    --b $OUT/${tag}_racme_axis.pt --label-a balanced --label-b axis \
    "2>&1 | tee $OUT/cmp_dec_${METHOD}_k${K}.log"
  else echo ">>> only one router arm present, skipping racme_compare"; fi
done
echo "=== deception battery done: $METHOD K=$* ==="
echo "summarize: python scripts/battery_summary.py --prefix ${PFX}"
