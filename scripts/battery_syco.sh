#!/usr/bin/env bash
# SOCIAL-SYCOPHANCY METHOD BATTERY: AIE / fidelity / selectivity at K in {1,3,5,7,9}.
#
# methods:
#   soft   soft-concepts train -> ladder fidelity -> ceiling+distill (AIE) -> sweep (honest sel)
#   bsae   behavior TopK SAE (sae_baseline)   -> SAE-NAMED fidelity -> causal legs on decoder dirs
#   psae   pretrained SAE via $PSAE_WEIGHTS   -> same (BLOCKED until a Qwen3-4B residual SAE exists)
#   dmpca  DiffMean + within-class-PC dirs    -> DIR-NAMED fidelity -> causal legs on raw dirs
#   reft   LoReFT trained end-to-end (rank-K orthonormal subspace, CE on the positive
#          class) -> its learned subspace rows captioned + causal legs via dir_baseline
#
# fidelity comes from each build's log (Fidelity(named), atom-aligned protocol everywhere);
# AIE from the ceiling+distill leg (own-AIE @ a8 + prompting oracle); selectivity from the
# sweep leg (PER-ATOM BEST ALPHA, honest select-even/score-odd). K=1 selectivity = NaN
# (no off-target exists).
#
# usage: bash scripts/battery_syco.sh <GPU> <method> <K...>
#   bash scripts/battery_syco.sh 0 soft 1 3 5 7 9
#   bash scripts/battery_syco.sh 1 dmpca 1 3 5 7 9
set -e -o pipefail
GPU=$1; METHOD=$2; shift 2
U=/scratch/mech-taxonomy
# SUBJECT-MODEL OVERRIDES (defaults = Qwen3-4B L20): make a config with
# scripts/make_subject_config.py, extract fresh caches, then e.g.
#   BAT_CFG=configs/experiments/..._g12.yaml BAT_UNITS=$U/units_g12_syco.npz \
#   BAT_PACTS=$U/prompt_last_g12_syco.npz BAT_PREFIX=bat_g12_ bash scripts/battery_syco.sh 0 soft 1 3 5
CFG=${BAT_CFG:-configs/experiments/sycophancy_grounded_sparse_decomposition_l20_resp.yaml}
UNITS=${BAT_UNITS:-$U/units_l20.npz}
PACTS=${BAT_PACTS:-$U/prompt_last_l20.npz}
PFX=${BAT_PREFIX:-bat_}
BT=${BAT_BATCH_TEXTS:-256}
RJP=${BAT_RACME_PROVIDER:-local}               # matches battery_dec/emo; 'openai' needs OPENAI_API_KEY
JT=${BAT_JUDGE_TEMP:-1.5}                        # raise (3-4) for judges that rate soft prompts a hard 0 (saturated digits = no gradient)                      # drop to 96-128 for a ~35B subject (judge backprop VRAM)
MPC=${BAT_MAX_PROMPT_CHARS:-400}   # 67% of OEQ prompts exceed this; 1200 = the
                                   # robustness control (OEQ task is front-loaded,
                                   # so truncation costs detail, not the question)
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
# SEED: vary for cross-seed robustness (use a distinct BAT_PREFIX per seed so runs
# do not overwrite each other). causal_diag must see the SAME seed as training.
SEED=${BAT_SEED:-0}
DATA="--config $CFG --units $UNITS --prompt-acts $PACTS --seed $SEED"
CD="--config $CFG --units $UNITS --acts-key sent_acts --prompt-acts $PACTS --max-prompt-chars $MPC --seed $SEED"
ENV="CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src"

run() {
  if [[ -n "$BAT_SKIP" ]]; then
    local o; o=$(echo "$*" | grep -o '\-\-out [^ ]*\.pt' | head -1 | awk '{print $2}')
    if [[ -n "$o" && -f "$o" ]]; then echo ">>> skip (exists): $o"; return 0; fi
  fi
  echo ">>> $*"; eval "$ENV $*"
}
# the pre-axis racme leg used the DEFAULT linear router -- the weakest arm in the router
# battery. Kept for continuity, opt-in via BAT_LEGACY_RACME=1; the axis/balanced ACME block
# after the case statement is the standard one now.
legacy_racme() {
  if [[ -n "${BAT_LEGACY_RACME}" ]]; then run python scripts/routed_acme.py "$@";
  else echo ">>> legacy linear-router racme skipped (BAT_LEGACY_RACME=1 to run)"; fi; }

for K in "$@"; do
  tag="${METHOD}_k${K}"
  case $METHOD in
    soft)
      # THE FULL METHOD: stage-1 train -> stage-2 cycle (sayability + anchor coordinate-ascent)
      # -> causal legs read off the CYCLE ANCHORS (--names-from). Snap-only naming collapses to
      # the advice family and caps selectivity at ~1 -- the anchors are where sel>=2 comes from.
      if [[ -f $OUT/${PFX}${tag}_s1.pt ]]; then
        echo ">>> stage-1 exists ($OUT/${PFX}${tag}_s1.pt), skipping retrain"
      else
        run python scripts/soft_concepts.py $DATA --acts-key sent_acts --num-concepts "$K" \
          --target acts_wc_axisw --lambda-fid $LF --lambda-div $LD --lambda-lang $LL \
          --steps 400 --batch-texts $BT --judge-temp $JT --fit-ridge 0.2 \
          --out $OUT/${PFX}${tag}_s1.pt "2>&1 | tee $OUT/${PFX}${tag}_s1.log"
      fi
      run python scripts/soft_concepts.py $DATA --acts-key sent_acts --num-concepts "$K" \
        --target acts_wc_axisw --lambda-fid $LF --lambda-div $LD --lambda-lang $LL \
        --lambda-cyc 0.9 --cyc-warmup 0 --cyc-every 1 --resume-from $OUT/${PFX}${tag}_s1.pt \
        --steps 100 --batch-texts $BT --judge-temp $JT --fit-ridge 0.2 \
        --out $OUT/${PFX}${tag}.pt "2>&1 | tee $OUT/${PFX}${tag}.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${PFX}${tag}.pt \
        --names-from $OUT/${PFX}${tag}.pt --only $CONLY --alpha 8 --interv-prompts 48 \
        --out $OUT/${PFX}${tag}_causal.pt "2>&1 | tee $OUT/${PFX}${tag}_causal.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${PFX}${tag}.pt \
        --names-from $OUT/${PFX}${tag}.pt --only sweep --dirs-from $OUT/${PFX}${tag}_causal.pt \
        --alphas $SWA --interv-prompts 96 \
        --out $OUT/${PFX}${tag}_sweep.pt "2>&1 | tee $OUT/${PFX}${tag}_sweep.log"
      legacy_racme $DATA --checkpoint $OUT/${PFX}${tag}.pt \
        --dirs-from $OUT/${PFX}${tag}_causal.pt --alpha-train 2 --alpha-eval 1,2,4 --judge-provider $RJP \
        --out $OUT/${PFX}${tag}_racme.pt "2>&1 | tee $OUT/${PFX}${tag}_racme.log"
      ;;
    bsae|psae)
      EXTRA=""
      if [[ $METHOD == psae ]]; then
        [[ -z "$PSAE_WEIGHTS" ]] && { echo "psae: set PSAE_WEIGHTS=<W_enc/b_enc/W_dec .pt>"; exit 1; }
        EXTRA="--sae-from $PSAE_WEIGHTS"
      fi
      run python scripts/sae_baseline.py $DATA --num-concepts "$K" --support-min 0.05 \
        --dict $SD --topk $STK $EXTRA \
        --out $OUT/${PFX}${tag}.pt "2>&1 | tee $OUT/${PFX}${tag}.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${PFX}${tag}.pt \
        --only $CONLY --alpha 8 --interv-prompts 48 \
        --out $OUT/${PFX}${tag}_causal.pt "2>&1 | tee $OUT/${PFX}${tag}_causal.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${PFX}${tag}.pt \
        --dirs causal --only sweep --alphas $SWA --interv-prompts 96 \
        --out $OUT/${PFX}${tag}_sweep.pt "2>&1 | tee $OUT/${PFX}${tag}_sweep.log"
      legacy_racme $DATA --checkpoint $OUT/${PFX}${tag}.pt \
        --alpha-train 2 --alpha-eval 1,2,4 --judge-provider $RJP --out $OUT/${PFX}${tag}_racme.pt "2>&1 | tee $OUT/${PFX}${tag}_racme.log"
      ;;
    dmpca)
      run python scripts/dir_baseline.py $DATA --method diffmean_pca --num-concepts "$K" \
        --out $OUT/${PFX}${tag}.pt "2>&1 | tee $OUT/${PFX}${tag}.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${PFX}${tag}.pt \
        --only $CONLY --alpha 8 --interv-prompts 48 \
        --out $OUT/${PFX}${tag}_causal.pt "2>&1 | tee $OUT/${PFX}${tag}_causal.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${PFX}${tag}.pt \
        --dirs causal --only sweep --alphas $SWA --interv-prompts 96 \
        --out $OUT/${PFX}${tag}_sweep.pt "2>&1 | tee $OUT/${PFX}${tag}_sweep.log"
      legacy_racme $DATA --checkpoint $OUT/${PFX}${tag}.pt \
        --alpha-train 2 --alpha-eval 1,2,4 --judge-provider $RJP --out $OUT/${PFX}${tag}_racme.pt "2>&1 | tee $OUT/${PFX}${tag}_racme.log"
      ;;
    reft)
      # LoReFT (Wu et al. 2024) trained END-TO-END: frozen model, trainables are a rank-K
      # orthonormal projection + an affine map inside that subspace, CE on positive-class
      # responses. The K learned subspace rows are the atoms. (The old leg ranked DiffMean
      # directions of rl_concept_discovery concepts -- a frozen basis, not ReFT; set
      # REFT_CONCEPTS to fall back to it.)
      if [[ -n "$REFT_CONCEPTS" ]]; then
        run python scripts/make_reft_basis.py --config $CFG --concepts "$REFT_CONCEPTS" \
          --units $UNITS --k "$K" --out $U/${PFX}reft_basis_k${K}.npy \
          "2>&1 | tee $OUT/${PFX}${tag}_basis.log"
      else
        run python scripts/reft_baseline.py $DATA --rank "$K" \
          --steps ${BAT_REFT_STEPS:-400} --out $U/${PFX}reft_basis_k${K}.npy \
          "2>&1 | tee $OUT/${PFX}${tag}_basis.log"
      fi
      run python scripts/dir_baseline.py $DATA --dirs-npy $U/${PFX}reft_basis_k${K}.npy \
        --num-concepts "$K" --out $OUT/${PFX}${tag}.pt "2>&1 | tee $OUT/${PFX}${tag}.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${PFX}${tag}.pt \
        --only $CONLY --alpha 8 --interv-prompts 48 \
        --out $OUT/${PFX}${tag}_causal.pt "2>&1 | tee $OUT/${PFX}${tag}_causal.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${PFX}${tag}.pt \
        --dirs causal --only sweep --alphas $SWA --interv-prompts 96 \
        --out $OUT/${PFX}${tag}_sweep.pt "2>&1 | tee $OUT/${PFX}${tag}_sweep.log"
      legacy_racme $DATA --checkpoint $OUT/${PFX}${tag}.pt \
        --alpha-train 2 --alpha-eval 1,2,4 --judge-provider $RJP --out $OUT/${PFX}${tag}_racme.pt "2>&1 | tee $OUT/${PFX}${tag}_racme.log"
      ;;
    expert)
      # EXPERT TAXONOMY: hand-defined categories as named directions (judge-scored
      # cov(acts, category score)). No training stage -- the taxonomy is given. Runs the
      # SAME causal legs as the learned methods so the columns are comparable.
      run python scripts/expert_taxonomy.py $DATA --domain sycophancy \
        --out $OUT/${PFX}${tag}_dirs.pt "2>&1 | tee $OUT/${PFX}${tag}_dirs.log"
      # LADDER through dir_baseline so structR2 / namedR2 / Fidelity(named) are computed by
      # the SAME code as the dmpca/bsae rows; --names-from keeps the expert's own labels
      # (re-captioning them would measure our captioner, not the expert taxonomy).
      run python scripts/dir_baseline.py $DATA --method external \
        --dirs-npy $OUT/${PFX}${tag}_dirs.npy --names-from $OUT/${PFX}${tag}_dirs.pt \
        --num-concepts "$K" --out $OUT/${PFX}${tag}.pt "2>&1 | tee $OUT/${PFX}${tag}.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${PFX}${tag}.pt \
        --only $CONLY --alpha 8 --interv-prompts 48 \
        --out $OUT/${PFX}${tag}_causal.pt "2>&1 | tee $OUT/${PFX}${tag}_causal.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${PFX}${tag}.pt \
        --dirs causal --only sweep --alphas $SWA --interv-prompts 96 \
        --out $OUT/${PFX}${tag}_sweep.pt "2>&1 | tee $OUT/${PFX}${tag}_sweep.log"
      ;;
    bank)
      # NO-METHOD ABLATION: no NLD at all -- take the top-K concepts off the ~1024-phrase
      # candidate bank by correlation with the behavior label, build each direction as
      # cov(acts, that concept's judged score), and run the SAME legs. BAT_BANK_SELECT=greedy
      # swaps plain top-K for redundancy-aware forward selection.
      run python scripts/bank_topk.py $DATA --num-concepts "$K" \
        --select ${BAT_BANK_SELECT:-top} --max-prompt-chars $MPC \
        --out $OUT/${PFX}${tag}.pt "2>&1 | tee $OUT/${PFX}${tag}.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${PFX}${tag}.pt \
        --only $CONLY --alpha 8 --interv-prompts 48 \
        --out $OUT/${PFX}${tag}_causal.pt "2>&1 | tee $OUT/${PFX}${tag}_causal.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${PFX}${tag}.pt \
        --dirs causal --only sweep --alphas $SWA --interv-prompts 96 \
        --out $OUT/${PFX}${tag}_sweep.pt "2>&1 | tee $OUT/${PFX}${tag}_sweep.log"
      ;;
    *) echo "unknown method: $METHOD (soft|bsae|psae|dmpca|reft)"; exit 1 ;;
  esac
  # ---- ACME: axis (signed axis reconstruction) vs balanced (maximin over the three expert
  # categories). balanced prints 'WORST category cos: axis -> balanced' BEFORE generating:
  # a negative worst-cosine means the axis MUST trade that category away.
  for R in $ROUTERS; do
    rlog=$OUT/${PFX}${tag}_racme_${R}.log
    if [[ -f $rlog ]]; then echo ">>> $rlog exists, skipping"; else
      DF=""; [[ -f $OUT/${PFX}${tag}_causal.pt ]] && DF="--dirs-from $OUT/${PFX}${tag}_causal.pt"
      run python scripts/routed_acme.py $DATA --checkpoint $OUT/${PFX}${tag}.pt $DF \
        --router $R --max-prompt-chars $MPC --alpha-eval 2,4,8 --interv-prompts 96 \
        --judge-provider $RJP --out $OUT/${PFX}${tag}_racme_${R}.pt "2>&1 | tee $rlog"
    fi
  done
  if [[ -f $OUT/${PFX}${tag}_racme_balanced.pt && -f $OUT/${PFX}${tag}_racme_axis.pt ]]; then
    run python scripts/racme_compare.py --a $OUT/${PFX}${tag}_racme_balanced.pt \
    --b $OUT/${PFX}${tag}_racme_axis.pt --label-a balanced --label-b axis \
    "2>&1 | tee $OUT/cmp_syco_${METHOD}_k${K}.log"
  else echo ">>> only one router arm present, skipping racme_compare"; fi
done
echo "=== battery done: $METHOD K=$* ==="
echo "collect: grep -E 'Fidelity|PROMPTING CEILING.*median|DISTILLED.*median|PER-ATOM-ALPHA' $OUT/${PFX}${METHOD}_k*[!p].log $OUT/${PFX}${METHOD}_k*_causal.log $OUT/${PFX}${METHOD}_k*_sweep.log"
