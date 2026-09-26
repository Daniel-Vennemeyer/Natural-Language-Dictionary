#!/usr/bin/env bash
# EMOBANK-EMOTION METHOD BATTERY: fidelity / AIE / selectivity at K, mirroring battery_dec.
#
# Domain: EmoBank valence tails (label 1 = V >= mean+1sd, 0 = V <= mean-1sd; ~1.1k/1.4k,
# n=2578 -> sycophancy-scale, default probe/CYCV splits are fine). Found texts with one
# constant prompt frame (difraud pattern). ACME leg runs via routed_acme --domain emotion
# (three banded facets: positive_affect / warmth / optimism), axis vs balanced routers.
# reft = LoReFT trained end-to-end (no rl artifact needed). Held-out judge = gemma-2-9b-it.
#
# ONE-TIME EXTRACTION (before the first battery run):
#   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/extract_units.py \
#     --config configs/experiments/emotion_emobank_qwen4b_l20.yaml \
#     --out /scratch/mech-taxonomy/units_emo_l20.npz
#   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/extract_prompt_acts.py \
#     --config configs/experiments/emotion_emobank_qwen4b_l20.yaml \
#     --out /scratch/mech-taxonomy/prompt_last_emo_l20.npz
#
# usage: bash scripts/battery_emo.sh <GPU> <method> <K...>
#   bash scripts/battery_emo.sh 0 soft 1 3 5 7
#   bash scripts/battery_emo.sh 1 dmpca 1 3 5 7
#   bash scripts/battery_emo.sh 1 bsae 1 3 5 7
# summarize: python scripts/battery_summary.py --prefix ebat_
set -e
GPU=$1; METHOD=$2; shift 2
U=/scratch/mech-taxonomy
CFG=configs/experiments/emotion_emobank_qwen4b_l20.yaml
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
DATA="--config $CFG --units $U/units_emo_l20.npz --prompt-acts $U/prompt_last_emo_l20.npz --seed $SEED"
# GENERATION PROMPTS: the dataset 'prompt' is one constant frame (found texts), so the
# causal legs need real prompt variety. Continuation-of-corpus-text tests style imitation,
# not the model's own affect -- instead we use AFFECT-ELICITING first-person prompts:
# describe/recount/react frames crossed with deliberately MUNDANE topics. A neutral
# situation leaves the response's emotional coloring free (baseline mid-scale), so a
# steered atom has room to express and the judge has range to detect it -- the analog of
# the OEQ advice questions in the sycophancy battery.
GP=$U/emo_gen_prompts.txt
if [[ ! -f $GP ]]; then
  python3 - <<EOF
templates = [
    "In a few sentences, describe {}.",
    "Write a short diary entry about {}.",
    "Tell a friend, in a few sentences, about {}.",
    "Write the opening lines of a story that begins with {}.",
    "In a few sentences, how would you feel about {}?",
    "Recount, in a few sentences, an experience involving {}.",
    "Write a short message to a coworker about {}.",
    "In a few sentences, what would it be like to deal with {}?",
]
topics = [
    "your morning commute", "a change in the weather", "moving to a new apartment",
    "cooking dinner on a weeknight", "a long meeting at work", "waiting in line at the DMV",
    "a phone call from an old friend", "a package arriving at the door",
    "the first day at a new job", "a walk through the neighborhood",
    "a family dinner on a Sunday", "a delayed train", "reorganizing a closet",
    "a trip to the grocery store", "an email you have been putting off",
    "a neighbor playing music", "a quiet afternoon at home", "planning a weekend trip",
    "a doctor's appointment", "returning a purchase to the store",
    "a coworker leaving the company", "learning to use a new phone",
    "watering the plants", "an unexpected knock at the door",
]
with open("$GP", "w") as f:
    for t in templates:
        for tp in topics:
            f.write(t.format(tp) + "\n")
print(f"[emo] wrote {len(templates) * len(topics)} affect-eliciting prompts -> $GP")
EOF
fi
CD="--config $CFG --units $U/units_emo_l20.npz --acts-key sent_acts --prompt-acts $U/prompt_last_emo_l20.npz --gen-prompts $GP --seed $SEED"
RP=${BAT_RACME_PROVIDER:-local}                  # ACME judge: local 30B (free) or openai
# emotion ACME: three banded facets (positive_affect / warmth / optimism). The dataset
# prompt is a constant frame, so the ACME leg needs --gen-prompts too, exactly like the
# causal legs -- otherwise 96 identical prompts collapse to ONE effective sample.
RACME="--config $CFG --units $U/units_emo_l20.npz --prompt-acts $U/prompt_last_emo_l20.npz \
 --domain emotion --gen-prompts $GP --judge-provider $RP --alpha-eval 2,4,8 --interv-prompts 96"
ENV="CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src"

run() {
  if [[ -n "$BAT_SKIP" ]]; then
    local o; o=$(echo "$*" | grep -o '\-\-out [^ ]*\.pt' | head -1 | awk '{print $2}')
    if [[ -n "$o" && -f "$o" ]]; then echo ">>> skip (exists): $o"; return 0; fi
  fi
  echo ">>> $*"; eval "$ENV $*"
}

for K in "$@"; do
  tag="ebat_${METHOD}_k${K}"
  case $METHOD in
    soft)
      if [[ -f $OUT/${tag}_s1.pt ]]; then
        echo ">>> stage-1 exists ($OUT/${tag}_s1.pt), skipping retrain"
      else
        run python scripts/soft_concepts.py $DATA --acts-key sent_acts --num-concepts "$K" \
          --target acts_wc_axisw --lambda-fid $LF --lambda-div $LD --lambda-lang $LL \
          --steps 300 --batch-texts 256 --fit-ridge 0.2 \
          --out $OUT/${tag}_s1.pt "2>&1 | tee $OUT/${tag}_s1.log"
      fi
      run python scripts/soft_concepts.py $DATA --acts-key sent_acts --num-concepts "$K" \
        --target acts_wc_axisw --lambda-fid $LF --lambda-div $LD --lambda-lang $LL \
        --lambda-cyc 0.9 --cyc-warmup 0 --cyc-every 1 \
        --resume-from $OUT/${tag}_s1.pt --steps 100 --batch-texts 256 --fit-ridge 0.2 \
        --eval-judge google/gemma-2-9b-it \
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
      run python scripts/expert_taxonomy.py $DATA --domain emotion \
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
        --steps ${BAT_REFT_STEPS:-400} --out $U/ebat_reft_basis_k${K}.npy \
        "2>&1 | tee $OUT/${tag}_basis.log"
      run python scripts/dir_baseline.py $DATA --dirs-npy $U/ebat_reft_basis_k${K}.npy \
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
      # gen_bank.py emotion bank).
      [[ -n "$BANK_FILE" ]] || { echo "bank leg needs BANK_FILE=<gen_bank output>"; exit 1; }
      run python scripts/bank_topk.py $DATA --num-concepts "$K" \
        --select ${BAT_BANK_SELECT:-top} --max-prompt-chars ${MPC:-400} \
        --out $OUT/${tag}.pt "2>&1 | tee $OUT/${tag}.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${tag}.pt \
        --only $CONLY --alpha 8 --interv-prompts 48 \
        --out $OUT/${tag}_causal.pt "2>&1 | tee $OUT/${tag}_causal.log"
      run python scripts/causal_diag.py $CD --checkpoint $OUT/${tag}.pt \
        --dirs causal --only sweep --alphas $SWA --interv-prompts 96 \
        --out $OUT/${tag}_sweep.pt "2>&1 | tee $OUT/${tag}_sweep.log"
      ;;
    *) echo "unknown method: $METHOD (soft|bsae|dmpca|expert|reft|bank; psae not available for emotion)"; exit 1 ;;
  esac
  # ---- ACME: axis vs balanced (maximin over the three valence facets) + paired compare
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
    "2>&1 | tee $OUT/cmp_emo_${METHOD}_k${K}.log"
  else echo ">>> only one router arm present, skipping racme_compare"; fi
done
echo "=== emotion battery done: $METHOD K=$* ==="
echo "summarize: python scripts/battery_summary.py --prefix ebat_"
