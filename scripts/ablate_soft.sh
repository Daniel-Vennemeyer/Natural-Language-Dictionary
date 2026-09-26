#!/usr/bin/env bash
# NLD soft-concepts ABLATION GRID -- how simple can the pipeline get while remaining useful?
#
# Every run trains on its ABLATED prep/target but scores the final ladder on the CANONICAL
# target (--eval-canonical), so SOFT/NAMED/SET R^2 are comparable across rows. K=4, 300 steps,
# random init, no bank, held-out judge -- identical protocol per row.
#
#   prep ablations             regularizer ablations
#   B  - length deconfound     G  - lambda_fid   (interpretation fidelity)
#   C  - prompt residual       H  - lambda_div   (behavior-corr distinctness)
#   D  - axisw weighting       I  - lambda_lang  (sayability pull)
#   E  - class conditioning    J  recon only (all three off)
#   F  minimal prep (B+C+E)    A  baseline (full prep, all regs)
#
# usage: bash scripts/ablate_soft.sh <GPU> <row...>     e.g.:
#   bash scripts/ablate_soft.sh 0 A B C D E     # prep rows on GPU 0
#   bash scripts/ablate_soft.sh 1 F G H I J     # reg rows on GPU 1
set -e
GPU=$1; shift
U=/scratch/mech-taxonomy
BASE="--config configs/experiments/sycophancy_grounded_sparse_decomposition_l20_resp.yaml \
 --units $U/units_l20.npz --acts-key sent_acts --prompt-acts $U/prompt_last_l20.npz \
 --num-concepts 4 --steps 300 --batch-texts 256 --fit-ridge 0.2 \
 --eval-judge Qwen/Qwen2.5-7B-Instruct --eval-canonical"
REG="--lambda-fid 1.0 --lambda-div 2.0 --lambda-lang 0.4"
declare -A ABL=(
  [A]="--target acts_wc_axisw $REG"
  [B]="--target acts_wc_axisw $REG --no-len-deconf"
  [C]="--target acts_wc_axisw $REG --no-prompt-residual"
  [D]="--target acts_wc $REG"
  [E]="--target acts_pc $REG"
  [F]="--target acts_pc $REG --no-len-deconf --no-prompt-residual"
  [G]="--target acts_wc_axisw --lambda-fid 0 --lambda-div 2.0 --lambda-lang 0.4"
  [H]="--target acts_wc_axisw --lambda-fid 1.0 --lambda-div 0 --lambda-lang 0.4"
  [I]="--target acts_wc_axisw --lambda-fid 1.0 --lambda-div 2.0 --lambda-lang 0"
  [J]="--target acts_wc_axisw --lambda-fid 0 --lambda-div 0 --lambda-lang 0"
)
for name in "$@"; do
  if [[ -z "${ABL[$name]}" ]]; then echo "unknown row: $name"; exit 1; fi
  out=outputs/runs/soft_wc/abl_${name}.pt
  echo "=== ablation ${name}: ${ABL[$name]} ==="
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
    python scripts/soft_concepts.py $BASE ${ABL[$name]} --out "$out" \
    2>&1 | tee "outputs/runs/soft_wc/abl_${name}.log"
done
