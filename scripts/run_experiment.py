"""Build and evaluate one dictionary: extract activations, fit, then measure selectivity, AIE and transfer.

  python scripts/run_experiment.py sycophancy --method nld --k 5
  python scripts/run_experiment.py deception --method pca --k 3 --stages fit selectivity aie
  python scripts/run_experiment.py emotion --method expert --dry-run

Stages (default: all):
  fit          extract activations once per domain, then fit the dictionary -> dictionary.pt
  selectivity  steer along each entry's direction at alpha 8 on 48 prompts -> selectivity.pt
  aie          broad-behavior effect of the balanced combination of the K directions
               (sycophancy: local LLM judge; emotion: VADER; deception: MASK lying rate)
  transfer     AITA (sycophancy) or LiarsBench (deception) detection from direction and concept scores

Outputs go to <output-root>/<domain>/<method>-k<K>-seed<seed>/, activations to <output-root>/<domain>/activations/.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

STAGES = ("fit", "selectivity", "aie", "transfer")
MAX_PROMPT_CHARS = {"sycophancy": 400, "deception": 800, "emotion": 400}
EXPERT_K = {"sycophancy": 3, "deception": 3, "emotion": 6}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("domain", choices=tuple(MAX_PROMPT_CHARS))
    parser.add_argument("--method", choices=("nld", "pca", "sae", "expert"), default="nld")
    parser.add_argument("--k", type=int, default=5, help="dictionary size (the expert dictionary has a fixed size)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--pca", choices=("diffmean", "top"), default="diffmean",
                        help="diffmean: mean-difference direction plus the top K-1 within-class components; "
                             "top: the top K components of behavior-present activations")
    parser.add_argument("--prefix-cache-gb", type=float, default=0.0,
                        help="host memory (GB) for caching encoded texts during NLD training; faster when available")
    parser.add_argument("--config", type=Path, help="override the domain config (model, layer, dataset)")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = args.config.resolve() if args.config else root / "configs/experiments" / f"{args.domain}.yaml"
    out = args.output_root.resolve() / args.domain
    cache = out / "activations"
    units, prompts = cache / "units.npz", cache / "prompts.npz"
    k = EXPERT_K[args.domain] if args.method == "expert" else args.k
    run_dir = out / f"{args.method}-k{k}-seed{args.seed}"
    dictionary, steering = run_dir / "dictionary.pt", run_dir / "selectivity.pt"
    emotion_prompts = root / "configs/emotion_steering_prompts.txt"

    def run(script, *options):
        command = [sys.executable, str(root / "scripts" / f"{script}.py"), "--config", str(config), *map(str, options)]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=root, check=True)

    if not args.dry_run:
        check_inputs(parser, root, config, cache, units, prompts)
        run_dir.mkdir(parents=True, exist_ok=True)
    shared = ("--units", units, "--prompt-acts", prompts, "--seed", args.seed,
              "--max-prompt-chars", MAX_PROMPT_CHARS[args.domain])

    if "fit" in args.stages:
        if not units.exists():
            run("extract_units", "--out", units)
        if not prompts.exists():
            run("extract_prompt_acts", "--out", prompts)
        fit(run, args, shared, k, run_dir, dictionary)

    if "selectivity" in args.stages and k > 1:
        options = [*shared, "--acts-key", "sent_acts", "--checkpoint", dictionary,
                   "--only", "distill", "--alpha", 8, "--interv-prompts", 48, "--out", steering]
        if args.method == "nld":
            options += ["--names-from", dictionary]
        if args.domain == "emotion":
            options += ["--gen-prompts", emotion_prompts]
        run("selectivity", *options)

    if "aie" in args.stages:
        steer = [*shared, "--checkpoint", dictionary, "--router", "balanced", "--domain", args.domain]
        if steering.exists() or args.dry_run:
            steer += ["--dirs-from", steering]
        if args.domain == "deception":
            vector = run_dir / "aie_vector.pt"
            run("steer_behavior", *steer, "--emit-vector", "--out", vector)
            run("mask_eval", "--vector", vector, "--mask-config", "known_facts", "--n", 200,
                "--alphas", "0,4,8", "--out", run_dir / "aie.pt")
        else:
            judge = (["--judge-provider", "local"] if args.domain == "sycophancy"
                     else ["--judge-provider", "vader", "--gen-prompts", emotion_prompts])
            run("steer_behavior", *steer, *judge, "--alpha-eval", "2,4,8", "--interv-prompts", 96,
                "--out", run_dir / "aie.pt")

    if "transfer" in args.stages:
        if args.domain == "sycophancy":
            run("transfer_aita", "--checkpoints", dictionary, "--n", 600, "--out", run_dir / "transfer_aita.pt")
        elif args.domain == "deception":
            run("transfer_liarsbench", "--subset", "instructed-deception", "--strip-system", "--balance-by-model",
                "--len-deconf", "--checkpoints", dictionary, "--out", run_dir / "transfer_liarsbench.json")


def fit(run, args, shared, k, run_dir, dictionary):
    if args.method == "nld":
        stage1 = run_dir / "stage1.pt"
        settings = (*shared, "--acts-key", "sent_acts", "--num-concepts", k, "--target", "acts_wc_axisw",
                    "--lambda-fid", 1, "--lambda-div", 2, "--lambda-lang", 0.4, "--fit-ridge", 0.2,
                    "--batch-texts", 256, "--probe-n", 256 if args.domain == "deception" else 512,
                    "--judge-temp", 1.5, "--prefix-cache-gb", args.prefix_cache_gb)
        run("train_nld", *settings, "--steps", 400 if args.domain == "sycophancy" else 300, "--out", stage1)
        run("train_nld", *settings, "--steps", 100, "--resume-from", stage1,
            "--lambda-cyc", 0.9, "--cyc-warmup", 0, "--cyc-every", 1, "--out", dictionary)
    elif args.method == "pca":
        run("describe_directions", *shared, "--num-concepts", k,
            "--method", "diffmean_pca" if args.pca == "diffmean" else "pca", "--out", dictionary)
    elif args.method == "sae":
        run("sae_baseline", *shared, "--num-concepts", k, "--dict", 1024, "--topk", 64,
            "--support-min", 0.05, "--out", dictionary)
    else:
        expert = run_dir / "expert.pt"
        run("expert_baseline", *shared, "--domain", args.domain, "--out", expert)
        run("describe_directions", *shared, "--num-concepts", k, "--method", "external",
            "--dirs-npy", expert.with_suffix(".npy"), "--names-from", expert, "--out", dictionary)


def check_inputs(parser, root, config, cache, units, prompts):
    """Refuse to reuse cached activations that were built from a different config or dataset."""
    import yaml

    with config.open() as file:
        dataset = root / yaml.safe_load(file)["behavior"]["train_data_dir"]
    if not dataset.is_file():
        parser.error(f"dataset missing at {dataset}; see the README for data setup")
    signature = sha256(config.read_bytes()).hexdigest() + "\n" + sha256(dataset.read_bytes()).hexdigest() + "\n"
    manifest = cache / "inputs.sha256"
    if (units.exists() or prompts.exists()) and (not manifest.is_file() or manifest.read_text() != signature):
        parser.error("cached activations were built from other inputs; use a separate --output-root")
    cache.mkdir(parents=True, exist_ok=True)
    manifest.write_text(signature)


if __name__ == "__main__":
    main()
