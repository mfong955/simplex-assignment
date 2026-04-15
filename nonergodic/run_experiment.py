"""End-to-end experiment runner.

Usage:
    # Default run (CPU auto-upgrades to CUDA/MPS if available)
    python -m nonergodic.run_experiment

    # Custom output dir
    python -m nonergodic.run_experiment --out_dir /path/to/results

    # Skip training, load existing checkpoint
    python -m nonergodic.run_experiment --load_from results/model_final.pt

    # Shorter debug run
    python -m nonergodic.run_experiment --n_steps 500 --batch_size 64 --seq_len 16
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from nonergodic.config import (
    ExperimentConfig,
    DataConfig,
    ModelConfig,
    TrainConfig,
    AnalysisConfig,
    get_default_config,
)
from nonergodic.train import build_model, train, load_checkpoint
from nonergodic.analyze import run_full_analysis
from nonergodic.figures import make_all_figures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Non-ergodic Mess3 transformer experiment")
    p.add_argument("--out_dir", type=str, default="results",
                   help="Directory for checkpoints, results, and figures")
    p.add_argument("--load_from", type=str, default=None,
                   help="Load model from checkpoint instead of training")
    p.add_argument("--n_steps", type=int, default=None,
                   help="Override number of training steps")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--n_layers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None,
                   help="Force device: cpu | cuda | mps")
    p.add_argument("--skip_analysis", action="store_true",
                   help="Skip analysis/figures (just train)")
    p.add_argument("--skip_figures", action="store_true",
                   help="Skip figure generation (just save results pickle)")
    p.add_argument("--analysis_sequences", type=int, default=None)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = get_default_config()
    cfg.out_dir = args.out_dir

    if args.n_steps is not None:
        cfg.data.n_train_batches = args.n_steps
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.seq_len is not None:
        cfg.data.seq_len = args.seq_len
        # ensure model context window is large enough
        cfg.model.n_ctx = max(cfg.model.n_ctx, args.seq_len + 4)
    if args.d_model is not None:
        cfg.model.d_model = args.d_model
        cfg.model.d_mlp = 4 * args.d_model
    if args.n_layers is not None:
        cfg.model.n_layers = args.n_layers
    if args.seed is not None:
        cfg.data.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    if args.analysis_sequences is not None:
        cfg.data.n_analysis_sequences = args.analysis_sequences

    # Auto-detect GPU
    if cfg.device == "cpu":
        if torch.cuda.is_available():
            cfg.device = "cuda"
        elif torch.backends.mps.is_available():
            cfg.device = "mps"

    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Non-ergodic Mess3 Transformer Experiment")
    print("=" * 60)
    print(f"  K sources:    {3}")
    print(f"  Sequence len: {cfg.data.seq_len}")
    print(f"  Batch size:   {cfg.data.batch_size}")
    print(f"  Train steps:  {cfg.data.n_train_batches}")
    print(f"  Model:        d={cfg.model.d_model}, L={cfg.model.n_layers}, "
          f"H={cfg.model.n_heads}, ctx={cfg.model.n_ctx}")
    print(f"  Device:       {cfg.device}")
    print(f"  Output:       {out_dir.resolve()}")
    print("=" * 60)

    # ---- Phase 1: Training ----
    if args.load_from:
        print(f"\nLoading checkpoint from {args.load_from}")
        model = load_checkpoint(args.load_from, device=cfg.device)
        history = {"steps": [], "losses": [], "lrs": []}
    else:
        print("\nPhase 1: Training")
        model, history = train(cfg)
        # Save history
        with open(out_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    if args.skip_analysis:
        print("Skipping analysis (--skip_analysis). Done.")
        return

    # ---- Phase 2: Analysis ----
    print("\nPhase 2: Analysis")
    results = run_full_analysis(model, cfg, device=cfg.device)

    # Save results (without large numpy arrays that can be recomputed)
    results_to_save = {
        k: v for k, v in results.items()
        if k not in ("acts", "pca_global")   # skip large arrays
    }
    with open(out_dir / "analysis_results.pkl", "wb") as f:
        pickle.dump(results_to_save, f)
    print(f"  Results saved → {out_dir/'analysis_results.pkl'}")

    # Also save a compact JSON summary
    summary = {
        "eff_rank_global": int(results["eff_rank_global"]),
        "r2_joint_cv_mean": float(results["reg_joint"]["r2_cv"].mean()),
        "r2_joint_cv_std":  float(results["reg_joint"]["r2_cv"].std()),
        "r2_source_cv_mean": float(results["reg_source"]["r2_cv"].mean()),
        "r2_hidden_cv_mean": float(results["reg_hidden"]["r2_cv"].mean()),
        "eff_rank_by_pos_min": int(results["eff_rank_by_pos"].min()),
        "eff_rank_by_pos_max": int(results["eff_rank_by_pos"].max()),
        "r2_source_at_pos_0": float(results["r2_source_by_pos"][0]),
        "r2_source_at_pos_last": float(results["r2_source_by_pos"][-1]),
        "overlap_matrix": results["overlap_matrix"].tolist(),
        "overlap_labels": results["overlap_labels"],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved → {out_dir/'summary.json'}")

    # ---- Phase 3: Figures ----
    if not args.skip_figures:
        print("\nPhase 3: Figures")
        make_all_figures(history, results, cfg)

    print("\nExperiment complete!")
    print_summary(summary)


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Global effective rank:    {summary['eff_rank_global']}")
    print(f"  Joint belief R² (CV):     {summary['r2_joint_cv_mean']:.3f} ± {summary['r2_joint_cv_std']:.3f}")
    print(f"  Source weight R² (CV):    {summary['r2_source_cv_mean']:.3f}")
    print(f"  Hidden belief R² (CV):    {summary['r2_hidden_cv_mean']:.3f}")
    print(f"  Eff rank range (context): {summary['eff_rank_by_pos_min']}–{summary['eff_rank_by_pos_max']}")
    print(f"  Source decodability:      pos=0: {summary['r2_source_at_pos_0']:.3f}  "
          f"pos=last: {summary['r2_source_at_pos_last']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
