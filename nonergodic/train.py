"""Training loop for non-ergodic Mess3 transformer.

Uses TransformerLens HookedTransformer for easy activation extraction.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import transformer_lens
from transformer_lens import HookedTransformer, HookedTransformerConfig

from nonergodic.config import (
    ExperimentConfig,
    get_default_config,
    BOS_TOKEN,
    VOCAB_SIZE,
)
from nonergodic.data import generate_batch


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(cfg: ExperimentConfig) -> HookedTransformer:
    """Instantiate a HookedTransformer from the model config."""
    mc = cfg.model
    hooked_cfg = HookedTransformerConfig(
        d_model=mc.d_model,
        d_head=mc.d_head,
        n_heads=mc.n_heads,
        n_layers=mc.n_layers,
        d_mlp=mc.d_mlp,
        n_ctx=mc.n_ctx,
        d_vocab=mc.vocab_size,
        act_fn=mc.act_fn,
        normalization_type=mc.normalization_type,
        attn_only=False,
    )
    model = HookedTransformer(hooked_cfg)
    model = model.to(cfg.device)
    return model


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_ntp_loss(
    model: HookedTransformer,
    tokens: torch.Tensor,    # (B, T+1) including BOS
) -> torch.Tensor:
    """Cross-entropy next-token-prediction loss.

    Input  tokens[:, :-1] -> predict tokens[:, 1:]
    """
    logits = model(tokens)                   # (B, T+1, V)
    # shift: predict token t+1 from position t
    logits_shifted = logits[:, :-1, :]      # (B, T, V)
    targets = tokens[:, 1:]                  # (B, T)
    loss = F.cross_entropy(
        logits_shifted.reshape(-1, logits_shifted.size(-1)),
        targets.reshape(-1),
    )
    return loss


# ---------------------------------------------------------------------------
# LR scheduler: linear warmup then cosine decay
# ---------------------------------------------------------------------------

def get_scheduler(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    cfg: ExperimentConfig | None = None,
    model: HookedTransformer | None = None,
    checkpoint_dir: str | None = None,
) -> tuple[HookedTransformer, dict]:
    """Train the model and return (model, history dict).

    Parameters
    ----------
    cfg            : ExperimentConfig (defaults to get_default_config())
    model          : pre-built model (constructed if None)
    checkpoint_dir : where to save .pt checkpoints (defaults to cfg.out_dir)

    Returns
    -------
    model   : trained HookedTransformer
    history : dict with keys "steps", "losses", "lrs"
    """
    if cfg is None:
        cfg = get_default_config()

    # Resolve device
    if cfg.device == "cpu":
        if torch.cuda.is_available():
            cfg.device = "cuda"
            print("GPU detected — switching to CUDA.")
        elif torch.backends.mps.is_available():
            cfg.device = "mps"
            print("Apple Silicon MPS detected — switching to MPS.")
    print(f"Using device: {cfg.device}")

    if model is None:
        model = build_model(cfg)
    model.train()

    out_dir = Path(checkpoint_dir or cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = get_scheduler(
        optimizer,
        warmup_steps=cfg.train.warmup_steps,
        total_steps=cfg.data.n_train_batches,
    )

    rng = np.random.default_rng(cfg.data.seed)

    history: dict[str, list] = {"steps": [], "losses": [], "lrs": []}
    t0 = time.time()

    for step in range(cfg.data.n_train_batches):
        # ---- generate batch ----
        batch = generate_batch(cfg.data.seq_len, cfg.data.batch_size, rng)
        tokens_np = batch["tokens"]                    # (B, seq_len+1)
        tokens = torch.tensor(tokens_np, dtype=torch.long, device=cfg.device)

        # ---- forward / backward ----
        optimizer.zero_grad()
        loss = compute_ntp_loss(model, tokens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()

        # ---- logging ----
        if step % cfg.train.log_every == 0 or step == cfg.data.n_train_batches - 1:
            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            print(
                f"step {step:6d}/{cfg.data.n_train_batches} | "
                f"loss {loss_val:.4f} | lr {lr_now:.2e} | {elapsed:.1f}s"
            )
            history["steps"].append(step)
            history["losses"].append(loss_val)
            history["lrs"].append(lr_now)

        # ---- checkpoint ----
        if step > 0 and step % cfg.train.checkpoint_every == 0:
            ckpt_path = out_dir / f"model_step{step}.pt"
            save_checkpoint(model, optimizer, step, loss_val, ckpt_path)
            print(f"  Saved checkpoint → {ckpt_path}")

    # Final checkpoint
    final_path = out_dir / "model_final.pt"
    save_checkpoint(model, optimizer, cfg.data.n_train_batches, loss_val, final_path)
    print(f"Training complete. Final model → {final_path}")

    model.eval()
    return model, history


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: HookedTransformer,
    optimizer,
    step: int,
    loss: float,
    path: Path,
) -> None:
    torch.save(
        {
            "step": step,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_cfg": model.cfg,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str = "cpu") -> HookedTransformer:
    """Load a model from a checkpoint file."""
    ckpt = torch.load(path, map_location=device)
    model = HookedTransformer(ckpt["model_cfg"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def extract_residual_stream(
    model: HookedTransformer,
    tokens: torch.Tensor,       # (B, T+1)
    layer: int | None = None,   # default: last layer
) -> np.ndarray:
    """Extract residual stream activations after `layer`.

    Returns
    -------
    acts : float array (B, T+1, d_model)
    """
    if layer is None:
        layer = model.cfg.n_layers - 1
    hook_name = f"blocks.{layer}.hook_resid_post"

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)

    acts = cache[hook_name]          # (B, T+1, d_model)
    return acts.cpu().numpy()


def extract_all_layers(
    model: HookedTransformer,
    tokens: torch.Tensor,
) -> dict[int, np.ndarray]:
    """Extract residual stream from ALL layers."""
    hook_names = [f"blocks.{l}.hook_resid_post" for l in range(model.cfg.n_layers)]

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_names)

    return {
        l: cache[f"blocks.{l}.hook_resid_post"].cpu().numpy()
        for l in range(model.cfg.n_layers)
    }
