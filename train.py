"""Train a small transformer on the non-ergodic Mess3 dataset and save the model.

Each training sequence is generated entirely by one randomly-chosen Mess3 source.

Usage:
    python train.py                        # default 5000 steps, saves to model.pt
    python train.py --n_steps 500          # quick debug run
    python train.py --out model_final.pt   # custom output path
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from transformer_lens import HookedTransformer, HookedTransformerConfig

from fwh_core.generative_processes.builder import (
    build_nonergodic_hidden_markov_model,
    build_transition_matrices,
)
from fwh_core.generative_processes.transition_matrices import (
    HMM_MATRIX_FUNCTIONS,
    get_stationary_state,
)
from fwh_core.generative_processes.torch_generator import generate_data_batch

# ---------------------------------------------------------------------------
# Experiment constants — shared with analysis scripts
# ---------------------------------------------------------------------------
SOURCES = [
    {"name": "mess3", "params": {"a": 0.95, "x": 0.05}},   # Source 0: highly structured
    {"name": "mess3", "params": {"a": 0.60, "x": 0.15}},   # Source 1: moderate
    {"name": "mess3", "params": {"a": 0.30, "x": 0.25}},   # Source 2: noisy
]
K = len(SOURCES)
N_STATES_PER_SOURCE = 3   # Mess3 has 3 hidden states
BOS_TOKEN = 3
VOCAB_SIZE = 4             # tokens {0,1,2} + BOS=3
SEQ_LEN = 16
OPTIMAL_LOSS = 0.89        # theoretical entropy-rate lower bound (nats)


def build_hmm():
    """Build the composite non-ergodic HMM and per-source initial states."""
    hmm = build_nonergodic_hidden_markov_model(
        process_names=[s["name"] for s in SOURCES],
        process_params=[s["params"] for s in SOURCES],
        process_weights=[1.0, 1.0, 1.0],
    )

    # One initial belief state per source: stationary dist in source k's block only.
    source_starts = []
    for k, source in enumerate(SOURCES):
        T_k = build_transition_matrices(HMM_MATRIX_FUNCTIONS, source["name"], source["params"])
        pi_k = get_stationary_state(T_k.sum(axis=0).T)
        s = jnp.zeros(K * N_STATES_PER_SOURCE)
        s = s.at[k * N_STATES_PER_SOURCE:(k + 1) * N_STATES_PER_SOURCE].set(pi_k)
        source_starts.append(s)

    return hmm, jnp.stack(source_starts)   # hmm, (K, 9)


def build_model(seq_len: int, device: str) -> HookedTransformer:
    """Build a small 2-layer HookedTransformer."""
    cfg = HookedTransformerConfig(
        d_model=128,
        d_head=32,
        n_heads=4,
        n_layers=2,
        d_mlp=512,
        n_ctx=seq_len + 1,
        d_vocab=VOCAB_SIZE,
        act_fn="gelu",
        normalization_type="LN",
        attn_only=False,
    )
    return HookedTransformer(cfg).to(device)


def train(n_steps: int, batch_size: int, out_path: Path, seed: int = 42):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    hmm, per_source_initial_states = build_hmm()
    model = build_model(SEQ_LEN, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {n_steps} steps | batch={batch_size} | seq_len={SEQ_LEN}")
    print(f"Theoretical optimal loss: ~{OPTIMAL_LOSS} nats\n")

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    history = []

    for step in range(n_steps):
        key, subkey = jax.random.split(key)

        source_ids = rng.integers(0, K, size=batch_size)
        gen_states = per_source_initial_states[source_ids]

        gen_states, inputs, labels = generate_data_batch(
            gen_states, hmm, batch_size, SEQ_LEN, subkey,
            bos_token=BOS_TOKEN, device=device,
        )

        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB_SIZE),
            labels[:, :-1].reshape(-1).long(),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
        history.append(loss_val)

        if step % 500 == 0 or step == n_steps - 1:
            print(f"  step {step:5d}/{n_steps} | loss {loss_val:.4f}")

    # Save model + config so analysis scripts can reload without redefining architecture
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": model.cfg,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "seq_len": SEQ_LEN,
            "sources": SOURCES,
            "final_loss": history[-1],
        },
        out_path,
    )
    print(f"\nSaved model → {out_path}")
    print(f"Final loss: {history[-1]:.4f}  (optimal: ~{OPTIMAL_LOSS})")

    # Also save loss history as JSON for plotting
    history_path = out_path.with_suffix(".json")
    with open(history_path, "w") as f:
        json.dump({"loss": history}, f)
    print(f"Saved loss history → {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out", type=str, default="model.pt")
    parser.add_argument("--seed", type=int, default=42)

    # parse_known_args ignores unrecognised args (e.g. Jupyter kernel flags)
    args, _ = parser.parse_known_args()

    train(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        out_path=Path(args.out),
        seed=args.seed,
    )
