"""Task 3: Residual stream geometry analysis.

Loads the trained model, generates an analysis set with ground-truth belief states,
extracts residual stream activations, and runs three analyses:

  1. PCA scatter — are there 3 clusters, one per source?
  2. Effective rank — how many dimensions does the model actually use?
  3. Linear decodability — can we read source weights and hidden beliefs from activations?

Usage (script):
    python analyze.py                   # loads model.pt, saves figures/
    python analyze.py --model model.pt  # explicit path

Usage (interactive): call each section function directly after running the setup cells.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from transformer_lens import HookedTransformer

from fwh_core.generative_processes.builder import (
    build_nonergodic_hidden_markov_model,
    build_transition_matrices,
)
from fwh_core.generative_processes.transition_matrices import (
    HMM_MATRIX_FUNCTIONS,
    get_stationary_state,
)
from fwh_core.generative_processes.torch_generator import generate_data_batch_with_full_history

# ---------------------------------------------------------------------------
# Constants — must match train.py exactly
# ---------------------------------------------------------------------------
SOURCES = [
    {"name": "mess3", "params": {"a": 0.95, "x": 0.05}},
    {"name": "mess3", "params": {"a": 0.60, "x": 0.15}},
    {"name": "mess3", "params": {"a": 0.30, "x": 0.25}},
]
K = len(SOURCES)
N_STATES_PER_SOURCE = 3
BOS_TOKEN = 3
VOCAB_SIZE = 4
SEQ_LEN = 16
N_ANALYSIS = 512          # sequences for analysis
SOURCE_COLORS = ["tab:blue", "tab:orange", "tab:green"]
SOURCE_LABELS = ["Source 0 (structured)", "Source 1 (moderate)", "Source 2 (noisy)"]


# ---------------------------------------------------------------------------
# Setup: build HMM + per-source initial states
# ---------------------------------------------------------------------------
def build_hmm():
    hmm = build_nonergodic_hidden_markov_model(
        process_names=[s["name"] for s in SOURCES],
        process_params=[s["params"] for s in SOURCES],
        process_weights=[1.0, 1.0, 1.0],
    )
    source_starts = []
    for k, source in enumerate(SOURCES):
        T_k = build_transition_matrices(HMM_MATRIX_FUNCTIONS, source["name"], source["params"])
        pi_k = get_stationary_state(T_k.sum(axis=0).T)
        s = jnp.zeros(K * N_STATES_PER_SOURCE).at[k * N_STATES_PER_SOURCE:(k + 1) * N_STATES_PER_SOURCE].set(pi_k)
        source_starts.append(s)
    return hmm, jnp.stack(source_starts)


def decompose_belief(belief_states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose 9-dim composite belief into source weights and hidden beliefs.

    Args:
        belief_states: (B, T, 9) composite belief over all (source, hidden_state) pairs

    Returns:
        source_weights: (B, T, K)    — P(source=k | obs_{1:t})
        hidden_beliefs: (B, T, K, 3) — P(hidden | obs_{1:t}, source=k)
    """
    B, T, _ = belief_states.shape
    blocks = belief_states.reshape(B, T, K, N_STATES_PER_SOURCE)   # (B, T, K, 3)
    source_weights = blocks.sum(axis=-1)                            # (B, T, K)
    # Normalize within each source block; avoid div-by-zero for inactive sources
    denom = source_weights[..., np.newaxis] + 1e-300
    hidden_beliefs = blocks / denom                                  # (B, T, K, 3)
    return source_weights, hidden_beliefs


# ---------------------------------------------------------------------------
# Data generation with ground-truth belief states
# ---------------------------------------------------------------------------
def generate_analysis_data(hmm, per_source_initial_states, seed=0):
    """Generate N_ANALYSIS sequences with known sources and ground-truth beliefs."""
    rng = np.random.default_rng(seed)
    source_ids = rng.integers(0, K, size=N_ANALYSIS)
    gen_states = per_source_initial_states[source_ids]

    key = jax.random.PRNGKey(seed)
    result = generate_data_batch_with_full_history(
        gen_states, hmm, N_ANALYSIS, SEQ_LEN, key, bos_token=BOS_TOKEN,
    )

    belief_states = np.array(result["belief_states"])    # (B, T, 9)
    tokens = np.array(result["inputs"])                  # (B, T) with BOS at pos 0

    source_weights, hidden_beliefs = decompose_belief(belief_states)

    return {
        "tokens": tokens,
        "source_ids": source_ids,
        "belief_states": belief_states,
        "source_weights": source_weights,   # (B, T, K)
        "hidden_beliefs": hidden_beliefs,   # (B, T, K, 3)
    }


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------
def extract_activations(model, tokens_np: np.ndarray, layer: int = -1) -> np.ndarray:
    """Extract residual stream after `layer` (-1 = last layer).

    Returns:
        acts: (B, T, d_model)
    """
    n_layers = model.cfg.n_layers
    layer_idx = n_layers - 1 if layer == -1 else layer
    hook_name = f"blocks.{layer_idx}.hook_resid_post"

    tokens = torch.tensor(tokens_np, dtype=torch.long)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)

    return cache[hook_name].cpu().numpy()   # (B, T, d_model)


# ---------------------------------------------------------------------------
# Analysis 1: PCA scatter colored by source (final token position)
# ---------------------------------------------------------------------------
def analyze_pca(acts: np.ndarray, source_ids: np.ndarray, out_dir: Path):
    """PCA on activations at the final token position, colored by source."""
    from sklearn.decomposition import PCA

    # Use activations at the last token position
    X = acts[:, -1, :]   # (B, d_model)

    pca = PCA(n_components=10)
    Z = pca.fit_transform(X)   # (B, 10)

    explained = pca.explained_variance_ratio_
    effective_rank = int(np.searchsorted(np.cumsum(explained), 0.95)) + 1
    print(f"[PCA] Effective rank (95% variance, last position): {effective_rank}")
    print(f"[PCA] Variance explained by first 3 PCs: {explained[:3].sum():.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: PC1 vs PC2 scatter
    ax = axes[0]
    for k in range(K):
        mask = source_ids == k
        ax.scatter(Z[mask, 0], Z[mask, 1], s=10, alpha=0.5,
                   color=SOURCE_COLORS[k], label=SOURCE_LABELS[k])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Residual stream PCA (final position)")
    ax.legend(markerscale=2)

    # Right: cumulative explained variance
    ax = axes[1]
    ax.plot(np.cumsum(explained) * 100, marker="o", markersize=4)
    ax.axhline(95, color="red", linestyle="--", label="95% threshold")
    ax.axvline(effective_rank - 1, color="gray", linestyle="--",
               label=f"eff. rank = {effective_rank}")
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Cumulative variance explained (%)")
    ax.set_title("Cumulative explained variance")
    ax.legend()

    plt.tight_layout()
    path = out_dir / "fig1_pca.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved → {path}")

    return effective_rank


# ---------------------------------------------------------------------------
# Analysis 1b: 3D PCA scatter with 2D projections
# ---------------------------------------------------------------------------
def analyze_pca_3d(acts: np.ndarray, source_ids: np.ndarray, out_dir: Path):
    """3D scatter of PC1/PC2/PC3 with projections onto each pair of axes."""
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    X = acts[:, -1, :]
    pca = PCA(n_components=3)
    Z = pca.fit_transform(X)   # (B, 3)
    explained = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(16, 14))

    # ---- 3D scatter (top-left, large) ----
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    for k in range(K):
        mask = source_ids == k
        ax3d.scatter(Z[mask, 0], Z[mask, 1], Z[mask, 2],
                     s=12, alpha=0.5, color=SOURCE_COLORS[k], label=SOURCE_LABELS[k])
    ax3d.set_xlabel(f"PC1 ({explained[0]:.1%})", labelpad=8)
    ax3d.set_ylabel(f"PC2 ({explained[1]:.1%})", labelpad=8)
    ax3d.set_zlabel(f"PC3 ({explained[2]:.1%})", labelpad=8)
    ax3d.set_title("3D PCA (final position)")
    ax3d.legend(markerscale=2, loc="upper left", fontsize=8)

    # ---- 2D projections ----
    projection_pairs = [
        (0, 1, "PC1", "PC2", fig.add_subplot(2, 2, 2)),
        (0, 2, "PC1", "PC3", fig.add_subplot(2, 2, 3)),
        (1, 2, "PC2", "PC3", fig.add_subplot(2, 2, 4)),
    ]
    for xi, yi, xlabel, ylabel, ax in projection_pairs:
        for k in range(K):
            mask = source_ids == k
            ax.scatter(Z[mask, xi], Z[mask, yi], s=10, alpha=0.5,
                       color=SOURCE_COLORS[k], label=SOURCE_LABELS[k])
        ax.set_xlabel(f"{xlabel} ({explained[xi]:.1%})")
        ax.set_ylabel(f"{ylabel} ({explained[yi]:.1%})")
        ax.set_title(f"{xlabel} vs {ylabel}")
        ax.legend(markerscale=2, fontsize=7)

    plt.suptitle("Residual stream PCA — 3D view and 2D projections", fontsize=13, y=1.01)
    plt.tight_layout()
    path = out_dir / "fig1b_pca_3d.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Analysis 2: Effective rank across all token positions
# ---------------------------------------------------------------------------
def analyze_effective_rank_by_position(acts: np.ndarray, out_dir: Path):
    """Compute effective rank at each token position."""
    from sklearn.decomposition import PCA

    T = acts.shape[1]
    eff_ranks = []

    for t in range(T):
        X = acts[:, t, :]
        pca = PCA(n_components=min(X.shape))
        pca.fit(X)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        eff_ranks.append(int(np.searchsorted(cumvar, 0.95)) + 1)

    print(f"[Eff. rank] t=0 (BOS): {eff_ranks[0]}  |  t=-1 (last): {eff_ranks[-1]}")
    print(f"[Eff. rank] Prediction: decreases from ~8 → ~2")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(T), eff_ranks, marker="o", markersize=4)
    ax.axhline(8, color="gray", linestyle="--", alpha=0.5, label="predicted start (~8)")
    ax.axhline(2, color="gray", linestyle=":",  alpha=0.5, label="predicted end (~2)")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Effective rank (95% variance)")
    ax.set_title("Effective dimensionality vs context position")
    ax.legend()

    plt.tight_layout()
    path = out_dir / "fig2_eff_rank_by_position.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved → {path}")

    return eff_ranks


# ---------------------------------------------------------------------------
# Analysis 3: Linear decodability of source weights and hidden beliefs
# ---------------------------------------------------------------------------
def analyze_decodability(acts: np.ndarray, source_ids: np.ndarray,
                         source_weights: np.ndarray,
                         hidden_beliefs: np.ndarray, out_dir: Path):
    """Ridge regression R² from activations → belief state components.

    Reports R² at the final token position for:
      - Source weights (K=3 dims)
      - Hidden beliefs for each source (3 dims each)
    """
    X = acts[:, -1, :]   # (B, d_model) — final position activations

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}

    # Source weights
    y_src = source_weights[:, -1, :]   # (B, K)
    scores = cross_val_score(Ridge(alpha=1.0), X_scaled, y_src, cv=5, scoring="r2")
    results["source_weights"] = scores.mean()
    print(f"[Decodability] Source weights R²: {scores.mean():.3f} ± {scores.std():.3f}")

    # Hidden beliefs per source — only evaluate on sequences from that source,
    # since hidden beliefs are undefined (NaN) for inactive sources.
    for k in range(K):
        mask = source_ids == k
        if mask.sum() < 10:
            print(f"[Decodability] Hidden beliefs (source {k}): too few sequences, skipping")
            continue
        y_h = hidden_beliefs[mask, -1, k, :]          # (n_k, 3)
        X_k = scaler.transform(acts[mask, -1, :])     # same scaler, subset
        scores = cross_val_score(Ridge(alpha=1.0), X_k, y_h, cv=5, scoring="r2")
        results[f"hidden_source_{k}"] = scores.mean()
        print(f"[Decodability] Hidden beliefs (source {k}) R²: {scores.mean():.3f} ± {scores.std():.3f}")

    # Bar plot
    labels = ["Source\nweights"] + [f"Hidden\n(src {k})" for k in range(K)]
    values = [results["source_weights"]] + [results[f"hidden_source_{k}"] for k in range(K)]
    colors = ["gray"] + SOURCE_COLORS

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, alpha=0.8)
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.3, label="perfect R²=1")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Cross-validated R²")
    ax.set_title("Linear decodability from residual stream (final position)")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = out_dir / "fig3_decodability.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved → {path}")

    return results


# ---------------------------------------------------------------------------
# Analysis 4: Decodability by token position
# ---------------------------------------------------------------------------
def analyze_decodability_by_position(acts: np.ndarray, source_ids: np.ndarray,
                                      source_weights: np.ndarray,
                                      hidden_beliefs: np.ndarray, out_dir: Path):
    """R² from activations → belief components at each token position."""
    T = acts.shape[1]
    scaler = StandardScaler()

    src_r2 = []
    hidden_r2 = [[] for _ in range(K)]

    for t in range(T):
        X_t = scaler.fit_transform(acts[:, t, :])

        # Source weights (all sequences)
        y_src = source_weights[:, t, :]
        s = cross_val_score(Ridge(alpha=1.0), X_t, y_src, cv=5, scoring="r2")
        src_r2.append(s.mean())

        # Hidden beliefs (source-k sequences only)
        for k in range(K):
            mask = source_ids == k
            if mask.sum() < 10:
                hidden_r2[k].append(np.nan)
                continue
            y_h = hidden_beliefs[mask, t, k, :]
            X_k = scaler.transform(acts[mask, t, :])
            s = cross_val_score(Ridge(alpha=1.0), X_k, y_h, cv=5, scoring="r2")
            hidden_r2[k].append(s.mean())

    positions = list(range(T))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(positions, src_r2, color="gray", linewidth=2, label="Source weights")
    for k in range(K):
        ax.plot(positions, hidden_r2[k], color=SOURCE_COLORS[k],
                linewidth=2, label=SOURCE_LABELS[k] + " hidden")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Cross-validated R²")
    ax.set_title("Linear decodability of belief components vs context position")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = out_dir / "fig4_decodability_by_position.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_analysis(model_path: str = "model.pt", out_dir: str = "figures"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    # Load model
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = HookedTransformer(ckpt["model_cfg"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {model_path}  (trained {ckpt['n_steps']} steps, "
          f"final loss {ckpt['final_loss']:.4f})")

    # Generate analysis data
    print("\nGenerating analysis sequences...")
    hmm, per_source_initial_states = build_hmm()
    data = generate_analysis_data(hmm, per_source_initial_states)
    print(f"Source distribution: {np.bincount(data['source_ids'], minlength=K)}")

    # Extract activations (last layer, all positions)
    print("\nExtracting residual stream activations...")
    acts = extract_activations(model, data["tokens"])
    print(f"Activations shape: {acts.shape}  (sequences, positions, d_model)")

    # Run analyses
    print("\n--- Analysis 1: PCA ---")
    analyze_pca(acts, data["source_ids"], out)

    print("\n--- Analysis 1b: 3D PCA ---")
    analyze_pca_3d(acts, data["source_ids"], out)

    print("\n--- Analysis 2: Effective rank by position ---")
    analyze_effective_rank_by_position(acts, out)

    print("\n--- Analysis 3: Linear decodability ---")
    analyze_decodability(acts, data["source_ids"], data["source_weights"], data["hidden_beliefs"], out)

    print("\n--- Analysis 4: Decodability by position ---")
    analyze_decodability_by_position(acts, data["source_ids"], data["source_weights"], data["hidden_beliefs"], out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model.pt")
    parser.add_argument("--out_dir", type=str, default="figures")
    args, _ = parser.parse_known_args()
    run_analysis(model_path=args.model, out_dir=args.out_dir)
