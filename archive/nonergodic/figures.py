"""Figure generation for the non-ergodic Mess3 experiment.

All figures are saved to cfg.out_dir/figures/.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D     # noqa: F401 (needed for 3D projection)

from nonergodic.config import ExperimentConfig, K, N_OBS, MESS3_PARAMS, get_default_config

# Color palette: one colour per source
SOURCE_COLORS = ["#E64B35", "#4DBBD5", "#00A087"]   # red, teal, green
SOURCE_NAMES  = [f"Source {k}\n(a={MESS3_PARAMS[k]['a']}, x={MESS3_PARAMS[k]['x']})"
                 for k in range(K)]


def save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 1: Training loss curve
# ---------------------------------------------------------------------------

def fig_training_loss(history: dict, out_dir: Path) -> None:
    steps = history["steps"]
    losses = history["losses"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, losses, linewidth=1.5, color="#333333")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss (nats)")
    ax.set_title("Training Loss — Non-ergodic Mess3 (K=3)")
    ax.grid(True, alpha=0.3)

    # Reference: random-guess loss = log(vocab_size-1) since model never predicts BOS
    # After BOS the distribution over {0,1,2} matters; uniform = log(3)
    ax.axhline(np.log(3), color="gray", linestyle="--", alpha=0.6, label="log(3) uniform")
    ax.legend()
    save_fig(fig, out_dir / "fig1_training_loss.png")


# ---------------------------------------------------------------------------
# Figure 2: Residual stream PCA (scatter by source)
# ---------------------------------------------------------------------------

def fig_pca_scatter(results: dict, out_dir: Path) -> None:
    acts = results["acts"]                  # (N, T+1, d_model)
    source_ids = results["source_ids"]      # (N,)

    # Use last context position (most structured)
    X_last = acts[:, -1, :].astype(np.float64)
    X_last -= X_last.mean(axis=0)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_last)

    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.4)

    pairs = [(0, 1), (0, 2), (1, 2)]
    for ax_idx, (dim_a, dim_b) in enumerate(pairs):
        ax = fig.add_subplot(gs[ax_idx])
        for k in range(K):
            mask = source_ids == k
            ax.scatter(
                X_pca[mask, dim_a], X_pca[mask, dim_b],
                color=SOURCE_COLORS[k], alpha=0.4, s=8, label=f"Source {k}"
            )
        ax.set_xlabel(f"PC {dim_a+1} ({pca.explained_variance_ratio_[dim_a]*100:.1f}%)")
        ax.set_ylabel(f"PC {dim_b+1} ({pca.explained_variance_ratio_[dim_b]*100:.1f}%)")
        ax.set_title(f"PC{dim_a+1} vs PC{dim_b+1}")
        if ax_idx == 0:
            ax.legend(fontsize=8, markerscale=2)
    fig.suptitle("Residual Stream PCA at Final Context Position", fontsize=12)
    save_fig(fig, out_dir / "fig2_pca_scatter.png")


# ---------------------------------------------------------------------------
# Figure 3: CEV curve
# ---------------------------------------------------------------------------

def fig_cev_curve(results: dict, cfg: ExperimentConfig, out_dir: Path) -> None:
    cev = results["cev_global"]
    eff_rank = results["eff_rank_global"]

    dims = np.arange(1, len(cev) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(dims, cev, "o-", markersize=4, linewidth=1.5, color="#333333")
    ax.axhline(cfg.analysis.cev_threshold, color="firebrick", linestyle="--",
               alpha=0.7, label=f"{cfg.analysis.cev_threshold*100:.0f}% threshold")
    ax.axvline(eff_rank, color="steelblue", linestyle="--",
               alpha=0.7, label=f"Effective rank = {eff_rank}")
    # Mark theoretical predictions
    ax.axvline(2, color="darkorange", linestyle=":", alpha=0.7, label="2D (single triangle)")
    ax.axvline(8, color="purple", linestyle=":", alpha=0.7, label="8D (full joint)")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("CEV — Global (all positions pooled)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(30, len(cev)) + 1)
    save_fig(fig, out_dir / "fig3_cev_curve.png")


# ---------------------------------------------------------------------------
# Figure 4: Belief regression summary
# ---------------------------------------------------------------------------

def fig_belief_regression(results: dict, out_dir: Path) -> None:
    reg_j = results["reg_joint"]
    reg_s = results["reg_source"]
    reg_h = results["reg_hidden"]

    labels = ["Joint belief\n(3K dims)", "Source weights\n(K dims)", "Hidden beliefs\n(K×3 dims)"]
    means = [reg_j["r2_cv"].mean(), reg_s["r2_cv"].mean(), reg_h["r2_cv"].mean()]
    stds  = [reg_j["r2_cv"].std(),  reg_s["r2_cv"].std(),  reg_h["r2_cv"].std()]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=["#4C72B0", "#DD8452", "#55A868"],
                  edgecolor="black", linewidth=0.8, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Cross-validated R²")
    ax.set_ylim(0, 1.05)
    ax.set_title("Linear Decodability of Belief States from Residual Stream")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.02,
                f"{mean:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, out_dir / "fig4_belief_regression.png")


# ---------------------------------------------------------------------------
# Figure 5: Per-dimension R² for joint belief regression
# ---------------------------------------------------------------------------

def fig_per_dim_r2(results: dict, out_dir: Path) -> None:
    r2_dims = results["reg_joint"]["r2_per_dim"]   # (3K,)

    dim_labels = []
    for k in range(K):
        for s in range(N_OBS):
            dim_labels.append(f"k={k},s={s}")

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = []
    for k in range(K):
        colors.extend([SOURCE_COLORS[k]] * N_OBS)
    bars = ax.bar(np.arange(len(dim_labels)), r2_dims, color=colors, edgecolor="black",
                  linewidth=0.6, alpha=0.85)
    ax.set_xticks(np.arange(len(dim_labels)))
    ax.set_xticklabels(dim_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("R²")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-dimension R² for Joint Belief Regression")
    patches = [mpatches.Patch(color=SOURCE_COLORS[k], label=f"Source {k}") for k in range(K)]
    ax.legend(handles=patches, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, out_dir / "fig5_per_dim_r2.png")


# ---------------------------------------------------------------------------
# Figure 6: Subspace overlap matrix
# ---------------------------------------------------------------------------

def fig_subspace_overlap(results: dict, out_dir: Path) -> None:
    mat = results["overlap_matrix"]
    labels = results["overlap_labels"]

    # Prettify labels
    pretty = []
    for l in labels:
        if l == "source_subspace":
            pretty.append("Source\nweights")
        else:
            k = int(l.split("_")[-1])
            pretty.append(f"Hidden\nSource {k}")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn_r")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(pretty, fontsize=9)
    ax.set_yticklabels(pretty, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=10, color="black")
    ax.set_title("Pairwise Subspace Overlap (0=orthogonal, 1=identical)")
    save_fig(fig, out_dir / "fig6_subspace_overlap.png")


# ---------------------------------------------------------------------------
# Figure 7 & 8: Context-position dimensionality (Task 4)
# ---------------------------------------------------------------------------

def fig_context_position_geometry(results: dict, out_dir: Path) -> None:
    eff_ranks = results["eff_rank_by_pos"]         # (T+1,)
    r2_source = results["r2_source_by_pos"]        # (T+1,)
    r2_hidden = results["r2_hidden_by_pos"]        # (T+1,)
    source_weights = results["source_weights"]     # (N, T+1, K)

    T_plus_1 = len(eff_ranks)
    positions = np.arange(T_plus_1)  # 0=BOS, 1=after 1 obs, ...

    # Mean source entropy per position
    eps = 1e-300
    sw = source_weights
    src_entropy = -np.sum(sw * np.log(sw + eps), axis=2).mean(axis=0)   # (T+1,)

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    # ---- Effective rank ----
    ax = axes[0]
    ax.plot(positions, eff_ranks, "o-", ms=4, lw=1.5, color="#333333")
    ax.axhline(2, color="darkorange", ls="--", alpha=0.7, label="2D prediction (converged)")
    ax.axhline(8, color="purple", ls="--", alpha=0.7, label="8D prediction (early)")
    ax.set_ylabel("Effective rank (95% CEV)")
    ax.set_title("Effective Dimensionality of Residual Stream vs Context Position")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(eff_ranks) + 2)

    # ---- Source decodability ----
    ax = axes[1]
    ax.plot(positions, r2_source, "s-", ms=4, lw=1.5, color="#E64B35",
            label="Source weights R²")
    ax.plot(positions, r2_hidden, "^-", ms=4, lw=1.5, color="#4DBBD5",
            label="Hidden beliefs R²")
    ax.set_ylabel("Linear R²")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Belief Decodability vs Context Position")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Source entropy ----
    ax = axes[2]
    ax.plot(positions, src_entropy, "D-", ms=4, lw=1.5, color="#00A087")
    ax.axhline(np.log(K), color="gray", ls="--", alpha=0.6, label=f"Max entropy = log({K})")
    ax.set_xlabel("Context position t  (0 = after BOS)")
    ax.set_ylabel("Mean source entropy H[k | obs₁:t]")
    ax.set_title("Source Uncertainty vs Context Position")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, out_dir / "fig7_context_position_geometry.png")


def fig_pca_per_position(results: dict, out_dir: Path, n_positions: int = 6) -> None:
    """PCA scatter at selected context positions to visualise cluster formation."""
    acts = results["acts"]                 # (N, T+1, d_model)
    source_ids = results["source_ids"]
    T_plus_1 = acts.shape[1]

    positions_to_plot = np.linspace(0, T_plus_1 - 1, n_positions, dtype=int)

    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, pos in zip(axes.flat, positions_to_plot):
        X = acts[:, pos, :].astype(np.float64)
        X -= X.mean(axis=0)
        pca = PCA(n_components=2)
        X2 = pca.fit_transform(X)
        for k in range(K):
            mask = source_ids == k
            ax.scatter(X2[mask, 0], X2[mask, 1],
                       color=SOURCE_COLORS[k], alpha=0.4, s=6, label=f"S{k}")
        ax.set_title(f"Position t={pos}", fontsize=10)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)", fontsize=8)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)", fontsize=8)
        ax.tick_params(labelsize=7)

    handles = [mpatches.Patch(color=SOURCE_COLORS[k], label=SOURCE_NAMES[k]) for k in range(K)]
    fig.legend(handles=handles, loc="lower center", ncol=K, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Residual Stream PCA at Selected Context Positions", fontsize=12)
    fig.tight_layout()
    save_fig(fig, out_dir / "fig8_pca_per_position.png")


# ---------------------------------------------------------------------------
# Figure 9: Belief simplex (ground truth vs decoded)
# ---------------------------------------------------------------------------

def _plot_simplex_scatter(ax, beliefs_2d, colors, alpha=0.3, s=5):
    """Plot belief states in barycentric coordinates (projected to 2D)."""
    # Barycentric to Cartesian for equilateral triangle
    B = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    coords = beliefs_2d @ B   # (N, 2)
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=alpha, s=s)
    # Draw triangle
    tri = plt.Polygon(B, fill=False, edgecolor="black", linewidth=1.5)
    ax.add_patch(tri)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_belief_simplex(results: dict, out_dir: Path) -> None:
    """Plot true hidden belief states on the 2-simplex for each source."""
    from sklearn.linear_model import Ridge

    hidden_beliefs = results["hidden_beliefs"]    # (N, T+1, K, 3)
    source_ids = results["source_ids"]
    source_weights = results["source_weights"]    # (N, T+1, K)

    # Use last position
    hb_last = hidden_beliefs[:, -1, :, :]         # (N, K, 3)
    sw_last = source_weights[:, -1, :]             # (N, K)

    # Re-run regression specifically for last position
    acts = results["acts"][:, -1, :].astype(np.float64)   # (N, d_model)
    acts_c = acts - acts.mean(axis=0)
    joint_last = (sw_last[:, :, None] * hb_last).reshape(len(acts_c), K * N_OBS)
    reg = Ridge(alpha=1e-3)
    reg.fit(acts_c, joint_last)
    decoded_flat = reg.predict(acts_c)             # (N, 3K)
    decoded = decoded_flat.reshape(-1, K, N_OBS)
    # Normalize decoded to simplex
    decoded = np.abs(decoded)
    decoded = decoded / (decoded.sum(axis=2, keepdims=True) + 1e-12)

    fig, axes = plt.subplots(2, K, figsize=(5 * K, 9))

    for k in range(K):
        # True beliefs (colored by sequence source, opacity by weight)
        ax = axes[0, k]
        # Only show sequences where source k is dominant (w_k > 0.5)
        mask = sw_last[:, k] > 0.5
        if mask.sum() > 0:
            _plot_simplex_scatter(ax, hb_last[mask, k, :],
                                  colors=SOURCE_COLORS[k], alpha=0.5, s=8)
        ax.set_title(f"True hidden beliefs — Source {k}", fontsize=10)

        # Decoded beliefs
        ax = axes[1, k]
        if mask.sum() > 0:
            _plot_simplex_scatter(ax, decoded[mask, k, :],
                                  colors=SOURCE_COLORS[k], alpha=0.5, s=8)
        ax.set_title(f"Decoded hidden beliefs — Source {k}", fontsize=10)

    axes[0, 0].set_ylabel("Ground truth", fontsize=11)
    axes[1, 0].set_ylabel("Linear readout", fontsize=11)
    fig.suptitle("Belief State Geometry on the 2-Simplex (final context position)", fontsize=12)
    save_fig(fig, out_dir / "fig9_belief_simplex.png")


# ---------------------------------------------------------------------------
# All-in-one entry point
# ---------------------------------------------------------------------------

def make_all_figures(
    history: dict,
    results: dict,
    cfg: ExperimentConfig | None = None,
) -> None:
    if cfg is None:
        cfg = get_default_config()
    out_dir = Path(cfg.out_dir) / "figures"
    print(f"\nGenerating figures → {out_dir}")

    fig_training_loss(history, out_dir)
    fig_pca_scatter(results, out_dir)
    fig_cev_curve(results, cfg, out_dir)
    fig_belief_regression(results, out_dir)
    fig_per_dim_r2(results, out_dir)
    fig_subspace_overlap(results, out_dir)
    fig_context_position_geometry(results, out_dir)
    fig_pca_per_position(results, out_dir)
    fig_belief_simplex(results, out_dir)

    print("All figures saved.")
