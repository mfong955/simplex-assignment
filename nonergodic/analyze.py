"""Geometric analysis of residual stream activations.

Implements:
    1. PCA / Cumulative Explained Variance (CEV) → effective rank
    2. Linear regression from activations to belief state vectors
    3. Subspace identification via "vary-one" analysis
    4. Subspace overlap metric (principal angles)
    5. Context-position dependent dimensionality (Task 4)
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import torch

from nonergodic.config import ExperimentConfig, K, N_OBS, get_default_config
from nonergodic.data import compute_belief_states_batch, generate_batch
from nonergodic.train import extract_residual_stream


# ---------------------------------------------------------------------------
# 1. PCA / Cumulative Explained Variance
# ---------------------------------------------------------------------------

def run_pca(
    acts: np.ndarray,          # (..., d_model)  any leading shape
    n_components: int | None = None,
) -> tuple[PCA, np.ndarray]:
    """Fit PCA on the (flattened) activation matrix.

    Returns
    -------
    pca    : fitted sklearn PCA object
    X_pca  : projected activations, shape (N, n_components)
    """
    X = acts.reshape(-1, acts.shape[-1]).astype(np.float64)
    X = X - X.mean(axis=0)
    if n_components is None:
        n_components = min(X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return pca, X_pca


def effective_rank(pca: PCA, threshold: float = 0.95) -> int:
    """Number of principal components needed to explain `threshold` of variance."""
    cev = np.cumsum(pca.explained_variance_ratio_)
    n = int(np.searchsorted(cev, threshold)) + 1
    return min(n, len(cev))


def cev_curve(acts: np.ndarray, max_dims: int = 50) -> np.ndarray:
    """Return cumulative explained variance for dims 1..max_dims."""
    X = acts.reshape(-1, acts.shape[-1]).astype(np.float64)
    X = X - X.mean(axis=0)
    max_dims = min(max_dims, X.shape[0], X.shape[1])
    pca = PCA(n_components=max_dims)
    pca.fit(X)
    return np.cumsum(pca.explained_variance_ratio_)


# ---------------------------------------------------------------------------
# 2. Linear regression: activations → belief states
# ---------------------------------------------------------------------------

def belief_regression(
    acts: np.ndarray,          # (N, d_model)
    targets: np.ndarray,       # (N, D_belief)
    n_splits: int = 5,
    alpha: float = 1e-3,
) -> dict[str, np.ndarray]:
    """Cross-validated ridge regression from activations to belief vectors.

    Returns
    -------
    dict with keys:
        "r2_cv"         : (n_splits,) R² per fold
        "rmse_cv"       : (n_splits,) RMSE per fold
        "r2_per_dim"    : (D_belief,) R² for each belief dimension (in-sample)
        "predictions"   : (N, D_belief) in-sample predictions
        "coef"          : (D_belief, d_model) regression coefficients
    """
    X = acts.astype(np.float64)
    Y = targets.astype(np.float64)

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    r2_folds, rmse_folds = [], []

    for train_idx, val_idx in kf.split(X_scaled):
        reg = Ridge(alpha=alpha)
        reg.fit(X_scaled[train_idx], Y[train_idx])
        preds = reg.predict(X_scaled[val_idx])
        ss_res = ((Y[val_idx] - preds) ** 2).sum(axis=0)
        ss_tot = ((Y[val_idx] - Y[val_idx].mean(axis=0)) ** 2).sum(axis=0)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        rmse = np.sqrt(((Y[val_idx] - preds) ** 2).mean(axis=0))
        r2_folds.append(r2.mean())
        rmse_folds.append(rmse.mean())

    # In-sample fit for coefficient extraction and per-dim R²
    reg_full = Ridge(alpha=alpha)
    reg_full.fit(X_scaled, Y)
    preds_full = reg_full.predict(X_scaled)
    ss_res = ((Y - preds_full) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    r2_per_dim = 1 - ss_res / (ss_tot + 1e-12)

    return {
        "r2_cv": np.array(r2_folds),
        "rmse_cv": np.array(rmse_folds),
        "r2_per_dim": r2_per_dim,
        "predictions": preds_full,
        "coef": reg_full.coef_,        # (D_belief, d_model)
        "intercept": reg_full.intercept_,
        "scaler_X": scaler_X,
    }


# ---------------------------------------------------------------------------
# 3. Vary-one subspace identification
# ---------------------------------------------------------------------------

def identify_factor_subspace(
    acts: np.ndarray,                  # (N, d_model)
    varying_factor_values: np.ndarray, # (N, d_factor)  the factor that varies
    fixed_factor_mask: np.ndarray,     # (N,) bool: which rows have other factors fixed
    n_dims: int = 2,
) -> np.ndarray:
    """Identify the subspace in activation space that encodes one factor.

    Strategy: among samples where all OTHER factors are held approximately
    fixed (fixed_factor_mask=True), compute the top PCA directions of the
    activation variance.  These directions span the factor's subspace.

    Returns
    -------
    basis : (d_model, n_dims)  orthonormal basis of the factor subspace
    """
    X = acts[fixed_factor_mask].astype(np.float64)
    X = X - X.mean(axis=0)
    pca = PCA(n_components=n_dims)
    pca.fit(X)
    return pca.components_.T    # (d_model, n_dims)


def subspace_overlap(A: np.ndarray, B: np.ndarray) -> float:
    """Measure overlap between two subspaces spanned by column matrices A, B.

    Uses the formula:  overlap(A,B) = Tr(P_A P_B) / min(dim_A, dim_B)
    where P_X = X (X^T X)^{-1} X^T  is the projection onto col(X).

    Equivalently, computed via singular values of A^T B:
        overlap = sum(sigma_i^2) / min(dim_A, dim_B)

    Returns a scalar in [0, 1].
    """
    d_min = min(A.shape[1], B.shape[1])
    # Orthonormalize both
    A_orth, _ = np.linalg.qr(A)
    B_orth, _ = np.linalg.qr(B)
    M = A_orth[:, :A.shape[1]].T @ B_orth[:, :B.shape[1]]
    singular_values = np.linalg.svd(M, compute_uv=False)
    return float(np.sum(singular_values ** 2) / d_min)


# ---------------------------------------------------------------------------
# 4. Factor subspace extraction (full pipeline)
# ---------------------------------------------------------------------------

def extract_factor_subspaces(
    acts: np.ndarray,                  # (N, d_model)
    source_weights: np.ndarray,        # (N, K)
    hidden_beliefs: np.ndarray,        # (N, K, 3)
    n_dims: int = 2,
    source_entropy_threshold: float = 0.3,
    hidden_entropy_threshold: float = 0.3,
) -> dict[str, np.ndarray]:
    """Extract orthogonal subspaces for (a) source identity and (b) K hidden factors.

    Returns
    -------
    dict with keys:
        "source_subspace"   : (d_model, n_dims)  basis for source-weight subspace
        "hidden_subspace_k" : (d_model, n_dims)  basis for k-th hidden state subspace
    """
    K_local = source_weights.shape[1]
    N = acts.shape[0]

    # Mean-center activations
    acts_centered = acts - acts.mean(axis=0)

    # --- Source subspace ---
    # Fix hidden beliefs (low hidden entropy within at least one source),
    # let source weights vary (high source entropy samples excluded — we want
    # to see where source-uncertainty is encoded).
    # Strategy: use all samples, regress source weights, take top directions.
    reg = Ridge(alpha=1e-3)
    reg.fit(acts_centered, source_weights)
    # The row-space of the coefficient matrix spans the source subspace
    source_coef = reg.coef_    # (K, d_model)
    u, s, vt = np.linalg.svd(source_coef, full_matrices=False)
    source_basis = vt[:n_dims].T   # (d_model, n_dims)

    # --- Per-source hidden subspaces ---
    hidden_bases = {}
    for k in range(K_local):
        # Only use samples where source k is dominant (w_k > 0.8)
        dominant_mask = source_weights[:, k] > 0.8
        if dominant_mask.sum() < 10:
            # Fall back: use all samples
            dominant_mask = np.ones(N, dtype=bool)

        reg_k = Ridge(alpha=1e-3)
        reg_k.fit(acts_centered[dominant_mask], hidden_beliefs[dominant_mask, k, :])
        coef_k = reg_k.coef_       # (3, d_model)
        u_k, s_k, vt_k = np.linalg.svd(coef_k, full_matrices=False)
        hidden_bases[k] = vt_k[:n_dims].T    # (d_model, n_dims)

    result = {"source_subspace": source_basis}
    for k in range(K_local):
        result[f"hidden_subspace_{k}"] = hidden_bases[k]
    return result


def compute_subspace_overlap_matrix(
    subspaces: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Compute the pairwise subspace overlap matrix.

    Returns
    -------
    overlap_matrix : (n_subspaces, n_subspaces)
    labels         : list of subspace names in order
    """
    labels = sorted(subspaces.keys())
    n = len(labels)
    mat = np.zeros((n, n))
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            mat[i, j] = subspace_overlap(subspaces[li], subspaces[lj])
    return mat, labels


# ---------------------------------------------------------------------------
# 5. Context-position dependent geometry (Task 4)
# ---------------------------------------------------------------------------

def context_position_effective_rank(
    acts: np.ndarray,      # (N, T+1, d_model)  all context positions
    threshold: float = 0.95,
    max_dims: int = 30,
) -> np.ndarray:
    """Compute effective rank at each context position.

    Returns
    -------
    eff_ranks : (T+1,) int array  — effective rank at each position
    """
    T_plus_1 = acts.shape[1]
    eff_ranks = np.zeros(T_plus_1, dtype=int)
    for pos in range(T_plus_1):
        X = acts[:, pos, :].astype(np.float64)
        X = X - X.mean(axis=0)
        n = min(max_dims, X.shape[0], X.shape[1])
        if n < 2:
            eff_ranks[pos] = 1
            continue
        pca = PCA(n_components=n)
        pca.fit(X)
        cev = np.cumsum(pca.explained_variance_ratio_)
        eff_ranks[pos] = int(np.searchsorted(cev, threshold)) + 1
    return eff_ranks


def context_position_source_decodability(
    acts: np.ndarray,          # (N, T+1, d_model)
    source_weights: np.ndarray,  # (N, T+1, K)
) -> np.ndarray:
    """At each context position, measure how well source identity can be decoded.

    Uses the R² of a linear regression from residual stream to source weights.

    Returns
    -------
    r2_per_position : (T+1,) float array
    """
    T_plus_1 = acts.shape[1]
    r2_per_pos = np.zeros(T_plus_1)
    for pos in range(T_plus_1):
        X = acts[:, pos, :].astype(np.float64)
        X = X - X.mean(axis=0)
        Y = source_weights[:, pos, :].astype(np.float64)
        reg = Ridge(alpha=1e-3)
        reg.fit(X, Y)
        preds = reg.predict(X)
        ss_res = ((Y - preds) ** 2).sum()
        ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum()
        r2_per_pos[pos] = float(1 - ss_res / (ss_tot + 1e-12))
    return r2_per_pos


def context_position_hidden_decodability(
    acts: np.ndarray,            # (N, T+1, d_model)
    hidden_beliefs: np.ndarray,  # (N, T+1, K, 3)
    source_weights: np.ndarray,  # (N, T+1, K)
) -> np.ndarray:
    """At each context position, for the dominant source, how well can we decode
    the hidden belief state?

    Returns
    -------
    r2_per_position : (T+1,) float array  (averaged over sources weighted by w_k)
    """
    T_plus_1 = acts.shape[1]
    K_local = source_weights.shape[2]
    r2_per_pos = np.zeros(T_plus_1)
    for pos in range(T_plus_1):
        X = acts[:, pos, :].astype(np.float64)
        X = X - X.mean(axis=0)
        r2_k = []
        for k in range(K_local):
            Y_k = hidden_beliefs[:, pos, k, :].astype(np.float64)
            w_k = source_weights[:, pos, k]           # (N,)
            if w_k.mean() < 0.05:
                r2_k.append(0.0)
                continue
            reg = Ridge(alpha=1e-3)
            reg.fit(X, Y_k, sample_weight=w_k)
            preds = reg.predict(X)
            ss_res = (w_k[:, None] * (Y_k - preds) ** 2).sum()
            ss_tot = (w_k[:, None] * (Y_k - Y_k.mean(axis=0)) ** 2).sum()
            r2_k.append(float(1 - ss_res / (ss_tot + 1e-12)))
        r2_per_pos[pos] = np.mean(r2_k)
    return r2_per_pos


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------

def run_full_analysis(
    model,
    cfg: ExperimentConfig | None = None,
    device: str = "cpu",
) -> dict:
    """Run all analyses and return results dict.

    Parameters
    ----------
    model  : trained HookedTransformer
    cfg    : ExperimentConfig
    device : torch device string

    Returns
    -------
    Large results dict with sub-dicts for each analysis component.
    """
    if cfg is None:
        cfg = get_default_config()

    model.eval()

    rng = np.random.default_rng(cfg.data.seed + 9999)
    batch = generate_batch(cfg.data.seq_len, cfg.data.n_analysis_sequences, rng)
    tokens_np = batch["tokens"]                         # (N, T+1)
    obs_np = batch["obs"]                               # (N, T)
    source_ids_np = batch["source_ids"]                 # (N,)

    print("Computing ground-truth belief states…")
    beliefs = compute_belief_states_batch(obs_np)       # dict of (N, T+1, ...)

    # Align with token positions: position 0 = after BOS, position t = after t obs
    source_weights = beliefs["source_weights"]          # (N, T+1, K)
    hidden_beliefs = beliefs["hidden_beliefs"]          # (N, T+1, K, 3)
    joint_beliefs = beliefs["joint_beliefs"]            # (N, T+1, 3K)

    print("Extracting residual stream activations…")
    CHUNK = 256
    all_acts = []
    for start in range(0, len(tokens_np), CHUNK):
        tokens_chunk = torch.tensor(
            tokens_np[start:start+CHUNK], dtype=torch.long, device=device
        )
        acts_chunk = extract_residual_stream(model, tokens_chunk)
        all_acts.append(acts_chunk)
    acts = np.concatenate(all_acts, axis=0)             # (N, T+1, d_model)

    print(f"  acts shape: {acts.shape}")

    # ---- 1. Global PCA (all positions pooled) ----
    print("PCA analysis…")
    acts_flat = acts.reshape(-1, acts.shape[-1])        # (N*(T+1), d_model)
    pca_global, _ = run_pca(acts_flat)
    eff_rank_global = effective_rank(pca_global, cfg.analysis.cev_threshold)
    cev_global = np.cumsum(pca_global.explained_variance_ratio_)
    print(f"  Global effective rank: {eff_rank_global}")

    # ---- 2. Belief regression (last position for most signal) ----
    print("Belief regression…")
    # Use all positions for richness
    acts_all_pos = acts.reshape(-1, acts.shape[-1])
    joint_all_pos = joint_beliefs.reshape(-1, joint_beliefs.shape[-1])
    source_all_pos = source_weights.reshape(-1, source_weights.shape[-1])
    hidden_all_pos = hidden_beliefs.reshape(-1, K * N_OBS)

    reg_joint = belief_regression(acts_all_pos, joint_all_pos, cfg.analysis.regression_n_splits)
    reg_source = belief_regression(acts_all_pos, source_all_pos, cfg.analysis.regression_n_splits)
    reg_hidden = belief_regression(acts_all_pos, hidden_all_pos, cfg.analysis.regression_n_splits)
    print(f"  Joint belief R² (CV): {reg_joint['r2_cv'].mean():.3f}")
    print(f"  Source weight R² (CV): {reg_source['r2_cv'].mean():.3f}")
    print(f"  Hidden belief R² (CV): {reg_hidden['r2_cv'].mean():.3f}")

    # ---- 3. Factor subspace extraction ----
    print("Factor subspace extraction…")
    # Use last position (most informative)
    last_pos_acts = acts[:, -1, :]                      # (N, d_model)
    last_pos_sw = source_weights[:, -1, :]              # (N, K)
    last_pos_hb = hidden_beliefs[:, -1, :, :]           # (N, K, 3)

    subspaces = extract_factor_subspaces(
        last_pos_acts, last_pos_sw, last_pos_hb,
        n_dims=cfg.analysis.n_subspace_dims,
    )
    overlap_matrix, overlap_labels = compute_subspace_overlap_matrix(subspaces)
    print("  Subspace overlap matrix:")
    print("  Labels:", overlap_labels)
    print("  Matrix:\n", np.round(overlap_matrix, 3))

    # ---- 4. Context-position dependent geometry (Task 4) ----
    print("Context-position analysis…")
    eff_rank_by_pos = context_position_effective_rank(
        acts, cfg.analysis.cev_threshold, max_dims=30
    )
    r2_source_by_pos = context_position_source_decodability(acts, source_weights)
    r2_hidden_by_pos = context_position_hidden_decodability(acts, hidden_beliefs, source_weights)
    print(f"  Eff rank range: {eff_rank_by_pos.min()}–{eff_rank_by_pos.max()}")

    # ---- Pack results ----
    return {
        # Raw data
        "acts": acts,                               # (N, T+1, d_model)
        "source_ids": source_ids_np,               # (N,)
        "source_weights": source_weights,          # (N, T+1, K)
        "hidden_beliefs": hidden_beliefs,          # (N, T+1, K, 3)
        "joint_beliefs": joint_beliefs,            # (N, T+1, 3K)

        # PCA
        "pca_global": pca_global,
        "eff_rank_global": eff_rank_global,
        "cev_global": cev_global,

        # Regression
        "reg_joint": reg_joint,
        "reg_source": reg_source,
        "reg_hidden": reg_hidden,

        # Subspaces
        "subspaces": subspaces,
        "overlap_matrix": overlap_matrix,
        "overlap_labels": overlap_labels,

        # Context-position
        "eff_rank_by_pos": eff_rank_by_pos,
        "r2_source_by_pos": r2_source_by_pos,
        "r2_hidden_by_pos": r2_hidden_by_pos,
    }
