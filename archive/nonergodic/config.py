"""Experiment configuration for non-ergodic Mess3 transformer study."""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Mess3 source parameters
# Three sources chosen to be clearly distinguishable:
#   Source 0 — high autocorrelation (structured), easy to identify early
#   Source 1 — medium structure (paper parameters)
#   Source 2 — low autocorrelation (noisy), harder to identify
# ---------------------------------------------------------------------------
MESS3_PARAMS = [
    {"a": 0.95, "x": 0.05},   # Source 0: highly structured (autocorr ≈ 0.82, identified in ~16 tokens)
    {"a": 0.60, "x": 0.15},   # Source 1: moderate — Shai2026 params (autocorr ≈ 0.40)
    {"a": 0.10, "x": 0.45},   # Source 2: nearly i.i.d. (autocorr ≈ 0.27, very slow identification)
]
K = len(MESS3_PARAMS)          # number of sources (3)

# Token vocabulary: {0, 1, 2} are Mess3 observations; 3 is BOS
VOCAB_SIZE = 4
BOS_TOKEN = 3
N_OBS = 3                      # observation alphabet size for each source


@dataclass
class DataConfig:
    """Data generation parameters."""
    seq_len: int = 48           # observable tokens per sequence (not counting BOS)
    batch_size: int = 256
    n_train_batches: int = 10_000
    n_analysis_sequences: int = 4_096   # sequences for post-hoc analysis
    seed: int = 42


@dataclass
class ModelConfig:
    """HookedTransformer architecture parameters."""
    d_model: int = 128
    d_head: int = 32
    n_heads: int = 4            # d_model = d_head * n_heads
    n_layers: int = 2
    d_mlp: int = 512            # 4 * d_model
    n_ctx: int = 64             # must be >= seq_len + 1 (BOS)
    vocab_size: int = VOCAB_SIZE
    act_fn: str = "gelu"
    normalization_type: Optional[str] = "LN"


@dataclass
class TrainConfig:
    """Optimizer / scheduler parameters."""
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    log_every: int = 200        # log loss every N steps
    analyze_every: int = 2_000  # run analysis every N steps
    checkpoint_every: int = 5_000


@dataclass
class AnalysisConfig:
    """Post-training analysis parameters."""
    cev_threshold: float = 0.95     # fraction of variance for effective rank
    regression_n_splits: int = 5    # cross-val folds for belief regression
    context_positions: list = field(
        default_factory=lambda: list(range(1, 49))  # positions 1..48
    )
    n_subspace_dims: int = 2        # expected dims per factor subspace (2 for simplex)


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    out_dir: str = "results"
    run_name: str = "nonergodic_mess3_k3"
    device: str = "cpu"             # overridden at runtime if GPU available


def get_default_config() -> ExperimentConfig:
    return ExperimentConfig()
