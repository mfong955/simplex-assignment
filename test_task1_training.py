"""Task 1 test: train a small transformer on the non-ergodic Mess3 dataset.

Each training sequence is generated entirely by one randomly-chosen Mess3 source,
matching the Task 1 requirement. Source assignment is explicit per-sequence per-batch.

Run with:
    python test_task1_training.py
"""

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
# 1. Define the three Mess3 sources
# ---------------------------------------------------------------------------
SOURCES = [
    {"name": "mess3", "params": {"a": 0.95, "x": 0.05}},
    {"name": "mess3", "params": {"a": 0.60, "x": 0.15}},
    {"name": "mess3", "params": {"a": 0.30, "x": 0.25}},
]
K = len(SOURCES)
BOS_TOKEN = 3
VOCAB_SIZE = 4
SEQ_LEN = 16
BATCH_SIZE = 64
N_STEPS = 1000
N_STATES_PER_SOURCE = 3   # Mess3 has 3 hidden states

# ---------------------------------------------------------------------------
# 2. Build the composite non-ergodic HMM
# ---------------------------------------------------------------------------
hmm = build_nonergodic_hidden_markov_model(
    process_names=[s["name"] for s in SOURCES],
    process_params=[s["params"] for s in SOURCES],
    process_weights=[1.0, 1.0, 1.0],
)

# Build one initial state per source:
#   source k start = stationary dist of source k in its block, zeros elsewhere.
# The composite state has shape (K * N_STATES_PER_SOURCE,) = (9,).
per_source_initial_states = []
for k, source in enumerate(SOURCES):
    T_k = build_transition_matrices(HMM_MATRIX_FUNCTIONS, source["name"], source["params"])
    pi_k = get_stationary_state(T_k.sum(axis=0).T)        # stationary dist, shape (3,)
    s = jnp.zeros(K * N_STATES_PER_SOURCE)
    s = s.at[k * N_STATES_PER_SOURCE : (k + 1) * N_STATES_PER_SOURCE].set(pi_k)
    per_source_initial_states.append(s)

per_source_initial_states = jnp.stack(per_source_initial_states)   # (K, 9)
print("Per-source initial states (each row sums to 1/3 of total):")
print(np.array(per_source_initial_states))


def make_gen_states(source_ids: np.ndarray) -> jax.Array:
    """Build per-sequence initial belief states from an array of source ids."""
    return per_source_initial_states[source_ids]   # (B, 9)


# ---------------------------------------------------------------------------
# 3. Build a small HookedTransformer (1-3 layers, small context)
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

cfg = HookedTransformerConfig(
    d_model=128,
    d_head=32,
    n_heads=4,
    n_layers=2,
    d_mlp=512,
    n_ctx=SEQ_LEN + 1,
    d_vocab=VOCAB_SIZE,
    act_fn="gelu",
    normalization_type="LN",
    attn_only=False,
)
model = HookedTransformer(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ---------------------------------------------------------------------------
# 4. Training loop — source is sampled fresh each batch
# ---------------------------------------------------------------------------
print(f"\nTraining for {N_STEPS} steps (batch={BATCH_SIZE}, seq_len={SEQ_LEN})...")
print("Theoretical optimal loss: ~0.89 nats\n")

rng = np.random.default_rng(42)
key = jax.random.PRNGKey(0)

for step in range(N_STEPS):
    key, subkey = jax.random.split(key)

    # Sample one source per sequence, fresh every batch
    source_ids = rng.integers(0, K, size=BATCH_SIZE)
    gen_states = make_gen_states(source_ids)

    gen_states, inputs, labels = generate_data_batch(
        gen_states, hmm, BATCH_SIZE, SEQ_LEN, subkey,
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

    if step % 100 == 0 or step == N_STEPS - 1:
        source_counts = np.bincount(source_ids, minlength=K)
        print(f"  step {step:4d} | loss {loss.item():.4f} | sources {source_counts}")

print("\nDone.")
