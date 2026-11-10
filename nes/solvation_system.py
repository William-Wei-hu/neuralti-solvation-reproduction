"""LJ solute in LJ solvent (Mate, Fleuret, Bereau, JCP 162, 124107 (2025)) as NES paths.

Uses the upstream code read-only (balintmate/neuralTI-solvation f756b30):
  data/target_systems.py  LJ3D_with_solute   (U, U_ij, U0, U1, Ut)
  src/stochastic_interpolant.py               (learned U_t = t(1-t) U_NN + U(x, LJsoft(t)))
  src/U_GNN.py                                (EGNN U_NN)

Paths
  base     U_t(x) = target_system.Ut(t, x) = U(x, a=(1-t)[0,1,0,1])                 (no model)
  learned  U_t(x) = stochastic_interpolant.U(params, x, t)                          (checkpoint)
Both equal U0 at t=0 and U1 at t=1 exactly (checked in smoke.py).

delta(x, ta, tb) = U_tb(x) - U_ta(x) is evaluated in ONE call on the same x: the pairwise LJ
matrices at ta and tb are subtracted element-wise before the (identical) reduction used by
target_system.U, and the NN terms are subtracted afterwards.  This avoids subtracting two
O(10^3) kBT totals.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

UPSTREAM = Path(os.environ.get('NTI_UPSTREAM', Path(__file__).resolve().parents[1] / 'upstream' / 'neuralTI-solvation'))  # patched upstream checkout (scripts/get_upstream.sh)
NES_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = NES_ROOT / 'models' / 'egnn-seed0.model'  # TSM EGNN, upstream settings
UPSTREAM_COMMIT = 'f756b308c4391a4dfcde49d7e72c5922ebb623e5'

for p in (UPSTREAM / 'src', UPSTREAM):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data.target_systems import LJ3D_with_solute  # noqa: E402
from distance_on_torus import dist2_on_torus_matrix, dR_on_torus_matrix  # noqa: E402

def target():
    return LJ3D_with_solute()


def _pair_sum(ts, x, Uij_fn):
    """Reproduce target_system.U's reduction for an arbitrary pair matrix builder."""
    n = x.shape[0]
    eye = jnp.eye(n, dtype=x.dtype)
    r2 = dist2_on_torus_matrix(x) + eye
    mask = (1 - eye) * 0.5
    return 0.5 * (Uij_fn(r2) * mask).sum()


def base_a(t):
    # identical expression to LJ3D_with_solute.Ut
    return (1 - t) * jnp.array([0.0, 1.0, 0.0, 1.0])


def make_base_path(ts=None):
    ts = ts or target()

    def energy(x, t):
        return ts.Ut(t, x)

    def delta(x, ta, tb):
        aa, ab = base_a(ta), base_a(tb)
        return _pair_sum(ts, x, lambda r2: ts.U_ij(r2=r2, a=ab) - ts.U_ij(r2=r2, a=aa))

    vg = jax.value_and_grad(energy)
    return dict(name='base', energy=energy, value_and_grad=vg, delta=delta, aux=None, ts=ts, info={})


def load_checkpoint(path=DEFAULT_CKPT):
    path = Path(path)
    raw = path.read_bytes()
    with path.open('rb') as f:
        ck = pickle.load(f)
    return ck, hashlib.sha256(raw).hexdigest()


def make_learned_path(ckpt=DEFAULT_CKPT, dtype=None):
    """Rebuild stochastic_interpolant exactly as experiments/main.py does and load params."""
    import hydra
    ck, sha = load_checkpoint(ckpt)
    cfg = ck['cfg']
    SI = hydra.utils.instantiate(cfg.model)
    SI.num_features = SI.target_system.num_dim
    SI.U_model = hydra.utils.instantiate(cfg.U_model)
    params = ck['params']
    dtype = dtype or (jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)
    params = jax.tree_util.tree_map(lambda v: jnp.asarray(v, dtype=dtype), params)
    ts = SI.target_system
    try:
        step = int(np.asarray(ck['opt_state'][0].count))  # main.py: step = opt_state[0].count
    except Exception as e:  # diagnostic only
        step = f'unknown ({type(e).__name__}: {e})'

    def t1(t):
        return jnp.reshape(t, (1,))

    def energy(x, t):
        return SI.U(params, x, t1(t))

    def nn(x, t):
        return SI.U_model.apply(params['U'], x, t1(t)).reshape()

    def delta(x, ta, tb):
        la = SI.getLJ_softening(params['LJ'], t1(ta))
        lb = SI.getLJ_softening(params['LJ'], t1(tb))
        base = _pair_sum(ts, x, lambda r2: ts.U_ij(r2=r2, a=lb) - ts.U_ij(r2=r2, a=la))
        return base + tb * (1 - tb) * nn(x, tb) - ta * (1 - ta) * nn(x, ta)

    U_model = SI.U_model

    def edge_count(x):
        """Same edge rule as U_GNN.U_model; upstream pads/truncates the list at size_to_pad."""
        dR = dR_on_torus_matrix(x) / U_model.cutoff
        D2 = (dR ** 2).sum(-1) + 10 * jnp.eye(len(x), dtype=x.dtype)
        div = (ts.sigma_AB / ts.sigma) ** 2
        D2 = D2.at[:, 0].divide(div).at[0, :].divide(div)
        return jnp.sum(D2 < 1).astype(x.dtype)

    vg = jax.value_and_grad(energy)
    from omegaconf import OmegaConf
    info = dict(checkpoint=str(ckpt), checkpoint_sha256=sha, optimizer_step=step,
                size_to_pad=int(U_model.size_to_pad), params_dtype=str(dtype.__name__ if hasattr(dtype, '__name__') else dtype),
                cfg_model=OmegaConf.to_container(cfg.model, resolve=True),
                cfg_U_model=OmegaConf.to_container(cfg.U_model, resolve=True))
    return dict(name='learned', energy=energy, value_and_grad=vg, delta=delta, aux=edge_count, ts=ts,
                info=info, SI=SI, params=params, nn=nn)


def make_path(name, ckpt=DEFAULT_CKPT):
    if name == 'base':
        return make_base_path()
    if name == 'learned':
        return make_learned_path(ckpt)
    raise ValueError(name)


def disperse(key, n_chains, ts, max_iter=200000):
    """Upstream Canonical_Sampler initialisation: uniform positions, Adam on sum 1/(r^2+1e-4)
    until the minimum pair distance over ALL chains is >= sqrt(0.7) sigma."""
    import optax
    N = ts.N
    x0 = jax.random.uniform(key, (n_chains, N, ts.num_dim), dtype=jnp.float64 if jax.config.jax_enable_x64
                            else jnp.float32)

    def loss_one(x):
        D = dist2_on_torus_matrix(x)
        mask = 1 - jnp.eye(N, dtype=x.dtype)
        return ((1 / (D + 1e-4)) * mask).sum()

    def loss(x):
        return jax.vmap(loss_one)(x).mean()

    optim = optax.adam(learning_rate=1e-4)
    opt_state = optim.init(x0)

    @jax.jit
    def move(x0, opt_state):
        g = jax.grad(loss)(x0)
        updates, opt_state = optim.update(g, opt_state, x0)
        x0 = optax.apply_updates(x0, updates) % 1
        D = jax.vmap(dist2_on_torus_matrix)(x0) / ts.sigma ** 2
        eye = jnp.eye(N, dtype=x0.dtype).reshape(1, N, N)
        return (D + eye).min(), x0, opt_state

    D_min, it = 0.0, 0
    while D_min < 0.7:
        D_min, x0, opt_state = move(x0, opt_state)
        it += 1
        if it > max_iter:
            raise RuntimeError(f'dispersal did not reach D_min>=0.7 (D_min={float(D_min)})')
    return x0, it
