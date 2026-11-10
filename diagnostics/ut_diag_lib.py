"""U_t self-consistency diagnostic for trained neuralTI-solvation models.

rho~_t : training-time interpolant density, x_t = I(x0, x1, t) (+ sigma_t z for DSM runs), x0/x1 Hungarian-aligned
U_t    : learned energy of the final checkpoint, U_t = t(1-t) U_NN + U_LJ(x; a(t))

For t on a grid:
  I(t) = E_{x ~ rho~_t}[dU_t/dt]      the neural-TI integrand (evaluated on interpolant samples)
  E(t) = E_{x ~ exp(-U_t)}[dU_t/dt]   Metropolis MCMC under the learned U_t, walkers started from rho~_t samples

The softening network gives a(0) = (0,1,0,1) and a(1) = (0,0,0,0) and t(1-t) U_NN vanishes at both ends, so
U_0 = U_A and U_1 = U_B exactly.  Hence int_0^1 E(t) dt = F_B - F_A for ANY parameters (plain TI along the learned
path), whereas int_0^1 I(t) dt is the neural-TI estimate.  I(t) = E(t) for all t iff exp(-U_t) is consistent with
rho~_t; I(t) - E(t) localises the neural-TI bias in t.  MCMC drift away from the step-0 value is the same signal.
"""
import os
import pickle
import sys
from functools import partial
from pathlib import Path

sys.dont_write_bytecode = True  # keep the source trees free of __pycache__ (python3 -I ignores the env variable)

ROOT = Path(__file__).resolve().parents[1]
CODE = Path(os.environ.get("NTI_UPSTREAM", ROOT / "upstream" / "neuralTI-solvation"))  # patched f756b30; U_t unchanged
RUNS = Path(os.environ.get("NTI_RUNS", ROOT / "runs"))  # one directory per training run (upstream layout)
DATA = RUNS / "egnn-seed0" / "data"  # endpoint data; byte-identical copies in every run dir
for _p in (str(CODE / "src"), str(CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp
import numpy as np


def set_jax_flags():
    # as experiments/config.yaml, except debug_nans: non-finite MCMC proposals are rejected and counted instead
    for k, v in dict(jax_enable_x64=False, jax_numpy_rank_promotion="raise", jax_debug_nans=False).items():
        jax.config.update(k, v)


# ----------------------------------------------------------------------------- data
def load_endpoints():
    with open(DATA / "LJ3Dwithsolute_0.00", "rb") as fh:
        X0 = np.asarray(pickle.load(fh))
    with open(DATA / "LJ3Dwithsolute_1.00", "rb") as fh:
        X1 = np.asarray(pickle.load(fh))
    return X0, X1


def aligned_pairs(n_pairs, seed, chunk, cache):
    """Random (x0, x1) pairs, solute-centred and Hungarian-aligned exactly as dataloader.batch_align; cached as npz."""
    cache = Path(cache)
    if cache.exists():
        d = np.load(cache)
        assert int(d["seed"]) == seed and len(d["b0"]) == n_pairs, (cache, int(d["seed"]), len(d["b0"]))
        return d["b0"], d["b1"]
    assert n_pairs % chunk == 0, (n_pairs, chunk)
    from dataloader import batch_align

    X0, X1 = load_endpoints()
    rng = np.random.default_rng(seed)
    i0 = rng.choice(len(X0), n_pairs, replace=False)
    i1 = rng.choice(len(X1), n_pairs, replace=False)
    align = jax.jit(batch_align)
    b0s, b1s = [], []
    for i in range(0, n_pairs, chunk):
        b0, b1 = align(jnp.asarray(X0[i0[i:i + chunk]]), jnp.asarray(X1[i1[i:i + chunk]]))
        b0s.append(np.asarray(b0))
        b1s.append(np.asarray(b1))
        print(f"aligned {i + chunk}/{n_pairs} pairs", flush=True)
    b0, b1 = np.concatenate(b0s), np.concatenate(b1s)
    cache.parent.mkdir(parents=True, exist_ok=True)
    # atomic: several jobs may build the same (deterministic) cache at once
    tmp = cache.with_name(f"{cache.stem}.tmp{os.getpid()}.npz")
    np.savez(tmp, b0=b0, b1=b1, i0=i0, i1=i1, seed=seed)
    os.replace(tmp, cache)
    return b0, b1


def endpoint_frames(t, n, seed):
    """n frames of the endpoint MCMC data for t = 0 (rho_A) or t = 1 (rho_B): there exp(-U_t) is exactly the
    sampled Boltzmann density for any parameters, so E(t) needs no new MCMC."""
    X0, X1 = load_endpoints()
    X = X0 if t == 0.0 else X1
    idx = np.random.default_rng(seed + (0 if t == 0.0 else 1)).choice(len(X), n, replace=False)
    return X[np.sort(idx)]


# ----------------------------------------------------------------------------- model
EXPECTED = {  # run dir -> settings the checkpoint must carry
    "egnn-seed0": dict(loss_type="tsm", loss_x_weight="lambda t: 1e-3", PRNGKey=0),
    "egnn-seed0-oldweight": dict(loss_type="tsm", loss_x_weight="lambda t: 1e-3 * jnp.exp(10*t*(t-1))", PRNGKey=0),
    "egnn-dsm-smid0.005-seed0": dict(loss_type="dsm", sigma_mid=5e-3, loss_x_weight="lambda t: 1.0", PRNGKey=0),
    "egnn-dsm-smid0.01-seed0": dict(loss_type="dsm", sigma_mid=1e-2, loss_x_weight="lambda t: 1.0", PRNGKey=0),
    "egnn-dsm-smid0.01-seed1": dict(loss_type="dsm", sigma_mid=1e-2, loss_x_weight="lambda t: 1.0", PRNGKey=1),
    "egnn-dsm-smid0.02-seed0": dict(loss_type="dsm", sigma_mid=2e-2, loss_x_weight="lambda t: 1.0", PRNGKey=0),
}


def load_model(run):
    """Final checkpoint of runs/<run> (written at grad step 50000) -> (stochastic_interpolant with params, target, info)."""
    import hydra

    with open(RUNS / run / "experiments/LJsolute.model", "rb") as fh:
        ck = pickle.load(fh)
    cfg = ck["cfg"]
    ts = hydra.utils.instantiate(cfg.target_system)
    si = hydra.utils.instantiate(cfg.model)
    si.num_features = ts.num_dim
    si.U_model = hydra.utils.instantiate(cfg.U_model)
    si.params = ck["params"]
    loss_type = str(cfg.model.get("loss_type", "tsm"))  # TSM checkpoints come from the unpatched upstream config
    info = dict(
        run=run,
        step=int(ck["opt_state"][0].count),
        loss_type=loss_type,
        sigma_min=float(cfg.model.sigma_min) if loss_type == "dsm" else 0.0,
        sigma_mid=float(cfg.model.sigma_mid) if loss_type == "dsm" else 0.0,
        loss_x_weight=str(cfg.model.loss_x_weight),
        PRNGKey=int(cfg.PRNGKey),
        cutoff=float(cfg.U_model.cutoff),
        size_to_pad=int(cfg.U_model.size_to_pad),
    )
    return si, ts, info


def check_info(info):
    exp = EXPECTED.get(info["run"])
    bad = [] if exp is not None else ["run not in EXPECTED"]
    for k, v in (exp or {}).items():
        if (abs(info[k] - v) > 1e-12) if isinstance(v, float) else (info[k] != v):
            bad.append(f"{k}={info[k]!r} expected {v!r}")
    if info["step"] != 50000:
        bad.append(f"step={info['step']} expected 50000")
    return bad


def make_interp_eval(si, loss_type):
    """Per-sample interpolant point and integrand, op-for-op as loss_one in stochastic_interpolant.loss_fn(_dsm).
    vmapped over (x0, x1, t[(1,)], z); returns (x_t, dU/dt, U, DSM score loss (0 for TSM))."""

    def one(params, x0, x1, t, z):
        dx = ((x1 - x0) + 0.5) % 1 - 0.5
        xt = x0 + t[0] * dx
        if loss_type == "dsm":
            s = si.sigma(t[0])
            xt = (xt + s * z) % 1
        else:
            xt = xt % 1
        U, (dUdx, dUdt) = jax.value_and_grad(partial(si.U, params), (0, 1))(xt, t)
        score = ((s * dUdx - z) ** 2).mean() if loss_type == "dsm" else jnp.zeros(())
        return xt, dUdt[0], U, score

    return jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0, 0)))


# ----------------------------------------------------------------------------- MCMC
def make_mcmc(U_fn):
    """Random-walk Metropolis as data/canonical_mcmc.py: x' = (x + sqrt(2 dt) N(0, 1)) mod 1 for all particles of a
    walker at once, accepted if u < exp(-(U' - U)).  Proposals with non-finite U' are rejected and counted.
    U_fn(params, x, t) -> scalar, t of shape (1,).  Returns (vmapped U, run) with
    run(params, x, U, key, t, step_std, n_steps) -> (x, U, key, n_accepted[W], n_nonfinite[W])."""
    Uv = jax.jit(jax.vmap(U_fn, in_axes=(None, 0, None)))

    @partial(jax.jit, static_argnames=("n_steps",))
    def run(params, x, U_curr, key, t, step_std, n_steps):
        def body(i, carry):
            x, U_curr, key, n_acc, n_bad = carry
            key, k1, k2 = jax.random.split(key, 3)
            x_prop = (x + step_std * jax.random.normal(k1, x.shape)) % 1
            U_prop = jax.vmap(U_fn, in_axes=(None, 0, None))(params, x_prop, t)
            ok = jnp.isfinite(U_prop)
            take = ok & (jax.random.uniform(k2, U_curr.shape) < jnp.exp(-(U_prop - U_curr)))
            x = jnp.where(take[:, None, None], x_prop, x)
            U_curr = jnp.where(take, U_prop, U_curr)
            return x, U_curr, key, n_acc + take, n_bad + (~ok)

        zero = jnp.zeros(U_curr.shape, jnp.int32)
        return jax.lax.fori_loop(0, n_steps, body, (x, U_curr, key, zero, zero))

    return Uv, run


def make_dudt(U_fn):
    """Per-walker dU_t/dt, vmapped over x."""
    g = jax.vmap(jax.grad(U_fn, argnums=2), in_axes=(None, 0, None))
    return jax.jit(lambda params, x, t: g(params, x, t).reshape(-1))


def make_structure(ts, cutoff):
    """Per-walker GNN edge count (same criterion as U_GNN; edges beyond size_to_pad are silently dropped there),
    solute coordination number (solvent within 1.5 sigma_AB), closest solvent-solvent distance / sigma and
    closest solute-solvent distance / sigma_AB."""
    from distance_on_torus import dR_on_torus_matrix, dist2_on_torus_matrix

    divide_by = (ts.sigma_AB / ts.sigma) ** 2

    def one(x):
        dR = dR_on_torus_matrix(x) / cutoff
        D2 = (dR**2).sum(-1) + 10 * jnp.eye(len(x))
        D2 = D2.at[:, 0].divide(divide_by)
        D2 = D2.at[0, :].divide(divide_by)
        r2 = dist2_on_torus_matrix(x)
        r2_ab = r2[0, 1:]
        r2_ss = r2[1:, 1:] + 10 * jnp.eye(len(x) - 1)
        return dict(
            n_edges=(D2 < 1).sum(),
            coord=(r2_ab < (1.5 * ts.sigma_AB) ** 2).sum(),
            rmin_ss=jnp.sqrt(r2_ss.min()) / ts.sigma,
            rmin_ab=jnp.sqrt(r2_ab.min()) / ts.sigma_AB,
        )

    return jax.jit(jax.vmap(one))


def walker_stats(rec):
    """rec [n_records, W] -> (mean, SE over independent walkers)."""
    wm = np.asarray(rec, np.float64).mean(0)
    return float(wm.mean()), float(wm.std(ddof=1) / np.sqrt(len(wm)))


def make_chain_fns(U_fn):
    """Build (and cache via jit) everything run_chain needs for one energy function; t is a traced argument."""
    Uv, run = make_mcmc(U_fn)
    return dict(Uv=Uv, run=run, dudt=make_dudt(U_fn))


def run_chain(params, fns, x, t, key, step_std, n_steps, record_every, structure=None, log=None):
    """MCMC from x under fns (make_chain_fns) at fixed t; records dU/dt, U (and structure) every record_every
    steps, step 0 included."""
    assert n_steps % record_every == 0, (n_steps, record_every)
    Uv, run, dudt = fns["Uv"], fns["run"], fns["dudt"]
    U_curr = Uv(params, x, t)
    rec = {"step": [0], "dUdt": [np.asarray(dudt(params, x, t))], "U": [np.asarray(U_curr)]}
    s_names = []
    if structure is not None:
        s0 = structure(x)
        s_names = list(s0)
        for k in s_names:
            rec[k] = [np.asarray(s0[k])]
    acc, bad = [], 0
    for c in range(n_steps // record_every):
        x, U_curr, key, n_acc, n_bad = run(params, x, U_curr, key, t, step_std, n_steps=record_every)
        rec["step"].append((c + 1) * record_every)
        rec["dUdt"].append(np.asarray(dudt(params, x, t)))
        rec["U"].append(np.asarray(U_curr))
        if structure is not None:
            s = structure(x)
            for k in s_names:
                rec[k].append(np.asarray(s[k]))
        acc.append(np.asarray(n_acc) / record_every)
        bad += int(np.asarray(n_bad).sum())
        if log is not None:
            log(c + 1, n_steps // record_every, rec, acc[-1])
    out = {k: np.stack(v) for k, v in rec.items() if k != "step"}
    out["step"] = np.asarray(rec["step"])
    out["acceptance"] = np.stack(acc) if acc else np.zeros((0, len(x)))
    out["n_nonfinite_proposals"] = bad
    return out, x, key


def trapezoid(ts, ys):
    ts, ys = np.asarray(ts, float), np.asarray(ys, float)
    return float(((ys[1:] + ys[:-1]) * np.diff(ts)).sum() / 2)
