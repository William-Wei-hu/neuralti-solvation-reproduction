"""Bidirectional nonequilibrium switching (NES) core, system independent.

Conventions (beta = 1, energies in kBT)
  path      U_t(x), t in [0, 1];  pi_t ∝ exp(-U_t);  dF = F_1 - F_0 = -ln(Z_1 / Z_0)
  grid      0 = t_0 < t_1 < ... < t_K = 1
  forward   x ~ pi_0.  for k = 0..K-1:  W += U_{t_{k+1}}(x) - U_{t_k}(x)   (one call, same x)
                                         x  = T_{t_{k+1}}^M (x)
  reverse   identical with the grid reversed (s_k = t_{K-k}), x ~ pi_1.
  Jarzynski <exp(-W_F)> = exp(-dF),  <exp(-W_R)> = exp(+dF);  Crooks P_F(W)/P_R(-W) = exp(W - dF).

T_t is a Metropolis-Hastings kernel that leaves pi_t exactly invariant and is reversible
(detailed balance), so the reverse protocol is the exact time reversal of the forward one and
Jarzynski / Crooks / BAR hold for ANY path U_t.  The kernel applied after the last increment
does not change W (kept only so every stage has the same shape).

Kernels (proposal noise std sqrt(2h) per coordinate, the Canonical_Sampler convention):
  rw    x' = x + sqrt(2h) xi                                   (symmetric; = upstream sampler)
  mala  x' = x + D(x) + sqrt(2h) xi,  D(x) = -h grad U_t(x), optionally "tamed":
        per particle the drift vector is rescaled to norm <= cap (cap = drift_cap_factor*sqrt(2h)).
        D is a deterministic function of x, so q(x'|x) is an exact Gaussian and MH is exact.
  hmc   L leapfrog steps of size eps = sqrt(2h) with unit mass, momentum p ~ N(0, I) refreshed every MH
        step, accept with min(1, exp(-dH)).  Leapfrog is volume preserving and time reversible, so with the
        momentum refresh the kernel is reversible w.r.t. pi_t.  hmc with L=1 is algebraically identical to
        untamed mala with the same h (same random numbers -> same trajectories; checked in toy_tests.py).
  ula   negative control ONLY: MALA proposal always accepted (does not preserve pi_t).

Torus: positions are wrapped mod 1 after the move.  The MH ratio uses the UNWRAPPED displacement d
(x' = (x + d) mod 1, reverse residual = -d - D(x')).  This is the exact MH ratio of the lifted chain
on R^{3N} with a periodic potential; its projection onto the torus satisfies detailed balance w.r.t.
pi_t on the torus (README, section "MALA on the torus").  No minimum-image approximation is involved.
HMC needs no displacement in its ratio; wrapping positions inside leapfrog does not change U or grad U
(both periodic), so the same lifted-chain argument applies.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

KINDS = ('mala', 'rw', 'hmc', 'ula')


def wrap_torus(x):
    return x % 1.0


def wrap_none(x):
    return x


def tamed_drift(g, h, cap):
    """-h*g, each particle row rescaled to norm <= cap (cap=None or inf: plain MALA)."""
    d = -h * g
    if cap is None:
        return d
    norm = jnp.sqrt(jnp.sum(d * d, axis=-1, keepdims=True))
    scale = jnp.minimum(1.0, cap / jnp.maximum(norm, 1e-300))
    return d * scale


def mh_step(x, U, g, t, h, key, *, kind, energy, value_and_grad, wrap, cap_factor, L=1):
    """One MH step targeting exp(-U_t) for a single configuration x (shape (n, dim)).

    U, g are U_t(x) and grad U_t(x) at the CURRENT t (g unused for rw).  L: leapfrog steps (hmc only).
    Returns x, U, g, acceptance probability min(1, alpha), accepted flag.
    """
    k1, k2 = jax.random.split(key)
    xi = jax.random.normal(k1, x.shape, x.dtype)
    s = jnp.sqrt(2.0 * h)
    cap = None if cap_factor is None else cap_factor * s
    if kind == 'rw':
        d = s * xi
        xp = wrap(x + d)
        Up = energy(xp, t)
        gp = g
        log_alpha = -(Up - U)
    elif kind in ('mala', 'ula'):
        D = tamed_drift(g, h, cap)
        d = D + s * xi
        xp = wrap(x + d)
        Up, gp = value_and_grad(xp, t)
        Dp = tamed_drift(gp, h, cap)
        lq_fwd = -jnp.sum((d - D) ** 2) / (4.0 * h)      # log q(x'|x)  (= -|xi|^2/2)
        lq_rev = -jnp.sum((-d - Dp) ** 2) / (4.0 * h)    # log q(x|x'), unwrapped residual
        log_alpha = -(Up - U) + lq_rev - lq_fwd
    elif kind == 'hmc':
        eps = s
        p = xi

        def leap(i, c):
            y, q, Uy, gy = c
            q = q - 0.5 * eps * gy
            y = wrap(y + eps * q)
            Uy, gy = value_and_grad(y, t)
            q = q - 0.5 * eps * gy
            return y, q, Uy, gy

        xp, q, Up, gp = jax.lax.fori_loop(0, L, leap, (x, p, U, g))
        log_alpha = -(Up - U) - 0.5 * (jnp.sum(q * q) - jnp.sum(p * p))
    else:
        raise ValueError(kind)
    log_alpha = jnp.where(jnp.isnan(log_alpha), -jnp.inf, log_alpha)
    finite = jnp.isfinite(Up) & jnp.all(jnp.isfinite(gp))
    if kind == 'ula':
        accept = finite
        acc_prob = jnp.exp(jnp.minimum(log_alpha, 0.0))
    else:
        accept = (jnp.log(jax.random.uniform(k2, (), x.dtype)) < log_alpha) & finite
        acc_prob = jnp.where(finite, jnp.exp(jnp.minimum(log_alpha, 0.0)), 0.0)
    x = jnp.where(accept, xp, x)
    U = jnp.where(accept, Up, U)
    g = jnp.where(accept, gp, g)
    return x, U, g, acc_prob, accept


def kernel_M(x, t, h, key, M, *, kind, energy, value_and_grad, wrap, cap_factor, L=1):
    """M MH steps at fixed t. Returns new x and mean acceptance probability."""
    if kind == 'rw':
        U = energy(x, t)
        g = jnp.zeros_like(x)
    else:
        U, g = value_and_grad(x, t)

    def body(m, carry):
        x, U, g, acc = carry
        x, U, g, a, _ = mh_step(x, U, g, t, h, jax.random.fold_in(key, m), kind=kind, energy=energy,
                                value_and_grad=value_and_grad, wrap=wrap, cap_factor=cap_factor, L=L)
        return x, U, g, acc + a

    x, U, g, acc = jax.lax.fori_loop(0, M, body, (x, U, g, jnp.zeros((), x.dtype)))
    return x, acc / M


def traj_key(seed, direction, root, rep):
    """Key depends only on (seed, direction, root, rep): batching/chunking/restarts give identical paths."""
    k = jax.random.PRNGKey(seed)
    k = jax.random.fold_in(k, {'fwd': 1, 'rev': 2}[direction])
    k = jax.random.fold_in(k, root)
    return jax.random.fold_in(k, rep)


def make_chunk_runner(*, energy, value_and_grad, delta, kind, wrap, M, cap_factor, nbins, aux=None, L=1):
    """Build a jitted runner advancing a batch of trajectories through stages [k0, k0+n).

    energy(x,t), value_and_grad(x,t), delta(x,ta,tb)=U_tb(x)-U_ta(x) act on ONE configuration.
    aux(x) -> scalar diagnostic evaluated at every stage point (max kept), e.g. GNN edge count.
    Grid arrays (ts, hs, bins) are passed in already ordered in the direction of travel.
    """
    if kind not in KINDS:
        raise ValueError(kind)
    kern = partial(kernel_M, M=M, kind=kind, energy=energy, value_and_grad=value_and_grad, wrap=wrap,
                   cap_factor=cap_factor, L=L)

    def one(x, W, Wbins, auxmax, nonfinite, key, ts, hs, bins, ks):
        def stage(carry, inp):
            x, W, Wbins, auxmax, nonfinite = carry
            k, ta, tb, hb, b = inp
            dW = delta(x, ta, tb)
            nonfinite = nonfinite + (~jnp.isfinite(dW)).astype(jnp.int32)
            W = W + dW
            Wbins = Wbins.at[b].add(dW)
            if aux is not None:
                auxmax = jnp.maximum(auxmax, aux(x))
            x, acc = kern(x, tb, hb, jax.random.fold_in(key, k))
            return (x, W, Wbins, auxmax, nonfinite), acc

        inputs = (ks, ts[:-1], ts[1:], hs[1:], bins)
        (x, W, Wbins, auxmax, nonfinite), acc = jax.lax.scan(stage, (x, W, Wbins, auxmax, nonfinite), inputs)
        return x, W, Wbins, auxmax, nonfinite, acc

    batched = jax.vmap(one, in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None))
    return jax.jit(batched)


def grads_per_stage(kind, M, L=1):
    """value_and_grad evaluations per stage (stage start + proposals); rw counts energy calls."""
    return 1 + M * (L if kind == 'hmc' else 1)


def grid(K, direction):
    """Linear grid in the direction of travel, and t-bin index of each increment's midpoint."""
    t = np.linspace(0.0, 1.0, K + 1)
    if direction == 'rev':
        t = t[::-1].copy()
    return t


def bin_index(t_dir, nbins):
    mid = 0.5 * (t_dir[:-1] + t_dir[1:])
    return np.minimum((mid * nbins).astype(np.int64), nbins - 1)


def run_protocol(runner, x0, keys, t_dir, h_dir, nbins, chunk, state=None, on_chunk=None, aux_init=0.0):
    """Drive a batch through all K stages in chunks (host loop). Returns dict of numpy arrays.

    state: optional dict from a previous partial run (resume). on_chunk(state) is called after each chunk.
    """
    K = len(t_dir) - 1
    B = x0.shape[0]
    bins = bin_index(t_dir, nbins)
    if state is None:
        state = dict(k=0, x=np.asarray(x0), W=np.zeros(B), Wbins=np.zeros((B, nbins)),
                     auxmax=np.full(B, aux_init), nonfinite=np.zeros(B, np.int32),
                     acc=np.zeros((B, K), np.float32))
    while state['k'] < K:
        k0 = state['k']
        k1 = min(K, k0 + chunk)
        sl = slice(k0, k1 + 1)
        out = runner(jnp.asarray(state['x']), jnp.asarray(state['W']), jnp.asarray(state['Wbins']),
                     jnp.asarray(state['auxmax']), jnp.asarray(state['nonfinite']), keys,
                     jnp.asarray(t_dir[sl]), jnp.asarray(h_dir[sl]), jnp.asarray(bins[k0:k1]),
                     jnp.arange(k0, k1))
        x, W, Wbins, auxmax, nonfinite, acc = jax.device_get(out)
        state.update(k=k1, x=x, W=W, Wbins=Wbins, auxmax=auxmax, nonfinite=nonfinite)
        state['acc'][:, k0:k1] = acc
        if on_chunk is not None:
            on_chunk(state)
    return state
