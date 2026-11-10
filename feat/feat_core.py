"""FEAT-style escorted transport with the forward-backward Radon-Nikodym derivative (FB RND).

He et al., J. Stat. Mech. (2026) 074003 (FEAT), eqs. 31-36.  Conventions (beta = 1, energies in kBT):

  Grid in the direction of travel t_0, ..., t_K (forward 0 -> 1, reverse 1 -> 0), dt_k = |t_{k+1} - t_k|.
  Fields at (x, t):  v(x, t) (velocity), g(x, t) = grad_x U_theta(x, t)  (score = -g).
  Drift of the kernel that travels forward in time:   D+(x, t) = dt (+v - eps g)          (eq. 31)
  Drift of the kernel that travels backward in time:  D-(x, t) = dt (-v - eps g)          (eq. 32)
  Both are optionally "tamed" per particle (norm <= cap_factor * sqrt(2 eps dt)); any deterministic
  drift function keeps the FB RND exact as long as the same function is used in both kernels.

  One EM step from (x, t_k) to (x', t_{k+1}):
      D_own = drift of the kernel travelling in the run's direction, at (x, t_k)
      d     = D_own + s eta,   s = sqrt(2 eps dt),  eta ~ N(0, I);   x' = wrap(x + d)
      D_oth = drift of the opposite kernel, at (x', t_{k+1})
      c     = -(D_own + D_oth) / s
      w     = -eta.c + |c|^2 / 2          (= ln N_own(x'|x) - ln N_oth(x|x'), no large cancellation)
  HMC block at t_{k+1} (exact, reversible w.r.t. exp(-U_theta(., t_{k+1}))):  w += U(x_before) - U(x_after)

  Work, the same formula in both directions:
      W = U_finish(x_K) - U_begin(x_0) + sum_k w_k
  forward:  U_begin = U_0 (true endpoint), U_finish = U_1  ->  W_F  (dF = -ln <exp(-W_F)>)
  reverse:  U_begin = U_1, U_finish = U_0                   ->  W_R  (dF = +ln <exp(-W_R)>, raw reverse work)
  so estimators.py (BAR etc.) of the NES directory is used unchanged.

  Torus: x' = (x + d) mod 1 with the UNWRAPPED increment d in the RND.  Augmenting the path with the
  increments (d_1..d_K), (x_0, d_1..d_K) <-> (x_K, d_K..d_1) is a bijection with unit Jacobian, so the
  identity holds exactly on the augmented space (same lifting argument as the NES MALA kernel).

  eps = 0 is allowed only with v == 0 (identity transport, e.g. pure HMC NES; regression test T4).

Random streams: EM noise of stage k uses fold_in(fold_in(key, EM_STREAM), k); the HMC block of stage k
uses fold_in(key, k) exactly like nes_core.make_chunk_runner, so v = 0, eps = 0 with HMC at every level
reproduces the NES trajectories of the same trajectory key.
"""
from __future__ import annotations

from functools import partial
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

NES_CODE = Path(__file__).resolve().parents[1] / 'nes'
if str(NES_CODE) not in sys.path:
    sys.path.insert(0, str(NES_CODE))
import nes_core as C  # noqa: E402  (kernel_M, traj_key, grid, bin_index, wrap_*)

EM_STREAM = 0x7FFFFFFF
VARIANTS = ('exact', 'other_at_x', 'minimage')   # the last two are negative / sensitivity controls only


def tame(D, cap):
    """Rescale each particle row of D to norm <= cap (cap None: unchanged). Returns (D, any_capped)."""
    if cap is None:
        return D, jnp.zeros((), bool)
    norm = jnp.sqrt(jnp.sum(D * D, axis=-1, keepdims=True))
    scale = jnp.minimum(1.0, cap / jnp.maximum(norm, 1e-300))
    return D * scale, jnp.any(norm > cap)


def two_scalar_grads(f2):
    """f2(x, t) -> (U, phi) from one shared backbone.  Returns g(x, t) -> (U, phi, grad U, grad phi)
    with ONE forward pass and two VJPs."""
    def g(x, t):
        (a, b), vjp = jax.vjp(lambda y: f2(y, t), x)
        ga, = vjp((jnp.ones_like(a), jnp.zeros_like(b)))
        gb, = vjp((jnp.zeros_like(a), jnp.ones_like(b)))
        return a, b, ga, gb
    return g


def fields_shared(f2):
    """Shared backbone, two scalar heads: v = grad phi, g = grad U."""
    tg = two_scalar_grads(f2)

    def fields(x, t):
        _, _, gU, gphi = tg(x, t)
        return gphi, gU
    return fields


def fields_separate(v_fn, U_fn):
    """v_fn(x, t) -> velocity array (e.g. grad of a separate phi network); U_fn(x, t) -> scalar U_theta."""
    gU = jax.grad(U_fn)

    def fields(x, t):
        return v_fn(x, t), gU(x, t)
    return fields


def make_runner(*, fields, eps, direction, wrap, cap_factor=None, nbins=100, hmc=None, aux=None,
                variant='exact', record=False):
    """Jitted, vmapped runner over a contiguous block of stages.

    fields(x, t) -> (v, g) for ONE configuration (unused when eps == 0).
    hmc: None or dict(energy, value_and_grad, h, L, M); flags (bool per stage, shared by the batch) say
         at which stages the HMC block runs.
    Returns (init, run):
      init(x0, t0) -> state dict for a batch
      run(state, keys, ts, bins, flags, ks) -> (state, acc (B, n), xs (B, n, ...) or None)
    """
    if variant not in VARIANTS:
        raise ValueError(variant)
    eps = float(eps)
    sig = {'fwd': 1.0, 'rev': -1.0}[direction]
    if eps == 0.0 and variant != 'exact':
        raise ValueError('controls need eps > 0')
    kern = None
    if hmc is not None:
        kern = partial(C.kernel_M, M=int(hmc['M']), kind='hmc', energy=hmc['energy'],
                       value_and_grad=hmc['value_and_grad'], wrap=wrap, cap_factor=None, L=int(hmc['L']))

    def own(v, g, dt):
        return dt * (sig * v - eps * g)

    def oth(v, g, dt):
        return dt * (-sig * v - eps * g)

    def init_one(x, t0):
        if eps > 0.0:
            v, g = fields(x, t0)
        else:
            v = g = jnp.zeros_like(x)
        z = jnp.zeros((), x.dtype)
        return dict(x=x, v=v, g=g, S=z, Sbins=jnp.zeros(nbins, x.dtype), nonfinite=jnp.zeros((), jnp.int32),
                    auxmax=z, ncapped=jnp.zeros((), jnp.int32), csq=z)

    def run_one(st, key, ts, bins, flags, ks):
        def step(carry, inp):
            x, v, g, S, Sb, nonf, auxm, ncap, csq = carry
            k, ta, tb, b, flag = inp
            dt = jnp.abs(tb - ta)
            if eps > 0.0:
                s = jnp.sqrt(2.0 * eps * dt)
                cap = None if cap_factor is None else cap_factor * s
                D, c1 = tame(own(v, g, dt), cap)
                eta = jax.random.normal(jax.random.fold_in(jax.random.fold_in(key, EM_STREAM), k), x.shape, x.dtype)
                d = D + s * eta
                xp = wrap(x + d)
                vp, gp = fields(xp, tb)
                if variant == 'other_at_x':
                    Do, c2 = tame(oth(v, g, dt), cap)       # WRONG on purpose: opposite kernel's drift at x
                else:
                    Do, c2 = tame(oth(vp, gp, dt), cap)
                if variant == 'minimage':                    # WRONG on purpose: minimum-image residual
                    r = -d - Do
                    r = r - jnp.round(r)
                    w = -0.5 * jnp.sum(eta * eta) + jnp.sum(r * r) / (2.0 * s * s)
                    cc = jnp.zeros((), x.dtype)
                else:
                    c = -(D + Do) / s
                    cc = jnp.sum(c * c)
                    w = -jnp.sum(eta * c) + 0.5 * cc
                ncap = ncap + (c1 | c2).astype(jnp.int32)
                csq = csq + cc
            else:
                xp, vp, gp = x, v, g
                w = jnp.zeros((), x.dtype)
            acc = jnp.full((), -1.0, x.dtype)
            if kern is not None:
                def do(a):
                    y, vy, gy = a
                    Ub = hmc['energy'](y, tb)
                    yq, ac = kern(y, tb, hmc['h'], jax.random.fold_in(key, k))
                    Ua = hmc['energy'](yq, tb)
                    if eps > 0.0:
                        vq, gq = fields(yq, tb)
                    else:
                        vq, gq = vy, gy
                    return yq, vq, gq, Ub - Ua, ac

                def skip(a):
                    y, vy, gy = a
                    return y, vy, gy, jnp.zeros((), x.dtype), jnp.full((), -1.0, x.dtype)

                xp, vp, gp, hw, acc = jax.lax.cond(flag, do, skip, (xp, vp, gp))
                w = w + hw
            nonf = nonf + (~jnp.isfinite(w)).astype(jnp.int32)
            S = S + w
            Sb = Sb.at[b].add(w)
            if aux is not None:
                auxm = jnp.maximum(auxm, aux(xp))
            return (xp, vp, gp, S, Sb, nonf, auxm, ncap, csq), (acc, xp if record else jnp.zeros((), x.dtype))

        carry = (st['x'], st['v'], st['g'], st['S'], st['Sbins'], st['nonfinite'], st['auxmax'], st['ncapped'],
                 st['csq'])
        inputs = (ks, ts[:-1], ts[1:], bins, flags)
        carry, (acc, xs) = jax.lax.scan(step, carry, inputs)
        names = ('x', 'v', 'g', 'S', 'Sbins', 'nonfinite', 'auxmax', 'ncapped', 'csq')
        return dict(zip(names, carry)), acc, xs

    init = jax.jit(jax.vmap(init_one, in_axes=(0, None)))
    run = jax.jit(jax.vmap(run_one, in_axes=(0, 0, None, None, None, None)))
    return init, run


def run_protocol(init, run, x0, keys, t_dir, flags, nbins, chunk=None):
    """Host loop over chunks of stages. Returns dict of numpy arrays (+ 'xs' if the runner records)."""
    K = len(t_dir) - 1
    chunk = chunk or K
    bins = C.bin_index(t_dir, nbins)
    st = init(jnp.asarray(x0), jnp.asarray(t_dir[0]))
    accs, xss = [], []
    k0 = 0
    while k0 < K:
        k1 = min(K, k0 + chunk)
        st, acc, xs = run(st, keys, jnp.asarray(t_dir[k0:k1 + 1]), jnp.asarray(bins[k0:k1]),
                          jnp.asarray(flags[k0:k1]), jnp.arange(k0, k1))
        accs.append(np.asarray(acc))
        xss.append(np.asarray(xs))
        k0 = k1
    out = {k: np.asarray(v) for k, v in jax.device_get(st).items()}
    out['acc'] = np.concatenate(accs, axis=1)
    out['xs'] = np.concatenate(xss, axis=1)
    return out


def work(out, x0, U_begin, U_finish):
    """W = U_finish(x_K) - U_begin(x_0) + S (U_* act on a batch)."""
    return np.asarray(U_finish(jnp.asarray(out['x']))) - np.asarray(U_begin(jnp.asarray(x0))) + out['S']
