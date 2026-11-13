# Used by lj_fewstep.py: copy of escort_nes.py (used by lj_arms.py) with ONE addition:
# map_kind='table': params is an array (K, 3B) of logits ALONG t_dir (row k = the stage t_dir[k] -> t_dir[k+1], global
# stage index k, as passed in `ks`); the reverse direction is given the forward table reversed by the caller.  Every
# other line is the same as in escort_nes.py.
"""Bidirectional NES on the LJ base path with a local escort map inserted in every stage.

Uses the repository's nes/ package unchanged: nes_core (kernels, keys, grid, runner),
solvation_system (LJ base path, t = 0 decoupled, t = 1 coupled), estimators (BAR etc.).

Stage from ta to tb (either direction; beta = 1, energies in kBT), with Phi = the map of the pair (min, max):
  forward (ta < tb):  y = Phi(x);       dW = [U_tb(x) - U_ta(x)] + [U_tb(y) - U_tb(x)] - ln|det Phi'(x)|
  reverse (ta > tb):  y = Phi^{-1}(x);  dW = [U_tb(x) - U_ta(x)] + [U_tb(y) - U_tb(x)] + ln|det Phi'(y)|
  then M MH steps of the exact kernel at tb starting from y.
  The first bracket is nes_core's `delta` (one call, same x, as the baseline); the second is U_tb(y) (computed anyway
  by the kernel's first value_and_grad) minus one extra energy call U_tb(x).  The reverse stage is the time reversal
  of the forward stage: the kernel at the higher-index time comes before Phi^{-1} because the previous reverse stage
  ended with that kernel.  map=None: the stage is exactly nes_core.make_chunk_runner's stage.
Negative controls (tests only): 'flip_logdet', 'reverse_uses_forward_map', 'reverse_map_after_kernel'.
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent / 'nes'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import escort_map as EM  # noqa: E402
import nes_core as C  # noqa: E402

CONTROLS = (None, 'flip_logdet', 'reverse_uses_forward_map', 'reverse_map_after_kernel')


def kernel_from(y, Uy, gy, t, h, key, M, *, kind, energy, value_and_grad, wrap, L):
    """nes_core.kernel_M without the initial value_and_grad (given Uy, gy at y).  Returns x, U, acc."""
    def body(m, carry):
        x, U, g, acc = carry
        x, U, g, a, _ = C.mh_step(x, U, g, t, h, jax.random.fold_in(key, m), kind=kind, energy=energy,
                                  value_and_grad=value_and_grad, wrap=wrap, cap_factor=None, L=L)
        return x, U, g, acc + a
    x, U, g, acc = jax.lax.fori_loop(0, M, body, (y, Uy, gy, jnp.zeros((), y.dtype)))
    return x, U, acc / M


def make_escort_runner(*, path, kind, M, L, nbins, map_kind=None, r_c=0.35, B=8, control=None):
    """Batched runner with signature of nes_core.make_chunk_runner's runner plus `params` (map parameters).
    map_kind None -> returns nes_core.make_chunk_runner itself (wrapped to accept and ignore params)."""
    energy, vg, delta = path['energy'], path['value_and_grad'], path['delta']
    if control not in CONTROLS:
        raise ValueError(control)
    if map_kind is None:
        base = C.make_chunk_runner(energy=energy, value_and_grad=vg, delta=delta, kind=kind, wrap=C.wrap_torus, M=M,
                                   cap_factor=None, nbins=nbins, aux=None, L=L)

        def runner0(params, *args):
            out = base(*args)
            return out + (jnp.zeros_like(out[1]),)
        return runner0
    kern = partial(kernel_from, M=M, kind=kind, energy=energy, value_and_grad=vg, wrap=C.wrap_torus, L=L)

    def one(params, x, W, Wbins, auxmax, nonfinite, key, ts, hs, bins, ks):
        def stage(carry, inp):
            x, W, Wbins, auxmax, nonfinite, LD = carry
            k, ta, tb, hb, b = inp
            theta = params[k] if map_kind == 'table' else EM.stage_theta(map_kind, params, ta, tb, B)
            fwd = tb > ta
            d0 = delta(x, ta, tb)
            if control == 'reverse_map_after_kernel':
                # WRONG on purpose (reverse only): kernel first at tb from x, inverse map afterwards
                def rev_wrong(_):
                    U0, g0 = vg(x, tb)
                    xk, _, acc = kern(x, U0, g0, tb, hb, jax.random.fold_in(key, k))
                    y, ld = EM.apply_inverse(xk, theta, r_c, B)
                    return y, d0 + ld, acc, ld

                def fwd_ok(_):
                    y, ld = EM.apply_map(x, theta, r_c, B)
                    Uy, gy = vg(y, tb)
                    dW = d0 + (Uy - energy(x, tb)) - ld
                    xn, _, acc = kern(y, Uy, gy, tb, hb, jax.random.fold_in(key, k))
                    return xn, dW, acc, -ld
                xn, dW, acc, ldr = jax.lax.cond(fwd, fwd_ok, rev_wrong, None)
            else:
                def go_fwd(_):
                    y, ld = EM.apply_map(x, theta, r_c, B)
                    return y, -ld if control != 'flip_logdet' else ld

                def go_rev(_):
                    if control == 'reverse_uses_forward_map':
                        y, ld = EM.apply_map(x, theta, r_c, B)          # WRONG on purpose
                        return y, -ld
                    y, ld = EM.apply_inverse(x, theta, r_c, B)
                    return y, ld if control != 'flip_logdet' else -ld
                y, ldr = jax.lax.cond(fwd, go_fwd, go_rev, None)
                Uy, gy = vg(y, tb)
                dW = d0 + (Uy - energy(x, tb)) + ldr
                xn, _, acc = kern(y, Uy, gy, tb, hb, jax.random.fold_in(key, k))
            nonfinite = nonfinite + (~jnp.isfinite(dW)).astype(jnp.int32)
            W = W + dW
            Wbins = Wbins.at[b].add(dW)
            return (xn, W, Wbins, auxmax, nonfinite, LD + ldr), acc

        inputs = (ks, ts[:-1], ts[1:], hs[1:], bins)
        z = jnp.zeros((), x.dtype)
        (x, W, Wbins, auxmax, nonfinite, LD), acc = jax.lax.scan(stage, (x, W, Wbins, auxmax, nonfinite, z), inputs)
        return x, W, Wbins, auxmax, nonfinite, acc, LD

    return jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0, 0, 0, 0, None, None, None, None)))


def run(runner, params, x0, keys, t_dir, h, nbins, chunk=200):
    """Host loop over chunks (as nes_core.run_protocol).  Returns dict of numpy arrays incl. summed log-det LD."""
    K = len(t_dir) - 1
    B = x0.shape[0]
    bins = C.bin_index(t_dir, nbins)
    h_dir = np.full(K + 1, h)
    st = dict(x=np.asarray(x0), W=np.zeros(B), Wbins=np.zeros((B, nbins)), auxmax=np.zeros(B),
              nonfinite=np.zeros(B, np.int32), acc=np.zeros((B, K), np.float32), LD=np.zeros(B))
    k0 = 0
    while k0 < K:
        k1 = min(K, k0 + chunk)
        out = runner(params, jnp.asarray(st['x']), jnp.asarray(st['W']), jnp.asarray(st['Wbins']),
                     jnp.asarray(st['auxmax']), jnp.asarray(st['nonfinite']), keys, jnp.asarray(t_dir[k0:k1 + 1]),
                     jnp.asarray(h_dir[k0:k1 + 1]), jnp.asarray(bins[k0:k1]), jnp.arange(k0, k1))
        x, W, Wb, am, nf, acc, LD = jax.device_get(out)
        st.update(x=x, W=W, Wbins=Wb, auxmax=am, nonfinite=nf, LD=st['LD'] + LD)
        st['acc'][:, k0:k1] = acc
        k0 = k1
    return st


def densified_grid(K, profile_json, direction, floor=0.5):
    """Grid with local density proportional to sqrt(max(d(t), floor)), d = baseline per-bin dissipation density
    (forward_mean_dWdt - reverse_minus_mean_dWdt) from a baseline summary.json, smoothed over 5 bins.  Fixed rule."""
    import json
    p = json.loads(Path(profile_json).read_text())['profile']
    c = np.asarray(p['t_bin_centers'])
    d = np.asarray(p['forward_mean_dWdt']) - np.asarray(p['reverse_minus_mean_dWdt'])
    d = np.convolve(d, np.ones(5) / 5, mode='same')
    rho = np.sqrt(np.maximum(d, floor))
    edges = np.linspace(0, 1, len(c) + 1)
    cdf = np.concatenate([[0], np.cumsum(rho * np.diff(edges))])
    cdf /= cdf[-1]
    t = np.interp(np.linspace(0, 1, K + 1), cdf, edges)
    t[0], t[-1] = 0.0, 1.0
    return t if direction == 'fwd' else t[::-1].copy()
