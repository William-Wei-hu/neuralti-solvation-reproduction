"""Bidirectional NES on the 4D-decoupling path of an MLIP (MACE-OFF methane in water).

Units: energy eV, length A, mass amu, time fs.  Momentum p = m v (amu A / fs); kinetic energy = sum p^2 / 2m * MVV2E.
Work is reported in kBT.  lambda = 0 coupled, lambda = 1 decoupled; forward = 0 -> 1.

Model interface (energy_fn), one configuration:  efn(x[n,3] A, lam) -> (U eV, F[n,3] eV/A, n_edges)
  U is the interaction energy (total minus the constant E0) in float64; F = -dU/dx.  Built from an e_and_f that returns
  (E_total, E_int, F, n_edges), e.g. mace_iface.MaceIface._one, by `from_mlip4d`.

One stage k (lam_k -> lam_{k+1}), the same in both directions (NES with exact kernels, as nes/nes_core.make_chunk_runner):
  work += (U(x, lam_{k+1}) - U(x, lam_k)) / kBT            (U(x, lam_k) is cached from the previous stage)
  then one MH step of an exact kernel at lam_{k+1}:
    K1 / K2  'hmc'  : p ~ N(0, m kBT) fresh; L leapfrog steps; accept with min(1, exp(-dH/kBT))
    K3       'ghmc' : p <- a p + sqrt(1 - a^2) xi,  a = exp(-gamma L dt); L leapfrog steps; accept, else p <- -p
  Both leave pi_lam(x) N(p) invariant (GHMC with the momentum flip: skew detailed balance); the kinetic energy does
  not depend on lam, so the work is only the potential-energy jump.
Positions are wrapped per atom (x mod Lbox); the energy uses minimum-image pairs and r_max < Lbox / 2.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

KB_EV = 8.617333262e-5            # eV / K
MVV2E = 1.66053906660e-27 * 1e10 / 1.602176634e-19   # 1 amu A^2 / fs^2 in eV  (= 103.6427 eV)
MASS = {1: 1.008, 6: 12.011, 8: 15.999}               # amu (IUPAC conventional)


def kbt(T=298.15):
    return KB_EV * T


def masses_for(Z):
    return np.array([MASS[int(z)] for z in Z], np.float64)


def from_mlip4d(e_and_f):
    def efn(x, lam):
        _, eint, F, ne = e_and_f(x, lam)
        return eint, F, ne
    return efn


def kinetic(p, m):
    return 0.5 * jnp.sum(p * p / m[:, None]) * MVV2E


def draw_p(key, m, kT, dtype):
    sd = jnp.sqrt(m * kT / MVV2E)[:, None]
    return sd * jax.random.normal(key, (len(m), 3), dtype)


def leapfrog(efn, x, p, F, lam, m, dt, L, box):
    """L velocity-Verlet steps; returns x, p, U, F, max n_edges seen."""
    def body(c, _):
        x, p, F, U, ne = c
        p = p + 0.5 * dt * F / MVV2E
        x = (x + dt * p / m[:, None]) % box
        U, F, n = efn(x, lam)
        p = p + 0.5 * dt * F / MVV2E
        return (x, p, F, U, jnp.maximum(ne, jnp.asarray(n).astype(jnp.int32))), None   # n_edges may be int64 (x64)
    U0 = jnp.zeros((), x.dtype)
    (x, p, F, U, ne), _ = jax.lax.scan(body, (x, p, F, U0, jnp.zeros((), jnp.int32)), None, length=L)
    return x, p, U, F, ne


def mh_step(efn, state, lam, key, *, kind, m, dt, L, box, kT, gamma=None):
    """One exact MH step at fixed lam.  state = dict(x, p, U, F) with U, F at (x, lam)."""
    k1, k2 = jax.random.split(key)
    x, U, F = state['x'], state['U'], state['F']
    if kind == 'hmc':
        p = draw_p(k1, m, kT, x.dtype)
    elif kind == 'ghmc':
        a = math.exp(-gamma * L * dt)
        p = a * state['p'] + math.sqrt(1.0 - a * a) * draw_p(k1, m, kT, x.dtype)
    else:
        raise ValueError(kind)
    H0 = U + kinetic(p, m)
    xn, pn, Un, Fn, ne = leapfrog(efn, x, p, F, lam, m, dt, L, box)
    H1 = Un + kinetic(pn, m)
    dH = H1 - H0
    log_a = jnp.where(jnp.isfinite(dH), -dH / kT, -jnp.inf)
    acc = jnp.log(jax.random.uniform(k2, (), x.dtype)) < log_a
    new = dict(x=jnp.where(acc, xn, x), U=jnp.where(acc, Un, U), F=jnp.where(acc, Fn, F),
               p=jnp.where(acc, pn, -p) if kind == 'ghmc' else jnp.where(acc, pn, p))
    return new, jnp.exp(jnp.minimum(log_a, 0.0)), dH, ne


def make_switch(efn, *, kind, m, dt, L, box, kT, gamma=None, nbins=20, record=False, probes=None):
    """Returns run(x0, p0, lams, keys) for ONE trajectory (vmap it).  lams: (K+1,) in the direction of travel.
    probes(x) -> dict of scalars recorded as running extrema (e.g. min solute-solvent distance, max O-H bond)."""
    def run(x0, p0, lams, keys):
        U0, F0, ne0 = efn(x0, lams[0])
        st = dict(x=x0, p=p0, U=U0, F=F0)
        mid = 0.5 * (lams[:-1] + lams[1:])
        bins = jnp.minimum((mid * nbins).astype(jnp.int32), nbins - 1)

        def stage(c, inp):
            st, W, Wb, nemax, nonf = c
            la, lb, b, key = inp
            Ub, Fb, ne1 = efn(st['x'], lb)
            dW = (Ub - st['U']) / kT
            nonf = nonf + (~jnp.isfinite(dW)).astype(jnp.int32)
            W = W + dW
            Wb = Wb.at[b].add(dW)
            st = dict(st, U=Ub, F=Fb)
            st, acc, dH, ne2 = mh_step(efn, st, lb, key, kind=kind, m=m, dt=dt, L=L, box=box, kT=kT, gamma=gamma)
            nemax = jnp.maximum(nemax, jnp.maximum(jnp.asarray(ne1).astype(jnp.int32), ne2))
            out = dict(acc=acc, dH=dH)
            if record:
                out['x_before_jump'] = c[0]['x']
            if probes is not None:
                out.update(probes(st['x']))
            return (st, W, Wb, nemax, nonf), out

        z = jnp.zeros((), x0.dtype)
        init = (st, z, jnp.zeros(nbins, x0.dtype), jnp.asarray(ne0).astype(jnp.int32), jnp.zeros((), jnp.int32))
        (st, W, Wb, nemax, nonf), out = jax.lax.scan(stage, init, (lams[:-1], lams[1:], bins, keys))
        return dict(W=W, Wbins=Wb, n_edges_max=nemax, nonfinite=nonf, x=st['x'], p=st['p'], U=st['U'], **out)
    return run


def stage_keys(seed, direction, root, K):
    k = jax.random.fold_in(jax.random.PRNGKey(seed), {'fwd': 1, 'rev': 2}[direction])
    k = jax.random.fold_in(k, root)
    return jax.random.split(k, K + 1)          # [0] for the initial momentum, [1:] for the stages


def schedule(K, direction):
    lam = np.linspace(0.0, 1.0, K + 1)
    return lam if direction == 'fwd' else lam[::-1].copy()


def min_image(d, box):
    return d - box * jnp.round(d / box)


def make_probes(solute_idx, solvent_idx, oh_pairs, box):
    """Running diagnostics per stage: min distance solute atom - solvent atom; max bonded O-H distance."""
    s = jnp.asarray(solute_idx)
    w = jnp.asarray(solvent_idx)
    o, h = jnp.asarray(oh_pairs[:, 0]), jnp.asarray(oh_pairs[:, 1])

    def probes(x):
        d = min_image(x[s][:, None, :] - x[w][None, :, :], box)
        dmin = jnp.sqrt((d ** 2).sum(-1).min())
        b = min_image(x[h] - x[o], box)
        return dict(min_solute_solvent=dmin, max_oh=jnp.sqrt((b ** 2).sum(-1).max()))
    return probes


def oh_pairs_from(Z, x, box, solute_mask):
    """Each solvent H is bonded to its nearest O (minimum image)."""
    Z = np.asarray(Z)
    sol = np.asarray(solute_mask, bool)
    O = np.where((Z == 8) & ~sol)[0]
    H = np.where((Z == 1) & ~sol)[0]
    d = x[H][:, None, :] - x[O][None, :, :]
    d -= box * np.round(d / box)
    j = np.argmin((d ** 2).sum(-1), axis=1)
    pairs = np.stack([O[j], H], 1)
    assert np.all(np.bincount(j, minlength=len(O)) == 2), 'water assignment is not 2 H per O'
    return pairs
