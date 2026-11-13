"""MACE-OFF energy interface with 4D decoupling of the solute (methane in water).

Loads a MACE-OFF model converted from mace-torch to a mace-jax bundle (see convert_mace_off.py); batched call with one
lambda per configuration; edge-capacity overflow (n_edges >= pad) makes E_total, E_int and every force component NaN
(in addition to returning n_edges and pad); masses, E0 and a system builder.

Units: positions Angstrom, energies eV, forces eV/Angstrom, masses amu.  Cubic box of side L (Angstrom); positions
may be wrapped or not (every pair uses the minimum image; needs r_max < L/2).
JAX settings (the CALLER sets them before importing mace_jax / this module):
    jax.config.update('jax_enable_x64', True); jax.config.update('jax_default_matmul_precision', 'highest')
Environment: mace-jax (commit e8a6f2d was used) and, for the conversion only, mace-torch 0.3.17.
Bundle location: MaceIface(..., bundle=DIR), else $MACE_OFF_BUNDLE_DIR/<model>, else bundles/<model> (relative to the
current directory).  The bundle holds config.json and params.msgpack as written by convert_mace_off.py.

API
  sysinfo = build_system(n_water, seed, density=0.997)        -> dict(Z, masses, solute, L, pos0, notes)   numpy only
  m = MaceIface('off23s', sysinfo['Z'], sysinfo['L'], sysinfo['solute'], pad=None, backend='cueq', ref_pos=pos0)
  E, Eint, F, n_edges = m.e_and_f(pos[B, N, 3], lam[B])        jitted, vmapped; one lambda per configuration
  m.pad (int), m.r_max, m.info (model, bundle path, sha256, pad, backend)
  overflow: n_edges >= m.pad  -> E, Eint, F are NaN for that configuration (never silently truncated)
  pos_relaxed, log = m.relax(pos[B, N, 3], lam=0.0, fmax=1.0, max_steps=200)    FIRE, runs the model (GPU)
  Ep, Fp = m.parts(pos[B, N, 3])        E(solvent alone) + E(solute alone) and the combined forces, lam = 0 each
4D decoupling (Picha et al., JPCL 16 (2025), doi:10.1021/acs.jpclett.5c02980, eq. 1b, with eps multiplied by lambda
so lambda = 0 is exact):
  solute-solvent edges r -> r + lam * (max(0, r_max - r) + eps), direction unchanged.  lam = 0 coupled, lam = 1
  decoupled (all solute-solvent edges beyond r_max where MACE's envelope is exactly 0).
Identities checked on the GPU before production (values measured in our runs in brackets):
  I1  E - Eint = sum of E0(Z), the same for every configuration and lambda   [|dE0| = 0 between Torch and JAX]
  I2  lam = 1: E(lam=1) = Ep (parts), F(lam=1) = Fp                       [5.8e-11 eV, 7e-15 eV/A]
  I3  F = -dE/dx: central difference along a random unit vector, h = 1e-4 A, absolute error <= 1e-6 eV/A
      (the f64 rounding floor of a 4.5e5 eV total is ~1e-6 eV/A; relative criteria are not meaningful)  [<= 2.2e-7]
  I4  periodic translation x -> x + c (all atoms): E unchanged to <= 1e-8 eV, F unchanged to <= 1e-8 eV/A
  I5  lam = 0 equals mace-torch MACECalculator to ~3e-4 eV / 7e-5 eV/A (not within 1e-5 eV / 1e-6 eV/A;
      cause not found; the configuration-dependent part is <= 0.003 kBT) - informational, not a pass criterion
  I6  pad: setting pad below n_edges gives NaN E, Eint, F and n_edges >= pad
Stress / virial: not implemented (one extra JVP, d/ds E(s*pos, s*L) at s = 1, would give the trace of the virial).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

MODELS = ('off23s', 'off24m')                     # MACE-OFF23 small, MACE-OFF24 medium (only off23s was used)
MASS = {1: 1.008, 6: 12.011, 8: 15.999}          # amu, IUPAC conventional atomic weights
KBT_298 = 0.0256926                              # eV at 298.15 K
AMU_A2_PER_FS2_IN_EV = 103.6427                  # 1 amu A^2 / fs^2 = 103.6427 eV


# ------------------------------------------------------------------------------------------------ builder (numpy)
def _rot(rng):
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def build_system(n_water, seed, density=0.997, exclude=3.2, tries=30):
    """CH4 at the box centre + n_water rigid-geometry waters on random sites of the smallest n^3 grid that leaves
    enough sites farther than `exclude` A from the centre; orientation = best of `tries` random rotations (largest
    minimum distance to atoms already placed).  Atom order: C, H, H, H, H, then (O, H, H) per water.
    Independent chains: different seeds.  Not relaxed: call MaceIface.relax (runs the model)."""
    rng = np.random.default_rng(seed)
    mass = n_water * 18.015 + 16.043
    L = (mass / (density * 0.60221408)) ** (1 / 3)
    c = np.full(3, L / 2)
    n = 2
    while True:
        s = L / n
        sites = (np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing='ij'), -1).reshape(-1, 3) + 0.5) * s
        cand = sites[np.linalg.norm(sites - c, axis=1) > exclude]
        if len(cand) >= n_water:
            break
        n += 1
    sel = cand[rng.choice(len(cand), n_water, replace=False)]
    tet = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    pos = [c] + [c + 1.087 * v for v in tet]
    Z = [6, 1, 1, 1, 1]
    a = np.deg2rad(104.52) / 2
    hloc = 0.9572 * np.array([[np.sin(a), 0, np.cos(a)], [-np.sin(a), 0, np.cos(a)]])

    def mind(new, old):
        dd = new[:, None, :] - np.asarray(old)[None, :, :]
        dd -= L * np.round(dd / L)
        return np.sqrt((dd ** 2).sum(-1)).min()
    for o in sel:
        best = None
        for _ in range(tries):
            h = o + hloc @ _rot(rng).T
            m = mind(np.vstack([o, h]), pos)
            if best is None or m > best[0]:
                best = (m, h)
        pos += [o, best[1][0], best[1][1]]
        Z += [8, 1, 1]
    Z = np.array(Z)
    pos = np.array(pos) % L
    solute = np.arange(len(Z)) < 5
    d = pos[None] - pos[:, None]
    d -= L * np.round(d / L)
    r = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(r, np.inf)
    return dict(Z=Z, masses=np.array([MASS[int(z)] for z in Z]), solute=solute, L=float(L), pos0=pos,
                notes=dict(n_water=n_water, seed=seed, density=density, grid=n, n_sites=len(cand),
                           min_pair=float(r.min()), builder='mace_iface.build_system (numpy, not relaxed)'))


def bundle_dir(model, bundle=None):
    """Directory with config.json and params.msgpack of a converted model."""
    if bundle is not None:
        return Path(bundle)
    if model not in MODELS:
        raise ValueError(f'unknown model {model}')
    return Path(os.environ.get('MACE_OFF_BUNDLE_DIR', 'bundles')) / model


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ interface
class MaceIface:
    def __init__(self, model, Z, L, solute, pad=None, backend='cueq', ref_pos=None, pad_factor=1.25, pad_extra=256,
                 eps=1e-3, bundle=None):
        import jax
        if not jax.config.jax_enable_x64:
            raise RuntimeError('enable jax_enable_x64 before creating MaceIface (everything runs in float64)')
        from mace_jax.tools import bundle as B
        d = bundle_dir(model, bundle)
        if backend == 'jax':
            b = B.load_model_bundle(str(d), 'float64')
        else:
            cfg = json.loads((d / 'config.json').read_text())
            cfg['equivariance_config'] = dict(backend='cueq', layout='mul_ir', group='O3', optimize_all=True,
                                              conv_fusion=True)
            b = B._restore_bundle(cfg, (d / 'params.msgpack').read_bytes(), 'float64', context=str(d))
        self.Z = np.asarray(Z)
        self.L = float(L)
        self.solute = np.asarray(solute, bool)
        self.masses = np.array([MASS[int(z)] for z in self.Z])
        self.r_max = float(b.config['r_max'])
        if not self.r_max < self.L / 2:
            raise ValueError(f'r_max {self.r_max} must be < L/2 = {self.L / 2}')
        if pad is None:
            if ref_pos is None:
                raise ValueError('give pad or ref_pos')
            pad = int(pad_factor * self.count_edges(np.asarray(ref_pos))) + pad_extra
        self.pad = int(pad)
        self.eps = float(eps)
        self._bundle = b
        self.info = dict(model=model, bundle=str(d), sha256={f: sha256(d / f) for f in ('config.json', 'params.msgpack')},
                         backend=backend, pad=self.pad, r_max=self.r_max, L=self.L, n_atoms=len(self.Z),
                         n_solute=int(self.solute.sum()), license='Academic Software Licence (ASL), non-commercial')
        self._one = self._make_one(self.Z, self.solute, self.pad)
        self.e_and_f = jax.jit(jax.vmap(self._one, in_axes=(0, 0)))
        self._parts = None

    def count_edges(self, pos):
        """numpy edge count (directed pairs within r_max, minimum image) of one configuration."""
        d = pos[None] - pos[:, None]
        d -= self.L * np.round(d / self.L)
        r2 = (d ** 2).sum(-1)
        np.fill_diagonal(r2, np.inf)
        return int((r2 < self.r_max ** 2).sum())

    def _make_one(self, Z, solute_mask, pad):
        import jax
        import jax.numpy as jnp
        b, L, r_max, eps = self._bundle, self.L, self.r_max, self.eps
        zlist = [int(z) for z in b.config['atomic_numbers']]
        idx = jnp.asarray(np.array([zlist.index(int(z)) for z in Z], np.int32))
        n = len(Z)
        solute = jnp.asarray(np.asarray(solute_mask, bool))
        apply = b.graphdef.apply(b.params)

        def energy(pos, lam):
            dt = pos.dtype
            node_attrs = jax.nn.one_hot(idx, len(zlist), dtype=dt)
            d = pos[None, :, :] - pos[:, None, :]                       # d[i, j] = x_j - x_i
            d = d - L * jnp.round(d / L)
            r2 = (d ** 2).sum(-1)
            inside = (r2 < r_max ** 2) & ~jnp.eye(n, dtype=bool)
            n_edges = inside.sum()
            send, recv = jnp.where(inside, size=pad, fill_value=-1)
            used = send >= 0
            send, recv = jnp.where(used, send, 0), jnp.where(used, recv, 0)
            vec = d[send, recv]
            r = jnp.sqrt(jnp.maximum((vec ** 2).sum(-1), 1e-30))
            cross = solute[send] != solute[recv]
            r_new = r + lam * (jnp.maximum(0., r_max - r) + eps)
            vec = vec * jnp.where(cross & used, r_new / r, 1.)[:, None]
            vec = jnp.where(used[:, None], vec, jnp.array([[1.5 * r_max, 0., 0.]], dt))
            shifts = vec - (pos[recv] - pos[send])
            data = dict(positions=pos, node_attrs=node_attrs, node_attrs_index=idx,
                        edge_index=jnp.stack([send, recv]).astype(jnp.int32), shifts=shifts,
                        batch=jnp.zeros(n, jnp.int32), ptr=jnp.array([0, n], jnp.int32),
                        cell=(L * jnp.eye(3, dtype=dt))[None], unit_shifts=jnp.zeros_like(shifts),
                        head=jnp.array([0], jnp.int32))
            out, _ = apply(data, compute_force=False)
            # overflow: NaN in the values and, through `poison`, in every gradient component
            nanify = jnp.where(n_edges >= pad, jnp.nan, 1.).astype(dt)
            poison = (nanify - 1) * pos.sum()
            E = out['energy'].reshape(()) * nanify + poison
            Eint = out['interaction_energy'].reshape(()) * nanify + poison
            return E, (Eint, n_edges)

        def one(pos, lam):
            (E, (Eint, ne)), g = jax.value_and_grad(energy, has_aux=True)(pos, lam)
            return E, Eint, -g, ne
        return one

    def parts(self, pos):
        """E(solvent alone) + E(solute alone) and combined forces (lam = 0 for each part); same box.  For identity I2."""
        import jax
        import jax.numpy as jnp
        if self._parts is None:
            ms, mu = ~self.solute, self.solute
            pad_s = self.pad
            pad_u = int(mu.sum() * (mu.sum() - 1)) + 8
            fs = jax.jit(jax.vmap(self._make_one(self.Z[ms], np.zeros(ms.sum(), bool), pad_s), in_axes=(0, 0)))
            fu = jax.jit(jax.vmap(self._make_one(self.Z[mu], np.ones(mu.sum(), bool), pad_u), in_axes=(0, 0)))
            self._parts = (fs, fu)
        fs, fu = self._parts
        pos = jnp.asarray(pos)
        z = jnp.zeros(pos.shape[0], pos.dtype)
        Es, _, Fs, _ = fs(pos[:, ~self.solute], z)
        Eu, _, Fu, _ = fu(pos[:, self.solute], z)
        F = jnp.zeros_like(pos).at[:, ~self.solute].set(Fs).at[:, self.solute].set(Fu)
        return Es + Eu, F

    def relax(self, pos, lam=0.0, fmax=1.0, max_steps=200, dt=0.1, dt_max=1.0, max_step=0.1):
        """FIRE (Bitzek et al. 2006) on a batch of configurations with the model (runs the model).
        Returns (pos, log) with log = dict(steps, fmax per configuration, E per configuration, converged)."""
        import jax.numpy as jnp
        x = np.array(pos, np.float64)
        B = x.shape[0]
        lam_b = jnp.full(B, lam, jnp.float64)
        v = np.zeros_like(x)
        dts = np.full(B, dt)
        alpha = np.full(B, 0.1)
        n_pos = np.zeros(B, int)
        for step in range(max_steps):
            E, _, F, ne = (np.asarray(o) for o in self.e_and_f(jnp.asarray(x), lam_b))
            if not np.all(np.isfinite(F)):
                raise FloatingPointError(f'non-finite force at FIRE step {step} (n_edges {ne.max()} pad {self.pad})')
            fm = np.abs(F).max(axis=(1, 2))
            if np.all(fm < fmax):
                break
            P = (F * v).sum(axis=(1, 2))
            fn = np.sqrt((F ** 2).sum(axis=(1, 2)))
            vn = np.sqrt((v ** 2).sum(axis=(1, 2)))
            v = (1 - alpha)[:, None, None] * v + (alpha * vn / np.maximum(fn, 1e-30))[:, None, None] * F
            up = P > 0
            n_pos = np.where(up, n_pos + 1, 0)
            grow = up & (n_pos > 5)
            dts = np.where(grow, np.minimum(dts * 1.1, dt_max), np.where(up, dts, dts * 0.5))
            alpha = np.where(grow, alpha * 0.99, np.where(up, alpha, 0.1))
            v = np.where(up[:, None, None], v, 0.)
            v = v + dts[:, None, None] * F
            dx = dts[:, None, None] * v
            dn = np.sqrt((dx ** 2).sum(-1, keepdims=True))
            dx = dx * np.minimum(1., max_step / np.maximum(dn, 1e-30))
            x = (x + dx) % self.L
        return x, dict(steps=step, fmax=fm.tolist(), E=E.tolist(), converged=bool(np.all(fm < fmax)))


def check_waters_intact(Z, pos, L, oh_max=1.5):
    """Number of O with != 2 H within oh_max A, and whether C keeps 4 H within 1.5 A (numpy)."""
    d = pos[None] - pos[:, None]
    d -= L * np.round(d / L)
    r = np.sqrt((d ** 2).sum(-1))
    O = np.where(Z == 8)[0]
    H = np.where(Z == 1)[0]
    bad_O = int(sum((r[o, H] < oh_max).sum() != 2 for o in O))
    c_ok = bool((r[0, H] < 1.5).sum() == 4) if Z[0] == 6 else None
    return dict(bad_waters=bad_O, methane_intact=c_ok)
