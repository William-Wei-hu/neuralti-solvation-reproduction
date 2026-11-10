"""MACE (mace-jax) energy network for the solvation NeuralTI code (balintmate/neuralTI-solvation f756b30).

Drop-in replacement for src/U_GNN.U_model: module.apply(params, x, t) with x (N, 3) in the unit periodic box (particle
0 is the solute) and t (1,), returns the scalar U_NN(x, t). stochastic_interpolant.U forms t(1-t) U_NN + U_base, so the
end points stay the exact U0 and U1 whatever the network does.

Derived from 2026-11-04-NeuralTI-MACE-100k/code/mace_energy.py (LJ, one species). Changes:
* Two species: node_attrs is one-hot [solvent, solute], as U_GNN's "is solute" node feature.
* Length unit sigma_A: positions are L*x with L = 1/target_system.sigma (box 9 sigma); r_max = cutoff*L (2 sigma for
  U_GNN's cutoff 0.222). Edge vectors with the solute are shortened by sigma/sigma_AB before MACE sees them, so the
  solute's neighbours reach 2 sigma_AB = 3 sigma. This is U_GNN's edge rule (its D2 is divided by (sigma_AB/sigma)^2
  in the solute's row and column), so both networks build the same graph.
* Edge capacity: U_GNN keeps size_to_pad edges and silently drops the rest. Here an overflow prints
  EDGE_CAPACITY_EXCEEDED (host callback, only when it happens) and returns NaN, so with the configuration's
  jax_debug_nans=True a run stops instead of training on a truncated graph. checkify is not used because the upstream
  training loop does not functionalise it.
* mace-jax relies on implicit rank promotion (mace_jax/modules/radial.py: bessel_weights (8,) * lengths (E, 1)), so
  the solvation configuration's jax_numpy_rank_promotion='raise' must be 'allow' with this network (config_mace.yaml).
* t enters through mace-jax's continuous per-node embedding of [t, sin(pi k t), cos(pi k t)], k = 1..num_frequencies
  (input layer only), as MaceEnergy(t_embed=True).
* Optional soft core r_floor (sigma), as MaceEnergy: an edge shorter than r_floor (as MACE sees it) is smoothly raised to
  r_floor. 0 = off.
MACE's vectors are pos[recv] - pos[send] + shifts. The shifts here are differentiable (edge_vecs - position difference),
so MACE's vectors equal edge_vecs exactly and the gradients in x are those of edge_vecs.
Needs the mace-jax environment (jax 0.10, flax 0.12 nnx; mace-jax e8a6f2d) and, on PYTHONPATH, the solvation code's
src/ (distance_on_torus.dR_on_torus_matrix).
"""
import functools
import json
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import nnx
from mace_jax.tools.model_builder import _build_jax_model

from distance_on_torus import dR_on_torus_matrix   # solvation src: dR[i, j] = x_i - x_j, minimum image

SIZES = {   # as mace_energy.py
    'S': dict(hidden_irreps='32x0e + 32x1o', max_ell=2, correlation=2, num_interactions=2, num_bessel=8,
              MLP_irreps='16x0e', radial_MLP=[32, 32, 32]),
    'M': dict(hidden_irreps='64x0e + 64x1o', max_ell=3, correlation=3, num_interactions=2, num_bessel=8,
              MLP_irreps='16x0e', radial_MLP=[64, 64, 64]),
}
SIZES['M3'] = dict(SIZES['M'], num_interactions=3)


def mace_config(size, r_max, avg_num_neighbors, num_frequencies):
    cfg = dict(r_max=float(r_max), num_polynomial_cutoff=6, radial_type='bessel',
               interaction_cls='RealAgnosticResidualInteractionBlock',
               interaction_cls_first='RealAgnosticInteractionBlock',
               atomic_numbers=[18, 54], atomic_energies=[0., 0.],   # labels only: [solvent, solute]
               gate='silu', avg_num_neighbors=float(avg_num_neighbors),
               embedding_specs={'t': dict(type='continuous', per='node', emb_dim=32, in_dim=2*num_frequencies+1)})
    cfg.update(SIZES[size])
    return cfg


def _name(path):
    parts = []
    for k in path:
        for attr in ('key', 'idx', 'name'):
            if hasattr(k, attr):
                parts.append(str(getattr(k, attr)))
                break
        else:
            parts.append(str(k))
    return '.'.join(parts)


@functools.lru_cache(maxsize=None)
def template(cfg_json, seed, x64):
    """(graphdef, param treedef, param names, initial params, constant state); x64 fixes the build dtype."""
    del x64                                             # cache key only: mace-jax builds in the default dtype
    with jax.ensure_compile_time_eval():
        model = _build_jax_model(json.loads(cfg_json), rngs=nnx.Rngs(seed))
        graphdef, params, rest = nnx.split(model, nnx.Param, ...)
        leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
        names = [_name(p) for p, _ in leaves]
        assert len(set(names)) == len(names), 'parameter names collide'
        init = {k: np.asarray(v) for k, (_, v) in zip(names, leaves)}
    return graphdef, treedef, tuple(names), init, rest


def _report_overflow(overflow, required, capacity):
    """Host-side print, only when an overflow happens (a lax.cond print would fire for every vmapped sample)."""
    if np.any(overflow):
        print(f'EDGE_CAPACITY_EXCEEDED required={int(np.max(required))} capacity={capacity}', flush=True)


def edge_rule(x, target_system, cutoff):
    """Edge vectors as MACE sees them (sigma units, solute edges shortened) and the edge mask (no self loops)."""
    L = 1./target_system.sigma
    r_max = cutoff*L
    n = len(x)
    solute = jnp.arange(n) == 0
    vec = -L*dR_on_torus_matrix(x)                              # vec[i, j] = x_j - x_i
    with_solute = solute[:, None] | solute[None, :]
    scale = jnp.where(with_solute, target_system.sigma/target_system.sigma_AB, 1.).astype(x.dtype)
    vec = vec*scale[:, :, None]
    r2 = (vec**2).sum(-1)
    inside = (r2 < r_max**2) & ~jnp.eye(n, dtype=bool)
    return vec, inside, r_max


class MaceSolvation(nn.Module):
    target_system: Any
    size: str = 'M'
    cutoff: float = 0.222               # unit-box length, as U_GNN (2 sigma)
    size_to_pad: int = 16384            # edge capacity; an overflow returns NaN
    num_frequencies: int = 8
    avg_num_neighbors: float = 24.      # solvent neighbours within 2 sigma at 512/9^3 sigma^-3
    seed: int = 0
    r_floor: float = 0.                 # soft core in sigma; 0 = off
    floor_sharpness: float = 20.        # 1/sigma

    @nn.compact
    def __call__(self, x, t):
        dtype = x.dtype
        vec, inside, r_max = edge_rule(x, self.target_system, self.cutoff)
        cfg = mace_config(self.size, r_max, self.avg_num_neighbors, self.num_frequencies)
        graphdef, treedef, names, init, rest = template(json.dumps(cfg, sort_keys=True), self.seed,
                                                        bool(jax.config.jax_enable_x64))
        mp = self.param('mace', lambda key: {k: jnp.asarray(v, dtype) for k, v in init.items()})
        params = jax.tree_util.tree_unflatten(treedef, [mp[k] for k in names])
        model = nnx.merge(graphdef, params, rest)

        n = len(x)
        n_edges = inside.sum()
        overflow = n_edges > self.size_to_pad
        jax.debug.callback(functools.partial(_report_overflow, capacity=self.size_to_pad), overflow, n_edges)
        send, recv = jnp.where(inside, size=self.size_to_pad, fill_value=-1)
        used = send >= 0
        send, recv = jnp.where(used, send, 0), jnp.where(used, recv, 0)
        far = jnp.array([[1.5*r_max, 0., 0.]], dtype)          # unused slots: envelope is exactly 0 there
        edge_vecs = jnp.where(used[:, None], vec[send, recv], far)
        if self.r_floor > 0:
            r = jnp.sqrt(jnp.maximum((edge_vecs**2).sum(-1, keepdims=True), 1e-24))
            k = self.floor_sharpness
            r_eff = self.r_floor + jax.nn.softplus(k*(r - self.r_floor))/k
            edge_vecs = jnp.where(used[:, None], edge_vecs*(r_eff/r), far)
        pos = x/self.target_system.sigma
        shifts = edge_vecs - (pos[recv] - pos[send])            # differentiable: MACE's vectors are edge_vecs
        solute = (jnp.arange(n) == 0).astype(dtype)
        data = dict(positions=pos, node_attrs=jnp.stack([1. - solute, solute], -1), edge_index=jnp.stack([send, recv]),
                    shifts=shifts, cell=(jnp.eye(3, dtype=dtype)/self.target_system.sigma)[None],
                    batch=jnp.zeros(n, jnp.int32), ptr=jnp.array([0, n]), head=jnp.array([0]))
        k = jnp.arange(1, self.num_frequencies + 1, dtype=dtype)
        tt = jnp.reshape(t, ()).astype(dtype)
        tf = jnp.concatenate([tt[None], jnp.sin(jnp.pi*k*tt), jnp.cos(jnp.pi*k*tt)])
        data['t'] = jnp.broadcast_to(tf[None, :], (n, tf.shape[0]))
        out = model(data, compute_force=False)
        energy = out['node_energy'].sum()
        return jnp.where(overflow, jnp.nan, energy)
