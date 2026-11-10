"""MACE backbone with two heads (energy/score and velocity) for FEAT-style transport.

One MACE backbone (mace-jax e8a6f2d), two heads:
  energy head   U_NN(x, t) = MACE's own energy readouts, exactly as MaceSolvation / MaceSolvationCueq in mace_solvation.py
                (same mace-jax configuration, same parameter names, same edge rule), so that
                U_theta(x, t) = t(1-t) U_NN(x, t) + U_LJ(x, LJsoft_theta(t))       (upstream stochastic_interpolant.U)
                and score = -grad_x U_theta.  U_theta(0) = U0 and U_theta(1) = U1 exactly.
  velocity head v_mode 'grad'   (default, plan 1.2): scalar phi(x, t), v = grad_x phi.
                  phi = sum_i MLP([invariant node features of every MACE layer, t features]_i).
                v_mode 'vector' (option): v_i = sigma * [u_i - mean_j u_j],
                  u_i = sum_{j in N(i)} m_ij (r_ij / |r_ij|),  m_ij = MLP([a_i, a_j, rbf(|r_ij|)]) * f_cut(|r_ij|),
                  a_i = Dense(invariant node features, t features)_i; r_ij the edge vector as MACE sees it.
                  O(3)-equivariant (1o), translation invariant, zero net drift.  Not a gradient field; the FB RND
                  estimate is exact for any v (FEAT prop. 3.3), so this changes only the variance.
Interface (one configuration x (N, 3) in the unit box, particle 0 the solute; t scalar or shape (1,)):
  MaceFeat.apply(params_U, x, t) -> (U_NN, phi) or (U_NN, v)
  FeatInterpolant (subclass of upstream stochastic_interpolant): .U(params, x, t) (as upstream, so upstream loss_fn and
      main.py work unchanged), .heads(params, x, t) -> (U_theta, phi | v), .fields_one(params, x, t) -> (U, grad U, v)
  make_path(params, model_cfg) -> dict with the keys of NES solvation_system.make_learned_path (energy,
      value_and_grad, delta, aux, ts, info, SI, params, nn) plus f2 (v_mode 'grad': x, t -> (U_theta, phi), for
      feat_core.fields_shared), fields (x, t -> (v, grad U), for feat_core.make_runner), velocity.
  fields: one backbone forward and two VJPs ('grad') or one VJP ('vector').
Options (model_cfg, none fixed yet; see README.zh-CN.md): size 'M' (2 layers) / 'M3' (3 layers); t_mode 'input' (t only
  through mace-jax's input node embedding, as in mace_solvation.py) / 'every' (also a zero-initialised additive embedding of
  the t features in the invariant channels before every later interaction); r_floor (soft core, sigma; 0 = off);
  backend 'cueq' / 'jax'; v_mode 'grad' / 'vector'.  Learning-rate presets: LR_PRESETS, make_optimizer.
Edge capacity: an overflow prints EDGE_CAPACITY_EXCEEDED and makes BOTH heads NaN, and, unlike MaceSolvation
  (jnp.where on the value only), also every entry of their gradients in x: each output is multiplied by
  nanify = where(overflow, nan, 1) and gets + (nanify - 1) * sum(x), which is exactly 0 without overflow.
mace-jax needs implicit rank promotion; the network is traced under jax.numpy_rank_promotion('allow'), so the global
  setting (upstream 'raise') can stay.
Needs the mace-jax venv and, on sys.path (inserted below unless already there), mace_solvation.py (same directory)
  (mace_solvation.py, read only) and the upstream solvation code (stochastic_interpolant, data.target_systems).
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Sequence

MACE_CODE = Path(__file__).resolve().parent
UPSTREAM = Path(os.environ.get('NTI_UPSTREAM', Path(__file__).resolve().parents[1] / 'upstream' / 'neuralTI-solvation'))
for _p in (UPSTREAM, UPSTREAM / 'src', MACE_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import flax.linen as nn  # noqa: E402
from flax import nnx  # noqa: E402
from flax.core import unfreeze  # noqa: E402
from mace_jax.modules.utils import prepare_graph  # noqa: E402

from mace_solvation import SIZES, _report_overflow, edge_rule, mace_config, template  # noqa: E402
from stochastic_interpolant import stochastic_interpolant  # noqa: E402
from data.target_systems import LJ3D_with_solute  # noqa: E402
from distance_on_torus import dist2_on_torus_matrix  # noqa: E402

PRESETS = {
    # energy head = MaceSolvationCueq, as trained with config_mace
    'solvation': dict(size='M', t_mode='input', r_floor=0.0, backend='cueq', v_mode='grad'),
    # settings that worked on the LJ fluid: 3 layers, soft core 0.5 sigma
    'lj-fluid': dict(size='M3', t_mode='input', r_floor=0.5, backend='cueq', v_mode='grad'),
}
LR_PRESETS = {   # optax.exponential_decay arguments; optimizer adam as upstream
    'upstream': dict(init_value=1e-3, decay_rate=0.1, transition_steps=10000, end_value=1e-5),  # config_mace_cueq.yaml
    'lj-fluid': dict(init_value=1e-2, decay_rate=0.1, transition_steps=10000, end_value=1e-4),  # 1e-2 -> 1e-4;
    # upstream schedule form with 10x the values
}
VERSION = 'mace_feat 2026-11-08'


def make_optimizer(name='upstream'):
    import optax
    return optax.adam(learning_rate=optax.exponential_decay(**LR_PRESETS[name]))


def irreps_dims(s):
    """(number of 0e channels, total dimension) of an irreps string such as '64x0e + 64x1o'; 0e must come first."""
    terms = [u.strip() for u in s.split('+')]
    assert terms[0].endswith('0e'), s
    n0 = tot = 0
    for u in terms:
        mul, ir = u.split('x')
        ell = int(ir[:-1])
        tot += int(mul) * (2 * ell + 1)
        n0 += int(mul) if ell == 0 else 0
    return n0, tot


def layer_dims(size):
    """Feature dimension of every MACE layer's output: hidden irreps, the last layer only its 0e part (mace-jax)."""
    n0, tot = irreps_dims(SIZES[size]['hidden_irreps'])
    L = SIZES[size]['num_interactions']
    return n0, [tot] * (L - 1) + [n0]


def mace_forward(model, data, layer_add):
    """mace-jax e8a6f2d MACE._energy_fn (modules/models.py, lines 409-595) for this use only (no LAMMPS, no comm, no
    pair repulsion, no embedding readout), with node_feats = layer_add(idx, node_feats) before interaction idx.
    Returns (energy, [node features of every layer]).  With layer_add = identity it is model(data) (CPU test C2)."""
    ctx = prepare_graph(data)
    arange, node_heads = ctx.num_atoms_arange, ctx.node_heads
    node_attrs = data['node_attrs']
    need_index = model.pair_repulsion or model.distance_transform in {'Agnesi', 'Soft'}
    cueq_config = getattr(model.equivariance_config, 'cueq_config', None)
    if cueq_config is not None and getattr(cueq_config, 'enabled', False):
        need_index = need_index or bool(getattr(cueq_config, 'optimize_all', False)
                                        or getattr(cueq_config, 'optimize_symmetric', False))
    node_attrs_index = jnp.argmax(node_attrs, axis=1).astype(jnp.int32) if need_index else None
    energy = model.atomic_energies_fn(node_attrs)[arange, node_heads].sum()
    node_feats = model.node_embedding(node_attrs)
    edge_attrs = model.spherical_harmonics(ctx.vectors)
    edge_feats, cutoff = model.radial_embedding(ctx.lengths, node_attrs, data['edge_index'], model._atomic_numbers,
                                                node_attrs_index=node_attrs_index)
    node_feats = node_feats + model.joint_embedding(data['batch'], {k: data[k] for k in model._embedding_names})
    feats = []
    for idx, (interaction, product) in enumerate(zip(model.interactions, model.products)):
        node_feats = layer_add(idx, node_feats)
        node_feats, sc = interaction(node_attrs=node_attrs, node_feats=node_feats, edge_attrs=edge_attrs,
                                     edge_feats=edge_feats, edge_index=data['edge_index'], cutoff=cutoff, n_real=None,
                                     first_layer=(idx == 0))
        node_feats = product(node_feats=node_feats, sc=sc, node_attrs=node_attrs, node_attrs_index=node_attrs_index)
        feats.append(node_feats)
    for idx, readout in enumerate(model.readouts):
        f = feats[-1 if len(model.readouts) == 1 else idx]
        energy = energy + readout(f, node_heads)[arange, node_heads].sum()
    return energy, feats


def poly_cutoff(u, p=6):
    """MACE's polynomial envelope (p = 6), 0 for u >= 1."""
    f = 1 - (p + 1) * (p + 2) / 2 * u ** p + p * (p + 2) * u ** (p + 1) - p * (p + 1) / 2 * u ** (p + 2)
    return jnp.where(u < 1, f, 0.)


class MaceFeat(nn.Module):
    target_system: Any
    size: str = 'M'
    cutoff: float = 0.222               # unit-box length, as U_GNN (2 sigma); solute edges reach 3 sigma
    size_to_pad: int = 16384            # edge capacity; an overflow makes both heads and their gradients NaN
    num_frequencies: int = 8
    avg_num_neighbors: float = 24.
    seed: int = 0
    r_floor: float = 0.                 # soft core (sigma, edge length as MACE sees it); 0 = off
    floor_sharpness: float = 20.
    backend: str = 'cueq'               # 'cueq' (= MaceSolvationCueq) or 'jax' (= MaceSolvation)
    layout: str = 'mul_ir'
    conv_fusion: bool = True
    t_mode: str = 'input'               # 'input' or 'every'
    v_mode: str = 'grad'                # 'grad' (v = grad phi) or 'vector' (1o output)
    head_hidden: Sequence[int] = (64, 64)
    edge_hidden: Sequence[int] = (64, 64)
    node_proj: int = 32
    n_rbf: int = 8

    def mace_cfg(self, r_max):
        cfg = mace_config(self.size, r_max, self.avg_num_neighbors, self.num_frequencies)
        if self.backend == 'cueq':       # identical to MaceSolvationCueq.mace_cfg
            cfg['equivariance_config'] = dict(backend='cueq', layout=self.layout, group='O3', optimize_all=True,
                                              conv_fusion=self.conv_fusion)
        elif self.backend != 'jax':
            raise ValueError(self.backend)
        return cfg

    def __call__(self, x, t):
        if self.t_mode not in ('input', 'every') or self.v_mode not in ('grad', 'vector'):
            raise ValueError((self.t_mode, self.v_mode))
        with jax.numpy_rank_promotion('allow'):
            return self.forward(x, t)

    @nn.compact
    def forward(self, x, t):
        ts = self.target_system
        dtype = x.dtype
        # ---- graph and mace-jax input: as MaceSolvation.__call__
        vec, inside, r_max = edge_rule(x, ts, self.cutoff)
        cfg = self.mace_cfg(r_max)
        graphdef, treedef, names, init, rest = template(json.dumps(cfg, sort_keys=True), self.seed,
                                                        bool(jax.config.jax_enable_x64))
        mp = self.param('mace', lambda key: {k: jnp.asarray(v, dtype) for k, v in init.items()})
        model = nnx.merge(graphdef, jax.tree_util.tree_unflatten(treedef, [mp[k] for k in names]), rest)
        n = len(x)
        n_edges = inside.sum()
        overflow = n_edges > self.size_to_pad
        jax.debug.callback(functools.partial(_report_overflow, capacity=self.size_to_pad), overflow, n_edges)
        send, recv = jnp.where(inside, size=self.size_to_pad, fill_value=-1)
        used = send >= 0
        send, recv = jnp.where(used, send, 0), jnp.where(used, recv, 0)
        far = jnp.array([[1.5 * r_max, 0., 0.]], dtype)          # unused slots: envelope exactly 0
        edge_vecs = jnp.where(used[:, None], vec[send, recv], far)
        if self.r_floor > 0:
            r = jnp.sqrt(jnp.maximum((edge_vecs ** 2).sum(-1, keepdims=True), 1e-24))
            kk = self.floor_sharpness
            r_eff = self.r_floor + jax.nn.softplus(kk * (r - self.r_floor)) / kk
            edge_vecs = jnp.where(used[:, None], edge_vecs * (r_eff / r), far)
        pos = x / ts.sigma
        shifts = edge_vecs - (pos[recv] - pos[send])              # MACE's vectors are exactly edge_vecs
        solute = (jnp.arange(n) == 0).astype(dtype)
        data = dict(positions=pos, node_attrs=jnp.stack([1. - solute, solute], -1), edge_index=jnp.stack([send, recv]),
                    shifts=shifts, cell=(jnp.eye(3, dtype=dtype) / ts.sigma)[None],
                    batch=jnp.zeros(n, jnp.int32), ptr=jnp.array([0, n]), head=jnp.array([0]))
        kf = jnp.arange(1, self.num_frequencies + 1, dtype=dtype)
        tt = jnp.reshape(t, ()).astype(dtype)
        tf = jnp.concatenate([tt[None], jnp.sin(jnp.pi * kf * tt), jnp.cos(jnp.pi * kf * tt)])
        data['t'] = jnp.broadcast_to(tf[None, :], (n, tf.shape[0]))

        # ---- backbone and energy head
        n0, dims = layer_dims(self.size)
        if self.t_mode == 'input':
            out = model(data, compute_force=False)
            U_nn = out['node_energy'].sum()
            nf = out['node_feats']
            assert nf.shape[-1] == sum(dims), (nf.shape, dims)
            offs = np.cumsum([0] + dims)
            feats = [nf[:, offs[i]:offs[i + 1]] for i in range(len(dims))]
        else:
            dense = [nn.Dense(n0, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros,
                              name=f't_layer{i}') for i in range(1, len(dims))]

            def layer_add(idx, f):
                if idx == 0:
                    return f
                return f.at[:, :n0].add(dense[idx - 1](tf)[None, :].astype(f.dtype))
            U_nn, feats = mace_forward(model, data, layer_add)
        scal = jnp.concatenate([f[:, :n0] for f in feats], -1)       # invariant channels of every layer
        h_in = jnp.concatenate([scal, jnp.broadcast_to(tf[None, :], (n, tf.shape[0]))], -1)

        # ---- velocity head
        if self.v_mode == 'grad':
            h = h_in
            for i, w in enumerate(self.head_hidden):
                h = nn.silu(nn.Dense(w, name=f'phi_dense{i}')(h))
            second = nn.Dense(1, name='phi_out')(h).sum()
        else:
            a = nn.Dense(self.node_proj, name='v_node')(h_in)
            r = jnp.sqrt(jnp.maximum((edge_vecs ** 2).sum(-1), 1e-24))
            mu = jnp.linspace(0., r_max, self.n_rbf, dtype=dtype)
            rbf = jnp.exp(-((r[:, None] - mu[None, :]) / (r_max / self.n_rbf)) ** 2)
            m = jnp.concatenate([a[send], a[recv], rbf.astype(a.dtype)], -1)
            for i, w in enumerate(self.edge_hidden):
                m = nn.silu(nn.Dense(w, name=f'v_edge{i}')(m))
            m = nn.Dense(1, name='v_edge_out')(m)[:, 0] * poly_cutoff(r / r_max) * used
            u = jax.ops.segment_sum((m / r)[:, None] * edge_vecs, send, num_segments=n)
            second = (u - u.mean(0, keepdims=True)) * ts.sigma        # zero net drift; sigma -> box units
        nanify = jnp.where(overflow, jnp.nan, 1.).astype(dtype)
        poison = (nanify - 1) * x.sum()      # exactly 0 (value and gradient) without overflow; NaN in every x entry
        return U_nn * nanify + poison, second * nanify + poison


class FeatInterpolant(stochastic_interpolant):
    """Upstream stochastic_interpolant with the two-head U_model; .U is upstream's formula on the energy head.
    U_model may be set afterwards, as upstream main.py does (cfg.model._target_: mace_feat.FeatInterpolant,
    cfg.U_model._target_: mace_feat.MaceFeat), so the energy head can also be trained with upstream main.py (TSM)."""

    def __init__(self, target_system, U_model=None, loss_t_weight=1e-8, loss_x_weight='lambda t: 1e-3'):
        super().__init__(target_system, loss_t_weight, loss_x_weight)
        self.num_features = target_system.num_dim
        self.U_model = U_model

    def heads(self, params, x, t):
        t = jnp.reshape(t, (1,))
        U_NN, second = self.U_model.apply(params['U'], x, t)
        LJsoft = self.getLJ_softening(params['LJ'], t)
        U_base = self.target_system.U(x, LJsoft)
        return ((t * (1 - t)) * U_NN.reshape() + U_base).reshape(), second

    def U(self, params, x, t):
        return self.heads(params, x, t)[0]

    def fields_one(self, params, x, t):
        """(U_theta, grad_x U_theta, v) for one configuration: one forward, two VJPs ('grad') or one ('vector')."""
        (U, b), vjp = jax.vjp(lambda y: self.heads(params, y, t), x)
        gU, = vjp((jnp.ones_like(U), jnp.zeros_like(b)))
        if self.U_model.v_mode == 'vector':
            return U, gU, b
        gphi, = vjp((jnp.zeros_like(U), jnp.ones_like(b)))
        return U, gU, gphi


def build(model_cfg, target_system=None):
    ts = target_system or LJ3D_with_solute()
    return FeatInterpolant(ts, MaceFeat(target_system=ts, **model_cfg))


def init_params(model_cfg, key):
    """Fresh parameters {'U', 'LJ'} with upstream's init_params (47 random particles, keys split as upstream)."""
    SI = build(model_cfg)
    SI.init_params(key)
    return unfreeze(SI.params)                          # plain nested dicts (editable, e.g. warm_start_from_tsm)


def count_params(params):
    return int(sum(np.size(v) for v in jax.tree_util.tree_leaves(params)))


def load_tsm_checkpoint(path):
    """Upstream pickle {params, opt_state, cfg} of a MaceSolvation(Cueq) TSM run (e.g. mace-cueq-seed0)."""
    raw = Path(path).read_bytes()
    ck = pickle.loads(raw)
    return ck, hashlib.sha256(raw).hexdigest()


def warm_start_from_tsm(path, model_cfg, key):
    """New two-head parameters whose energy head and LJ softening are those of a MaceSolvation(Cueq) checkpoint.
    The velocity head (and t_mode 'every' layers, zero-initialised) are fresh, so U_theta equals the checkpoint's."""
    ck, sha = load_tsm_checkpoint(path)
    cu = ck['cfg'].U_model
    target = str(cu['_target_'])
    want = 'mace_solvation_cueq.MaceSolvationCueq' if model_cfg.get('backend', 'cueq') == 'cueq' \
        else 'mace_solvation.MaceSolvation'
    checks = dict(target=(target, want), size=(cu.get('size', 'M'), model_cfg.get('size', 'M')),
                  r_floor=(float(cu.get('r_floor', 0.)), float(model_cfg.get('r_floor', 0.))),
                  cutoff=(float(cu.get('cutoff', .222)), float(model_cfg.get('cutoff', .222))),
                  num_frequencies=(int(cu.get('num_frequencies', 8)), int(model_cfg.get('num_frequencies', 8))),
                  avg_num_neighbors=(float(cu.get('avg_num_neighbors', 24.)),
                                     float(model_cfg.get('avg_num_neighbors', 24.))))
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise ValueError(f'checkpoint and model_cfg differ: {bad}')
    params = jax.tree_util.tree_map(np.asarray, init_params(model_cfg, key))
    mace_ck = ck['params']['U']['params']['mace']
    assert set(mace_ck) == set(params['U']['params']['mace']), 'mace parameter names differ'
    params['U']['params']['mace'] = {k: np.asarray(v) for k, v in mace_ck.items()}
    params['LJ'] = jax.tree_util.tree_map(np.asarray, ck['params']['LJ'])
    return params, dict(checkpoint=str(path), sha256=sha)


def _pair_sum(x, Uij_fn):
    """target_system.U's reduction for an arbitrary pair matrix (as NES solvation_system._pair_sum)."""
    n = x.shape[0]
    eye = jnp.eye(n, dtype=x.dtype)
    r2 = dist2_on_torus_matrix(x) + eye
    mask = (1 - eye) * 0.5
    return 0.5 * (Uij_fn(r2) * mask).sum()


def make_path(params, model_cfg, dtype=None, name='mace-feat', info=None):
    """Path dict for feat_core / nes_core (keys as NES solvation_system.make_learned_path, plus f2, fields, velocity).
    params are cast to dtype (default: float64 if x64 is on)."""
    SI = build(model_cfg)
    ts = SI.target_system
    dtype = dtype or (jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)
    params = jax.tree_util.tree_map(lambda v: jnp.asarray(v, dtype=dtype), params)
    mdl = SI.U_model

    def t1(t):
        return jnp.reshape(t, (1,))

    def energy(x, t):
        return SI.U(params, x, t)

    def nn_(x, t):
        return mdl.apply(params['U'], x, t1(t))[0].reshape()

    def delta(x, ta, tb):
        la = SI.getLJ_softening(params['LJ'], t1(ta))
        lb = SI.getLJ_softening(params['LJ'], t1(tb))
        base = _pair_sum(x, lambda r2: ts.U_ij(r2=r2, a=lb) - ts.U_ij(r2=r2, a=la))
        return base + tb * (1 - tb) * nn_(x, tb) - ta * (1 - ta) * nn_(x, ta)

    def edge_count(x):
        with jax.numpy_rank_promotion('allow'):
            return edge_rule(x, ts, mdl.cutoff)[1].sum().astype(x.dtype)

    def fields(x, t):
        _, gU, v = SI.fields_one(params, x, t)
        return v, gU

    def velocity(x, t):
        return fields(x, t)[0]

    f2 = (lambda x, t: SI.heads(params, x, t)) if mdl.v_mode == 'grad' else None
    inf = dict(version=VERSION, model_cfg=dict(model_cfg), n_params=count_params(params), size_to_pad=mdl.size_to_pad,
               params_dtype=str(jnp.dtype(dtype)), layer_dims=layer_dims(mdl.size)[1])
    inf.update(info or {})
    return dict(name=name, energy=energy, value_and_grad=jax.value_and_grad(energy), delta=delta, aux=edge_count,
                ts=ts, info=inf, SI=SI, params=params, nn=nn_, f2=f2, fields=fields, velocity=velocity)
