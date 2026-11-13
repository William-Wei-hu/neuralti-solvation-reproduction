"""LJ experiment for the local radial escort map (training, NES arms, analysis).  One GPU, about 0.7 GPU-h on an A100.
  NTI_NES_RUNS=runs python escort/lj_arms.py OUTDIR

Inputs under $NTI_NES_RUNS (default runs/), produced by the nes/ scripts:
  endpoints/end{0,1}.npz            nes/sample_endpoints.py (estimation roots x, chain_id and pilot chains x_pilot)
  base-hmc-G2400-M1/summary.json    nes/analyze_nes.py on the HMC base-path run with --grad-budget 2400 --M 1
                                    (its dissipation profile defines the densified grid)
  base-hmc-G{2400,24000}-M1/{fwd,rev}.npz   optional: the baseline arms are compared with these saved works

Stages
  S1  training samples from the PILOT chains only (never the estimation roots 0..255):
      (a) annealed HMC along the training grid t_g = g/40 (both directions), n_frames per chain per t;
      (b) endpoint samples: each pilot chain at its own end, n_end_frames frames (for the one-step map).
  S2  training (final parameters used; no selection on validation):
      learned-KL     stage-wise forward + reverse KL on training-grid stages (t_g -> t_{g+1})
      learned-BDHD   Baron & Panagiotopoulos (arXiv 2606.25229): Bhattacharyya distance for half the steps, then a
                     per-stage BAR estimate on the training samples, then the harmonic-overlap loss for the other half
      control        one scalar a, BD-HD (same recipe)
      one-step       MNF-style single map t = 0 -> 1 (24 free logits), BD-HD on the endpoint samples
      Stage maps theta = dt * net(t_mid): trained on stages of width 1/40, applied with dt = 1/K (linear scaling).
  S3  arms on the estimation roots 0..255 (seed 7110, as the NES baseline); one-step estimates on the same roots
  S4  analysis and the criteria fixed before the run (see s4_analysis)
Work conventions (beta = 1): stage forward  W~F = U_b(Phi(x)) - U_a(x) - ln det;  reverse (raw) W~R = U_a(Phi^-1(y))
  - U_b(y) + ln det(Phi^-1(y)).  BD = -ln<exp(-W~F/2)>_a - ln<exp(-W~R/2)>_b (dF cancels);
  HD(dF) = -ln<2/(1+exp(W~F - dF))>_a - ln<2/(1+exp(W~R + dF))>_b.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'nes'))   # nes_core, solvation_system, estimators
import jax  # noqa: E402

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from jax.scipy.special import logsumexp  # noqa: E402

import escort_map as EM  # noqa: E402
import escort_nes as EN  # noqa: E402
import estimators as ES  # noqa: E402
import nes_core as C  # noqa: E402
import solvation_system as S  # noqa: E402

NES_RUNS = Path(os.environ.get('NTI_NES_RUNS', 'runs'))
TRUTH, TRUTH_SE = -20.365, 0.013
H, L_HMC, M = 2e-6, 20, 1
R_C, B = 0.35, 8
SEED_EST, SEED_SAMPLES, SEED_NET, SEED_TRAIN = 7110, 20261111, 0, 20261112
PROFILE = NES_RUNS / 'base-hmc-G2400-M1' / 'summary.json'

SC = dict(n_roots=256, n_grid=41, pilot_chains=16, n_eq=400, n_frames=4, n_gap=100, n_end_frames=32,
          holdout=4, steps_mlp=3000, steps_ctl=1000, steps_one=1000, batch_g=4, batch_x=32, K_main=114,
          K_extra=1143, chunk=57, extra_budget_s=2400)


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def save_json(p, obj):
    tmp = Path(str(p) + '.tmp')
    tmp.write_text(json.dumps(ES.json_safe(obj), indent=1))
    tmp.replace(p)


def kern_fn(path):
    return jax.jit(jax.vmap(lambda x, t, k: C.kernel_M(x, t, H, k, 1, kind='hmc', energy=path['energy'],
                                                       value_and_grad=path['value_and_grad'], wrap=C.wrap_torus,
                                                       cap_factor=None, L=L_HMC), in_axes=(0, None, 0)))


# ------------------------------------------------------------------------------------------------ S1 samples
def s1_samples(path, sc, out):
    kern = kern_fn(path)
    T = np.linspace(0, 1, sc['n_grid'])
    P = sc['pilot_chains']
    samples = np.zeros((len(T), 2 * P * sc['n_frames'], 513, 3))
    chain_of = np.zeros(samples.shape[1], int)
    ends = np.zeros((2, P * sc['n_end_frames'], 513, 3))
    ends_chain = np.zeros((2, ends.shape[1]), int)
    acc_log = np.zeros((2, len(T)))
    base = jax.random.PRNGKey(SEED_SAMPLES)
    for side, (end, order) in enumerate(((0, range(len(T))), (1, range(len(T) - 1, -1, -1)))):
        with np.load(NES_RUNS / 'endpoints' / f'end{end}.npz') as z:
            x0 = jnp.asarray(z['x_pilot'][:P])
        step = 0
        # (b) endpoint frames at the chain's own end
        x = x0
        tend = jnp.float64(float(end))
        for s in range(sc['n_end_frames'] * sc['n_gap']):
            keys = jax.random.split(jax.random.fold_in(jax.random.fold_in(base, 10 + side), step), P)
            x, _ = kern(x, tend, keys)
            step += 1
            if (s + 1) % sc['n_gap'] == 0:
                f = (s + 1) // sc['n_gap'] - 1
                ends[end, f * P:(f + 1) * P] = np.asarray(x)
                ends_chain[end, f * P:(f + 1) * P] = np.arange(P)
        # (a) annealing along the grid, restarting from the pilot frames
        x = x0
        step = 0
        for ti in order:
            t = jnp.float64(T[ti])
            accs = []
            for s in range(sc['n_eq'] + sc['n_frames'] * sc['n_gap']):
                keys = jax.random.split(jax.random.fold_in(jax.random.fold_in(base, side), step), P)
                x, a = kern(x, t, keys)
                accs.append(np.asarray(a))
                step += 1
                s2 = s - sc['n_eq'] + 1
                if s2 > 0 and s2 % sc['n_gap'] == 0:
                    f = s2 // sc['n_gap'] - 1
                    j0 = side * P * sc['n_frames'] + f * P
                    samples[ti, j0:j0 + P] = np.asarray(x)
                    chain_of[j0:j0 + P] = side * P + np.arange(P)
            acc_log[side, ti] = float(np.mean(accs))
        log('S1 side', side, 'done; mean acceptance', float(acc_log[side].mean()))
    np.savez(out / 'train_samples.npz', samples=samples, t=T, chain=chain_of, acc=acc_log, ends=ends,
             ends_chain=ends_chain)
    return dict(samples=samples, T=T, chain_of=chain_of, ends=ends, ends_chain=ends_chain)


# ------------------------------------------------------------------------------------------------ losses
def stage_works(U, kind, params, xa, ya, ta, tb):
    """Mapped works of one stage (ta < tb) for batches xa ~ pi_ta, ya ~ pi_tb."""
    th = EM.stage_theta(kind, params, ta, tb, B)

    def wf(x):
        y, ld = EM.apply_map(x, th, R_C, B)
        return U(y, tb) - U(x, ta) - ld

    def wr(y):
        x, ld = EM.apply_inverse(y, th, R_C, B)
        return U(x, ta) - U(y, tb) + ld
    return jax.vmap(wf)(xa), jax.vmap(wr)(ya)


def loss_kl(WF, WR):
    return jnp.mean(WF) + jnp.mean(WR)


def loss_bd(WF, WR):
    n_f, n_r = WF.shape[0], WR.shape[0]
    return -(logsumexp(-WF / 2) - jnp.log(n_f)) - (logsumexp(-WR / 2) - jnp.log(n_r))


def loss_hd(WF, WR, dF):
    return -(logsumexp(jnp.log(2.0) - jnp.logaddexp(0.0, WF - dF)) - jnp.log(WF.shape[0])) \
        - (logsumexp(jnp.log(2.0) - jnp.logaddexp(0.0, WR + dF)) - jnp.log(WR.shape[0]))


# ------------------------------------------------------------------------------------------------ S2 training
def s2_train(path, sc, D, out):
    U = path['energy']
    T = D['T']
    G = len(T) - 1
    P = sc['pilot_chains']
    train_mask = (D['chain_of'] % P) < P - sc['holdout']
    idx_train = np.where(train_mask)[0]
    idx_val = np.where(~train_mask)[0]
    X = jnp.asarray(D['samples'])
    rng = np.random.default_rng(SEED_TRAIN)

    def batch_loss(kind, lossname, params, gs, xa, ya, dFs):
        def one(g, a, b, dF):
            WF, WR = stage_works(U, kind, params, a, b, jnp.asarray(T)[g], jnp.asarray(T)[g + 1])
            if lossname == 'kl':
                return loss_kl(WF, WR)
            if lossname == 'bd':
                return loss_bd(WF, WR)
            return loss_hd(WF, WR, dF)
        return jnp.mean(jax.vmap(one)(gs, xa, ya, dFs))

    def per_stage_bar(kind, params):
        dfs = np.zeros(G)
        f = jax.jit(lambda p, a, b, ta, tb: stage_works(U, kind, p, a, b, ta, tb))
        for g in range(G):
            WF, WR = f(params, X[g, idx_train], X[g + 1, idx_train], jnp.float64(T[g]), jnp.float64(T[g + 1]))
            dfs[g] = ES.bar(np.asarray(WF), np.asarray(WR))
        return jnp.asarray(dfs)

    def train(kind, p0, steps, lr, recipe):
        opt = optax.adam(lr)
        st = opt.init(p0)
        p = p0
        hist = []
        dFs = jnp.zeros(G)
        phases = [('kl', steps)] if recipe == 'kl' else [('bd', steps // 2), ('hd', steps - steps // 2)]
        for lossname, n in phases:
            if lossname == 'hd':
                dFs = per_stage_bar(kind, p)
            vg = jax.jit(jax.value_and_grad(lambda pp, gs, xa, ya, d: batch_loss(kind, lossname, pp, gs, xa, ya, d)))
            for it in range(n):
                gs = rng.integers(0, G, sc['batch_g'])
                ia = rng.choice(idx_train, (sc['batch_g'], sc['batch_x']))
                ib = rng.choice(idx_train, (sc['batch_g'], sc['batch_x']))
                val, gr = vg(p, jnp.asarray(gs), X[gs[:, None], ia], X[gs[:, None] + 1, ib], dFs[gs])
                upd, st = opt.update(gr, st, p)
                p = optax.apply_updates(p, upd)
                hist.append((lossname, float(val)))
        # validation (report only): mean KL and BD over all stages on held-out chains
        v = {}
        for lname in ('kl', 'bd'):
            f = jax.jit(lambda pp, gs, xa, ya: batch_loss(kind, lname, pp, gs, xa, ya, jnp.zeros(len(gs))))
            iv = idx_val[:8]
            gs = np.arange(G)
            v[lname] = float(f(p, jnp.asarray(gs), X[gs[:, None], iv[None, :]], X[gs[:, None] + 1, iv[None, :]]))
            v[lname + '_identity'] = float(f(p0, jnp.asarray(gs), X[gs[:, None], iv[None, :]],
                                             X[gs[:, None] + 1, iv[None, :]]))
        return p, dict(history=hist, validation=v)

    results, info = {}, {}
    t0 = time.time()
    for name, kind, p0, steps, lr, recipe in (
            ('learned-KL', 'mlp', EM.init_mlp(jax.random.PRNGKey(SEED_NET), B), sc['steps_mlp'], 3e-3, 'kl'),
            ('learned-BDHD', 'mlp', EM.init_mlp(jax.random.PRNGKey(SEED_NET), B), sc['steps_mlp'], 3e-3, 'bdhd'),
            ('control', 'control', dict(a=jnp.zeros((), jnp.float64)), sc['steps_ctl'], 1e-1, 'bdhd')):
        p, inf = train(kind, p0, steps, lr, recipe)
        results[name] = jax.device_get(p)
        info[name] = inf
        log('S2', name, 'last loss', inf['history'][-1] if inf['history'] else None, 'val', inf['validation'],
            f'{time.time() - t0:.0f}s')
    # one-step MNF-style map t = 0 -> 1 on the endpoint samples (BD half, BAR, HD half)
    E = jnp.asarray(D['ends'])
    hold = D['ends_chain'][0] >= P - sc['holdout']
    it_tr = np.where(~hold)[0]
    p = dict(theta=jnp.zeros(EM.n_logits(B), jnp.float64))
    opt = optax.adam(3e-2)
    st = opt.init(p)
    hist = []
    for lossname, n in (('bd', sc['steps_one'] // 2), ('hd', sc['steps_one'] - sc['steps_one'] // 2)):
        dF = 0.0
        if lossname == 'hd':
            WF, WR = jax.jit(lambda pp: stage_works(U, 'fixed', pp, E[0, it_tr], E[1, it_tr], 0.0, 1.0))(p)
            dF = ES.bar(np.asarray(WF), np.asarray(WR))
        vg = jax.jit(jax.value_and_grad(lambda pp, a, b: (lambda w: loss_bd(*w) if lossname == 'bd' else
                                                           loss_hd(*w, dF))(stage_works(U, 'fixed', pp, a, b, 0.0, 1.0))))
        for it in range(n):
            ia = rng.choice(it_tr, min(64, len(it_tr)))
            ib = rng.choice(it_tr, min(64, len(it_tr)))
            val, gr = vg(p, E[0, ia], E[1, ib])
            upd, st = opt.update(gr, st, p)
            p = optax.apply_updates(p, upd)
            hist.append((lossname, float(val)))
    results['one-step'] = jax.device_get(p)
    info['one-step'] = dict(history=hist)
    log('S2 one-step done', hist[-1] if hist else None)
    with open(out / 'trained_params.pkl', 'wb') as fh:
        pickle.dump(results, fh)
    save_json(out / 'train_history.json', info)
    return results


# ------------------------------------------------------------------------------------------------ S3 arms
def arms_list(sc):
    K = sc['K_main']
    a = [('baseline', None, 'uniform', K), ('learned-BDHD', 'mlp', 'uniform', K), ('learned-KL', 'mlp', 'uniform', K),
         ('control', 'control', 'densified', K), ('diag_densify_only', None, 'densified', K)]
    if sc['K_extra']:
        Kx = sc['K_extra']
        a += [('baseline', None, 'uniform', Kx), ('learned-BDHD', 'mlp', 'uniform', Kx),
              ('learned-KL', 'mlp', 'uniform', Kx)]
    return a


def load_roots(sc, end):
    with np.load(NES_RUNS / 'endpoints' / f'end{end}.npz') as z:
        return z['x'][:sc['n_roots']], z['chain_id'][:sc['n_roots']]


def s3_arms(path, sc, params, out):
    res = {}
    for arm, mk, grid_kind, K in arms_list(sc):
        name = f'{arm}-K{K}'
        if K != sc['K_main'] and time.time() - T0 > sc['extra_budget_s']:
            log('S3 time gate: skipping', name)
            res.setdefault('_skipped', []).append(name)
            continue
        runner = EN.make_escort_runner(path=path, kind='hmc', M=M, L=L_HMC, nbins=100, map_kind=mk, r_c=R_C, B=B)
        p = None if mk is None else params[arm]
        r = {}
        for direction, end in (('fwd', 0), ('rev', 1)):
            X, ch = load_roots(sc, end)
            keys = jax.vmap(lambda c: C.traj_key(SEED_EST, direction, c, 0))(jnp.asarray(ch))
            t_dir = C.grid(K, direction) if grid_kind == 'uniform' else EN.densified_grid(K, PROFILE, direction)
            bins = C.bin_index(t_dir, 100)
            Bn = X.shape[0]
            st = dict(x=np.asarray(X), W=np.zeros(Bn), Wbins=np.zeros((Bn, 100)), auxmax=np.zeros(Bn),
                      nonfinite=np.zeros(Bn, np.int32), acc=np.zeros((Bn, K), np.float32), LD=np.zeros(Bn))
            times, k0 = [], 0
            while k0 < K:
                k1 = min(K, k0 + sc['chunk'])
                t1 = time.time()
                o = runner(p, jnp.asarray(st['x']), jnp.asarray(st['W']), jnp.asarray(st['Wbins']),
                           jnp.asarray(st['auxmax']), jnp.asarray(st['nonfinite']), keys,
                           jnp.asarray(t_dir[k0:k1 + 1]), jnp.asarray(np.full(k1 - k0 + 1, H)),
                           jnp.asarray(bins[k0:k1]), jnp.arange(k0, k1))
                x, W, Wb, am, nf, acc, LD = jax.device_get(o)
                times.append((k1 - k0, time.time() - t1))
                st.update(x=x, W=W, Wbins=Wb, nonfinite=nf, LD=st['LD'] + LD)
                st['acc'][:, k0:k1] = acc
                k0 = k1
            steady = [s / n for n, s in times[1:]] or [times[0][1] / times[0][0]]
            np.savez(out / f'{name}-{direction}.npz', W=st['W'], Wbins=st['Wbins'], LD=st['LD'], acc=st['acc'],
                     nonfinite=st['nonfinite'], root_id=ch, t_dir=t_dir, chunk_times=np.array(times))
            r[direction] = dict(W=st['W'], nonfinite=int(st['nonfinite'].sum()), acc_mean=float(st['acc'].mean()),
                                s_per_stage_steady=float(np.median(steady)), root_id=ch, calls_per_traj=K * 22)
            log('S3', name, direction, 'mean W', float(st['W'].mean()), 'acc', float(st['acc'].mean()))
        res[name] = r
    # one-step estimates on the same roots: identity (plain two-state BAR) and the trained one-step map
    U = path['energy']
    X0, c0 = load_roots(sc, 0)
    X1, c1 = load_roots(sc, 1)
    for name, p in (('one-step-identity', dict(theta=jnp.zeros(EM.n_logits(B), jnp.float64))),
                    ('one-step-MNF', params['one-step'])):
        f = jax.jit(lambda pp, a, b: stage_works(U, 'fixed', pp, a, b, 0.0, 1.0))
        WF, WR = (np.asarray(v) for v in f(p, jnp.asarray(X0), jnp.asarray(X1)))
        np.savez(out / f'{name}.npz', WF=WF, WR=WR)
        res[name] = dict(fwd=dict(W=WF, nonfinite=int((~np.isfinite(WF)).sum()), acc_mean=np.nan,
                                  s_per_stage_steady=np.nan, root_id=c0, calls_per_traj=2),
                         rev=dict(W=WR, nonfinite=int((~np.isfinite(WR)).sum()), acc_mean=np.nan,
                                  s_per_stage_steady=np.nan, root_id=c1, calls_per_traj=2))
        log('S3', name, 'mean WF', float(WF.mean()), 'mean WR', float(WR.mean()))
    return res


# ------------------------------------------------------------------------------------------------ S4 analysis
def s4_analysis(sc, res, out):
    A = dict(truth=TRUTH, truth_se=TRUTH_SE, arms={}, skipped_by_time_gate=res.pop('_skipped', []))
    rng = np.random.default_rng(0)
    for name, r in res.items():
        wf, wr = r['fwd']['W'], r['rev']['W']
        try:
            s = ES.summarize(wf, wr, r['fwd']['root_id'], r['rev']['root_id'], n_boot=2000, seed=0, truth=TRUTH)
            bar, se = s['point']['BAR'], s['bootstrap']['BAR']['se']
            extra = dict(overlap=s['point']['logistic_overlap_sum'], ess_F=s['point']['ess_forward'],
                         ess_R=s['point']['ess_reverse'], dissipation_F=s['point']['dissipation_forward'],
                         dissipation_R=s['point']['dissipation_reverse'])
        except Exception as e:  # e.g. BAR not bracketed for a hopeless one-step estimate
            bar, se, extra = float('nan'), float('nan'), dict(error=str(e))
        diss = 0.5 * (wf.mean() + wr.mean())
        bs = [0.5 * (wf[rng.integers(0, len(wf), len(wf))].mean() + wr[rng.integers(0, len(wr), len(wr))].mean())
              for _ in range(2000)]
        nf = r['fwd']['nonfinite'] + r['rev']['nonfinite']
        A['arms'][name] = dict(BAR=bar, BAR_se=se, mean_dissipation=diss, mean_dissipation_se=float(np.std(bs, ddof=1)),
                               nonfinite=nf, acc=0.5 * (r['fwd']['acc_mean'] + r['rev']['acc_mean']),
                               s_per_stage=0.5 * (r['fwd']['s_per_stage_steady'] + r['rev']['s_per_stage_steady']),
                               calls_per_traj=r['fwd']['calls_per_traj'],
                               rigor=bool(np.isfinite(bar) and abs(bar - TRUTH) <= 3 * np.sqrt(se ** 2 + TRUTH_SE ** 2)
                                          and nf == 0), **extra)
    a = A['arms']
    K = sc['K_main']
    b, c = a.get(f'baseline-K{K}'), a.get(f'control-K{K}')
    crit = dict(R_rigor_NES_arms=all(v['rigor'] for k, v in a.items() if not k.startswith('one-step')))
    for lv in ('learned-BDHD', 'learned-KL'):
        l = a.get(f'{lv}-K{K}')
        if b and c and l:
            crit[lv] = dict(G1=l['mean_dissipation'] <= 0.5 * b['mean_dissipation'],
                            G2=l['mean_dissipation'] <= 0.7 * c['mean_dissipation'],
                            overhead=l['s_per_stage'] / b['s_per_stage'] - 1.0,
                            O=l['s_per_stage'] / b['s_per_stage'] - 1.0 <= 0.05,
                            ratio_vs_baseline=l['mean_dissipation'] / b['mean_dissipation'],
                            ratio_vs_control=l['mean_dissipation'] / c['mean_dissipation'])
    if b and c:
        crit['control_vs_baseline'] = c['mean_dissipation'] / b['mean_dissipation']
    # one-step vs the best NES arm at K_main (same roots)
    one = a.get('one-step-MNF')
    nes = [v for k, v in a.items() if k.endswith(f'-K{K}') and v['rigor'] and np.isfinite(v['BAR_se'])]
    if one and nes:
        best = min(nes, key=lambda v: v['BAR_se'])
        crit['one_step'] = dict(rigor=one['rigor'], se_one_step=one['BAR_se'], se_best_nes=best['BAR_se'],
                                J_not_needed=bool(one['rigor'] and one['BAR_se'] <= best['BAR_se']),
                                J_doubtful=bool(one['rigor'] and one['BAR_se'] <= 2 * best['BAR_se']))
    A['criteria'] = crit
    rep = {}
    for g, Kb in (('2400', 114), ('24000', 1143)):
        nm = f'baseline-K{Kb}'
        if nm in res and all((NES_RUNS / f'base-hmc-G{g}-M1' / f'{d}.npz').exists() for d in ('fwd', 'rev')):
            for d in ('fwd', 'rev'):
                with np.load(NES_RUNS / f'base-hmc-G{g}-M1' / f'{d}.npz') as z:
                    Wn = z['W'][:len(res[nm][d]['W'])]
                rep[f'{nm}-{d}'] = float(np.abs(Wn - res[nm][d]['W']).max())
    A['baseline_vs_NES_saved_max_abs_dW'] = rep
    save_json(out / 'analysis.json', A)
    log('S4', json.dumps(ES.json_safe(crit)))
    return A


T0 = time.time()


def main():
    out = Path(sys.argv[1])
    sc = SC
    out.mkdir(parents=True, exist_ok=False)
    print('devices', jax.devices(), 'jax', jax.__version__, flush=True)
    save_json(out / 'meta.json', dict(sc=sc, H=H, L=L_HMC, M=M, R_C=R_C, B=B,
                                      seeds=dict(est=SEED_EST, samples=SEED_SAMPLES, net=SEED_NET, train=SEED_TRAIN),
                                      nes_runs=str(NES_RUNS)))
    path = S.make_base_path()
    try:
        D = s1_samples(path, sc, out)
        params = s2_train(path, sc, D, out)
        res = s3_arms(path, sc, params, out)
        s4_analysis(sc, res, out)
    except Exception:
        traceback.print_exc()
        return 1
    log('DONE')
    return 0


if __name__ == '__main__':
    sys.exit(main())
