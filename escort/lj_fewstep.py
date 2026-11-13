"""Few-stage scan of the local radial escort on LJ, and the K = 114 map + densified-grid arm.  One GPU.
  NTI_NES_RUNS=runs python escort/lj_fewstep.py ARMS_DIR OUTDIR

ARMS_DIR is the output directory of lj_arms.py, re-used here:
  train_samples.npz (pilot-chain samples at t = g/40), trained_params.pkl (learned-BDHD MLP, used only at K = 114),
  and the K = 114, M = 1 arms baseline / diag_densify_only / learned-BDHD (W of the same 256 + 256 roots).
Inputs under $NTI_NES_RUNS as for lj_arms.py.
Stages
  S1  pilot-chain samples at the 1/80 midpoints needed by the snapped densified grids (start: the lj_arms.py frames of the
      neighbouring 1/40 point, side 0 from below, side 1 from above; n_eq_mid HMC steps, then 4 frames 100 apart)
  I   identity check: per M, zero-logit table vs the baseline runner, first stage, 4 roots per direction:
      max|dW| <= 1e-12 and max|dx| <= 1e-12 (hard criterion)
  S2  per (K, grid): free logits per stage (K x 24), zero init, BD-HD (MNF recipe: BD half, per-stage BAR on the
      training samples, HD half), all stages every step, Adam; final parameters used
  S3  arms (priority order, time gate): link (baseline K114 M1 rerun) -> accounting (ii) -> accounting (i) -> K114 map+dens
  S4  analysis and the criteria fixed before the run (see s4)
"""
from __future__ import annotations

import json
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

import escort_map as EM  # noqa: E402
import escort_nes_v3 as EN  # noqa: E402
import estimators as ES  # noqa: E402
import lj_arms as LA  # noqa: E402   (helpers only: stage_works, losses, kern_fn, load_roots; its main is not run)
import nes_core as C  # noqa: E402
import solvation_system as S  # noqa: E402

V2 = None                     # output directory of lj_arms.py (ARMS_DIR), set in main
TRUTH, TRUTH_SE = LA.TRUTH, LA.TRUTH_SE
H, L_HMC, R_C, B = LA.H, LA.L_HMC, LA.R_C, LA.B
SEED_EST, SEED_MID, SEED_TRAIN = 7110, 20261113, 20261114
CALLS_REF = 114 * 22          # 2508 = lj_arms.py K = 114, M = 1 (convention: K (2 + L M))
LOW_OVERLAP = 0.05
M_II = {5: 25, 10: 12, 20: 6, 40: 3}  # accounting (ii): M = round((2508 / K - 2) / 20)

SC = dict(n_roots=256, Ks=(5, 10, 20, 40), n_eq_mid=200, n_gap=100, steps=600, batch_x=16, lr=1e-2,
          icheck_roots=4, gate_s=2400, K114=True, chunk114=57)
P, NF, HOLD = 16, 4, 4        # layout of train_samples.npz: 2 sides x 16 chains x 4 frames; chains >= 12 held out


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def save_json(p, obj):
    tmp = Path(str(p) + '.tmp')
    tmp.write_text(json.dumps(ES.json_safe(obj), indent=1))
    tmp.replace(p)


# ------------------------------------------------------------------------------------------------ grids
def uniform80(K):
    return np.arange(K + 1) * (80 // K)


def dens80(K):
    j = np.round(EN.densified_grid(K, LA.PROFILE, 'fwd') * 80).astype(int)
    assert len(np.unique(j)) == K + 1 and j[0] == 0 and j[-1] == 80 and np.all(np.diff(j) > 0), (K, j)
    return j


def grid_j(K, grid):
    return uniform80(K) if grid == 'uni' else dens80(K)


def t_of(j, direction):
    t = np.asarray(j, float) / 80.0
    return t if direction == 'fwd' else t[::-1].copy()


# ------------------------------------------------------------------------------------------------ S1 midpoints
def s1_midpoints(path, sc, out, needed):
    with np.load(V2 / 'train_samples.npz') as z:
        v2 = z['samples']
    kern = LA.kern_fn(path)
    base = jax.random.PRNGKey(SEED_MID)
    mid = {}
    for j in needed:
        arr = np.zeros((2 * P * NF, 513, 3))
        for side in (0, 1):
            g = (j - 1) // 2 if side == 0 else (j + 1) // 2     # neighbour below (side 0) / above (side 1)
            j0 = side * P * NF + (NF - 1) * P
            x = jnp.asarray(v2[g, j0:j0 + P])
            t = jnp.float64(j / 80.0)
            step = 0
            for s in range(sc['n_eq_mid'] + NF * sc['n_gap']):
                keys = jax.random.split(jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(base, side), j), step), P)
                x, _ = kern(x, t, keys)
                step += 1
                s2 = s - sc['n_eq_mid'] + 1
                if s2 > 0 and s2 % sc['n_gap'] == 0:
                    f = s2 // sc['n_gap'] - 1
                    k0 = side * P * NF + f * P
                    arr[k0:k0 + P] = np.asarray(x)
        mid[j] = arr
        log('S1 midpoint j', j, 'done')
    np.savez(out / 'mid_samples.npz', **{f'j{j}': v for j, v in mid.items()})
    samples = {2 * g: v2[g] for g in range(41)}
    samples.update(mid)
    return samples


# ------------------------------------------------------------------------------------------------ S2 training
def s2_train(path, sc, samples, K, grid):
    U = path['energy']
    j = grid_j(K, grid)
    T = j / 80.0
    chain = (np.arange(2 * P * NF) % P)
    itr, ival = np.where(chain < P - HOLD)[0], np.where(chain >= P - HOLD)[0]
    X = jnp.asarray(np.stack([samples[int(v)] for v in j]))           # (K+1, 128, 513, 3)
    rng = np.random.default_rng(SEED_TRAIN + 1000 * K + (grid == 'dens'))
    Ta, Tb = jnp.asarray(T[:-1]), jnp.asarray(T[1:])

    def works(th, xa, ya, ta, tb):
        return LA.stage_works(U, 'fixed', dict(theta=th), xa, ya, ta, tb)

    vw = jax.vmap(works)

    def loss(th, xa, ya, dF, name):
        WF, WR = vw(th, xa, ya, Ta, Tb)
        if name == 'bd':
            return jnp.mean(jax.vmap(LA.loss_bd)(WF, WR))
        return jnp.mean(jax.vmap(LA.loss_hd)(WF, WR, dF))

    w1 = jax.jit(works)          # one stage at a time (memory: ~10 MB per configuration with gradients)

    def stage_bar(th, idx):
        out = []
        for k in range(K):
            WF, WR = jax.device_get(w1(th[k], X[k, idx], X[k + 1, idx], Ta[k], Tb[k]))
            out.append(ES.bar(WF, WR))
        return np.array(out)

    th = jnp.zeros((K, EM.n_logits(B)), jnp.float64)
    opt = optax.adam(sc['lr'])
    st = opt.init(th)
    hist = []
    dF = jnp.zeros(K)
    t0 = time.time()
    for name, n in (('bd', sc['steps'] // 2), ('hd', sc['steps'] - sc['steps'] // 2)):
        if name == 'hd':
            dF = jnp.asarray(stage_bar(th, itr))
        vg = jax.jit(jax.value_and_grad(lambda p, a, b, d: loss(p, a, b, d, name)))
        for it in range(n):
            ia = rng.choice(itr, (K, sc['batch_x']))
            ib = rng.choice(itr, (K, sc['batch_x']))
            v, gr = vg(th, X[np.arange(K)[:, None], ia], X[np.arange(K)[:, None] + 1, ib], dF)
            upd, st = opt.update(gr, st, th)
            th = optax.apply_updates(th, upd)
            hist.append((name, float(v)))
    # validation (report only): per-stage BD on the held-out chains, trained vs identity
    def val_bd(p):
        return np.array([float(LA.loss_bd(*w1(p[k], X[k, ival], X[k + 1, ival], Ta[k], Tb[k]))) for k in range(K)])
    bd_tr, bd_id = val_bd(th), val_bd(jnp.zeros_like(th))
    info = dict(K=K, grid=grid, j=j.tolist(), seconds=time.time() - t0, last=hist[-1], history=hist[::10],
                val_bd_trained=bd_tr.tolist(), val_bd_identity=bd_id.tolist(),
                val_bd_trained_sum=float(bd_tr.sum()), val_bd_identity_sum=float(bd_id.sum()),
                max_abs_logit=float(jnp.max(jnp.abs(th))))
    log('S2', K, grid, 'val BD sum trained', info['val_bd_trained_sum'], 'identity', info['val_bd_identity_sum'],
        'max|logit|', info['max_abs_logit'], f"{info['seconds']:.0f}s")
    return np.asarray(th), info


def table_from_mlp(params, t_fwd):
    """lj_arms.py learned-BDHD map as a table along the forward grid: theta_k = dt_k * net(t_mid_k) (same formula)."""
    ta, tb = jnp.asarray(t_fwd[:-1]), jnp.asarray(t_fwd[1:])
    return np.asarray(jax.vmap(lambda a, b: EM.stage_theta('mlp', params, a, b, B))(ta, tb))


# ------------------------------------------------------------------------------------------------ S3 arms
RUNNERS = {}


def runner(M, mk):
    if (M, mk) not in RUNNERS:
        RUNNERS[(M, mk)] = EN.make_escort_runner(path=PATH, kind='hmc', M=M, L=L_HMC, nbins=100, map_kind=mk,
                                                 r_c=R_C, B=B)
    return RUNNERS[(M, mk)]


def run_dir(sc, M, table_fwd, t_fwd, direction, n_roots, chunk, stages=None):
    end = 0 if direction == 'fwd' else 1
    X, ch = LA.load_roots(dict(n_roots=n_roots), end)
    keys = jax.vmap(lambda c: C.traj_key(SEED_EST, direction, c, 0))(jnp.asarray(ch))
    t_dir = t_fwd if direction == 'fwd' else t_fwd[::-1].copy()
    K = len(t_dir) - 1 if stages is None else stages
    t_dir = t_dir[:K + 1]
    mk = None if table_fwd is None else 'table'
    p = None if table_fwd is None else jnp.asarray(table_fwd if direction == 'fwd' else table_fwd[::-1].copy())
    bins = C.bin_index(t_dir, 100)
    Bn = X.shape[0]
    st = dict(x=np.asarray(X), W=np.zeros(Bn), Wbins=np.zeros((Bn, 100)), auxmax=np.zeros(Bn),
              nonfinite=np.zeros(Bn, np.int32), acc=np.zeros((Bn, K), np.float32), LD=np.zeros(Bn))
    times, k0 = [], 0
    run = runner(M, mk)
    while k0 < K:
        k1 = min(K, k0 + chunk)
        t1 = time.time()
        o = run(p, jnp.asarray(st['x']), jnp.asarray(st['W']), jnp.asarray(st['Wbins']), jnp.asarray(st['auxmax']),
                jnp.asarray(st['nonfinite']), keys, jnp.asarray(t_dir[k0:k1 + 1]), jnp.asarray(np.full(k1 - k0 + 1, H)),
                jnp.asarray(bins[k0:k1]), jnp.arange(k0, k1))
        x, W, Wb, am, nf, acc, LD = jax.device_get(o)
        times.append((k1 - k0, time.time() - t1))
        st.update(x=x, W=W, Wbins=Wb, nonfinite=nf, LD=st['LD'] + LD)
        st['acc'][:, k0:k1] = acc
        k0 = k1
    st.update(root_id=ch, t_dir=t_dir, chunk_times=np.array(times))
    return st


def icheck(sc, Ms):
    """Identity check per M: zero table vs baseline runner, first stage, both directions.
    Criterion revised once, after a dry run and before the production run:
      original: max|dW| <= 1e-12 and max|dx| <= 1e-12 after the M HMC steps of the first stage, for every M.
        dry run: dW 1.3e-15 / 4.5e-13 for every M; dx after M steps 7.8e-16 (M=1), 1.5e-13 / 4.0e-13 (M=12),
        5.5e-11 / 4.2e-10 (M=25) -> FAIL at M=25 (chaotic growth inside the stage).
      revised: max|dW| <= 1e-12 for every M (unchanged);  max|dx| after the FIRST HMC step of the first stage <= 1e-12
        for every M.  The first step of an M-step stage is the M = 1 stage: both kernels (nes_core.kernel_M and
        escort_nes_v3.kernel_from) start from value_and_grad at the same point and use key fold_in(stage_key, m) with
        m = 0 for the first step, so the state after step 1 does not depend on M; it is measured with the M = 1 runners
        (reading it inside an M-step runner would need changes to the runner code).  dx after M steps: report only,
        with a check that it grows monotonically with M (also report only)."""
    out, ok = {}, True
    dx_after = {}
    for M in sorted(set(Ms) | {1}):
        for direction in ('fwd', 'rev'):
            t_fwd = np.array([0.0, 1.0 / 80, 2.0 / 80]) if direction == 'fwd' else np.array([78.0, 79.0, 80.0]) / 80
            z = np.zeros((2, EM.n_logits(B)))
            a = run_dir(sc, M, None, t_fwd, direction, sc['icheck_roots'], 1, stages=1)
            b = run_dir(sc, M, z, t_fwd, direction, sc['icheck_roots'], 1, stages=1)
            dW = float(np.abs(a['W'] - b['W']).max())
            dx = float(np.abs(((a['x'] - b['x']) + 0.5) % 1.0 - 0.5).max())
            dx_after[(M, direction)] = dx
            good = dW <= 1e-12 and (dx <= 1e-12 if M == 1 else True)
            ok &= good
            out[f'M{M}-{direction}'] = dict(max_dW=dW, max_dx_after_M_steps=dx, PASS=good,
                                            dx_role='criterion (= first HMC step, all M)' if M == 1 else 'report only')
            log('I', M, direction, 'max dW', dW, 'max dx after M steps', dx, 'PASS' if good else 'FAIL')
    for direction in ('fwd', 'rev'):
        seq = [dx_after[(M, direction)] for M in sorted(set(Ms) | {1})]
        out[f'dx_monotone_in_M-{direction}'] = dict(M=sorted(set(Ms) | {1}), dx=seq,
                                                    monotone=bool(all(np.diff(seq) >= 0)))
        log('I dx vs M', direction, seq, 'monotone', bool(all(np.diff(seq) >= 0)))
    return ok, out


def calls(K, M, mapped):
    return K * (2 + L_HMC * M + (1 if mapped else 0))


def arm_list(sc):
    a = [('base-uni', 114, 1, 'link')]
    for acc_name, Mf in (('ii', lambda K: M_II[K]), ('i', lambda K: 1)):
        for K in sc['Ks']:
            for arm in ('base-uni', 'dens', 'map-uni', 'map-dens'):
                a.append((arm, K, Mf(K), acc_name))
    if sc['K114']:
        a.append(('map-dens', 114, 1, 'k114'))
    return a


def s3_arms(sc, out, tables, mlp_params):
    res, skipped = {}, []
    for arm, K, M, tag in arm_list(sc):
        name = f'{arm}-K{K}-M{M}'
        if name in res:
            continue
        if time.time() - T0 > sc['gate_s']:
            skipped.append(name)
            log('S3 time gate: skipping', name)
            continue
        grid = 'uni' if arm.endswith('uni') else 'dens'
        if K == 114:
            t_fwd = C.grid(114, 'fwd') if grid == 'uni' else EN.densified_grid(114, LA.PROFILE, 'fwd')
            table = table_from_mlp(mlp_params, t_fwd) if arm.startswith('map') else None
            chunk = sc['chunk114']
        else:
            t_fwd = grid_j(K, grid) / 80.0
            table = tables[(K, grid)] if arm.startswith('map') else None
            chunk = K
        r = {}
        for direction in ('fwd', 'rev'):
            st = run_dir(sc, M, table, t_fwd, direction, sc['n_roots'], chunk)
            np.savez(out / f'{name}-{direction}.npz', W=st['W'], LD=st['LD'], acc=st['acc'], nonfinite=st['nonfinite'],
                     root_id=st['root_id'], t_dir=st['t_dir'], chunk_times=st['chunk_times'])
            steady = [s / n for n, s in st['chunk_times'][1:]] or [st['chunk_times'][0][1] / st['chunk_times'][0][0]]
            r[direction] = dict(W=st['W'], nonfinite=int(st['nonfinite'].sum()), acc_mean=float(st['acc'].mean()),
                                s_per_stage=float(np.median(steady)), seconds=float(st['chunk_times'][:, 1].sum()),
                                root_id=st['root_id'])
            log('S3', name, direction, 'mean W', float(st['W'].mean()), 'acc', float(st['acc'].mean()),
                f"{r[direction]['seconds']:.1f}s")
        r['meta'] = dict(arm=arm, K=K, M=M, grid=grid, mapped=arm.startswith('map'), calls=calls(K, M, arm.startswith('map')),
                         source='lj_fewstep.py')
        res[name] = r
    return res, skipped


def load_v2_arm(v2name, arm, n_roots):
    r = {}
    for direction in ('fwd', 'rev'):
        with np.load(V2 / f'{v2name}-{direction}.npz') as z:
            ct = z['chunk_times']
            steady = [s / n for n, s in ct[1:]] or [ct[0][1] / ct[0][0]]
            r[direction] = dict(W=z['W'][:n_roots], nonfinite=int(z['nonfinite'][:n_roots].sum()),
                                acc_mean=float(z['acc'][:n_roots].mean()), s_per_stage=float(np.median(steady)),
                                seconds=float(ct[:, 1].sum()), root_id=z['root_id'][:n_roots])
    r['meta'] = dict(arm=arm, K=114, M=1, grid='uni' if arm.endswith('uni') else 'dens', mapped=arm.startswith('map'),
                     calls=calls(114, 1, arm.startswith('map')), source=f'lj_arms.py {v2name}')
    return r


# ------------------------------------------------------------------------------------------------ S4 analysis
def summarize(res):
    A = {}
    rng = np.random.default_rng(0)
    for name, r in res.items():
        wf, wr = r['fwd']['W'], r['rev']['W']
        try:
            s = ES.summarize(wf, wr, r['fwd']['root_id'], r['rev']['root_id'], n_boot=2000, seed=0, truth=TRUTH)
            bar, se = s['point']['BAR'], s['bootstrap']['BAR']['se']
            extra = dict(overlap=s['point']['logistic_overlap_sum'], ess_F=s['point']['ess_forward'],
                         ess_R=s['point']['ess_reverse'])
        except Exception as e:
            bar, se, extra = float('nan'), float('nan'), dict(error=str(e), overlap=float('nan'))
        diss = 0.5 * (wf.mean() + wr.mean())
        bs = [0.5 * (wf[rng.integers(0, len(wf), len(wf))].mean() + wr[rng.integers(0, len(wr), len(wr))].mean())
              for _ in range(2000)]
        nf = r['fwd']['nonfinite'] + r['rev']['nonfinite']
        m = r['meta']
        rig = bool(np.isfinite(bar) and abs(bar - TRUTH) <= 3 * np.sqrt(se ** 2 + TRUTH_SE ** 2) and nf == 0)
        A[name] = dict(**m, BAR=bar, BAR_se=se, SE_adj=se * np.sqrt(m['calls'] / CALLS_REF) if np.isfinite(se) else np.nan,
                       mean_dissipation=diss, mean_dissipation_se=float(np.std(bs, ddof=1)), nonfinite=nf,
                       acc=0.5 * (r['fwd']['acc_mean'] + r['rev']['acc_mean']),
                       s_per_stage=0.5 * (r['fwd']['s_per_stage'] + r['rev']['s_per_stage']),
                       seconds=r['fwd']['seconds'] + r['rev']['seconds'], rigor=rig,
                       low_overlap=bool(not np.isfinite(extra['overlap']) or extra['overlap'] < LOW_OVERLAP),
                       SE2_x_calls=se ** 2 * m['calls'] if np.isfinite(se) else np.nan, **extra)
    return A


def verdict_ratio(x):
    if not np.isfinite(x):
        return 'no eligible arm'
    return 'clear advantage' if x <= 0.7 else ('small advantage' if x <= 0.9 else 'no advantage')


def additivity(A, K, M):
    g = {a: A.get(f'{a}-K{K}-M{M}') for a in ('base-uni', 'dens', 'map-uni', 'map-dens')}
    if any(v is None for v in g.values()):
        return None
    D = {a: v['mean_dissipation'] for a, v in g.items()}
    S_ = {a: v['mean_dissipation_se'] for a, v in g.items()}
    rm, rd, rb = D['map-uni'] / D['base-uni'], D['dens'] / D['base-uni'], D['map-dens'] / D['base-uni']
    best_single = 'map-uni' if D['map-uni'] <= D['dens'] else 'dens'
    gap = D[best_single] - D['map-dens']
    se_gap = np.sqrt(S_[best_single] ** 2 + S_['map-dens'] ** 2)
    if rb <= 0.9 * min(rm, rd) and gap > 2 * se_gap:
        v = 'additive'
    elif rb > max(rm, rd) and -gap > 2 * se_gap:
        v = 'interfering'
    else:
        v = 'not additive'
    return dict(r_map=rm, r_dens=rd, r_both=rb, product=rm * rd, best_single=best_single, gap=gap, se_gap=se_gap,
                verdict=v, only_meaningful=bool(rm <= 0.9 and rd <= 0.9))


def s4(sc, res, skipped, out, extra):
    A = summarize(res)
    crit = dict(skipped_by_time_gate=skipped, **extra)
    alarm = [n for n, v in A.items() if not v['rigor'] and not v['low_overlap']]
    crit['R'] = dict(all_rigor=all(v['rigor'] for v in A.values()), implementation_alarm=alarm,
                     low_overlap_arms=[n for n, v in A.items() if v['low_overlap']],
                     failed_rigor=[n for n, v in A.items() if not v['rigor']])
    elig = {n: v for n, v in A.items() if v['rigor']}
    nomap = {n: v for n, v in elig.items() if not v['mapped'] and (v['K'] == 114 or v['M'] == M_II.get(v['K']))}
    mapped = {n: v for n, v in elig.items() if v['mapped'] and v['K'] <= 40 and v['M'] == M_II.get(v['K'])}
    bn = min(nomap, key=lambda n: nomap[n]['SE_adj']) if nomap else None
    bm = min(mapped, key=lambda n: mapped[n]['SE_adj']) if mapped else None
    ratio = (mapped[bm]['SE_adj'] / nomap[bn]['SE_adj']) if (bn and bm) else float('nan')
    crit['main_ii'] = dict(best_nomap=bn, best_nomap_SE_adj=nomap[bn]['SE_adj'] if bn else None, best_map=bm,
                           best_map_SE_adj=mapped[bm]['SE_adj'] if bm else None, ratio=ratio,
                           verdict=verdict_ratio(ratio) if not alarm else 'not judged (implementation alarm)')
    crit['additivity_ii'] = {K: additivity(A, K, M_II[K]) for K in sc['Ks']}
    crit['additivity_i'] = {K: additivity(A, K, 1) for K in sc['Ks']}
    save_json(out / 'analysis.json', dict(truth=TRUTH, calls_ref=CALLS_REF, arms=A, criteria=crit))
    log('S4', json.dumps(ES.json_safe(crit)))
    return crit


T0 = time.time()
PATH = None


def main():
    global PATH, V2
    V2, out = Path(sys.argv[1]), Path(sys.argv[2])
    sc = SC
    out.mkdir(parents=True, exist_ok=False)
    print('devices', jax.devices(), 'jax', jax.__version__, flush=True)
    save_json(out / 'meta.json', dict(sc=sc, M_II=M_II, H=H, L=L_HMC, R_C=R_C, B=B, calls_ref=CALLS_REF,
                                      seeds=dict(est=SEED_EST, mid=SEED_MID, train=SEED_TRAIN), arms_dir=str(V2)))
    PATH = S.make_base_path()
    try:
        grids = {(K, g): grid_j(K, g).tolist() for K in sc['Ks'] for g in ('uni', 'dens')}
        save_json(out / 'grids.json', {f'K{K}-{g}': v for (K, g), v in grids.items()})
        needed = sorted({j for v in grids.values() for j in v if j % 2})
        log('S1 needed midpoints (1/80 units)', needed)
        samples = s1_midpoints(PATH, sc, out, needed)
        Ms = sorted({M_II[K] for K in sc['Ks']} | {1})
        ok, ires = icheck(sc, Ms)
        extra = dict(identity_check=ires, identity_check_pass=ok)
        if not ok:
            save_json(out / 'analysis.json', dict(criteria=extra))
            log('IDENTITY CHECK FAILED: stop')
            return 1
        tables, tinfo = {}, {}
        for K in sc['Ks']:
            for g in ('uni', 'dens'):
                tables[(K, g)], tinfo[f'K{K}-{g}'] = s2_train(PATH, sc, samples, K, g)
        with open(out / 'tables.pkl', 'wb') as fh:
            pickle.dump(tables, fh)
        save_json(out / 'train_info.json', tinfo)
        with open(V2 / 'trained_params.pkl', 'rb') as fh:
            mlp = pickle.load(fh)['learned-BDHD']
        res, skipped = s3_arms(sc, out, tables, mlp)
        # link: rerun of the lj_arms.py baseline K114 must equal its W bit for bit
        link = {}
        for direction in ('fwd', 'rev'):
            with np.load(V2 / f'baseline-K114-{direction}.npz') as z:
                link[direction] = float(np.abs(z['W'][:sc['n_roots']] - res['base-uni-K114-M1'][direction]['W']).max())
        extra['link_baseline_K114_max_abs_dW'] = link
        extra['link_pass'] = bool(max(link.values()) == 0.0)
        log('link max|dW| vs lj_arms.py', link)
        res['dens-K114-M1'] = load_v2_arm('diag_densify_only-K114', 'dens', sc['n_roots'])
        res['map-uni-K114-M1'] = load_v2_arm('learned-BDHD-K114', 'map-uni', sc['n_roots'])
        s4(sc, res, skipped, out, extra)
    except Exception:
        traceback.print_exc()
        return 1
    log('DONE')
    return 0


if __name__ == '__main__':
    sys.exit(main())
