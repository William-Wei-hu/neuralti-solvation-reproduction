"""NeuralTI's own free-energy estimate for a trained checkpoint (MACE or EGNN), on one fixed set of interpolant samples.

Estimator (the paper's; same as D1 of ../2026-11-07-NeuralTI-solvation-NES/code/interp_diag.py, rewritten here, that
directory is only read):
  dF_NTI = int_0^1 E_{x ~ rho_t^I}[ dU_theta/dt (x, t) ] dt,
  the partial t-derivative at fixed x, as upstream stochastic_interpolant.loss_fn's dUdt.
  rho_t^I: x0, x1 drawn independently from the training end-point samples, solute-centred and Hungarian-aligned by the
  upstream src/dataloader.batch_align (float32, x64 off: the training code path); x_t = (x0 + t*minimage(x1 - x0)) mod 1.
  Each pair gets J stratified t, t_pj = (j + u_pj)/J. Its mean over j is an unbiased estimate of the integral and the
  pairs are independent -> dF = mean over pairs, SE = sd/sqrt(n_pairs).

Stages (separate processes, so that jax_enable_x64 is fixed before anything else is imported):
  align    x64 off. Draw and align the pairs, draw u; check the matching against scipy on the first 256 pairs.
           -> <out>.npz (A0, A1 float32; u float64; indices), <out>.json
  eval     one checkpoint in one numerical mode (--x64 0|1, --matmul default|highest):
           per-sample (U, dU/dt) on every (pair, stratum) of the pairs file, interpolation done in the working dtype
           inside the jitted function; the upstream SI.loss_fn path (aux = score loss, dudt loss, mean dU/dt) on
           --loss-batches batches of 24 pairs (the quantity the training log records); the checkpoint's dUdt_hist.
           -> <out>.npz (per-sample values, float64), <out>.json
  summary  combine: estimates, paired MACE - EGNN differences on the same samples, f32 - f64 differences per model,
           references, rules fixed before running, plot.
Runs under the mace-jax venv (jax 0.10.2) and under the neuralti-lj env (jax 0.4.37) that trained the EGNN.
"""
import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

# References (kBT), copied from ../2026-11-07-NeuralTI-solvation-NES (README.zh-CN.md, analysis/test-c-base.json,
# analysis/interp-diag.json) on 2026-11-08; that directory is not read at run time.
REF = dict(
    NES_base_BAR=-20.365, NES_base_BAR_se=0.013,              # bidirectional NES, base path, HMC G=2.4e5 (exact)
    TI65_trapezoid=-20.073, TI65_simpson=-20.208, TI65_author=-20.181,   # standard TI, 65 slices (upstream run_standardTI.sh)
    EGNN_D1=-2.606, EGNN_D1_se=0.481,                         # NES interp_diag D1, EGNN seed0, f64, 8192 x 16
    EGNN_train_hist_mean_all=-14.720, EGNN_train_hist_last20=-14.243,
)
TOL_ABS = 0.5      # NES test (c): |estimate - reference| <= 3 SE + 0.5


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, obj):
    tmp = f'{path}.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    os.replace(tmp, path)


def jax_setup(x64, matmul):
    import jax
    jax.config.update('jax_enable_x64', bool(x64))
    if matmul != 'default':
        jax.config.update('jax_default_matmul_precision', matmul)
    return jax


def mean_se(v):
    v = np.asarray(v, np.float64)
    return float(v.mean()), float(v.std(ddof=1)/np.sqrt(len(v))) if len(v) > 1 else float('nan')


# ---------------------------------------------------------------------------------------------------------------- align
def stage_align(a):
    jax = jax_setup(False, 'default')
    import jax.numpy as jnp
    import dataloader as DL                      # upstream src/dataloader.py
    from scipy.optimize import linear_sum_assignment
    t_start = time.time()
    rng = np.random.default_rng(a.seed)
    X, meta = [], dict(stage='align', seed=a.seed, devices=[str(d) for d in jax.devices()], jax=jax.__version__)
    for k, p in (('data0', a.data0), ('data1', a.data1)):
        with open(p, 'rb') as f:
            X.append(np.asarray(pickle.load(f), dtype=np.float32))
        meta[k] = dict(path=p, sha256=sha256(p), shape=list(X[-1].shape))
    X0, X1 = X
    n = a.n_pairs
    i0 = rng.choice(len(X0), n, replace=False)
    i1 = rng.choice(len(X1), n, replace=False)
    align = jax.jit(DL.batch_align)
    A0, A1 = [], []
    for s in range(0, n, a.batch):
        b0, b1 = align(jnp.asarray(X0[i0[s:s + a.batch]]), jnp.asarray(X1[i1[s:s + a.batch]]))
        A0.append(np.asarray(b0, np.float32))
        A1.append(np.asarray(b1, np.float32))
        print(f'  aligned {s + len(A0[-1])}/{n}  {time.time() - t_start:.0f}s', flush=True)
        if time.time() - t_start > a.budget_s:
            print(f'  ALIGN TIME BUDGET reached after {s + len(A0[-1])} pairs', flush=True)
            break
    A0, A1 = np.concatenate(A0), np.concatenate(A1)
    n = len(A0)

    def d2(p, q):
        d = p[:, None, :].astype(np.float64) - q[None, :, :]
        d = d - np.round(d)
        return (d*d).sum(-1)

    rel = []
    for j in range(min(256, n)):
        c = d2(A0[j, 1:], A1[j, 1:])
        r, cidx = linear_sum_assignment(c)
        best = float(c[r, cidx].sum())
        rel.append((float(np.trace(c)) - best)/best)
    rel = np.array(rel)
    u = rng.random((n, a.strata))
    np.savez(a.out + '.npz', A0=A0, A1=A1, u=u, i0=i0[:n], i1=i1[:n])
    meta.update(n_pairs=n, strata=a.strata, seconds=time.time() - t_start,
                solute_at_origin_max=float(np.abs(((A0[:, 0] + 0.5) % 1) - 0.5).max()
                                           + np.abs(((A1[:, 0] + 0.5) % 1) - 0.5).max()),
                matching_vs_scipy=dict(n_checked=len(rel), max_rel_excess_cost=float(rel.max()),
                                       n_rel_excess_gt_1e5=int((rel > 1e-5).sum()),
                                       note='upstream matched cost minus scipy optimal cost, relative'),
                precision='upstream batch_align, float32, x64 off (training code path)')
    atomic_json(a.out + '.json', meta)
    print('ALIGN', json.dumps(meta), flush=True)


# ----------------------------------------------------------------------------------------------------------------- eval
def stage_eval(a):
    jax = jax_setup(a.x64, a.matmul)
    import jax.numpy as jnp
    import hydra
    from omegaconf import OmegaConf
    t_start = time.time()
    dt = jnp.float64 if a.x64 else jnp.float32
    mode = f"{'f64' if a.x64 else 'f32'}-{a.matmul}"
    with open(a.ckpt, 'rb') as f:
        ck = pickle.load(f)
    cfg = ck['cfg']
    for k, v in cfg.jax_config.items():                  # rank promotion as in training; x64 and debug_nans are ours
        if k not in ('jax_enable_x64', 'jax_debug_nans'):
            jax.config.update(k, v)
    SI = hydra.utils.instantiate(cfg.model)               # as experiments/main.py
    SI.num_features = SI.target_system.num_dim
    SI.U_model = hydra.utils.instantiate(cfg.U_model)
    params = jax.tree_util.tree_map(lambda v: jnp.asarray(v, dtype=dt), ck['params'])
    try:
        step = int(np.asarray(ck['opt_state'][0].count))
    except Exception as e:
        step = f'unknown ({type(e).__name__}: {e})'
    res = dict(stage='eval', name=a.name, mode=mode, x64=bool(a.x64), matmul=a.matmul, jax=jax.__version__,
               devices=[str(d) for d in jax.devices()], ckpt=a.ckpt, ckpt_sha256=sha256(a.ckpt), optimizer_step=step,
               n_params=int(sum(np.size(v) for v in jax.tree_util.tree_leaves(params))),
               cfg_U_model=OmegaConf.to_container(cfg.U_model, resolve=True),
               jax_config={k: v for k, v in OmegaConf.to_container(cfg.jax_config).items()},
               rank_promotion=str(getattr(jax.config, 'jax_numpy_rank_promotion', 'unreadable')),
               matmul_precision_config=str(getattr(jax.config, 'jax_default_matmul_precision', 'unreadable')),
               x64_config=str(getattr(jax.config, 'jax_enable_x64', 'unreadable')))
    print('EVAL', json.dumps({k: v for k, v in res.items() if k != 'cfg_U_model'}), flush=True)

    P = np.load(a.pairs)
    A0, A1, u = P['A0'], P['A1'], P['u']
    n, J = u.shape
    T = (np.arange(J)[None, :] + u)/J                    # (n, J) stratified t

    def one(p, x0, x1, t):
        dx = ((x1 - x0) + 0.5) % 1 - 0.5
        xt = (x0 + t*dx) % 1
        return jax.value_and_grad(lambda tt: SI.U(p, xt, jnp.reshape(tt, (1,))))(t)   # partial d/dt at fixed x_t

    f = jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0)))  # one vmap over B*J samples, as upstream loss_fn and NES D1
    B = a.pairs_per_batch
    U, G = [], []
    t0 = time.time()
    first = None
    for s in range(0, n - n % B, B):
        Ub, Gb = f(params, jnp.asarray(np.repeat(A0[s:s + B], J, axis=0), dt),
                   jnp.asarray(np.repeat(A1[s:s + B], J, axis=0), dt), jnp.asarray(T[s:s + B].reshape(-1), dt))
        U.append(np.asarray(Ub, np.float64).reshape(B, J))
        G.append(np.asarray(Gb, np.float64).reshape(B, J))
        done = s + B
        if first is None:
            first = time.time() - t0
            print(f'  first batch (compile + run) {first:.1f}s', flush=True)
        el = time.time() - t0
        if done % max(B, (n//20)//B*B) == 0:
            g = np.concatenate(G)
            print(f'  {done}/{n} pairs  {el:.0f}s  ETA {(el - first)/max(done - B, 1)*(n - done):.0f}s  '
                  f'running dF {g.mean(1).mean():.3f}  nonfinite {int((~np.isfinite(g)).sum())}', flush=True)
        if el > a.budget_s:
            print(f'  EVAL TIME BUDGET reached after {done} pairs', flush=True)
            break
    secs = time.time() - t0
    U, G = np.concatenate(U), np.concatenate(G)
    m = len(G)
    finite = np.isfinite(G).all(1) & np.isfinite(U).all(1)
    per_pair = G.mean(1)
    dF, se = mean_se(per_pair) if finite.all() else (float('nan'), float('nan'))
    dF_f, se_f = mean_se(per_pair[finite]) if finite.any() else (float('nan'), float('nan'))
    tt = T[:m].reshape(-1)
    g = G.reshape(-1)
    nb = 100
    b = np.minimum((tt*nb).astype(int), nb - 1)
    ok = np.isfinite(g)
    cnt = np.bincount(b[ok], minlength=nb)
    gm = np.bincount(b[ok], weights=g[ok], minlength=nb)/np.maximum(cnt, 1)
    gv = np.bincount(b[ok], weights=g[ok]**2, minlength=nb)/np.maximum(cnt, 1) - gm**2
    res.update(n_pairs=m, n_pairs_available=n, strata=J, pairs_per_batch=B, seconds_eval=secs, first_batch_s=first,
               s_per_sample=(secs - first)/max((m - B)*J, 1),
               nonfinite_dUdt=int((~np.isfinite(G)).sum()), nonfinite_U=int((~np.isfinite(U)).sum()),
               dF_NTI=dF, dF_NTI_se=se, dF_NTI_finite_pairs=dF_f, dF_NTI_finite_pairs_se=se_f,
               n_finite_pairs=int(finite.sum()),
               dUdt_rms=float(np.sqrt(np.mean(g[ok]**2))), dUdt_sd_within_pair=float(G[finite].std(1, ddof=1).mean()),
               U_mean=float(np.nanmean(U)),
               profile=dict(t_bin_centers=((np.arange(nb) + .5)/nb).tolist(), mean_dUdt=gm.tolist(),
                            se_dUdt=np.sqrt(np.maximum(gv, 0)/np.maximum(cnt - 1, 1)).tolist(), count=cnt.tolist()))
    np.savez(a.out + '.npz', U=U, dUdt=G, t=T[:m])
    print('ESTIMATE', json.dumps({k: res[k] for k in ('name', 'mode', 'n_pairs', 'dF_NTI', 'dF_NTI_se', 'nonfinite_dUdt',
                                                      'dUdt_rms', 'seconds_eval', 's_per_sample')}), flush=True)
    atomic_json(a.out + '.json', res)

    # upstream loss_fn path, 24 pairs per batch as in training (t drawn inside loss_fn from the key)
    lf = jax.jit(lambda p, b0, b1, k: SI.loss_fn(p, (b0, b1), k)[1])
    rows = []
    t1 = time.time()
    for i in range(a.loss_batches):
        s = (i*24) % (n - 24)
        aux = lf(params, jnp.asarray(A0[s:s + 24], dt), jnp.asarray(A1[s:s + 24], dt), jax.random.PRNGKey(1000 + i))
        rows.append([float(v) for v in aux])
    rows = np.array(rows)
    res['loss_fn_path'] = dict(batches=len(rows), batch=24, seconds=time.time() - t1,
                               score_loss=mean_se(rows[:, 0]), dudt_loss=mean_se(rows[:, 1]),
                               mean_dUdt=mean_se(rows[:, 2]), nonfinite=int((~np.isfinite(rows)).sum()),
                               note='[mean, SE] over batches; mean_dUdt is what the training log calls running dF')
    print('LOSS_FN', json.dumps(res['loss_fn_path']), flush=True)

    hp = Path(a.ckpt).parent/'dUdt_hist'
    try:
        with open(hp, 'rb') as fh:
            hist = np.asarray(pickle.load(fh)['train'], np.float64)
        row = hist.mean(1)                               # one row per 20-step block
        k80 = int(0.8*len(row))
        k20 = 1000                                       # row 1000 = step 20000: learning rate has reached 1e-5
        res['training_history'] = dict(path=str(hp), sha256=sha256(hp), shape=list(hist.shape),
                                       mean_all=mean_se(row), mean_after_step20000=mean_se(row[k20:]),
                                       mean_last20pct=mean_se(row[k80:]),
                                       note='[mean, naive SE]; rows = 20-step blocks; parameters change along it')
    except Exception as e:
        res['training_history'] = dict(error=repr(e))
    print('HISTORY', json.dumps(res['training_history']), flush=True)
    res['seconds_total'] = time.time() - t_start
    atomic_json(a.out + '.json', res)


# -------------------------------------------------------------------------------------------------------------- summary
def stage_summary(a):
    """Rules (fixed before the run):
    R1 replication: our EGNN f64 estimate reproduces NES D1 if |ours - (-2.606)| <= 3 sqrt(SE^2 + 0.481^2).
    R2 accuracy:    a model's f64 estimate agrees with the exact reference (NES base BAR -20.365 +- 0.013) if
                    |dF_NTI - ref| <= 3 sqrt(SE^2 + 0.013^2) + 0.5  (the tolerance of NES test (c)).
    R3 numerics:    f32-default and f64 differ for a model if the paired per-pair difference exceeds 4 SE."""
    d = Path(a.dir)
    runs = {}
    for p in sorted(d.glob('*.json')):
        r = json.loads(p.read_text())
        if r.get('stage') == 'eval':
            r['_npz'] = str(p.with_suffix('.npz'))
            runs[(r['name'], r['mode'])] = r
    out = dict(stage='summary', dir=str(d), references=REF, rules=stage_summary.__doc__, runs={}, paired={},
               precision={}, verdicts={})
    al = d/'pairs.json'
    if al.exists():
        out['alignment'] = json.loads(al.read_text())
    G = {}
    for (name, mode), r in runs.items():
        key = f'{name}/{mode}'
        out['runs'][key] = {k: r.get(k) for k in ('dF_NTI', 'dF_NTI_se', 'n_pairs', 'nonfinite_dUdt', 'dUdt_rms',
                                                  'dUdt_sd_within_pair', 'seconds_eval', 'first_batch_s',
                                                  's_per_sample', 'optimizer_step', 'n_params', 'ckpt_sha256',
                                                  'jax', 'devices', 'loss_fn_path', 'training_history')}
        if os.path.exists(r['_npz']):
            G[(name, mode)] = np.load(r['_npz'])['dUdt']
    names = sorted({k[0] for k in runs})
    modes = sorted({k[1] for k in runs})
    for mode in modes:                                   # paired model differences on the same samples
        have = [nm for nm in names if (nm, mode) in G]
        for i in range(len(have)):
            for j in range(i + 1, len(have)):
                g1, g2 = G[(have[i], mode)], G[(have[j], mode)]
                k = min(len(g1), len(g2))
                dd = g1[:k].mean(1) - g2[:k].mean(1)
                out['paired'][f'{mode}: {have[i]} - {have[j]}'] = dict(n_pairs=k, diff=mean_se(dd))
    for nm in names:                                     # numerics per model, same samples
        ref = G.get((nm, 'f64-highest'))
        for mode in modes:
            if mode == 'f64-highest' or (nm, mode) not in G or ref is None:
                continue
            g = G[(nm, mode)]
            k = min(len(g), len(ref))
            ds = (g[:k] - ref[:k]).reshape(-1)
            dp = g[:k].mean(1) - ref[:k].mean(1)
            m_, s_ = mean_se(dp)
            out['precision'][f'{nm}: {mode} - f64-highest'] = dict(
                n_pairs=k, per_pair_diff=[m_, s_], per_sample_rms_diff=float(np.sqrt(np.nanmean(ds**2))),
                per_sample_max_abs_diff=float(np.nanmax(np.abs(ds))),
                R3_differ=bool(abs(m_) > 4*s_) if np.isfinite(s_) else None)
    for nm in names:
        r = runs.get((nm, 'f64-highest'))
        if r is None or not np.isfinite(r['dF_NTI']):
            continue
        dev = r['dF_NTI'] - REF['NES_base_BAR']
        tol = 3*np.hypot(r['dF_NTI_se'], REF['NES_base_BAR_se']) + TOL_ABS
        out['verdicts'][f'R2 {nm}'] = dict(dF_NTI=r['dF_NTI'], se=r['dF_NTI_se'], minus_ref=dev, tol=tol,
                                           agrees=bool(abs(dev) <= tol))
        if 'egnn' in nm:
            dev1 = r['dF_NTI'] - REF['EGNN_D1']
            tol1 = 3*np.hypot(r['dF_NTI_se'], REF['EGNN_D1_se'])
            out['verdicts'][f'R1 {nm}'] = dict(minus_NES_D1=dev1, tol=tol1, reproduces=bool(abs(dev1) <= tol1))
    if a.gpumem and os.path.exists(a.gpumem) and a.phases and os.path.exists(a.phases):
        mem = []
        for line in open(a.gpumem):
            parts = line.replace('MiB', '').split(',')
            try:
                mem.append((int(parts[0]), int(parts[1])))
            except (ValueError, IndexError):
                pass
        ph = [line.split() for line in open(a.phases) if line.strip()]   # name start end
        out['peak_MiB'] = {}
        for p in ph:
            if len(p) == 3:
                v = [x for t, x in mem if int(p[1]) <= t <= int(p[2]) + 5]
                out['peak_MiB'][p[0]] = max(v) if v else None
    atomic_json(a.out, out)
    print('SUMMARY', json.dumps({k: out[k] for k in ('paired', 'precision', 'verdicts')}), flush=True)
    for k, v in out['runs'].items():
        print('RUN', k, json.dumps({kk: v[kk] for kk in ('dF_NTI', 'dF_NTI_se', 'n_pairs', 'nonfinite_dUdt',
                                                         'dUdt_rms', 'seconds_eval')}), flush=True)
    if a.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4))
            for (nm, mode), r in sorted(runs.items()):
                p = r['profile']
                ax.plot(p['t_bin_centers'], p['mean_dUdt'], lw=1.5 if mode == 'f64-highest' else .8,
                        ls='-' if mode == 'f64-highest' else '--',
                        label=f"{nm} {mode}: dF = {r['dF_NTI']:.2f} +- {r['dF_NTI_se']:.2f}")
            ax.axhline(0, color='k', lw=.5)
            ax.set_xlabel('t')
            ax.set_ylabel('<dU/dt> on interpolant samples (kBT)')
            ax.set_title(f"NeuralTI integrand; exact dF = {REF['NES_base_BAR']} (NES)")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(a.plot, dpi=120)
            plt.close(fig)
        except Exception as e:
            print('PLOT FAILED', repr(e), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='stage', required=True)
    s = sub.add_parser('align')
    s.add_argument('--data0', required=True)
    s.add_argument('--data1', required=True)
    s.add_argument('--n-pairs', type=int, default=8192)
    s.add_argument('--strata', type=int, default=16)
    s.add_argument('--seed', type=int, default=20261108)
    s.add_argument('--batch', type=int, default=256)
    s.add_argument('--budget-s', type=float, default=1800)
    s.add_argument('--out', required=True, help='prefix: <out>.npz, <out>.json')
    s = sub.add_parser('eval')
    s.add_argument('--name', required=True)
    s.add_argument('--ckpt', required=True)
    s.add_argument('--pairs', required=True)
    s.add_argument('--x64', type=int, choices=(0, 1), required=True)
    s.add_argument('--matmul', choices=('default', 'highest'), required=True)
    s.add_argument('--pairs-per-batch', type=int, default=2)
    s.add_argument('--loss-batches', type=int, default=256)
    s.add_argument('--budget-s', type=float, default=2700)
    s.add_argument('--out', required=True, help='prefix: <out>.npz, <out>.json')
    s = sub.add_parser('summary')
    s.add_argument('--dir', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--plot')
    s.add_argument('--gpumem')
    s.add_argument('--phases')
    a = ap.parse_args()
    dict(align=stage_align, eval=stage_eval, summary=stage_summary)[a.stage](a)


if __name__ == '__main__':
    main()
