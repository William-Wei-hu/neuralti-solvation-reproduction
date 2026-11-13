"""Tier C of the MACE-OFF methane switching runs, plus the three-tier analysis (tiers A, B from j2_switch.py).
Same physics as j2_switch.py: K2 = HMC full refresh, L = 40, dt = 0.5 fs, float64, linear schedule, K = T / 20 fs;
tier C T = 32 ps, K = 1600, stage keys stage_keys(20261112, dir, 10*chain+segment, K).

  python j3_switch.py run --roots ROOTS_DIR --ref J2_OUT_DIR --out OUT_DIR
  python j3_switch.py analyze OUT_DIR --ref J2_OUT_DIR

Differences from tier C as planned in j2_switch.py:
  * 14 pairs instead of 16 (budget only): roots at positions 64-77 of the (segment, chain) order, i.e. the first 14 of
    the tier-C block, chosen without looking at any result.  Chains 18 and 19 of each end are not used in tier C.
  * Forward and reverse trajectories of tier C run in ONE device call of 28 (vmap over x0, p0, schedule and keys).
    Each trajectory is the same computation as before (per-trajectory keys and schedule); vmap only batches.  Both
    files are written as soon as that call returns.
  * Count thresholds of G3 / G4 for 14 pairs: lim = 2 if n_pairs >= 32 else 1, i.e. lim = 1.
Analysis (all gates and thresholds as in j2_switch.py; only outputs added):
  tiers A, B are read from the j2_switch.py output (--ref), tier C from this run; G1-G4 per tier; G5 among tiers
  passing G1-G4; STOP rules: tier B (as j2_switch.py) and tier C: mean dissipation C >= 0.8 x mean dissipation B.
  Added outputs: mean dissipation per tier = (mean W_F + mean W_R) / 2 (dF cancels) with chain-cluster bootstrap SE;
  ratios B/A and C/B with bootstrap SE and 95 % percentile interval (2000 draws, seed 0, chains resampled
  independently per tier and direction); per-direction mean work per lambda bin (reverse sign-flipped so that
  fwd + rev = the dissipation profile; the split between directions in one bin contains the unknown dF of that bin);
  extrapolated calls to +-0.05 kBT: n_pairs = c W_d / 0.05^2 with c = 1.2 (fitted on LJ, not refitted), calls =
  n_pairs x 2 x (41 K + 1).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_default_matmul_precision', 'highest')
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'nes'))   # estimators
import mlip_nes as M  # noqa: E402

KT = M.kbt(298.15)
SEED, NBINS, L_LEAP, DT, STAGE_FS = 20261112, 20, 40, 0.5, 20.0
C_EXTRAP, TARGET = 1.2, 0.05
TIERS = dict(A=dict(T=2000.0, n=32), B=dict(T=8000.0, n=32), C=dict(T=32000.0, sl=slice(64, 78), n=14))
ROOTS = REF = RUN = None                       # set from the command line
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def load_roots():
    out = {}
    for e in (0, 1):
        z = np.load(ROOTS / f'roots_end{e}.npz')
        o = np.lexsort((z['chain_id'], z['segment']))
        r = {k: z[k] for k in z.files}
        for k in ('x', 'chain_id', 'segment', 'kernel'):
            r[k] = r[k][o]
        out[e] = r
    return out


def run():
    import mace_iface as MI
    RUN.mkdir(parents=True, exist_ok=False)
    roots = load_roots()
    r0 = roots[0]
    Z, L, sol = r0['Z'], float(r0['L']), r0['solute'].astype(bool)
    pad_src = json.loads((ROOTS / 'j1.json').read_text())['info']['pad']
    m = MI.MaceIface('off23s', Z, L, sol, pad=int(pad_src))
    log('devices', jax.devices(), json.dumps(m.info))
    one = m._one

    def efn(x, lam):
        _, eint, F, ne = one(x, lam)
        return eint, F, jnp.asarray(ne).astype(jnp.int32)
    mass = jnp.asarray(m.masses)
    pairs = M.oh_pairs_from(Z, r0['x'][0], L, sol)
    sidx, widx = np.where(sol)[0], np.where(~sol)[0]
    base = M.make_probes(sidx, widx, pairs, L)
    C = sidx[int(np.where(Z[sol] == 6)[0][0])]
    Hm = sidx[np.where(Z[sol] == 1)[0]]

    def probes(x):
        d = base(x)
        ch = M.min_image(x[Hm] - x[C], L)
        d['max_ch'] = jnp.sqrt((ch ** 2).sum(-1).max())
        return d
    cfg = TIERS['C']
    K = int(round(cfg['T'] / STAGE_FS))
    idx = np.arange(cfg['sl'].start, cfg['sl'].stop)
    n = len(idx)
    calls_dir = n * (K * (L_LEAP + 1) + 1)
    meta = dict(info=m.info, roots=str(ROOTS), ref=str(REF), kernel='K2 hmc L=40', dt=DT,
                tier_C=dict(T_fs=cfg['T'], K=K, roots=[int(idx[0]), int(idx[-1]) + 1], n_pairs=n,
                            chains_fwd=roots[0]['chain_id'][idx].tolist(), chains_rev=roots[1]['chain_id'][idx].tolist()),
                batch='fwd and rev together, one call of 2n', chunks=[])
    (RUN / 'meta.json').write_text(json.dumps(meta, indent=1, default=float))
    log(f'tier C: {n} pairs, K = {K}')
    sw = M.make_switch(efn, kind='hmc', m=mass, dt=DT, L=L_LEAP, box=L, kT=KT, nbins=NBINS, record=False, probes=probes)
    swb = jax.jit(jax.vmap(sw, in_axes=(0, 0, 0, 0)))
    xs, ps, lams, keys, info = [], [], [], [], []
    for d, e in (('fwd', 0), ('rev', 1)):
        R = roots[e]
        rid = 10 * R['chain_id'][idx] + R['segment'][idx]
        k = jnp.stack([M.stage_keys(SEED, d, int(r), K) for r in rid])
        xs.append(jnp.asarray(R['x'][idx]))
        ps.append(jax.vmap(lambda kk: M.draw_p(kk, mass, KT, jnp.float64))(k[:, 0]))
        lams.append(jnp.broadcast_to(jnp.asarray(M.schedule(K, d)), (n, K + 1)))
        keys.append(k[:, 1:])
        info.append((d, R))
    ts = time.time()
    out = jax.block_until_ready(swb(jnp.concatenate(xs), jnp.concatenate(ps), jnp.concatenate(lams),
                                    jnp.concatenate(keys)))
    sec = time.time() - ts
    for j, (d, R) in enumerate(info):
        s = slice(j * n, (j + 1) * n)
        x = np.asarray(out['x'][s])
        dC = M.min_image(x[:, widx] - x[:, [C]], L)
        dC = np.sqrt((dC ** 2).sum(-1)).min(1)
        rec = dict(W=np.asarray(out['W'][s]), Wbins=np.asarray(out['Wbins'][s]), acc=np.asarray(out['acc'][s]),
                   n_edges_max=np.asarray(out['n_edges_max'][s]), nonfinite=np.asarray(out['nonfinite'][s]),
                   min_ss=np.asarray(out['min_solute_solvent'][s]).min(1),
                   max_oh=np.asarray(out['max_oh'][s]).max(1), max_ch=np.asarray(out['max_ch'][s]).max(1),
                   final_C_water=dC, x_final=x.astype(np.float32), chain_id=R['chain_id'][idx],
                   segment=R['segment'][idx], root_pos=idx, seconds=np.float64(sec / 2),
                   seconds_shared_call=np.float64(sec), calls=np.int64(calls_dir))
        f = RUN / f'tierC-{d}-c00.npz'
        np.savez(f, **rec)
        meta['chunks'].append(dict(file=f.name, seconds_shared_call=sec, calls=calls_dir,
                                   ms_per_config=1e3 * sec / (2 * calls_dir)))
        log(f'tier C {d}: <W>={rec["W"].mean():.3f} sd={rec["W"].std():.3f} acc={np.nanmean(rec["acc"]):.3f} '
            f'edges={int(rec["n_edges_max"].max())} nonfinite={int(rec["nonfinite"].sum())} '
            f'oh={rec["max_oh"].max():.3f} {sec:.0f}s (shared) {1e3 * sec / (2 * calls_dir):.2f} ms/config')
    (RUN / 'meta.json').write_text(json.dumps(meta, indent=1, default=float))


# ------------------------------------------------------------------------------------------------ analysis
def load_tier(run, t, d):
    fs = sorted(run.glob(f'tier{t}-{d}-c*.npz'))
    if not fs:
        return None
    parts = [dict(np.load(f)) for f in fs]
    return {k: (np.concatenate([p[k] for p in parts]) if np.ndim(parts[0][k]) else np.array([p[k] for p in parts]))
            for k in parts[0]}


def boot_mean_diss(F, R, rng, nb):
    """Chain-cluster bootstrap of (mean W_F + mean W_R)/2; chains resampled independently per direction."""
    out = np.empty(nb)
    groups = []
    for X in (F, R):
        ids = X['chain_id']
        u = np.unique(ids)
        groups.append([X['W'][ids == c] for c in u])
    for b in range(nb):
        mm = []
        for g in groups:
            pick = rng.integers(0, len(g), len(g))
            w = np.concatenate([g[i] for i in pick])
            mm.append(w.mean())
        out[b] = 0.5 * (mm[0] + mm[1])
    return out


def tier_stats(E, t, F, R, npairs, K):
    lim = 2 if npairs >= 32 else 1
    rid_f = 10 * F['chain_id'] + F['segment']
    rid_r = 10 * R['chain_id'] + R['segment']
    s_root = E.summarize(F['W'], R['W'], rid_f, rid_r, n_boot=2000, seed=0, truth=-3.38)
    s_chain = E.summarize(F['W'], R['W'], F['chain_id'], R['chain_id'], n_boot=2000, seed=0, truth=-3.38)
    p = s_root['point']
    se_root, se_chain = s_root['bootstrap']['BAR']['se'], s_chain['bootstrap']['BAR']['se']
    se = max(se_root, se_chain)
    fin = bool(np.isfinite(F['W']).all() and np.isfinite(R['W']).all())
    nonf = int(F['nonfinite'].sum() + R['nonfinite'].sum())
    pad = PAD
    edges_over = int((F['n_edges_max'] >= pad).sum() + (R['n_edges_max'] >= pad).sum())
    bonds = int((F['max_oh'] > 1.5).sum() + (R['max_oh'] > 1.5).sum() + (F['max_ch'] > 1.5).sum() + (R['max_ch'] > 1.5).sum())
    g1 = fin and nonf == 0 and edges_over == 0 and bonds == 0
    g2 = p['logistic_overlap_sum'] >= 0.1 and s_root['crooks']['bins_used'] >= 3
    stuck = {d: int(((X['acc'].astype(float) < 0.05).mean(1) >= 0.05).sum()) for d, X in (('fwd', F), ('rev', R))}
    g3 = max(stuck.values()) <= lim
    Wr = R['W']
    med = np.median(Wr)
    mad = np.median(np.abs(Wr - med))
    extreme = int((np.abs(Wr - med) > 10 * mad).sum()) if mad > 0 else 0
    order = np.argsort(-np.abs(Wr - med))
    drop = {}
    for k in (1, 2, 3):
        keep = np.ones(len(Wr), bool)
        keep[order[:k]] = False
        try:
            drop[k] = float(E.bar(F['W'], Wr[keep]) - p['BAR'])
        except Exception as ex:
            drop[k] = repr(ex)
    close = int((R['final_C_water'] < 1.5).sum())
    g4a, g4b, g4c = extreme <= lim, (isinstance(drop[2], float) and abs(drop[2]) <= 2 * se), close == 0
    g4 = g4a and g4b and g4c
    diss = [p['dissipation_forward'], p['dissipation_reverse']]
    fb, rb = F['Wbins'].mean(0), R['Wbins'].mean(0)
    mdiss = 0.5 * (p['mean_WF'] + p['mean_WR'])
    n_ext = C_EXTRAP * mdiss / TARGET ** 2
    os_ = {side: dict(value=p[k], se=s_root['bootstrap'][k]['se'], ess=p[e_], cite_50=bool(p[e_] >= 50),
                      cite_500=bool(p[e_] >= 500))
           for side, k, e_ in (('forward', 'forward_jarzynski', 'ess_forward'),
                               ('reverse', 'reverse_jarzynski', 'ess_reverse'))}
    return dict(
        n=[len(F['W']), len(R['W'])], n_pairs_planned=npairs, count_limit=lim, K=K,
        BAR=p['BAR'], SE_root=se_root, SE_chain=se_chain,
        mean_W=[p['mean_WF'], p['mean_WR']], sd_W=[p['sd_WF'], p['sd_WR']], dissipation=diss,
        mean_dissipation=float(mdiss), logistic_overlap=p['logistic_overlap_sum'],
        histogram_overlap=p['histogram_overlap'], crooks_bins=s_root['crooks']['bins_used'],
        ess=[p['ess_forward'], p['ess_reverse']], one_sided=os_,
        W_range=dict(fwd=[float(F['W'].min()), float(F['W'].max())], minus_rev=[float(-R['W'].max()), float(-R['W'].min())]),
        G1=dict(passed=bool(g1), finite=fin, nonfinite=nonf, edges_over_pad=edges_over, bond_breaks=bonds,
                max_oh=float(max(F['max_oh'].max(), R['max_oh'].max())),
                max_ch=float(max(F['max_ch'].max(), R['max_ch'].max())),
                n_edges_max=int(max(F['n_edges_max'].max(), R['n_edges_max'].max())), pad=pad),
        G2=dict(passed=bool(g2)), G3=dict(passed=bool(g3), stuck=stuck, limit=lim),
        G4=dict(passed=bool(g4), extreme_rev=extreme, limit=lim, bar_change_drop=drop, se_used=se,
                final_C_water_lt_1p5=close, min_final_C_water=float(R['final_C_water'].min()),
                a=bool(g4a), b=bool(g4b), c=bool(g4c)),
        coarse_check=dict(expected=-3.38, diff=p['BAR'] + 3.38, tol=max(1.0, 3 * se),
                          within=bool(abs(p['BAR'] + 3.38) <= max(1.0, 3 * se))),
        dissipation_profile_20bins=(fb + rb).tolist(),
        work_bins_fwd=fb.tolist(), work_bins_rev_signflip_equiv=rb.tolist(),
        extrapolation=dict(c=C_EXTRAP, c_source='fitted on LJ, not refitted on the pilot', target=TARGET,
                           W_d=float(mdiss), n_pairs=float(n_ext), calls=float(n_ext * 2 * (K * (L_LEAP + 1) + 1))),
        acc_mean=[float(np.nanmean(F['acc'])), float(np.nanmean(R['acc']))],
        gpu_seconds=float(F['seconds'].sum() + R['seconds'].sum()),
        calls=int(F['calls'].sum() + R['calls'].sum()))


PAD = None


def analyze(run):
    global PAD
    import estimators as E
    run = Path(run)
    meta = json.loads((run / 'meta.json').read_text())
    PAD = int(meta['info']['pad'])
    res = dict(meta=meta, ref=str(REF), tiers={})
    data = {}
    for t in ('A', 'B', 'C'):
        src = run if t == 'C' else REF
        F, R = load_tier(src, t, 'fwd'), load_tier(src, t, 'rev')
        if F is None or R is None:
            res['tiers'][t] = dict(error=f'missing files in {src}')
            continue
        K = int(round(TIERS[t]['T'] / STAGE_FS))
        try:
            res['tiers'][t] = tier_stats(E, t, F, R, TIERS[t]['n'], K)
            res['tiers'][t]['source'] = str(src)
            data[t] = (F, R)
        except Exception as ex:
            res['tiers'][t] = dict(error=repr(ex))
    # bootstrap of mean dissipation and ratios
    rng = np.random.default_rng(0)
    bs = {t: boot_mean_diss(F, R, rng, 2000) for t, (F, R) in data.items()}
    for t, b in bs.items():
        res['tiers'][t]['mean_dissipation_boot'] = dict(se=float(b.std(ddof=1)),
                                                        ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])
    ratios = {}
    for num, den in (('B', 'A'), ('C', 'B')):
        if num in bs and den in bs:
            r = bs[num] / bs[den]
            pt = res['tiers'][num]['mean_dissipation'] / res['tiers'][den]['mean_dissipation']
            ratios[f'{num}/{den}'] = dict(point=float(pt), se=float(r.std(ddof=1)),
                                          ci95=[float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))],
                                          le_0p4=bool(pt <= 0.4), gt_0p6=bool(pt > 0.6))
    res['ratios'] = ratios
    ok = {t: v for t, v in res['tiers'].items() if 'BAR' in v and v['G1']['passed'] and v['G2']['passed']
          and v['G3']['passed'] and v['G4']['passed']}
    g5 = {}
    ts = sorted(ok)
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            a, b = ok[ts[i]], ok[ts[j]]
            g5[f'{ts[i]}-{ts[j]}'] = bool(abs(a['BAR'] - b['BAR']) <= 2 * np.hypot(a['SE_chain'], b['SE_chain']))
    res['tiers_passing_G1_G4'] = ts
    res['G5'] = g5
    B, Ct = res['tiers'].get('B', {}), res['tiers'].get('C', {})
    stop_B = bool('BAR' in B and min(B['dissipation']) > 20 and B['logistic_overlap'] < 1e-3)
    stop_C = bool('BAR' in B and 'BAR' in Ct and Ct['mean_dissipation'] >= 0.8 * B['mean_dissipation'])
    bond_stop = any(v.get('G1', {}).get('bond_breaks', 0) > 0 for v in res['tiers'].values())
    res['stop_rule_B'] = stop_B
    res['stop_rule_C_ge_0p8B'] = stop_C
    res['bond_break_stop'] = bond_stop
    (run / 'analysis.json').write_text(json.dumps(res, indent=1, default=float))
    short = {t: {k: v[k] for k in ('BAR', 'SE_root', 'SE_chain', 'mean_W', 'dissipation', 'mean_dissipation',
                                    'logistic_overlap', 'crooks_bins', 'ess', 'G1', 'G2', 'G3', 'G4')}
             if 'BAR' in v else v for t, v in res['tiers'].items()}
    log('ANALYSIS', json.dumps(dict(tiers=short, ratios=ratios, G5=g5, stop_rule_B=stop_B, stop_rule_C=stop_C,
                                     bond_break_stop=bond_stop), default=float))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MACE-OFF methane NES switching, tier C and three-tier analysis')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--roots', required=True, help='output directory of j1_endpoints.py')
    r.add_argument('--ref', required=True, help='output directory of j2_switch.py (tiers A, B)')
    r.add_argument('--out', required=True, help='new output directory (must not exist)')
    an = sub.add_parser('analyze')
    an.add_argument('run_dir')
    an.add_argument('--ref', required=True, help='output directory of j2_switch.py (tiers A, B)')
    a = ap.parse_args()
    REF = Path(a.ref)
    if a.cmd == 'run':
        ROOTS, RUN = Path(a.roots), Path(a.out)
        run()
        if (RUN / 'tierC-fwd-c00.npz').exists():
            analyze(RUN)
    else:
        analyze(a.run_dir)
