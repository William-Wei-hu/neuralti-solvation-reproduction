"""Tiers A and B: bidirectional NES switching on the 4D-decoupling path of MACE-OFF23(S), float64 everywhere.
The kernel and dt are those chosen by j1_endpoints.py (K2 = HMC full refresh, L = 40, dt = 0.5 fs).
Interface: mace_iface.py (cueq).  Tier C and the three-tier analysis are in j3_switch.py.

  python j2_switch.py run A B --roots ROOTS_DIR --out OUT_DIR     |    python j2_switch.py analyze OUT_DIR

Stage (mlip_nes.make_switch, kind 'hmc', L = 40): work += [E_int(x, lam_{k+1}) - E_int(x, lam_k)] / kBT, then one HMC
  MH step (fresh momentum, 40 leapfrog steps of 0.5 fs, MH) at lam_{k+1}.  41 force calls and 20 fs per stage.
Schedule: linear, K = T / 20 fs.
  tier A: T = 2 ps,  K = 100;  tier B: T = 8 ps, K = 400;  tier C (j3_switch.py): T = 32 ps, K = 1600.
Roots: ROOTS_DIR/roots_end{0,1}.npz from j1_endpoints.py (80 per end) ordered by (segment, chain); tier A uses
  positions 0-31, B 32-63, C 64-79 of that order (each root used once).  Forward starts from end 0 roots, reverse from
  end 1 roots.
Keys: mlip_nes.stage_keys(20261112, direction, 10 * chain_id + segment, K).
Batch: chunks of 16 trajectories per device call; each chunk is written to OUT before the next starts.
Records per trajectory: W (kBT), W in 20 lambda bins, acceptance per stage, max n_edges, per stage min solute-solvent
  distance, max O-H, max C-H; final configuration.

GATES (fixed before the run; each tier separately; n_pairs = 32 in A, B, 16 in C):
  G1 numerics  : non-finite work, n_edges >= pad (the interface returns NaN), O-H > 1.5 A or C-H > 1.5 A anywhere:
                 all zero.  A bond break anywhere -> STOP the whole route (potential unstable).
  G2 overlap   : logistic overlap >= 0.1 and Crooks usable bins >= 3, else no estimate from this tier.
  G3 stuck     : a trajectory with >= 5 % of stages at acceptance < 0.05 is stuck; <= 2 per direction (scaled to
                 <= 1 for 16 pairs), else the tier is invalid.
  G4 heavy tail (reverse 1 -> 0): (a) trajectories with W outside median +- 10 MAD: <= 2 of 32 (<= 1 of 16);
                 (b) BAR change after removing the 2 reverse trajectories with the largest |W - median|: <= 2 SE
                 (changes after removing 1 and 3 are reported); (c) trajectories whose final configuration (lam = 0)
                 has a water atom closer than 1.5 A to the methane C: 0.  Any failure -> tier judged "heavy tail".
  G5 agreement : for tiers passing G1-G4: |BAR_i - BAR_j| <= 2 sqrt(SE_i^2 + SE_j^2) (SE = chain-clustered).
  STOP rule    : tier B with both dissipations > 20 kBT and logistic overlap < 1e-3 -> STOP (no tier C).
                 (tier C relative to B: mean dissipation C >= 0.8 x B -> STOP; evaluated in j3_switch.py.)
  Tier C precondition: no G1 bond break, no STOP rule in A, B, and tier B not "heavy tail".
  SE: cluster bootstrap 2000 draws, seed 0, (i) by root (one trajectory per root) and (ii) by endpoint chain (roots of
      the same chain are clustered); G4 / G5 use (ii), the larger in practice; both reported.
  One-sided Jarzynski: cited only with ESS >= 50; also listed against ESS >= 500.
  Coarse check (report only): expected dF(0 -> 1) = -3.38 kBT (Picha et al. +2.0 kcal/mol); tolerance max(1, 3 SE).
  Dissipation profile: per lambda bin, mean_F(bin) + mean_R(bin) (both in the same t bins).
Time guard: a chunk is not started if its projected end is later than --time-limit-min minus 8 min (default 90).
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
TIERS = dict(A=dict(T=2000.0, sl=slice(0, 32)), B=dict(T=8000.0, sl=slice(32, 64)), C=dict(T=32000.0, sl=slice(64, 80)))
CHUNK = 16
ROOTS = RUN = None                             # set from the command line
T_LIMIT = 90 * 60
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


def run(tiers):
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
    meta = dict(info=m.info, roots=str(ROOTS), kernel='K2 hmc L=40', dt=DT, chunk=CHUNK,
                tiers={k: dict(T_fs=v['T'], K=int(round(v['T'] / STAGE_FS)), roots=[v['sl'].start, v['sl'].stop])
                       for k, v in TIERS.items()}, chunks=[])
    (RUN / 'meta.json').write_text(json.dumps(meta, indent=1, default=float))
    sw = M.make_switch(efn, kind='hmc', m=mass, dt=DT, L=L_LEAP, box=L, kT=KT, nbins=NBINS, record=False, probes=probes)
    swb = jax.jit(jax.vmap(sw, in_axes=(0, 0, None, 0)))
    per_call = None
    for t in tiers:
        cfg = TIERS[t]
        K = int(round(cfg['T'] / STAGE_FS))
        for d, e in (('fwd', 0), ('rev', 1)):
            R = roots[e]
            idx = np.arange(cfg['sl'].start, cfg['sl'].stop)
            lams = jnp.asarray(M.schedule(K, d))
            for c0 in range(0, len(idx), CHUNK):
                ci = idx[c0:c0 + CHUNK]
                if per_call is not None:
                    proj = len(ci) * (K * (L_LEAP + 1) + 1) * per_call
                    if time.time() - T0 + proj > T_LIMIT - 8 * 60:
                        log(f'time guard: tier {t} {d} chunk {c0} not started (projected {proj:.0f}s)')
                        meta['time_guard'] = f'stopped before tier {t} {d} chunk {c0}'
                        (RUN / 'meta.json').write_text(json.dumps(meta, indent=1, default=float))
                        return
                rid = 10 * R['chain_id'][ci] + R['segment'][ci]
                keys = jnp.stack([M.stage_keys(SEED, d, int(r), K) for r in rid])
                p0 = jax.vmap(lambda k: M.draw_p(k, mass, KT, jnp.float64))(keys[:, 0])
                ts = time.time()
                out = jax.block_until_ready(swb(jnp.asarray(R['x'][ci]), p0, lams, keys[:, 1:]))
                sec = time.time() - ts
                calls = len(ci) * (K * (L_LEAP + 1) + 1)
                per_call = sec / calls                      # the first chunk includes compilation (conservative)
                x = np.asarray(out['x'])
                dC = M.min_image(x[:, widx] - x[:, [C]], L)
                dC = np.sqrt((dC ** 2).sum(-1)).min(1)
                rec = dict(W=np.asarray(out['W']), Wbins=np.asarray(out['Wbins']), acc=np.asarray(out['acc']),
                           n_edges_max=np.asarray(out['n_edges_max']), nonfinite=np.asarray(out['nonfinite']),
                           min_ss=np.asarray(out['min_solute_solvent']).min(1),
                           max_oh=np.asarray(out['max_oh']).max(1), max_ch=np.asarray(out['max_ch']).max(1),
                           final_C_water=dC, x_final=x.astype(np.float32), chain_id=R['chain_id'][ci],
                           segment=R['segment'][ci], root_pos=ci, seconds=np.float64(sec), calls=np.int64(calls))
                f = RUN / f'tier{t}-{d}-c{c0 // CHUNK:02d}.npz'
                np.savez(f, **rec)
                meta['chunks'].append(dict(file=f.name, seconds=sec, calls=calls, ms_per_config=1e3 * sec / calls))
                (RUN / 'meta.json').write_text(json.dumps(meta, indent=1, default=float))
                log(f'tier {t} {d} chunk {c0 // CHUNK}: <W>={rec["W"].mean():.3f} sd={rec["W"].std():.3f} '
                    f'acc={np.nanmean(rec["acc"]):.3f} edges={int(rec["n_edges_max"].max())} '
                    f'nonfinite={int(rec["nonfinite"].sum())} oh={rec["max_oh"].max():.3f} '
                    f'{sec:.0f}s {1e3 * sec / calls:.2f} ms/config')


# ------------------------------------------------------------------------------------------------ analysis
def load_tier(run, t, d):
    fs = sorted(run.glob(f'tier{t}-{d}-c*.npz'))
    if not fs:
        return None
    parts = [dict(np.load(f)) for f in fs]
    return {k: (np.concatenate([p[k] for p in parts]) if np.ndim(parts[0][k]) else np.array([p[k] for p in parts]))
            for k in parts[0]}


def analyze(run):
    import estimators as E
    run = Path(run)
    meta = json.loads((run / 'meta.json').read_text())
    res = dict(meta=meta, tiers={})
    for t in ('A', 'B', 'C'):
        F, R = load_tier(run, t, 'fwd'), load_tier(run, t, 'rev')
        if F is None or R is None:
            continue
        npairs = 32 if t in 'AB' else 16
        lim = 2 if npairs >= 32 else 1
        rid_f = 1000 * 0 + 10 * F['chain_id'] + F['segment']
        rid_r = 10 * R['chain_id'] + R['segment']
        try:
            s_root = E.summarize(F['W'], R['W'], rid_f, rid_r, n_boot=2000, seed=0, truth=-3.38)
            s_chain = E.summarize(F['W'], R['W'], F['chain_id'], R['chain_id'], n_boot=2000, seed=0, truth=-3.38)
        except Exception as ex:                   # e.g. BAR root not bracketed
            res['tiers'][t] = dict(error=repr(ex))
            continue
        p = s_root['point']
        se_root, se_chain = s_root['bootstrap']['BAR']['se'], s_chain['bootstrap']['BAR']['se']
        se = max(se_root, se_chain)
        fin = bool(np.isfinite(F['W']).all() and np.isfinite(R['W']).all())
        nonf = int(F['nonfinite'].sum() + R['nonfinite'].sum())
        pad = int(meta['info']['pad'])
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
        prof = (F['Wbins'].mean(0) + R['Wbins'].mean(0)).tolist()
        os_ = {side: dict(value=p[k], se=s_root['bootstrap'][k]['se'], ess=p[e_], cite_50=bool(p[e_] >= 50),
                          cite_500=bool(p[e_] >= 500))
               for side, k, e_ in (('forward', 'forward_jarzynski', 'ess_forward'),
                                   ('reverse', 'reverse_jarzynski', 'ess_reverse'))}
        res['tiers'][t] = dict(
            n=[len(F['W']), len(R['W'])], BAR=p['BAR'], SE_root=se_root, SE_chain=se_chain,
            mean_W=[p['mean_WF'], p['mean_WR']], sd_W=[p['sd_WF'], p['sd_WR']], dissipation=diss,
            mean_dissipation=float(np.mean(diss)), logistic_overlap=p['logistic_overlap_sum'],
            histogram_overlap=p['histogram_overlap'], crooks_bins=s_root['crooks']['bins_used'],
            ess=[p['ess_forward'], p['ess_reverse']], one_sided=os_,
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
            dissipation_profile_20bins=prof,
            acc_mean=[float(np.nanmean(F['acc'])), float(np.nanmean(R['acc']))],
            gpu_seconds=float(F['seconds'].sum() + R['seconds'].sum()),
            calls=int(F['calls'].sum() + R['calls'].sum()))
    ok = {t: v for t, v in res['tiers'].items() if 'BAR' in v and v['G1']['passed'] and v['G2']['passed']
          and v['G3']['passed'] and v['G4']['passed']}
    g5 = {}
    ts = sorted(ok)
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            a, b = ok[ts[i]], ok[ts[j]]
            g5[f'{ts[i]}-{ts[j]}'] = bool(abs(a['BAR'] - b['BAR']) <= 2 * np.hypot(a['SE_chain'], b['SE_chain']))
    res['G5'] = g5
    B = res['tiers'].get('B')
    stop_B = bool(B and 'BAR' in B and min(B['dissipation']) > 20 and B['logistic_overlap'] < 1e-3)
    bond_stop = any(v.get('G1', {}).get('bond_breaks', 0) > 0 for v in res['tiers'].values())
    res['stop_rule_B'] = stop_B
    res['bond_break_stop'] = bond_stop
    res['tierC_precondition'] = bool(not bond_stop and not stop_B and B is not None and 'BAR' in B and B['G4']['passed'])
    (run / 'analysis.json').write_text(json.dumps(res, indent=1, default=float))
    short = {t: {k: v[k] for k in ('BAR', 'SE_root', 'SE_chain', 'mean_W', 'dissipation', 'logistic_overlap',
                                    'crooks_bins', 'ess', 'G1', 'G2', 'G3', 'G4')} if 'BAR' in v else v
             for t, v in res['tiers'].items()}
    log('ANALYSIS', json.dumps(dict(tiers=short, G5=g5, stop_rule_B=stop_B, bond_break_stop=bond_stop,
                                     tierC_precondition=res['tierC_precondition']), default=float))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MACE-OFF methane NES switching, tiers A and B')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('tiers', nargs='+', choices=['A', 'B', 'C'])
    r.add_argument('--roots', required=True, help='output directory of j1_endpoints.py')
    r.add_argument('--out', required=True, help='new output directory (must not exist)')
    r.add_argument('--time-limit-min', type=float, default=90.0)
    an = sub.add_parser('analyze')
    an.add_argument('run_dir')
    a = ap.parse_args()
    if a.cmd == 'run':
        ROOTS, RUN, T_LIMIT = Path(a.roots), Path(a.out), 60 * a.time_limit_min
        run(a.tiers)
        analyze(RUN)
    else:
        analyze(a.run_dir)
