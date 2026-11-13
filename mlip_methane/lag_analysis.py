"""Structural lag at the end of NES switching (MACE-OFF23(S), CH4 in 216 water).  numpy only; reads saved runs only.

  python lag_analysis.py --roots ROOTS_DIR --j2 J2_OUT_DIR --j3 J3_OUT_DIR --out results.json [--r-shell 5.5]

Compares structural quantities of the final configurations of the switching runs (tiers A, B from j2_switch.py, C from
j3_switch.py) with the equilibrium roots of j1_endpoints.py.  Quantities (per configuration, around the methane C):
  radial  n_shell (waters with r_CO < r_shell), r_min (nearest C-O distance)
  orient  cos_dip (dipole vs radial direction, shell average), p2_oh (P2 of O-H vs radial direction)
  hbond   hb_shell (H bonds per shell water), hb_shell_out (shell-to-outside H bonds per shell water)
  H bond: R_OO < 3.5 A and angle H-O_donor..O_acceptor < 30 deg.
Judgment rule (fixed before any number was computed; reverse direction): a class is "not in place" if any of its
  quantities has |z| > 2 against the target-end equilibrium.  Radial only -> "mainly geometric (radial)"; orient or
  hbond only -> "mainly many-body rearrangement"; otherwise "not distinguishable from final states".
Sensitivity (reported only, not used for the judgment): --r-shell 5.1.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

RSHELL, RIN, ROO, HBANG = 5.5, 4.0, 3.5, 30.0
COS30 = np.cos(np.deg2rad(HBANG))
PRIMARY = dict(radial=['n_shell', 'r_min'], orient=['cos_dip', 'p2_oh'], hbond=['hb_shell', 'hb_shell_out'])
TIER_FILES = dict(A=('j2', 'A'), B=('j2', 'B'), C=('j3', 'C'))
RBINS = np.arange(0.0, 9.01, 0.25)
LABEL_GEOM, LABEL_MANY, LABEL_NONE = ('mainly geometric (radial)', 'mainly many-body rearrangement',
                                      'not distinguishable from final states')


def mi(d, L):
    return d - L * np.round(d / L)


def features(x, Z, L, ci):
    x = np.asarray(x, np.float64)
    sol = np.zeros(len(Z), bool)
    sol[:5] = True
    O = np.where((Z == 8) & ~sol)[0]
    H = np.where((Z == 1) & ~sol)[0]
    dHO = mi(x[H][:, None] - x[O][None], L)
    j = np.argmin((dHO ** 2).sum(-1), 1)
    ok = bool(np.all(np.bincount(j, minlength=len(O)) == 2))
    hh = np.stack([H[j == k] for k in range(len(O))]) if ok else None
    xO = x[O]
    vOH = np.stack([mi(x[hh[:, 0]] - xO, L), mi(x[hh[:, 1]] - xO, L)], 1)      # (nO, 2, 3)
    rCO_v = mi(xO - x[ci], L)
    rCO = np.linalg.norm(rCO_v, axis=1)
    shell = rCO < RSHELL
    u = rCO_v / rCO[:, None]
    dip = vOH.sum(1)
    cos_dip = (dip * u).sum(1) / np.linalg.norm(dip, axis=1)
    cos_oh = (vOH * u[:, None]).sum(-1) / np.linalg.norm(vOH, axis=-1)       # (nO, 2)
    dOO = mi(xO[None] - xO[:, None], L)                                         # [d, a] = O_a - O_d
    rOO = np.linalg.norm(dOO, axis=-1)
    np.fill_diagonal(rOO, np.inf)
    hb = np.zeros((len(O), len(O)), bool)
    for k in range(2):
        v = vOH[:, k]
        c = (v[:, None] * dOO).sum(-1) / (np.linalg.norm(v, axis=1)[:, None] * rOO)
        hb |= (rOO < ROO) & (c > COS30)
    hbs = hb | hb.T                                                             # undirected pair bonded
    nhb = hb.sum(1) + hb.sum(0)                                                 # donor + acceptor count
    out = ~shell
    ns = int(shell.sum())
    f = dict(
        n_shell=float(ns), r_min=float(rCO.min()), n_in4=float((rCO < RIN).sum()),
        cos_dip=float(cos_dip[shell].mean()) if ns else np.nan,
        p2_oh=float((1.5 * cos_oh[shell] ** 2 - 0.5).mean()) if ns else np.nan,
        frac_oh_in=float((cos_oh[shell] < -0.5).mean()) if ns else np.nan,
        frac_oh_out=float((cos_oh[shell] > 0.5).mean()) if ns else np.nan,
        hb_shell=float(nhb[shell].mean()) if ns else np.nan,
        hb_shell_out=float((hb[shell][:, out].sum() + hb[out][:, shell].sum()) / ns) if ns else np.nan,
        hb_shell_in=float(hb[shell][:, shell].sum() / ns) if ns else np.nan,
        hb_bulk=float(nhb[out].mean()),
    )
    gr = np.histogram(rCO, RBINS)[0].astype(float)
    hist = dict(cos_dip=np.histogram(cos_dip[shell], np.linspace(-1, 1, 11))[0].tolist(),
                cos_oh=np.histogram(cos_oh[shell].ravel(), np.linspace(-1, 1, 11))[0].tolist())
    return f, gr, hist, ok


def stats(vals, groups):
    v = np.asarray(vals, float)
    m = np.nanmean(v)
    n = np.isfinite(v).sum()
    se_iid = np.nanstd(v, ddof=1) / np.sqrt(n)
    ug = np.unique(groups)
    if len(ug) >= 3 and len(ug) < n:
        gm = np.array([np.nanmean(v[groups == g]) for g in ug])
        se_cl = np.std(gm, ddof=1) / np.sqrt(len(ug))
    else:
        se_cl = se_iid
    return float(m), float(max(se_iid, se_cl)), int(n)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main(roots, dirs, out):
    TIERS = {t: (dirs[src], tt) for t, (src, tt) in TIER_FILES.items()}
    z0 = np.load(Path(roots) / 'roots_end0.npz')
    Z, L = z0['Z'], float(z0['L'])
    ci = int(np.where((Z == 6) & z0['solute'])[0][0])
    res = dict(settings=dict(r_shell=RSHELL, hb=f'R_OO<{ROO} A, angle H-Od..Oa<{HBANG} deg', L=L), eq={}, tiers={},
               water_assignment_failures=0)
    keys = None
    eq = {}
    for e in (0, 1):
        z = np.load(Path(roots) / f'roots_end{e}.npz')
        F, G = [], []
        for xx in z['x']:
            f, gr, h, ok = features(xx, Z, L, ci)
            res['water_assignment_failures'] += int(not ok)
            F.append(f)
            G.append(gr)
        keys = list(F[0])
        eq[e] = dict(F={k: np.array([f[k] for f in F]) for k in keys}, chain=z['chain_id'], gr=np.mean(G, 0))
        res['eq'][f'end{e}'] = {k: stats(eq[e]['F'][k], z['chain_id']) for k in keys}
    shellvol = 4 / 3 * np.pi * (RBINS[1:] ** 3 - RBINS[:-1] ** 3)
    rho = 216 / L ** 3
    res['gr'] = dict(r=((RBINS[1:] + RBINS[:-1]) / 2).tolist(),
                     end0=(eq[0]['gr'] / shellvol / rho).tolist(), end1=(eq[1]['gr'] / shellvol / rho).tolist())
    for t, (run, tt) in TIERS.items():
        res['tiers'][t] = {}
        for d, target, start in (('fwd', 1, 0), ('rev', 0, 1)):
            fs = sorted(glob.glob(str(Path(run) / f'tier{tt}-{d}-c*.npz')))
            W, chains, F, G = [], [], [], []
            for fn in fs:
                z = np.load(fn)
                W.append(z['W'])
                chains.append(z['chain_id'])
                for xx in z['x_final']:
                    f, gr, h, ok = features(xx, Z, L, ci)
                    res['water_assignment_failures'] += int(not ok)
                    F.append(f)
                    G.append(gr)
            W = np.concatenate(W)
            chains = np.concatenate(chains)
            Fa = {k: np.array([f[k] for f in F]) for k in keys}
            rows = {}
            for k in keys:
                m, se, n = stats(Fa[k], chains)
                me, see, _ = eq[target]['F'][k].mean(), res['eq'][f'end{target}'][k][1], None
                ms, ses = res['eq'][f'end{start}'][k][0], res['eq'][f'end{start}'][k][1]
                zval = (m - me) / np.hypot(se, see)
                gap = ms - me
                frac = float(1 - (m - me) / gap) if abs(gap) > 2 * np.hypot(ses, see) else None
                ok = np.isfinite(Fa[k])
                rows[k] = dict(final=[m, se, n], eq_target=[float(me), see], eq_start=[ms, ses], z=float(zval),
                               frac_covered=frac,
                               pearson_W=float(np.corrcoef(W[ok], Fa[k][ok])[0, 1]),
                               spearman_W=spearman(W[ok], Fa[k][ok]))
            res['tiers'][t][d] = dict(n=len(W), W_mean=float(W.mean()), q=rows,
                                      gr_final=(np.mean(G, 0) / shellvol / rho).tolist(),
                                      _W=W.tolist(), _F={k: Fa[k].tolist() for k in keys})
    # judgment (reverse direction), rule in the module docstring
    judg = {}
    for t in TIERS:
        q = res['tiers'][t]['rev']['q']
        notin = {c: any(abs(q[k]['z']) > 2 for k in ks) for c, ks in PRIMARY.items()}
        if notin['radial'] and not notin['orient'] and not notin['hbond']:
            lab = LABEL_GEOM
        elif not notin['radial'] and (notin['orient'] or notin['hbond']):
            lab = LABEL_MANY
        else:
            lab = LABEL_NONE
        judg[t] = dict(not_in_place=notin, label=lab, z={k: q[k]['z'] for ks in PRIMARY.values() for k in ks})
    labs = {v['label'] for v in judg.values()}
    res['judgment'] = dict(per_tier=judg, overall=labs.pop() if len(labs) == 1 else LABEL_NONE + ' (tiers disagree)')
    # pooled correlation (reverse), W and quantities standardized within tier
    pool = {}
    for k in keys:
        a, b = [], []
        for t in TIERS:
            Wt = np.array(res['tiers'][t]['rev']['_W'])
            qt = np.array(res['tiers'][t]['rev']['_F'][k])
            if np.nanstd(qt) == 0:
                continue
            a.append((Wt - Wt.mean()) / Wt.std())
            b.append((qt - np.nanmean(qt)) / np.nanstd(qt))
        a, b = np.concatenate(a), np.concatenate(b)
        ok = np.isfinite(b)
        pool[k] = dict(pearson=float(np.corrcoef(a[ok], b[ok])[0, 1]), spearman=spearman(a[ok], b[ok]), n=int(ok.sum()))
    res['pooled_rev_corr'] = pool
    Path(out).write_text(json.dumps(res, indent=1))
    # summary
    print('assignment failures', res['water_assignment_failures'])
    print('eq end0', {k: np.round(v[:2], 3).tolist() for k, v in res['eq']['end0'].items()})
    print('eq end1', {k: np.round(v[:2], 3).tolist() for k, v in res['eq']['end1'].items()})
    for t in TIERS:
        for d in ('fwd', 'rev'):
            q = res['tiers'][t][d]['q']
            print(t, d, ' '.join(f"{k}:{q[k]['final'][0]:.3f}±{q[k]['final'][1]:.3f} z={q[k]['z']:+.1f} "
                                 f"r={q[k]['pearson_W']:+.2f}" for k in keys))
    print(json.dumps(res['judgment']))
    print('pooled', {k: round(v['pearson'], 2) for k, v in pool.items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='structural lag at the end of switching')
    ap.add_argument('--roots', required=True, help='output directory of j1_endpoints.py')
    ap.add_argument('--j2', required=True, help='output directory of j2_switch.py (tiers A, B)')
    ap.add_argument('--j3', required=True, help='output directory of j3_switch.py (tier C)')
    ap.add_argument('--out', required=True, help='output json file')
    ap.add_argument('--r-shell', type=float, default=5.5)
    a = ap.parse_args()
    RSHELL = a.r_shell
    main(a.roots, dict(j2=a.j2, j3=a.j3), a.out)
