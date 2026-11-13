"""Hysteresis loops on the 21-point lambda grid of lagB_switch.py output (numpy only).

  python hysteresis_analysis.py LAGB_OUT_DIR --roots ROOTS_DIR [--out hysteresis-results.json]

For every structural quantity of lag_analysis.py and every grid point lambda = g / 20, loop = mean(reverse) -
mean(forward) with chain-clustered SE, z = loop / SE.  Also the first-peak position of the C-O g(r) (bootstrap SE).
Rules (fixed before the data existed), for each class (radial, orient, hbond):
  a grid point is flagged if max |z| over the class's quantities > 3; the class lags in the hard part if two adjacent
  grid points among lambda = 0.05 ... 0.5 are flagged.  Label: radial only -> "mainly geometric (radial)";
  orient / hbond only -> "mainly many-body rearrangement"; both -> "both"; neither -> "no distinguishable loop".
  Evaluated for shell radii 5.1 and 5.5 A; overall label = the common label, else "depends on shell radius".
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lag_analysis as LA  # noqa: E402

PRIMARY = dict(radial=['r_min', 'n_shell'], orient=['cos_dip', 'p2_oh'], hbond=['hb_shell', 'hb_shell_out'])
ZTH, HARD = 3.0, list(range(1, 11))
PBINS = np.arange(2.0, 6.01, 0.1)
NG = 21


def peak_pos(rs):
    """rs: list of per-config C-O distance arrays -> smoothed first-peak position of the mean histogram."""
    h = np.mean([np.histogram(r, PBINS)[0] for r in rs], 0)
    h = h / (4.0 / 3.0 * np.pi * (PBINS[1:] ** 3 - PBINS[:-1] ** 3))     # counts -> g(r) shape (raw counts would
    s = np.convolve(h, np.ones(3) / 3, mode='same')                     # bias the peak outward, ~ r^2)
    c = 0.5 * (PBINS[1:] + PBINS[:-1])
    return float(c[np.argmax(s)])


def load(run, d):
    fs = sorted(glob.glob(str(Path(run) / f'lagB-{d}-c*.npz')))
    parts = [dict(np.load(f)) for f in fs]
    cat = lambda k: np.concatenate([p[k] for p in parts])   # noqa: E731
    out = dict(frames=cat('frames'), W=cat('W'), Wbins=cat('Wbins'), chain=cat('chain_id'), lam=parts[0]['frame_lam'])
    g_of_frame = np.rint(out['lam'] * 20).astype(int)
    order = np.argsort(g_of_frame)                          # frames sorted by grid index 0..20
    out['frames'] = out['frames'][:, order]
    out['lam'] = out['lam'][order]
    return out


def analyze(run, rshell, roots):
    LA.RSHELL = rshell
    z0 = np.load(Path(roots) / 'roots_end0.npz')
    Z, L = z0['Z'], float(z0['L'])
    ci = int(np.where((Z == 6) & z0['solute'])[0][0])
    O = np.where((Z == 8) & ~z0['solute'])[0]
    D = {d: load(run, d) for d in ('fwd', 'rev')}
    keys = None
    feats, peaks, peaks_boot = {}, {}, {}
    rng = np.random.default_rng(0)
    for d, X in D.items():
        n = len(X['W'])
        F = np.empty((n, NG), object)
        RS = np.empty((n, NG), object)
        for i in range(n):
            for g in range(NG):
                x = X['frames'][i, g].astype(np.float64)
                f, _, _, ok = LA.features(x, Z, L, ci)
                assert ok
                F[i, g] = f
                RS[i, g] = np.linalg.norm(LA.mi(x[O] - x[ci], L), axis=1)
        keys = list(F[0, 0])
        feats[d] = {k: np.array([[F[i, g][k] for g in range(NG)] for i in range(n)]) for k in keys}
        peaks[d] = [peak_pos(list(RS[:, g])) for g in range(NG)]
        bs = np.array([[peak_pos(list(RS[rng.integers(0, n, n), g])) for g in range(NG)] for _ in range(1000)])
        peaks_boot[d] = bs.std(0, ddof=1).tolist()
    loops = {}
    for k in keys:
        rows = []
        for g in range(NG):
            mf, sf, _ = LA.stats(feats['fwd'][k][:, g], D['fwd']['chain'])
            mr, sr, _ = LA.stats(feats['rev'][k][:, g], D['rev']['chain'])
            dl = mr - mf
            se = float(np.hypot(sf, sr))
            rows.append(dict(lam=round(g / 20, 2), fwd=[mf, sf], rev=[mr, sr], delta=dl, z=dl / se if se > 0 else 0.0))
        loops[k] = rows
    pk = [dict(lam=round(g / 20, 2), fwd=[peaks['fwd'][g], peaks_boot['fwd'][g]], rev=[peaks['rev'][g], peaks_boot['rev'][g]],
               delta=peaks['rev'][g] - peaks['fwd'][g],
               z=(peaks['rev'][g] - peaks['fwd'][g]) / max(np.hypot(peaks_boot['fwd'][g], peaks_boot['rev'][g]), 1e-12))
          for g in range(NG)]
    # judgment
    cls = {}
    for c, ks in PRIMARY.items():
        zabs = np.array([[abs(loops[k][g]['z']) for g in range(NG)] for k in ks]).max(0)
        flag = zabs > ZTH
        adj = any(flag[g] and flag[g + 1] for g in HARD[:-1])
        onset = next((round(g / 20, 2) for g in range(1, NG - 1) if flag[g]), None)
        gmax = int(np.argmax(zabs[1:NG - 1])) + 1
        # equilibrium gap between the ends: roots are fwd frame 0 (lam = 0) and rev frame at lam = 1
        eq_gap = {k: abs(loops[k][NG - 1]['rev'][0] - loops[k][0]['fwd'][0]) for k in ks}
        norm = {k: [loops[k][g]['delta'] / eq_gap[k] if eq_gap[k] > 0 else None for g in range(NG)] for k in ks}
        cls[c] = dict(lag_in_hard=bool(adj), flagged_lams=[round(g / 20, 2) for g in range(NG) if flag[g]],
                      isolated_in_hard=[round(g / 20, 2) for g in HARD if flag[g] and not ((g > 1 and flag[g - 1]) or flag[g + 1])],
                      onset_lam=onset, max_lam=round(gmax / 20, 2), max_abs_z=float(zabs[gmax]),
                      zabs_profile=zabs.tolist(), normalized_delta=norm, eq_gap=eq_gap)
    r, o, h = cls['radial']['lag_in_hard'], cls['orient']['lag_in_hard'], cls['hbond']['lag_in_hard']
    label = (LA.LABEL_GEOM if r and not (o or h) else LA.LABEL_MANY if (not r) and (o or h) else
             'both' if r and (o or h) else 'no distinguishable loop')
    # dissipation per bin and loop profiles
    db = D['fwd']['Wbins'].mean(0) + D['rev']['Wbins'].mean(0)
    prof = {}
    for c in PRIMARY:
        zp = np.array(cls[c]['zabs_profile'])
        pb = 0.5 * (zp[:-1] + zp[1:])
        prof[c] = dict(spearman_with_dissipation=LA.spearman(db, pb), peak_bin_lam=[round(int(np.argmax(pb)) / 20, 2),
                       round(int(np.argmax(pb)) / 20 + 0.05, 2)], bin_profile=pb.tolist())
    # per trajectory: reverse work over lam < 0.3 vs quantities at lam = 0.2 (grid 4)
    wr = D['rev']['Wbins'][:, :6].sum(1)
    traj = {k: LA.spearman(wr, feats['rev'][k][:, 4]) for ks in PRIMARY.values() for k in ks}
    return dict(r_shell=rshell, n=[len(D['fwd']['W']), len(D['rev']['W'])], loops=loops, first_peak=pk, classes=cls,
                label=label, dissipation_bins=db.tolist(), dissipation_peak_bin_lam=round(int(np.argmax(db)) / 20, 2),
                profiles=prof, traj_rev_W_lam_lt_0p3_vs_q_at_0p2=traj,
                W_mean=[float(D['fwd']['W'].mean()), float(D['rev']['W'].mean())])


def main():
    ap = argparse.ArgumentParser(description='hysteresis loops on the lagB_switch.py frames')
    ap.add_argument('run_dir', help='output directory of lagB_switch.py')
    ap.add_argument('--roots', required=True, help='output directory of j1_endpoints.py')
    ap.add_argument('--out', default='hysteresis-results.json', help='output json file')
    a = ap.parse_args()
    run = Path(a.run_dir)
    res = {str(rs): analyze(run, rs, a.roots) for rs in (5.1, 5.5)}
    labs = {v['label'] for v in res.values()}
    res['overall'] = labs.pop() if len(labs) == 1 else 'depends on shell radius'
    res['run'] = str(run)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    for rs in ('5.1', '5.5'):
        v = res[rs]
        print(f'r_shell {rs}: label {v["label"]}; dissipation peak bin lam {v["dissipation_peak_bin_lam"]}')
        for c, x in v['classes'].items():
            print(f'  {c}: lag_in_hard={x["lag_in_hard"]} flagged={x["flagged_lams"]} onset={x["onset_lam"]} '
                  f'max |z|={x["max_abs_z"]:.1f} at {x["max_lam"]}; rho(diss)={v["profiles"][c]["spearman_with_dissipation"]:+.2f}')
        for k in [k for ks in PRIMARY.values() for k in ks]:
            print(f'   {k:13s} z:', ' '.join(f'{r["z"]:+.1f}' for r in v['loops'][k]))
        print('   first_peak  z:', ' '.join(f'{r["z"]:+.1f}' for r in v['first_peak']))
        print('   traj rho:', {k: round(x, 2) for k, x in v['traj_rev_W_lam_lt_0p3_vs_q_at_0p2'].items()})
    print('overall', res['overall'])


if __name__ == '__main__':
    main()
