"""Analyse forward/reverse NES runs: estimates, dissipation, overlap, ESS, Crooks plot, TI comparison.

  python analyze_nes.py --runs ../runs/base-mala-K400-M5 ../runs/base-mala-K4000-M5 ... \
      --out ../analysis/test-c-base.json [--ti-log <standardTI log>]

Per run dir (fwd.npz + rev.npz) writes <run>/summary.json, <run>/crooks.png, <run>/profile.png.
The combined JSON adds: TI reference, BAR - TI, internal consistency of BAR across protocol lengths,
and the sign-flip negative control on the real data.
Test (c) criterion (fixed before running): for the longest protocol with Crooks bins_used >= 3,
|BAR - TI_65| <= 3*SE_BAR + 0.5 kBT  (0.5 kBT allowance: TI from the log has no error bar);
also reported against the paper range [-21, -19] kBT.  The same rule is reported per path in 'vs_TI'
(dF does not depend on the path); 'test_c' is the base-path entry.

  python analyze_nes.py --runs ../runs/learned-hmc-G2400-M1 ../runs/learned-hmc-G24000-M1 \
      --compare-with ../runs/base-hmc-G2400-M1 ../runs/base-hmc-G24000-M1 --out ../analysis/learned-D.json
adds 'path_comparison': same kernel / gradients per trajectory, learned vs base dissipation, overlap, ESS
fraction and BAR variance x GPU seconds (the --compare-with summary.json files are only read).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import estimators as E
import ti_reference

PAPER_TI_RANGE = (-21.0, -19.0)


def load(run):
    out = {}
    for d in ('fwd', 'rev'):
        with np.load(Path(run) / f'{d}.npz') as z:
            out[d] = {k: z[k] for k in z.files}
            out[d]['meta'] = json.loads(str(out[d]['meta']))
    return out


def plots(run, r, s, ti):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    wf, wr = r['fwd']['W'], r['rev']['W']
    p = s['point']
    fig, ax = plt.subplots(figsize=(6, 4))
    lo, hi = min(wf.min(), (-wr).min()), max(wf.max(), (-wr).max())
    bins = np.linspace(lo, hi, 50)
    ax.hist(wf, bins, density=True, alpha=.5, label=r'$W_F$')
    ax.hist(-wr, bins, density=True, alpha=.5, label=r'$-W_R$')
    ax.axvline(p['BAR'], color='k', lw=1, label=f"BAR {p['BAR']:.2f}")
    if ti is not None:
        ax.axvline(ti, color='r', ls='--', lw=1, label=f'TI {ti:.2f}')
    ax.set_xlabel('work / kBT')
    ax.set_title(Path(run).name)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(run) / 'crooks.png', dpi=120)
    plt.close(fig)

    nb = r['fwd']['Wbins'].shape[1]
    tc = (np.arange(nb) + 0.5) / nb
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tc, r['fwd']['Wbins'].mean(0) * nb, label='forward <dW>/dt')
    ax.plot(tc, -r['rev']['Wbins'].mean(0) * nb, label='reverse -<dW>/dt')
    if s.get('ti_slices'):
        t = np.array([float(k) for k in s['ti_slices']])
        ax.plot(t, list(s['ti_slices'].values()), 'k.', ms=3, label='TI <dU/dt> (log)')
    ax.set_xlabel('t')
    ax.set_ylabel('kBT')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(run) / 'profile.png', dpi=120)
    plt.close(fig)


def analyze_run(run, n_boot, seed, ti):
    r = load(run)
    wf, wr = r['fwd']['W'], r['rev']['W']
    rf, rr = r['fwd']['root_id'], r['rev']['root_id']
    s = E.summarize(wf, wr, rf, rr, n_boot=n_boot, seed=seed)
    s['protocol'] = {k: r['fwd']['meta'].get(k) for k in ('path', 'K', 'M', 'kernel', 'L', 'grads_per_stage', 'h',
                                                          'h_source', 'cap_factor', 'n_roots', 'reps', 'precision')}
    if r['fwd']['meta'].get('h') != r['rev']['meta'].get('h') or r['fwd']['meta'].get('L') != r['rev']['meta'].get('L'):
        s['WARNING'] = 'forward and reverse used different kernel parameters: Crooks/BAR pairing invalid'
    s['cost'] = {d: dict(wall_seconds=r[d]['meta']['wall_seconds'], us_per_traj_stage=r[d]['meta']['us_per_traj_stage'])
                 for d in ('fwd', 'rev')}
    s['acceptance'] = {d: dict(mean=float(r[d]['acc_traj_mean'].mean()),
                               per_stage_min=float(r[d]['acc'].astype(float).mean(0).min()),
                               per_traj_min=float(r[d]['acc_traj_mean'].min())) for d in ('fwd', 'rev')}
    s['nonfinite_increments'] = {d: int(r[d]['nonfinite'].sum()) for d in ('fwd', 'rev')}
    if r['fwd']['meta']['path'] == 'learned':
        s['gnn_max_edges'] = {d: float(r[d]['auxmax'].max()) for d in ('fwd', 'rev')}
        s['gnn_size_to_pad'] = r['fwd']['meta']['path_info'].get('size_to_pad')
    flip = E.point_estimates(-wf, -wr)
    s['signflip_control'] = {k: flip[k] for k in ('BAR', 'forward_jarzynski', 'reverse_jarzynski')}
    s['jarzynski_one_sided_se'] = dict(
        forward=E.one_sided_bootstrap(wf, rf, E.jarzynski_forward, n_boot, seed + 1),
        reverse=E.one_sided_bootstrap(wr, rr, E.jarzynski_reverse, n_boot, seed + 2))
    nb = r['fwd']['Wbins'].shape[1]
    s['profile'] = dict(t_bin_centers=((np.arange(nb) + .5) / nb).tolist(),
                        forward_mean_dWdt=(r['fwd']['Wbins'].mean(0) * nb).tolist(),
                        reverse_minus_mean_dWdt=(-r['rev']['Wbins'].mean(0) * nb).tolist())
    s['ti_slices'] = ti.get('slices') if ti else None
    try:
        plots(run, r, s, ti.get('TI') if ti else None)
    except Exception as e:  # plotting must not kill the analysis
        s['plot_error'] = repr(e)
    E.atomic_json(Path(run) / 'summary.json', s)
    return s


def make_row(s):
    """One table row from a run summary (fresh from analyze_run, or an existing summary.json)."""
    p, b, pr = s['point'], s['bootstrap'], s['protocol']
    n_traj = int(pr.get('n_roots') or 0) * int(pr.get('reps') or 1)
    gpu = {d: c['us_per_traj_stage'] * 1e-6 * n_traj * pr['K'] for d, c in s['cost'].items()}
    gpu_total = float(sum(gpu.values()))
    return dict(
        path=pr.get('path'), K=pr['K'], M=pr['M'], kernel=pr['kernel'], h=pr['h'], L=pr['L'],
        grads_per_traj=(pr['grads_per_stage'] or 0) * pr['K'], n_roots=pr.get('n_roots'), reps=pr.get('reps'),
        jarzynski_se_caveat='kernel_checks.py: at ESS ~1e2-1e3 the bootstrap SE of one-sided exponential '
                            'estimators was 1.6-1.8x too small and they carry finite-sample bias; judge with BAR',
        BAR=p['BAR'], BAR_se=b['BAR']['se'], forward=p['forward_jarzynski'], forward_se=b['forward_jarzynski']['se'],
        reverse=p['reverse_jarzynski'], reverse_se=b['reverse_jarzynski']['se'],
        dissipation_forward=p['dissipation_forward'], dissipation_reverse=p['dissipation_reverse'],
        logistic_overlap_sum=p['logistic_overlap_sum'], histogram_overlap=p['histogram_overlap'],
        ess_forward=p['ess_forward'], ess_reverse=p['ess_reverse'],
        ess_fraction=dict(forward=p['ess_forward'] / n_traj, reverse=p['ess_reverse'] / n_traj) if n_traj else None,
        crooks_bins_used=s['crooks']['bins_used'], crooks_dF=s['crooks']['dF_slope1'], acc=s['acceptance'],
        cost=s['cost'], gpu_seconds=gpu, gpu_seconds_total=gpu_total,
        BAR_se_x_sqrt_traj=b['BAR']['se'] * np.sqrt(n_traj),
        BAR_var_x_gpu_seconds=b['BAR']['se'] ** 2 * gpu_total,
        signflip_BAR=s['signflip_control']['BAR'], gnn_max_edges=s.get('gnn_max_edges'),
        gnn_size_to_pad=s.get('gnn_size_to_pad'), nonfinite_increments=s.get('nonfinite_increments'))


def vs_ti(items, ti):
    """Longest protocol (by gradients) with Crooks histogram overlap; |BAR - TI| <= 3 SE + 0.5 kBT.
    dF is path independent, so the same rule applies to the learned path."""
    usable = [(k, v) for k, v in items if v['crooks_bins_used'] >= 3]
    if not usable:
        return dict(passed=False, reason='no run with Crooks histogram overlap (bins_used>=3)')
    k, v = max(usable, key=lambda kv: kv[1]['grads_per_traj'] or kv[1]['K'] * kv[1]['M'])
    crit = dict(run=k, BAR=v['BAR'], BAR_se=v['BAR_se'],
                in_paper_range=PAPER_TI_RANGE[0] <= v['BAR'] <= PAPER_TI_RANGE[1])
    if 'TI' in ti:
        tol = 3 * v['BAR_se'] + 0.5
        crit.update(TI=ti['TI'], TI_n_slices=ti['best_n'], TI_complete=ti['complete'],
                    diff=v['BAR'] - ti['TI'], tolerance=tol, passed=abs(v['BAR'] - ti['TI']) <= tol)
    return crit


COMPARE_KEYS = ('path', 'n_roots', 'K', 'BAR', 'BAR_se', 'dissipation_forward', 'dissipation_reverse',
                'logistic_overlap_sum', 'histogram_overlap', 'ess_fraction', 'crooks_bins_used', 'gpu_seconds_total',
                'BAR_se_x_sqrt_traj', 'BAR_var_x_gpu_seconds')


def path_comparison(ok, ref):
    """Pair every analysed run with every run of a DIFFERENT path at the same kernel, M, L and gradients per
    trajectory.  Efficiency: BAR variance x total GPU seconds (fwd + rev), lower is better; it is independent of
    the number of trajectories, so runs with different root counts can be compared."""
    pool = dict(ok) | {k: v for k, v in ref.items() if not v.get('missing')}
    out = []
    for k, v in ok:
        for kr, vr in pool.items():
            if kr == k or vr['path'] == v['path']:
                continue
            if (vr['kernel'], vr['M'], vr['L'], vr['grads_per_traj']) != (v['kernel'], v['M'], v['L'], v['grads_per_traj']):
                continue
            se = float(np.hypot(v['BAR_se'], vr['BAR_se']))
            mean_diss = lambda r: 0.5 * (r['dissipation_forward'] + r['dissipation_reverse'])  # noqa: E731
            out.append(dict(
                grads_per_traj=v['grads_per_traj'], kernel=v['kernel'], L=v['L'], this=k, ref=kr,
                rows={k: {c: v[c] for c in COMPARE_KEYS}, kr: {c: vr[c] for c in COMPARE_KEYS}},
                BAR_diff=v['BAR'] - vr['BAR'], BAR_diff_z=(v['BAR'] - vr['BAR']) / se if se > 0 else None,
                mean_dissipation_ratio_this_over_ref=mean_diss(v) / mean_diss(vr) if mean_diss(vr) else None,
                gpu_seconds_per_traj_ratio_this_over_ref=(v['gpu_seconds_total'] / (v['n_roots'] * v['reps'])) /
                (vr['gpu_seconds_total'] / (vr['n_roots'] * vr['reps'])),
                var_x_gpu_ratio_this_over_ref=v['BAR_var_x_gpu_seconds'] / vr['BAR_var_x_gpu_seconds']
                if vr['BAR_var_x_gpu_seconds'] else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--compare-with', nargs='*', default=[],
                    help='already analysed run dirs (their summary.json is read, never rewritten) for path_comparison')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ti-log', default=str(ti_reference.DEFAULT_LOG))
    ap.add_argument('--n-boot', type=int, default=2000)
    a = ap.parse_args()
    try:
        ti = ti_reference.reference(a.ti_log)
    except Exception as e:
        ti = dict(error=repr(e))
    rows = {}
    for i, run in enumerate(a.runs):
        if not ((Path(run) / 'fwd.npz').exists() and (Path(run) / 'rev.npz').exists()):
            rows[Path(run).name] = dict(missing=True)
            continue
        s = analyze_run(run, a.n_boot, 4242 + 10 * i, ti if 'TI' in ti else None)
        rows[Path(run).name] = make_row(s)
        print(Path(run).name, json.dumps(E.json_safe(rows[Path(run).name])), flush=True)
    ref = {}
    for run in a.compare_with:
        f = Path(run) / 'summary.json'
        ref[Path(run).name] = make_row(json.loads(f.read_text())) if f.exists() else dict(missing=True)
    ok = [(k, v) for k, v in rows.items() if not v.get('missing')]
    comb = dict(runs=rows, compare_with=ref, ti_reference=ti, paper_TI_range=PAPER_TI_RANGE)
    if ok:
        # internal consistency: every pair of BAR estimates (dF is path independent, so cross-path pairs count too)
        pairs = []
        for i in range(len(ok)):
            for j in range(i + 1, len(ok)):
                (ka, va), (kb, vb) = ok[i], ok[j]
                se = float(np.hypot(va['BAR_se'], vb['BAR_se']))
                pairs.append(dict(a=ka, b=kb, diff=va['BAR'] - vb['BAR'], se=se,
                                  z=(va['BAR'] - vb['BAR']) / se if se > 0 else None))
        comb['BAR_pairwise'] = pairs
        by_path = {}
        for k, v in ok:
            by_path.setdefault(v['path'], []).append((k, v))
        comb['vs_TI'] = {p: vs_ti(items, ti) for p, items in by_path.items()}
        if 'base' in comb['vs_TI']:
            comb['test_c'] = comb['vs_TI']['base']
        if ref:
            comb['path_comparison'] = path_comparison(ok, ref)
    # TI slice values vs the NES <dW/dt> profiles (forward and minus reverse) of the longest overlapping BASE run
    # (TI slices are <dU_t/dt> of the base path; the learned path has a different integrand)
    if ok and ti.get('slices') and comb.get('test_c', {}).get('run'):
        with open(Path([r for r in a.runs if Path(r).name == comb['test_c']['run']][0]) / 'summary.json') as f:
            prof = json.load(f)['profile']
        tc = np.array(prof['t_bin_centers'])
        fw = np.array(prof['forward_mean_dWdt'])
        rv = np.array(prof['reverse_minus_mean_dWdt'])
        rows_p = []
        for k, v in ti['slices'].items():
            t = float(k)
            nf, nr = float(np.interp(t, tc, fw)), float(np.interp(t, tc, rv))
            rows_p.append(dict(t=t, TI=v, NES_fwd=nf, NES_minus_rev=nr, TI_minus_NES_mean=v - (nf + nr) / 2))
        d = np.array([r['TI_minus_NES_mean'] for r in rows_p])
        comb['ti_vs_nes_profile'] = dict(
            run=comb['test_c']['run'], rows=rows_p, rms_TI_minus_NES=float(np.sqrt((d ** 2).mean())),
            mean_TI_minus_NES=float(d.mean()), fwd_rev_profile_rms_gap=float(np.sqrt(((fw - rv) ** 2).mean())),
            note='NES profile = mean work increment per unit t in 100 bins (interpolated to slice t); forward and '
                 'minus-reverse coincide when switching is near-reversible')
    E.atomic_json(a.out, comb)
    print(json.dumps(E.json_safe(comb.get('vs_TI')), indent=1), flush=True)
    for c in comb.get('path_comparison', []):
        print({k: v for k, v in c.items() if k != 'rows'}, flush=True)
    if 'ti_vs_nes_profile' in comb:
        print({k: v for k, v in comb['ti_vs_nes_profile'].items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    main()
