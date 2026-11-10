"""S3a analysis: applies the criteria fixed in estimate_feat.py's header (plan section 3) to runs/s3a.
numpy / scipy only (estimators.py of the NES directory, unchanged).  Writes analysis/s3a-estimate.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'nes'))
from estimators import atomic_json, json_safe, summarize  # noqa: E402

RUNS = ROOT / 'runs' / 's3a'
OUT = ROOT / 'analysis' / 's3a-estimate.json'
TRUTH, TRUTH_SE = -20.365, 0.013
EDGE_CAP = 15000
BASE = {'G2.4e3': dict(var_x_gpu=1.07, diss=(4.98, 3.97), se=0.16), 'G2.4e4': dict(var_x_gpu=0.82),
        'G2.4e5': dict(var_x_gpu=0.59), 'jobD': dict(var_x_gpu=639)}
S2_TRAIN_GPU_SECONDS = 4266.0   # wall time of the velocity-field training run (1 GPU)


def load(name, d):
    f = RUNS / f'{name}-{d}.npz'
    if not f.exists():
        return None
    with np.load(f) as z:
        r = {k: z[k] for k in z.files}
    r['meta'] = json.loads(str(r['meta']))
    return r


def pair(name, sel=None):
    f, r = load(name, 'fwd'), load(name, 'rev')
    if f is None or r is None:
        return None
    if sel is not None:
        f = {k: (v[np.isin(f['chain_id'], sel)] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(f['W'])
                 else v) for k, v in f.items()}
        r = {k: (v[np.isin(r['chain_id'], sel)] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(r['W'])
                 else v) for k, v in r.items()}
    return f, r


def evaluate(f, r):
    s = summarize(f['W'], r['W'], f['chain_id'], r['chain_id'], n_boot=2000, seed=0, truth=TRUTH)
    p, b = s['point'], s['bootstrap']
    se = b['BAR']['se']
    gpu = float(f['seconds'].sum() + r['seconds'].sum())
    finite = bool(np.isfinite(f['W']).all() and np.isfinite(r['W']).all())
    nonf = int(f['nonfinite'].sum() + r['nonfinite'].sum())
    edges = float(max(f['auxmax'].max(), r['auxmax'].max()))
    g1 = abs(p['BAR'] - TRUTH) <= 3 * np.sqrt(se ** 2 + TRUTH_SE ** 2)
    g2 = s['crooks']['bins_used'] >= 3 and p['logistic_overlap_sum'] >= 0.1
    g3 = finite and nonf == 0 and edges < EDGE_CAP
    one_sided = {}
    for side, key, ess in (('forward', 'forward_jarzynski', 'ess_forward'), ('reverse', 'reverse_jarzynski', 'ess_reverse')):
        one_sided[side] = dict(value=p[key], se=b[key]['se'], ess=p[ess],
                               cited_ess_ge_50=bool(p[ess] >= 50), cited_ess_ge_500=bool(p[ess] >= 500))
    return dict(
        n=[len(f['W']), len(r['W'])], BAR=p['BAR'], SE=se, CI95=b['BAR']['ci95'], error=p['BAR'] - TRUTH,
        z_vs_truth=(p['BAR'] - TRUTH) / np.sqrt(se ** 2 + TRUTH_SE ** 2),
        dissipation=[p['dissipation_forward'], p['dissipation_reverse']],
        dissipation_se=[b['dissipation_forward']['se'], b['dissipation_reverse']['se']],
        mean_diss=0.5 * (p['dissipation_forward'] + p['dissipation_reverse']),
        mean_W=[p['mean_WF'], p['mean_WR']], sd_W=[p['sd_WF'], p['sd_WR']],
        logistic_overlap=p['logistic_overlap_sum'], histogram_overlap=p['histogram_overlap'],
        crooks_bins=s['crooks']['bins_used'], crooks_dF_slope1=s['crooks']['dF_slope1'],
        ess=[p['ess_forward'], p['ess_reverse']], one_sided_jarzynski=one_sided,
        gpu_seconds=gpu, gpu_seconds_dir=[float(f['seconds'].sum()), float(r['seconds'].sum())],
        BAR_var_x_gpu_seconds=se ** 2 * gpu,
        nonfinite=nonf, edges_max=edges, edges_max_dir=[float(f['auxmax'].max()), float(r['auxmax'].max())],
        capped_steps_mean=[float(f['ncapped'].mean()), float(r['ncapped'].mean())],
        csq_mean=[float(f['csq'].mean()), float(r['csq'].mean())],
        Sbins_mean_fwd=f['Sbins'].mean(0).tolist(), Sbins_mean_rev=r['Sbins'].mean(0).tolist(),
        hmc_acc_mean=[float(f['acc'][f['acc'] >= 0].mean()) if (f['acc'] >= 0).any() else None,
                      float(r['acc'][r['acc'] >= 0].mean()) if (r['acc'] >= 0).any() else None],
        eps=[f['meta']['eps'], r['meta']['eps']], K=f['meta']['K'],
        gates=dict(G1_consistent=bool(g1), G2_overlap=bool(g2), G3_numerics=bool(g3),
                   G4='preliminary (single root block)', all_pass=bool(g1 and g2 and g3)))


def main():
    rep = dict(truth=[TRUTH, TRUTH_SE], base=BASE)
    # eps choice from both jobs
    ch = {}
    for j in ('fwd', 'rev'):
        f = RUNS / f'eps-choice-{j}job.json'
        ch[j] = json.loads(f.read_text()) if f.exists() else None
    rep['eps_choice'] = ch
    same = all(c is not None for c in ch.values()) and ch['fwd']['eps'] == ch['rev']['eps']
    rep['eps_choice_agree'] = bool(same)
    # pilot (from both jobs; BAR per eps with each job's pilot, and the job-to-job difference)
    pil = {}
    for j in ('fwd', 'rev'):
        for e in (3e-5, 1.2e-4, 2.4e-4):
            pr = pair(f'pilot-{j}job/eps{e:.1e}-K120')
            if pr is not None:
                pil[f'{j}job/eps{e:.1e}'] = evaluate(*pr)
    for e in (3e-5, 1.2e-4, 2.4e-4):
        a, b = pair(f'pilot-fwdjob/eps{e:.1e}-K120'), pair(f'pilot-revjob/eps{e:.1e}-K120')
        if a and b:
            pil[f'jobdiff/eps{e:.1e}'] = dict(max_abs_dW_fwd=float(np.abs(a[0]['W'] - b[0]['W']).max()),
                                              max_abs_dW_rev=float(np.abs(a[1]['W'] - b[1]['W']).max()))
    rep['pilot'] = pil
    # production
    main_ = {}
    for K in (120, 400, 1200):
        pr = pair(f'main-K{K}')
        if pr is None:
            main_[f'K{K}'] = 'missing'
            continue
        if pr[0]['meta']['eps'] != pr[1]['meta']['eps']:
            main_[f'K{K}'] = dict(error='fwd and rev used different eps; no BAR formed',
                                  eps=[pr[0]['meta']['eps'], pr[1]['meta']['eps']])
            continue
        main_[f'K{K}'] = evaluate(*pr)
    rep['main'] = main_
    # metrics
    ok = {k: v for k, v in main_.items() if isinstance(v, dict) and 'BAR' in v}
    m = {}
    m['M1_var_x_gpu'] = {k: v['BAR_var_x_gpu_seconds'] for k, v in ok.items()}
    m['M1_pass_vs_0.59'] = {k: bool(v['gates']['all_pass'] and v['BAR_var_x_gpu_seconds'] <= 0.59) for k, v in ok.items()}
    if 'K120' in ok:
        m['M1_K120_vs_G2.4e4_0.82'] = bool(ok['K120']['gates']['all_pass'] and ok['K120']['BAR_var_x_gpu_seconds'] <= 0.82)
    if 'K1200' in ok:
        v = ok['K1200']
        m['M1_K1200_vs_G2.4e5_0.59'] = bool(v['gates']['all_pass'] and v['BAR_var_x_gpu_seconds'] <= 0.59)
        m['M2_K1200'] = dict(SE=v['SE'], dissipation=v['dissipation'],
                             passed=bool(v['gates']['all_pass'] and v['SE'] < 0.16 and v['dissipation'][0] < 4.98
                                         and v['dissipation'][1] < 3.97))
        m['stop_rule_triggered'] = bool(min(v['dissipation']) > 20 and v['logistic_overlap'] < 1e-3)
    best = min(ok.values(), key=lambda v: v['BAR_var_x_gpu_seconds']) if ok else None
    if best is not None:
        m['M3'] = dict(train_gpu_seconds=S2_TRAIN_GPU_SECONDS, best_K=best['K'],
                       best_var_x_gpu=best['BAR_var_x_gpu_seconds'],
                       note='report only; estimates needed to amortise = train / (gpu_base - gpu_feat) at equal variance',
                       gpu_seconds_for_SE_0p013=dict(feat=best['BAR_var_x_gpu_seconds'] / 0.013 ** 2, base=0.59 / 0.013 ** 2))
        g = m['M3']['gpu_seconds_for_SE_0p013']
        m['M3']['estimates_to_amortise'] = (S2_TRAIN_GPU_SECONDS / (g['base'] - g['feat'])) if g['base'] > g['feat'] else None
    m['P1_effective'] = bool(any(v['gates']['all_pass'] and v['BAR_var_x_gpu_seconds'] <= 64 for v in ok.values()))
    # controls (paired with production K = 400 on the same 64 roots)
    ctrl = {}
    sel = None
    for name in ('ctrl-v0-K400', 'ctrl-H1-K400', 'ctrl-H2-K400'):
        pr = pair(name)
        if pr is None:
            ctrl[name] = 'missing'
            continue
        ctrl[name] = evaluate(*pr)
        sel = pr[0]['chain_id']
    if sel is not None and load('main-K400', 'fwd') is not None and load('main-K400', 'rev') is not None:
        ctrl['main-K400-on-control-roots'] = evaluate(*pair('main-K400', sel))
        if isinstance(ctrl.get('ctrl-v0-K400'), dict):
            a, b = ctrl['main-K400-on-control-roots']['mean_diss'], ctrl['ctrl-v0-K400']['mean_diss']
            m['v_helps_K400'] = dict(learned_mean_diss=a, v0_mean_diss=b, ratio=a / b if b else None,
                                     passed=bool(a < b / 3))
    rep['controls'] = ctrl
    rep['metrics'] = m
    # status files
    rep['status'] = {f.name: json.loads(f.read_text()) for f in sorted(RUNS.glob('status-*.json'))}
    atomic_json(OUT, json_safe(rep))
    short = {k: ({kk: v[kk] for kk in ('BAR', 'SE', 'dissipation', 'logistic_overlap', 'crooks_bins', 'ess',
                                       'gpu_seconds', 'BAR_var_x_gpu_seconds', 'edges_max', 'nonfinite', 'gates')}
                 if isinstance(v, dict) and 'BAR' in v else v) for k, v in {**main_, **ctrl}.items()}
    print(json.dumps(json_safe(dict(eps_choice_agree=same, eps=[c and c['eps'] for c in ch.values()],
                                    runs=short, metrics=m)), indent=1))


if __name__ == '__main__':
    main()
