"""Free-energy estimators and diagnostics from forward / reverse NES work (numpy, CPU).

Sign convention (beta = 1):  W_F = forward work (0 -> 1),  W_R = reverse work (1 -> 0), both as measured.
  forward Jarzynski  dF_F = -ln <exp(-W_F)>
  reverse Jarzynski  dF_R = +ln <exp(-W_R)>
  BAR                sum_F f(M + W_F - dF) = sum_R f(-M + W_R + dF),  f(z) = 1/(1+e^z),  M = ln(n_F/n_R)
  dissipation        <W_F> - dF  >= 0,      dF + <W_R> >= 0
  Crooks             P_F(W) / P_R(-W) = exp(W - dF): W_F and -W_R histograms cross at W = dF.

Adapted (to this sign convention) from our earlier NeuralTI estimator code:
    estimator_analysis.py   estimates(): stable-log BAR root with bracketing + brentq, logistic overlap
    known_certificates.py   markov_interval(): Kish ESS of log-weights; per-root log-mean-exp weights
    analyze_campaign.py     cluster bootstrap that resamples whole root chains
In the 10-25 code `wr` meant -W_R (it entered as logsumexp(+wr)); here W_R is the raw reverse work.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logsumexp


# ----------------------------------------------------------------------------- point estimates
def jarzynski_forward(wf):
    wf = np.asarray(wf, float)
    return float(-(logsumexp(-wf) - np.log(len(wf))))


def jarzynski_reverse(wr):
    wr = np.asarray(wr, float)
    return float(logsumexp(-wr) - np.log(len(wr)))


def bar(wf, wr):
    """Bennett acceptance ratio, maximum-likelihood form (Shirts et al. 2003). Stable in logs."""
    wf = np.asarray(wf, float)
    wr = np.asarray(wr, float)
    m = np.log(len(wf) / len(wr))

    def eq(f):
        return float(logsumexp(-np.logaddexp(0.0, m + wf - f)) - logsumexp(-np.logaddexp(0.0, -m + wr + f)))

    center = float((np.median(wf) - np.median(wr)) / 2)
    span = 64.0
    for _ in range(100):
        if eq(center - span) <= 0 <= eq(center + span):
            break
        span *= 2
    else:
        raise ValueError('BAR root could not be bracketed')
    return float(brentq(eq, center - span, center + span, xtol=1e-12))


def kish_ess(logw):
    logw = np.asarray(logw, float)
    p = logw - logsumexp(logw)
    return float(np.exp(-logsumexp(2 * p)))


def per_root_logmeanexp(logw, roots):
    logw = np.asarray(logw, float)
    return np.array([logsumexp(logw[g]) - np.log(len(g)) for g in _groups(roots)])


def histogram_overlap(wf, wr, bins=60):
    """Overlap coefficient int min(p_F(w), p_R(-w)) dw of W_F and -W_R (0 = disjoint, 1 = identical)."""
    a = np.asarray(wf, float)
    b = -np.asarray(wr, float)
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, edges, density=True)
    pb, _ = np.histogram(b, edges, density=True)
    return float(np.sum(np.minimum(pa, pb)) * (edges[1] - edges[0]))


def crooks_crossing(wf, wr, bins=40, min_count=5):
    """Fit ln p_F(w) - ln p_R(-w) = w - dF on bins where both histograms have >= min_count entries.

    Returns the inverse-variance weighted dF estimate (slope fixed to 1), the free-slope fit, and the
    number of usable bins (0 -> no histogram overlap, crossing undefined)."""
    a = np.asarray(wf, float)
    b = -np.asarray(wr, float)
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    edges = np.linspace(lo, hi, bins + 1)
    na, _ = np.histogram(a, edges)
    nb, _ = np.histogram(b, edges)
    centers = 0.5 * (edges[1:] + edges[:-1])
    ok = (na >= min_count) & (nb >= min_count)
    out = dict(bins_used=int(ok.sum()), edges=edges.tolist(), counts_F=na.tolist(), counts_minus_R=nb.tolist())
    if ok.sum() == 0:
        out.update(dF_slope1=None, slope=None, intercept_dF=None)
        return out
    lr = np.log(na[ok] / len(a)) - np.log(nb[ok] / len(b))
    var = 1.0 / na[ok] + 1.0 / nb[ok]
    w = 1.0 / var
    est = centers[ok] - lr
    out['dF_slope1'] = float(np.sum(w * est) / np.sum(w))
    if ok.sum() >= 2:
        A = np.vstack([centers[ok], np.ones(ok.sum())]).T
        coef, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], lr * np.sqrt(w), rcond=None)
        out['slope'] = float(coef[0])
        out['intercept_dF'] = float(-coef[1] / coef[0]) if coef[0] != 0 else None
    else:
        out.update(slope=None, intercept_dF=None)
    return out


def point_estimates(wf, wr, roots_f=None, roots_r=None):
    wf = np.asarray(wf, float)
    wr = np.asarray(wr, float)
    dF = bar(wf, wr)
    m = np.log(len(wf) / len(wr))
    oF = float(expit(-(m + wf - dF)).mean())       # mean_F f(M + W_F - dF)
    oR = float(expit(-(-m + wr + dF)).mean())      # mean_R f(-M + W_R + dF)
    out = dict(
        BAR=dF,
        forward_jarzynski=jarzynski_forward(wf),
        reverse_jarzynski=jarzynski_reverse(wr),
        mean_WF=float(wf.mean()), mean_WR=float(wr.mean()),
        sd_WF=float(wf.std(ddof=1)), sd_WR=float(wr.std(ddof=1)),
        dissipation_forward=float(wf.mean() - dF),
        dissipation_reverse=float(dF + wr.mean()),
        bound_lower_minus_mean_WR=float(-wr.mean()), bound_upper_mean_WF=float(wf.mean()),
        midpoint_mean_work=float((wf.mean() - wr.mean()) / 2),
        bar_overlap_F=oF, bar_overlap_R=oR, logistic_overlap_sum=oF + oR,
        histogram_overlap=histogram_overlap(wf, wr),
        ess_forward=kish_ess(-wf), ess_reverse=kish_ess(-wr),
        n_F=int(len(wf)), n_R=int(len(wr)),
    )
    if roots_f is not None:
        out['ess_forward_roots'] = kish_ess(per_root_logmeanexp(-wf, roots_f))
        out['n_roots_F'] = int(len(np.unique(roots_f)))
    if roots_r is not None:
        out['ess_reverse_roots'] = kish_ess(per_root_logmeanexp(-wr, roots_r))
        out['n_roots_R'] = int(len(np.unique(roots_r)))
    return out


BOOT_KEYS = ('BAR', 'forward_jarzynski', 'reverse_jarzynski', 'dissipation_forward', 'dissipation_reverse',
             'mean_WF', 'mean_WR', 'logistic_overlap_sum')


def _groups(roots):
    """Indices of each root's trajectories (computed once)."""
    roots = np.asarray(roots)
    order = np.argsort(roots, kind='stable')
    _, start = np.unique(roots[order], return_index=True)
    return np.split(order, start[1:])


def _cluster_index(groups, rng):
    pick = rng.integers(len(groups), size=len(groups))
    return np.concatenate([groups[j] for j in pick])


def cluster_bootstrap(wf, wr, roots_f, roots_r, n_boot=2000, seed=0, keys=BOOT_KEYS):
    """Resample whole root chains (with replacement) independently in each direction."""
    wf = np.asarray(wf, float)
    wr = np.asarray(wr, float)
    rng = np.random.default_rng(seed)
    gf, gr = _groups(roots_f), _groups(roots_r)
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        fi = _cluster_index(gf, rng)
        ri = _cluster_index(gr, rng)
        p = point_estimates(wf[fi], wr[ri])
        for k in keys:
            draws[k].append(p[k])
    out = {}
    for k, v in draws.items():
        v = np.asarray(v)
        out[k] = dict(se=float(v.std(ddof=1)), ci95=np.quantile(v, [0.025, 0.975]).tolist())
    return out


def one_sided_bootstrap(w, roots, fn, n_boot=2000, seed=0):
    w = np.asarray(w, float)
    rng = np.random.default_rng(seed)
    g = _groups(roots)
    v = np.array([fn(w[_cluster_index(g, rng)]) for _ in range(n_boot)])
    return dict(se=float(v.std(ddof=1)), ci95=np.quantile(v, [0.025, 0.975]).tolist())


def summarize(wf, wr, roots_f, roots_r, n_boot=2000, seed=0, truth=None):
    p = point_estimates(wf, wr, roots_f, roots_r)
    b = cluster_bootstrap(wf, wr, roots_f, roots_r, n_boot=n_boot, seed=seed)
    out = dict(point=p, bootstrap=b, crooks=crooks_crossing(wf, wr))
    if truth is not None:
        out['truth'] = float(truth)
        out['z'] = {k: float((p[k] - truth) / b[k]['se']) if b[k]['se'] > 0 else None
                    for k in ('BAR', 'forward_jarzynski', 'reverse_jarzynski')}
        out['error'] = {k: float(p[k] - truth) for k in ('BAR', 'forward_jarzynski', 'reverse_jarzynski')}
    return out


# ----------------------------------------------------------------------------- io helpers
def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else str(float(obj))
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=1) + '\n')
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
