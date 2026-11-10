"""Tests (a) and (b) on toys with exact dF, using the SAME nes_core kernels / work accumulation.

(a) positive tests: forward, reverse and BAR must hit the exact dF
    HO   U_t = k_t |x|^2 / 2 on R^d (x shape (4,3), d=12), k_t = k0 + t (k1-k0),  dF = (d/2) ln(k1/k0)
    VM   U_t = -kappa_t sum cos(2 pi x) on the unit torus (x shape (2,3), d=6),  Z = I0(kappa)^d,
         dF = -d ln(I0(kappa1)/I0(kappa0)); large MALA steps so that wrapping mod 1 happens often
         (tests the unwrapped-displacement MH ratio used for the solvation system).
    Kernels: mala, tamed mala, rw, hmc (L leapfrog steps; on the torus with trajectories long enough to wrap).
    Roots are exact independent samples; 2 trajectories per root -> cluster bootstrap by root.
    Identity check: hmc with L=1 must reproduce untamed mala with the same h and seeds (max |dW| <= 1e-8).
    Pass rule (fixed before running): |z_BAR| <= 3.5 for every case; |z_F|, |z_R| <= 3.5 where that
    direction's Kish ESS >= 50 (exponential estimators are not judged below that ESS).
(b) negative controls: must fail clearly
    b1  work sign flipped (W -> -W) on the (a) data: |error| >= 1 kBT and |z| >= 10 for F, R, BAR
    b2  ULA (MALA proposal, MH correction removed -> pi_t not preserved): |error_BAR| >= 0.5 and |z_BAR| >= 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import i0e  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'nes'))
import nes_core as C  # noqa: E402
from estimators import atomic_json, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def ho_system(k0, k1, shape=(4, 3)):
    d = int(np.prod(shape))

    def kt(t):
        return k0 + t * (k1 - k0)

    def energy(x, t):
        return 0.5 * kt(t) * jnp.sum(x * x)

    def delta(x, ta, tb):
        return 0.5 * (kt(tb) - kt(ta)) * jnp.sum(x * x)

    def sample(rng, n, end):
        k = k0 if end == 0 else k1
        return rng.normal(size=(n,) + shape) / np.sqrt(k)

    return dict(energy=energy, value_and_grad=jax.value_and_grad(energy), delta=delta, sample=sample,
                wrap=C.wrap_none, dF=0.5 * d * np.log(k1 / k0), desc=f'HO d={d} k {k0}->{k1}')


def vm_system(c0, c1, shape=(2, 3)):
    d = int(np.prod(shape))

    def ct(t):
        return c0 + t * (c1 - c0)

    def energy(x, t):
        return -ct(t) * jnp.sum(jnp.cos(2 * np.pi * x))

    def delta(x, ta, tb):
        return -(ct(tb) - ct(ta)) * jnp.sum(jnp.cos(2 * np.pi * x))

    def sample(rng, n, end):
        c = c0 if end == 0 else c1
        th = rng.vonmises(0.0, c, size=(n,) + shape)
        return (th / (2 * np.pi)) % 1.0

    # ln I0(k) = ln i0e(k) + k
    lnI0 = lambda k: np.log(i0e(k)) + k  # noqa: E731
    return dict(energy=energy, value_and_grad=jax.value_and_grad(energy), delta=delta, sample=sample,
                wrap=C.wrap_torus, dF=-d * (lnI0(c1) - lnI0(c0)), desc=f'von Mises torus d={d} kappa {c0}->{c1}')


def run_case(sysd, *, kind, h, K, M, cap, n_roots, reps, seed, L=1):
    rng = np.random.default_rng(seed)
    out = {}
    for direction, end in (('fwd', 0), ('rev', 1)):
        x0 = sysd['sample'](rng, n_roots, end)
        roots = np.repeat(np.arange(n_roots), reps)
        reps_ = np.tile(np.arange(reps), n_roots)
        x0 = jnp.asarray(np.repeat(x0, reps, axis=0))
        keys = jax.vmap(lambda r, p: C.traj_key(seed, direction, r, p))(jnp.asarray(roots), jnp.asarray(reps_))
        runner = C.make_chunk_runner(energy=sysd['energy'], value_and_grad=sysd['value_and_grad'],
                                     delta=sysd['delta'], kind=kind, wrap=sysd['wrap'], M=M, cap_factor=cap,
                                     nbins=10, L=L)
        t_dir = C.grid(K, direction)
        st = C.run_protocol(runner, x0, keys, t_dir, np.full(K + 1, h), 10, chunk=K)
        out[direction] = dict(W=st['W'], roots=roots, acc=float(st['acc'].mean()),
                              nonfinite=int(st['nonfinite'].sum()))
    return out


def judge_positive(s, ess_min=50.0, zmax=3.5):
    p = s['point']
    z = s['z']
    checks = {'BAR': abs(z['BAR']) <= zmax}
    judged = {'BAR': True}
    for k, ess in (('forward_jarzynski', p['ess_forward']), ('reverse_jarzynski', p['ess_reverse'])):
        judged[k] = ess >= ess_min
        checks[k] = (abs(z[k]) <= zmax) if judged[k] else None
    passed = all(v for v in checks.values() if v is not None)
    return dict(passed=bool(passed), checks=checks, judged=judged, rule=f'|z|<={zmax}; F/R judged if ESS>={ess_min}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-roots', type=int, default=4000)
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--out', default=str(ROOT / 'analysis' / 'toy-tests.json'))
    a = ap.parse_args()
    started = time.time()
    print('devices', jax.devices(), flush=True)

    HO = ho_system(1.0, 4.0)
    VM = vm_system(0.5, 4.0)
    positive = [
        ('HO-mala-K10-M1', HO, dict(kind='mala', h=0.1, K=10, M=1, cap=None)),
        ('HO-mala-K100-M1', HO, dict(kind='mala', h=0.1, K=100, M=1, cap=None)),
        ('HO-rw-K30-M2', HO, dict(kind='rw', h=0.1, K=30, M=2, cap=None)),
        ('HO-tamedmala-cap0.3-K30-M1', HO, dict(kind='mala', h=0.1, K=30, M=1, cap=0.3)),
        ('VM-mala-K20-M1-bigstep', VM, dict(kind='mala', h=0.02, K=20, M=1, cap=None)),
        ('VM-rw-K20-M1-bigstep', VM, dict(kind='rw', h=0.05, K=20, M=1, cap=None)),
        ('VM-tamedmala-cap1-K20-M1', VM, dict(kind='mala', h=0.02, K=20, M=1, cap=1.0)),
        ('HO-hmc-L5-K10-M1', HO, dict(kind='hmc', h=0.02, K=10, M=1, cap=None, L=5)),
        ('HO-hmc-L10-K30-M1', HO, dict(kind='hmc', h=0.01, K=30, M=1, cap=None, L=10)),
        ('VM-hmc-L5-K20-M1-wraps', VM, dict(kind='hmc', h=0.02, K=20, M=1, cap=None, L=5)),
    ]
    results = dict(positive={}, negative={})
    seed = 9100
    signflip_src = {}
    for name, sysd, kw in positive:
        t0 = time.time()
        r = run_case(sysd, n_roots=a.n_roots, reps=a.reps, seed=seed, **kw)
        s = summarize(r['fwd']['W'], r['rev']['W'], r['fwd']['roots'], r['rev']['roots'], n_boot=a.n_boot,
                      seed=seed, truth=sysd['dF'])
        s.pop('crooks', None)
        s['judgement'] = judge_positive(s)
        s['setup'] = dict(system=sysd['desc'], **kw, n_roots=a.n_roots, reps=a.reps, seed=seed,
                          acc_fwd=r['fwd']['acc'], acc_rev=r['rev']['acc'],
                          nonfinite=r['fwd']['nonfinite'] + r['rev']['nonfinite'], seconds=time.time() - t0)
        results['positive'][name] = s
        signflip_src[name] = (r, sysd)
        p = s['point']
        print(f"{name}: dF={sysd['dF']:.4f} BAR={p['BAR']:.4f} F={p['forward_jarzynski']:.4f} "
              f"R={p['reverse_jarzynski']:.4f} z={s['z']} diss=({p['dissipation_forward']:.2f},"
              f"{p['dissipation_reverse']:.2f}) ESS=({p['ess_forward']:.0f},{p['ess_reverse']:.0f}) "
              f"acc={r['fwd']['acc']:.3f} -> {'PASS' if s['judgement']['passed'] else 'FAIL'}", flush=True)
        seed += 1

    # b1: sign flip on (a) data
    for name in ('HO-mala-K10-M1', 'VM-mala-K20-M1-bigstep'):
        r, sysd = signflip_src[name]
        s = summarize(-r['fwd']['W'], -r['rev']['W'], r['fwd']['roots'], r['rev']['roots'], n_boot=a.n_boot,
                      seed=seed, truth=sysd['dF'])
        s.pop('crooks', None)
        ks = ('BAR', 'forward_jarzynski', 'reverse_jarzynski')
        fails = {k: bool(abs(s['error'][k]) >= 1.0 and abs(s['z'][k]) >= 10) for k in ks}
        s['judgement'] = dict(passed=all(fails.values()), failed_clearly=fails,
                              rule='control passes if every estimator misses by >=1 kBT and >=10 SE')
        results['negative'][f'b1-signflip-{name}'] = s
        print(f"b1 signflip {name}: errors={s['error']} z={s['z']} -> control "
              f"{'PASS' if s['judgement']['passed'] else 'FAIL'}", flush=True)
        seed += 1

    # identity: hmc L=1 == untamed mala (same h, seeds, roots)
    ra = run_case(HO, kind='mala', h=0.1, K=10, M=1, cap=None, n_roots=500, reps=1, seed=777)
    rb = run_case(HO, kind='hmc', h=0.1, K=10, M=1, cap=None, n_roots=500, reps=1, seed=777, L=1)
    dmax = max(float(np.abs(ra[d]['W'] - rb[d]['W']).max()) for d in ('fwd', 'rev'))
    results['hmc_L1_equals_mala'] = dict(max_abs_dW=dmax, passed=bool(dmax <= 1e-8))
    print(f'hmc(L=1) vs mala: max|dW|={dmax:.3e}', flush=True)

    # b2: ULA (no MH correction)
    r = run_case(HO, kind='ula', h=0.3, K=100, M=5, cap=None, n_roots=a.n_roots, reps=a.reps, seed=seed)
    s = summarize(r['fwd']['W'], r['rev']['W'], r['fwd']['roots'], r['rev']['roots'], n_boot=a.n_boot, seed=seed,
                  truth=HO['dF'])
    s.pop('crooks', None)
    ok = bool(abs(s['error']['BAR']) >= 0.5 and abs(s['z']['BAR']) >= 5)
    s['judgement'] = dict(passed=ok, rule='control passes if BAR misses by >=0.5 kBT and >=5 SE')
    s['setup'] = dict(system=HO['desc'], kind='ula', h=0.3, K=100, M=5)
    results['negative']['b2-ULA-no-MH-HO'] = s
    print(f"b2 ULA: errors={s['error']} z={s['z']} -> control {'PASS' if ok else 'FAIL'}", flush=True)

    results['all_positive_passed'] = (all(v['judgement']['passed'] for v in results['positive'].values())
                                      and results['hmc_L1_equals_mala']['passed'])
    results['all_negative_controls_passed'] = all(v['judgement']['passed'] for v in results['negative'].values())
    results['seconds'] = time.time() - started
    atomic_json(a.out, results)
    print(json.dumps(dict(all_positive_passed=results['all_positive_passed'],
                          all_negative_controls_passed=results['all_negative_controls_passed'],
                          out=a.out, seconds=results['seconds'])), flush=True)


if __name__ == '__main__':
    main()
