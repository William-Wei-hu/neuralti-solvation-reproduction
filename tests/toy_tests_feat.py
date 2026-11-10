"""CPU tests T1-T6 for feat/feat_core.py on toy systems with exact dF.  CPU only.

Toys (exact dF, exact independent roots, one trajectory per root, N_ROOTS per direction):
  HO  U_t = k_t |x|^2 / 2, x shape (4, 3), k_t = 1 + 3t, dF = 6 ln 4           (NES toy_tests.ho_system)
  VM  U_t = -kappa_t sum cos 2 pi x on the torus, x shape (2, 3), kappa_t = 0.5 + 3.5t   (vm_system)
  "arbitrary" fields (deliberately not the transport of the path):
      HO  v = (-0.8 + 0.5t) x + 0.3 sin x,       g = (1 + 2.5t + 0.4 sin pi t) x + 0.2 tanh x
      VM  v = 0.15 (1 + t) sin(2 pi x + 1),       g = 2 pi (0.5 + 3t + 0.5 sin pi t) sin 2 pi x
  "exact" HO fields: v = -(3 / (2 k_t)) x, g = k_t x  (the SDE then preserves N(0, 1/k_t)).
  eps: HO 0.5; VM 0.02 K (per-coordinate EM noise sqrt(2 eps dt) = 0.2: frequent wrapping).
  VM2 (same endpoints and dF as VM):
      v = 0.05 (1 + t) sin(2 pi x + 1)   (non-zero divergence, so control a is sensitive)
      g = 2 pi kappa_t sin 2 pi x        (exact score of the path U_t; the Langevin part relaxes towards pi_t)
      eps = 0.25 at every K; K = 100 / 200 / 400 -> per-step noise sqrt(2 eps dt) = 0.071 / 0.050 / 0.035.
      The modes of exp(kappa cos 2 pi x) sit at x = 0 = 1, i.e. on the wrap point, and the total noise
      displacement sqrt(2 eps) = 0.71 per coordinate, so wrapping happens many times on every path.

The old VM cases (eps = 0.02 K) are REPORT ONLY in T1, T3 and T5: with them even a correct implementation has
overlap ~0 and ESS 1-7, so they cannot discriminate.  VM2 is the required torus test.

Pass rules (fixed before the tests were run):
  T1  arbitrary fields, HO K = 10 / 30 / 100 and VM2 K = 100 / 200 / 400 (required):
      |z_BAR| <= 3.5; |z_F|, |z_R| <= 3.5 where that direction's Kish ESS >= 50 (ESS >= 500 also reported).
      VM2 additionally: logistic overlap >= 0.1 and mean boundary crossings per trajectory >= 5
      (crossing = a coordinate's wrapped position jumps by more than 0.5 within one step).
      Old VM K = 10 / 30 / 100: report only.
  T2  HO exact fields, K = 10 / 30 / 100 / 300: |z_BAR| <= 3.5 at every K; SD(W_F) and SD(W_R) strictly
      decrease with K and SD(K=300) < SD(K=10) / 3.  Generalized work (eq. 25, exact divergence,
      trapezoid on the same paths) at K = 300: SD <= 1e-10 and |mean - dF| <= 1e-3 in both directions.
      Report only: eq.-25 BAR with the arbitrary HO fields at K = 1000 (O(dt) bias expected).
  T3  negative controls must fail clearly; required on HO (K = 30) and on VM2 (K = 200):
      a  opposite kernel's drift taken at x instead of x':   |error_BAR| >= 0.1 and |z_BAR| >= 5   (required)
      c  endpoint energies omitted (W = S):                    |error_BAR| >= 0.1 and |z_BAR| >= 5   (required)
      d  work sign flipped:                                    |error_BAR| >= 1   and |z_BAR| >= 10  (required)
      Old VM K = 30: report only.
  T3b sensitivity only (no pass rule): minimum-image residual instead of the unwrapped one, old VM K = 10 and
      VM2 K = 200 (with per-step noise 0.05 the two residuals almost never differ, so no effect is expected).
  T5  taming, cap_factor 5 and 0.5: HO K = 30 and VM2 K = 200 required |z_BAR| <= 3.5; old VM K = 30 report
      only; capped fraction reported.
  T6  two_scalar_grads on a shared-backbone toy network: max |difference| to separate jax.grad <= 1e-12.
  T4  real solvation system, learned EGNN path (NES inputs/ checkpoint), 2 roots x K = 3, both directions:
      NES (nes_core, HMC h = 2e-6, L = 20, M = 1, seed 7110 keys) vs feat_core with v = 0, eps = 0 and an HMC
      block at every level: max |x_final difference| <= 1e-10 (bitwise status reported) and
      max |W difference| <= 1e-8 kBT.
"""
from __future__ import annotations

import argparse
import os
import json
import sys
import time
import traceback
from pathlib import Path

import jax

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_default_matmul_precision', 'highest')
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'nes'), str(HERE.parent / 'feat')]
import feat_core as F  # noqa: E402
import nes_core as C  # noqa: E402
from estimators import atomic_json, summarize  # noqa: E402
from toy_tests import ho_system, vm_system  # noqa: E402

N_ROOTS = 4000
N_BOOT = 600
REPORT = dict(tests={}, started=time.strftime('%Y-%m-%dT%H:%M:%S'))
OUT = [ROOT / 'outputs' / 'feat-toy-tests.json']     # set by --out


# ----------------------------------------------------------------------------- toy fields
def ho_fields(kind):
    if kind == 'exact':
        def v(x, t):
            return -(3.0 / (2.0 * (1.0 + 3.0 * t))) * x

        def U(x, t):
            return 0.5 * (1.0 + 3.0 * t) * jnp.sum(x * x)
    else:
        def v(x, t):
            return (-0.8 + 0.5 * t) * x + 0.3 * jnp.sin(x)

        def U(x, t):
            kk = 1.0 + 2.5 * t + 0.4 * jnp.sin(np.pi * t)
            return 0.5 * kk * jnp.sum(x * x) + 0.2 * jnp.sum(jnp.log(jnp.cosh(x)))
    return v, U


def vm_fields():
    def v(x, t):
        return 0.15 * (1.0 + t) * jnp.sin(2 * np.pi * x + 1.0)

    def U(x, t):
        kk = 0.5 + 3.0 * t + 0.5 * jnp.sin(np.pi * t)
        return -kk * jnp.sum(jnp.cos(2 * np.pi * x))
    return v, U


def vm2_fields():
    def v(x, t):
        return 0.05 * (1.0 + t) * jnp.sin(2 * np.pi * x + 1.0)

    def U(x, t):
        return -(0.5 + 3.5 * t) * jnp.sum(jnp.cos(2 * np.pi * x))
    return v, U


def toy(name):
    if name == 'HO':
        sysd = ho_system(1.0, 4.0)
    else:
        sysd = vm_system(0.5, 4.0)
    U0 = jax.jit(jax.vmap(lambda x: sysd['energy'](x, 0.0)))
    U1 = jax.jit(jax.vmap(lambda x: sysd['energy'](x, 1.0)))
    return sysd, U0, U1


def run_dir(name, fkind, direction, K, eps, *, seed, cap_factor=None, variant='exact', record=False, n=N_ROOTS):
    sysd, U0, U1 = toy(name)
    v, U = ho_fields(fkind) if name == 'HO' else (vm2_fields() if name == 'VM2' else vm_fields())
    fields = F.fields_separate(v, U)
    count_crossings = name == 'VM2'
    record = record or count_crossings
    init, run = F.make_runner(fields=fields, eps=eps, direction=direction, wrap=sysd['wrap'],
                              cap_factor=cap_factor, nbins=20, variant=variant, record=record)
    rng = np.random.default_rng([seed, 0 if direction == 'fwd' else 1, K])
    x0 = sysd['sample'](rng, n, 0 if direction == 'fwd' else 1)
    keys = jax.vmap(lambda r: C.traj_key(seed, direction, r, 0))(jnp.arange(n))
    t_dir = C.grid(K, direction)
    out = F.run_protocol(init, run, x0, keys, t_dir, np.zeros(K, bool), 20)
    Ub, Uf = (U0, U1) if direction == 'fwd' else (U1, U0)
    W = F.work(out, x0, Ub, Uf)
    res = dict(W=W, S=out['S'], x0=x0, out=out, t_dir=t_dir, dF=sysd['dF'], v=v, U=U)
    if count_crossings:
        xs = np.concatenate([x0[:, None], out['xs']], axis=1)
        jumps = np.abs(np.diff(xs, axis=1)) > 0.5
        res['crossings_per_traj'] = float(jumps.reshape(len(x0), -1).sum(1).mean())
    return res


def judge_sum(s, ess_min=50.0, zmax=3.5):
    z = s['z']
    p = s['point']
    ok = abs(z['BAR']) <= zmax
    if p['ess_forward'] >= ess_min:
        ok &= abs(z['forward_jarzynski']) <= zmax
    if p['ess_reverse'] >= ess_min:
        ok &= abs(z['reverse_jarzynski']) <= zmax
    return bool(ok)


def one_sided(s):
    """One-sided Jarzynski verdicts under both thresholds (ESS >= 50: rule; ESS >= 500: reported)."""
    out = {}
    for thr in (50, 500):
        z, p = s['z'], s['point']
        judged = {d: bool(p[e] >= thr) for d, e in (('F', 'ess_forward'), ('R', 'ess_reverse'))}
        okd = {d: (abs(z[q]) <= 3.5) if judged[d] else None
               for d, q in (('F', 'forward_jarzynski'), ('R', 'reverse_jarzynski'))}
        out[f'ess>={thr}'] = dict(judged=judged, ok=okd)
    return out


def brief(s):
    p, b = s['point'], s['bootstrap']
    return dict(BAR=p['BAR'], se=b['BAR']['se'], err=s['error']['BAR'], z=s['z'],
                ess=[p['ess_forward'], p['ess_reverse']], diss=[p['dissipation_forward'], p['dissipation_reverse']],
                overlap=p['logistic_overlap_sum'], sd=[p['sd_WF'], p['sd_WR']], one_sided=one_sided(s))


def pair(name, fkind, K, eps, seed, **kw):
    f = run_dir(name, fkind, 'fwd', K, eps, seed=seed, **kw)
    r = run_dir(name, fkind, 'rev', K, eps, seed=seed, **kw)
    roots = np.arange(N_ROOTS)
    s = summarize(f['W'], r['W'], roots, roots, n_boot=N_BOOT, seed=seed, truth=f['dF'])
    return f, r, s


def eps_for(name, K):
    return {'HO': 0.5, 'VM2': 0.25}.get(name, 0.02 * K)


def crossings(f, r):
    return [f.get('crossings_per_traj'), r.get('crossings_per_traj')]


def section(name):
    def deco(fn):
        def wrapped(*a, **k):
            t0 = time.time()
            try:
                res = fn(*a, **k)
            except Exception:
                res = dict(passed=False, error=traceback.format_exc())
            res['seconds'] = time.time() - t0
            REPORT['tests'][name] = res
            print(f'== {name}: passed={res.get("passed")}  ({res["seconds"]:.0f}s)', flush=True)
            print(json.dumps(F_json(res), indent=1)[:4000], flush=True)
            atomic_json(OUT[0], F_json(REPORT))
            return res
        return wrapped
    return deco


def F_json(o):
    from estimators import json_safe
    return json_safe(o)


# ----------------------------------------------------------------------------- tests
@section('T1')
def t1():
    cases = {}
    ok = True
    for name, Ks, required in (('HO', (10, 30, 100), True), ('VM2', (100, 200, 400), True),
                               ('VM', (10, 30, 100), False)):
        for K in Ks:
            f, r, s = pair(name, 'arb', K, eps_for(name, K), seed=100 + K)
            good = judge_sum(s)
            cr = crossings(f, r)
            if name == 'VM2':
                good = good and s['point']['logistic_overlap_sum'] >= 0.1 and min(cr) >= 5
            if required:
                ok &= good
            cases[f'{name}-K{K}'] = dict(passed=good, required=required, eps=eps_for(name, K),
                                         crossings_per_traj=cr, **brief(s))
    return dict(passed=bool(ok), cases=cases)


def eq25_work(run, fkind, direction):
    """Generalized work (eq. 25) by the trapezoid rule on the recorded grid, exact divergence via jacfwd.
    W~ = int (-div v + grad U . v + d_t U) dt + [U_a(x0) - U_theta0(x0)] - [U_b(x1) - U_theta1(x1)] (forward
    time order); W_F = W~ (forward paths), W_R = -W~ (reverse paths)."""
    v, U = run['v'], run['U']
    xs = np.concatenate([run['x0'][:, None], run['out']['xs']], axis=1)       # (B, K+1, ...) in travel order
    t = run['t_dir']

    def integrand(x, tt):
        vv = v(x, tt)
        g = jax.grad(U)(x, tt)
        J = jax.jacfwd(lambda y: v(y, tt))(x).reshape(x.size, x.size)
        dtU = jax.grad(U, argnums=1)(x, tt)
        return -jnp.trace(J) + jnp.sum(g * vv) + dtU

    f = jax.jit(jax.vmap(jax.vmap(integrand, in_axes=(0, 0)), in_axes=(0, None)))(jnp.asarray(xs), jnp.asarray(t))
    f = np.asarray(f)
    dt = np.abs(np.diff(t))
    integ = np.sum(0.5 * (f[:, 1:] + f[:, :-1]) * dt, axis=1)
    sysd, _, _ = toy('HO')
    Ua = jax.vmap(lambda x: sysd['energy'](x, 0.0))
    Ub = jax.vmap(lambda x: sysd['energy'](x, 1.0))
    Ut0 = jax.vmap(lambda x: U(x, 0.0))
    Ut1 = jax.vmap(lambda x: U(x, 1.0))
    if direction == 'fwd':
        x0, x1 = xs[:, 0], xs[:, -1]
    else:
        x0, x1 = xs[:, -1], xs[:, 0]
    # eq. 25 correction: -U_a(X0) - U_theta1(X1) + U_b(X1) + U_theta0(X0).
    corr = np.asarray(Ut0(x0) - Ua(x0)) + np.asarray(Ub(x1) - Ut1(x1))
    Wt = integ + corr
    return Wt if direction == 'fwd' else -Wt


@section('T2')
def t2():
    res = {}
    sds_f, sds_r = [], []
    ok = True
    for K in (10, 30, 100, 300):
        f, r, s = pair('HO', 'exact', K, 0.5, seed=200 + K, record=(K == 300))
        ok &= abs(s['z']['BAR']) <= 3.5
        sds_f.append(float(np.std(f['W'], ddof=1)))
        sds_r.append(float(np.std(r['W'], ddof=1)))
        res[f'K{K}'] = brief(s)
        if K == 300:
            e = {}
            for d, run in (('fwd', f), ('rev', r)):
                w25 = eq25_work(run, 'exact', d)
                target = f['dF'] if d == 'fwd' else -f['dF']
                e[d] = dict(sd=float(np.std(w25, ddof=1)), mean_minus_target=float(w25.mean() - target),
                            max_abs_dev=float(np.max(np.abs(w25 - target))))
                ok &= (e[d]['sd'] <= 1e-10) and (abs(e[d]['mean_minus_target']) <= 1e-3)
            res['eq25_exact_K300'] = e
    mono = all(a > b for a, b in zip(sds_f, sds_f[1:])) and all(a > b for a, b in zip(sds_r, sds_r[1:]))
    shrink = sds_f[-1] < sds_f[0] / 3 and sds_r[-1] < sds_r[0] / 3
    ok &= mono and shrink
    res.update(sd_WF=sds_f, sd_WR=sds_r, monotone=mono, shrink_factor_3=shrink)
    # report only: eq. 25 with arbitrary fields at K = 1000
    try:
        f = run_dir('HO', 'arb', 'fwd', 1000, 0.5, seed=299, record=True, n=1000)
        r = run_dir('HO', 'arb', 'rev', 1000, 0.5, seed=299, record=True, n=1000)
        wf, wr = eq25_work(f, 'arb', 'fwd'), eq25_work(r, 'arb', 'rev')
        roots = np.arange(1000)
        s25 = summarize(wf, wr, roots, roots, n_boot=N_BOOT, seed=299, truth=f['dF'])
        sfb = summarize(f['W'], r['W'], roots, roots, n_boot=N_BOOT, seed=299, truth=f['dF'])
        res['report_eq25_arbitrary_K1000'] = dict(eq25=brief(s25), fbrnd_same_paths=brief(sfb))
    except Exception:
        res['report_eq25_arbitrary_K1000'] = dict(error=traceback.format_exc())
    return dict(passed=bool(ok), **res)


def failed_clearly(s, err_min=0.1, z_min=5.0):
    return bool(abs(s['error']['BAR']) >= err_min and abs(s['z']['BAR']) >= z_min)


@section('T3')
def t3():
    res = {}
    ok = True
    roots = np.arange(N_ROOTS)
    for name, K, required in (('HO', 30, True), ('VM2', 200, True), ('VM', 30, False)):
        eps = eps_for(name, K)
        # a: opposite drift at x
        _, _, sa = pair(name, 'arb', K, eps, seed=300 + K, variant='other_at_x')
        fa = failed_clearly(sa)
        # c, d on the exact-variant data
        f, r, s0 = pair(name, 'arb', K, eps, seed=310 + K)
        sc = summarize(f['S'], r['S'], roots, roots, n_boot=N_BOOT, seed=1, truth=f['dF'])
        fc = failed_clearly(sc)
        sd = summarize(-f['W'], -r['W'], roots, roots, n_boot=N_BOOT, seed=2, truth=f['dF'])
        fd = failed_clearly(sd, err_min=1.0, z_min=10.0)
        if required:
            ok &= fa and fc and fd
        res[name] = dict(required=required, K=K, correct_variant_same_seed=brief(s0),
                         a_other_at_x=dict(failed=fa, **brief(sa)),
                         c_no_endpoint_energy=dict(failed=fc, **brief(sc)),
                         d_sign_flip=dict(failed=fd, **brief(sd)))
    # T3b: sensitivity only
    for name, K in (('VM', 10), ('VM2', 200)):
        try:
            _, _, sb = pair(name, 'arb', K, eps_for(name, K), seed=330, variant='minimage')
            res[f'T3b_minimage_{name}_K{K}_report_only'] = brief(sb)
        except Exception:
            res[f'T3b_minimage_{name}_K{K}_report_only'] = dict(error=traceback.format_exc())
    return dict(passed=bool(ok), **res)


@section('T5')
def t5():
    res = {}
    ok = True
    for name, K, required in (('HO', 30, True), ('VM2', 200, True), ('VM', 30, False)):
        for cf in (5.0, 0.5):
            f, r, s = pair(name, 'arb', K, eps_for(name, K), seed=500 + int(cf * 10), cap_factor=cf)
            good = abs(s['z']['BAR']) <= 3.5
            if required:
                ok &= good
            res[f'{name}-K{K}-cap{cf}'] = dict(passed=good, required=required,
                                          capped_fraction=[float(f['out']['ncapped'].mean() / K),
                                                           float(r['out']['ncapped'].mean() / K)], **brief(s))
    return dict(passed=bool(ok), **res)


@section('T6')
def t6():
    key = jax.random.PRNGKey(6)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    W1 = jax.random.normal(k1, (32, 12))
    b1 = jax.random.normal(k2, (32,))
    w2 = jax.random.normal(k3, (32,))
    w3 = jax.random.normal(k4, (32,))

    def f2(x, t):
        h = jnp.tanh(W1 @ x.reshape(-1) + b1 * t)
        return jnp.dot(w2, h) * (1 + t), jnp.dot(w3, h * h)

    tg = F.two_scalar_grads(f2)
    x = jax.random.normal(jax.random.PRNGKey(7), (4, 3))
    t = 0.37
    a, b, ga, gb = tg(x, t)
    ga2 = jax.grad(lambda y: f2(y, t)[0])(x)
    gb2 = jax.grad(lambda y: f2(y, t)[1])(x)
    va, vb = f2(x, t)
    fv, fg = F.fields_shared(f2)(x, t)
    diffs = dict(U=float(abs(a - va)), phi=float(abs(b - vb)), gradU=float(jnp.max(jnp.abs(ga - ga2))),
                 gradphi=float(jnp.max(jnp.abs(gb - gb2))),
                 fields_v_is_gradphi=float(jnp.max(jnp.abs(fv - gb2))),
                 fields_g_is_gradU=float(jnp.max(jnp.abs(fg - ga2))))
    return dict(passed=bool(max(diffs.values()) <= 1e-12), max_abs=diffs)


@section('T4')
def t4():
    import solvation_system as S
    t0 = time.time()
    path = S.make_learned_path()
    ts = path['ts']
    U0 = jax.jit(jax.vmap(ts.U0))
    U1 = jax.jit(jax.vmap(ts.U1))
    K, h, L, M, seed = 3, 2e-6, 20, 1, 7110
    endp = Path(os.environ.get('NTI_RUNS', HERE.parent / 'runs')) / 'endpoints'
    res = dict(load_seconds=time.time() - t0, info=dict(ckpt_sha256=path['info']['checkpoint_sha256'],
                                                       optimizer_step=path['info']['optimizer_step']))
    ok = True
    nes_runner = C.make_chunk_runner(energy=path['energy'], value_and_grad=path['value_and_grad'],
                                     delta=path['delta'], kind='hmc', wrap=C.wrap_torus, M=M, cap_factor=None,
                                     nbins=100, aux=None, L=L)
    hmc = dict(energy=path['energy'], value_and_grad=path['value_and_grad'], h=h, L=L, M=M)
    for direction in ('fwd', 'rev'):
        with np.load(endp / f'end{0 if direction == "fwd" else 1}.npz') as z:
            X = z['x'][:2]
            chain = z['chain_id'][:2]
        keys = jax.vmap(lambda r: C.traj_key(seed, direction, r, 0))(jnp.asarray(chain))
        t_dir = C.grid(K, direction)
        tn = time.time()
        st = C.run_protocol(nes_runner, jnp.asarray(X), keys, t_dir, np.full(K + 1, h), 100, K)
        tn = time.time() - tn
        init, run = F.make_runner(fields=None, eps=0.0, direction=direction, wrap=C.wrap_torus, hmc=hmc, nbins=100)
        tf = time.time()
        out = F.run_protocol(init, run, X, keys, t_dir, np.ones(K, bool), 100)
        tf = time.time() - tf
        Ub, Uf = (U0, U1) if direction == 'fwd' else (U1, U0)
        Wf = F.work(out, X, Ub, Uf)
        dx = float(np.max(np.abs(np.asarray(st['x']) - out['x'])))
        dW = float(np.max(np.abs(np.asarray(st['W']) - Wf)))
        bitwise = bool(np.array_equal(np.asarray(st['x']), out['x']))
        good = dx <= 1e-10 and dW <= 1e-8
        ok &= good
        res[direction] = dict(passed=good, max_abs_dx=dx, x_bitwise=bitwise, max_abs_dW=dW,
                              W_nes=np.asarray(st['W']).tolist(), W_feat=Wf.tolist(),
                              acc_nes=np.asarray(st['acc']).tolist(), acc_feat=out['acc'].tolist(),
                              seconds_nes=tn, seconds_feat=tf)
    return dict(passed=bool(ok), **res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--out', default=str(ROOT / 'analysis' / 's0-cpu-tests.json'))
    a = ap.parse_args()
    OUT[0] = Path(a.out)
    if OUT[0].exists():
        raise SystemExit(f'{OUT[0]} exists; refusing to overwrite')
    REPORT['devices'] = str(jax.devices())
    REPORT['jax'] = jax.__version__
    tests = dict(T6=t6, T1=t1, T2=t2, T3=t3, T5=t5, T4=t4)
    for k, fn in tests.items():
        if a.only and k not in a.only:
            continue
        fn()
    REPORT['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    REPORT['all_passed'] = all(v.get('passed') for v in REPORT['tests'].values())
    atomic_json(OUT[0], F_json(REPORT))
    print('SUMMARY', json.dumps({k: v.get('passed') for k, v in REPORT['tests'].items()}),
          'all_passed=', REPORT['all_passed'], flush=True)
    return 0 if REPORT['all_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
