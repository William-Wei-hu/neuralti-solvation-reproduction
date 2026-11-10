"""FEAT-style estimate: escorted transport + FB RND work + nonequilibrium BAR on the solvation system.

Protocol: pilot 3 eps x 64 roots at K = 120; production K in {120, 400, 1200}, 256 roots per direction; controls
at K = 400 (v = 0, H1, H2), 64 roots each; forward and reverse as two parallel jobs.

Fields:  U_theta = frozen TSM EGNN (nes/solvation_system.make_learned_path, models/egnn-seed0.model);
         v = grad_x phi, phi = S2 snapshot runs/s2-phi/snapshots/step20000.pkl, EMA params (sha256 checked).
         Everything float64 (jax_enable_x64, matmul precision 'highest'); phi params cast float32 -> float64.
Kernel:  feat_core.make_runner, eps constant, linear grid, tamed drift cap_factor 5 (per particle <= 5 sqrt(2 eps dt)).
Work:    W = U_finish(x_K) - U_begin(x_0) + sum FB RND (+ HMC block terms), true endpoint energies U0 / U1.
Roots (S1b runs/endpoints-s1b, independent chains):  pilot block chains 0-255 -> eps pilot uses chains 0-63 only;
         block B chains 256-511 -> production (all 256) and controls (chains 256-319).  Block C (512-767) unused.
Keys:    traj_key(seed, direction, chain_id, rep): pilot seed 20261109, rep = K; production / controls seed 20261110,
         rep = K.  Controls use the same roots and keys as production K = 400 (paired comparison).
HMC (controls H1, H2): exact HMC w.r.t. exp(-U_theta(., t_{k+1})), M = 1, L = 20, h = 2e-6 (= job D, chosen there by
         the pre-set pilot rule); H1: after stage k if t_{k+1} <= 0.03 or t_{k+1} >= 0.97; H2: after stage k if
         (k + 1) % 10 == 0.  v = 0 control: same U_theta, same eps, no HMC.

======================================================================  FIXED BEFORE SUBMISSION (plan section 3)
Reference dF = -20.365 +- 0.013 (base HMC G = 2.4e5).
eps selection rule (pilot roots only): eps in {3e-5, 1.2e-4, 2.4e-4}, K = 120 (eps*dt <= 2e-6 for all).  Each job runs
  the pilot in BOTH directions (same roots / keys), so both jobs choose with the same data.  Choose the eps with the
  smallest mean dissipation over the two directions, (<W_F> + <W_R>) / 2 (= (diss_F + diss_R) / 2, dF cancels);
  a candidate with any non-finite work in either direction is excluded; exact tie -> smaller eps.  The chosen eps is
  used unchanged for K = 400 and 1200 and for the controls.  If both jobs' choices differ (GPU nondeterminism),
  that is reported and no BAR is formed across different eps.
Validity gates per K (no comparison conclusion unless all pass):
  G1 |dF_BAR + 20.365| <= 3 sqrt(SE_boot^2 + 0.013^2)
  G2 Crooks usable bins >= 3 and logistic overlap >= 0.1.  One-sided Jarzynski is cited (diagnostic only) when that
     direction's Kish ESS >= 50; it is ALSO listed against ESS >= 500 (the 50 -> 500 change loosens the citation
     rule; both are reported, the 50 rule is the plan's).
  G3 no non-finite work; max GNN edge count along every trajectory (incl. roots) < 15000 (EGNN truncates silently
     above size_to_pad; recorded for every run).
  G4 only one root block (S3a) -> every conclusion is labelled preliminary.
Metrics (estimation GPU seconds = sum of batch wall seconds fwd + rev of that K, compile included, as the NES runs):
  M1 BAR var x GPU-s <= 0.59 (base best); per K: K = 120 vs base G = 2.4e4 (<= 0.82), K = 1200 vs G = 2.4e5 (<= 0.59).
  M2 K = 1200: SE < 0.16 and diss_F < 4.98 and diss_R < 3.97 (base G = 2.4e3).
  M3 report only: training GPU-s (S2) and the number of estimates needed to amortise it under M1.
  "P1 effective": gates pass and some K has BAR var x GPU-s <= 64.
  "v helps": at K = 400 on the 64 control roots, (diss_F + diss_R)/2 with learned v < 1/3 of that with v = 0.
  H1 / H2: reported only (plan prediction: H2 worse than no HMC).
Stop rule: K = 1200 with both dissipations > 20 kBT and logistic overlap < 1e-3 -> negative result, stop, report.
Error: cluster bootstrap over roots (estimators.cluster_bootstrap, 2000 draws, seed 0; one trajectory per root).
======================================================================

Order inside each job (time guard --max-seconds stops cleanly between batches; finished runs are kept):
  pilot (both directions) -> eps choice -> production K = 120, 400, 1200 -> controls v0, H1, H2 (own direction).
Outputs (never overwritten): runs/s3a/<run>-<dir>.npz (+ per-batch partial files), runs/s3a/eps-choice-<job>.json,
  runs/s3a/done-<dir>.json; the job that finishes last (waits up to --partner-wait s) runs analyze_s3a.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import jax

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_default_matmul_precision', 'highest')
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import feat_core as F  # noqa: E402
import nes_core as C  # noqa: E402
import solvation_system as S  # noqa: E402
from estimators import atomic_json, atomic_npz, json_safe  # noqa: E402

PHI_SNAP = ROOT / 'runs' / 's2-phi' / 'snapshots' / 'step20000.pkl'
PHI_SHA = 'cea4cd63ac4ccb872ab98d04db87ad332f8064391a5dffbf8447ed996edbd160'
ROOTS = ROOT / 'runs' / 'endpoints-s1b'
OUT = ROOT / 'runs' / 's3a'
EPS_GRID = (3e-5, 1.2e-4, 2.4e-4)
K_PILOT, K_MAIN, K_CTRL = 120, (120, 400, 1200), 400
SEED_PILOT, SEED_MAIN = 20261109, 20261110
PILOT_ROOTS = slice(0, 64)
MAIN_ROOTS = slice(256, 512)
CTRL_ROOTS = slice(256, 320)
CAP = 5.0
NBINS = 20
CHUNK = 40
HMC = dict(h=2e-6, L=20, M=1)


def h1_flags(t_dir):
    tb = t_dir[1:]
    return (tb <= 0.03 + 1e-12) | (tb >= 0.97 - 1e-12)


def h2_flags(t_dir):
    return (np.arange(len(t_dir) - 1) + 1) % 10 == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--direction', choices=['fwd', 'rev'], required=True)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--max-seconds', type=float, default=8400)
    ap.add_argument('--partner-wait', type=float, default=900)
    ap.add_argument('--time-limit', type=float, default=9000, help='Slurm limit (s); the partner wait never runs into it')
    a = ap.parse_args()
    t_start = time.time()
    job = os.environ.get('SLURM_JOB_ID', 'none')
    OUT.mkdir(parents=True, exist_ok=True)
    log = dict(job=job, direction=a.direction, started=time.strftime('%Y-%m-%dT%H:%M:%S'), devices=str(jax.devices()))

    # ---------------- fields
    path = S.make_learned_path()
    ts = path['ts']
    raw = PHI_SNAP.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == PHI_SHA, f'phi snapshot sha256 {sha} != {PHI_SHA}'
    snap = pickle.loads(raw)
    assert snap['step'] == 20000
    assert snap['cfg_U_model'] == path['info']['cfg_U_model'], 'phi architecture differs from the checkpoint U_model'
    U_model = path['SI'].U_model
    pphi = jax.tree_util.tree_map(lambda v: jnp.asarray(v, jnp.float64), snap['ema'])

    def phi(x, t):
        return U_model.apply(pphi, x, jnp.reshape(t, (1,))).reshape()
    gphi = jax.grad(phi)
    zero = lambda x, t: jnp.zeros_like(x)  # noqa: E731
    FIELDS = {'phi': F.fields_separate(gphi, path['energy']), 'v0': F.fields_separate(zero, path['energy'])}
    hmc = dict(energy=path['energy'], value_and_grad=path['value_and_grad'], **HMC)
    aux = path['aux']
    auxb = jax.jit(jax.vmap(aux))
    U0b = jax.jit(jax.vmap(ts.U0))
    U1b = jax.jit(jax.vmap(ts.U1))
    log.update(ckpt_sha256=path['info']['checkpoint_sha256'], ckpt_step=path['info']['optimizer_step'],
               size_to_pad=path['info']['size_to_pad'], phi_sha256=sha, phi_step=int(snap['step']),
               phi_params='ema', precision='float64', matmul_precision='highest')

    roots = {}
    for e, d in ((0, 'fwd'), (1, 'rev')):
        with np.load(ROOTS / f'end{e}.npz') as z:
            roots[d] = (z['x'], z['chain_id'])

    runners = {}

    def runner(vkind, eps, direction, with_hmc):
        key = (vkind, eps, direction, with_hmc)
        if key not in runners:
            runners[key] = F.make_runner(fields=FIELDS[vkind], eps=eps, direction=direction, wrap=C.wrap_torus,
                                         cap_factor=CAP, nbins=NBINS, hmc=hmc if with_hmc else None, aux=aux)
        return runners[key]

    def over_time():
        return time.time() - t_start > a.max_seconds

    def run_set(name, direction, vkind, eps, K, sl, seed, flags_fn=None):
        """One run = one (setting, direction) over a root slice, in batches; returns the npz path or None."""
        out = OUT / f'{name}-{direction}.npz'
        if out.exists():
            raise SystemExit(f'{out} exists; refusing to overwrite')
        X, ch = roots[direction]
        X, ch = X[sl], ch[sl]
        t_dir = C.grid(K, direction)
        flags = np.zeros(K, bool) if flags_fn is None else flags_fn(t_dir)
        init, run = runner(vkind, eps, direction, flags_fn is not None)
        Ub, Uf = (U0b, U1b) if direction == 'fwd' else (U1b, U0b)
        res = []
        for b0 in range(0, len(X), a.batch):
            part = out.with_name(out.stem + f'.b{b0 // a.batch:03d}.npz')
            if over_time():
                print(f'{name}-{direction}: max-seconds reached before batch {b0 // a.batch}', flush=True)
                return None
            xb, cb = X[b0:b0 + a.batch], ch[b0:b0 + a.batch]
            keys = jax.vmap(lambda r: C.traj_key(seed, direction, r, K))(jnp.asarray(cb))
            tb = time.time()
            o = F.run_protocol(init, run, xb, keys, t_dir, flags, NBINS, chunk=CHUNK)
            W = F.work(o, xb, Ub, Uf)
            e0 = np.asarray(auxb(jnp.asarray(xb)))
            r = dict(W=W, S=o['S'], Sbins=o['Sbins'], nonfinite=o['nonfinite'],
                     auxmax=np.maximum(o['auxmax'], e0), ncapped=o['ncapped'], csq=o['csq'],
                     acc=o['acc'].astype(np.float32), chain_id=cb, seconds=np.float64(time.time() - tb))
            atomic_npz(part, **r)
            res.append(r)
            print(f'  {name}-{direction} batch {b0 // a.batch}: <W>={W.mean():.3f} sd={W.std():.3f} '
                  f'edges={r["auxmax"].max():.0f} capped={o["ncapped"].mean():.1f} nonfinite={int(o["nonfinite"].sum())} '
                  f'{r["seconds"]:.0f}s', flush=True)
        merged = {k: np.concatenate([np.atleast_1d(r[k]) if k == 'seconds' else r[k] for r in res])
                  for k in res[0]}
        meta = dict(name=name, direction=direction, v=vkind, eps=eps, K=K, roots=[sl.start, sl.stop], seed=seed,
                    rep='K', cap_factor=CAP, nbins=NBINS, chunk=CHUNK, batch=a.batch,
                    hmc=(HMC if flags_fn is not None else None), hmc_stages=int(flags.sum()),
                    gpu_seconds=float(merged['seconds'].sum()), job=job, **{k: log[k] for k in (
                        'ckpt_sha256', 'phi_sha256', 'precision')})
        atomic_npz(out, meta=np.array(json.dumps(meta)), **merged)
        for b in range(len(res)):
            out.with_name(out.stem + f'.b{b:03d}.npz').unlink()
        print(f'{name}-{direction}: <W>={merged["W"].mean():.4f} sd={merged["W"].std(ddof=1):.4f} '
              f'edges_max={merged["auxmax"].max():.0f} nonfinite={int(merged["nonfinite"].sum())} '
              f'gpu_s={meta["gpu_seconds"]:.0f}', flush=True)
        return out

    status = dict(log=log, runs={}, complete=False)

    def save_status():
        atomic_json(OUT / f'status-{a.direction}-{job}.json', json_safe(status))

    # ---------------- pilot, both directions, then the fixed eps rule
    pdir = f'pilot-{a.direction}job'
    (OUT / pdir).mkdir(exist_ok=True)
    pilot = {}
    for eps in EPS_GRID:
        for d in ('fwd', 'rev'):
            p = run_set(f'{pdir}/eps{eps:.1e}-K{K_PILOT}', d, 'phi', eps, K_PILOT, PILOT_ROOTS, SEED_PILOT)
            if p is None:
                status['stopped'] = 'max-seconds during pilot'
                save_status()
                return 1
            with np.load(p) as z:
                pilot[(eps, d)] = dict(mean_W=float(z['W'].mean()), finite=bool(np.isfinite(z['W']).all()),
                                       nonfinite=int(z['nonfinite'].sum()), edges=float(z['auxmax'].max()))
            status['runs'][str(p.name)] = pilot[(eps, d)]
            save_status()
    score = {}
    for eps in EPS_GRID:
        f, r = pilot[(eps, 'fwd')], pilot[(eps, 'rev')]
        ok = f['finite'] and r['finite'] and f['nonfinite'] == 0 and r['nonfinite'] == 0
        score[eps] = 0.5 * (f['mean_W'] + r['mean_W']) if ok else np.inf
    eps_star = min(EPS_GRID, key=lambda e: (score[e], e))
    choice = dict(rule='argmin (<W_F>+<W_R>)/2 at K=120 over pilot roots 0-63; non-finite excluded; tie -> smaller',
                  score={f'{e:.1e}': score[e] for e in EPS_GRID}, pilot={f'{e:.1e}/{d}': v for (e, d), v in pilot.items()},
                  eps=eps_star, job=job, direction_of_job=a.direction)
    if not np.isfinite(score[eps_star]):
        choice['eps'] = None
        atomic_json(OUT / f'eps-choice-{a.direction}job.json', json_safe(choice))
        status['stopped'] = 'no eps candidate with finite pilot work'
        save_status()
        print('STOP: no eps candidate with finite work', flush=True)
        return 1
    atomic_json(OUT / f'eps-choice-{a.direction}job.json', json_safe(choice))
    print('eps choice', json.dumps(json_safe(choice)), flush=True)
    status['eps'] = eps_star

    # ---------------- production and controls, own direction
    plan = [(f'main-K{K}', 'phi', K, MAIN_ROOTS, None) for K in K_MAIN]
    plan += [('ctrl-v0-K400', 'v0', K_CTRL, CTRL_ROOTS, None),
             ('ctrl-H1-K400', 'phi', K_CTRL, CTRL_ROOTS, h1_flags),
             ('ctrl-H2-K400', 'phi', K_CTRL, CTRL_ROOTS, h2_flags)]
    for name, vk, K, sl, ff in plan:
        p = run_set(name, a.direction, vk, eps_star, K, sl, SEED_MAIN, ff)
        if p is None:
            status['stopped'] = f'max-seconds before finishing {name}'
            save_status()
            break
        with np.load(p) as z:
            status['runs'][p.name] = dict(mean_W=float(z['W'].mean()), sd_W=float(z['W'].std(ddof=1)),
                                          nonfinite=int(z['nonfinite'].sum()), edges=float(z['auxmax'].max()),
                                          gpu_seconds=json.loads(str(z['meta']))['gpu_seconds'])
        save_status()
    else:
        status['complete'] = True
    try:
        status['jax_peak_bytes'] = int(jax.devices()[0].memory_stats()['peak_bytes_in_use'])
    except Exception as e:  # diagnostic only
        status['jax_peak_bytes'] = f'unavailable: {e}'
    status['seconds'] = time.time() - t_start
    save_status()
    atomic_json(OUT / f'done-{a.direction}.json', json_safe(dict(job=job, complete=status['complete'],
                                                                 time=time.strftime('%Y-%m-%dT%H:%M:%S'))))

    # ---------------- analysis by the job that finishes last
    other = 'rev' if a.direction == 'fwd' else 'fwd'
    t0 = time.time()
    wait = max(0.0, min(a.partner_wait, a.time_limit - (time.time() - t_start) - 600))
    while not (OUT / f'done-{other}.json').exists() and time.time() - t0 < wait:
        time.sleep(20)
    if (OUT / f'done-{other}.json').exists():
        try:
            os.mkdir(OUT / 'analysis.lock')
        except FileExistsError:
            print('partner job runs the analysis', flush=True)
        else:
            rc = subprocess.call([sys.executable, str(HERE / 'analyze_s3a.py')], env={**os.environ, 'JAX_PLATFORMS': 'cpu'})
            print(f'analyze_s3a exit={rc}', flush=True)
    else:
        print(f'partner not done after {wait:.0f}s; analysis not run here', flush=True)
    return 0 if status['complete'] else 1


if __name__ == '__main__':
    sys.exit(main())
