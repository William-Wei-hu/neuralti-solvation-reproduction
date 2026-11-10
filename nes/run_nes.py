"""Run one direction of an NES protocol on the solvation system and store per-trajectory work.

  python run_nes.py --path base|learned --direction fwd|rev --K 4000 --M 5 --kernel mala \
      --endpoints ../runs/endpoints --out ../runs/base-mala-K4000-M5/fwd.npz

Roots: forward uses end0.npz (pi_0), reverse uses end1.npz (pi_1); first --n-roots production chains,
--reps independent trajectories per root.  Trajectory randomness depends only on
(seed, direction, root, rep) so batches / chunks / restarts reproduce the same paths.
Partial progress is checkpointed (<out>.bXXX.state.npz) and resumed automatically.

Output npz: W, root_id, rep, acc (n_traj, K) float16 = mean MH acceptance probability per stage,
acc_traj_mean, Wbins (n_traj, nbins) = work increments summed in t-bins (t-bin of each increment's
midpoint; reverse increments are stored in the same t-bins, so -Wbins_R ~ +Wbins_F), x_final (float32),
U_start, nonfinite (count of non-finite increments), auxmax (learned: max GNN edge count seen), meta (json).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_default_matmul_precision', 'highest')
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import nes_core as C  # noqa: E402
import solvation_system as S  # noqa: E402
from estimators import atomic_npz  # noqa: E402


def resolve_kernel(kernel, h, L, endpoints):
    """Return (kind, h, L, source).  Fixed rule, decided before any production run:
    among pilot-scan settings with mean acceptance >= 0.5 at BOTH ends, take the one maximising the
    smaller of the two ends' ESJD per gradient evaluation.  kernel='auto' compares mala and hmc."""
    if kernel == 'rw':
        return 'rw', (float(S.target().sampling_dt) if h in (None, 'auto') else float(h)), 1, \
            ('upstream sampling_dt' if h in (None, 'auto') else 'cli')
    if h not in (None, 'auto') and (kernel == 'mala' or L not in (None, 'auto')):
        return kernel, float(h), (1 if kernel == 'mala' else int(L)), 'cli'
    metas = [json.loads((Path(endpoints) / f'end{e}.json').read_text()) for e in (0, 1)]
    cands = {}
    for e, meta in enumerate(metas):
        for r in meta.get('mala_scan', []) + meta.get('hmc_scan', []):
            kind = r.get('kind', 'mala')
            cands.setdefault((kind, r['h'], int(r.get('L', 1))), {})[e] = r
    keep = []
    for (kind, hh, LL), rr in cands.items():
        if len(rr) < 2 or (kernel != 'auto' and kind != kernel):
            continue
        if h not in (None, 'auto') and hh != float(h):
            continue
        if L not in (None, 'auto') and kind == 'hmc' and LL != int(L):
            continue
        if min(rr[0]['acc_mean'], rr[1]['acc_mean']) >= 0.5:
            keep.append(((kind, hh, LL), min(rr[0]['esjd_per_grad'] if 'esjd_per_grad' in rr[0] else
                                             rr[0]['esjd_per_particle'],
                                             rr[1]['esjd_per_grad'] if 'esjd_per_grad' in rr[1] else
                                             rr[1]['esjd_per_particle'])))
    if not keep:
        raise RuntimeError(f'no {kernel} setting with acceptance >= 0.5 at both ends; pass --h/--L explicitly')
    (kind, hh, LL), score = max(keep, key=lambda kv: kv[1])
    return kind, float(hh), LL, f'auto ({kernel}): max min-end ESJD/grad={score:.3e} with acc>=0.5 (pilots only)'


def resolve_h(arg, endpoints, kernel):
    """Backward-compatible helper (smoke.py): h only."""
    kind, h, L, src = resolve_kernel(kernel, arg, None, endpoints)
    return h, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', choices=['base', 'learned'], required=True)
    ap.add_argument('--direction', choices=['fwd', 'rev'], required=True)
    ap.add_argument('--K', type=int, default=None)
    ap.add_argument('--grad-budget', type=int, default=None,
                    help='gradient evaluations per trajectory; K = round(G / grads_per_stage(kernel, M, L))')
    ap.add_argument('--M', type=int, default=5)
    ap.add_argument('--kernel', choices=['mala', 'rw', 'hmc', 'auto'], default='mala')
    ap.add_argument('--h', default='auto')
    ap.add_argument('--L', default='auto', help='HMC leapfrog steps (auto: from pilot scan)')
    ap.add_argument('--cap-factor', type=float, default=5.0, help='MALA drift cap per particle, units of sqrt(2h); <=0 disables')
    ap.add_argument('--endpoints', required=True)
    ap.add_argument('--ckpt', default=str(S.DEFAULT_CKPT))
    ap.add_argument('--n-roots', type=int, default=256)
    ap.add_argument('--reps', type=int, default=1)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--chunk', type=int, default=500)
    ap.add_argument('--nbins', type=int, default=100)
    ap.add_argument('--seed', type=int, default=7110)
    ap.add_argument('--checkpoint-seconds', type=float, default=600)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        print('exists, skipping', out)
        return
    started = time.time()
    path = S.make_path(a.path, a.ckpt)
    kind, h, L, h_source = resolve_kernel(a.kernel, a.h, a.L, a.endpoints)
    if (a.K is None) == (a.grad_budget is None):
        raise SystemExit('give exactly one of --K / --grad-budget')
    if a.grad_budget is not None:
        a.K = max(1, int(round(a.grad_budget / C.grads_per_stage(kind, a.M, L))))
    cap = a.cap_factor if (kind == 'mala' and a.cap_factor > 0) else None
    end = 0 if a.direction == 'fwd' else 1
    with np.load(Path(a.endpoints) / f'end{end}.npz') as z:
        X = z['x'][:a.n_roots]
        chain = z['chain_id'][:a.n_roots]
    if len(X) < a.n_roots:
        raise RuntimeError(f'only {len(X)} roots available')
    root_id = np.repeat(chain, a.reps)
    rep = np.tile(np.arange(a.reps), len(chain))
    x_all = np.repeat(X, a.reps, axis=0)
    n = len(root_id)

    t_dir = C.grid(a.K, a.direction)
    h_dir = np.full(a.K + 1, h)
    runner = C.make_chunk_runner(energy=path['energy'], value_and_grad=path['value_and_grad'], delta=path['delta'],
                                 kind=kind, wrap=C.wrap_torus, M=a.M, cap_factor=cap, nbins=a.nbins,
                                 aux=path['aux'], L=L)
    e0 = jax.jit(jax.vmap(path['energy'], in_axes=(0, None)))
    print(f'path={a.path} dir={a.direction} K={a.K} M={a.M} kernel={kind} L={L} h={h} ({h_source}) cap={cap} '
          f'roots={a.n_roots} reps={a.reps} batch={a.batch} devices={jax.devices()} info={path["info"]}', flush=True)

    results = []
    for b0 in range(0, n, a.batch):
        b = b0 // a.batch
        sl = slice(b0, min(n, b0 + a.batch))
        done = out.with_name(out.name + f'.b{b:03d}.npz')
        if done.exists():
            with np.load(done) as z:
                results.append({k: z[k] for k in z.files})
            print(f'batch {b}: loaded', flush=True)
            continue
        state_file = out.with_name(out.name + f'.b{b:03d}.state.npz')
        state = None
        if state_file.exists():
            with np.load(state_file) as z:
                state = {k: z[k] for k in z.files}
            state['k'] = int(state['k'])
            print(f'batch {b}: resuming at stage {state["k"]}', flush=True)
        keys = jax.vmap(lambda r, p: C.traj_key(a.seed, a.direction, r, p))(
            jnp.asarray(root_id[sl]), jnp.asarray(rep[sl]))
        x0 = jnp.asarray(x_all[sl])
        U_start = np.asarray(e0(x0, jnp.float64(t_dir[0])))
        tb = time.time()
        last = [time.time()]

        def on_chunk(st, state_file=state_file, tb=tb, last=last):
            el = time.time() - tb
            print(f'  batch {b} stage {st["k"]}/{a.K}  <W>={st["W"].mean():.3f}  '
                  f'acc={st["acc"][:, :st["k"]].mean():.3f}  {el:.0f}s', flush=True)
            if time.time() - last[0] > a.checkpoint_seconds and st['k'] < a.K:
                atomic_npz(state_file, **st)
                last[0] = time.time()

        st = C.run_protocol(runner, x0, keys, t_dir, h_dir, a.nbins, a.chunk, state=state, on_chunk=on_chunk)
        res = dict(W=st['W'], Wbins=st['Wbins'], acc=st['acc'].astype(np.float16),
                   acc_traj_mean=st['acc'].mean(1), x_final=st['x'].astype(np.float32), U_start=U_start,
                   nonfinite=st['nonfinite'], auxmax=st['auxmax'], root_id=root_id[sl], rep=rep[sl],
                   seconds=np.float64(time.time() - tb))
        atomic_npz(done, **res)
        if state_file.exists():
            state_file.unlink()
        results.append(res)
        print(f'batch {b}: done in {time.time() - tb:.0f}s', flush=True)

    merged = {k: np.concatenate([r[k] for r in results]) for k in results[0] if k != 'seconds'}
    gpu_seconds = float(sum(float(r['seconds']) for r in results))
    meta = dict(path=a.path, direction=a.direction, K=a.K, grad_budget=a.grad_budget, M=a.M, kernel=kind,
                kernel_requested=a.kernel, L=L,
                grads_per_stage=C.grads_per_stage(kind, a.M, L), h=h, h_source=h_source,
                cap_factor=cap, n_roots=a.n_roots, reps=a.reps, batch=a.batch, seed=a.seed, nbins=a.nbins,
                grid='linear', precision='float64', matmul_precision='highest',
                endpoints=str(Path(a.endpoints).resolve()), path_info=path['info'],
                upstream_commit=S.UPSTREAM_COMMIT, batch_seconds_sum=gpu_seconds,
                wall_seconds=time.time() - started,
                us_per_traj_stage=1e6 * gpu_seconds / (n * a.K))
    atomic_npz(out, meta=np.array(json.dumps(meta)), **merged)
    for b in range(len(results)):
        f = out.with_name(out.name + f'.b{b:03d}.npz')
        if f.exists():
            f.unlink()
    print(json.dumps(dict(out=str(out), W_mean=float(merged['W'].mean()), W_sd=float(merged['W'].std()),
                          nonfinite=int(merged['nonfinite'].sum()), acc=float(merged['acc_traj_mean'].mean()),
                          wall_seconds=meta['wall_seconds'])), flush=True)


if __name__ == '__main__':
    main()
