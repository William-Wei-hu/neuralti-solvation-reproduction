"""Train the velocity potential phi (v = grad_x phi) by flow-matching regression (FEAT eq. 19, gamma = 0).

U_theta (TSM EGNN) is NOT used here:
the loss depends only on the paired data, so phi can later be combined with any frozen U_theta that was trained
on the same gamma = 0 interpolant (EGNN TSM, MACE TSM).

Data and interpolant (identical to the upstream U_theta training):
  upstream paired_dataloader(cfg) on the upstream training frames (runs/egnn-seed0/data/LJ3Dwithsolute_{0,1}.00),
  read-only: two independent draws, solute translated to the origin, solvent Hungarian-matched.
  dx = minimage(x1 - x0);  x_t = (x0 + t dx) mod 1;  t ~ U(0, 1) per pair.
Target and loss:
  target = dx - mean_particles(dx)   (phi is translation invariant, so sum_i grad_i phi = 0; the uniform part of dx
                                      is orthogonal to every such field and only adds a constant to the loss)
  L = mean_{pairs, particles, coords} |grad_x phi(x_t, t) - target|^2      (lambda_t = 1, as FEAT)
Network: upstream U_GNN.U_model architecture (cfg.U_model of the TSM run), new init.
Training loop = upstream main.py structure: per row a new batch of 24 pairs (key1 from split(key)), then 20 steps on
  that batch, each step key = split(key)[0] for t.  Adam, lr exponential_decay(1e-3, 0.1 per 1e4 steps, end 1e-5)
  = upstream optim.  20 000 steps = 1000 rows.  jax flags = upstream cfg.jax_config (x64 off, rank promotion
  'raise', debug_nans on); matmul precision unset (= upstream training numerics).
Why float32 for training: the FB RND estimate is exact for ANY v, so training precision changes only how good v
  is, not the correctness of the estimate; S1 check C3 concerned the network inside the FB RND sum (estimation
  stays float64, phi params are cast to float64 there).  float32 is also what U_theta was trained with.
EMA of params, decay 0.999, updated after every step.
Snapshots (never overwritten): runs/s2-phi/snapshots/step{NNNNN}.pkl every 1000 steps (params, ema, opt_state,
  step, key) ; logs: runs/s2-phi/train_log.npz (per-step training loss, per-row key data, validation curve).
Validation (fixed before training): S1b pilot roots (independent chains, never in training), pairs i <-> i of
  end0 / end1 chains 0..255, aligned with the upstream batch_align, 4 stratified t per pair (fixed rng);
  every 500 steps for params and EMA; baseline = same loss with v = 0 (mean target^2), R^2 = 1 - L / baseline.
Selection rule (plan 2.2, fixed): S3 uses the final EMA params; other snapshots only report validation losses.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NES_CODE = ROOT / 'nes'
sys.path.insert(0, str(NES_CODE))
import solvation_system as S  # noqa: E402  (puts upstream src/ on sys.path; sets no jax flags)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

TRAIN_RUN = Path(os.environ.get('NTI_RUNS', ROOT / 'runs')) / 'egnn-seed0'  # training frames and config
EXP = TRAIN_RUN / 'experiments'
VAL_ROOTS = ROOT / 'runs' / 'endpoints-s1b'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=1108)
    ap.add_argument('--snapshot-every', type=int, default=1000)
    ap.add_argument('--val-every', type=int, default=500)
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--n-val', type=int, default=256)
    ap.add_argument('--max-seconds', type=float, default=8400, help='stop cleanly (with a snapshot) before the limit')
    ap.add_argument('--out', default=str(ROOT / 'runs' / 's2-phi'))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'{out} exists; refusing to overwrite')
    (out / 'snapshots').mkdir(parents=True)
    t_start = time.time()

    import hydra
    import optax
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(EXP / 'config.yaml')
    for name, value in cfg.jax_config.items():          # upstream main.py: before any computation
        jax.config.update(name, value)
    nb = int(cfg.new_batch_every_n_step)
    assert a.steps % nb == 0 and a.snapshot_every % nb == 0 and a.val_every % nb == 0

    U_model = hydra.utils.instantiate(cfg.U_model)
    os.chdir(EXP)                                       # upstream dataloader reads ../data/<name>
    from dataloader import paired_dataloader, batch_align
    dl = paired_dataloader(cfg)
    info = dict(n_frames=[int(len(dl.X0)), int(len(dl.X1))], batch_size=int(dl.batch_size), rows_per_batch=nb,
                jax=jax.__version__, devices=str(jax.devices()), jax_config=OmegaConf.to_container(cfg.jax_config),
                cfg_U_model=OmegaConf.to_container(cfg.U_model, resolve=True), seed=a.seed, steps=a.steps,
                ema=a.ema, optim=OmegaConf.to_container(cfg.optim))
    print('info', json.dumps(info), flush=True)

    key = jax.random.PRNGKey(a.seed)
    kinit, key = jax.random.split(key)
    k1, k2 = jax.random.split(kinit)
    params = U_model.init(rngs=k1, t=jnp.ones(1), x=jax.random.uniform(k2, (47, 3)))
    n_params = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
    optim = hydra.utils.instantiate(cfg.optim)
    opt_state = optim.init(params)
    ema = params

    def loss_one(p, x0, x1, t):
        dx = ((x1 - x0) + 0.5) % 1 - 0.5
        xt = (x0 + t[0] * dx) % 1
        target = dx - dx.mean(0, keepdims=True)
        g = jax.grad(lambda y: U_model.apply(p, y, t).reshape())(xt)
        return ((g - target) ** 2).mean(), (target ** 2).mean()

    def loss_fn(p, batch, key):
        b0, b1 = batch
        t = jax.random.uniform(key, (len(b0), 1))
        L, base = jax.vmap(loss_one, in_axes=(None, 0, 0, 0))(p, b0, b1, t)
        return L.mean(), base.mean()

    @jax.jit
    def body(i, carry):
        params, ema, opt_state, key, losses, batch = carry
        key, _ = jax.random.split(key)                  # upstream body_fun
        (L, base), grad = jax.value_and_grad(loss_fn, has_aux=True)(params, batch, key)
        updates, opt_state = optim.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        ema = jax.tree_util.tree_map(lambda e, p: a.ema * e + (1 - a.ema) * p, ema, params)
        losses = losses.at[i].set(jnp.stack([L, base]))
        return params, ema, opt_state, key, losses, batch

    def row(params, ema, opt_state, key, batch):
        carry = (params, ema, opt_state, key, jnp.zeros((nb, 2)), batch)
        return jax.lax.fori_loop(0, nb, body, carry)
    row = jax.jit(row)

    # validation set (independent S1b pilot chains)
    with np.load(VAL_ROOTS / 'end0.npz') as z:
        V0 = z['x'][:a.n_val]
    with np.load(VAL_ROOTS / 'end1.npz') as z:
        V1 = z['x'][:a.n_val]
    V0, V1 = batch_align(jnp.asarray(V0, jnp.float32), jnp.asarray(V1, jnp.float32))
    rng = np.random.default_rng(20261108)
    tv = ((np.arange(4)[None, :] + rng.random((a.n_val, 4))) / 4).astype(np.float32)   # stratified

    @jax.jit
    def val_loss(p):
        def one(x0, x1, ts):
            return jax.vmap(lambda t: jnp.stack(loss_one(p, x0, x1, t.reshape(1))))(ts)
        r = jax.lax.map(lambda a3: one(*a3), (V0, V1, jnp.asarray(tv)))   # (n_val, 4, 2)
        return r.mean((0, 1))

    n_rows = a.steps // nb
    train_losses = np.zeros((n_rows, nb, 2), np.float32)
    row_keys = np.zeros((n_rows, 2), np.uint32)
    val = []
    snaps = []

    def snapshot(step):
        f = out / 'snapshots' / f'step{step:05d}.pkl'
        with open(f, 'wb') as fh:
            pickle.dump(dict(params=jax.device_get(params), ema=jax.device_get(ema),
                             opt_state=jax.device_get(opt_state), step=step, key=np.asarray(key),
                             seed=a.seed, cfg_U_model=OmegaConf.to_container(cfg.U_model, resolve=True)), fh)
        snaps.append(str(f.name))

    def do_val(step):
        lp = np.asarray(val_loss(params))
        le = np.asarray(val_loss(ema))
        rec = dict(step=step, val_params=float(lp[0]), val_ema=float(le[0]), baseline=float(lp[1]),
                   R2_params=float(1 - lp[0] / lp[1]), R2_ema=float(1 - le[0] / le[1]))
        val.append(rec)
        return rec

    def save_logs(done_rows):
        np.savez(out / 'train_log.npz', train_loss=train_losses[:done_rows, :, 0],
                 train_baseline=train_losses[:done_rows, :, 1], row_key_data=row_keys[:done_rows],
                 val=np.array(json.dumps(val)))

    print(f'n_params={n_params}  rows={n_rows}  t_setup={time.time() - t_start:.0f}s', flush=True)
    rec = do_val(0)
    print('val', rec, flush=True)
    t_train = time.time()
    done = 0
    stopped_early = False
    for r in range(n_rows):
        k1, k2 = jax.random.split(key)                  # upstream: key1 -> batch, key2 -> fori_loop chain
        row_keys[r] = np.asarray(k1)
        batch = dl.next(k1)
        params, ema, opt_state, key, losses, _ = row(params, ema, opt_state, k2, batch)
        train_losses[r] = np.asarray(losses)
        done = r + 1
        step = done * nb
        if step % a.val_every == 0:
            rec = do_val(step)
            el = time.time() - t_train
            print(f'step {step}  train {train_losses[r, :, 0].mean():.4e}  val {rec["val_params"]:.4e}  '
                  f'ema {rec["val_ema"]:.4e}  R2 {rec["R2_params"]:.4f}/{rec["R2_ema"]:.4f}  '
                  f'{el:.0f}s  {el / step:.3f} s/step', flush=True)
        if step % a.snapshot_every == 0:
            snapshot(step)
            save_logs(done)
        if time.time() - t_start > a.max_seconds and step < a.steps:
            stopped_early = True
            if step % a.snapshot_every != 0:
                snapshot(step)
            print(f'max-seconds reached at step {step}; stopping cleanly', flush=True)
            break
    save_logs(done)
    try:
        peak = int(jax.devices()[0].memory_stats()['peak_bytes_in_use'])
    except Exception as e:  # diagnostic only
        peak = f'unavailable: {e}'
    summary = dict(info=info, n_params=n_params, steps_done=done * nb, stopped_early=stopped_early,
                   snapshots=snaps, val=val, train_seconds=time.time() - t_train,
                   total_seconds=time.time() - t_start, s_per_step=(time.time() - t_train) / max(1, done * nb),
                   jax_peak_bytes=peak, final_selection='ema at the last step (plan 2.2)')
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('SUMMARY', json.dumps({k: v for k, v in summary.items() if k not in ('val', 'info')}), flush=True)


if __name__ == '__main__':
    main()
