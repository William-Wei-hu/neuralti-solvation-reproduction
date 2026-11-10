"""Independent equilibrium roots at one end of the path (t=0 decoupled, t=1 coupled).

Why: the upstream endpoint files (runs/egnn-seed0/data/LJ3Dwithsolute_{0,1}.00) are 32 MCMC chains thinned
and then randomly permuted, so frames are strongly correlated and their chain identity is lost.  They
are NOT used here.

Procedure (same settings as upstream data/canonical_mcmc.py Canonical_Sampler):
  * n_chains independent chains, each from its own uniform random start, dispersed with the upstream
    Adam repulsion until min pair distance >= sqrt(0.7) sigma (over all chains).
  * random-walk MH, proposal x + N(0, 2*sampling_dt) mod 1, sampling_dt = 8e-8, target U0 or U1
    (target_system.U0 / U1, identical to both paths at t=0 / 1).
  * burn-in n_burn steps; keep ONLY the final frame of each chain -> one root per chain, root id = chain id.
  * the last n_pilot chains are pilots: used only to tune the MALA step h (never as NES roots).
Diagnostics: per-chain U trace every `record_every` steps, acceptance, drift of the chain-mean U between
the last two quarters of the burn-in (independent chains -> honest SE), MALA h and HMC (h, L) scans on
pilots (acceptance, expected squared jump distance ESJD and ESJD per gradient evaluation), timing.
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
from estimators import atomic_json, atomic_npz  # noqa: E402


def rw_sampler(energy, h, n_steps):
    step = jax.vmap(lambda x, U, k: C.mh_step(x, U, jnp.zeros_like(x), None, h, k, kind='rw',
                                              energy=lambda y, t: energy(y), value_and_grad=None,
                                              wrap=C.wrap_torus, cap_factor=None))

    @jax.jit
    def run(x, U, key):
        def body(i, carry):
            x, U, acc = carry
            keys = jax.random.split(jax.random.fold_in(key, i), x.shape[0])
            x, U, _, a, _ = step(x, U, keys)
            return x, U, acc + a
        x, U, acc = jax.lax.fori_loop(0, n_steps, body, (x, U, jnp.zeros(x.shape[0], x.dtype)))
        return x, U, acc / n_steps
    return run


def mala_scan(path, x, t, hs, n_steps, key, cap_factor, n_skip=50, kind='mala', L=1):
    """Short MALA / HMC runs from pilot frames for each h; acceptance and expected squared jump distance."""
    vg = path['value_and_grad']
    rows = []
    for j, h in enumerate(hs):
        step = jax.jit(jax.vmap(lambda x, U, g, k, h=h: C.mh_step(
            x, U, g, t, h, k, kind=kind, energy=path['energy'], value_and_grad=vg,
            wrap=C.wrap_torus, cap_factor=cap_factor if kind == 'mala' else None, L=L)))
        y = jnp.asarray(x)
        U, g = jax.jit(jax.vmap(lambda z: vg(z, t)))(y)
        accs, esjd = [], []
        t0 = None
        for i in range(n_steps):
            if i == n_skip:
                jax.block_until_ready(y)
                t0 = time.time()
            keys = jax.random.split(jax.random.fold_in(jax.random.fold_in(key, j), i), y.shape[0])
            y_new, U, g, a, _ = step(y, U, g, keys)
            if i >= n_skip:
                dr = ((y_new - y) + 0.5) % 1 - 0.5
                esjd.append(np.asarray(jnp.sum(dr ** 2, axis=(1, 2)) / y.shape[1]))
                accs.append(np.asarray(a))
            y = y_new
        jax.block_until_ready(y)
        sec = time.time() - t0
        gpe = L if kind == 'hmc' else 1
        rows.append(dict(kind=kind, L=int(L), h=float(h), noise_std=float(np.sqrt(2 * h)),
                         acc_mean=float(np.mean(accs)),
                         acc_per_chain_min=float(np.min(np.mean(accs, axis=0))),
                         esjd_per_particle=float(np.mean(esjd)), grads_per_step=gpe,
                         esjd_per_grad=float(np.mean(esjd)) / gpe,
                         us_per_config_step=1e6 * sec / ((n_steps - n_skip) * y.shape[0])))
        print(f'{kind} scan', rows[-1], flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--end', type=int, choices=[0, 1], required=True)
    ap.add_argument('--n-chains', type=int, default=256)
    ap.add_argument('--n-pilot', type=int, default=16)
    ap.add_argument('--n-burn', type=int, default=300000)
    ap.add_argument('--record-every', type=int, default=1000)
    ap.add_argument('--h', type=float, default=None, help='RW sampling_dt; default target_system.sampling_dt')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--mala-h', default='8e-8,2e-7,5e-7,1e-6,2e-6,4e-6,8e-6')
    ap.add_argument('--mala-steps', type=int, default=300)
    ap.add_argument('--hmc-h', default='1e-7,2.5e-7,5e-7,1e-6,2e-6')
    ap.add_argument('--hmc-L', default='5,10,20')
    ap.add_argument('--hmc-steps', type=int, default=120)
    ap.add_argument('--cap-factor', type=float, default=5.0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f'end{a.end}.npz'
    meta_file = out / f'end{a.end}.json'
    hs = [float(v) for v in a.mala_h.split(',')]
    if dest.exists():
        meta = json.loads(meta_file.read_text())
        if meta.get('mala_scan_done') and meta.get('hmc_scan_done'):
            print('exists, skipping', dest)
            return
        print('roots exist; running only the missing pilot scans', flush=True)
        with np.load(dest) as z:
            pilots = z['x_pilot']
        run_scan(meta, meta_file, pilots, a, hs)
        return
    started = time.time()
    ts = S.target()
    h = ts.sampling_dt if a.h is None else a.h
    seed = (20261107 + 1000 * a.end) if a.seed is None else a.seed
    energy = ts.U0 if a.end == 0 else ts.U1
    n_tot = a.n_chains + a.n_pilot
    key = jax.random.PRNGKey(seed)
    kinit, krun = jax.random.split(key, 2)
    print(f'end={a.end} chains={a.n_chains}+{a.n_pilot} pilots burn={a.n_burn} h={h} seed={seed} '
          f'devices={jax.devices()}', flush=True)

    t0 = time.time()
    x, iters = S.disperse(kinit, n_tot, ts)
    x = jnp.asarray(x)
    print(f'dispersal: {iters} Adam steps, {time.time() - t0:.0f}s', flush=True)

    Uf = jax.jit(jax.vmap(energy))
    U = Uf(x)
    run = rw_sampler(energy, h, a.record_every)
    n_rec = a.n_burn // a.record_every
    U_trace = np.zeros((n_tot, n_rec))
    acc_trace = np.zeros((n_tot, n_rec))
    t0 = time.time()
    for r in range(n_rec):
        x, U, acc = run(x, U, jax.random.fold_in(krun, r))
        U_trace[:, r] = np.asarray(U)
        acc_trace[:, r] = np.asarray(acc)
        if r % max(1, n_rec // 30) == 0 or r == n_rec - 1:
            el = time.time() - t0
            print(f'burn {(r + 1) * a.record_every}/{a.n_burn}  <U>={U_trace[:, r].mean():.3f}  '
                  f'acc={acc_trace[:, r].mean():.3f}  {el:.0f}s', flush=True)
    rw_sec = time.time() - t0
    # recompute final energy from scratch to check the running value
    U_check = np.asarray(Uf(x))
    x = np.asarray(x)

    q = n_rec // 4
    m3 = U_trace[:, 2 * q:3 * q].mean(1)
    m4 = U_trace[:, 3 * q:].mean(1)
    d = m4 - m3
    diag = dict(
        mean_U_q3=float(m3.mean()), mean_U_q4=float(m4.mean()),
        drift_q4_minus_q3=float(d.mean()), drift_se=float(d.std(ddof=1) / np.sqrt(len(d))),
        final_U_mean=float(U_check.mean()), final_U_sd=float(U_check.std(ddof=1)),
        final_U_running_vs_recomputed_maxabs=float(np.max(np.abs(U_check - U_trace[:, -1]))),
        acc_mean_last_quarter=float(acc_trace[:, 3 * q:].mean()),
        rw_us_per_config_step=1e6 * rw_sec / (a.n_burn * n_tot),
    )
    print('diagnostics', diag, flush=True)

    pilots = x[a.n_chains:]
    chain_id = np.arange(n_tot)
    # save roots first: a failure in the pilot scan must not lose the burn-in
    atomic_npz(dest, x=x[:a.n_chains], chain_id=chain_id[:a.n_chains], U=U_check[:a.n_chains],
               x_pilot=pilots, chain_id_pilot=chain_id[a.n_chains:], U_trace=U_trace, acc_trace=acc_trace,
               record_every=a.record_every)
    meta = dict(end=a.end, n_chains=a.n_chains, n_pilot=a.n_pilot, n_burn=a.n_burn, rw_h=h, seed=seed,
                dispersal_adam_steps=int(iters), target='target_system.U0' if a.end == 0 else 'target_system.U1',
                upstream_commit=S.UPSTREAM_COMMIT, precision='float64', diagnostics=diag,
                mala_scan=[], mala_scan_done=False, mala_cap_factor=a.cap_factor,
                burn_seconds=time.time() - started,
                independence='one final frame per independently initialised chain; root id = chain id')
    atomic_json(meta_file, meta)
    run_scan(meta, meta_file, pilots, a, hs)
    print(json.dumps(dict(dest=str(dest), seconds=time.time() - started)), flush=True)


def run_scan(meta, meta_file, pilots, a, hs):
    if len(pilots) == 0:
        meta['mala_scan_done'] = meta['hmc_scan_done'] = True
        atomic_json(meta_file, meta)
        return
    path = S.make_base_path()          # U_t at t=end equals U0/U1 for both paths
    t = jnp.float64(meta['end'])
    if not meta.get('mala_scan_done'):
        t0 = time.time()
        kscan = jax.random.fold_in(jax.random.PRNGKey(meta['seed']), 777)
        meta['mala_scan'] = mala_scan(path, jnp.asarray(pilots), t, hs, a.mala_steps, kscan, a.cap_factor)
        meta['mala_scan_done'] = True
        meta['mala_scan_seconds'] = time.time() - t0
        atomic_json(meta_file, meta)
    if not meta.get('hmc_scan_done'):
        t0 = time.time()
        rows = []
        hh = [float(v) for v in a.hmc_h.split(',')]
        for L in [int(v) for v in a.hmc_L.split(',')]:
            kscan = jax.random.fold_in(jax.random.PRNGKey(meta['seed']), 888 + L)
            rows += mala_scan(path, jnp.asarray(pilots), t, hh, a.hmc_steps, kscan, None,
                              n_skip=min(20, a.hmc_steps // 3), kind='hmc', L=L)
        meta['hmc_scan'] = rows
        meta['hmc_scan_done'] = True
        meta['hmc_scan_seconds'] = time.time() - t0
        atomic_json(meta_file, meta)


if __name__ == '__main__':
    main()
