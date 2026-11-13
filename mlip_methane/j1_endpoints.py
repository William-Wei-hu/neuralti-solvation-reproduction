"""Endpoint equilibrium roots for the MACE-OFF23(S) methane switching runs.  NOT a free-energy run.
One GPU (A100 80 GB), float64 everywhere, model via mace_iface.py (backend cueq).

  python j1_endpoints.py --out runs/methane/endpoints [--resume-s1 EARLIER_OUT]

Chains: 20 per end (end 0: lam = 0 coupled; end 1: lam = 1 decoupled), chain c of end e built by
  build_system(216, seed = 1000 * e + c) and relaxed with FIRE at that end's lam (fmax < 1 eV/A, <= 200 steps).
  All 40 chains run in ONE batch (lam per configuration).
Stages (each writes its results to OUT before the next one starts):
  S1 equilibration: 10 ps Langevin (BAOAB, dt 0.5 fs, gamma 10/ps) per chain, float64.
  S2 dt scan (fixed rule): chains 0-7 of each end, copies at dt in {0.25, 0.35, 0.5, 0.7} fs,
     HMC (K1) L = 10, 100 MH steps each.  Choose the LARGEST dt with mean acceptance >= 0.6 at BOTH ends and every
     chain's acceptance >= 0.4.  None -> STOP (write the scan, no production).
  S3 production with the chosen dt: 4 segments x 2.5 ps per chain; one root at the end of every segment.
     Kernels, all exact, written as blocks of 4 sub-steps of 10 leapfrog steps (40 force calls
     per block, the same for every kernel, so they share one batch):
       K1 HMC full refresh, L = 10   : refresh (a = 0) and MH after every sub-step
       K2 HMC full refresh, L = 40   : refresh at sub-step 0 only, MH after sub-step 3 only (one 40-step trajectory)
       K3 GHMC, L = 10, gamma 1/ps   : partial refresh a = exp(-gamma * 10 * dt) and MH after every sub-step;
                                       on rejection p <- -p (momentum flip)
     Chains 0-6 of each end use K1, 7-13 K2, 14-19 K3.  One extra call per block evaluates E_int(x, lam = 1).
Rules (fixed before the run):
  kernel choice  observable: end 0 coupling energy E_int(x,1) - E_int(x,0); end 1 E_int(x,1).  Integrated
                 autocorrelation time tau (Sokal window, c = 5) from the block series of each kernel's chains (pooled
                 ACF), tau_calls = tau_blocks * 40; per kernel the larger of the two ends.  Eligible: mean acceptance
                 >= 0.6 (K1, K2) / >= 0.8 (K3) at both ends.  Choose argmin tau_calls among eligible kernels.
                 (No eligible kernel -> reported; roots are still valid, every kernel is exact.)
  decorrelation  roots of a chain are spaced by the segment length S; if S >= 2 tau_time of that chain's kernel at
                 that end, all 4 roots are usable, otherwise only the last root of each chain (no extension).
  drift          per end, mean observable over segments 3-4 minus segments 1-2 (chain means): |diff| <= 2 SE -> ok,
                 otherwise reported.
  stability      any O-H bond > 1.5 A or a methane C-H > 1.5 A, or any non-finite energy (incl. edge overflow, which
                 the interface turns into NaN): the chain is flagged; ANY flagged chain = "potential unstable" -> STOP.
Time guard (wall-clock limit 1 h 30 min; with --resume-s1 the limit is 1 h and the three times below become
  45 / 48 / 52 min): after S2 the production time is projected from the measured S1 speed; if it would end later than
  75 min after start, every segment is shortened by the same factor (not below 1.25 ps) and this is reported; the
  decorrelation rule then uses the actual S.  A segment is not started after 80 min or if it would end after 85 min;
  finished segments are kept.  (The roots used for the reported results came from a --resume-s1 run in which the
  guard shortened the segments to 2.28 ps.)
Memory: S2 and S3 call the device in chunks of CHUNK = 20 configurations (64 at once ran out of memory on an A100).
Resume: --resume-s1 DIR reuses the finished S1 of an earlier run (same S1 code) and skips relax and S1.
Outputs (never overwritten): OUT/ s1.npz, s2-scan.json, seg{1..4}.npz, roots_end{0,1}.npz, j1.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import jax

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_default_matmul_precision', 'highest')
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mlip_nes as M  # noqa: E402
import mace_iface as MI  # noqa: E402

RUN = None                                        # output directory, set from --out
RESUME_S1 = None                                  # earlier output directory with a finished S1, from --resume-s1
KT = M.kbt(298.15)
DT_EQ, GAMMA_EQ = 0.5, 0.01                       # fs, 1/fs
DTS, NSEG = (0.25, 0.35, 0.5, 0.7), 4
GAMMA_K3 = 1e-3                                   # 1/fs = 1/ps
NCH, NW, T_EQ, SCAN_CH, SCAN_STEPS, SEG_PS, SEG_MIN_PS = 20, 216, 10000.0, 8, 100, 2.5, 1.25
KERNEL_OF_CHAIN = ['K1'] * 7 + ['K2'] * 7 + ['K3'] * 6
T_PLAN, T_NOSTART, T_END = 75 * 60, 80 * 60, 85 * 60     # with --resume-s1: 45, 48, 52 min (set in __main__)
CHUNK = 20                                        # chains per device call in S2 / S3 (memory)
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def save_json(name, obj):
    (RUN / name).write_text(json.dumps(obj, indent=1, default=float))


# ------------------------------------------------------------------------------------------------ building blocks
def setup(do_relax=True):
    RUN.mkdir(parents=True, exist_ok=False)
    systems = [MI.build_system(NW, seed=1000 * e + c) for e in (0, 1) for c in range(NCH)]
    s0 = systems[0]
    for s in systems:
        assert np.array_equal(s['Z'], s0['Z']) and np.array_equal(s['solute'], s0['solute']) and s['L'] == s0['L']
    probe = MI.MaceIface('off23s', s0['Z'], s0['L'], s0['solute'], ref_pos=s0['pos0'])
    pad = int(1.25 * max(probe.count_edges(s['pos0']) for s in systems)) + 256
    m = MI.MaceIface('off23s', s0['Z'], s0['L'], s0['solute'], pad=pad)
    log('devices', jax.devices(), json.dumps(m.info))
    lam_end = np.array([0.0] * NCH + [1.0] * NCH)
    X = np.zeros((2 * NCH,) + s0['pos0'].shape)
    relax = {}
    for e in ((0, 1) if do_relax else ()):
        x, lg = m.relax(np.stack([s['pos0'] for s in systems[e * NCH:(e + 1) * NCH]]), lam=float(e), fmax=1.0,
                        max_steps=200)
        X[e * NCH:(e + 1) * NCH] = x
        relax[e] = lg
        log('relax end', e, json.dumps(dict(steps=lg['steps'], converged=lg['converged'], fmax_max=max(lg['fmax']))))
    return m, s0, X, lam_end, relax


def make_tools(m, s0):
    Z, L, sol = s0['Z'], float(s0['L']), s0['solute'].astype(bool)
    mass = jnp.asarray(m.masses)
    one = m._one

    def efn(x, lam):
        _, eint, F, ne = one(x, lam)
        return eint, F, jnp.asarray(ne).astype(jnp.int32)

    pairs = M.oh_pairs_from(Z, s0['pos0'], L, sol)
    sidx, widx = np.where(sol)[0], np.where(~sol)[0]
    probes = M.make_probes(sidx, widx, pairs, L)
    C = int(np.where(Z[sol] == 6)[0][0])
    Hm = np.where(Z[sol] == 1)[0]

    def diag(x):
        d = probes(x)
        ch = M.min_image(x[sidx[Hm]] - x[sidx[C]], L)
        d['max_ch'] = jnp.sqrt((ch ** 2).sum(-1).max())
        return d
    return efn, mass, L, diag


def langevin(efn, mass, L, x, lam, key, nsteps, dt, gamma):
    """BAOAB, float64; returns final x, p and per-chunk diagnostics are taken by the caller."""
    U, F, ne = efn(x, lam)
    p = M.draw_p(jax.random.fold_in(key, 0), mass, KT, x.dtype)
    c1 = math.exp(-gamma * dt)
    c2 = math.sqrt(1 - c1 * c1)

    def body(c, k):
        x, p, F, U, ne = c
        p = p + 0.5 * dt * F / M.MVV2E
        x = (x + 0.5 * dt * p / mass[:, None]) % L
        p = c1 * p + c2 * M.draw_p(k, mass, KT, x.dtype)
        x = (x + 0.5 * dt * p / mass[:, None]) % L
        U, F, n = efn(x, lam)
        p = p + 0.5 * dt * F / M.MVV2E
        return (x, p, F, U, jnp.maximum(ne, n)), None
    keys = jax.random.split(jax.random.fold_in(key, 1), nsteps)
    (x, p, F, U, ne), _ = jax.lax.scan(body, (x, p, F, U, ne), keys)
    return x, U, ne


def block(efn, mass, L, st, lam, key, dt, prm):
    """One block = 4 sub-steps x 10 leapfrog steps; prm = dict(a[4], refresh[4], mh[4], flip) per chain."""
    x, p, U, F = st['x'], st['p'], st['U'], st['F']
    xs, ps, Us, Fs, H0 = x, p, U, F, U + M.kinetic(p, mass)
    accs, nemax = [], jnp.zeros((), jnp.int32)
    for s in range(4):
        k1, k2 = jax.random.split(jax.random.fold_in(key, s))
        a = prm['a'][s]
        pn = a * p + jnp.sqrt(1 - a * a) * M.draw_p(k1, mass, KT, x.dtype)
        r = prm['refresh'][s]
        p = jnp.where(r, pn, p)
        xs, ps, Us, Fs = (jnp.where(r, x, xs), jnp.where(r, p, ps), jnp.where(r, U, Us), jnp.where(r, F, Fs))
        H0 = jnp.where(r, U + M.kinetic(p, mass), H0)
        x, p, U, F, ne = M.leapfrog(efn, x, p, F, lam, mass, dt, 10, L)
        nemax = jnp.maximum(nemax, ne)
        dH = U + M.kinetic(p, mass) - H0
        log_a = jnp.where(jnp.isfinite(dH), -dH / KT, -jnp.inf)
        acc = jnp.log(jax.random.uniform(k2, (), x.dtype)) < log_a
        do = prm['mh'][s]
        keep = (~do) | acc
        x, U, F = jnp.where(keep, x, xs), jnp.where(keep, U, Us), jnp.where(keep, F, Fs)
        p = jnp.where(keep, p, jnp.where(prm['flip'], -ps, ps))
        accs.append(jnp.where(do, jnp.exp(jnp.minimum(log_a, 0.0)), jnp.nan))
    return dict(x=x, p=p, U=U, F=F), jnp.stack(accs), nemax


def kernel_params(names, dt):
    a, refresh, mh, flip = [], [], [], []
    for k in names:
        if k == 'K1':
            a.append([0.] * 4); refresh.append([1] * 4); mh.append([1] * 4); flip.append(0)
        elif k == 'K2':
            a.append([0., 1., 1., 1.]); refresh.append([1, 0, 0, 0]); mh.append([0, 0, 0, 1]); flip.append(0)
        elif k == 'K3':
            g = math.exp(-GAMMA_K3 * 10 * dt)
            a.append([g] * 4); refresh.append([1] * 4); mh.append([1] * 4); flip.append(1)
    return dict(a=jnp.asarray(a, jnp.float64).T, refresh=jnp.asarray(refresh, bool).T,
                mh=jnp.asarray(mh, bool).T, flip=jnp.asarray(flip, bool))


# ------------------------------------------------------------------------------------------------ analysis helpers
def tau_int(series, c=5.0):
    """series (n_chains, n): pooled normalized ACF, Sokal window.  Returns tau in samples (>= 1 / 2 lower bound)."""
    x = series - series.mean(1, keepdims=True)
    n = x.shape[1]
    var = (x ** 2).mean()
    if not np.isfinite(var) or var <= 0:
        return float('nan')
    rho = np.array([(x[:, :n - t] * x[:, t:]).mean() / var for t in range(n // 2)])
    tau = 1.0
    for M_ in range(1, len(rho)):
        tau = 1.0 + 2.0 * rho[1:M_ + 1].sum()
        if M_ >= c * tau:
            break
    return float(max(tau, 0.5))


def main():
    resume = RESUME_S1                            # output dir of an earlier run whose S1 finished (same S1 code)
    m, s0, X, lam_end, relax = setup(do_relax=not resume)
    efn, mass, L, diag = make_tools(m, s0)
    lamj = jnp.asarray(lam_end)
    out = dict(info=m.info, relax={str(k): v for k, v in relax.items()}, deviations=[
        'S1 equilibration in float64 (a float32 Langevin was planned): the interface is float64 only'])
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(20261111), i))(jnp.arange(2 * NCH))
    if resume:
        src = Path(resume)
        z = np.load(src / 's1.npz')
        assert np.array_equal(z['lam'], lam_end) and z['x'].shape == X.shape
        assert int(json.loads((src / 'j1.json').read_text())['info']['pad']) == m.pad, 'pad differs from the S1 run'
        x = jnp.asarray(z['x'])
        np.savez(RUN / 's1.npz', **{k: z[k] for k in z.files})
        out['s1'] = dict(json.loads((src / 'j1.json').read_text())['s1'], resumed_from=str(src))
        s1_sec = out['s1']['seconds']
        log('S1 resumed from', src)
    else:
        # S1 equilibration
        lang = jax.jit(jax.vmap(lambda x, lam, k: langevin(efn, mass, L, x, lam, k, int(T_EQ / DT_EQ), DT_EQ,
                                                           GAMMA_EQ)))
        t1 = time.time()
        x, U, ne = jax.block_until_ready(lang(jnp.asarray(X), lamj, keys))
        s1_sec = time.time() - t1
        d = jax.vmap(diag)(x)
        np.savez(RUN / 's1.npz', x=np.asarray(x), U=np.asarray(U), n_edges=np.asarray(ne), lam=lam_end,
                 **{k: np.asarray(v) for k, v in d.items()})
        out['s1'] = dict(seconds=s1_sec, n_edges_max=int(np.asarray(ne).max()), pad=m.pad,
                         U_finite=bool(np.all(np.isfinite(np.asarray(U)))), max_oh=float(np.asarray(d['max_oh']).max()),
                         max_ch=float(np.asarray(d['max_ch']).max()))
    calls_s1 = 2 * NCH * int(T_EQ / DT_EQ)
    sec_per_batch_call = s1_sec / int(T_EQ / DT_EQ)
    log('S1', json.dumps(out['s1']))
    save_json('j1.json', out)

    # S2 dt scan
    sel = np.r_[0:min(SCAN_CH, NCH), NCH:NCH + min(SCAN_CH, NCH)]
    xs = np.repeat(np.asarray(x)[sel], len(DTS), axis=0)
    lams = np.repeat(lam_end[sel], len(DTS))
    dts = np.tile(np.array(DTS), len(sel))
    prm1 = kernel_params(['K1'], 0.5)
    prm1 = {k: v[..., 0] for k, v in prm1.items()}

    def scan_chain(x0, lam, dt, key):
        Uq, Fq, _ = efn(x0, lam)
        st = dict(x=x0, p=jnp.zeros_like(x0), U=Uq, F=Fq)

        def body(st, k):
            # one K1 sub-step = HMC L = 10 (use block with refresh/mh only at sub-step 0 is not needed: run 1 block = 4)
            st, acc, ne = block(efn, mass, L, st, lam, k, dt, prm1)
            return st, (acc, ne)
        st, (acc, ne) = jax.lax.scan(body, st, jax.random.split(key, SCAN_STEPS // 4))
        return acc, ne
    sc = jax.jit(jax.vmap(scan_chain))
    t2 = time.time()
    kscan = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(7), i))(jnp.arange(len(xs)))
    accs, nes = [], []
    for c0 in range(0, len(xs), CHUNK):              # chunks of CHUNK configurations (memory)
        sl = slice(c0, c0 + CHUNK)
        a_, n_ = sc(jnp.asarray(xs[sl]), jnp.asarray(lams[sl]), jnp.asarray(dts[sl]), kscan[sl])
        accs.append(np.asarray(a_))
        nes.append(np.asarray(n_))
    acc = np.concatenate(accs).reshape(len(xs), -1)
    ne = np.concatenate(nes)
    s2_sec = time.time() - t2
    rows = []
    for j, dtv in enumerate(DTS):
        r = {}
        for e in (0, 1):
            idx = [i for i in range(len(xs)) if dts[i] == dtv and lams[i] == float(e)]
            per = np.nanmean(acc[idx], axis=1)
            r[f'end{e}'] = dict(mean=float(np.nanmean(per)), min_chain=float(np.nanmin(per)))
        r['dt'] = dtv
        r['ok'] = all(r[f'end{e}']['mean'] >= 0.6 and r[f'end{e}']['min_chain'] >= 0.4 for e in (0, 1))
        rows.append(r)
    ok_dts = [r['dt'] for r in rows if r['ok']]
    out['s2'] = dict(rule='largest dt with mean acc >= 0.6 at both ends and every chain >= 0.4', rows=rows,
                     chosen_dt=max(ok_dts) if ok_dts else None, seconds=s2_sec, n_edges_max=int(np.asarray(ne).max()))
    save_json('s2-scan.json', out['s2'])
    save_json('j1.json', out)
    log('S2', json.dumps(out['s2']))
    if not ok_dts:
        out['verdict'] = 'STOP: no dt satisfies the acceptance rule'
        save_json('j1.json', out)
        log('STOP no dt')
        return 2
    dt = max(ok_dts)

    # S3 production
    blk_fs = 40 * dt
    remaining = T_PLAN - (time.time() - T0)
    per_block = 41 * sec_per_batch_call                   # 40 leapfrog calls + 1 observation call, same batch
    nb_full = int(round(SEG_PS * 1000 / blk_fs))
    need = NSEG * nb_full * per_block
    factor = 1.0 if need <= remaining else max(SEG_MIN_PS / SEG_PS, remaining / need)
    nb = max(1, int(nb_full * factor))
    seg_ps = nb * blk_fs / 1000
    out['s3_plan'] = dict(dt=dt, blocks_per_segment=nb, segment_ps=seg_ps, shortened=bool(nb < nb_full),
                          projected_seconds=NSEG * nb * per_block, remaining_seconds=remaining)
    log('S3 plan', json.dumps(out['s3_plan']))
    save_json('j1.json', out)
    names = KERNEL_OF_CHAIN * 2
    prm = kernel_params(names, dt)

    def seg_chain(st, lam, key, prm_c):
        def body(st, k):
            st, acc, ne = block(efn, mass, L, st, lam, k, dt, prm_c)
            obs, _, _ = efn(st['x'], 1.0)
            return st, dict(acc=acc, ne=ne, U=st['U'], E1=obs, **diag(st['x']))
        return jax.lax.scan(body, st, jax.random.split(key, nb))
    segf = jax.jit(jax.vmap(seg_chain, in_axes=(0, 0, 0, 0)))
    U0, F0, _ = jax.vmap(efn)(x, lamj)
    st = dict(x=x, p=jax.vmap(lambda k: M.draw_p(k, mass, KT, jnp.float64))(keys), U=U0, F=F0)
    prm_c = {k: (v.T if v.ndim == 2 else v) for k, v in prm.items()}   # per chain: a[4], refresh[4], mh[4], flip
    series, roots, done = [], [], 0
    for sgi in range(NSEG):
        if time.time() - T0 > T_NOSTART or time.time() - T0 + nb * per_block > T_END:
            log('hard time stop before segment', sgi + 1)
            break
        ts = time.time()
        k = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(100 + sgi), i))(jnp.arange(2 * NCH))
        parts_st, parts_rec = [], []
        for c0 in range(0, 2 * NCH, CHUNK):           # chunks of CHUNK chains (memory)
            sl = slice(c0, c0 + CHUNK)
            st_c, rec_c = segf({kk: v[sl] for kk, v in st.items()}, lamj[sl], k[sl],
                               {kk: v[sl] for kk, v in prm_c.items()})
            parts_st.append(st_c)
            parts_rec.append({kk: np.asarray(v) for kk, v in rec_c.items()})
        st = {kk: jnp.concatenate([p_[kk] for p_ in parts_st]) for kk in st}
        rec = {kk: np.concatenate([r_[kk] for r_ in parts_rec]) for kk in parts_rec[0]}
        np.savez(RUN / f'seg{sgi + 1}.npz', x_end=np.asarray(st['x']), **rec)
        series.append(rec)
        roots.append(np.asarray(st['x']))
        done += 1
        log(f'S3 segment {sgi + 1}', f'{time.time() - ts:.0f}s', 'acc', float(np.nanmean(rec['acc'])),
            'edges', int(rec['ne'].max()), 'max_oh', float(rec['max_oh'].max()))
    out['s3'] = dict(segments_done=done, seconds=time.time() - T0)

    # analysis
    A = {kk: np.concatenate([s[kk] for s in series], axis=1) for kk in series[0]}   # (chains, blocks, ...)
    obs = np.where(lam_end[:, None] == 0.0, A['E1'] - A['U'], A['U'])
    finite = np.isfinite(A['U']).all(1) & np.isfinite(A['E1']).all(1)
    unstable = (A['max_oh'].max(1) > 1.5) | (A['max_ch'].max(1) > 1.5) | ~finite
    kern = np.array(names)
    taus, accm = {}, {}
    for kname in ('K1', 'K2', 'K3'):
        for e in (0, 1):
            ch = (kern == kname) & (lam_end == float(e)) & ~unstable
            if ch.sum() < 1:
                taus[f'{kname}/end{e}'] = float('nan')
                accm[f'{kname}/end{e}'] = float('nan')
                continue
            taus[f'{kname}/end{e}'] = tau_int(obs[ch])
            accm[f'{kname}/end{e}'] = float(np.nanmean(A['acc'][ch]))
    sel_rows = {}
    for kname, thr in (('K1', 0.6), ('K2', 0.6), ('K3', 0.8)):
        tb = max(taus[f'{kname}/end0'], taus[f'{kname}/end1'])
        sel_rows[kname] = dict(tau_blocks=tb, tau_calls=tb * 40, tau_fs=tb * blk_fs,
                               acc=[accm[f'{kname}/end0'], accm[f'{kname}/end1']],
                               eligible=bool(min(accm[f'{kname}/end0'], accm[f'{kname}/end1']) >= thr))
    elig = {k: v for k, v in sel_rows.items() if v['eligible'] and np.isfinite(v['tau_calls'])}
    chosen = min(elig, key=lambda k: elig[k]['tau_calls']) if elig else None
    # decorrelation and roots
    usable = np.zeros((2 * NCH, done), bool)
    for i in range(2 * NCH):
        if unstable[i]:
            continue
        tf = taus[f'{kern[i]}/end{int(lam_end[i])}'] * blk_fs
        usable[i] = True if (np.isfinite(tf) and seg_ps * 1000 >= 2 * tf) else np.arange(done) == done - 1
    for e in (0, 1):
        ch = np.where(lam_end == float(e))[0]
        xs_e = np.stack([roots[s][i] for i in ch for s in range(done) if usable[i, s]]) if usable[ch].any() else np.zeros((0,))
        cid = np.array([1000 * e + (i - e * NCH) for i in ch for s in range(done) if usable[i, s]])
        seg = np.array([s + 1 for i in ch for s in range(done) if usable[i, s]])
        kk = np.array([kern[i] for i in ch for s in range(done) if usable[i, s]])
        np.savez(RUN / f'roots_end{e}.npz', x=xs_e, chain_id=cid, segment=seg, kernel=kk, Z=s0['Z'], L=s0['L'],
                 solute=s0['solute'], masses=m.masses)
    # drift
    drift = {}
    if done == NSEG:
        nbk = A['U'].shape[1] // NSEG
        for e in (0, 1):
            ch = (lam_end == float(e)) & ~unstable
            first = obs[ch, :2 * nbk].mean(1)
            last = obs[ch, 2 * nbk:].mean(1)
            dd = last - first
            se = dd.std(ddof=1) / np.sqrt(len(dd)) if len(dd) > 1 else float('nan')
            drift[f'end{e}'] = dict(diff=float(dd.mean()), se=float(se), ok=bool(abs(dd.mean()) <= 2 * se))
    out['analysis'] = dict(kernel_rule='argmin tau_calls (max over ends) among eligible (acc >= 0.6 K1/K2, >= 0.8 K3)',
                           kernels=sel_rows, chosen_kernel=chosen, tau_by_end=taus,
                           roots_usable=[int(usable[lam_end == float(e)].sum()) for e in (0, 1)],
                           unstable_chains=np.where(unstable)[0].tolist(), drift=drift,
                           n_edges_max=int(A['ne'].max()), pad=m.pad,
                           calls=dict(s1=calls_s1, s2=int(len(xs) * SCAN_STEPS // 4 * 40),
                                      s3=int(2 * NCH * done * nb * 41)))
    out['verdict'] = ('STOP: potential unstable (flagged chains)' if unstable.any() else
                      'OK' if chosen else 'no eligible kernel (reported)')
    out['seconds_total'] = time.time() - T0
    save_json('j1.json', out)
    log('ANALYSIS', json.dumps(out['analysis'], default=float))
    log('VERDICT', out['verdict'])
    return 0 if out['verdict'] == 'OK' else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True, help='new output directory (must not exist)')
    ap.add_argument('--resume-s1', default=None, help='earlier output directory whose S1 finished')
    a = ap.parse_args()
    RUN = Path(a.out)
    RESUME_S1 = a.resume_s1
    if RESUME_S1:
        T_PLAN, T_NOSTART, T_END = 45 * 60, 48 * 60, 52 * 60
    sys.exit(main())
