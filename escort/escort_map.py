"""Local radial escort map around the solute, LJ version.

Single-atom solvent in the unit periodic box; the solute is particle 0 and is NOT moved by the map.
For every solvent particle i with minimum-image vector d_i = x_i - c (c = x_0) and r = |d_i|:
    y_i = c + g(r) d_i / r      if r < r_c,     y_i = x_i otherwise          (then wrapped mod 1)
g is a monotone rational-quadratic spline on [0, r_c] (Durkan et al. 2019, NeurIPS, "Neural Spline Flows",
eqs. 4-6 and the inverse in their appendix) with g(0) = 0, g(r_c) = r_c, g'(r_c) = 1, so g maps [0, r_c] onto
itself, is C^1 at r_c and the identity outside.  Requires r_c < 1/2 (minimum image single valued).
Jacobian: block diagonal over particles; for a particle inside, det = g'(r) (g(r)/r)^2, so
    ln|det J| = sum_{r_i < r_c} [ ln g'(r_i) + 2 ln(g(r_i)/r_i) ]                         (closed form)
Inverse: g^{-1} in closed form (root of a quadratic), so Phi^{-1}(Phi(x)) = x to rounding.

Spline parametrisation (n_bins = B):
  logits theta = (w[B], h[B], d[B-1], d0[1]);  widths = r_c (m + (1 - B m) softmax(w)), heights likewise,
  interior knot derivatives = m_d + softplus(d + c1),  derivative at r = 0 = m_d + softplus(d0 + c1),
  derivative at r = r_c fixed to 1;  c1 = log(exp(1 - m_d) - 1) so that theta = 0 gives every derivative = 1.
  theta = 0  ->  equal widths and heights and unit derivatives  ->  g(r) = r (identity up to rounding).
Stage maps: theta_k = dt_k * net(t_mid_k), net zero-initialised in its last layer, so the map is the identity at
initialisation and as dt -> 0.  Learned arm: net = MLP on t features.  Control arm: theta = dt * a * b with one
scalar a and a fixed pattern b (heights of the inner bins up, outer bins down: pushes particles outward).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

MIN_W = 1e-3        # minimum relative bin width / height (fraction of r_c)
MIN_D = 1e-3        # minimum derivative
C1 = float(np.log(np.expm1(1.0 - MIN_D)))


def n_logits(B):
    return 3 * B


def spline_params(theta, r_c, B):
    w, h, d, d0 = theta[:B], theta[B:2 * B], theta[2 * B:3 * B - 1], theta[3 * B - 1:3 * B]
    widths = r_c * (MIN_W + (1 - B * MIN_W) * jax.nn.softmax(w))
    heights = r_c * (MIN_W + (1 - B * MIN_W) * jax.nn.softmax(h))
    xk = jnp.concatenate([jnp.zeros(1, theta.dtype), jnp.cumsum(widths)])
    yk = jnp.concatenate([jnp.zeros(1, theta.dtype), jnp.cumsum(heights)])
    xk = xk.at[-1].set(r_c)
    yk = yk.at[-1].set(r_c)
    dk = jnp.concatenate([MIN_D + jax.nn.softplus(d0 + C1), MIN_D + jax.nn.softplus(d + C1), jnp.ones(1, theta.dtype)])
    return xk, yk, dk


def _bin(knots, v):
    B = knots.shape[0] - 1
    return jnp.clip(jnp.searchsorted(knots, v, side='right') - 1, 0, B - 1)


def rq_forward(r, xk, yk, dk):
    """g(r) and g'(r) for r in [0, r_c] (callers mask r >= r_c)."""
    k = _bin(xk, r)
    x0, x1, y0, y1 = xk[k], xk[k + 1], yk[k], yk[k + 1]
    d0, d1 = dk[k], dk[k + 1]
    w = x1 - x0
    s = (y1 - y0) / w
    xi = (r - x0) / w
    om = xi * (1 - xi)
    den = s + (d1 + d0 - 2 * s) * om
    g = y0 + (y1 - y0) * (s * xi ** 2 + d0 * om) / den
    dg = s ** 2 * (d1 * xi ** 2 + 2 * s * om + d0 * (1 - xi) ** 2) / den ** 2
    return g, dg


def rq_inverse(y, xk, yk, dk):
    """g^{-1}(y) for y in [0, r_c]."""
    k = _bin(yk, y)
    x0, x1, y0, y1 = xk[k], xk[k + 1], yk[k], yk[k + 1]
    d0, d1 = dk[k], dk[k + 1]
    w = x1 - x0
    hh = y1 - y0
    s = hh / w
    t = y - y0
    a = hh * (s - d0) + t * (d1 + d0 - 2 * s)
    b = hh * d0 - t * (d1 + d0 - 2 * s)
    c = -s * t
    disc = jnp.maximum(b ** 2 - 4 * a * c, 0.0)
    xi = (2 * c) / (-b - jnp.sqrt(disc))
    return x0 + xi * w


def _radial(x, r_c):
    c = x[0]
    d = x - c
    d = d - jnp.round(d)
    r = jnp.sqrt(jnp.sum(d * d, axis=-1))
    inside = (r < r_c) & (jnp.arange(x.shape[0]) > 0)
    return c, d, r, inside


def apply_map(x, theta, r_c, B):
    """y = Phi(x) (wrapped) and ln|det dPhi/dx (x)|."""
    xk, yk, dk = spline_params(theta, r_c, B)
    c, d, r, inside = _radial(x, r_c)
    rs = jnp.where(inside, r, 0.5 * r_c)                     # safe value outside (masked)
    g, dg = rq_forward(rs, xk, yk, dk)
    ratio = jnp.where(rs > 1e-12, g / jnp.maximum(rs, 1e-300), dk[0])
    y = jnp.where(inside[:, None], c + d * ratio[:, None], x)
    ld = jnp.sum(jnp.where(inside, jnp.log(dg) + 2 * jnp.log(ratio), 0.0))
    return y % 1.0, ld


def apply_inverse(y, theta, r_c, B):
    """x = Phi^{-1}(y) (wrapped) and ln|det dPhi/dx (x)| at that x (to be ADDED to the reverse work)."""
    xk, yk, dk = spline_params(theta, r_c, B)
    c, d, ry, inside = _radial(y, r_c)                        # g maps [0, r_c) onto itself: inside-ness is preserved
    rys = jnp.where(inside, ry, 0.5 * r_c)
    rx = rq_inverse(rys, xk, yk, dk)
    ratio_inv = jnp.where(rys > 1e-12, rx / jnp.maximum(rys, 1e-300), 1.0 / dk[0])
    x = jnp.where(inside[:, None], c + d * ratio_inv[:, None], y)
    g, dg = rq_forward(rx, xk, yk, dk)
    ratio = jnp.where(rx > 1e-12, g / jnp.maximum(rx, 1e-300), dk[0])
    ld = jnp.sum(jnp.where(inside, jnp.log(dg) + 2 * jnp.log(ratio), 0.0))
    return x % 1.0, ld


# ------------------------------------------------------------------------------------------------ parameter nets
def t_features(t, n_freq=4):
    k = jnp.arange(1, n_freq + 1, dtype=jnp.result_type(t, jnp.float32))
    return jnp.concatenate([jnp.reshape(t, (1,)), jnp.sin(jnp.pi * k * t), jnp.cos(jnp.pi * k * t)])


def init_mlp(key, B, width=64, n_freq=4, dtype=jnp.float64):
    """MLP t-features -> 3B logits; last layer zero (identity map at initialisation)."""
    d_in = 1 + 2 * n_freq
    k1, k2 = jax.random.split(key)
    return dict(W1=jax.random.normal(k1, (d_in, width), dtype) / np.sqrt(d_in), b1=jnp.zeros(width, dtype),
                W2=jax.random.normal(k2, (width, width), dtype) / np.sqrt(width), b2=jnp.zeros(width, dtype),
                W3=jnp.zeros((width, n_logits(B)), dtype), b3=jnp.zeros(n_logits(B), dtype))


def mlp_logits(p, t_mid):
    h = jnp.tanh(t_features(t_mid) @ p['W1'] + p['b1'])
    h = jnp.tanh(h @ p['W2'] + p['b2'])
    return h @ p['W3'] + p['b3']


def control_pattern(B, dtype=jnp.float64):
    """Fixed pattern for the one-parameter control map: inner bins taller, outer bins shorter (outward push)."""
    hpat = jnp.cos(jnp.pi * (jnp.arange(B, dtype=dtype) + 0.5) / B)
    return jnp.concatenate([jnp.zeros(B, dtype), hpat, jnp.zeros(B, dtype)])


def stage_theta(kind, params, ta, tb, B):
    """Logits of the map for the stage between ta and tb (either order): dt * net(t_mid).  kind: 'mlp' | 'control'."""
    lo, hi = jnp.minimum(ta, tb), jnp.maximum(ta, tb)
    dt = hi - lo
    tm = 0.5 * (lo + hi)
    if kind == 'mlp':
        return dt * mlp_logits(params, tm)
    if kind == 'control':
        return dt * params['a'] * control_pattern(B, jnp.result_type(dt))
    if kind == 'fixed':                       # the same logits in every stage (not scaled by dt): one-stage maps
        return params['theta'] + 0.0 * dt
    raise ValueError(kind)
