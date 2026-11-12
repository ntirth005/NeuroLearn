"""Numerical checks for the fourth external review (NumPy only). Results: 05_RESEARCH_NOTES.md section 3.6.

    python3 verify_gate_and_retraction.py
"""
import numpy as np

rng = np.random.default_rng(0)
d_in, r, tau, delta, phi = 5120, 8, 0.5, 0.1, -1.1


def gate(e, phi=phi):
    g = np.log(e * d_in / r)
    s = 1 / (1 + np.exp(-(g - phi) / tau))
    return s, np.maximum(0.0, (s - delta) / (1 - delta))


# 1. gate ramp at the cut s = delta
e_cut = (r / d_in) * np.exp(phi) * (delta / (1 - delta)) ** tau
h = e_cut * 1e-6
dw_de = (gate(e_cut + h)[1] - gate(e_cut)[1]) / h
print("1. e at the cut                         : %.3e   (chance level r/d = %.3e)" % (e_cut, r / d_in))
print("   dw/de just above the cut (numeric)   : %.1f   (review's closed form: %.1f)"
      % (dw_de, d_in / r * delta / tau * ((1 - delta) / delta) ** tau * np.exp(-phi)))
print("   same slope in log-lift units, dw/dg  : %.3f   (= delta/tau; e-units inflate it by 1/e)" % (dw_de * e_cut))
print("   dw/dphi at the cut                   : %.3f" % (-dw_de * e_cut))
# gradient of w w.r.t. A has norm (dw/de) * 2*sqrt(e*(1-e)) for a unit input; scan the whole open range
es = np.geomspace(e_cut * (1 + 1e-9), 1 - 1e-9, 200001)
s, _ = gate(es)
dwde = s * (1 - s) / ((1 - delta) * tau * es)
gA = dwde * 2 * np.sqrt(es * (1 - es))
print("   |dw/dA| per token: at cut %.1f, at chance %.1f, max over open range %.1f (finite; a kink, not a pole)"
      % (gA[0], gA[np.argmin(abs(es - r / d_in))], gA.max()))

# 2. QR sign: R's diagonal is not guaranteed positive
A = rng.standard_normal((r, d_in)) / np.sqrt(d_in)
Q, _ = np.linalg.qr(A.T)
A0 = Q.T                                             # canonical
neg, dev_raw, dev_fix = 0, 0.0, 0.0
for _ in range(200):
    A1 = A0 + 1e-3 * rng.standard_normal(A0.shape) / np.sqrt(d_in)     # one small optimiser step
    Q, R = np.linalg.qr(A1.T)
    neg += int((np.diag(R) < 0).any())
    dev_raw = max(dev_raw, np.abs(R - np.eye(r)).max())
    sgn = np.sign(np.diag(R))
    dev_fix = max(dev_fix, np.abs(sgn[:, None] * R - np.eye(r)).max())
print("2. QR after a small step, 200 trials    : %d with a negative diagonal in R" % neg)
print("   max |R - I| as returned %.3f ; after sign fix %.2e" % (dev_raw, dev_fix))

# 2b. the same when the canonical A did not come from QR (e.g. after pack or prune, which use an SVD)
U, _, Vt = np.linalg.svd(A, full_matrices=False)
A_svd = U @ Vt                                       # orthonormal rows, not QR-shaped
neg, dev_raw = 0, 0.0
for _ in range(200):
    A1 = A_svd + 1e-3 * rng.standard_normal(A_svd.shape) / np.sqrt(d_in)
    Q, R = np.linalg.qr(A1.T)
    neg += int((np.diag(R) < 0).any())
    dev_raw = max(dev_raw, np.abs(R - np.eye(r)).max())
print("2b. A from an SVD, 200 trials           : %d with a negative diagonal; max |R - I| as returned %.3f" % (neg, dev_raw))

# 3. step on A: full Euclidean update vs update with its row-space part removed
D = 1e-2 * rng.standard_normal(A0.shape) / np.sqrt(d_in)
for name, step in (("full update", D), ("row-space part removed", D - (D @ A0.T) @ A0)):
    Q, R = np.linalg.qr((A0 + step).T)
    R = np.sign(np.diag(R))[:, None] * R
    print("3. %-24s |R - I| max = %.2e" % (name, np.abs(R - np.eye(r)).max()))

# 4. fast-slab decay applied to A and B, then retraction
gam = 0.01
B = rng.standard_normal((64, r))
Ad, Bd = (1 - gam) * A0, (1 - gam) * B
Q, R = np.linalg.qr(Ad.T)
A2, B2 = Q.T, Bd @ R.T
print("4. decay on A and B, then retract: |A row norm| = %.4f, ||B'||/||B|| = %.6f (=(1-g)^2 = %.6f)"
      % (np.linalg.norm(A2[0]), np.linalg.norm(B2) / np.linalg.norm(B), (1 - gam) ** 2))

# 5. guard: chance that 21 independent 'not worse' checks produce a false failure
print("5. P(at least one false failure), 21 checks at 2.5%% each: %.3f ; 7 checks: %.3f ; 7 with Holm at 5%%: <= 0.050"
      % (1 - 0.975 ** 21, 1 - 0.975 ** 7))

# 6. KV cache size, Qwen2.5-14B, bf16
layers, kv_heads, head_dim = 48, 8, 128
for ctx in (32768, 16384):
    print("6. KV cache at %5d tokens: %.2f GiB" % (ctx, layers * 2 * kv_heads * head_dim * ctx * 2 / 2 ** 30))

# 7. calibrated margin with a tiny own-domain std
for std in (0.02, 0.1):
    print("7. std %.2f: token 0.1 above the slab's mean margin -> %.1f ; token 2.5 below -> %.1f"
          % (std, 0.1 / std, -2.5 / std))
