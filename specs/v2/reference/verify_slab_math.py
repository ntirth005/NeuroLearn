"""Numerical checks for the slab algebra used in the v2 architecture spec (NumPy only)."""
import numpy as np

rng = np.random.default_rng(0)
d_in, d_out, r, S = 5120, 1024, 8, 6


def canon(B, A):
    """Rewrite B@A so that A has orthonormal rows. Returns (B', A')."""
    Q, R = np.linalg.qr(A.T)          # A.T = Q R,  A = R.T Q.T
    return B @ R.T, Q.T


slabs = []
for _ in range(S):
    A = rng.standard_normal((r, d_in)) / np.sqrt(d_in)
    B = rng.standard_normal((d_out, r)) * 0.02
    slabs.append((B, A))

# 1. canonical form preserves the update
B0, A0 = slabs[0]
Bc, Ac = canon(B0, A0)
print("1. canonicalisation error      :", np.abs(B0 @ A0 - Bc @ Ac).max())
print("   rows orthonormal            :", np.abs(Ac @ Ac.T - np.eye(r)).max())
canon_slabs = [canon(B, A) for B, A in slabs]

# 2. stacked two-GEMM forward == per-slab sum, with independent gates
x = rng.standard_normal((4, d_in))                       # 4 tokens
A_hot = np.concatenate([A for _, A in canon_slabs], 0)   # (S*r, d_in)
B_hot = np.concatenate([B for B, _ in canon_slabs], 1)   # (d_out, S*r)
u = x @ A_hot.T                                          # GEMM 1
e = (u.reshape(4, S, r) ** 2).sum(-1) / (x ** 2).sum(-1, keepdims=True)
theta, tau = 0.0016, 0.0005
w = 1 / (1 + np.exp(-(e - theta) / tau))                 # independent sigmoid gates
y = (u * np.repeat(w, r, axis=1)) @ B_hot.T              # GEMM 2
y_ref = sum(w[:, [b]] * (x @ A.T @ B.T) for b, (B, A) in enumerate(canon_slabs))
print("2. stacked vs per-slab forward :", np.abs(y - y_ref).max())

# 3. gate energy: fraction of input energy inside the slab's row space
print("3. energy, random x (expect r/d = %.5f): %.5f" % (r / d_in, e.mean()))
x_in = rng.standard_normal((4, r)) @ canon_slabs[0][1]   # x inside slab 0's row space
e_in = ((x_in @ canon_slabs[0][1].T) ** 2).sum(-1) / (x_in ** 2).sum(-1)
print("   energy, x in row space (expect 1):", e_in.round(6))

# 4. Frobenius cosine from r x r products
def cos(s1, s2):
    (B1, A1), (B2, A2) = s1, s2
    inner = np.trace((A1 @ A2.T) @ (B2.T @ B1))
    n1 = np.sqrt(np.trace((A1 @ A1.T) @ (B1.T @ B1)))
    n2 = np.sqrt(np.trace((A2 @ A2.T) @ (B2.T @ B2)))
    return inner / (n1 * n2)

D1, D2 = slabs[0][0] @ slabs[0][1], slabs[1][0] @ slabs[1][1]
print("4. cosine small-product vs dense:", cos(slabs[0], slabs[1]),
      (D1 * D2).sum() / (np.linalg.norm(D1) * np.linalg.norm(D2)))
cs = [abs(cos(slabs[i], slabs[j])) for i in range(S) for j in range(i)]
print("   |cos| between independent slabs: max %.4f  (PRD merge threshold 0.92)" % max(cs))

# 5. merge by QR + small SVD == truncated SVD of the dense sum
def merge(s1, s2, r_out):
    (B1, A1), (B2, A2) = s1, s2
    Bc_, Ac_ = np.concatenate([B1, B2], 1), np.concatenate([A1, A2], 0)   # rank 2r
    Qb, Rb = np.linalg.qr(Bc_)
    Qa, Ra = np.linalg.qr(Ac_.T)
    U, s, Vt = np.linalg.svd(Rb @ Ra.T)                                   # 2r x 2r core
    kept = (s[:r_out] ** 2).sum() / (s ** 2).sum()
    return (Qb @ U[:, :r_out]) * s[:r_out], Vt[:r_out] @ Qa.T, kept, s

Bm, Am, kept, s = merge(slabs[0], slabs[1], r)
U, sd, Vt = np.linalg.svd(D1 + D2, full_matrices=False)
best = (U[:, :r] * sd[:r]) @ Vt[:r]
print("5. merge vs dense truncated SVD:", np.abs(Bm @ Am - best).max())
print("   merged A rows orthonormal   :", np.abs(Am @ Am.T - np.eye(r)).max())
print("   energy kept, two unrelated rank-8 slabs -> rank 8: %.3f" % kept)

# 5b. two slabs with effective rank 3 and 4 pack into one rank-8 slab without loss
def low(rank):
    A = rng.standard_normal((rank, d_in)) / np.sqrt(d_in)
    B = rng.standard_normal((d_out, rank)) * 0.02
    return B, A
s3, s4 = low(3), low(4)
Bm, Am, kept, _ = merge(s3, s4, r)
print("   energy kept, rank 3 + rank 4 -> rank 8: %.6f, error %.2e"
      % (kept, np.abs(Bm @ Am - (s3[0] @ s3[1] + s4[0] @ s4[1])).max()))

# 6. sizes
for name, d, L, kv in [("0.5B", 896, 24, 128), ("1.5B", 1536, 28, 256), ("7B", 3584, 28, 512),
                       ("14B", 5120, 48, 1024), ("32B", 5120, 64, 1024)]:
    p = r * ((d + d) + (d + kv)) * L
    print("6. %-4s params/rank-8 slab (q+v, all layers): %6.2f M  bf16 %5.1f MB  64 hot: %6.1f MB"
          % (name, p / 1e6, p * 2 / 2**20, 64 * p * 2 / 2**20))
