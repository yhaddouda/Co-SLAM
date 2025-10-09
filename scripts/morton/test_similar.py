import numpy as np
import pandas as pd
from collections import defaultdict

ORIG = "inputs.csv"
MORT = "inputs_sorted.csv"
SEP = ","    # change if needed
HAS_HEADER = False
ATOL = 0  # tighten/loosen if you know your precision
RTOL = 0

def read64(path, sep=",", header=False):
    df = pd.read_csv(path, sep=sep, header=0 if header else None, dtype=np.float64)
    arr = df.values
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{path}: expected (N,3), got {arr.shape}")
    return arr

A = read64(ORIG, SEP, HAS_HEADER)     # shape (N,3)
B = read64(MORT, SEP, HAS_HEADER)

print("Shapes:", A.shape, B.shape)
if len(A) != len(B):
    raise SystemExit("Different lengths — not a permutation.")

# Quick multiset check (lex-sort + allclose)
def lexsort_rows(X):
    order = np.lexsort((X[:,2], X[:,1], X[:,0]))
    return X[order], order

A_sorted, A_order = lexsort_rows(A)
B_sorted, B_order = lexsort_rows(B)

same_multiset = np.allclose(A_sorted, B_sorted, rtol=RTOL, atol=ATOL, equal_nan=True)
print("Multiset equality (after lexsort, with tol):", same_multiset)

# If not equal, report max diff for diagnosis
if not same_multiset:
    if A_sorted.shape == B_sorted.shape:
        diff = np.abs(A_sorted - B_sorted)
        print("Max abs diff:", diff.max())
    raise SystemExit("Files are not the same set of points (beyond tolerance).")

# Build a permutation P such that B = A[P]
# Handles duplicates by mapping each quantized key to a queue of A-indices.
# Quantize to integer grid to make hashing robust to tiny fp jitter.
eps = max(ATOL, 1e-12)
QA = np.rint(A / eps).astype(np.int64)
QB = np.rint(B / eps).astype(np.int64)

buckets = defaultdict(list)
for i, key in enumerate(map(tuple, QA)):
    buckets[key].append(i)

P = np.empty(len(B), dtype=np.int64)
missing = 0
for j, key in enumerate(map(tuple, QB)):
    lst = buckets.get(key)
    if lst:
        P[j] = lst.pop()
        if not lst:
            del buckets[key]
    else:
        missing += 1

if missing or buckets:
    raise SystemExit(f"Could not match {missing} rows from B; extra keys in A: {len(buckets)}")

# Verify permutation
ok = np.allclose(A[P], B, rtol=RTOL, atol=ATOL, equal_nan=True)
print("Permutation verified:", ok)
if ok:
    print("Example mapping: B[0] == A[P[0]] =", A[P[0]], "vs", B[0])
