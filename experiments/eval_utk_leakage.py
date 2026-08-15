"""
UTKFace leakage check. UTK has no identity labels, so the paper splits by image;
the same person could fall in both train and test. We build an approximate
identity-disjoint split by clustering UTK via ArcFace similarity (union-find over
pairs with cos > tau), then re-run text-to-face retrieval on the cluster-disjoint
split and compare mean AP to the image-split number. If mAP is unchanged, image-level
leakage was not inflating the retrieval result.
Out: eval_out/utk_leakage.csv (split,target,source,mAP)
"""
import os
import numpy as np
from sklearn.metrics import average_precision_score
import eval_text_retrieval as r

TAU = 0.5          # conservative genuine-match threshold for same-identity linking
SEED = r.SEED
FACE = ["arcface", "adaface", "adaface_vit", "kprpe"]
TARGETS = ["clip", "metaclip", "siglip"]
OUT = "eval_out"


def union_find_clusters(E, tau, block=2000):
    n = len(E)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i0 in range(0, n, block):
        S = E[i0:i0+block] @ E.T                      # (block, n)
        for r_ in range(S.shape[0]):
            gi = i0 + r_
            js = np.where(S[r_, gi+1:] > tau)[0] + gi + 1
            for j in js:
                union(gi, j)
    roots = np.array([find(i) for i in range(n)])
    _, cl = np.unique(roots, return_inverse=True)
    return cl


def mean_ap(E, T, Q, te):
    aps = []
    for qi, (q, rel_all) in enumerate(Q):
        rel = rel_all[te]
        if rel.sum() < 5 or rel.mean() > 0.98:
            continue
        aps.append(average_precision_score(rel, E @ T[qi]))
    return float(np.mean(aps))


def main():
    os.makedirs(OUT, exist_ok=True)
    emb_dir, n, Q, _ = r.utk_queries()
    A0 = r.l2(np.load(f"{emb_dir}/arcface.npy").astype(np.float64))
    cl = union_find_clusters(A0.astype(np.float32), TAU)
    ncl = cl.max() + 1
    multi = np.bincount(cl); n_multi = int((multi > 1).sum())
    print(f"UTK {n} imgs -> {ncl} clusters ({n_multi} multi-image; "
          f"largest {multi.max()}); {n - ncl} imgs share an identity", flush=True)

    # cluster-disjoint 70/30 split
    cids = np.arange(ncl)
    cids = cids[np.random.RandomState(SEED).permutation(ncl)]
    kc = int(ncl * r.TR_RATIO)
    tr_cl = set(cids[:kc].tolist())
    mask = np.array([c in tr_cl for c in cl])
    tr_d, te_d = np.where(mask)[0], np.where(~mask)[0]

    # image split (paper baseline), same seed
    idx = np.random.RandomState(SEED).permutation(n); k = int(n * r.TR_RATIO)
    tr_i, te_i = idx[:k], idx[k:]

    rows = []
    for ftag in TARGETS:
        C = np.load(f"{emb_dir}/{ftag}.npy").astype(np.float64)
        T = r.encode_text(ftag, [q for q, _ in Q])
        for split, (trs, tes) in [("image", (tr_i, te_i)), ("identity", (tr_d, te_d))]:
            muC = C[trs].mean(0)
            rows.append((split, ftag, "native", mean_ap(r.l2(C[tes]), T, Q, tes)))
            for s in FACE:
                A = r.l2(np.load(f"{emb_dir}/{s}.npy").astype(np.float64))
                muA = A[trs].mean(0)
                W = np.linalg.lstsq(A[trs]-muA, C[trs]-muC, rcond=None)[0]
                rows.append((split, ftag, s, mean_ap(r.l2((A[tes]-muA) @ W + muC), T, Q, tes)))
        b = {(sp, s): m for sp, tt, s, m in rows if tt == ftag}
        print(f"  {ftag}: " + " ".join(
            f"{s}[img {b[('image',s)]:.3f}/id {b[('identity',s)]:.3f}]"
            for s in ["native"] + FACE), flush=True)

    with open(f"{OUT}/utk_leakage.csv", "w") as f:
        f.write("split,target,source,mAP\n")
        for x in rows:
            f.write(f"{x[0]},{x[1]},{x[2]},{x[3]:.4f}\n")
    print(f"\nsaved {OUT}/utk_leakage.csv")


if __name__ == "__main__":
    main()
