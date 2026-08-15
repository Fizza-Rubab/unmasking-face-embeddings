"""
Multi-seed text-to-face retrieval (error bars). Reuses eval_text_retrieval helpers.
5 splits (identity-disjoint on CelebA, image-disjoint on UTK) -> mean +/- std mAP.
Text encoded ONCE per target (split-independent), cached across seeds.
Out: eval_out/text_retrieval_ms.csv  (dataset,target,ranker,mean,std,n)
"""
import os
import collections
import numpy as np
from sklearn.metrics import average_precision_score
import eval_text_retrieval as r

SEEDS = [42, 1, 2, 3, 4]
FACE = ["arcface", "adaface", "adaface_vit", "kprpe"]
TARGETS = ["clip", "metaclip", "siglip"]
TR = r.TR_RATIO
OUT = "eval_out"


def split(n, idents, seed):
    if idents is not None:
        uids = np.unique(idents)
        uids = uids[np.random.RandomState(seed).permutation(len(uids))]
        ku = int(len(uids) * TR)
        tr_ids = set(uids[:ku].tolist())
        mask = np.array([i in tr_ids for i in idents])
        return np.where(mask)[0], np.where(~mask)[0]
    idx = np.random.RandomState(seed).permutation(n)
    k = int(n * TR)
    return idx[:k], idx[k:]


def main():
    os.makedirs(OUT, exist_ok=True)
    # per-seed mAP grid: map_grid[(ds, ftag, ranker)] = list over seeds
    grid = collections.defaultdict(list)

    for ds, fn in r.DATASETS.items():
        emb_dir, n, Q, idents = fn()
        # cache text + embeddings (split-independent)
        Ccache = {t: np.load(f"{emb_dir}/{t}.npy").astype(np.float64) for t in TARGETS}
        Tcache = {t: r.encode_text(t, [q for q, _ in Q]) for t in TARGETS}
        Acache = {s: r.l2(np.load(f"{emb_dir}/{s}.npy").astype(np.float64)) for s in FACE}
        print(f"\n#### {ds}: {n} imgs, {len(Q)} queries", flush=True)

        for seed in SEEDS:
            tr, te = split(n, idents, seed)
            for ftag in TARGETS:
                C = Ccache[ftag]; T = Tcache[ftag]
                muC = C[tr].mean(0)
                variants = {"native": r.l2(C[te])}
                for src in FACE:
                    A = Acache[src]; muA = A[tr].mean(0)
                    W = np.linalg.lstsq(A[tr]-muA, C[tr]-muC, rcond=None)[0]
                    variants[f"bridged-{src}"] = r.l2((A[te]-muA) @ W + muC)
                # single shared unaligned/random (arcface) to match paper floors
                A0 = Acache["arcface"]
                U = A0[te]
                if U.shape[1] < C.shape[1]:
                    U = np.pad(U, ((0, 0), (0, C.shape[1]-U.shape[1])))
                variants["unaligned"] = r.l2(U[:, :C.shape[1]])
                d = A0.shape[1]
                Wr = np.random.RandomState(seed).normal(0, 1/np.sqrt(d), (d, C.shape[1]))
                variants["random"] = r.l2((A0[te]-A0[tr].mean(0)) @ Wr + muC)

                for lbl, E in variants.items():
                    aps = []
                    for qi, (q, rel_all) in enumerate(Q):
                        rel = rel_all[te]
                        if rel.sum() < 5 or rel.mean() > 0.98:
                            continue
                        aps.append(average_precision_score(rel, E @ T[qi]))
                    grid[(ds, ftag, lbl)].append(float(np.mean(aps)))

    with open(f"{OUT}/text_retrieval_ms.csv", "w") as f:
        f.write("dataset,target,ranker,mean,std,n\n")
        for k in sorted(grid):
            v = np.array(grid[k])
            f.write(f"{k[0]},{k[1]},{k[2]},{v.mean():.4f},{v.std():.4f},{len(v)}\n")

    for ds in r.DATASETS:
        print(f"\n=== {ds} (mean±std mAP, {len(SEEDS)} seeds) ===")
        for lbl in ["native"] + [f"bridged-{s}" for s in FACE] + ["unaligned", "random"]:
            row = "  %-18s " % lbl
            for ftag in TARGETS:
                v = np.array(grid[(ds, ftag, lbl)])
                row += f"{ftag}={v.mean():.3f}±{v.std():.3f}  "
            print(row)
    print(f"\nsaved {OUT}/text_retrieval_ms.csv")


if __name__ == "__main__":
    main()
