"""
Cross-dataset text-to-face retrieval: does the FR->foundation map transfer across
datasets for the *retrieval* task (not just naming)?
Two directions on the SAME held-out probes:
  CelebA->UTK : fit map on all CelebA, eval UTK retrieval (9 demographic queries)
  UTK->CelebA : fit map on all UTK,    eval CelebA retrieval (22 queries)
Each direction reports, on the target's 30% test split (seed 42):
  native (ceiling) / within-fit / cross-fit  mean AP.
Out: eval_out/retrieval_crossds.csv (direction,target,source,fit,mAP)
"""
import os
import numpy as np
from sklearn.metrics import average_precision_score
import eval_text_retrieval as r

SEED = r.SEED
FACE = ["arcface", "adaface", "adaface_vit", "kprpe"]
TARGETS = ["clip", "metaclip", "siglip"]
OUT = "eval_out"


def img_split(n, idents):
    if idents is not None:
        uids = np.unique(idents)
        uids = uids[np.random.RandomState(SEED).permutation(len(uids))]
        ku = int(len(uids) * r.TR_RATIO)
        tr_ids = set(uids[:ku].tolist())
        mask = np.array([i in tr_ids for i in idents])
        return np.where(mask)[0], np.where(~mask)[0]
    idx = np.random.RandomState(SEED).permutation(n)
    k = int(n * r.TR_RATIO)
    return idx[:k], idx[k:]


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
    data = {"utk": r.utk_queries(), "celeba": r.celeba_queries()}
    rows = []
    # (source_ds fit) -> (target_ds eval)
    for src_ds, tgt_ds in [("celeba", "utk"), ("utk", "celeba")]:
        s_dir, s_n, _, s_id = data[src_ds]
        t_dir, t_n, Q, t_id = data[tgt_ds]
        _, te = img_split(t_n, t_id)            # target test probes
        ttr, _ = img_split(t_n, t_id)           # target train (for within fit)
        print(f"\n#### {src_ds}->{tgt_ds}: {len(te)} target-test probes, {len(Q)} queries", flush=True)
        for ftag in TARGETS:
            Ct = np.load(f"{t_dir}/{ftag}.npy").astype(np.float64)   # target foundation
            Cs = np.load(f"{s_dir}/{ftag}.npy").astype(np.float64)   # source foundation
            T = r.encode_text(ftag, [q for q, _ in Q])
            rows.append((f"{src_ds}->{tgt_ds}", ftag, "-", "native",
                         mean_ap(r.l2(Ct[te]), T, Q, te)))
            for s in FACE:
                At = r.l2(np.load(f"{t_dir}/{s}.npy").astype(np.float64))
                As = r.l2(np.load(f"{s_dir}/{s}.npy").astype(np.float64))
                # within-target fit
                muA, muC = At[ttr].mean(0), Ct[ttr].mean(0)
                W = np.linalg.lstsq(At[ttr]-muA, Ct[ttr]-muC, rcond=None)[0]
                Ew = r.l2((At[te]-muA) @ W + muC)
                rows.append((f"{src_ds}->{tgt_ds}", ftag, s, "within", mean_ap(Ew, T, Q, te)))
                # cross fit (map on all source); strict: no target-side stats
                muAs, muCs = As.mean(0), Cs.mean(0)
                Wc = np.linalg.lstsq(As-muAs, Cs-muCs, rcond=None)[0]
                Ec = r.l2((At[te]-muAs) @ Wc + muCs)
                rows.append((f"{src_ds}->{tgt_ds}", ftag, s, "cross", mean_ap(Ec, T, Q, te)))
            # readable line per target
            def g(s, fit): return next(x[4] for x in rows if x[0].endswith(tgt_ds)
                                       and x[1] == ftag and x[2] == s and x[3] == fit)
            print(f"  {ftag}: native {g('-','native'):.3f} | "
                  + " ".join(f"{s}[w{g(s,'within'):.2f}/c{g(s,'cross'):.2f}]" for s in FACE), flush=True)

    with open(f"{OUT}/retrieval_crossds.csv", "w") as f:
        f.write("direction,target,source,fit,mAP\n")
        for x in rows:
            f.write(f"{x[0]},{x[1]},{x[2]},{x[3]},{x[4]:.4f}\n")
    print(f"\nsaved {OUT}/retrieval_crossds.csv")


if __name__ == "__main__":
    main()
