"""
Does naming accuracy scale with a person's web exposure? Use LFW #images-per-identity
as an exposure proxy (LFW ranges 1 .. 500+ images). Bin held-out probes by their
identity's image count and report top-1 naming accuracy per bin, for the aligned
KPRPE->CLIP map and the native ceiling. Answers R2/R3 (naming limited to public figures)
with a monotone finding instead of a caveat.
Out: eval_out/naming_exposure.csv (config,bin,lo,hi,n_probes,top1)
"""
import os
import numpy as np
from eval_naming import l2, name_anchors, SEED, TR

LFW = "embeddings/lfw"
BINS = [(1, 1), (2, 3), (4, 10), (11, 50), (51, 10**9)]
OUT = "eval_out"


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = np.load(f"{LFW}/lfw_metadata.npy", allow_pickle=True)
    ids = np.array([int(r["identity"]) for r in meta])
    names = [None] * ids.max()
    for r in meta:
        names[int(r["identity"]) - 1] = r["name"]
    counts = np.bincount(ids)[1:]                 # counts[k] = #imgs of identity k+1

    rng = np.random.RandomState(SEED)
    uniq = np.array(sorted(set(ids))); rng.shuffle(uniq)
    trset = set(uniq[:int(len(uniq) * TR)].tolist())
    tr = np.array([i for i, x in enumerate(ids) if x in trset])
    te = np.array([i for i, x in enumerate(ids) if x not in trset])
    gt = ids[te] - 1
    probe_count = counts[gt]                       # exposure of each probe's identity

    rows = []
    F = l2(np.load(f"{LFW}/clip.npy"))
    Tc = l2(name_anchors("clip", names)); Tc = l2(Tc - Tc.mean(0))
    A = l2(np.load(f"{LFW}/kprpe.npy"))
    muA, muF = A[tr].mean(0), F[tr].mean(0)
    W = np.linalg.lstsq(A[tr]-muA, F[tr]-muF, rcond=None)[0]
    configs = {
        "native":       l2(F[te] - F[tr].mean(0)),
        "aligned-kprpe": l2(l2((A[te]-muA) @ W + muF) - l2((A[tr]-muA) @ W + muF).mean(0)),
    }
    for cfg, E in configs.items():
        pred = (E @ Tc.T).argmax(1)
        correct = (pred == gt)
        print(f"\n=== {cfg} (overall top1 {100*correct.mean():.1f}%) ===", flush=True)
        for lo, hi in BINS:
            m = (probe_count >= lo) & (probe_count <= hi)
            if m.sum() == 0:
                continue
            acc = 100 * correct[m].mean()
            rows.append((cfg, f"{lo}-{hi if hi < 10**9 else '+'}", lo, hi, int(m.sum()), acc))
            print(f"  imgs {lo:>2}-{hi if hi<10**9 else '+':<3}  n={m.sum():5d}  top1 {acc:5.1f}", flush=True)

    with open(f"{OUT}/naming_exposure.csv", "w") as f:
        f.write("config,bin,lo,hi,n_probes,top1\n")
        for x in rows:
            f.write(f"{x[0]},{x[1]},{x[2]},{x[3]},{x[4]},{x[5]:.3f}\n")
    print(f"\nsaved {OUT}/naming_exposure.csv")


if __name__ == "__main__":
    main()
