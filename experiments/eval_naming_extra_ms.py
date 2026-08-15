"""
Multi-seed (5) for two additional naming results:
  (A) matched-pair CNN (adaface_cnn_wf4m) top-1 on CLIP/MetaCLIP/SigLIP
  (B) vocabulary-size ablation: arcface & kprpe top-1 through CLIP at vocab 500..4000
Reuses eval_naming helpers. Out: eval_out/naming_extra_ms.csv
"""
import os
import numpy as np
import eval_naming as e

SEEDS = [42, 1, 2, 3, 4]
TR = e.TR
OUT = "eval_out"


def split(ids, seed):
    rng = np.random.RandomState(seed)
    uniq = np.array(sorted(set(ids))); rng.shuffle(uniq)
    trset = set(uniq[:int(len(uniq) * TR)])
    tr = np.array([i for i, x in enumerate(ids) if x in trset])
    te = np.array([i for i, x in enumerate(ids) if x not in trset])
    return tr, te


def main():
    os.makedirs(OUT, exist_ok=True)
    names = [ln.strip() for ln in open(e.NAMES_TXT) if ln.strip()]
    n_names = len(names)
    distract = e.build_distractors(names, max(e.VOCAB_SIZES) - n_names)
    allnames = names + distract
    meta = np.load(f"{e.EMB}/cfp_metadata.npy", allow_pickle=True)
    ids = np.array([int(r["identity"]) for r in meta])

    from collections import defaultdict
    res = defaultdict(list)

    # (A) matched CNN across 3 targets (top-1)
    for ftag in ["clip", "metaclip", "siglip"]:
        F = e.l2(np.load(f"{e.EMB}/{ftag}.npy"))
        Tc = e.l2(e.name_anchors(ftag, names)); Tc = e.l2(Tc - Tc.mean(0))
        A = e.l2(np.load(f"{e.EMB}/adaface_cnn_wf4m.npy"))
        for seed in SEEDS:
            tr, te = split(ids, seed); gt = ids[te] - 1
            muA, muF = A[tr].mean(0), F[tr].mean(0)
            W = np.linalg.lstsq(A[tr]-muA, F[tr]-muF, rcond=None)[0]
            Ec = e.l2(e.l2((A[te]-muA) @ W + muF) - e.l2((A[tr]-muA) @ W + muF).mean(0))
            t1 = e.topk_stats(Ec @ Tc.T, gt)[0]
            res[("matchedCNN", ftag)].append(t1)

    # (B) vocab-size ablation through CLIP for arcface & kprpe (top-1)
    F = e.l2(np.load(f"{e.EMB}/clip.npy"))
    Tall = e.name_anchors("clip", allnames)          # 500 real + distractors
    for src in ["arcface", "kprpe"]:
        A = e.l2(np.load(f"{e.EMB}/{src}.npy"))
        for seed in SEEDS:
            tr, te = split(ids, seed); gt = ids[te] - 1
            muA, muF = A[tr].mean(0), F[tr].mean(0)
            W = np.linalg.lstsq(A[tr]-muA, F[tr]-muF, rcond=None)[0]
            Ete = e.l2((A[te]-muA) @ W + muF); Etrm = e.l2((A[tr]-muA) @ W + muF).mean(0)
            Ec = e.l2(Ete - Etrm)
            for V in e.VOCAB_SIZES:
                Tv = e.l2(Tall[:V] - Tall[:V].mean(0))
                res[(f"vocab-{src}", V)].append(e.topk_stats(Ec @ Tv.T, gt)[0])

    with open(f"{OUT}/naming_extra_ms.csv", "w") as f:
        f.write("group,key,mean,std,n\n")
        for (g, k), vals in res.items():
            v = np.array(vals)
            f.write(f"{g},{k},{v.mean():.3f},{v.std():.3f},{len(v)}\n")
    print("=== matched CNN (adaface_cnn_wf4m) top-1, 5 seeds ===")
    for ftag in ["clip", "metaclip", "siglip"]:
        v = np.array(res[("matchedCNN", ftag)]); print(f"  {ftag}: {v.mean():.1f}±{v.std():.1f}")
    print("=== vocab ablation top-1 (CLIP), 5 seeds ===")
    for src in ["arcface", "kprpe"]:
        for V in e.VOCAB_SIZES:
            v = np.array(res[(f"vocab-{src}", V)]); print(f"  {src} @{V}: {v.mean():.1f}±{v.std():.1f}")
    print(f"saved {OUT}/naming_extra_ms.csv")


if __name__ == "__main__":
    main()
