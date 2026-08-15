"""
Cross-dataset naming: does the FR->foundation alignment transfer across datasets?
Fit the linear map on ALL of CFP; evaluate naming on the held-out LFW test split.
Compared side-by-side with (a) native LFW ceiling and (b) within-LFW-fit map,
on the SAME LFW test probes, so the only variable is where the map was estimated.
Vocabulary = all LFW names. Targets clip/metaclip/siglip; sources arc/ada/adavit/kprpe.
Out: eval_out/naming_crossds.csv (foundation,source,fit,top1,top5,top10,dir1,dir10,auroc)
"""
import os
import numpy as np
from eval_naming import (l2, name_anchors, dir_at_far, auroc, topk_stats, SEED, TR)

CFP = "embeddings/cfp"
LFW = "embeddings/lfw"
TARGETS = ["clip", "metaclip", "siglip"]
SOURCES = ["arcface", "adaface", "adaface_vit", "kprpe"]
OUT = "eval_out"


def main():
    os.makedirs(OUT, exist_ok=True)
    # ---- LFW names + identity-disjoint 70/30 split (seed 42, matches within-LFW run)
    lmeta = np.load(f"{LFW}/lfw_metadata.npy", allow_pickle=True)
    lids = np.array([int(r["identity"]) for r in lmeta])
    lnames = [None] * lids.max()
    for r in lmeta:
        lnames[int(r["identity"]) - 1] = r["name"]
    n_names = len(lnames)
    rng = np.random.RandomState(SEED)
    uniq = np.array(sorted(set(lids))); rng.shuffle(uniq)
    trset = set(uniq[:int(len(uniq) * TR)].tolist())
    ltr = np.array([i for i, x in enumerate(lids) if x in trset])
    lte = np.array([i for i, x in enumerate(lids) if x not in trset])
    gt = lids[lte] - 1
    te_ids = np.array(sorted(set(lids[lte])))
    r2 = np.random.RandomState(SEED)
    wl = set(r2.permutation(te_ids)[:len(te_ids) // 2])
    vocab = np.array(sorted(i - 1 for i in wl))
    inl = np.array([g in set(vocab) for g in gt])
    print(f"LFW test probes={len(lte)} names={n_names} chance={100/n_names:.3f}%", flush=True)

    # ---- CFP arrays (all, for cross-dataset fit)
    cfp = {s: l2(np.load(f"{CFP}/{s}.npy")) for s in SOURCES}
    cfp_found = {t: l2(np.load(f"{CFP}/{t}.npy")) for t in TARGETS}
    lfw_src = {s: l2(np.load(f"{LFW}/{s}.npy")) for s in SOURCES}

    def score(Ete, Etr_mean, Tc):
        Ec = l2(Ete - Etr_mean)
        S = Ec @ Tc.T
        t1, t5, t10 = topk_stats(S, gt)
        Sw = S[:, vocab]; smax, amax = Sw.max(1), Sw.argmax(1)
        ok = vocab[amax] == gt
        s_in, ok_in, s_out = smax[inl], ok[inl], smax[~inl]
        return (t1, t5, t10, 100*dir_at_far(s_in, ok_in, s_out, .01),
                100*dir_at_far(s_in, ok_in, s_out, .10), auroc(s_in, s_out))

    rows = []
    for ftag in TARGETS:
        Fl = l2(np.load(f"{LFW}/{ftag}.npy"))          # LFW foundation emb
        Fc = cfp_found[ftag]                            # CFP foundation emb
        Tc = l2(name_anchors(ftag, lnames))
        Tc = l2(Tc - Tc.mean(0))
        print(f"\n=== {ftag} ===", flush=True)

        # native ceiling on LFW test
        r_ = score(Fl[lte], Fl[ltr].mean(0), Tc)
        rows.append((ftag, "native", "-", *r_))
        print(f"  {'native':16s} [-]     top1 {r_[0]:5.1f} top10 {r_[2]:5.1f} AUROC {r_[5]:.3f}", flush=True)

        for src in SOURCES:
            Al = lfw_src[src]
            # (a) within-LFW fit: map estimated on LFW train
            muA, muF = Al[ltr].mean(0), Fl[ltr].mean(0)
            W = np.linalg.lstsq(Al[ltr]-muA, Fl[ltr]-muF, rcond=None)[0]
            Ete = l2((Al[lte]-muA) @ W + muF)
            Etr_mean = l2((Al[ltr]-muA) @ W + muF).mean(0)
            rw = score(Ete, Etr_mean, Tc)
            rows.append((ftag, src, "within", *rw))

            # (b) cross-dataset fit: map estimated on ALL CFP, applied to LFW
            Ac = cfp[src]
            muAc, muFc = Ac.mean(0), Fc.mean(0)
            Wc = np.linalg.lstsq(Ac-muAc, Fc-muFc, rcond=None)[0]
            EteC = l2((Al[lte]-muAc) @ Wc + muFc)
            EtrC_mean = l2((Ac-muAc) @ Wc + muFc).mean(0)   # center by aligned-source(CFP) mean (strict: no target-side stats)
            rc = score(EteC, EtrC_mean, Tc)
            rows.append((ftag, src, "cross", *rc))
            print(f"  {src:16s} within  top1 {rw[0]:5.1f} top10 {rw[2]:5.1f} AUROC {rw[5]:.3f}  "
                  f"|| cross  top1 {rc[0]:5.1f} top10 {rc[2]:5.1f} AUROC {rc[5]:.3f}", flush=True)

    with open(f"{OUT}/naming_crossds.csv", "w") as f:
        f.write("foundation,source,fit,top1,top5,top10,dir1,dir10,auroc\n")
        for r in rows:
            f.write(",".join(str(x) if isinstance(x, str) else f"{x:.3f}" for x in r) + "\n")
    print(f"\nsaved {OUT}/naming_crossds.csv")


if __name__ == "__main__":
    main()
