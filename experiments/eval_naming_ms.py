"""
Multi-seed naming eval (error bars). Reuses eval_naming helpers.
Full paper source grid {arcface, adaface, adaface_vit, kprpe} x targets {clip, metaclip, siglip},
plus native/unaligned/random. 5 identity-disjoint splits -> mean +/- std.
Text anchors encoded ONCE per target (split-independent), cached across seeds.
Out: eval_out/naming_ms.csv  (foundation,source,metric,mean,std)
"""
import os
import numpy as np
import eval_naming as e

SEEDS = [42, 1, 2, 3, 4]              # 5 splits, matches IJCB protocol
TARGETS = ["clip", "metaclip", "siglip"]
SOURCES = ["arcface", "adaface", "adaface_vit", "kprpe"]
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
    meta = np.load(f"{e.EMB}/cfp_metadata.npy", allow_pickle=True)
    ids = np.array([int(r["identity"]) for r in meta])

    # cache per-target text anchors (split-independent) and raw foundation emb
    print("encoding text anchors (once per target)...", flush=True)
    Tc_cache, F_cache = {}, {}
    for ftag in TARGETS:
        F_cache[ftag] = e.l2(np.load(f"{e.EMB}/{ftag}.npy"))
        T = e.name_anchors(ftag, names)
        Tc_cache[ftag] = e.l2(T - T.mean(0))
    A_cache = {s: e.l2(np.load(f"{e.EMB}/{s}.npy")) for s in SOURCES}

    # results[(ftag, src)][metric] = list over seeds
    from collections import defaultdict
    res = defaultdict(lambda: defaultdict(list))
    METRICS = ["top1", "top5", "top10", "dir1", "dir10", "auroc"]

    for seed in SEEDS:
        tr, te = split(ids, seed)
        gt = ids[te] - 1
        te_ids = np.array(sorted(set(ids[te])))
        # watchlist: half the test ids (seeded by the split seed)
        r2 = np.random.RandomState(seed)
        wl = set(r2.permutation(te_ids)[:len(te_ids) // 2])
        vocab = np.array(sorted(i - 1 for i in wl))
        inl = np.array([g in set(vocab) for g in gt])
        print(f"\n### seed {seed}: {len(tr)}/{len(te)} tr/te", flush=True)

        for ftag in TARGETS:
            F = F_cache[ftag]; Tc = Tc_cache[ftag]
            variants = {"native": (F[te], F[tr])}
            for src in SOURCES:
                A = A_cache[src]
                muA, muF = A[tr].mean(0), F[tr].mean(0)
                W = np.linalg.lstsq(A[tr] - muA, F[tr] - muF, rcond=None)[0]
                variants[f"bridged-{src}"] = (e.l2((A[te]-muA) @ W + muF),
                                              e.l2((A[tr]-muA) @ W + muF))
            A0 = A_cache["arcface"]
            U = A0 if A0.shape[1] >= F.shape[1] else \
                np.pad(A0, ((0, 0), (0, F.shape[1]-A0.shape[1])))
            variants["unaligned"] = (e.l2(U[te][:, :F.shape[1]]),
                                     e.l2(U[tr][:, :F.shape[1]]))
            G = np.random.RandomState(seed).randn(F.shape[1], A0.shape[1])
            Q, _ = np.linalg.qr(G)
            variants["random"] = (e.l2(A0[te] @ Q.T), e.l2(A0[tr] @ Q.T))

            for lbl, (E_te, E_tr) in variants.items():
                Ec = e.l2(E_te - E_tr.mean(0))
                S = Ec @ Tc.T
                t1, t5, t10 = e.topk_stats(S, gt)
                Sw = S[:, vocab]
                smax, amax = Sw.max(1), Sw.argmax(1)
                ok = vocab[amax] == gt
                s_in, ok_in, s_out = smax[inl], ok[inl], smax[~inl]
                d1 = 100 * e.dir_at_far(s_in, ok_in, s_out, .01)
                d10 = 100 * e.dir_at_far(s_in, ok_in, s_out, .10)
                au = e.auroc(s_in, s_out)
                for m, v in zip(METRICS, [t1, t5, t10, d1, d10, au]):
                    res[(ftag, lbl)][m].append(v)

    with open(f"{OUT}/naming_ms.csv", "w") as f:
        f.write("foundation,source,metric,mean,std,n\n")
        for (ftag, lbl), md in res.items():
            for m in METRICS:
                vals = np.array(md[m])
                f.write(f"{ftag},{lbl},{m},{vals.mean():.4f},{vals.std():.4f},{len(vals)}\n")
    # human-readable
    print(f"\nchance top-1 = {100/n_names:.2f}%  ({len(SEEDS)} seeds)\n")
    for ftag in TARGETS:
        print(f"=== {ftag} ===")
        for lbl in ["native"] + [f"bridged-{s}" for s in SOURCES] + ["unaligned", "random"]:
            md = res[(ftag, lbl)]
            print(f"  {lbl:16s} top1 {np.mean(md['top1']):5.1f}±{np.std(md['top1']):.1f} "
                  f"top10 {np.mean(md['top10']):5.1f}±{np.std(md['top10']):.1f} "
                  f"DIR@1% {np.mean(md['dir1']):5.1f}±{np.std(md['dir1']):.1f} "
                  f"AUROC {np.mean(md['auroc']):.3f}±{np.std(md['auroc']):.3f}")
    print(f"\nsaved {OUT}/naming_ms.csv")


if __name__ == "__main__":
    main()
