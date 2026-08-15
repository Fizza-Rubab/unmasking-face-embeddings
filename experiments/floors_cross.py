"""Compute native/unaligned/random floors for the cross-dataset table (CLIP, seed 42).
Naming: LFW test (CFP->LFW protocol).  Retrieval: UTK test (CelebA->UTK protocol).
Floors use no map, so they are identical whether the map was fit within or cross.
native rows here must match naming_crossds.csv (34.04) and retrieval_crossds.csv (0.768).
"""
import numpy as np
from sklearn.metrics import average_precision_score
import eval_naming as en
import eval_text_retrieval as rt

# NAMING floors on LFW test (CLIP)
LFW = "embeddings/lfw"
lmeta = np.load(f"{LFW}/lfw_metadata.npy", allow_pickle=True)
lids = np.array([int(r["identity"]) for r in lmeta])
lnames = [None] * lids.max()
for r in lmeta:
    lnames[int(r["identity"]) - 1] = r["name"]
rng = np.random.RandomState(en.SEED)
uniq = np.array(sorted(set(lids))); rng.shuffle(uniq)
trset = set(uniq[:int(len(uniq) * en.TR)].tolist())
ltr = np.array([i for i, x in enumerate(lids) if x in trset])
lte = np.array([i for i, x in enumerate(lids) if x not in trset])
gt = lids[lte] - 1
F = en.l2(np.load(f"{LFW}/clip.npy"))
Tc = en.l2(en.name_anchors("clip", lnames)); Tc = en.l2(Tc - Tc.mean(0))
A0 = en.l2(np.load(f"{LFW}/arcface.npy"))

def nm(Ete, Etrmean):
    return en.topk_stats(en.l2(Ete - Etrmean) @ Tc.T, gt)  # (t1,t5,t10)

nat = nm(F[lte], F[ltr].mean(0))
U = A0 if A0.shape[1] >= F.shape[1] else np.pad(A0, ((0, 0), (0, F.shape[1]-A0.shape[1])))
U = U[:, :F.shape[1]]
un = nm(en.l2(U[lte]), en.l2(U[ltr]).mean(0))
G = np.random.RandomState(en.SEED).randn(F.shape[1], A0.shape[1]); Q, _ = np.linalg.qr(G)
rn = nm(en.l2(A0[lte] @ Q.T), en.l2(A0[ltr] @ Q.T).mean(0))
print(f"NAMING  LFW CLIP  (chance top1={100/len(lnames):.3f}%)")
print(f"  native    top1 {nat[0]:6.2f}  top5 {nat[1]:6.2f}")
print(f"  unaligned top1 {un[0]:6.2f}  top5 {un[1]:6.2f}")
print(f"  random    top1 {rn[0]:6.2f}  top5 {rn[1]:6.2f}")

# RETRIEVAL floors on UTK test (CLIP)
u_dir, u_n, Qs, u_id = rt.utk_queries()

def img_split(n, idents):
    if idents is not None:
        uids = np.unique(idents); uids = uids[np.random.RandomState(rt.SEED).permutation(len(uids))]
        tr_ids = set(uids[:int(len(uids)*rt.TR_RATIO)].tolist())
        mask = np.array([i in tr_ids for i in idents]); return np.where(mask)[0], np.where(~mask)[0]
    idx = np.random.RandomState(rt.SEED).permutation(n); k = int(n*rt.TR_RATIO); return idx[:k], idx[k:]

utr, ute = img_split(u_n, u_id)
C = np.load(f"{u_dir}/clip.npy").astype(np.float64)
T = rt.encode_text("clip", [q for q, _ in Qs])
muC = C[utr].mean(0)
A = rt.l2(np.load(f"{u_dir}/arcface.npy").astype(np.float64)); muA = A[utr].mean(0); d = A.shape[1]

def mean_ap(E):
    aps = []
    for qi, (q, rel_all) in enumerate(Qs):
        rel = rel_all[ute]
        if rel.sum() < 5 or rel.mean() > 0.98:
            continue
        aps.append(average_precision_score(rel, E @ T[qi]))
    return float(np.mean(aps))

nat_r = mean_ap(rt.l2(C[ute]))
Uu = A[ute]
if Uu.shape[1] < C.shape[1]:
    Uu = np.pad(Uu, ((0, 0), (0, C.shape[1]-Uu.shape[1])))
elif Uu.shape[1] > C.shape[1]:
    Uu = Uu[:, :C.shape[1]]
un_r = mean_ap(rt.l2(Uu))
Wr = np.random.RandomState(rt.SEED).normal(0, 1/np.sqrt(d), (d, C.shape[1]))
rn_r = mean_ap(rt.l2((A[ute]-muA) @ Wr + muC))
print(f"RETRIEVAL  UTK CLIP")
print(f"  native {nat_r:.4f}  unaligned {un_r:.4f}  random {rn_r:.4f}")
