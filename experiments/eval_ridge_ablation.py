"""
Ridge ablation for the Method-section claim: top-1 naming accuracy (CFP, 500 names,
CLIP target) for the plain lstsq bridge vs ridge alpha = 1, 10, 100.
Reuses the protocol of eval_naming.py (identity-disjoint split, prompt-ensemble
anchors, image/text mean-centering at scoring time).
Output: eval_out/ridge_ablation.csv
"""
import os
CACHE = ".cache"
os.environ.setdefault("HF_HOME", os.path.join(CACHE, "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np

from eval_naming import l2, name_anchors, NAMES_TXT, EMB, SEED, TR, OUT

ALPHAS = [0.0, 1.0, 10.0, 100.0]
SOURCES = ["arcface", "kprpe"]
TARGET = "clip"


def fit_bridge(X, Y, alpha):
    if alpha == 0.0:
        return np.linalg.lstsq(X, Y, rcond=None)[0]
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + alpha * np.eye(d), X.T @ Y)


def main():
    names = [ln.strip() for ln in open(NAMES_TXT) if ln.strip()]
    meta = np.load(f"{EMB}/cfp_metadata.npy", allow_pickle=True)
    ids = np.array([int(r["identity"]) for r in meta])
    rng = np.random.RandomState(SEED)
    uniq = np.array(sorted(set(ids))); rng.shuffle(uniq)
    trset = set(uniq[:int(len(uniq) * TR)])
    tr = np.array([i for i, x in enumerate(ids) if x in trset])
    te = np.array([i for i, x in enumerate(ids) if x not in trset])
    gt = ids[te] - 1

    F = l2(np.load(f"{EMB}/{TARGET}.npy"))
    T = name_anchors(TARGET, names)
    muF = F[tr].mean(0)
    Tc = l2(T - T.mean(0))

    rows = []
    for src in SOURCES:
        A = l2(np.load(f"{EMB}/{src}.npy"))
        muA = A[tr].mean(0)
        for a in ALPHAS:
            W = fit_bridge(A[tr] - muA, F[tr] - muF, a)
            Bte = l2((A[te] - muA) @ W + muF)
            mu_img = l2((A[tr] - muA) @ W + muF).mean(0)
            S = l2(Bte - mu_img) @ Tc.T
            top1 = 100.0 * np.mean(np.argmax(S, 1) == gt)
            rows.append((src, a, top1))
            print(f"{src:8s} alpha={a:6.1f}  top-1 = {top1:.1f}%")

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/ridge_ablation.csv", "w") as f:
        f.write("source,alpha,top1\n")
        for s, a, t in rows:
            f.write(f"{s},{a},{t:.2f}\n")


if __name__ == "__main__":
    main()
