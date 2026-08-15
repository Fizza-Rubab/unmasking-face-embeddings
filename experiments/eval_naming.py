"""
Zero-shot naming and open-set identification from FR templates.

  - targets: clip, metaclip, siglip
  - sources: bridged FR (least squares), native (ceiling),
    unaligned (raw zero-padded FR), random rotation (floor)
  - closed-set top-1/5/10 over the 500-name CFP vocabulary
  - open-set watchlist: DIR@FAR 1%/10%, rejection AUROC
  - vocabulary-size ablation: real-celebrity distractors (FaceScrub, CFP overlaps
    removed) then synthetic names; top-k vs vocab size

Outputs: eval_out/naming_results.csv, eval_out/naming_ablation.csv, eval_out/tab_naming.tex
"""
import os
CACHE = ".cache"
os.environ.setdefault("HF_HOME", os.path.join(CACHE, "huggingface"))
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

EMB = "embeddings/cfp"
NAMES_TXT = os.path.expanduser(
    "~/.cache/kagglehub/datasets/chinafax/cfpw-dataset/versions/1/cfp-dataset/Data/list_name.txt")
SEED, TR = 42, 0.7
FACE = ["arcface", "adaface", "adaface_vit", "kprpe"]
FOUND = {
    "clip":     ("openai/clip-vit-base-patch32", "clip"),
    "metaclip": ("facebook/metaclip-b32-400m", "clip"),
    "siglip":   ("google/siglip-base-patch16-224", "siglip"),
}
PROMPTS = ["a photo of {}", "a portrait photo of {}",
           "a face photo of the celebrity {}", "{}"]
VOCAB_SIZES = [500, 1000, 2000, 4000]
OUT = "eval_out"


def l2(x):
    x = np.atleast_2d(np.asarray(x, np.float64))
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def enc_text(ftag, prompts):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mid, kind = FOUND[ftag]
    out = []
    if kind == "siglip":
        from transformers import SiglipModel, SiglipProcessor
        m = SiglipModel.from_pretrained(mid).to(dev).eval()
        pr = SiglipProcessor.from_pretrained(mid)
        with torch.no_grad():
            for i in range(0, len(prompts), 256):
                inp = pr(text=prompts[i:i+256], return_tensors="pt",
                         padding="max_length", truncation=True).to(dev)
                out.append(m.get_text_features(**inp).cpu().numpy())
    else:
        from transformers import CLIPModel, CLIPTokenizer
        proc = mid
        if "metaclip" in mid:
            proc = "openai/clip-vit-base-patch32"
        m = CLIPModel.from_pretrained(mid).to(dev).eval()
        tk = CLIPTokenizer.from_pretrained(proc)
        with torch.no_grad():
            for i in range(0, len(prompts), 256):
                inp = tk(prompts[i:i+256], return_tensors="pt", padding=True,
                         truncation=True).to(dev)
                out.append(m.get_text_features(**inp).cpu().numpy())
    del m
    if dev == "cuda":
        torch.cuda.empty_cache()
    return l2(np.concatenate(out))


def name_anchors(ftag, names):
    return l2(np.mean([enc_text(ftag, [p.format(n) for n in names])
                       for p in PROMPTS], axis=0))


def dir_at_far(s_in, ok_in, s_out, far):
    thr = np.quantile(s_out, 1.0 - far)
    return float(np.mean((s_in >= thr) & ok_in))


def auroc(pos, neg):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y == 1].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))


def build_distractors(cfp_names, total_needed, seed=0):
    """Real FaceScrub celebrities first (CFP overlap removed), then synthetic names."""
    low = {n.lower() for n in cfp_names}
    real = [n for n in (ln.strip() for ln in open(f"{OUT}/distractor_names.txt"))
            if n and n.lower() not in low]
    first = [ln.strip().title() for ln in open(f"{OUT}/first_names.txt") if ln.strip()]
    last = [ln.strip().title() for ln in open(f"{OUT}/last_names.txt") if ln.strip()]
    rng = np.random.RandomState(seed)
    used = low | {n.lower() for n in real}
    synth = []
    while len(synth) < max(0, total_needed - len(real)):
        nm = f"{rng.choice(first)} {rng.choice(last)}"
        if nm.lower() not in used:
            used.add(nm.lower()); synth.append(nm)
    return (real + synth)[:total_needed]


def topk_stats(S, gt, ks=(1, 5, 10)):
    rank = np.argsort(-S, axis=1)
    out = []
    for k in ks:
        out.append(100.0 * np.mean([gt[r] in rank[r, :k] for r in range(len(gt))]))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    names = [ln.strip() for ln in open(NAMES_TXT) if ln.strip()]
    n_names = len(names)
    distract = build_distractors(names, max(VOCAB_SIZES) - n_names)
    meta = np.load(f"{EMB}/cfp_metadata.npy", allow_pickle=True)
    ids = np.array([int(r["identity"]) for r in meta])

    rng = np.random.RandomState(SEED)
    uniq = np.array(sorted(set(ids))); rng.shuffle(uniq)
    trset = set(uniq[:int(len(uniq) * TR)])
    tr = np.array([i for i, x in enumerate(ids) if x in trset])
    te = np.array([i for i, x in enumerate(ids) if x not in trset])
    gt = ids[te] - 1
    te_ids = np.array(sorted(set(ids[te])))
    print(f"{n_names} names +{len(distract)} distractors | {len(tr)}/{len(te)} tr/te imgs")

    rows, abl = [], []
    for ftag in FOUND:
        F = l2(np.load(f"{EMB}/{ftag}.npy"))
        Tfull = name_anchors(ftag, names + distract)   # 500 CFP first, then distractors

        variants = {"native": (F[te], F[tr])}
        for src in FACE:
            A = l2(np.load(f"{EMB}/{src}.npy"))
            muA, muF = A[tr].mean(0), F[tr].mean(0)
            W = np.linalg.lstsq(A[tr] - muA, F[tr] - muF, rcond=None)[0]
            variants[f"bridged-{src}"] = (l2((A[te]-muA) @ W + muF),
                                          l2((A[tr]-muA) @ W + muF))
        A0 = l2(np.load(f"{EMB}/{FACE[0]}.npy"))
        U = A0 if A0.shape[1] >= F.shape[1] else \
            np.pad(A0, ((0, 0), (0, F.shape[1]-A0.shape[1])))
        U = U[:, :F.shape[1]]
        variants["unaligned"] = (l2(U[te]), l2(U[tr]))
        G = rng.randn(F.shape[1], A0.shape[1]); Q, _ = np.linalg.qr(G)
        variants["random"] = (l2(A0[te] @ Q.T), l2(A0[tr] @ Q.T))

        print(f"\n=== target {ftag} ===  (chance top-1 = {100/n_names:.2f}%)")
        for lbl, (E_te, E_tr) in variants.items():
            Ec = l2(E_te - E_tr.mean(0))
            # ---- closed set on the 500-name vocab
            Tc = l2(Tfull[:n_names] - Tfull[:n_names].mean(0))
            S = Ec @ Tc.T
            t1, t5, t10 = topk_stats(S, gt)

            # ---- open-set watchlist (half the test ids)
            r2 = np.random.RandomState(SEED)
            wl_ids = set(r2.permutation(te_ids)[:len(te_ids)//2])
            vocab = np.array(sorted(i - 1 for i in wl_ids))
            Sw = S[:, vocab]
            in_list = np.array([g in set(vocab) for g in gt])
            smax, amax = Sw.max(1), Sw.argmax(1)
            ok = vocab[amax] == gt
            s_in, ok_in, s_out = smax[in_list], ok[in_list], smax[~in_list]
            d1 = dir_at_far(s_in, ok_in, s_out, .01)
            d10 = dir_at_far(s_in, ok_in, s_out, .10)
            au = auroc(s_in, s_out)
            print(f"  {lbl:16s} top1 {t1:5.1f} top5 {t5:5.1f} top10 {t10:5.1f} | "
                  f"DIR@1% {100*d1:5.1f} DIR@10% {100*d10:5.1f} AUROC {au:.3f}")
            rows.append((ftag, lbl, t1, t5, t10, 100*d1, 100*d10, au))

            # ---- vocab-size ablation
            for V in VOCAB_SIZES:
                Tv = l2(Tfull[:V] - Tfull[:V].mean(0))
                ts = topk_stats(Ec @ Tv.T, gt)
                abl.append((ftag, lbl, V, *ts))

    with open(f"{OUT}/naming_results.csv", "w") as f:
        f.write("foundation,source,top1,top5,top10,dir_far1,dir_far10,auroc\n")
        for r in rows:
            f.write(",".join(f"{x:.3f}" if isinstance(x, float) else str(x)
                             for x in r) + "\n")
    with open(f"{OUT}/naming_ablation.csv", "w") as f:
        f.write("foundation,source,vocab_size,top1,top5,top10\n")
        for r in abl:
            f.write(",".join(f"{x:.3f}" if isinstance(x, float) else str(x)
                             for x in r) + "\n")

    # LaTeX table
    pretty = {"native": "native (ceiling)", "bridged-arcface": r"\textbf{bridged} ArcFace",
              "bridged-adaface": r"\textbf{bridged} AdaFace",
              "bridged-kprpe": r"\textbf{bridged} KPRPE",
              "unaligned": "unaligned", "random": "random rot."}
    order = ["native", "bridged-arcface", "bridged-adaface", "bridged-kprpe",
             "unaligned", "random"]
    with open(f"{OUT}/tab_naming.tex", "w") as f:
        f.write("% naming, auto-generated by eval_naming.py. chance top1=0.2%\n")
        f.write("\\begin{tabular}{llcccccc}\n\\toprule\n")
        f.write("Target & Source & Top-1 & Top-5 & Top-10 & DIR@1\\% & DIR@10\\% "
                "& AUROC \\\\\n\\midrule\n")
        for ftag in FOUND:
            for i, lbl in enumerate(order):
                r = next(x for x in rows if x[0] == ftag and x[1] == lbl)
                fn = ftag.upper() if i == 0 else ""
                f.write(f"{fn} & {pretty[lbl]} & " +
                        " & ".join(f"{v:.1f}" for v in r[2:7]) +
                        f" & {r[7]:.3f} \\\\\n")
            if ftag != list(FOUND)[-1]:
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"\nsaved {OUT}/naming_results.csv, naming_ablation.csv, tab_naming.tex")


if __name__ == "__main__":
    main()
