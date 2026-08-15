"""
Text-to-face retrieval from FR templates, full grid.

Datasets : UTKFace (gender/age/race labels) + CelebA (40 binary attrs, incl. composed sentences)
Sources  : arcface, adaface, adaface_vit, kprpe (FR templates)
Targets  : clip, metaclip, siglip (foundation text/image spaces)
Rankers  : bridged   = plain least-squares bridge FR->foundation (ours)
           native    = real foundation image embedding (ceiling)
           unaligned = raw FR embedding scored against text (zero-padded if dims differ)
           random    = untrained random map (floor)
Metric   : Average Precision per query; mean AP per (dataset, target, source).

Outputs  : eval_out/text_retrieval_full.csv        (per-query APs)
           eval_out/text_retrieval_summary.csv     (mAP grid)
           eval_out/tab_text_retrieval.tex         (LaTeX table)

"""
import os
CACHE = ".cache"
os.environ.setdefault("HF_HOME", os.path.join(CACHE, "huggingface"))
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from sklearn.metrics import average_precision_score

SEED, TR_RATIO = 42, 0.7
FACE = ["arcface", "adaface", "adaface_vit", "kprpe"]
FOUND = {
    "clip":     ("openai/clip-vit-base-patch32", "clip"),
    "metaclip": ("facebook/metaclip-b32-400m", "clip"),
    "siglip":   ("google/siglip-base-patch16-224", "siglip"),
}
OUT = "eval_out"


def l2(x):
    x = np.atleast_2d(np.asarray(x, np.float64))
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def encode_text(ftag, prompts):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mid, kind = FOUND[ftag]
    out = []
    if kind == "siglip":
        from transformers import SiglipModel, SiglipProcessor
        m = SiglipModel.from_pretrained(mid).to(dev).eval()
        pr = SiglipProcessor.from_pretrained(mid)
        with torch.no_grad():
            for i in range(0, len(prompts), 64):
                inp = pr(text=prompts[i:i+64], return_tensors="pt",
                         padding="max_length", truncation=True).to(dev)
                out.append(m.get_text_features(**inp).cpu().numpy())
    else:
        from transformers import CLIPModel, CLIPTokenizer
        proc_id = mid
        if "metaclip" in mid:
            proc_id = "openai/clip-vit-base-patch32"
        m = CLIPModel.from_pretrained(mid).to(dev).eval()
        tk = CLIPTokenizer.from_pretrained(proc_id)
        with torch.no_grad():
            for i in range(0, len(prompts), 64):
                inp = tk(prompts[i:i+64], return_tensors="pt", padding=True,
                         truncation=True).to(dev)
                out.append(m.get_text_features(**inp).cpu().numpy())
    del m
    if dev == "cuda":
        torch.cuda.empty_cache()
    return l2(np.concatenate(out))


# ---------------------------------------------------------------- datasets
def utk_queries():
    meta = np.load("embeddings/utk/utk_metadata.npy")
    age, gender, race = meta["age"], meta["gender"], meta["race"]
    n = len(age)
    Q = [
        ("a photo of a man",            gender == 0),
        ("a photo of a woman",          gender == 1),
        ("a photo of an old person",    age >= 50),
        ("a photo of a young person",   age <= 25),
        ("a photo of a baby",           age <= 3),
        ("a photo of a White person",   race == 0),
        ("a photo of a Black person",   race == 1),
        ("a photo of an Asian person",  race == 2),
        ("a photo of an Indian person", race == 3),
    ]
    return "embeddings/utk", n, Q, None


def celeba_queries():
    meta = np.load("embeddings/celeba/celeba_metadata.npz", allow_pickle=True)
    attrs, names = meta["attrs"].astype(bool), list(meta["attr_names"])
    # identity labels (official identity_CelebA.txt) -> identity-disjoint split
    id_map = {}
    with open("embeddings/celeba/identity_CelebA.txt") as f:
        for line in f:
            img, ident = line.split()
            id_map[img] = int(ident)
    idents = np.array([id_map[i] for i in meta["image_id"]])
    a = {nm: attrs[:, i] for i, nm in enumerate(names)}
    n = attrs.shape[0]
    Q = [
        # single attributes (visually grounded subset)
        ("a photo of a man",                        a["Male"]),
        ("a photo of a woman",                      ~a["Male"]),
        ("a photo of a smiling person",             a["Smiling"]),
        ("a person wearing eyeglasses",             a["Eyeglasses"]),
        ("a person with blond hair",                a["Blond_Hair"]),
        ("a person with black hair",                a["Black_Hair"]),
        ("a person with gray hair",                 a["Gray_Hair"]),
        ("a bald person",                           a["Bald"]),
        ("a person with bangs",                     a["Bangs"]),
        ("a person wearing a hat",                  a["Wearing_Hat"]),
        ("a man with a mustache",                   a["Mustache"]),
        ("a man with a beard",                      ~a["No_Beard"]),
        ("a person wearing lipstick",               a["Wearing_Lipstick"]),
        ("a young person",                          a["Young"]),
        ("a person with wavy hair",                 a["Wavy_Hair"]),
        ("a chubby person",                         a["Chubby"]),
        # composed sentences (AND of attributes)
        ("a smiling young woman with blond hair",
         a["Smiling"] & a["Young"] & ~a["Male"] & a["Blond_Hair"]),
        ("an old bald man with eyeglasses",
         ~a["Young"] & a["Bald"] & a["Male"] & a["Eyeglasses"]),
        ("a young man with black hair and a beard",
         a["Young"] & a["Male"] & a["Black_Hair"] & ~a["No_Beard"]),
        ("a smiling woman wearing a hat",
         a["Smiling"] & ~a["Male"] & a["Wearing_Hat"]),
        ("a woman with wavy hair wearing lipstick",
         ~a["Male"] & a["Wavy_Hair"] & a["Wearing_Lipstick"]),
        ("a serious man with a mustache and eyeglasses",
         ~a["Smiling"] & a["Male"] & a["Mustache"] & a["Eyeglasses"]),
    ]
    return "embeddings/celeba", n, Q, idents


DATASETS = {"utk": utk_queries, "celeba": celeba_queries}


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.RandomState(SEED)
    rows = []          # dataset,target,source,ranker,query,ap,prevalence
    for ds, fn in DATASETS.items():
        emb_dir, n, Q, idents = fn()
        if idents is not None:
            # identity-disjoint split: no identity appears in both train and test
            uids = np.unique(idents)
            uids = uids[np.random.RandomState(SEED).permutation(len(uids))]
            ku = int(len(uids) * TR_RATIO)
            tr_ids = set(uids[:ku].tolist())
            mask = np.array([i in tr_ids for i in idents])
            tr, te = np.where(mask)[0], np.where(~mask)[0]
            print(f"\n########## {ds}: identity-disjoint split "
                  f"({ku}/{len(uids)-ku} identities -> {len(tr)}/{len(te)} images)")
        else:
            idx = np.random.RandomState(SEED).permutation(n)
            k = int(n * TR_RATIO)
            tr, te = idx[:k], idx[k:]
        print(f"########## {ds}: gallery = {len(te)} test faces, {len(Q)} queries")
        for ftag in FOUND:
            C = np.load(f"{emb_dir}/{ftag}.npy").astype(np.float64)
            T = encode_text(ftag, [q for q, _ in Q])
            muC = C[tr].mean(0)
            native = l2(C[te])
            variants = {"native": native}
            for src in FACE:
                A = l2(np.load(f"{emb_dir}/{src}.npy").astype(np.float64))
                muA = A[tr].mean(0)
                d = A.shape[1]
                W = np.linalg.lstsq(A[tr]-muA, C[tr]-muC, rcond=None)[0]
                variants[f"bridged-{src}"] = l2((A[te]-muA) @ W + muC)
                # unaligned: raw FR embedding vs text (zero-pad if dims differ)
                U = A[te]
                if U.shape[1] < C.shape[1]:
                    U = np.pad(U, ((0, 0), (0, C.shape[1]-U.shape[1])))
                elif U.shape[1] > C.shape[1]:
                    U = U[:, :C.shape[1]]
                variants[f"unaligned-{src}"] = l2(U)
                Wr = rng.normal(0, 1/np.sqrt(d), (d, C.shape[1]))
                variants[f"random-{src}"] = l2((A[te]-muA) @ Wr + muC)
            for lbl, E in variants.items():
                for qi, (q, rel_all) in enumerate(Q):
                    rel = rel_all[te]
                    if rel.sum() < 5 or rel.mean() > 0.98:
                        continue
                    ap = average_precision_score(rel, E @ T[qi])
                    rows.append((ds, ftag, lbl, q, ap, rel.mean()))
            m = {lbl: np.mean([r[4] for r in rows
                               if r[0] == ds and r[1] == ftag and r[2] == lbl])
                 for lbl in variants}
            print(f"  {ftag}: " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))

    with open(f"{OUT}/text_retrieval_full.csv", "w") as f:
        f.write("dataset,target,ranker,query,ap,prevalence\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},\"{r[3]}\",{r[4]:.4f},{r[5]:.4f}\n")

    # summary grid: rows = (dataset, source, ranker), cols = targets
    import collections
    agg = collections.defaultdict(list)
    for ds, ftag, lbl, q, ap, p in rows:
        agg[(ds, ftag, lbl)].append(ap)
    mAP = {k: np.mean(v) for k, v in agg.items()}
    with open(f"{OUT}/text_retrieval_summary.csv", "w") as f:
        f.write("dataset,target,ranker,mAP\n")
        for k in sorted(mAP):
            f.write(f"{k[0]},{k[1]},{k[2]},{mAP[k]:.4f}\n")

    # LaTeX: one block per dataset; rows = ranker, cols = clip/metaclip/siglip
    order = (["native"] + [f"{r}-{s}" for s in FACE
             for r in ("bridged", "unaligned", "random")])
    pretty = {"native": r"native (ceiling)"}
    for s in FACE:
        pretty[f"bridged-{s}"] = rf"\textbf{{bridged}} {s}"
        pretty[f"unaligned-{s}"] = f"unaligned {s}"
        pretty[f"random-{s}"] = f"random map {s}"
    tnames = {"clip": "CLIP", "metaclip": "MetaCLIP", "siglip": "SigLIP"}
    with open(f"{OUT}/tab_text_retrieval.tex", "w") as f:
        f.write("% mean AP, text->face retrieval. auto-generated by eval_text_retrieval.py\n")
        f.write("\\begin{tabular}{ll" + "c"*len(FOUND) + "}\n\\toprule\n")
        f.write("Dataset & Ranker & " + " & ".join(tnames[t] for t in FOUND)
                + " \\\\\n\\midrule\n")
        for ds in DATASETS:
            for i, lbl in enumerate(order):
                cells = []
                for ftag in FOUND:
                    v = mAP.get((ds, ftag, lbl))
                    cells.append(f"{v:.3f}" if v is not None else "--")
                dsn = ds.upper() if i == 0 else ""
                f.write(f"{dsn} & {pretty[lbl]} & " + " & ".join(cells) + " \\\\\n")
            f.write("\\midrule\n" if ds != list(DATASETS)[-1] else "")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"\nsaved {OUT}/text_retrieval_full.csv, _summary.csv, tab_text_retrieval.tex")


if __name__ == "__main__":
    main()
