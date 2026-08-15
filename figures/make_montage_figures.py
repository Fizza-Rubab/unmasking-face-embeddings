"""
Rebuild all qualitative montage figures in one shared visual theme, as PDF (+PNG preview):
  figures/fig_freeform_retrieval.pdf  -- user-final 8 prompts x top-5 bridged retrievals
  figures/fig_naming.pdf              -- named probes (pred vs true, green/red)
  figures/fig_t2i_grid.pdf            -- original | native | bridged | unaligned per decoder
Theme matches make_paper_figures.py: DejaVu Sans, ink #0b0b0b / #52514e, correct #008300,
miss #e34948, thin #c9c8c2 frames, white surface.
"""
import os
CACHE = ".cache"
os.environ.setdefault("HF_HOME", os.path.join(CACHE, "huggingface"))
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GOOD, MISS, FRAME = "#0b0b0b", "#52514e", "#008300", "#e34948", "#c9c8c2"
plt.rcParams.update({"font.size": 8, "figure.dpi": 200, "text.color": INK,
                     "font.family": "sans-serif",
                     "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})

EMB = "embeddings/cfp"
IMG = os.path.expanduser(
    "~/.cache/kagglehub/datasets/chinafax/cfpw-dataset/versions/1/cfp-dataset/Data/Images")
NAMES_TXT = os.path.expanduser(
    "~/.cache/kagglehub/datasets/chinafax/cfpw-dataset/versions/1/cfp-dataset/Data/list_name.txt")
SEED, TR = 42, 0.7
TGT_ID, TGT_PROC = "openai/clip-vit-base-patch32", "openai/clip-vit-base-patch32"
TGT = "clip"
QUERIES = [
    "a person with dimples",
    "a woman wearing heavy makeup",
    "a bald smiling man",
    "a man with bushy eyebrows",
    "a woman with short bangs",
    "sharp cheekbones",
]
PROMPTS = ["a photo of {}", "a portrait photo of {}",
           "a face photo of the celebrity {}", "{}"]


def l2(x):
    x = np.atleast_2d(np.asarray(x, np.float64))
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def split_ids():
    meta = np.load(f"{EMB}/cfp_metadata.npy", allow_pickle=True)
    ids = np.array([int(r["identity"]) for r in meta])
    rel = np.array([r["rel_path"] for r in meta])
    rng = np.random.RandomState(SEED)
    uniq = np.array(sorted(set(ids))); rng.shuffle(uniq)
    trset = set(uniq[:int(len(uniq) * TR)])
    tr = np.array([i for i, x in enumerate(ids) if x in trset])
    te = np.array([i for i, x in enumerate(ids) if x not in trset])
    return ids, rel, tr, te


def bridge_target(tr, src="arcface"):
    A = l2(np.load(f"{EMB}/{src}.npy"))
    F = l2(np.load(f"{EMB}/{TGT}.npy"))
    muA, muF = A[tr].mean(0), F[tr].mean(0)
    W = np.linalg.lstsq(A[tr] - muA, F[tr] - muF, rcond=None)[0]
    return A, F, W, muA, muF


def enc_text(prompts):
    from transformers import CLIPModel, CLIPTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = CLIPModel.from_pretrained(TGT_ID).to(dev).eval()
    tk = CLIPTokenizer.from_pretrained(TGT_PROC)
    out = []
    with torch.no_grad():
        for i in range(0, len(prompts), 256):
            inp = tk(prompts[i:i+256], return_tensors="pt", padding=True,
                     truncation=True).to(dev)
            out.append(m.get_text_features(**inp).cpu().numpy())
    del m; torch.cuda.empty_cache()
    return l2(np.concatenate(out))


def cfp_img(rel_path):
    try:
        return Image.open(os.path.join(IMG, rel_path)).convert("RGB").resize((224, 224))
    except Exception:
        return Image.new("RGB", (224, 224), (235, 235, 235))


def frame(ax):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(FRAME); sp.set_linewidth(0.6)
    ax.set_xticks([]); ax.set_yticks([])


def save(fig, name):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"figures/{name}.{ext}", bbox_inches="tight", dpi=250)
    plt.close(fig)
    print(f"saved figures/{name}.pdf")


# 1. freeform
def fig_freeform(ids, rel, tr, te, src="arcface", name="fig_freeform_retrieval"):
    A, F, W, muA, muF = bridge_target(tr, src=src)
    bridged = l2((A[te] - muA) @ W + muF)
    T = enc_text(QUERIES)
    K = 5
    # geometry: label margin 34% of width; make cell height == cell width so rows are tight
    FW = 5.6
    cell = FW * 0.64 / K * (1 / 1.04)
    fig, axes = plt.subplots(len(QUERIES), K,
                             figsize=(FW, len(QUERIES) * cell * 1.06))
    for qi, q in enumerate(QUERIES):
        s = bridged @ T[qi]
        top, seen = [], set()
        for j in np.argsort(-s):
            if ids[te[j]] not in seen:
                seen.add(ids[te[j]]); top.append(j)
            if len(top) == K:
                break
        for c, j in enumerate(top):
            ax = axes[qi, c]
            ax.imshow(cfp_img(rel[te[j]])); frame(ax)
        axes[qi, 0].set_ylabel(f"“{q}”", rotation=0, ha="right", va="center",
                               fontsize=7.5, color=INK)
    fig.subplots_adjust(wspace=0.04, hspace=0.06, left=0.36, right=0.995,
                        top=0.995, bottom=0.005)
    save(fig, name)


# 2. naming
def fig_naming(ids, rel, tr, te):
    names = [ln.strip() for ln in open(NAMES_TXT) if ln.strip()]
    A, F, W, muA, muF = bridge_target(tr, src="kprpe")   # best bridged source
    Bte = l2((A[te] - muA) @ W + muF)
    Btr = l2((A[tr] - muA) @ W + muF)
    T = l2(np.mean([enc_text([p.format(n) for n in names]) for p in PROMPTS], axis=0))
    Tc = l2(T - T.mean(0))
    S = l2(Bte - Btr.mean(0)) @ Tc.T
    gt = ids[te] - 1
    pred = np.argsort(-S, axis=1)[:, 0]
    hit = pred == gt
    r = np.random.RandomState(1)
    pick = np.r_[r.choice(np.where(hit)[0], 6, replace=False),
                 r.choice(np.where(~hit)[0], 2, replace=False)]
    fig, axes = plt.subplots(2, 4, figsize=(4 * 1.35, 2 * 1.62))
    for k, j in enumerate(pick):
        ax = axes[k // 4, k % 4]
        ax.imshow(cfp_img(rel[te[j]])); frame(ax)
        ok = hit[j]
        ax.set_title(names[pred[j]], fontsize=7.5, pad=3,
                     color=GOOD if ok else MISS)
        ax.set_xlabel("" if ok else f"true: {names[gt[j]]}", fontsize=6.5, color=INK2,
                      labelpad=2)
    fig.subplots_adjust(wspace=0.06, hspace=0.28, left=0.01, right=0.99,
                        top=0.92, bottom=0.08)
    save(fig, "fig_naming")


# 3. t2i grid
def fig_t2i():
    GEN = "eval_out/t2i"
    N = 60
    ids, rel, tr, te = split_ids()
    pick = te[np.random.RandomState(SEED).choice(len(te), N, replace=False)]
    sel = np.random.RandomState(3).choice(N, 6, replace=False)
    sel = np.delete(sel, 1)                          # drop 2nd identity -> 5 rows
    show = [("original", None, None),
            (None, None, None),                     # spacer after original
            ("native", "kandinsky", "native"),
            ("aligned", "kandinsky", "bridged-kprpe"),
            ("unaligned", "kandinsky", "unaligned"),
            (None, None, None),                     # spacer between decoders
            ("native", "unclip", "native"),
            ("aligned", "unclip", "bridged-kprpe"),
            ("unaligned", "unclip", "unaligned")]
    widths = [0.18 if s == (None, None, None) else 1 for s in show]
    fig, axes = plt.subplots(len(sel), len(show),
                             figsize=(sum(widths) * 1.0, len(sel) * 1.06),
                             gridspec_kw={"width_ratios": widths})
    for ri, i in enumerate(sel):
        for ci, (lbl, dec, var) in enumerate(show):
            ax = axes[ri, ci]
            if (lbl, dec, var) == (None, None, None):
                ax.axis("off"); continue
            if dec is None:
                im = cfp_img(rel[pick[i]])
            else:
                fp = os.path.join(GEN, dec, var, f"{i:03d}.jpg")
                im = Image.open(fp).convert("RGB").resize((224, 224)) \
                    if os.path.exists(fp) else Image.new("RGB", (224, 224), (235,)*3)
            ax.imshow(im); frame(ax)
            if ri == 0:
                ax.set_title(lbl, fontsize=7.5, pad=3, color=INK)
    fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0.005, right=0.995,
                        top=0.93, bottom=0.005)
    # decoder group labels just above the column titles
    for cols, lbl in (((2, 4), "Kandinsky 2.2"), ((6, 8), "Stable unCLIP")):
        p0, p1 = axes[0, cols[0]].get_position(), axes[0, cols[1]].get_position()
        fig.text((p0.x0 + p1.x1) / 2, p0.y1 + 0.045, lbl, ha="center",
                 fontsize=8.5, color=INK2)
    save(fig, "fig_t2i_grid")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    ids, rel, tr, te = split_ids()
    fig_freeform(ids, rel, tr, te)
    fig_naming(ids, rel, tr, te)
    fig_t2i()
