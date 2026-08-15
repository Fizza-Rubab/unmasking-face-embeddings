"""
Overview figure (Fig. 1): one identity template, one linear bridge, three borrowed
capabilities -- assembled from REAL pipeline outputs, same theme as all other figures.
Layout:  face -> FR encoder -> template (barcode) -> linear bridge W -> foundation space
         -> { READ: text query retrieves faces | GENERATE: decoded face | NAME: top name }
Outputs figures/fig_overview.pdf/.png.
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, INK2, GOOD, FRAME = "#0b0b0b", "#52514e", "#008300", "#c9c8c2"
BLUE, AQUA = "#2a78d6", "#1baf7a"
plt.rcParams.update({"font.size": 8, "figure.dpi": 200, "text.color": INK,
                     "font.family": "sans-serif",
                     "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})

EMB = "embeddings/cfp"
IMG = os.path.expanduser(
    "~/.cache/kagglehub/datasets/chinafax/cfpw-dataset/versions/1/cfp-dataset/Data/Images")
NAMES_TXT = os.path.expanduser(
    "~/.cache/kagglehub/datasets/chinafax/cfpw-dataset/versions/1/cfp-dataset/Data/list_name.txt")
SEED, TR, N = 42, 0.7, 60
TGT_ID, TGT_PROC = "openai/clip-vit-base-patch32", "openai/clip-vit-base-patch32"
TGT = "clip"
PROMPTS = ["a photo of {}", "a portrait photo of {}",
           "a face photo of the celebrity {}", "{}"]
QUERY = "a smiling woman with blond hair"


def l2(x):
    x = np.atleast_2d(np.asarray(x, np.float64))
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


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
    return Image.open(os.path.join(IMG, rel_path)).convert("RGB").resize((224, 224))


def frame(ax, color=FRAME, lw=0.7):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(color); sp.set_linewidth(lw)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    meta = np.load(f"{EMB}/cfp_metadata.npy", allow_pickle=True)
    ids = np.array([int(r["identity"]) for r in meta])
    rel = np.array([r["rel_path"] for r in meta])
    names = [ln.strip() for ln in open(NAMES_TXT) if ln.strip()]
    rng = np.random.RandomState(SEED)
    uniq = np.array(sorted(set(ids))); rng.shuffle(uniq)
    trset = set(uniq[:int(len(uniq) * TR)])
    tr = np.array([i for i, x in enumerate(ids) if x in trset])
    te = np.array([i for i, x in enumerate(ids) if x not in trset])
    pick = te[np.random.RandomState(SEED).choice(len(te), N, replace=False)]
    sel = np.random.RandomState(3).choice(N, 6, replace=False)
    hero_local = int(sel[0])                      # same face as t2i grid row 1
    hero = pick[hero_local]

    A = l2(np.load(f"{EMB}/arcface.npy"))
    K = l2(np.load(f"{EMB}/kprpe.npy"))
    F = l2(np.load(f"{EMB}/{TGT}.npy"))
    muA, muK, muF = A[tr].mean(0), K[tr].mean(0), F[tr].mean(0)
    Wa = np.linalg.lstsq(A[tr] - muA, F[tr] - muF, rcond=None)[0]
    Wk = np.linalg.lstsq(K[tr] - muK, F[tr] - muF, rcond=None)[0]
    bridgedA = l2((A[te] - muA) @ Wa + muF)

    # READ: top-3 retrieval for QUERY (identity-deduped)
    Tq = enc_text([QUERY])[0]
    s = bridgedA @ Tq
    top3, seen = [], set()
    for j in np.argsort(-s):
        if ids[te[j]] not in seen:
            seen.add(ids[te[j]]); top3.append(j)
        if len(top3) == 3:
            break

    # NAME: kprpe->clip prediction for the hero probe
    T = l2(np.mean([enc_text([p.format(n) for n in names]) for p in PROMPTS], axis=0))
    Tc = l2(T - T.mean(0))
    Bk = l2((K - muK) @ Wk + muF)
    probe = l2(Bk[hero][None] - Bk[tr].mean(0))
    pred_name = names[int(np.argmax(probe @ Tc.T))]

    gen_fp = f"eval_out/t2i/kandinsky/bridged-kprpe/{hero_local:03d}.jpg"
    gen_im = Image.open(gen_fp).convert("RGB").resize((224, 224))

    # ---------------------------------------------------------------- layout
    fig = plt.figure(figsize=(7.2, 2.7))

    def axat(x, y, w, h):
        return fig.add_axes([x, y, w, h])

    # 1) face photo
    ax = axat(0.015, 0.30, 0.155, 0.46)
    ax.imshow(cfp_img(rel[hero])); frame(ax)
    ax.set_title("face image", fontsize=7.5, color=INK2, pad=2)

    # 2) template barcode
    axb = axat(0.225, 0.30, 0.032, 0.46)
    axb.imshow(A[hero][:, None].repeat(6, 1), aspect="auto", cmap="Greys")
    frame(axb)
    axb.set_title("template", fontsize=7.5, color=INK2, pad=2)
    axb.text(0.5, -0.13, "512-d", transform=axb.transAxes, ha="center",
             fontsize=6.5, color=INK2)

    # 3) bridge box
    bx, by, bw, bh = 0.315, 0.40, 0.115, 0.26
    fig.patches.append(FancyBboxPatch((bx, by), bw, bh,
                       boxstyle="round,pad=0.012", transform=fig.transFigure,
                       facecolor="#eaf1fb", edgecolor=BLUE, linewidth=1.0))
    fig.text(bx + bw/2, by + bh/2, "linear transformation\n$\\tilde c=(a-\\mu_a)W+\\mu_c$",
             ha="center", va="center", fontsize=7.5, color=INK)
    fig.text(bx + bw/2, by + bh + 0.10, "fit once, no training", ha="center",
             fontsize=6.5, color=INK2, style="italic")

    # arrows: photo -> template -> bridge -> outputs
    def arrow(x0, y0, x1, y1, color=INK2):
        fig.patches.append(FancyArrowPatch((x0, y0), (x1, y1),
                           transform=fig.transFigure, arrowstyle="-|>",
                           mutation_scale=9, linewidth=1.0, color=color))
    arrow(0.173, 0.53, 0.222, 0.53)
    fig.text(0.198, 0.57, "FR\nencoder", ha="center", fontsize=6.5, color=INK2)
    arrow(0.260, 0.53, 0.312, 0.53)
    fx = bx + bw + 0.015
    arrow(fx - 0.013, 0.53, fx + 0.032, 0.80, color=BLUE)   # to READ
    arrow(fx - 0.013, 0.53, fx + 0.032, 0.53, color=BLUE)   # to GENERATE
    arrow(fx - 0.013, 0.53, fx + 0.032, 0.24, color=BLUE)   # to NAME

    X0 = fx + 0.040
    # READ row
    fig.text(X0, 0.90, "READ", fontsize=8, color=BLUE, weight="bold")
    fig.text(X0 + 0.058, 0.90, f"“{QUERY}”", fontsize=6.5, color=INK, va="center")
    for k, j in enumerate(top3):
        axr = axat(X0 + 0.285 + k * 0.075, 0.775, 0.068, 0.185)
        axr.imshow(cfp_img(rel[te[j]])); frame(axr)
    fig.text(X0 + 0.272, 0.865, "→", fontsize=8, color=INK2)
    # GENERATE row
    fig.text(X0, 0.545, "GENERATE", fontsize=8, color=BLUE, weight="bold")
    axg = axat(X0 + 0.115, 0.40, 0.105, 0.28)
    axg.imshow(gen_im); frame(axg)
    fig.text(X0 + 0.235, 0.53, "decoded from the\ntemplate alone", fontsize=6.5,
             color=INK2, va="center")
    # NAME row
    fig.text(X0, 0.185, "NAME", fontsize=8, color=BLUE, weight="bold")
    fig.text(X0 + 0.085, 0.185, f"“{pred_name}”", fontsize=8.5, color=GOOD,
             weight="bold")
    fig.text(X0 + 0.085, 0.075, "matched against a 500-name text vocabulary",
             fontsize=6.5, color=INK2)

    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"figures/fig_overview.{ext}", bbox_inches="tight", dpi=250)
    print("saved figures/fig_overview.pdf   (name pred:", pred_name + ")")


if __name__ == "__main__":
    main()
