"""
Freeform open-vocabulary text->face retrieval montages (qualitative, CFP gallery).

Tries a pool of freeform queries, retrieves top-5 with the BRIDGED ranker
(KPRPE -> CLIP), and auto-scores each query by how well the retrieved faces
match the query in the native CLIP image space (independent judge). Saves one montage
per query plus a combined figure of the best-scoring queries.

Outputs: eval_out/freeform/<query>.jpg, figures/fig_freeform_retrieval.jpg,
         eval_out/freeform_scores.csv
"""
import os
CACHE = ".cache"
os.environ.setdefault("HF_HOME", os.path.join(CACHE, "huggingface"))
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

EMB = "embeddings/cfp"
IMG = os.path.expanduser(
    "~/.cache/kagglehub/datasets/chinafax/cfpw-dataset/versions/1/cfp-dataset/Data/Images")
SEED, TR = 42, 0.5
SRC, FTAG = "kprpe", "clip"
CLIP_ID = "openai/clip-vit-base-patch32"
K, N_BEST = 5, 8
OUT = "eval_out/freeform"

QUERIES = [
    "a person with dimples",
    "a woman wearing heavy makeup",
    "a bald smiling man",
    "a man with bushy eyebrows",
    "a woman with short bangs",
    "sharp cheekbones",
    "a woman with red hair",
    "a person looking grim"
]


def l2(x):
    x = np.atleast_2d(np.asarray(x, np.float64))
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def enc_text(mid, proc_id, prompts, dev):
    from transformers import CLIPModel, CLIPTokenizer
    m = CLIPModel.from_pretrained(mid).to(dev).eval()
    tk = CLIPTokenizer.from_pretrained(proc_id)
    with torch.no_grad():
        inp = tk(prompts, return_tensors="pt", padding=True, truncation=True).to(dev)
        t = m.get_text_features(**inp).cpu().numpy()
    del m; torch.cuda.empty_cache()
    return l2(t)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = np.load(f"{EMB}/cfp_metadata.npy", allow_pickle=True)
    ids = np.array([int(r["identity"]) for r in meta])
    rel = np.array([r["rel_path"] for r in meta])
    rng = np.random.RandomState(SEED)
    uniq = np.array(sorted(set(ids))); rng.shuffle(uniq)
    trset = set(uniq[:int(len(uniq) * TR)])
    tr = np.array([i for i, x in enumerate(ids) if x in trset])
    te = np.array([i for i, x in enumerate(ids) if x not in trset])

    A = l2(np.load(f"{EMB}/{SRC}.npy"))
    F = l2(np.load(f"{EMB}/{FTAG}.npy"))
    Cn = l2(np.load(f"{EMB}/clip.npy"))          # independent judge space
    muA, muF = A[tr].mean(0), F[tr].mean(0)
    W = np.linalg.lstsq(A[tr] - muA, F[tr] - muF, rcond=None)[0]
    bridged = l2((A[te] - muA) @ W + muF)

    Tf = enc_text(CLIP_ID, CLIP_ID, QUERIES, dev)       # ranker text (clip space)
    Tj = enc_text(CLIP_ID, CLIP_ID, QUERIES, dev)       # judge text (clip space)

    os.makedirs(OUT, exist_ok=True)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    TH = 180
    scores, tops = [], []
    for qi, q in enumerate(QUERIES):
        s = bridged @ Tf[qi]
        top, seen = [], set()
        for j in np.argsort(-s):                       # dedupe: one image per identity
            if ids[te[j]] not in seen:
                seen.add(ids[te[j]]); top.append(j)
            if len(top) == K:
                break
        top = np.array(top)
        judge = float((Cn[te][top] @ Tj[qi]).mean())    # native-clip agreement
        scores.append(judge); tops.append(top)
        strip = Image.new("RGB", (K * TH, TH + 26), "white")
        d = ImageDraw.Draw(strip)
        d.text((4, 4), f"“{q}”   (judge {judge:.3f})", fill=(0, 0, 0), font=font)
        for c, j in enumerate(top):
            try:
                im = Image.open(os.path.join(IMG, rel[te[j]])).convert("RGB").resize((TH, TH))
            except Exception:
                im = Image.new("RGB", (TH, TH), (40, 40, 40))
            strip.paste(im, (c * TH, 26))
        strip.save(os.path.join(OUT, q.replace(" ", "_")[:60] + ".jpg"), quality=92)

    with open("eval_out/freeform_scores.csv", "w") as f:
        f.write("query,judge_score\n")
        for q, s in zip(QUERIES, scores):
            f.write(f"\"{q}\",{s:.4f}\n")

    # combined figure: best N_BEST queries by judge score
    best = np.argsort(-np.array(scores))[:N_BEST]
    LW = 250
    sheet = Image.new("RGB", (LW + K * TH, N_BEST * TH), "white")
    d = ImageDraw.Draw(sheet)
    for r_, qi in enumerate(best):
        y = r_ * TH
        words, line, lines = QUERIES[qi].split(), "", []
        for w in words:
            if len(line) + len(w) > 24:
                lines.append(line); line = w
            else:
                line = (line + " " + w).strip()
        lines.append(line)
        for li, ln in enumerate(lines):
            d.text((8, y + TH // 2 - 10 * len(lines) + 20 * li), ln,
                   fill=(0, 0, 0), font=font)
        for c, j in enumerate(tops[qi]):
            try:
                im = Image.open(os.path.join(IMG, rel[te[j]])).convert("RGB").resize((TH, TH))
            except Exception:
                im = Image.new("RGB", (TH, TH), (40, 40, 40))
            sheet.paste(im, (LW + c * TH, y))
    os.makedirs("figures", exist_ok=True)
    sheet.save("figures/fig_freeform_retrieval.jpg", quality=95)
    print("saved figures/fig_freeform_retrieval.jpg + per-query strips in", OUT)
    for qi in np.argsort(-np.array(scores)):
        print(f"  {scores[qi]:.3f}  {QUERIES[qi]}")


if __name__ == "__main__":
    main()
