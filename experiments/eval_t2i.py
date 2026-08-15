"""
Template-to-image reconstruction: quantitative and qualitative, both decoders.

Stage GEN (resumable): decode N CFP test faces per decoder x variant:
    decoders : kandinsky (CLIP-bigG, steps 75 / guid 4)
               unclip    (CLIP ViT-H, steps 50 / guid 5 / noise 0, fp32)
    variants : bridged-arcface, bridged-adaface (lstsq bridge, ours)
               native    (decode real target embedding = ceiling)
               unaligned (raw zero-padded FR template fed to decoder)
               random    (untrained random map = floor)
Stage METRICS:
    attribute preservation (primary): CLIP zero-shot binary attrs on generated vs
        original image -> % agreement (per-attribute + mean)
    appearance: cos(DINOv2(gen), DINOv2(orig)) (independent space); LPIPS (alex)
    realism: FID (torch-fidelity) generated-set vs real CFP test crops
    identity (context, not headline): cos(FR(gen), FR(orig)) for ArcFace & AdaFace,
        with genuine/impostor reference means from real CFP images
Outputs: eval_out/t2i/<decoder>/<variant>/*.jpg, eval_out/t2i_metrics.csv,
         eval_out/tab_t2i.tex, figures/fig_t2i_grid.jpg

Usage: python eval_t2i.py gen [kandinsky|unclip]   (GPU, run per decoder)
       python eval_t2i.py metrics                  (GPU)
"""
import os, sys
CACHE_ROOT = ".cache"
os.environ.setdefault("HF_HOME", os.path.join(CACHE_ROOT, "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(CACHE_ROOT, "huggingface", "hub"))
os.environ.setdefault("TORCH_HOME", os.path.join(CACHE_ROOT, "torch"))
os.environ.setdefault("XDG_CACHE_HOME", CACHE_ROOT)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "RAYON_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from PIL import Image

EMB_DIR = "embeddings/cfp"
IMAGE_ROOT = os.path.expanduser(
    "~/.cache/kagglehub/datasets/chinafax/cfpw-dataset/versions/1/cfp-dataset/Data/Images")
SEED, TR_RATIO, N = 42, 0.7, 60
OUT = "eval_out"
GEN_DIR = os.path.join(OUT, "t2i")
FACE = ["arcface", "adaface", "adaface_vit", "kprpe"]
DECODERS = {
    "kandinsky": dict(tag="clip_kandinsky_bigg", steps=75, guid=4.0, size=512),
    "unclip":    dict(tag="clip_unclip_vith",    steps=50, guid=5.0, size=768, noise=0),
}
VARIANTS = ["bridged-arcface", "bridged-adaface", "bridged-adaface_vit",
            "bridged-kprpe", "native", "unaligned", "random"]
ARC = ("minchul/cvlface_arcface_ir101_webface4m",
       ".cache/hub/models--minchul--cvlface_arcface_ir101_webface4m")
ADA = ("minchul/cvlface_adaface_ir101_ms1mv2",
       ".cache/hub/models--minchul--cvlface_adaface_ir101_ms1mv2")
# binary zero-shot attribute probes (CLIP ViT-B/32 on orig vs generated)
ATTRS = [
    ("gender",  "a photo of a man",                "a photo of a woman"),
    ("age",     "a photo of a young person",       "a photo of an old person"),
    ("glasses", "a person wearing eyeglasses",     "a person without eyeglasses"),
    ("beard",   "a man with a beard",              "a clean-shaven person"),
    ("smile",   "a smiling person",                "a person with a neutral expression"),
    ("hair",    "a person with dark hair",         "a person with light or gray hair"),
    ("bald",    "a bald person",                   "a person with a full head of hair"),
]


def l2(x):
    x = np.atleast_2d(np.asarray(x, np.float64))
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def identity_split(ids):
    rng = np.random.RandomState(SEED)
    uniq = np.array(sorted(set(ids))); rng.shuffle(uniq)
    trset = set(uniq[:int(len(uniq) * TR_RATIO)])
    tr = np.array([i for i, x in enumerate(ids) if x in trset])
    te = np.array([i for i, x in enumerate(ids) if x not in trset])
    return tr, te


def setup():
    meta = np.load(os.path.join(EMB_DIR, "cfp_metadata.npy"), allow_pickle=True)
    ids = np.array([int(r["identity"]) for r in meta])
    rel = np.array([r["rel_path"] for r in meta])
    tr, te = identity_split(ids)
    pick = te[np.random.RandomState(SEED).choice(len(te), N, replace=False)]
    return ids, rel, tr, te, pick


def embeddings_for(dec, tr, pick, rng):
    """dict variant -> (N, d_target) embeddings to decode."""
    tag = DECODERS[dec]["tag"]
    C = np.load(f"{EMB_DIR}/{tag}.npy").astype(np.float64)
    muC = C[tr].mean(0)
    out = {"native": C[pick]}
    for src in FACE:
        A = l2(np.load(f"{EMB_DIR}/{src}.npy"))
        muA = A[tr].mean(0)
        W = np.linalg.lstsq(A[tr] - muA, C[tr] - muC, rcond=None)[0]
        out[f"bridged-{src}"] = (A[pick] - muA) @ W + muC
    A0 = l2(np.load(f"{EMB_DIR}/{FACE[0]}.npy"))
    scale = np.linalg.norm(C[tr], axis=1).mean()
    U = np.pad(A0[pick], ((0, 0), (0, max(0, C.shape[1]-A0.shape[1]))))[:, :C.shape[1]]
    out["unaligned"] = U * scale                       # raw template, decoder-space norm
    Wr = rng.normal(0, 1/np.sqrt(A0.shape[1]), (A0.shape[1], C.shape[1]))
    out["random"] = (A0[pick] - A0[tr].mean(0)) @ Wr + muC
    return out


def gen(dec):
    ids, rel, tr, te, pick = setup()
    rng = np.random.RandomState(SEED)
    embs = embeddings_for(dec, tr, pick, rng)
    cfg = DECODERS[dec]
    device = "cuda"
    if dec == "kandinsky":
        from diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline
        dtype = torch.float16
        prior = KandinskyV22PriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior", torch_dtype=dtype).to(device)
        neg = prior("", guidance_scale=4.0, num_inference_steps=25,
                    generator=torch.Generator(device).manual_seed(0)).negative_image_embeds
        del prior; torch.cuda.empty_cache()
        pipe = KandinskyV22Pipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=dtype).to(device)
        pipe.set_progress_bar_config(disable=True)

        def decode(e):
            emb = torch.tensor(e[None], device=device, dtype=dtype)
            g = torch.Generator(device).manual_seed(SEED)
            return pipe(image_embeds=emb, negative_image_embeds=neg,
                        height=cfg["size"], width=cfg["size"],
                        num_inference_steps=cfg["steps"], guidance_scale=cfg["guid"],
                        generator=g).images[0]
    else:
        from diffusers import StableUnCLIPImg2ImgPipeline
        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            "diffusers/stable-diffusion-2-1-unclip-i2i-h",
            torch_dtype=torch.float32).to(device)          # fp16 -> NaN
        pipe.set_progress_bar_config(disable=True)
        edt = next(pipe.image_encoder.parameters()).dtype

        def decode(e):
            emb = torch.tensor(e[None], device=device, dtype=edt)
            g = torch.Generator(device).manual_seed(SEED)
            return pipe(image_embeds=emb, prompt="", num_inference_steps=cfg["steps"],
                        guidance_scale=cfg["guid"], noise_level=cfg.get("noise", 0),
                        generator=g).images[0]

    for var in VARIANTS:
        d = os.path.join(GEN_DIR, dec, var); os.makedirs(d, exist_ok=True)
        for i in range(N):
            fp = os.path.join(d, f"{i:03d}.jpg")
            if os.path.exists(fp):
                continue
            decode(embs[var][i]).save(fp, quality=95)
        print(f"[{dec}] {var}: done")


# ------------------------------------------------------------------ metrics
def load_fr(spec, device):
    from ufe.model_loaders import load_model_by_repo_id
    repo, path = spec
    return load_model_by_repo_id(repo, save_path=path, HF_TOKEN=None).to(device).eval()


def metrics():
    from torchvision import transforms as T
    device = "cuda"
    ids, rel, tr, te, pick = setup()
    origs = [Image.open(os.path.join(IMAGE_ROOT, rel[i])).convert("RGB") for i in pick]

    # ---------- CLIP zero-shot attribute agreement + DINOv2 + LPIPS
    from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor, AutoModel, AutoImageProcessor
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    ctok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    cproc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    prompts = [p for _, a, b in ATTRS for p in (a, b)]
    with torch.no_grad():
        tt = clip.get_text_features(**ctok(prompts, return_tensors="pt",
                                           padding=True).to(device))
        tt = torch.nn.functional.normalize(tt, dim=-1).cpu().numpy()

    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    import lpips as lpips_pkg
    lp = lpips_pkg.LPIPS(net="alex").to(device)
    to256 = T.Compose([T.Resize((256, 256)), T.ToTensor()])

    def clip_img(images):
        out = []
        with torch.no_grad():
            for i in range(0, len(images), 32):
                inp = cproc(images=images[i:i+32], return_tensors="pt").to(device)
                v = clip.get_image_features(**inp)
                out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        return np.concatenate(out)

    def dino_emb(images):
        out = []
        with torch.no_grad():
            for i in range(0, len(images), 32):
                inp = dproc(images=images[i:i+32], return_tensors="pt").to(device)
                v = dino(**inp).last_hidden_state[:, 0]
                out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        return np.concatenate(out)

    def attr_preds(cemb):
        # (n, n_attr) boolean: first prompt of pair wins
        s = cemb @ tt.T
        return np.stack([s[:, 2*i] > s[:, 2*i+1] for i in range(len(ATTRS))], 1)

    orig_attr = attr_preds(clip_img(origs))
    orig_dino = dino_emb(origs)

    # ---------- FR identity + genuine/impostor reference
    fr = {}
    proc_face = T.Compose([T.Resize((112, 112)), T.ToTensor(),
                           T.Normalize([0.5]*3, [0.5]*3)])
    for name, spec in (("arcface", ARC), ("adaface", ADA)):
        fr[name] = load_fr(spec, device)
    def fr_emb(model, images):
        out = []
        with torch.no_grad():
            for i in range(0, len(images), 32):
                x = torch.stack([proc_face(im) for im in images[i:i+32]]).to(device)
                out.append(model(x).cpu().numpy())
        return l2(np.concatenate(out))
    ref = {}
    for name in fr:
        E = l2(np.load(f"{EMB_DIR}/{name}.npy"))
        same = ids[:, None] == ids[None, :]
        r = np.random.RandomState(0)
        ii = r.choice(len(ids), 4000); jj = r.choice(len(ids), 4000)
        m = ii != jj
        sims = (E[ii[m]] * E[jj[m]]).sum(1); lab = same[ii[m], jj[m]]
        ref[name] = (float(sims[lab].mean()), float(sims[~lab].mean()))
    orig_fr = {name: fr_emb(fr[name], origs) for name in fr}

    rows = []
    for dec in DECODERS:
        for var in VARIANTS:
            d = os.path.join(GEN_DIR, dec, var)
            if not os.path.isdir(d) or len(os.listdir(d)) < N:
                print(f"skip {dec}/{var} (images missing)"); continue
            gens = [Image.open(os.path.join(d, f"{i:03d}.jpg")).convert("RGB")
                    for i in range(N)]
            g_attr = attr_preds(clip_img(gens))
            attr_ag = float((g_attr == orig_attr).mean())
            per_attr = (g_attr == orig_attr).mean(0)
            dino_cos = float((dino_emb(gens) * orig_dino).sum(1).mean())
            with torch.no_grad():
                lpv = float(np.mean([lp(to256(a)[None].to(device)*2-1,
                                        to256(b)[None].to(device)*2-1).item()
                                     for a, b in zip(gens, origs)]))
            idcos = {name: float((fr_emb(fr[name], gens) * orig_fr[name]).sum(1).mean())
                     for name in fr}
            rows.append(dict(decoder=dec, variant=var, attr_agree=attr_ag,
                             dino=dino_cos, lpips=lpv,
                             id_arc=idcos["arcface"], id_ada=idcos["adaface"],
                             **{f"attr_{a[0]}": float(v)
                                for a, v in zip(ATTRS, per_attr)}))
            print(f"{dec:9s} {var:16s} attr {attr_ag:.3f}  dino {dino_cos:.3f}  "
                  f"lpips {lpv:.3f}  idA {idcos['arcface']:.3f} idAd {idcos['adaface']:.3f}")
    for m in fr.values():
        del m
    torch.cuda.empty_cache()

    # ---------- FID: each generated set vs 500 real CFP test faces
    import tempfile, shutil
    from torch_fidelity import calculate_metrics
    real_dir = os.path.join(GEN_DIR, "_real")
    if not os.path.isdir(real_dir) or len(os.listdir(real_dir)) < 500:
        os.makedirs(real_dir, exist_ok=True)
        r = np.random.RandomState(1)
        for k, i in enumerate(te[r.choice(len(te), 500, replace=False)]):
            Image.open(os.path.join(IMAGE_ROOT, rel[i])).convert("RGB")\
                 .resize((299, 299)).save(os.path.join(real_dir, f"{k:04d}.jpg"))
    for row in rows:
        d = os.path.join(GEN_DIR, row["decoder"], row["variant"])
        r = calculate_metrics(input1=d, input2=real_dir, cuda=True, fid=True,
                              verbose=False)
        row["fid"] = float(r["frechet_inception_distance"])
        print(f"FID {row['decoder']}/{row['variant']}: {row['fid']:.1f}")

    import csv
    keys = list(rows[0].keys())
    with open(f"{OUT}/t2i_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)

    pretty = {"bridged-arcface": r"\textbf{bridged} ArcFace",
              "bridged-adaface": r"\textbf{bridged} AdaFace",
              "bridged-adaface_vit": r"\textbf{bridged} AdaFace-ViT",
              "bridged-kprpe": r"\textbf{bridged} KPRPE",
              "native": "native (ceiling)", "unaligned": "unaligned",
              "random": "random map"}
    gm = {n: ref[n] for n in ref}
    with open(f"{OUT}/tab_t2i.tex", "w") as f:
        f.write("% template->image, auto-generated by eval_t2i.py\n")
        f.write(f"% FR ref means: arcface genuine {gm['arcface'][0]:.2f} / impostor "
                f"{gm['arcface'][1]:.2f}; adaface {gm['adaface'][0]:.2f} / {gm['adaface'][1]:.2f}\n")
        f.write("\\begin{tabular}{llcccccc}\n\\toprule\n")
        f.write("Decoder & Source & Attr.\\ agree $\\uparrow$ & DINOv2 $\\uparrow$ & "
                "LPIPS $\\downarrow$ & FID $\\downarrow$ & ArcFace cos & AdaFace cos \\\\\n\\midrule\n")
        for dec in DECODERS:
            first = True
            for var in ["native", "bridged-arcface", "bridged-adaface",
                        "bridged-adaface_vit", "bridged-kprpe", "unaligned", "random"]:
                row = next((r_ for r_ in rows
                            if r_["decoder"] == dec and r_["variant"] == var), None)
                if row is None:
                    continue
                dn = dec.capitalize() if first else ""
                first = False
                f.write(f"{dn} & {pretty[var]} & {row['attr_agree']:.3f} & "
                        f"{row['dino']:.3f} & {row['lpips']:.3f} & {row['fid']:.1f} & "
                        f"{row['id_arc']:.3f} & {row['id_ada']:.3f} \\\\\n")
            if dec != list(DECODERS)[-1]:
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"saved {OUT}/t2i_metrics.csv, tab_t2i.tex")

    # ---------- qualitative grid figure: 6 ids x (orig + 4 variants) x 2 decoders
    TH = 200
    show_vars = ["native", "bridged-arcface", "unaligned", "random"]
    r = np.random.RandomState(3)
    sel = r.choice(N, 6, replace=False)
    cols = 1 + len(show_vars) * len(DECODERS)
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    sheet = Image.new("RGB", (cols * TH, len(sel) * TH + 22), "white")
    dr = ImageDraw.Draw(sheet)
    heads = ["original"] + [f"{d[:4]}-{v.replace('bridged-','br.')}"
                            for d in DECODERS for v in show_vars]
    for c, h in enumerate(heads):
        dr.text((c * TH + 4, 4), h, fill=(0, 0, 0), font=font)
    for ri, i in enumerate(sel):
        y = 22 + ri * TH
        sheet.paste(origs[i].resize((TH, TH)), (0, y))
        c = 1
        for dec in DECODERS:
            for var in show_vars:
                fp = os.path.join(GEN_DIR, dec, var, f"{i:03d}.jpg")
                im = Image.open(fp).resize((TH, TH)) if os.path.exists(fp) \
                    else Image.new("RGB", (TH, TH), (230, 230, 230))
                sheet.paste(im, (c * TH, y)); c += 1
    os.makedirs("figures", exist_ok=True)
    sheet.save("figures/fig_t2i_grid.jpg", quality=95)
    print("saved figures/fig_t2i_grid.jpg")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "gen"
    if mode == "gen":
        gen(sys.argv[2] if len(sys.argv) > 2 else "kandinsky")
    else:
        metrics()
