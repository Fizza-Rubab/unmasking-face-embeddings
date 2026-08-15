"""
Metrics over the FULL held-out CFP test set for all generated variants in
eval_out/t2i_full/ (2 decoders x 7 variants) plus the Arc2Face reference.
Per variant: attribute agreement (7 zero-shot CLIP attrs), DINOv2 cosine, LPIPS,
identity cosine (ArcFace & AdaFace), FID vs all real CFP test faces.
Out: eval_out/t2i_full_metrics.csv
"""
import os, csv, glob
import numpy as np
import torch
from PIL import Image
import eval_t2i as t

FULL = os.path.join(t.OUT, "t2i_full")
REAL = os.path.join(FULL, "_real")


def main():
    from torchvision import transforms as T
    device = "cuda"
    meta = np.load(os.path.join(t.EMB_DIR, "cfp_metadata.npy"), allow_pickle=True)
    ids = np.array([int(r["identity"]) for r in meta])
    rel = np.array([r["rel_path"] for r in meta])
    tr, te = t.identity_split(ids)
    orig_paths = [os.path.join(t.IMAGE_ROOT, rel[i]) for i in te]   # index i -> file {i:04d}

    from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor, AutoModel, AutoImageProcessor
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    ctok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    cproc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    prompts = [p for _, a, b in t.ATTRS for p in (a, b)]
    with torch.no_grad():
        tt = torch.nn.functional.normalize(
            clip.get_text_features(**ctok(prompts, return_tensors="pt", padding=True).to(device)),
            dim=-1).cpu().numpy()
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    import lpips as lpips_pkg
    lp = lpips_pkg.LPIPS(net="alex").to(device)
    to256 = T.Compose([T.Resize((256, 256)), T.ToTensor()])

    def clip_img(images):
        out = []
        with torch.no_grad():
            for i in range(0, len(images), 64):
                v = clip.get_image_features(**cproc(images=images[i:i+64], return_tensors="pt").to(device))
                out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        return np.concatenate(out)

    def dino_emb(images):
        out = []
        with torch.no_grad():
            for i in range(0, len(images), 64):
                v = dino(**dproc(images=images[i:i+64], return_tensors="pt").to(device)).last_hidden_state[:, 0]
                out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        return np.concatenate(out)

    def attr_preds(cemb):
        s = cemb @ tt.T
        return np.stack([s[:, 2*i] > s[:, 2*i+1] for i in range(len(t.ATTRS))], 1)

    # FR models
    proc_face = T.Compose([T.Resize((112, 112)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    fr = {"arcface": t.load_fr(t.ARC, device), "adaface": t.load_fr(t.ADA, device)}

    def fr_emb(model, images):
        out = []
        with torch.no_grad():
            for i in range(0, len(images), 64):
                x = torch.stack([proc_face(im) for im in images[i:i+64]]).to(device)
                out.append(model(x).cpu().numpy())
        return t.l2(np.concatenate(out))

    # cache originals (all 1500)
    origs_all = [Image.open(p).convert("RGB") for p in orig_paths]
    orig_attr_all = attr_preds(clip_img(origs_all))
    orig_dino_all = dino_emb(origs_all)
    orig_fr_all = {n: fr_emb(fr[n], origs_all) for n in fr}

    # discover variants: t2i_full/<dec>/<var> and t2i_full/arc2face/arc2face
    tasks = []
    for dec in ("kandinsky", "unclip"):
        for var in t.VARIANTS:
            d = os.path.join(FULL, dec, var)
            if os.path.isdir(d) and glob.glob(d + "/*.jpg"):
                tasks.append((dec, var, d))
    a2f = os.path.join(FULL, "arc2face", "arc2face")
    if os.path.isdir(a2f) and glob.glob(a2f + "/*.jpg"):
        tasks.append(("arc2face", "arc2face", a2f))

    rows = []
    for dec, var, d in tasks:
        files = sorted(glob.glob(d + "/*.jpg"))
        idxs = [int(os.path.basename(f)[:4]) for f in files]
        gens = [Image.open(f).convert("RGB") for f in files]
        n = len(gens)
        oa = orig_attr_all[idxs]; od = orig_dino_all[idxs]
        g_attr = attr_preds(clip_img(gens))
        attr_ag = float((g_attr == oa).mean())
        dino_cos = float((dino_emb(gens) * od).sum(1).mean())
        with torch.no_grad():
            lpv = float(np.mean([lp(to256(g)[None].to(device)*2-1,
                                    to256(origs_all[j])[None].to(device)*2-1).item()
                                 for g, j in zip(gens, idxs)]))
        idcos = {nm: float((fr_emb(fr[nm], gens) * orig_fr_all[nm][idxs]).sum(1).mean()) for nm in fr}
        rows.append(dict(decoder=dec, variant=var, n=n, attr=attr_ag, dino=dino_cos,
                         lpips=lpv, id_arc=idcos["arcface"], id_ada=idcos["adaface"]))
        print(f"{dec:9s} {var:18s} n={n:4d} attr {attr_ag:.3f} dino {dino_cos:.3f} "
              f"lpips {lpv:.3f} idA {idcos['arcface']:.3f} idAd {idcos['adaface']:.3f}", flush=True)
    for m in fr.values():
        del m
    torch.cuda.empty_cache()

    # FID: all real CFP test faces
    os.makedirs(REAL, exist_ok=True)
    if len(glob.glob(REAL + "/*.jpg")) < len(te):
        for k, p in enumerate(orig_paths):
            Image.open(p).convert("RGB").resize((299, 299)).save(os.path.join(REAL, f"{k:04d}.jpg"))
    from torch_fidelity import calculate_metrics
    for row in rows:
        d = os.path.join(FULL, row["decoder"], row["variant"])
        r = calculate_metrics(input1=d, input2=REAL, cuda=True, fid=True, verbose=False)
        row["fid"] = float(r["frechet_inception_distance"])
        print(f"FID {row['decoder']}/{row['variant']}: {row['fid']:.1f}", flush=True)

    with open(os.path.join(t.OUT, "t2i_full_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["decoder", "variant", "n", "attr", "dino",
                                          "lpips", "id_arc", "id_ada", "fid"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nsaved {t.OUT}/t2i_full_metrics.csv")


if __name__ == "__main__":
    main()
