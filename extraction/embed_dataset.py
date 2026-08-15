"""
Generic embedder.  Usage:  python embed_dataset.py {cfp|utk|celeba|lfw}
Embeds each dataset with the face and foundation models and saves per-model npy
(resumable: skips any <model>.npy that already exists) plus a label metadata file.

Face: arcface, adaface, adaface_vit, kprpe, adaface_cnn_wf4m (CVLface).
Foundation: clip, metaclip, siglip, dinov2.
"""
import os, sys, glob
CACHE_ROOT = ".cache"
os.environ.setdefault("HF_HOME", os.path.join(CACHE_ROOT, "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(CACHE_ROOT, "huggingface", "hub"))
os.environ.setdefault("TORCH_HOME", os.path.join(CACHE_ROOT, "torch"))
os.environ.setdefault("XDG_CACHE_HOME", CACHE_ROOT)
os.environ.setdefault("KAGGLEHUB_CACHE", os.path.join(CACHE_ROOT, "kagglehub"))
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "RAYON_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.environ.get("HF_TOKEN")     # required for the gated CVLface face models
B = 64
KHUB = os.path.join(CACHE_ROOT, "kagglehub", "datasets")
FACE = {
    "arcface": ("minchul/cvlface_arcface_ir101_webface4m",
                ".cache/hub/models--minchul--cvlface_arcface_ir101_webface4m"),
    "adaface": ("minchul/cvlface_adaface_ir101_ms1mv2",
                ".cache/hub/models--minchul--cvlface_adaface_ir101_ms1mv2"),
    "adaface_vit": ("minchul/cvlface_adaface_vit_base_webface4m",
                    ".cache/hub/models--minchul--cvlface_adaface_vit_base_webface4m"),
    # AdaFace IR-101 on WebFace4M: matches adaface_vit in loss AND data, differs only in
    # backbone (the ms1mv2 `adaface` above differs in both).
    "adaface_cnn_wf4m": ("minchul/cvlface_adaface_ir101_webface4m",
                         ".cache/hub/models--minchul--cvlface_adaface_ir101_webface4m"),
}
proc_face = T.Compose([T.Resize((112, 112)), T.ToTensor(),
                       T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
KPRPE = ("minchul/cvlface_adaface_vit_base_kprpe_webface4m",
         ".cache/hub/models--minchul--cvlface_adaface_vit_base_kprpe_webface4m")
KPRPE_ALIGNER = ("minchul/cvlface_DFA_mobilenet",
                 ".cache/hub/models--minchul--cvlface_DFA_mobilenet")
CLIP_ID   = "openai/clip-vit-base-patch32"
METACLIP_ID = "facebook/metaclip-b32-400m"
DINO_ID   = "facebook/dinov2-base"
SIGLIP_ID = "google/siglip-base-patch16-224"
CELEBA_SUBSET = 40000     # random subset (202k is overkill for probing/text-AP)
WANT = set(sys.argv[2:])  # optional: `embed_dataset.py <ds> clip kprpe` -> embed only these models


# datasets
def load_cfp():
    # index the 10 frontal images per identity (5,000 images, 500 identities)
    root = os.path.expanduser(
        "~/.cache/kagglehub/datasets/chinafax/cfpw-dataset/versions/1/cfp-dataset/Data/Images")
    recs = []
    for person in sorted(os.listdir(root)):
        fdir = os.path.join(root, person, "frontal")
        if not os.path.isdir(fdir):
            continue
        for img in sorted(os.listdir(fdir)):
            if img.lower().endswith((".jpg", ".jpeg", ".png")):
                recs.append({"identity": person, "image_name": img,
                             "rel_path": os.path.join(person, "frontal", img)})
    os.makedirs("embeddings/cfp", exist_ok=True)
    if not os.path.exists("embeddings/cfp/cfp_metadata.npy"):
        np.save("embeddings/cfp/cfp_metadata.npy", recs, allow_pickle=True)
        print(f"saved cfp metadata ({len(recs)} imgs)")
    return [os.path.join(root, r["rel_path"]) for r in recs], "embeddings/cfp", None


def load_cfp_profile():
    root = os.path.expanduser(
        "~/.cache/kagglehub/datasets/chinafax/cfpw-dataset/versions/1/cfp-dataset/Data/Images")
    recs = []
    for person in sorted(os.listdir(root)):
        pdir = os.path.join(root, person, "profile")
        if not os.path.isdir(pdir):
            continue
        for img in sorted(os.listdir(pdir)):
            if img.lower().endswith((".jpg", ".jpeg", ".png")):
                recs.append({"identity": person, "image_name": img,
                             "rel_path": os.path.join(person, "profile", img)})
    os.makedirs("embeddings/cfp_profile", exist_ok=True)
    np.save("embeddings/cfp_profile/cfp_profile_metadata.npy", recs, allow_pickle=True)
    return [os.path.join(root, r["rel_path"]) for r in recs], "embeddings/cfp_profile", None


def load_utk():
    img = (".cache/kagglehub/datasets/"
           "jangedoo/utkface-new/versions/1/UTKFace")
    recs = []
    for p in sorted(glob.glob(os.path.join(img, "*.jpg"))):
        s = os.path.basename(p).split("_")
        try:
            a, g, r = int(s[0]), int(s[1]), int(s[2])
            if g in (0, 1) and r in range(5) and 0 <= a <= 120:
                recs.append((p, a, g, r))
        except (ValueError, IndexError):
            continue
    return [r[0] for r in recs], "embeddings/utk", None    # utk_metadata.npy already saved


def load_celeba():
    import kagglehub
    base = kagglehub.dataset_download("jessicali9530/celeba-dataset")
    imgdir = next(iter(glob.glob(os.path.join(base, "**", "img_align_celeba", "*.jpg"), recursive=True)
                       or glob.glob(os.path.join(base, "**", "*.jpg"), recursive=True)))
    imgdir = os.path.dirname(imgdir)
    attr_csv = glob.glob(os.path.join(base, "**", "list_attr_celeba.csv"), recursive=True)[0]
    print(f"celeba images: {imgdir}\nceleba attrs: {attr_csv}")
    rows = np.genfromtxt(attr_csv, delimiter=",", dtype=str, skip_header=1)
    names = np.genfromtxt(attr_csv, delimiter=",", dtype=str, max_rows=1)[1:]
    imgids = rows[:, 0]
    attrs = (rows[:, 1:].astype(int) > 0).astype(np.int8)      # -1/1 -> 0/1
    rng = np.random.RandomState(42)
    sel = rng.choice(len(imgids), min(CELEBA_SUBSET, len(imgids)), replace=False)
    sel.sort()
    paths = [os.path.join(imgdir, i) for i in imgids[sel]]
    os.makedirs("embeddings/celeba", exist_ok=True)
    if not os.path.exists("embeddings/celeba/celeba_metadata.npz"):
        np.savez("embeddings/celeba/celeba_metadata.npz",
                 image_id=imgids[sel], attrs=attrs[sel], attr_names=names)
        print(f"saved celeba metadata ({len(sel)} imgs, {attrs.shape[1]} attrs)")
    return paths, "embeddings/celeba", None


def load_lfw():
    # detected+cropped by crop_lfw.py; raw deep-funneled frames are mostly background
    root = "data/lfw_cropped"
    if not os.path.isdir(root):
        raise SystemExit(f"missing {root} -- run crop_lfw.py first")
    recs = []
    people = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    for pid, person in enumerate(people, 1):        # identity index 1-based, like CFP
        for img in sorted(os.listdir(os.path.join(root, person))):
            if img.lower().endswith((".jpg", ".jpeg", ".png")):
                recs.append({"identity": pid, "name": person.replace("_", " "),
                             "rel_path": os.path.join(person, img)})
    os.makedirs("embeddings/lfw", exist_ok=True)
    if not os.path.exists("embeddings/lfw/lfw_metadata.npy"):
        np.save("embeddings/lfw/lfw_metadata.npy", recs, allow_pickle=True)
        print(f"saved lfw metadata ({len(recs)} imgs, {len(people)} identities)")
    return [os.path.join(root, r["rel_path"]) for r in recs], "embeddings/lfw", None


LOADERS = {"cfp": load_cfp, "utk": load_utk, "celeba": load_celeba, "lfw": load_lfw}


# embedding
def run_model(name, embed_batch, paths, save_dir):
    if WANT and name not in WANT:
        return
    out = os.path.join(save_dir, f"{name}.npy")
    if os.path.exists(out):
        print(f"  {name}: cached, skip"); return
    embs = []
    for i in range(0, len(paths), B):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + B]]
        with torch.no_grad():
            embs.append(embed_batch(ims))
        if (i // B) % 40 == 0:
            print(f"    {name}: {i+len(ims)}/{len(paths)}", flush=True)
    np.save(out, np.concatenate(embs, 0).astype(np.float32))
    print(f"  {name}: saved {out}", flush=True)


def main():
    ds = sys.argv[1]
    paths, save_dir, _ = LOADERS[ds]()
    os.makedirs(save_dir, exist_ok=True)
    print(f"[{ds}] {len(paths)} images -> {save_dir}", flush=True)

    from ufe.model_loaders import load_model_by_repo_id
    for name, (repo, path) in FACE.items():
        try:
            m = load_model_by_repo_id(repo, save_path=path, HF_TOKEN=HF_TOKEN).eval()
            run_model(name, lambda ims, m=m: m(torch.stack([proc_face(i) for i in ims]).to(m.device)).cpu().numpy(),
                      paths, save_dir)
            del m; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {name}: FAILED -> {type(e).__name__}: {e}", flush=True)

    if not WANT or "kprpe" in WANT:
        try:
            km = load_model_by_repo_id(KPRPE[0], save_path=KPRPE[1], HF_TOKEN=HF_TOKEN).to(DEVICE).eval()
            ka = load_model_by_repo_id(KPRPE_ALIGNER[0], save_path=KPRPE_ALIGNER[1], HF_TOKEN=HF_TOKEN).to(DEVICE).eval()

            def embed_kprpe(ims):
                x = torch.stack([proc_face(i) for i in ims]).to(DEVICE)
                _, ldmks, _, _, _, _ = ka(x)
                return km(x, ldmks).cpu().numpy()
            run_model("kprpe", embed_kprpe, paths, save_dir)
            del km, ka; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  kprpe: FAILED -> {type(e).__name__}: {e}", flush=True)

    from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoImageProcessor
    if not WANT or "metaclip" in WANT:
        try:
            m2 = CLIPModel.from_pretrained(METACLIP_ID).to(DEVICE).eval()
            p2 = CLIPProcessor.from_pretrained(METACLIP_ID)
            run_model("metaclip", lambda ims, m2=m2, p2=p2: m2.get_image_features(
                **p2(images=ims, return_tensors="pt").to(DEVICE)).cpu().numpy(),
                paths, save_dir)
            del m2; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  metaclip FAILED -> {e}", flush=True)
    try:
        cm = CLIPModel.from_pretrained(CLIP_ID).to(DEVICE).eval()
        cp = CLIPProcessor.from_pretrained(CLIP_ID)
        run_model("clip", lambda ims: cm.get_image_features(**cp(images=ims, return_tensors="pt").to(DEVICE)).cpu().numpy(),
                  paths, save_dir); del cm; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  clip FAILED -> {e}", flush=True)
    try:
        dp = AutoImageProcessor.from_pretrained(DINO_ID)
        dm = AutoModel.from_pretrained(DINO_ID).to(DEVICE).eval()
        run_model("dinov2", lambda ims: dm(**dp(images=ims, return_tensors="pt").to(DEVICE)).last_hidden_state[:, 0].cpu().numpy(),
                  paths, save_dir); del dm; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  dinov2 FAILED -> {e}", flush=True)
    if not WANT or "siglip" in WANT:
        try:
            sp = AutoImageProcessor.from_pretrained(SIGLIP_ID)     # image-only: no SentencePiece tokenizer needed
            sm = AutoModel.from_pretrained(SIGLIP_ID).to(DEVICE).eval()
            run_model("siglip", lambda ims: sm.get_image_features(**sp(images=ims, return_tensors="pt").to(DEVICE)).cpu().numpy(),
                      paths, save_dir); del sm; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  siglip FAILED -> {e}", flush=True)

    print(f"[{ds}] done.", flush=True)


if __name__ == "__main__":
    main()
