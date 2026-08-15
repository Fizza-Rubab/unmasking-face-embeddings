"""
Generate ONE (decoder, variant) over the FULL held-out CFP test set (1500 imgs).
Reuses eval_t2i embedding + decoder logic; writes to eval_out/t2i_full/<dec>/<var>/.
Usage: python gen_t2i_one.py <decoder> <variant> [Nmax]
"""
import os, sys
import numpy as np
import torch
from PIL import Image
import eval_t2i as t

DEC, VAR = sys.argv[1], sys.argv[2]
NMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
FULL_DIR = os.path.join(t.OUT, "t2i_full")


def main():
    meta = np.load(os.path.join(t.EMB_DIR, "cfp_metadata.npy"), allow_pickle=True)
    ids = np.array([int(r["identity"]) for r in meta])
    tr, te = t.identity_split(ids)
    te = te[:NMAX]
    rng = np.random.RandomState(t.SEED)
    embs = t.embeddings_for(DEC, tr, te, rng)[VAR]     # (len(te), d_target)
    cfg = t.DECODERS[DEC]
    device = "cuda"
    print(f"{DEC}/{VAR}: {len(te)} images", flush=True)

    if DEC == "kandinsky":
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
            g = torch.Generator(device).manual_seed(t.SEED)
            return pipe(image_embeds=emb, negative_image_embeds=neg,
                        height=cfg["size"], width=cfg["size"],
                        num_inference_steps=cfg["steps"], guidance_scale=cfg["guid"],
                        generator=g).images[0]
    else:
        from diffusers import StableUnCLIPImg2ImgPipeline
        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            "diffusers/stable-diffusion-2-1-unclip-i2i-h",
            torch_dtype=torch.float32).to(device)
        pipe.set_progress_bar_config(disable=True)
        edt = next(pipe.image_encoder.parameters()).dtype

        def decode(e):
            emb = torch.tensor(e[None], device=device, dtype=edt)
            g = torch.Generator(device).manual_seed(t.SEED)
            return pipe(image_embeds=emb, prompt="", num_inference_steps=cfg["steps"],
                        guidance_scale=cfg["guid"], noise_level=cfg.get("noise", 0),
                        generator=g).images[0]

    d = os.path.join(FULL_DIR, DEC, VAR); os.makedirs(d, exist_ok=True)
    for i in range(len(te)):
        fp = os.path.join(d, f"{i:04d}.jpg")
        if os.path.exists(fp):
            continue
        decode(embs[i]).save(fp, quality=95)
        if i % 100 == 0:
            print(f"  {i}/{len(te)}", flush=True)
    print(f"DONE {DEC}/{VAR}", flush=True)


if __name__ == "__main__":
    main()
