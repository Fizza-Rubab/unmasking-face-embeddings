"""
Arc2Face reference generation: face-native decoder identity-retention ceiling.
For each held-out CFP test image, extract its antelopev2 (WebFace42M ArcFace) ID
embedding and generate a face with Arc2Face (SD1.5). Same test order/indexing as
gen_t2i_one.py so metrics compare against the same originals.
Out: eval_out/t2i_full/arc2face/arc2face/{idx:04d}.jpg  (+ present_idx.npy)
Usage: python gen_arc2face.py [Nmax]
"""
import os, sys
sys.path.insert(0, "arc2face_repo")
import numpy as np
import cv2
import torch
from PIL import Image
import eval_t2i as t

NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
OUTD = os.path.join(t.OUT, "t2i_full", "arc2face", "arc2face")
STEPS, GUID = 25, 3.0


def main():
    os.makedirs(OUTD, exist_ok=True)
    meta = np.load(os.path.join(t.EMB_DIR, "cfp_metadata.npy"), allow_pickle=True)
    ids = np.array([int(r["identity"]) for r in meta])
    rel = np.array([r["rel_path"] for r in meta])
    tr, te = t.identity_split(ids)
    te = te[:NMAX]
    device = "cuda"

    # Arc2Face pipeline
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
    from arc2face import CLIPTextModelWrapper, project_face_embs
    from insightface.app import FaceAnalysis
    enc = CLIPTextModelWrapper.from_pretrained("arc2face_repo/models", subfolder="encoder",
                                               torch_dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained("arc2face_repo/models", subfolder="arc2face",
                                                torch_dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", text_encoder=enc, unet=unet,
        torch_dtype=torch.float16, safety_checker=None).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    app = FaceAnalysis(name="antelopev2", root="arc2face_repo",
                       allowed_modules=["detection", "recognition"],
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print(f"arc2face: {len(te)} target images", flush=True)

    present = []
    for i, gi in enumerate(te):
        fp = os.path.join(OUTD, f"{i:04d}.jpg")
        if os.path.exists(fp):
            present.append(i); continue
        img = np.array(Image.open(os.path.join(t.IMAGE_ROOT, rel[gi])).convert("RGB"))[:, :, ::-1]
        # CFP images are tight face crops that fill the frame; pad so SCRFD can detect
        img = cv2.copyMakeBorder(img, 250, 250, 250, 250, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        faces = app.get(img)
        if not faces:
            continue
        f = sorted(faces, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        emb = torch.tensor(f['embedding'], dtype=torch.float16)[None].to(device)
        emb = emb / torch.norm(emb, dim=1, keepdim=True)
        with torch.no_grad():
            pe = project_face_embs(pipe, emb)
            out = pipe(prompt_embeds=pe, num_inference_steps=STEPS,
                       guidance_scale=GUID, num_images_per_prompt=1).images[0]
        out.save(fp, quality=95)
        present.append(i)
        if i % 100 == 0:
            print(f"  {i}/{len(te)} (detected {len(present)})", flush=True)
    np.save(os.path.join(t.OUT, "t2i_full", "arc2face", "present_idx.npy"), np.array(present))
    print(f"DONE arc2face: {len(present)}/{len(te)} faces detected+generated", flush=True)


if __name__ == "__main__":
    main()
