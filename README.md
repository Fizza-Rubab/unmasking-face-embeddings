# Unmasking Face Embeddings: Reading, Rendering and Naming with Foundation Models

Official code for the paper, accepted at the ECCV 2026 Workshop on Foundation and
Generative Models in Biometrics (FoundGen),
**"Unmasking Face Embeddings: Reading, Rendering and Naming with Foundation Models"**
Fizza Rubab, Yiying Tong, Arun Ross (Michigan State University).

A single linear map, estimated once from paired embeddings, aligns a face-recognition
(FR) model with an off-the-shelf foundation model. Once aligned, a face embedding can be
**read** with free-form text, **rendered** into a face image by an unmodified diffusion
decoder, and **named** against a text vocabulary of candidate names, all without training
or modifying either model.

---

## What this repository contains

Code to reproduce the experiments in the paper:

1. **Extract** embeddings for each model on each dataset (`extraction/`).
2. **Align** an FR space with a foundation space by a mean-centered linear map (`ufe/align.py`).
3. **Evaluate** the three capabilities (retrieval, embedding-to-image, and naming), plus
   the cross-dataset, web-exposure, and robustness analyses (`experiments/`).
4. **Render** the paper figures (`figures/`).


---

## Repository structure

```
ufe/               # importable library (installed with `pip install -e .`)
  align.py         #   the linear map: LinearAlignment.fit / .transform (paper Eq. 1)
  model_loaders.py #   CVLface / MagFace checkpoint loaders
extraction/        # embed_dataset.py (all datasets) + crop_lfw.py  (-> embeddings/<ds>/<model>.npy)
experiments/       # retrieval, generation, naming, cross-dataset, exposure, ablations
figures/           # overview + qualitative montages + result plots
```

At runtime the code also uses these git-ignored folders (create or symlink them):

```
data/          # raw datasets (see "Datasets")
checkpoints/   # optional MagFace weights
embeddings/    # generated: embeddings/<dataset>/<model>.npy  (+ <dataset>_metadata.npy)
eval_out/      # generated: per-experiment CSV outputs
.cache/        # Hugging Face / kagglehub cache (HF_HOME)
```

---

## Installation

```bash
conda create -n ufe python=3.11 -y
conda activate ufe
pip install -e .
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

If you only run the alignment/evaluation on existing embeddings, `pip install -e .` is enough.

---

## Datasets

| Dataset | Used for | Notes |
|---|---|---|
| CFP | naming, embedding-to-image | 500 named celebrities, 10 frontal images/identity |
| UTKFace | text-to-face retrieval | age/gender/ethnicity labels |
| CelebA | text-to-face retrieval | 40 binary attributes |
| LFW | cross-dataset naming, web-exposure | 5,749 named identities |

Datasets download to the Hugging Face / kagglehub cache under `.cache/`; the per-dataset
loaders in `extraction/embed_dataset.py` point at those cache paths. Splits are
identity-disjoint wherever identity labels exist.

## Models (paper Table 1)

Face-specific weights come from [CVLface](https://github.com/mk-minchul/CVLface) (gated:
set `export HF_TOKEN=...`); foundation models load from Hugging Face `transformers` /
`diffusers`. None of the foundation targets is face-tuned.

- **Face (source):** ArcFace (ir101, WebFace4M), AdaFace (ir101, MS1MV2),
  AdaFace-ViT (ViT-B, WebFace4M), KPRPE (ViT-B, WebFace4M)
- **Foundation (target):** CLIP ViT-B/32, MetaCLIP ViT-B/32, SigLIP ViT-B/16,
  Kandinsky 2.2 (CLIP ViT-bigG), Stable unCLIP (CLIP ViT-H)
- **Reference only:** DINOv2 ViT-B/14 (appearance metric), Arc2Face (face-native decoder)

---

## Pipeline

Run everything **from the repository root**.

### 1. Extract embeddings

```bash
export HF_TOKEN=...                        # gated CVLface face models
python extraction/embed_dataset.py cfp     # -> embeddings/cfp/<model>.npy
python extraction/embed_dataset.py utk     # -> embeddings/utk/<model>.npy
python extraction/embed_dataset.py celeba  # -> embeddings/celeba/<model>.npy
python extraction/crop_lfw.py              # LFW: central-crop the funneled frames first
python extraction/embed_dataset.py lfw     # -> embeddings/lfw/<model>.npy
```

Each writes `embeddings/<dataset>/<dataset>_metadata.npy` (per-image identity/path records,
row-aligned with the embeddings). Extraction is resumable and skips any `<model>.npy` that
already exists.

### 2. Run experiments

```bash
# Reading: text-to-face retrieval (Table 2)
python experiments/eval_text_retrieval.py        # full grid, single split
python experiments/eval_text_retrieval_ms.py     # 5 splits -> reported mean +/- std

# Rendering: embedding-to-image (Table 3, Fig 3)
python experiments/eval_t2i.py                   # quantitative + qualitative grid
python experiments/gen_t2i_one.py <decoder> <variant>   # full 1500-image generation
python experiments/metrics_t2i_full.py           # attr / DINOv2 / LPIPS / FID / id-cos
python experiments/gen_arc2face.py               # Arc2Face face-native reference row

# Naming (Table 4, Fig 4)
python experiments/eval_naming.py                # top-k, open-set, vocab-size ablation
python experiments/eval_naming_ms.py             # 5 splits -> reported mean +/- std
python experiments/eval_naming_extra_ms.py       # matched-CNN + vocab ablation (5 splits)

# Cross-dataset transfer (Table 5) and web exposure (Table 6)
python experiments/eval_naming_crossds.py        # naming CFP -> LFW
python experiments/eval_retrieval_crossds.py     # retrieval CelebA <-> UTK
python experiments/floors_cross.py               # native / random / unaligned floors
python experiments/eval_naming_exposure.py       # naming vs. images-per-identity

# Supporting claims
python experiments/eval_ridge_ablation.py        # unregularized vs. ridge (Method)
python experiments/eval_utk_leakage.py           # identity-disjoint re-check (Setup)
```

Outputs are written as CSV under `eval_out/`.

### 3. Figures

```bash
python figures/make_overview_figure.py     # Fig 1  overview
python figures/fig_freeform_retrieval.py   # Fig 2  free-form retrieval montage
python figures/make_paper_figures.py       # Fig 4  naming-vs-vocabulary plot
python figures/make_montage_figures.py     # Fig 5  naming montage
```

---

## The alignment API (`ufe/align.py`)

```python
from ufe import LinearAlignment, l2

algo = LinearAlignment()
algo.fit(A_train, C_train)     # A: face embeddings, C: foundation embeddings (paired)
C_hat = algo.transform(A_test) # align held-out face embeddings into the foundation space
# then apply any foundation head (text scoring, a diffusion decoder, name matching) to C_hat
```

---

## Scripts and paper artifacts

| Paper artifact | Script(s) |
|---|---|
| Table 2: retrieval | `experiments/eval_text_retrieval.py`, `eval_text_retrieval_ms.py` |
| Table 3 / Fig 3: embedding-to-image | `experiments/eval_t2i.py`, `gen_t2i_one.py`, `metrics_t2i_full.py`, `gen_arc2face.py` |
| Table 4 / Fig 4: naming | `experiments/eval_naming.py`, `eval_naming_ms.py`, `eval_naming_extra_ms.py`, `figures/make_paper_figures.py` |
| Table 5: cross-dataset | `experiments/eval_naming_crossds.py`, `eval_retrieval_crossds.py`, `floors_cross.py` |
| Table 6: web exposure | `experiments/eval_naming_exposure.py` |
| Method: unregularized map | `experiments/eval_ridge_ablation.py` |
| Setup: UTK leakage check | `experiments/eval_utk_leakage.py` |
| Fig 1 / Fig 2 / Fig 5: qualitative | `figures/make_overview_figure.py`, `fig_freeform_retrieval.py`, `make_montage_figures.py` |

---

## Citation

```bibtex
@inproceedings{rubab2026unmasking,
  title     = {Unmasking Face Embeddings: Reading, Rendering and Naming with Foundation Models},
  author    = {Rubab, Fizza and Tong, Yiying and Ross, Arun},
  booktitle = {ECCV Workshop on Foundation and Generative Models in Biometrics (FoundGen)},
  year      = {2026}
}
```

This work builds on our earlier study of face-embedding compatibility:

```bibtex
@inproceedings{rubab2026compatibility,
  title     = {Compatibility of Face Embeddings Across Deep Neural Networks},
  author    = {Rubab, Fizza and Tong, Yiying and Ross, Arun},
  booktitle = {IEEE International Joint Conference on Biometrics (IJCB)},
  year      = {2026}
}
```

## License

Released under the MIT License (see `LICENSE`).
