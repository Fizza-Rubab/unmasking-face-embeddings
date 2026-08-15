# Unmasking Face Embeddings: Reading, Rendering and Naming with Foundation Models

Official code for the paper, accepted at the ECCV 2026 Workshop on Foundation and
Generative Models in Biometrics (FoundGen),
**"Unmasking Face Embeddings: Reading, Rendering and Naming with Foundation Models"**
Fizza Rubab, Yiying Tong, Arun Ross (Michigan State University).

A single linear map, estimated once from paired embeddings, aligns a face-recognition
model with an off-the-shelf foundation model. Once aligned, a face embedding can be read
with free-form text, rendered into a face image by an unmodified diffusion decoder, and
named against a text vocabulary, without training or modifying either model.

## Repository structure

```
ufe/               importable library (pip install -e .): the linear map + model loaders
extraction/        embed_dataset.py (all datasets), crop_lfw.py
experiments/       retrieval, generation, naming, cross-dataset, exposure, ablations
figures/           overview, qualitative montages, result plots
```

## Installation

```bash
conda create -n ufe python=3.11 -y
conda activate ufe
pip install -e .
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

`pip install -e .` alone is enough to run the evaluations on existing embeddings.

## Datasets

| Dataset | Used for |
|---|---|
| CFP | naming, embedding-to-image |
| UTKFace | text-to-face retrieval |
| CelebA | text-to-face retrieval |
| LFW | cross-dataset naming, web exposure |

Datasets download to the Hugging Face / kagglehub cache under `.cache/`; the per-dataset
loaders in `extraction/embed_dataset.py` point at those paths.

## Models

Face weights come from CVLface (gated: `export HF_TOKEN=...`); foundation models load from
Hugging Face `transformers` / `diffusers` (paper Table 1). None of the foundation targets is
face-tuned.

Face: ArcFace, AdaFace, AdaFace-ViT, KPRPE.
Foundation: CLIP, MetaCLIP, SigLIP, Kandinsky 2.2, Stable unCLIP; DINOv2 (metric only).

## Pipeline

Run from the repository root.

```bash
export HF_TOKEN=...
python extraction/embed_dataset.py cfp
python extraction/embed_dataset.py utk
python extraction/embed_dataset.py celeba
python extraction/crop_lfw.py
python extraction/embed_dataset.py lfw

python experiments/eval_text_retrieval.py
python experiments/eval_t2i.py
python experiments/eval_naming.py

python figures/make_overview_figure.py
```

## Alignment API

```python
from ufe import LinearAlignment, l2

algo = LinearAlignment()
algo.fit(A_train, C_train)
C_hat = algo.transform(A_test)
```

## Scripts and analysis

| Paper artifact | Script(s) |
|---|---|
| Table 2: retrieval | `experiments/eval_text_retrieval.py`, `eval_text_retrieval_ms.py` |
| Table 3 / Fig 3: embedding-to-image | `experiments/eval_t2i.py`, `gen_t2i_one.py`, `metrics_t2i_full.py`, `gen_arc2face.py` |
| Table 4 / Fig 4: naming | `experiments/eval_naming.py`, `eval_naming_ms.py`, `eval_naming_extra_ms.py`, `figures/make_paper_figures.py` |
| Table 5: cross-dataset | `experiments/eval_naming_crossds.py`, `eval_retrieval_crossds.py`, `floors_cross.py` |
| Table 6: web exposure | `experiments/eval_naming_exposure.py` |
| Method: unregularized map | `experiments/eval_ridge_ablation.py` |
| Setup: UTK leakage check | `experiments/eval_utk_leakage.py` |
| Figures 1, 2, 5 | `figures/make_overview_figure.py`, `fig_freeform_retrieval.py`, `make_montage_figures.py` |

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
