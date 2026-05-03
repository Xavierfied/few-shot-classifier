Room Classification (ResNet + CLIP)
=================================

Simple toolkit for identifying room types from images using:
- ResNet50 few-shot (prototype embeddings + cosine similarity)
- CLIP zero-shot (text labels only)
- CLIP few-shot (support images -> prototypes)

Prerequisites
-------------
- Python 3.8+
- Virtualenv recommended
- Install dependencies from the provided `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate   # or `.venv\\Scripts\\Activate.ps1` on Windows PowerShell
pip install -r requirements.txt
```

Quick usage
-----------
- ResNet few-shot (uses `embeddings/support.pt`):

```bash
python main.py --model resnet_cosine --source kit1.jpg
```

- CLIP zero-shot (provide text labels):

```bash
python main.py --model clip_zeroshot --source kit1.jpg --labels kitchen bedroom "living room"
```

- CLIP few-shot (loads `embeddings/clip_support.pt` by default):

```bash
python main.py --model clip_fewshot --source kit1.jpg --embeddings embeddings/clip_support.pt
```

- Compare all models and save combined JSON:

```bash
python main.py --compare_all --source kit1.jpg
```

Support builders
----------------
- Build ResNet support embeddings:

```bash
python resnet_fewshot/build_support.py
```

- Build CLIP few-shot support embeddings:

```bash
python clip_model/build_support.py
```

Output
------
- Results saved under `results/`.
- Single-model runs create `<source>_<model>_result.json`.
- `--compare_all` creates `<source>_comparison_result.json` containing each model's predictions and ordered scores.

Configuration
-------------
- Default labels are in `utils/args.py` (`--labels`).
- Default embeddings paths:
  - ResNet: `embeddings/support.pt`
  - CLIP few-shot: `embeddings/clip_support.pt`

Notes
-----
- CLIP zero-shot uses text-label prompts (no support images required).
- Passing a single label can produce a scalar score; the code handles that case.
