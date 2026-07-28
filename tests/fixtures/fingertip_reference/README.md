# Fingertip Reference Fixture Set (flec-atl)

Annotated close-up hand images used to gate the `yolo26n-pose` fine-tune chain for
Flec Reading mode — which tracks a child's pointing fingertip to read words aloud.

## Dataset Source

**HaGRID** — HAnd Gesture Recognition Image Dataset  
- Repository: <https://github.com/hukenovs/hagrid>  
- HuggingFace: <https://huggingface.co/datasets/hagrid/hagrid-sample-30k-384p>  
- License: **CC BY 4.0** (Creative Commons Attribution 4.0 International)  
- Citation: Kapitanov et al., "HaGRID — HAnd Gesture Recognition Image Dataset", 2022  
  <https://arxiv.org/abs/2206.08219>

Gesture class used: **`point`** — pointing with index finger extended.  
This directly corresponds to the Reading mode intent signal in Flec.

## Proxy Note

Images are adult hand images used as proxies for child hands during development.
The model will be validated against child-specific data before production deployment
per the flec-atl acceptance criteria.

## Structure

```
fingertip_reference/
├── images/          # Downloaded JPG images (gitignored — use download script)
├── labels/          # YOLO pose annotations (.txt, one per image)
├── data.yaml        # YOLO HUB dataset config
├── README.md        # This file
└── .gitignore       # Excludes images/ from git
```

## Annotation Format

YOLO pose format — each `labels/<stem>.txt` contains one line per hand:

```
# <class> <cx> <cy> <w> <h> <kp_x> <kp_y> <kp_vis>
0 0.5 0.5 0.8 0.8  0.5 0.25 2
```

- **class**: 0 = hand
- **cx cy w h**: normalized bounding box centre + size (relative to image dimensions)
- **kp_x kp_y**: normalized index fingertip position
- **kp_vis**: keypoint visibility — `0` = placeholder/unlabeled, `2` = verified visible

Files with `kp_vis=0` are placeholder annotations suitable for human review and
refinement before fine-tuning.

## Populating Images

```bash
python scripts/download_fingertip_fixtures.py
```

Requires either:
- A HuggingFace account token: `export HUGGING_FACE_HUB_TOKEN=<token>` and
  `pip install huggingface_hub`, **or**
- Network access to the HaGRID direct download mirrors

The script downloads 60 images from the HaGRID "point" test split and creates
corresponding YOLO pose placeholder annotations.

## Annotation Refinement

After downloading, run:

```bash
python scripts/annotate_fingertip_fixtures.py
```

This script provides a heuristic estimate of fingertip positions and produces
annotations with `kp_vis=1` (estimated). Human review is recommended before
fine-tuning.

## License Attribution

Images and annotations derived from the HaGRID dataset are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Required attribution:
> Kapitanov A., Makhlyarchuk A., Kvanchiani K. et al., "HaGRID — HAnd Gesture
> Recognition Image Dataset", arXiv:2206.08219, 2022.
