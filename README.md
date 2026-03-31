# Learning-Based Conditional Image Compression

Official implementation of **"Learning-Based Conditional Image Compression"** (ISCAS 2024).

> Tianma Shen, Wen-Hsiao Peng, Huang-Chia Shih, Ying Liu  
> Santa Clara University · National Yang Ming Chiao Tung University · Yuan Ze University

[[Paper]](https://www.cse.scu.edu/~yliu1/papers/ISCAS2024-TianmaShen-FinalSubmission.pdf)

---

## Overview

We propose a novel transformer-based **conditional coding paradigm** for learned image compression. Rather than compressing the target image directly, the framework:

1. Compresses a 4× downsampled low-resolution version of the image using an existing codec.
2. Upscales the decoded low-resolution image via a pretrained **SwinIR** super-resolution network.
3. Uses the super-resolved image as **conditional information** to compress the original high-resolution image through a cross-attention Swin Transformer architecture.



---

## Installation

> This project heavily depends on [CompressAI](https://github.com/InterDigitalInc/CompressAI). If you encounter any installation issues, please refer to their documentation first.

```bash
pip install -e .
pip install -e '.[dev]'
pip install Cython
pip install scikit-image
pip3 install git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI
pip install compressai
pip install pybind11
```

---

## Datasets

**Training:** [OpenImages v4](https://storage.googleapis.com/openimages/web/index.html) — 300,000 images randomly sampled from the full training set.

**Evaluation benchmarks:**
| Dataset | Description |
|---------|-------------|
| [Kodak](http://r0k.us/graphics/kodak/) | 24 lossless true color images |
| [Tecnick](https://testimages.org/) | High-resolution test images |
| [CLIC](https://www.compression.cc/) | Workshop on Learned Image Compression |

---

## Training


```bash

python train_czigzag.py --model czigzag --lambda <λ>

```


---

## Evaluation

```bash
# Run the evaluation entrypoint used in this repo:
python compressai/utils/eval_model/__main__.py \
  --dataset <path_to_image_folder_or_coco_root> \
  --path <path_to_checkpoint> \
  --architecture czigzag


```

Bit rates (BPP) reported for our model include the bits used to encode **both** the original image `x` and the low-resolution image `x_LR`.


---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{shen2024conditional,
  title     = {Learning-Based Conditional Image Compression},
  author    = {Shen, Tianma and Peng, Wen-Hsiao and Shih, Huang-Chia and Liu, Ying},
  booktitle = {Proc. IEEE International Symposium on Circuits and Systems (ISCAS)},
  year      = {2024}
}
```

---

## Acknowledgements

This work is supported in part by the **National Science Foundation** under Grant ECCS-2138635 and the **NVIDIA Academic Hardware Grant**.

The super-resolution network used in this work is [SwinIR](https://github.com/JingyunLiang/SwinIR), pretrained on the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset.
