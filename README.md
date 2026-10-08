---
language: 
  - vi
license: other
library_name: pytorch
tags:
  - vietnamese
  - causal-lm
  - language-model
  - custom-architecture
datasets:
  - hoangduy02071997/tinyDTVi-dataset
---

# TinyDTVi-500M: Pretraining a Compact Vietnamese Language Model from Scratch
TinyDTVi is a ~500M parameter causal language model pretrained entirely from scratch on a single consumer-grade GPU (NVIDIA RTX 3080 12GB). 
This repository contains the core codebase for reproducing the model architecture, training loop, and data preparation pipeline.

**Paper:** [arXiv link coming soon]
**Model Weights (283,200 steps):** [hoangduy02071997/tinyDTVi](https://huggingface.co/hoangduy02071997/tinyDTVi)
**Pretraining Dataset:** [hoangduy02071997/tinyDTVi-dataset](https://huggingface.co/datasets/hoangduy02071997/tinyDTVi-dataset)

## 📌 Model Details & Reproducibility
To ensure exact reproducibility of the Phase-1 pretraining results reported in our paper, we provide the following strict metadata for this checkpoint:
```yaml
checkpoint_type: pretraining_checkpoint
alignment: none
training_steps: 283200
tokens_seen: 23199744000
precision: bfloat16
architecture:
  layers: 24
  hidden_size: 1088
  heads: 16
  context_length: 1024
vocab_size: 50304
```

## 📊 Dataset & Preprocessing
The original source corpora (comprising Wikipedia, VNews, and CulturaX-vi) contained **over 200GB** of raw/uncompressed text. After aggressive character-level whitelisting, document truncation, whitespace normalization, and bounded prefix-based deduplication (15M unique prefixes), the resulting high-quality training corpus yielded the following final metrics:
- **Documents:** ~28.6 million
- **Cleaned Size:** ~134 GB
- **Tokens:** ~28 Billion (using our custom 50k-token BPE tokenizer)

## Architecture Highlights
![TinyDTVi Architecture](architecture.gif)

- **Base Architecture:** Decoder-only Transformer (nanoGPT-inspired)
- **Parameters:** ~509 Million
- **Layers:** 24 | **Heads:** 16 | **Embed Dim:** 1088
- **Context Length:** 1024 tokens
- **Modernizations:**
  - RMSNorm (Pre-Layer Normalization)
  - SwiGLU Activations
  - Rotary Position Embeddings (RoPE)
  - No biases in linear layers
  - Tied embeddings

## Repository Structure
- `tinyDTVi_model.py`: Model architecture definition (Transformer, Blocks, Attention with RoPE, SwiGLU).
- `train_tinyDTVi.py`: The main pretraining loop featuring 8-bit AdamW, Gradient Checkpointing, and bfloat16 mixed precision.
- `cleanup_v2.py` & `clean_utils.py`: The aggressive whitelist-based data filtering and bounded prefix-based deduplication pipeline used to process over 200GB of raw text.
- `prepare_data.py`: Script to tokenize the cleaned 134GB text corpus into binary format (`train.bin`, `val.bin`) for training.
- `train_tokenizer.py`: Script to train our custom highly-compressed 50,304-token Vietnamese Byte-Pair Encoding (BPE) tokenizer.
- `benchmark_tokenizer.py`: Script evaluating tokenizer compression efficiency against multilingual models (Qwen2.5, Llama-3, etc.).
- `eval_ppl.py`: Validation script to measure the ~10.45 Perplexity (PPL) on the held-out Wikipedia data stream.
- `benchmark_bpc.py`: Script to reproduce the 42.29 PPL and 1.3889 Bits-Per-Character (BPC) on the raw XQuAD dataset as reported in the paper.
- `inference.py`: Script for qualitative text generation (Top-k sampling).

## Quick Start (Inference)
First, install the requirements:
```bash
pip install -r requirements.txt
```

To run text generation using the pretrained model:
```bash
# Note: Ensure you have downloaded the weights from Hugging Face
python inference.py
```

## Reproducing Evaluation Metrics
To explicitly reproduce the exact metrics reported in Section 6.1 of the paper:

**1. XQuAD (Out-of-Distribution, Raw Text)**
```bash
python benchmark_bpc.py
# Expected Output: Validation Loss ~3.7445, PPL ~42.29, BPC ~1.3889
```

**2. Wikipedia (In-Domain, Whitelisted Text)**
```bash
python eval_ppl.py
# Expected Output: PPL ~10.45
```

## Training & Environment Setup
We successfully trained this model on a single RTX 3080 (12GB VRAM). The training heavily relies on memory-bound optimizations. To reproduce the exact training and evaluation environment as described in Section 5.1 of our paper, use Docker with PyTorch 2.1.2:

```bash
# Build the Docker image
docker build -t tinydtvi .

# Launch the container with GPU support and 16GB shared memory
docker run --gpus all -it --ipc=host --shm-size=16g -v $(pwd):/workspace -w /workspace tinydtvi
```

*Note: The reported training run used `bitsandbytes` 8-bit AdamW to fit within 12GB VRAM. The released `train_tinyDTVi.py` script includes a standard 32-bit AdamW fallback for environments where `bitsandbytes` cannot be initialized.*

For hardware configurations and ablation studies, please refer to Section 5.1 of our paper.

## License & Use Restrictions
While the **source code** in this repository is licensed under the MIT License, the **model weights** and **training dataset** are subject to the licenses and terms of the respective upstream datasets (Wikipedia, CulturaX, and VNews). 
Because the pretraining corpus contains web-crawled data, it may contain trace amounts of PII, adult content, or bias. Therefore, TinyDTVi is released **strictly for research purposes only**. It should not be deployed in production environments without rigorous data scrubbing, safety alignment, and independent legal review.

## Citation
If you use this codebase or model in your research, please cite our paper:
```bibtex
@article{tinyDTVi2026,
  title={TinyDTVi: Pretraining a Compact Vietnamese Language Model from Scratch Under Consumer-Grade Constraints},
  author={Duy Hoang and Duy Tam Hoang Nguyen},
  year={2026}
}
```
