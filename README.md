# An Empirical Evaluation of Pruning and Caching under Adversarial Attacks

This repository provides the implementation for reproducing the pruning, fine-tuning, robustness evaluation, and DeepCache-based caching experiments for CIFAR-10 diffusion purifier models.

The code evaluates how **structural pruning** and **inference-time score caching** affect the robustness-efficiency trade-off of a **Score SDE / NCSN++ diffusion purifier** under adversarial attacks.

---

## Overview

This repository contains three main experiment scripts:

1. **`cifar10_pipeline.py`**  
   Runs the full pruning and fine-tuning pipeline.

2. **`cifar10_final_eval.py`**  
   Performs the final clean and adversarial robustness evaluation.

3. **`cifar10_deepcache_sweep.py`**  
   Runs DeepCache-based inference-time caching experiments.

---

## Repository Structure

```text
.
├── cifar10_pipeline.py
├── cifar10_final_eval.py
├── cifar10_deepcache_sweep.py
├── setup_score_sde.sh
├── checkpoints/
├── results/
└── README.md
```

---

## Score SDE / NCSN++ Backbone

We use the public **Score SDE / NCSN++** implementation of Song et al. as the baseline diffusion purifier backbone.

The pruning, fine-tuning, robustness evaluation, and caching analysis are built on top of this diffusion purifier model.

To keep third-party code attribution clear, this repository does not duplicate the full Score SDE source tree. Please run:

```bash
bash setup_score_sde.sh
```

This script prepares the required Score SDE components used by the experiments.

---

## Experimental Pipeline

The complete experimental workflow consists of three steps:

1. Structural pruning and fine-tuning
2. Final robustness evaluation
3. DeepCache-based caching evaluation

---

## Step 1: Structural Pruning and Fine-Tuning

The script **`cifar10_pipeline.py`** performs the full pruning pipeline.

It:

- loads the CIFAR-10 Score SDE / NCSN++ diffusion purifier,
- structurally prunes the model at a specified pruning ratio,
- fine-tunes the pruned purifier,
- evaluates the pruned model on randomly selected CIFAR-10 test images.

Example command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 cifar10_pipeline.py \
  --ratio 0.25 \
  --ft_steps 20000 \
  --device cuda:0
```

Here, `--ratio 0.25` means that 25% structural pruning is applied.

---

## Step 2: Final Robustness Evaluation

The script **`cifar10_final_eval.py`** runs the final evaluation using clean accuracy and PGD-based adversarial accuracy.

It reports:

- clean accuracy,
- purified clean accuracy,
- PGD adversarial accuracy,
- purified PGD adversarial accuracy.

Example command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 cifar10_final_eval.py \
  --ratio 0.25 \
  --eval_images 512 \
  --eval_eot 5 \
  --attack_eot 5 \
  --pgd_steps 20 \
  --device cuda:0
```

Important arguments:

```text
--ratio         Pruning ratio of the evaluated model.
--eval_images   Number of CIFAR-10 test images used for evaluation.
--eval_eot      Number of EOT samples used during evaluation.
--attack_eot    Number of EOT samples used during adversarial attack generation.
--pgd_steps     Number of PGD attack steps.
--device        CUDA device used for evaluation.
```

---

## Step 3: DeepCache-Based Caching Evaluation

The script **`cifar10_deepcache_sweep.py`** evaluates inference-time score caching using DeepCache-style reuse of U-Net score predictions.

The goal is to measure how reducing the number of full U-Net evaluations affects:

- inference latency,
- clean accuracy,
- adversarial robustness,
- robustness-efficiency trade-off.

Example command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 cifar10_deepcache_sweep.py \
  --checkpoints 25pct \
  --eval_images 512 \
  --attack_eot 5 \
  --device cuda:0
```

Typical cache intervals include:

```text
K = 1, 2, 3, 5
```

A larger cache interval reduces the number of full U-Net calls, but it may also affect purification quality and adversarial robustness.

---

## Pruning Ratios

The experiments can be run with different structural pruning ratios, for example:

```text
0.10
0.20
0.25
0.30
0.50
```

Example for 30% pruning:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 cifar10_pipeline.py \
  --ratio 0.30 \
  --ft_steps 20000 \
  --device cuda:0
```

---

## Evaluation Setting

The main evaluation setting is:

```text
Dataset: CIFAR-10
Purifier: Score SDE / NCSN++
Attack: PGD
PGD steps: 20
Evaluation images: 512
Evaluation EOT: 5
Attack EOT: 5
```

The evaluation measures the trade-off between:

- clean accuracy,
- adversarial accuracy,
- purified adversarial accuracy,
- model size,
- inference latency,
- number of U-Net calls.

---

## Output

The scripts save logs and results under the **`results/`** directory.

Depending on the script, the output may include:

- pruning ratio,
- checkpoint path,
- clean accuracy,
- adversarial accuracy,
- purified clean accuracy,
- purified adversarial accuracy,
- latency per image,
- cache interval,
- number of U-Net calls.

---

## Notes

- Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce CUDA memory fragmentation.
- Make sure the CIFAR-10 dataset and required Score SDE checkpoints are available before running the experiments.
- For fair comparison, use the same number of evaluation images, PGD steps, and EOT settings across all experiments.
- DeepCache latency may vary depending on GPU hardware and system configuration.
- This repository focuses on empirical evaluation of pruning and caching for diffusion purification under adversarial attacks.

---


Instead, you can simply write:

```markdown
## Citation

Citation information will be added.

---

## Acknowledgements

This work builds on the public Score SDE / NCSN++ implementation by Song et al.

We thank the authors of the original Score SDE framework for releasing their code.