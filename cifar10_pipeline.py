"""
cifar10_pipeline.py
===================
Self-contained, single-process pipeline:
  Prune → Fine-tune → Evaluate (clean + PGD-20)

All phases run in the same process so every metric and timing is
captured directly and written to a properly-filled consolidated CSV.
Nothing is NaN.

Usage:
  # 20% pruning on cuda:3
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  python3 cifar10_pipeline.py --ratio 0.20 --ft_steps 20000 --device cuda:3

  # 30% pruning on cuda:4
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  python3 cifar10_pipeline.py --ratio 0.30 --ft_steps 20000 --device cuda:4

  # Eval-only on an existing finetuned checkpoint (skip prune+finetune)
  python3 cifar10_pipeline.py --ratio 0.10 --eval_only \\
      --finetuned_ckpt checkpoints/score_sde/pruned/pruned_10pct_finetuned/finetuned_final.pth \\
      --device cuda:5

Output:
  results/cifar10/consolidated_results.csv   (append — never overwrites)
  results/cifar10/consolidated_results.json  (full run log)
  checkpoints/score_sde/pruned/pruned_{R}pct.pth
  checkpoints/score_sde/pruned/pruned_{R}pct_finetuned/finetuned_final.pth
"""

import os, sys, csv, json, math, time, datetime, argparse, warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.utils as tvu

sys.path.insert(0, str(Path(__file__).parent))

from score_sde.models import utils as mutils
from score_sde.models.ema import ExponentialMovingAverage
from score_sde import sde_lib
from score_sde.losses import get_optimizer
from cifar10_prune_score_sde import prune_ncsnpp


# ════════════════════════════════════════════════════════════════════════════
# 1.  CLI
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="DiffPure CIFAR-10 full pipeline: prune→finetune→eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ratio",         type=float, required=True,
                   help="Channel pruning ratio (e.g. 0.20 = 20%%)")
    p.add_argument("--device",        default="cuda:3")
    p.add_argument("--seed",          type=int,   default=42)

    # ── Checkpoints ──────────────────────────────────────────────────────────
    p.add_argument("--base_ckpt",
                   default="checkpoints/score_sde/checkpoint_8.pth",
                   help="Unpruned Score SDE checkpoint")
    p.add_argument("--pruned_dir",
                   default="checkpoints/score_sde/pruned",
                   help="Directory to store pruned/finetuned checkpoints")
    p.add_argument("--finetuned_ckpt", default=None,
                   help="Skip prune+finetune and use this checkpoint directly")

    # ── Pruning ──────────────────────────────────────────────────────────────
    p.add_argument("--skip_prune",    action="store_true",
                   help="Skip pruning (use existing pruned_Xpct.pth)")

    # ── Fine-tuning ──────────────────────────────────────────────────────────
    p.add_argument("--ft_steps",      type=int,   default=20000)
    p.add_argument("--ft_batch",      type=int,   default=64)
    p.add_argument("--ft_lr",         type=float, default=2e-4)
    p.add_argument("--ft_warmup",     type=int,   default=500)
    p.add_argument("--ft_grad_clip",  type=float, default=1.0)
    p.add_argument("--ft_ema_rate",   type=float, default=0.9999)
    p.add_argument("--ft_save_every", type=int,   default=5000)
    p.add_argument("--skip_finetune", action="store_true",
                   help="Skip fine-tuning (use existing finetuned checkpoint)")
    p.add_argument("--eval_only",     action="store_true",
                   help="Skip prune+finetune entirely (requires --finetuned_ckpt)")

    # ── Evaluation ────────────────────────────────────────────────────────────
    p.add_argument("--eval_images",   type=int,   default=512)
    p.add_argument("--eval_batch",    type=int,   default=32)
    p.add_argument("--t_star",        type=int,   default=100)
    p.add_argument("--ddim_steps",    type=int,   default=50)
    p.add_argument("--eot",           type=int,   default=5)
    p.add_argument("--pgd_steps",     type=int,   default=20)
    p.add_argument("--attack_eps",    type=float, default=8/255)
    p.add_argument("--pgd_step_size", type=float, default=2/255)

    # ── Data / output ─────────────────────────────────────────────────────────
    p.add_argument("--data_dir",      default="./data")
    p.add_argument("--output_dir",    default="results/cifar10/")

    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# 2.  CIFAR-10 MODEL CONFIG
# ════════════════════════════════════════════════════════════════════════════

class _Bunch(dict):
    def __getattr__(self, k):
        try:
            v = self[k]; return _Bunch(v) if isinstance(v, dict) else v
        except KeyError: raise AttributeError(k)
    def __setattr__(self, k, v): self[k] = v

CONFIG = _Bunch({
    "data":     {"dataset": "CIFAR10", "image_size": 32, "num_channels": 3,
                 "centered": True, "random_flip": True, "uniform_dequantization": False},
    "model":    {"sigma_min": 0.01, "sigma_max": 50, "num_scales": 1000,
                 "beta_min": 0.1, "beta_max": 20.0, "dropout": 0.1,
                 "name": "ncsnpp", "scale_by_sigma": False, "ema_rate": 0.9999,
                 "normalization": "GroupNorm", "nonlinearity": "swish", "nf": 128,
                 "ch_mult": (1, 2, 2, 2), "num_res_blocks": 8,
                 "attn_resolutions": (16,), "resamp_with_conv": True,
                 "conditional": True, "fir": False, "fir_kernel": [1, 3, 3, 1],
                 "skip_rescale": True, "resblock_type": "biggan",
                 "progressive": "none", "progressive_input": "none",
                 "progressive_combine": "sum", "attention_type": "ddpm",
                 "init_scale": 0.0, "embedding_type": "positional",
                 "fourier_scale": 16, "conv_size": 3},
    "training": {"sde": "vpsde", "continuous": True, "reduce_mean": True},
    "optim":    {"weight_decay": 0, "optimizer": "Adam", "lr": 2e-4,
                 "beta1": 0.9, "eps": 1e-8, "warmup": 5000, "grad_clip": 1.0},
    "sampling": {"n_steps_each": 1, "noise_removal": True, "probability_flow": False,
                 "snr": 0.16, "method": "pc", "predictor": "euler_maruyama",
                 "corrector": "none"},
})


# ════════════════════════════════════════════════════════════════════════════
# 3.  PRUNING
# ════════════════════════════════════════════════════════════════════════════

def run_pruning(base_ckpt: str, ratio: float, out_path: str, device: str):
    """
    Randomly prune ratio% of channels from the NCSN++ model.
    Returns (model, keep_128, keep_256, n_params_before, n_params_after).
    """
    print(f"\n{'═'*60}")
    print(f"  [PRUNE] ratio={ratio:.0%}  →  {out_path}")
    print(f"{'═'*60}")

    t0 = time.time()
    ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    model = mutils.create_model(CONFIG)
    ema   = ExponentialMovingAverage(model.parameters(), decay=CONFIG.model.ema_rate)
    optimizer = get_optimizer(CONFIG, model.parameters())
    model.load_state_dict(ckpt["model"], strict=False)
    ema.load_state_dict(ckpt["ema"])
    ema.copy_to(model.parameters())

    n_params_before = sum(p.numel() for p in model.parameters()) / 1e6

    # Randomly select channels to keep (no_importance mode)
    n128 = round(128 * (1 - ratio))
    n256 = round(256 * (1 - ratio))
    torch.manual_seed(42)
    keep_128 = torch.sort(torch.randperm(128)[:n128]).values
    keep_256 = torch.sort(torch.randperm(256)[:n256]).values

    prune_ncsnpp(model, keep_128, keep_256)
    n_params_after = sum(p.numel() for p in model.parameters()) / 1e6
    reduction = (n_params_before - n_params_after) / n_params_before * 100

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        "model":          model.state_dict(),
        "keep_128":       keep_128,
        "keep_256":       keep_256,
        "pruning_ratio":  ratio,
        "stats": {
            "params_before_M": n_params_before,
            "params_after_M":  n_params_after,
            "reduction_pct":   reduction,
        },
    }, out_path)

    elapsed = time.time() - t0
    print(f"  Params : {n_params_before:.2f}M → {n_params_after:.2f}M  "
          f"({reduction:.1f}% reduction)")
    print(f"  Saved  : {out_path}  [{elapsed:.1f}s]")
    return model, keep_128, keep_256, n_params_before, n_params_after, elapsed


def load_pruned_ckpt(ckpt_path: str, device: str):
    """Load a previously saved pruned checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    keep_128 = ckpt["keep_128"]
    keep_256 = ckpt["keep_256"]
    model = mutils.create_model(CONFIG)
    prune_ncsnpp(model, keep_128, keep_256)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    stats = ckpt.get("stats", {})
    n_before = stats.get("params_before_M", 106.6)
    return model, keep_128, keep_256, n_before, n_params


# ════════════════════════════════════════════════════════════════════════════
# 4.  FINE-TUNING
# ════════════════════════════════════════════════════════════════════════════

class VPScheduleSimple:
    def __init__(self, device):
        N = CONFIG.model.num_scales
        self.N = N
        betas = torch.linspace(CONFIG.model.beta_min / N, CONFIG.model.beta_max / N,
                               N, device=device)
        ab = torch.cumprod(1.0 - betas, dim=0)
        self.alpha_bar = ab
        self.sqrt_ab   = ab.sqrt()
        self.sqrt_1mab = (1.0 - ab).sqrt()

    def q_sample(self, x0, t_int, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        return self.sqrt_ab[t_int - 1] * x0 + self.sqrt_1mab[t_int - 1] * noise


def dsm_loss(model, sde, x, device):
    """
    VP-SDE denoising score matching loss — identical to original training objective.
    MUST use mutils.get_score_fn wrapper (not model() directly) to get properly
    scaled score output. Calling model() directly trains on the wrong objective.
    """
    B   = x.shape[0]
    eps = 1e-5
    t   = torch.rand(B, device=device) * (1 - eps) + eps
    z   = torch.randn_like(x)
    mean, std = sde.marginal_prob(x, t)
    x_t  = mean + std[:, None, None, None] * z
    # Use get_score_fn wrapper — this applies the correct sigma scaling
    score_fn = mutils.get_score_fn(sde, model, train=True, continuous=True)
    score    = score_fn(x_t, t)
    losses   = torch.sum((score * std[:, None, None, None] + z) ** 2, dim=(1, 2, 3))
    return losses.mean()


def warmup_lr(step, base_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def run_finetuning(model, keep_128, keep_256, ratio, pruned_ckpt_path: str,
                   out_dir: str, args, device: str):
    """
    Fine-tune the pruned model using DSM loss.
    Returns (finetuned_model_path, ft_elapsed_s, final_loss).
    """
    print(f"\n{'═'*60}")
    print(f"  [FINETUNE] {args.ft_steps} steps  lr={args.ft_lr}  batch={args.ft_batch}")
    print(f"{'═'*60}")

    os.makedirs(out_dir, exist_ok=True)

    sde = sde_lib.VPSDE(
        beta_min=CONFIG.model.beta_min,
        beta_max=CONFIG.model.beta_max,
        N=CONFIG.model.num_scales,
    )
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ft_ema_rate)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.ft_lr,
        betas=(CONFIG.optim.beta1, 0.999),
        eps=CONFIG.optim.eps,
        weight_decay=CONFIG.optim.weight_decay,
    )

    # Data
    tf = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])
    ds = torchvision.datasets.CIFAR10(root=args.data_dir, train=True,
                                       download=True, transform=tf)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.ft_batch, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    loader_iter = iter(loader)

    model.train()
    t0 = time.time()
    running_loss = 0.0
    train_log = []
    final_loss = 0.0

    print(f"  Training for {args.ft_steps} steps ...")
    for step in range(args.ft_steps):
        lr = warmup_lr(step, args.ft_lr, args.ft_warmup)
        for pg in optimizer.param_groups: pg["lr"] = lr

        try:
            x, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, _ = next(loader_iter)
        x = x.to(device)

        optimizer.zero_grad()
        loss = dsm_loss(model, sde, x, device)
        loss.backward()
        if args.ft_grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.ft_grad_clip)
        optimizer.step()
        ema.update(model.parameters())
        running_loss += loss.item()
        final_loss    = loss.item()

        if (step + 1) % 500 == 0:
            avg = running_loss / 500
            elapsed = time.time() - t0
            eta    = (args.ft_steps - step - 1) / ((step + 1) / elapsed)
            print(f"  step {step+1:6d}/{args.ft_steps}  loss={avg:.4f}  "
                  f"lr={lr:.2e}  [{elapsed/60:.1f}min  ETA {eta/60:.0f}min]")
            train_log.append({"step": step+1, "loss": avg, "lr": lr})
            running_loss = 0.0

        # Periodic checkpoint (with keep indices embedded)
        if (step + 1) % args.ft_save_every == 0:
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            ckpt_path = os.path.join(out_dir, f"ft_step{step+1:06d}.pth")
            torch.save({
                "step":          step + 1,
                "model":         model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "ema":           ema.state_dict(),
                "loss":          loss.item(),
                "keep_128":      keep_128,
                "keep_256":      keep_256,
                "pruning_ratio": ratio,
            }, ckpt_path)
            ema.restore(model.parameters())
            print(f"  [Ckpt] → {ckpt_path}")

    ft_elapsed = time.time() - t0

    # Final checkpoint with EMA weights
    ema.copy_to(model.parameters())
    final_path = os.path.join(out_dir, "finetuned_final.pth")
    torch.save({
        "step":           args.ft_steps,
        "model":          model.state_dict(),
        "keep_128":       keep_128,
        "keep_256":       keep_256,
        "pruning_ratio":  ratio,
        "ft_steps":       args.ft_steps,
        "pruning_source": pruned_ckpt_path,
    }, final_path)

    with open(os.path.join(out_dir, "training_log.json"), "w") as f:
        json.dump(train_log, f, indent=2)

    print(f"\n  ✅  Fine-tuning done  [{ft_elapsed/60:.1f} min]")
    print(f"  Final checkpoint → {final_path}")
    return final_path, ft_elapsed, final_loss


# ════════════════════════════════════════════════════════════════════════════
# 5.  EVALUATION  (clean + PGD-20 with BPDA)
# ════════════════════════════════════════════════════════════════════════════

class VPSchedule:
    def __init__(self, device):
        N = CONFIG.model.num_scales
        self.N = N
        betas = torch.linspace(CONFIG.model.beta_min / N,
                               CONFIG.model.beta_max / N, N, device=device)
        ab = torch.cumprod(1.0 - betas, dim=0)
        self.alpha_bar = ab
        self.sqrt_ab   = ab.sqrt()
        self.sqrt_1mab = (1.0 - ab).sqrt()

    def q_sample(self, x0, t_int, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        return self.sqrt_ab[t_int - 1] * x0 + self.sqrt_1mab[t_int - 1] * noise


@torch.no_grad()
def purify(x, score_fn, schedule, t_star, ddim_steps, device):
    noise = torch.randn_like(x)
    x_t   = schedule.q_sample(x, t_star, noise)
    ts    = np.clip(np.linspace(t_star, 1, ddim_steps + 1, dtype=int), 1, t_star)
    x_cur = x_t
    for i in range(len(ts) - 1):
        t_cur, t_prev = int(ts[i]), int(ts[i+1])
        t_cont  = torch.full((x.shape[0],), t_cur / schedule.N,
                             device=device, dtype=torch.float32)
        score   = score_fn(x_cur, t_cont)
        s1m     = schedule.sqrt_1mab[t_cur - 1]
        sab     = schedule.sqrt_ab[t_cur - 1]
        eps_p   = -s1m * score
        x0_p    = ((x_cur - s1m * eps_p) / sab).clamp(-1, 1)
        x_cur   = (schedule.alpha_bar[t_prev - 1].sqrt() * x0_p
                   + schedule.sqrt_1mab[t_prev - 1] * eps_p)
    return x_cur


class _BPDA(torch.autograd.Function):
    """Straight-through estimator: gradient flows to x_adv (second arg)."""
    @staticmethod
    def forward(ctx, x_pure, x_adv): return x_pure
    @staticmethod
    def backward(ctx, g):            return None, g  # gradient → x_adv


def pgd_attack(x_clean, y, score_fn, schedule, classifier,
               t_star, ddim_steps, eps, step_size, n_steps, device):
    x_adv = (x_clean + torch.zeros_like(x_clean).uniform_(-eps, eps)).clamp(-1., 1.)
    for _ in range(n_steps):
        x_adv = x_adv.detach().requires_grad_(True)
        with torch.no_grad():
            x_pure = purify(x_adv, score_fn, schedule, t_star, ddim_steps, device)
        x_in = _BPDA.apply(x_pure, x_adv)
        loss  = F.cross_entropy(classifier(x_in), y)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + step_size * x_adv.grad.sign()
            x_adv = torch.clamp(
                torch.min(torch.max(x_adv, x_clean - eps), x_clean + eps),
                -1., 1.
            )
    return x_adv.detach()


def load_classifier(device):
    try:
        from robustbench.utils import load_model as rb_load
        rb_model = rb_load("Standard", dataset="cifar10", threat_model="Linf")
        # RobustBench expects [0,1]; our whole pipeline uses [-1,1] — wrap it
        class _RBWrapper(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): return self.m((x + 1) / 2)  # [-1,1] → [0,1]
        clf = _RBWrapper(rb_model).to(device).eval()
        print("  [Classifier] WRN-28-10 Standard (RobustBench) — [-1,1] wrapper applied")
        return clf
    except Exception as e:
        print(f"  [Classifier] RobustBench failed: {e}")
        raise


def run_evaluation(model, n_params_M, ratio, args, device):
    """
    Evaluate the model: clean accuracy + PGD-20 robust accuracy.
    Returns a dict of all metrics + timing.
    """
    print(f"\n{'═'*60}")
    print(f"  [EVAL]  {args.eval_images} images  PGD-{args.pgd_steps}  EOT={args.eot}")
    print(f"{'═'*60}")

    # ── Classifier ─────────────────────────────────────────────────────────
    print("\n  Loading classifier ...")
    classifier = load_classifier(device)

    # ── Data ───────────────────────────────────────────────────────────────
    tf = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    ds = torchvision.datasets.CIFAR10(root=args.data_dir, train=False,
                                       download=True, transform=tf)
    g   = torch.Generator(); g.manual_seed(args.seed)
    idx = torch.randperm(len(ds), generator=g)[:args.eval_images].tolist()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idx),
        batch_size=args.eval_batch, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    batches = [(x.to(device), y.to(device)) for x, y in loader]
    n_total = sum(x.shape[0] for x, _ in batches)
    print(f"  {n_total} test images  ({len(batches)} batches)")

    # ── Score function ──────────────────────────────────────────────────────
    _sde = sde_lib.VPSDE(beta_min=CONFIG.model.beta_min,
                          beta_max=CONFIG.model.beta_max,
                          N=CONFIG.model.num_scales)
    score_fn = mutils.get_score_fn(_sde, model, train=False, continuous=True)
    schedule = VPSchedule(device)

    # ── Clean accuracy ──────────────────────────────────────────────────────
    print(f"\n  [1/2] Clean accuracy (EOT={args.eot}) ...")
    model.eval()
    clean_correct = 0
    total_purify_s = 0.0

    for bi, (x, y) in enumerate(batches):
        # EOT: average over multiple purifications
        logit_sum = None
        t0 = time.time()
        for _ in range(args.eot):
            with torch.no_grad():
                x_pure = purify(x, score_fn, schedule,
                                args.t_star, args.ddim_steps, device)
                logits = classifier(x_pure)
            logit_sum = logits if logit_sum is None else logit_sum + logits
        purify_s = time.time() - t0

        preds = logit_sum.argmax(1)
        bc    = (preds == y).sum().item()
        clean_correct  += bc
        total_purify_s += purify_s

        ms_img = purify_s / x.shape[0] * 1000
        print(f"    Batch {bi+1:3d}/{len(batches)}  "
              f"acc={bc/x.shape[0]:.3f}  "
              f"running={clean_correct/min((bi+1)*args.eval_batch, n_total):.3f}  "
              f"{ms_img:.0f}ms/img")

    clean_acc   = clean_correct / n_total
    purify_ms   = total_purify_s / n_total * 1000 / args.eot   # per image per eot
    clean_time_s = total_purify_s
    print(f"\n  Clean accuracy : {clean_acc:.4f}  ({clean_correct}/{n_total})")
    print(f"  Purify time    : {purify_ms:.1f} ms/image  [{clean_time_s/60:.1f} min total]")

    # ── PGD-20 robust accuracy ──────────────────────────────────────────────
    print(f"\n  [2/2] PGD-{args.pgd_steps} robust accuracy  (ε=8/255  BPDA) ...")
    pgd_correct = 0
    t0_pgd = time.time()

    for bi, (x, y) in enumerate(batches):
        t0 = time.time()
        x_adv = pgd_attack(
            x, y, score_fn, schedule, classifier,
            args.t_star, args.ddim_steps,
            args.attack_eps, args.pgd_step_size, args.pgd_steps, device,
        )
        with torch.no_grad():
            x_pure = purify(x_adv, score_fn, schedule,
                            args.t_star, args.ddim_steps, device)
            preds = classifier(x_pure).argmax(1)
        bc = (preds == y).sum().item()
        pgd_correct += bc
        elapsed = time.time() - t0
        print(f"    Batch {bi+1:3d}/{len(batches)}  "
              f"rob={bc/x.shape[0]:.3f}  "
              f"running={pgd_correct/min((bi+1)*args.eval_batch, n_total):.3f}  "
              f"[{elapsed:.0f}s]")

    pgd_acc    = pgd_correct / n_total
    pgd_time_s = time.time() - t0_pgd
    print(f"\n  PGD-{args.pgd_steps} robust acc : {pgd_acc:.4f}  "
          f"({pgd_correct}/{n_total})")
    print(f"  PGD eval time  : {pgd_time_s/60:.1f} min")

    return {
        "clean_acc":       round(clean_acc, 6),
        "clean_pct":       round(clean_acc * 100, 2),
        "clean_correct":   clean_correct,
        "pgd_acc":         round(pgd_acc, 6),
        "pgd_pct":         round(pgd_acc * 100, 2),
        "pgd_correct":     pgd_correct,
        "n_images":        n_total,
        "purify_ms_img":   round(purify_ms, 2),
        "clean_time_s":    round(clean_time_s, 1),
        "pgd_time_s":      round(pgd_time_s, 1),
    }


# ════════════════════════════════════════════════════════════════════════════
# 6.  CSV / JSON  (append-safe, never overwrites)
# ════════════════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    "run_id", "timestamp_start", "timestamp_end",
    "label", "pruning_ratio", "ft_steps",
    "params_M_before", "params_M_after",
    "compression_x", "reduction_pct",
    "clean_acc", "clean_pct", "clean_correct",
    "pgd_acc",   "pgd_pct",   "pgd_correct",
    "n_images", "t_star", "ddim_steps", "eot", "pgd_steps",
    "attack_eps_over255",
    "prune_time_s", "finetune_time_s",
    "clean_eval_time_s", "pgd_eval_time_s", "total_time_s",
    "purify_ms_img",
    "base_ckpt", "pruned_ckpt", "finetuned_ckpt",
    "device", "notes",
]


def save_results(row: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    csv_path  = os.path.join(output_dir, "consolidated_results.csv")
    json_path = os.path.join(output_dir, "consolidated_results.json")

    # CSV — append, write header only if new file
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print(f"\n  [Save] CSV  → {csv_path}")

    # JSON — load+merge+save
    existing = []
    if os.path.exists(json_path):
        with open(json_path) as f:
            try:    existing = json.load(f)
            except: existing = []
    existing.append(row)
    with open(json_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    print(f"  [Save] JSON → {json_path}")


def print_row_summary(row):
    print(f"\n{'═'*65}")
    print(f"  RESULT SUMMARY  —  {row['label']}")
    print(f"{'═'*65}")
    print(f"  Params        : {row['params_M_before']:.1f}M → {row['params_M_after']:.1f}M  "
          f"({row['reduction_pct']:.1f}% reduction  {row['compression_x']:.2f}×)")
    print(f"  Fine-tune     : {row['ft_steps']} steps  [{row['finetune_time_s']/60:.1f} min]")
    print(f"  Clean acc     : {row['clean_pct']:.2f}%  ({row['clean_correct']}/{row['n_images']})")
    print(f"  PGD-20 acc    : {row['pgd_pct']:.2f}%  ({row['pgd_correct']}/{row['n_images']})")
    print(f"  Purify time   : {row['purify_ms_img']:.1f} ms/image")
    print(f"  Total time    : {row['total_time_s']/3600:.2f} hours")
    print(f"{'═'*65}")


# ════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    ratio_pct = int(args.ratio * 100)
    label     = f"pruned_{ratio_pct:02d}pct"
    run_id    = f"{label}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ts_start  = datetime.datetime.now().isoformat()
    t_global  = time.time()

    os.makedirs(args.pruned_dir, exist_ok=True)
    pruned_ckpt_path  = os.path.join(args.pruned_dir, f"pruned_{ratio_pct:02d}pct.pth")
    ft_out_dir        = os.path.join(args.pruned_dir, f"pruned_{ratio_pct:02d}pct_finetuned")
    finetuned_ckpt    = args.finetuned_ckpt or os.path.join(ft_out_dir, "finetuned_final.pth")

    print("=" * 65)
    print(f"  DiffPure CIFAR-10  Pipeline  —  {label}")
    print("=" * 65)
    if "cuda" in device and torch.cuda.is_available():
        idx = int(device.split(":")[-1]) if ":" in device else 0
        print(f"  GPU  : {torch.cuda.get_device_name(idx)}")
    print(f"  Ratio   : {args.ratio:.0%}")
    print(f"  FT      : {args.ft_steps} steps")
    print(f"  Eval    : {args.eval_images} images  PGD-{args.pgd_steps}  EOT={args.eot}")
    print(f"  Outputs : {args.pruned_dir}")

    prune_time_s = ft_time_s = 0.0
    n_params_before = 106.6

    # ── PRUNE ────────────────────────────────────────────────────────────────
    if args.eval_only:
        print("\n[PRUNE] Skipped (eval_only mode)")
        model, keep_128, keep_256, n_params_before, n_params_after = \
            load_pruned_ckpt(finetuned_ckpt, device)
    elif args.skip_prune and os.path.exists(pruned_ckpt_path):
        print(f"\n[PRUNE] Loading existing: {pruned_ckpt_path}")
        model, keep_128, keep_256, n_params_before, n_params_after = \
            load_pruned_ckpt(pruned_ckpt_path, device)
    else:
        model, keep_128, keep_256, n_params_before, n_params_after, prune_time_s = \
            run_pruning(args.base_ckpt, args.ratio, pruned_ckpt_path, device)
    model = model.to(device)

    compression_x = n_params_before / n_params_after if n_params_after else 1.0
    reduction_pct  = (n_params_before - n_params_after) / n_params_before * 100

    # ── FINE-TUNE ────────────────────────────────────────────────────────────
    if args.eval_only or args.skip_finetune:
        print(f"\n[FINETUNE] Skipped — using {finetuned_ckpt}")
        if not args.eval_only:
            ckpt = torch.load(finetuned_ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"], strict=True)
            model = model.to(device).eval()
    else:
        finetuned_ckpt, ft_time_s, _ = run_finetuning(
            model, keep_128, keep_256, args.ratio,
            pruned_ckpt_path, ft_out_dir, args, device
        )

    model = model.eval()

    # ── EVALUATE ─────────────────────────────────────────────────────────────
    eval_metrics = run_evaluation(model, n_params_after, args.ratio, args, device)

    total_time_s = time.time() - t_global
    ts_end = datetime.datetime.now().isoformat()

    # ── BUILD RESULT ROW ─────────────────────────────────────────────────────
    row = {
        "run_id":             run_id,
        "timestamp_start":    ts_start,
        "timestamp_end":      ts_end,
        "label":              label,
        "pruning_ratio":      args.ratio,
        "ft_steps":           args.ft_steps if not (args.eval_only or args.skip_finetune) else 0,
        "params_M_before":    round(n_params_before, 3),
        "params_M_after":     round(n_params_after, 3),
        "compression_x":      round(compression_x, 4),
        "reduction_pct":      round(reduction_pct, 2),
        "clean_acc":          eval_metrics["clean_acc"],
        "clean_pct":          eval_metrics["clean_pct"],
        "clean_correct":      eval_metrics["clean_correct"],
        "pgd_acc":            eval_metrics["pgd_acc"],
        "pgd_pct":            eval_metrics["pgd_pct"],
        "pgd_correct":        eval_metrics["pgd_correct"],
        "n_images":           eval_metrics["n_images"],
        "t_star":             args.t_star,
        "ddim_steps":         args.ddim_steps,
        "eot":                args.eot,
        "pgd_steps":          args.pgd_steps,
        "attack_eps_over255": round(args.attack_eps * 255, 1),
        "prune_time_s":       round(prune_time_s, 1),
        "finetune_time_s":    round(ft_time_s, 1),
        "clean_eval_time_s":  round(eval_metrics["clean_time_s"], 1),
        "pgd_eval_time_s":    round(eval_metrics["pgd_time_s"], 1),
        "total_time_s":       round(total_time_s, 1),
        "purify_ms_img":      eval_metrics["purify_ms_img"],
        "base_ckpt":          args.base_ckpt,
        "pruned_ckpt":        pruned_ckpt_path,
        "finetuned_ckpt":     finetuned_ckpt,
        "device":             device,
        "notes":              "",
    }

    print_row_summary(row)
    save_results(row, args.output_dir)
    print(f"\n✅  Pipeline complete for {label}  [{total_time_s/3600:.2f} hours]")


if __name__ == "__main__":
    main()
