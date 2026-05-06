"""
cifar10_deepcache_sweep.py
==========================
Sweep all finetuned Score SDE checkpoints over multiple DeepCache K values.
This is a pure inference-time experiment — no pruning or fine-tuning needed.

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
        description="DeepCache sweep across all DiffPure CIFAR-10 checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoints", nargs="+",
                   default=["baseline", "10pct", "20pct", "25pct", "30pct", "50pct"],
                   help="Which model checkpoints to evaluate. "
                        "Options: baseline, 10pct, 20pct, 25pct, 30pct, 50pct")
    p.add_argument("--k_values", nargs="+", type=int, default=[1, 2, 3, 5],
                   help="DeepCache K values to sweep (K=1 = no caching)")
    p.add_argument("--baseline_ckpt",
                   default="checkpoints/score_sde/checkpoint_8.pth")
    p.add_argument("--pruned_dir",
                   default="checkpoints/score_sde/pruned")

    # Eval settings
    p.add_argument("--eval_images",   type=int,   default=512)
    p.add_argument("--eval_batch",    type=int,   default=32)
    p.add_argument("--t_star",        type=int,   default=100)
    p.add_argument("--ddim_steps",    type=int,   default=50)
    p.add_argument("--eot",           type=int,   default=5,
                   help="EOT for clean accuracy (averages purifications)")
    p.add_argument("--attack_eot",    type=int,   default=5,
                   help="EOT for PGD attack gradient (strong attack)")
    p.add_argument("--pgd_steps",     type=int,   default=20)
    p.add_argument("--attack_eps",    type=float, default=8/255)
    p.add_argument("--pgd_step_size", type=float, default=2/255)
    p.add_argument("--only_clean",    action="store_true",
                   help="Skip PGD evaluation — only measure clean acc + timing")

    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--data_dir",    default="./data")
    p.add_argument("--output_dir",  default="results/cifar10/")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# 2.  CIFAR-10 CONFIG
# ════════════════════════════════════════════════════════════════════════════

class _Bunch(dict):
    def __getattr__(self, k):
        try:
            v = self[k]; return _Bunch(v) if isinstance(v, dict) else v
        except KeyError: raise AttributeError(k)
    def __setattr__(self, k, v): self[k] = v

CONFIG = _Bunch({
    "data":     {"dataset": "CIFAR10", "image_size": 32, "num_channels": 3,
                 "centered": True, "random_flip": True,
                 "uniform_dequantization": False},
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
    "sampling": {"n_steps_each": 1, "noise_removal": True,
                 "probability_flow": False, "snr": 0.16,
                 "method": "pc", "predictor": "euler_maruyama",
                 "corrector": "none"},
})


# ════════════════════════════════════════════════════════════════════════════
# 3.  CHECKPOINT REGISTRY
# ════════════════════════════════════════════════════════════════════════════

def build_checkpoint_registry(args) -> dict:
    """
    Returns: {name: {"ckpt_path": str, "is_pruned": bool, "ratio": float}}
    Only includes checkpoints that actually exist on disk.
    """
    pruned = args.pruned_dir
    registry = {
        "baseline": {
            "ckpt_path": args.baseline_ckpt,
            "is_pruned": False,
            "ratio":     0.0,
            "label":     "baseline",
        },
        "10pct": {
            "ckpt_path": os.path.join(pruned, "pruned_10pct_finetuned", "finetuned_final.pth"),
            "is_pruned": True,
            "ratio":     0.10,
            "label":     "pruned_10pct",
        },
        "20pct": {
            "ckpt_path": os.path.join(pruned, "pruned_20pct_finetuned", "finetuned_final.pth"),
            "is_pruned": True,
            "ratio":     0.20,
            "label":     "pruned_20pct",
        },
        "25pct": {
            "ckpt_path": os.path.join(pruned, "pruned_25pct_finetuned", "finetuned_final.pth"),
            "is_pruned": True,
            "ratio":     0.25,
            "label":     "pruned_25pct",
        },
        "30pct": {
            "ckpt_path": os.path.join(pruned, "pruned_30pct_finetuned", "finetuned_final.pth"),
            "is_pruned": True,
            "ratio":     0.30,
            "label":     "pruned_30pct",
        },
        "50pct": {
            "ckpt_path": os.path.join(pruned, "pruned_50pct_finetuned", "finetuned_final.pth"),
            "is_pruned": True,
            "ratio":     0.50,
            "label":     "pruned_50pct",
        },
    }
    return registry


# ════════════════════════════════════════════════════════════════════════════
# 4.  MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_model(entry: dict, device: str):
    """
    Load a Score SDE model from a checkpoint entry.
    Returns (model, n_params_M, pruning_ratio).
    """
    ckpt_path = entry["ckpt_path"]
    if not os.path.exists(ckpt_path):
        return None, None, None

    ckpt     = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    keep_128 = ckpt.get("keep_128")
    keep_256 = ckpt.get("keep_256")

    if keep_128 is not None and keep_256 is not None:
        # Pruned checkpoint
        ratio = ckpt.get("pruning_ratio", entry["ratio"])
        model = mutils.create_model(CONFIG)
        prune_ncsnpp(model, keep_128, keep_256)
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        # Baseline checkpoint
        model     = mutils.create_model(CONFIG)
        optimizer = get_optimizer(CONFIG, model.parameters())
        ema       = ExponentialMovingAverage(model.parameters(),
                                             decay=CONFIG.model.ema_rate)
        model.load_state_dict(ckpt["model"], strict=False)
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
            ema.copy_to(model.parameters())
        ratio = 0.0

    model = model.to(device).eval()
    n_params_M = sum(p.numel() for p in model.parameters()) / 1e6
    return model, n_params_M, ratio


# ════════════════════════════════════════════════════════════════════════════
# 5.  CLASSIFIER (with [-1,1]→[0,1] wrapper)
# ════════════════════════════════════════════════════════════════════════════

def load_classifier(device):
    from robustbench.utils import load_model as rb_load
    rb_model = rb_load("Standard", dataset="cifar10", threat_model="Linf")
    class _W(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m((x + 1) / 2)  # [-1,1] → [0,1]
    clf = _W(rb_model).to(device).eval()
    print(f"  Classifier: WRN-28-10 Standard (RobustBench)  [-1,1]→[0,1] wrapped")
    return clf


# ════════════════════════════════════════════════════════════════════════════
# 6.  VP-SDE SCHEDULE
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


# ════════════════════════════════════════════════════════════════════════════
# 7.  PURIFICATION  (DDIM + DeepCache)
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def purify(x, score_fn, schedule, t_star, ddim_steps, device, deepcache_K=1):
    """
    DiffPure purification with optional DeepCache.

    deepcache_K=1  → standard DDIM (no caching, all steps compute UNet)
    deepcache_K=K  → reuse cached score on non-anchor steps (every K-th = anchor)

    Returns (x_pure, n_unet_calls).
    """
    noise  = torch.randn_like(x)
    x_t    = schedule.q_sample(x, t_star, noise)
    ts     = np.clip(np.linspace(t_star, 1, ddim_steps + 1, dtype=int), 1, t_star)

    x_cur        = x_t
    cached_score = None
    n_unet_calls = 0

    for i in range(len(ts) - 1):
        t_cur, t_prev = int(ts[i]), int(ts[i + 1])

        # DeepCache: skip UNet on non-anchor steps
        if deepcache_K > 1 and i > 0 and (i % deepcache_K != 0) \
                and cached_score is not None:
            score = cached_score      # reuse previous score
        else:
            t_cont       = torch.full((x.shape[0],), t_cur / schedule.N,
                                      device=device, dtype=torch.float32)
            score        = score_fn(x_cur, t_cont)
            cached_score = score
            n_unet_calls += 1

        s1m   = schedule.sqrt_1mab[t_cur - 1]
        sab   = schedule.sqrt_ab[t_cur - 1]
        eps_p = -s1m * score
        x0_p  = ((x_cur - s1m * eps_p) / sab).clamp(-1, 1)
        x_cur = (schedule.alpha_bar[t_prev - 1].sqrt() * x0_p
                 + schedule.sqrt_1mab[t_prev - 1] * eps_p)

    return x_cur, n_unet_calls


# ════════════════════════════════════════════════════════════════════════════
# 8.  ATTACKS
# ════════════════════════════════════════════════════════════════════════════

class _BPDA(torch.autograd.Function):
    """Straight-through estimator — gradient flows to x_adv (second arg)."""
    @staticmethod
    def forward(ctx, x_pure, x_adv): return x_pure
    @staticmethod
    def backward(ctx, g):            return None, g   # gradient → x_adv


def pgd_eot(x_clean, y, score_fn, schedule, classifier,
             t_star, ddim_steps, deepcache_K,
             eps, step_size, n_steps, attack_eot, device):
    x_adv = (x_clean + torch.zeros_like(x_clean).uniform_(-eps, eps)).clamp(-1., 1.)
    for _ in range(n_steps):
        x_adv = x_adv.detach().requires_grad_(True)
        for _ in range(attack_eot):
            with torch.no_grad():
                x_pure, _ = purify(x_adv, score_fn, schedule,
                                    t_star, ddim_steps, device, deepcache_K)
            x_in = _BPDA.apply(x_pure, x_adv)
            loss_sample = F.cross_entropy(classifier(x_in), y)
            loss_sample = loss_sample / attack_eot
            loss_sample.backward()
        with torch.no_grad():
            x_adv = x_adv + step_size * x_adv.grad.sign()
            x_adv = torch.clamp(
                torch.min(torch.max(x_adv, x_clean - eps), x_clean + eps),
                -1., 1.
            )
    return x_adv.detach()


# ════════════════════════════════════════════════════════════════════════════
# 9.  SINGLE-CONFIG EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def evaluate_config(model, classifier, batches, schedule, score_fn,
                    deepcache_K, args, device):
    """
    Evaluate one (model, K) configuration.
    Returns dict with all metrics.
    """
    n_total = sum(x.shape[0] for x, _ in batches)
    n_unet_expected = math.ceil(args.ddim_steps / deepcache_K) \
                      if deepcache_K > 1 else args.ddim_steps
    dc_tag = f"K={deepcache_K}" if deepcache_K > 1 else "K=1 (no cache)"
    print(f"\n  {'─'*55}")
    print(f"  DeepCache {dc_tag}  "
          f"(~{n_unet_expected}/{args.ddim_steps} UNet calls/sample)")
    print(f"  {'─'*55}")

    # ── Clean accuracy ──────────────────────────────────────────────────────
    clean_correct  = 0
    total_purify_s = 0.0
    total_unet     = 0
    model.eval()

    for bi, (x, y) in enumerate(batches):
        t0 = time.time()
        logit_sum = None
        unet_this = 0
        for _ in range(args.eot):
            with torch.no_grad():
                x_pure, nc = purify(x, score_fn, schedule,
                                    args.t_star, args.ddim_steps, device, deepcache_K)
                logits = classifier(x_pure)
            logit_sum = logits if logit_sum is None else logit_sum + logits
            unet_this = nc   # same every eot iteration

        purify_s = time.time() - t0
        preds = logit_sum.argmax(1)
        bc    = (preds == y).sum().item()
        clean_correct  += bc
        total_purify_s += purify_s
        total_unet     += unet_this

        ms_img = purify_s / x.shape[0] * 1000 / args.eot
        print(f"    [Clean] Batch {bi+1:3d}/{len(batches)}  "
              f"acc={bc/x.shape[0]:.3f}  "
              f"running={clean_correct/min((bi+1)*args.eval_batch, n_total):.3f}  "
              f"{ms_img:.0f}ms/img  UNet={unet_this}")

    clean_acc      = clean_correct / n_total
    purify_ms_img  = total_purify_s / n_total * 1000 / args.eot
    unet_calls_avg = total_unet / len(batches)

    print(f"\n  Clean acc  : {clean_acc:.4f}  ({clean_correct}/{n_total})")
    print(f"  Purify     : {purify_ms_img:.1f} ms/image")
    print(f"  UNet calls : {unet_calls_avg:.1f} / {args.ddim_steps}")

    # ── PGD-20 robust accuracy ──────────────────────────────────────────────
    pgd_acc = pgd_correct = None
    pgd_time_s = 0.0
    if not args.only_clean:
        print(f"\n  [PGD-{args.pgd_steps}  ε={args.attack_eps*255:.0f}/255  BPDA  EOT={args.attack_eot}]")
        pgd_correct_n = 0
        t0_pgd = time.time()

        for bi, (x, y) in enumerate(batches):
            t0    = time.time()
            x_adv = pgd_eot(
                x, y, score_fn, schedule, classifier,
                args.t_star, args.ddim_steps, deepcache_K,
                args.attack_eps, args.pgd_step_size, args.pgd_steps, args.attack_eot, device,
            )
            
            # Evaluate adversarial examples with eval_eot
            logit_sum = None
            with torch.no_grad():
                for _ in range(args.eot):
                    x_pure, _ = purify(x_adv, score_fn, schedule,
                                       args.t_star, args.ddim_steps, device, deepcache_K)
                    logits = classifier(x_pure)
                    logit_sum = logits if logit_sum is None else logit_sum + logits
                preds = logit_sum.argmax(1)
            bc = (preds == y).sum().item()
            pgd_correct_n += bc
            elapsed = time.time() - t0
            print(f"    [PGD]   Batch {bi+1:3d}/{len(batches)}  "
                  f"rob={bc/x.shape[0]:.3f}  "
                  f"running={pgd_correct_n/min((bi+1)*args.eval_batch, n_total):.3f}  "
                  f"[{elapsed:.0f}s]")

        pgd_acc     = pgd_correct_n / n_total
        pgd_correct = pgd_correct_n
        pgd_time_s  = time.time() - t0_pgd
        print(f"\n  PGD-{args.pgd_steps} acc: {pgd_acc:.4f}  "
              f"({pgd_correct}/{n_total})  [{pgd_time_s/60:.1f}min]")

    return {
        "clean_acc":      round(clean_acc,   6),
        "clean_pct":      round(clean_acc   * 100, 2),
        "clean_correct":  clean_correct,
        "pgd_acc":        round(pgd_acc,  6) if pgd_acc  is not None else "",
        "pgd_pct":        round(pgd_acc * 100, 2) if pgd_acc is not None else "",
        "pgd_correct":    pgd_correct         if pgd_correct is not None else "",
        "n_images":       n_total,
        "purify_ms_img":  round(purify_ms_img, 2),
        "unet_calls":     round(unet_calls_avg, 1),
        "clean_time_s":   round(total_purify_s, 1),
        "pgd_time_s":     round(pgd_time_s, 1),
    }


# ════════════════════════════════════════════════════════════════════════════
# 10.  RESULTS  (CSV + JSON, append-safe)
# ════════════════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    "run_id", "timestamp",
    "model_label", "pruning_ratio", "params_M", "compression_x", "reduction_pct",
    "deepcache_K", "unet_calls", "unet_calls_expected", "ddim_steps",
    "clean_acc", "clean_pct", "clean_correct",
    "pgd_acc",   "pgd_pct",   "pgd_correct",
    "n_images", "t_star", "eot", "attack_eot", "pgd_steps", "attack_eps_over255",
    "purify_ms_img", "speedup_vs_k1",
    "clean_time_s", "pgd_time_s",
    "ckpt_path", "device",
]


def save_rows(rows: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    csv_path  = os.path.join(output_dir, "deepcache_results.csv")
    json_path = os.path.join(output_dir, "deepcache_results.json")

    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\n  [Save] CSV  → {csv_path}  ({len(rows)} rows appended)")

    existing = []
    if os.path.exists(json_path):
        with open(json_path) as f:
            try:    existing = json.load(f)
            except: existing = []
    existing.extend(rows)
    with open(json_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    print(f"  [Save] JSON → {json_path}")


def print_summary_table(rows: list):
    """Print a presentation-ready table grouped by model."""
    print("\n" + "═" * 82)
    print("  DEEPCACHE SWEEP  —  SUMMARY TABLE")
    print("═" * 82)
    hdr = (f"  {'Model':<18} {'K':>3} {'UNet':>5} {'Speedup':>8} "
           f"{'Clean%':>7} {'PGD-20%':>8} {'ms/img':>8}")
    print(hdr)
    print("  " + "─" * 78)

    current_model = None
    for r in rows:
        if r["model_label"] != current_model:
            if current_model is not None:
                print()
            current_model = r["model_label"]

        speedup = f"{r['speedup_vs_k1']:.2f}×" if r.get("speedup_vs_k1") else "─"
        pgd_str = f"{r['pgd_pct']:.2f}"         if r["pgd_pct"] != "" else "─"
        print(f"  {r['model_label']:<18} {r['deepcache_K']:>3} "
              f"{r['unet_calls']:>5.1f} {speedup:>8} "
              f"{r['clean_pct']:>7.2f}% {pgd_str:>7}%  "
              f"{r['purify_ms_img']:>7.1f}")

    print("═" * 82)


# ════════════════════════════════════════════════════════════════════════════
# 11.  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 70)
    print("  DiffPure CIFAR-10  DeepCache Sweep")
    print("=" * 70)
    if "cuda" in device and torch.cuda.is_available():
        idx = int(device.split(":")[-1]) if ":" in device else 0
        print(f"  GPU        : {torch.cuda.get_device_name(idx)}")
    print(f"  Checkpoints: {args.checkpoints}")
    print(f"  K values   : {args.k_values}")
    print(f"  Eval images: {args.eval_images}  EOT={args.eot}")
    print(f"  PGD        : {'OFF (--only_clean)' if args.only_clean else f'PGD-{args.pgd_steps}  BPDA'}")
    print(f"  Output     : {args.output_dir}/deepcache_results.{{csv,json}}")

    # ── Build checkpoint registry ───────────────────────────────────────────
    registry = build_checkpoint_registry(args)

    # Filter to requested checkpoints and check existence
    selected = []
    for name in args.checkpoints:
        if name not in registry:
            print(f"  [WARN] Unknown checkpoint key '{name}' — skipping")
            continue
        entry = registry[name]
        if not os.path.exists(entry["ckpt_path"]):
            print(f"  [SKIP] {name}: checkpoint not found → {entry['ckpt_path']}")
            continue
        selected.append((name, entry))

    if not selected:
        print("[ERROR] No valid checkpoints found. Check --pruned_dir and --baseline_ckpt.")
        return

    print(f"\n  Will evaluate {len(selected)} checkpoints × {len(args.k_values)} K values "
          f"= {len(selected) * len(args.k_values)} total configurations")

    # ── Load classifier once (shared across all configs) ────────────────────
    print("\n[Step 1] Loading classifier ...")
    classifier = load_classifier(device)

    # ── Load data once (same 512 images for all configs) ────────────────────
    print("\n[Step 2] Loading CIFAR-10 test images ...")
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
    print(f"  {n_total} images loaded  ({len(batches)} batches)")

    # ── VP-SDE schedule (shared) ────────────────────────────────────────────
    schedule = VPSchedule(device)

    # ── Main sweep ──────────────────────────────────────────────────────────
    all_rows = []
    run_ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for ck_idx, (name, entry) in enumerate(selected):
        print(f"\n{'═'*70}")
        print(f"  [{ck_idx+1}/{len(selected)}]  {entry['label']}  "
              f"→  {entry['ckpt_path']}")
        print(f"{'═'*70}")

        print(f"\n  Loading model ...")
        model, n_params_M, ratio = load_model(entry, device)
        if model is None:
            print(f"  [SKIP] Failed to load {entry['ckpt_path']}")
            continue

        compression_x = round(106.6 / n_params_M, 4)
        reduction_pct = round((106.6 - n_params_M) / 106.6 * 100, 2)
        print(f"  Params : {n_params_M:.2f}M  "
              f"({compression_x:.2f}× compression  {reduction_pct:.1f}% reduction)")

        # Build score function for this model
        _sde     = sde_lib.VPSDE(beta_min=CONFIG.model.beta_min,
                                   beta_max=CONFIG.model.beta_max,
                                   N=CONFIG.model.num_scales)
        score_fn = mutils.get_score_fn(_sde, model, train=False, continuous=True)

        # Track K=1 timing for speedup computation
        k1_purify_ms = None
        model_rows   = []

        for K in args.k_values:
            print(f"\n  ── DeepCache K={K} ──────────────────────────────────")
            t_cfg = time.time()
            metrics = evaluate_config(
                model, classifier, batches, schedule, score_fn,
                K, args, device,
            )
            cfg_elapsed = time.time() - t_cfg

            if K == 1:
                k1_purify_ms = metrics["purify_ms_img"]

            speedup = round(k1_purify_ms / metrics["purify_ms_img"], 3) \
                      if k1_purify_ms and K != 1 else 1.0

            unet_expected = math.ceil(args.ddim_steps / K) if K > 1 else args.ddim_steps

            row = {
                "run_id":             f"{entry['label']}_K{K}_{run_ts}",
                "timestamp":          datetime.datetime.now().isoformat(),
                "model_label":        entry["label"],
                "pruning_ratio":      ratio,
                "params_M":           round(n_params_M, 3),
                "compression_x":      compression_x,
                "reduction_pct":      reduction_pct,
                "deepcache_K":        K,
                "unet_calls":         metrics["unet_calls"],
                "unet_calls_expected":unet_expected,
                "ddim_steps":         args.ddim_steps,
                "clean_acc":          metrics["clean_acc"],
                "clean_pct":          metrics["clean_pct"],
                "clean_correct":      metrics["clean_correct"],
                "pgd_acc":            metrics["pgd_acc"],
                "pgd_pct":            metrics["pgd_pct"],
                "pgd_correct":        metrics["pgd_correct"],
                "n_images":           metrics["n_images"],
                "t_star":             args.t_star,
                "eot":                args.eot,
                "attack_eot":         args.attack_eot if not args.only_clean else 0,
                "pgd_steps":          args.pgd_steps if not args.only_clean else 0,
                "attack_eps_over255": round(args.attack_eps * 255, 1),
                "purify_ms_img":      metrics["purify_ms_img"],
                "speedup_vs_k1":      speedup,
                "clean_time_s":       metrics["clean_time_s"],
                "pgd_time_s":         metrics["pgd_time_s"],
                "ckpt_path":          entry["ckpt_path"],
                "device":             device,
            }
            model_rows.append(row)
            all_rows.append(row)

            # Save after each (model, K) so we don't lose data on crash
            save_rows([row], args.output_dir)
            print(f"\n  Config done in {cfg_elapsed/60:.1f} min")

        # Free model memory before loading next
        del model, score_fn
        torch.cuda.empty_cache()

    # ── Final summary ────────────────────────────────────────────────────────
    if all_rows:
        print_summary_table(all_rows)
    print(f"\n DeepCache sweep complete — {len(all_rows)} configs evaluated")
    print(f"   Results: {args.output_dir}/deepcache_results.{{csv,json}}")


if __name__ == "__main__":
    main()
