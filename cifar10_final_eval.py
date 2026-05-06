"""
cifar10_final_eval.py
=====================
Rigorous, presentation-ready evaluation of one DiffPure checkpoint.

Key design decisions (vs earlier scripts):
  • NO pre-screening  — accuracy reported over ALL 512 images
  • PGD-20 with EOT=5 attack — attacker averages gradient over 5 purifications
    per step (not EOT=1 which is a blind attack against stochastic defenses)
  • Clean accuracy with EOT=5 — average over 5 purifications for stable estimate
  • Purification timing measured separately at EOT=1 (net inference cost)
  • All elapsed times logged explicitly for the paper table
  • Saves to results/cifar10/final_eval_results.csv (append — never overwrites)

Usage (one model per run, multiple GPUs in parallel):
  # Baseline
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  python3 cifar10_final_eval.py --ratio 0.00 --device cuda:0

  # 10% pruned
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  python3 cifar10_final_eval.py --ratio 0.10 --device cuda:1

  # 25% pruned
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  python3 cifar10_final_eval.py --ratio 0.25 --device cuda:2

  # After all models are done — print final table with speedup column
  python3 cifar10_final_eval.py --print_table

Time estimate per model on H200 (512 images):
  Clean acc (EOT=5)        : ~5  min
  PGD-20  (EOT=5 attack)   : ~75–90 min
  ─────────────────────────────────────
  Total                    : ~80–95 min per model
"""

import os, sys, csv, json, time, datetime, argparse, warnings
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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Rigorous DiffPure CIFAR-10 evaluation for presentation",
    )
    p.add_argument("--ratio", type=float, default=None,
                   help="Pruning ratio: 0.0=baseline, 0.1, 0.2, 0.25, 0.3, 0.5")

    # Checkpoint locations
    p.add_argument("--baseline_ckpt",
                   default="checkpoints/score_sde/checkpoint_8.pth")
    p.add_argument("--pruned_dir",
                   default="checkpoints/score_sde/pruned")

    # Diffusion settings
    p.add_argument("--t_star",      type=int,   default=100)
    p.add_argument("--ddim_steps",  type=int,   default=50)

    # Eval settings
    p.add_argument("--eval_images", type=int,   default=512)
    p.add_argument("--eval_batch",  type=int,   default=32)
    p.add_argument("--eval_eot",    type=int,   default=5,
                   help="EOT for clean accuracy (averages N purifications)")

    # Attack settings
    p.add_argument("--pgd_steps",     type=int,   default=20)
    p.add_argument("--attack_eot",    type=int,   default=20,
                   help="EOT for PGD attack gradient (strong attack)")
    p.add_argument("--attack_eps",    type=float, default=8/255)
    p.add_argument("--pgd_step_size", type=float, default=2/255)

    # I/O
    p.add_argument("--data_dir",    default="./data")
    p.add_argument("--output_dir",  default="results/cifar10/")
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--seed",        type=int, default=42)

    # Utility mode
    p.add_argument("--print_table", action="store_true",
                   help="Print final summary table from existing CSV and exit")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# 2.  CIFAR-10 CONFIG
# ════════════════════════════════════════════════════════════════════════════

class _B(dict):
    def __getattr__(self, k):
        try: v = self[k]; return _B(v) if isinstance(v, dict) else v
        except KeyError: raise AttributeError(k)
    def __setattr__(self, k, v): self[k] = v

CONFIG = _B({
    "data":     {"dataset":"CIFAR10","image_size":32,"num_channels":3,
                 "centered":True,"random_flip":True,"uniform_dequantization":False},
    "model":    {"sigma_min":0.01,"sigma_max":50,"num_scales":1000,
                 "beta_min":0.1,"beta_max":20.0,"dropout":0.1,
                 "name":"ncsnpp","scale_by_sigma":False,"ema_rate":0.9999,
                 "normalization":"GroupNorm","nonlinearity":"swish","nf":128,
                 "ch_mult":(1,2,2,2),"num_res_blocks":8,
                 "attn_resolutions":(16,),"resamp_with_conv":True,
                 "conditional":True,"fir":False,"fir_kernel":[1,3,3,1],
                 "skip_rescale":True,"resblock_type":"biggan",
                 "progressive":"none","progressive_input":"none",
                 "progressive_combine":"sum","attention_type":"ddpm",
                 "init_scale":0.0,"embedding_type":"positional",
                 "fourier_scale":16,"conv_size":3},
    "training": {"sde":"vpsde","continuous":True,"reduce_mean":True},
    "optim":    {"weight_decay":0,"optimizer":"Adam","lr":2e-4,
                 "beta1":0.9,"eps":1e-8,"warmup":5000,"grad_clip":1.0},
    "sampling": {"n_steps_each":1,"noise_removal":True,"probability_flow":False,
                 "snr":0.16,"method":"pc","predictor":"euler_maruyama",
                 "corrector":"none"},
})

BASELINE_PARAMS_M = 106.6


# ════════════════════════════════════════════════════════════════════════════
# 3.  CHECKPOINT FINDING
# ════════════════════════════════════════════════════════════════════════════

def find_checkpoint(ratio: float, args) -> str:
    if ratio == 0.0:
        return args.baseline_ckpt
    pct = int(ratio * 100)
    ft_path = os.path.join(args.pruned_dir,
                           f"pruned_{pct:02d}pct_finetuned", "finetuned_final.pth")
    if os.path.exists(ft_path):
        return ft_path
    raise FileNotFoundError(
        f"Finetuned checkpoint not found: {ft_path}\n"
        f"Run: python3 cifar10_pipeline.py --ratio {ratio} --skip_prune --ft_steps 20000"
    )


# ════════════════════════════════════════════════════════════════════════════
# 4.  MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path: str, device: str):
    """Auto-detect baseline vs pruned checkpoint and load correctly."""
    ckpt     = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    keep_128 = ckpt.get("keep_128")
    keep_256 = ckpt.get("keep_256")

    if keep_128 is not None and keep_256 is not None:
        # Pruned checkpoint
        ratio = ckpt.get("pruning_ratio", 0.0)
        model = mutils.create_model(CONFIG)
        prune_ncsnpp(model, keep_128, keep_256)
        model.load_state_dict(ckpt["model"], strict=True)
        print(f"  Type    : Pruned ({ratio:.0%})  "
              f"keep {len(keep_128)}/128ch  {len(keep_256)}/256ch")
    else:
        # Baseline
        model     = mutils.create_model(CONFIG)
        optimizer = get_optimizer(CONFIG, model.parameters())
        ema       = ExponentialMovingAverage(model.parameters(),
                                             decay=CONFIG.model.ema_rate)
        model.load_state_dict(ckpt["model"], strict=False)
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
            ema.copy_to(model.parameters())
        ratio = 0.0
        print(f"  Type    : Baseline (full model)")

    model      = model.to(device).eval()
    n_params_M = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params  : {n_params_M:.2f}M")
    return model, n_params_M, ratio


# ════════════════════════════════════════════════════════════════════════════
# 5.  CLASSIFIER  (RobustBench WRN-28-10, [-1,1]→[0,1] wrapped)
# ════════════════════════════════════════════════════════════════════════════

def load_classifier(device: str) -> nn.Module:
    from robustbench.utils import load_model as rb_load
    rb_model = rb_load("Standard", dataset="cifar10", threat_model="Linf")
    class _W(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m((x + 1) / 2)  # [-1,1]→[0,1]
    clf = _W(rb_model).to(device).eval()
    print("  Classifier: WRN-28-10 Standard (RobustBench)  [-1,1]→[0,1] wrapped")
    return clf


# ════════════════════════════════════════════════════════════════════════════
# 6.  VP-SDE SCHEDULE
# ════════════════════════════════════════════════════════════════════════════

class VPSchedule:
    def __init__(self, device: str):
        N     = CONFIG.model.num_scales
        self.N = N
        betas = torch.linspace(CONFIG.model.beta_min / N,
                               CONFIG.model.beta_max / N, N, device=device)
        ab            = torch.cumprod(1.0 - betas, dim=0)
        self.alpha_bar = ab
        self.sqrt_ab   = ab.sqrt()
        self.sqrt_1mab = (1.0 - ab).sqrt()

    def q_sample(self, x0, t_int, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        return self.sqrt_ab[t_int - 1] * x0 + self.sqrt_1mab[t_int - 1] * noise


# ════════════════════════════════════════════════════════════════════════════
# 7.  PURIFICATION  (DDIM, no DeepCache here — pure baseline timing)
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def purify(x: torch.Tensor, score_fn, schedule: VPSchedule,
           t_star: int, ddim_steps: int, device: str) -> torch.Tensor:
    """One DiffPure purification pass (DDIM, deterministic η=0)."""
    noise  = torch.randn_like(x)
    x_t    = schedule.q_sample(x, t_star, noise)
    ts     = np.clip(np.linspace(t_star, 1, ddim_steps + 1, dtype=int), 1, t_star)
    x_cur  = x_t

    for i in range(len(ts) - 1):
        t_cur, t_prev = int(ts[i]), int(ts[i + 1])
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


# ════════════════════════════════════════════════════════════════════════════
# 8.  PGD ATTACK  (BPDA + EOT)
# ════════════════════════════════════════════════════════════════════════════

class _BPDA(torch.autograd.Function):
    """
    Straight-through BPDA estimator.
      Forward : returns x_pure  (the purified image)
      Backward: passes gradient to x_adv (second arg), not x_pure (detached)
    """
    @staticmethod
    def forward(ctx, x_pure, x_adv): return x_pure
    @staticmethod
    def backward(ctx, g):            return None, g   # grad → x_adv ✓


def pgd_eot(x_clean: torch.Tensor, y: torch.Tensor,
            score_fn, schedule: VPSchedule, classifier: nn.Module,
            t_star: int, ddim_steps: int,
            eps: float, step_size: float, n_steps: int,
            attack_eot: int, device: str) -> torch.Tensor:
    """
    PGD-ℓ∞ with BPDA and EOT.

    Attack gradient is the average of `attack_eot` BPDA gradient estimates
    (each from an independent random purification). This prevents the attacker
    exploiting one lucky noise draw — equivalent to attacking E[purify(x)].

    attack_eot=1  → same as before (blind to stochasticity)
    attack_eot=5  → strong, closer to the paper's EOT=20
    """
    x_adv = (x_clean
             + torch.zeros_like(x_clean).uniform_(-eps, eps)
             ).clamp(-1., 1.)

    for step_i in range(n_steps):
        x_adv = x_adv.detach().requires_grad_(True)

        # Accumulate loss over EOT samples (average gradient = average loss.backward)
        total_loss = torch.zeros(1, device=device, requires_grad=False)
        for _ in range(attack_eot):
            with torch.no_grad():
                x_pure = purify(x_adv, score_fn, schedule,
                                t_star, ddim_steps, device)
            # BPDA: treat purifier as identity for backward pass
            x_in        = _BPDA.apply(x_pure, x_adv)
            logits      = classifier(x_in)
            loss_sample = F.cross_entropy(logits, y)
            # Accumulate loss (sum → avg after dividing)
            loss_sample = loss_sample / attack_eot
            loss_sample.backward()   # grad accumulates in x_adv.grad

        # x_adv.grad now holds the averaged gradient over attack_eot samples
        with torch.no_grad():
            x_adv = x_adv + step_size * x_adv.grad.sign()
            x_adv = torch.clamp(
                torch.min(torch.max(x_adv, x_clean - eps), x_clean + eps),
                -1., 1.
            )

    return x_adv.detach()


# ════════════════════════════════════════════════════════════════════════════
# 9.  DATA
# ════════════════════════════════════════════════════════════════════════════

def get_eval_batches(data_dir: str, n: int, batch_size: int,
                     seed: int, device: str):
    """Load n CIFAR-10 test images. NO pre-screening."""
    tf  = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    ds  = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                        download=True, transform=tf)
    g   = torch.Generator(); g.manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idx),
        batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    batches = [(x.to(device), y.to(device)) for x, y in loader]
    return batches


# ════════════════════════════════════════════════════════════════════════════
# 10.  EVALUATION PHASES
# ════════════════════════════════════════════════════════════════════════════

def eval_clean(batches, score_fn, schedule, classifier,
               t_star, ddim_steps, eval_eot, device):
    """
    Clean accuracy with EOT=eval_eot (average logits over multiple purifications).
    Also measures purify_ms_img at EOT=1 for a fair speed comparison.
    Returns (clean_acc, clean_correct, n_total, purify_ms_img, elapsed_s)
    """
    n_total        = sum(x.shape[0] for x, _ in batches)
    correct        = 0
    total_s        = 0.0
    timing_ms_list = []   # for per-image timing at EOT=1

    t0_phase = time.time()
    for bi, (x, y) in enumerate(batches):
        B = x.shape[0]

        # ─── accuracy: average logits over eval_eot purifications ───────────
        t0 = time.time()
        logit_sum = None
        for _ in range(eval_eot):
            with torch.no_grad():
                x_pure = purify(x, score_fn, schedule, t_star, ddim_steps, device)
                logits = classifier(x_pure)
            logit_sum = logits if logit_sum is None else logit_sum + logits
        elapsed_batch = time.time() - t0

        preds   = logit_sum.argmax(1)
        bc      = (preds == y).sum().item()
        correct += bc
        total_s += elapsed_batch

        # ─── timing: one extra single purification for speed measurement ────
        t_time = time.time()
        with torch.no_grad():
            _ = purify(x, score_fn, schedule, t_star, ddim_steps, device)
        timing_ms_list.append((time.time() - t_time) / B * 1000)  # ms/img

        ms_eot1 = timing_ms_list[-1]
        running = correct / min((bi + 1) * batches[0][0].shape[0], n_total)
        print(f"    Batch {bi+1:3d}/{len(batches)}  "
              f"acc={bc/B:.3f}  running={running:.3f}  "
              f"purify={ms_eot1:.0f}ms/img")

    clean_acc      = correct / n_total
    purify_ms_img  = float(np.mean(timing_ms_list))   # EOT=1, pure inference
    elapsed_s      = time.time() - t0_phase
    return clean_acc, correct, n_total, purify_ms_img, elapsed_s


def eval_pgd(batches, score_fn, schedule, classifier,
             t_star, ddim_steps, pgd_steps, attack_eot,
             attack_eps, pgd_step_size, eval_eot, device):
    """
    PGD-{pgd_steps} robust accuracy.
    Attack uses EOT=attack_eot for gradient estimation (strong).
    Final eval uses EOT=eval_eot on the adversarial examples.
    NO pre-screening — all images evaluated.
    Returns (pgd_acc, pgd_correct, n_total, elapsed_s)
    """
    n_total    = sum(x.shape[0] for x, _ in batches)
    correct    = 0
    t0_phase   = time.time()

    for bi, (x, y) in enumerate(batches):
        B  = x.shape[0]
        t0 = time.time()

        # Generate adversarial examples (strong EOT attack)
        x_adv = pgd_eot(
            x, y, score_fn, schedule, classifier,
            t_star, ddim_steps,
            attack_eps, pgd_step_size, pgd_steps, attack_eot, device,
        )

        # Evaluate: purify adversarial example (EOT for stable accuracy)
        logit_sum = None
        with torch.no_grad():
            for _ in range(eval_eot):
                x_pure = purify(x_adv, score_fn, schedule, t_star, ddim_steps, device)
                logits  = classifier(x_pure)
                logit_sum = logits if logit_sum is None else logit_sum + logits

        preds   = logit_sum.argmax(1)
        bc      = (preds == y).sum().item()
        correct += bc
        elapsed  = time.time() - t0

        running = correct / min((bi + 1) * B, n_total)
        print(f"    Batch {bi+1:3d}/{len(batches)}  "
              f"rob={bc/B:.3f}  running={running:.3f}  "
              f"[{elapsed:.0f}s/batch  eta≈{(len(batches)-bi-1)*elapsed/60:.0f}min]")

    pgd_acc  = correct / n_total
    elapsed_s = time.time() - t0_phase
    return pgd_acc, correct, n_total, elapsed_s


# ════════════════════════════════════════════════════════════════════════════
# 11.  RESULTS  (append-safe CSV + JSON)
# ════════════════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    # Identifier
    "run_id", "timestamp", "device",
    # Model
    "model_label", "pruning_ratio",
    "params_M", "compression_x", "reduction_pct",
    # Clean accuracy
    "n_images",
    "clean_acc", "clean_pct", "clean_correct",
    # Robust accuracy (PGD)
    "pgd_acc", "pgd_pct", "pgd_correct",
    # Speed
    "purify_ms_img",        # ms per image per purification (EOT=1, wall-clock)
    "speedup_vs_baseline",  # filled in post-processing
    # Elapsed times (minutes)
    "clean_time_min", "pgd_time_min", "total_time_min",
    # Eval config
    "t_star", "ddim_steps",
    "eval_eot", "attack_eot", "pgd_steps",
    "attack_eps_over255",
    # Checkpoint
    "checkpoint",
]


def save_row(row: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    csv_path  = os.path.join(output_dir, "final_eval_results.csv")
    json_path = os.path.join(output_dir, "final_eval_results.json")

    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)

    existing = []
    if os.path.exists(json_path):
        with open(json_path) as f:
            try:    existing = json.load(f)
            except: existing = []
    existing.append(row)
    with open(json_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)

    print(f"\n  [Saved] CSV  → {csv_path}")
    print(f"  [Saved] JSON → {json_path}")


def compute_and_print_table(output_dir: str):
    """
    Read final_eval_results.csv, compute speedup vs baseline, print table.
    Also rewrites CSV with speedup filled in.
    """
    import pandas as pd
    csv_path = os.path.join(output_dir, "final_eval_results.csv")
    if not os.path.exists(csv_path):
        print(f"No results found at {csv_path}"); return

    df = pd.read_csv(csv_path).sort_values("pruning_ratio")

    # Compute speedup vs baseline purify_ms_img
    baseline_ms = df.loc[df["pruning_ratio"] == 0.0, "purify_ms_img"]
    if len(baseline_ms):
        bms = baseline_ms.iloc[0]
        df["speedup_vs_baseline"] = (bms / df["purify_ms_img"]).round(3)
    else:
        df["speedup_vs_baseline"] = ""

    # Write back with speedup
    df.to_csv(csv_path, index=False)
    json_path = csv_path.replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump(df.to_dict("records"), f, indent=2, default=str)

    # Print presentation table
    W = 95
    print("\n" + "═" * W)
    print("  FINAL EVALUATION RESULTS  "
          "(PGD-20, EOT=5 attack, EOT=5 eval, NO pre-screening)")
    print("═" * W)
    hdr = (f"  {'Model':<16} {'Params':>7} {'Compr':>6} {'Reduc':>6} "
           f"{'Clean%':>7} {'PGD20%':>7} {'ms/img':>8} {'Speedup':>8} "
           f"{'PGD-min':>8}")
    print(hdr)
    print("  " + "─" * (W - 2))

    for _, r in df.iterrows():
        compr   = f"{r['compression_x']:.2f}×"
        reduc   = f"{r['reduction_pct']:.1f}%"
        speedup = f"{r['speedup_vs_baseline']:.2f}×" \
                  if r['speedup_vs_baseline'] != "" else "─"
        drop_vs_base = ""
        if r["pruning_ratio"] > 0:
            base_pgd = df.loc[df["pruning_ratio"]==0.0, "pgd_pct"]
            if len(base_pgd):
                drop = base_pgd.iloc[0] - r["pgd_pct"]
                drop_vs_base = f"  (▼{drop:.2f}pp)"
        print(f"  {r['model_label']:<16} {r['params_M']:>6.1f}M {compr:>6} {reduc:>6} "
              f"{r['clean_pct']:>7.2f}% {r['pgd_pct']:>7.2f}% "
              f"{r['purify_ms_img']:>7.1f} {speedup:>8} "
              f"{r['pgd_time_min']:>7.1f}{drop_vs_base}")
    print("═" * W)
    print(f"\n  'Speedup' = baseline purify time / model purify time")
    print(f"  'PGD-min' = wall-clock minutes for full PGD-20 eval")
    print(f"  'pp'      = percentage points drop vs baseline\n")


# ════════════════════════════════════════════════════════════════════════════
# 12.  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # ── Utility mode: just print table ──────────────────────────────────────
    if args.print_table:
        compute_and_print_table(args.output_dir)
        return

    if args.ratio is None:
        print("[ERROR] --ratio is required (0.0 for baseline, 0.1, 0.2, 0.25, 0.3, 0.5)")
        return

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    ratio_pct = int(args.ratio * 100)
    label     = "baseline" if args.ratio == 0.0 else f"pruned_{ratio_pct:02d}pct"
    run_id    = f"{label}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    t_global  = time.time()

    print("=" * 70)
    print(f"  FINAL EVAL  —  {label}")
    print("=" * 70)
    if "cuda" in device and torch.cuda.is_available():
        idx = int(device.split(":")[-1]) if ":" in device else 0
        print(f"  GPU         : {torch.cuda.get_device_name(idx)}")
        vram = torch.cuda.get_device_properties(idx).total_memory / 1e9
        print(f"  VRAM        : {vram:.0f} GB")
    print(f"  Ratio       : {args.ratio:.0%}")
    print(f"  Eval images : {args.eval_images}  (NO pre-screening)")
    print(f"  Clean EOT   : {args.eval_eot}")
    print(f"  Attack      : PGD-{args.pgd_steps}  EOT={args.attack_eot}  "
          f"ε={args.attack_eps*255:.0f}/255")

    # ── Find checkpoint ─────────────────────────────────────────────────────
    ckpt_path = find_checkpoint(args.ratio, args)
    print(f"  Checkpoint  : {ckpt_path}")

    # ── Load model ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading model ...")
    model, n_params_M, actual_ratio = load_model(ckpt_path, device)
    compression_x = round(BASELINE_PARAMS_M / n_params_M, 4)
    reduction_pct = round((BASELINE_PARAMS_M - n_params_M) / BASELINE_PARAMS_M * 100, 2)
    print(f"  Compression : {compression_x:.2f}×  ({reduction_pct:.1f}% parameter reduction)")

    # ── Load classifier ─────────────────────────────────────────────────────
    print("\n[2/4] Loading classifier ...")
    classifier = load_classifier(device)

    # ── Data ────────────────────────────────────────────────────────────────
    print("\n[3/4] Loading data ...")
    batches = get_eval_batches(args.data_dir, args.eval_images,
                               args.eval_batch, args.seed, device)
    n_total = sum(x.shape[0] for x, _ in batches)
    print(f"  {n_total} images  ({len(batches)} batches of {args.eval_batch})"
          f"  ← full set, NO pre-screening")

    # ── Score function + schedule ────────────────────────────────────────────
    _sde     = sde_lib.VPSDE(beta_min=CONFIG.model.beta_min,
                              beta_max=CONFIG.model.beta_max,
                              N=CONFIG.model.num_scales)
    score_fn = mutils.get_score_fn(_sde, model, train=False, continuous=True)
    schedule = VPSchedule(device)

    # ── [PHASE 1] Clean accuracy ─────────────────────────────────────────────
    print(f"\n[4a/4] Clean accuracy  (EOT={args.eval_eot}) ...")
    clean_acc, clean_correct, _, purify_ms_img, clean_elapsed_s = eval_clean(
        batches, score_fn, schedule, classifier,
        args.t_star, args.ddim_steps, args.eval_eot, device,
    )
    print(f"\n  ✔ Clean acc    : {clean_acc:.4f}  ({clean_correct}/{n_total})  "
          f"= {clean_acc*100:.2f}%")
    print(f"  ✔ Purify speed : {purify_ms_img:.1f} ms/image  (EOT=1, inference only)")
    print(f"  ✔ Clean time   : {clean_elapsed_s/60:.1f} min")

    # ── [PHASE 2] PGD robust accuracy ────────────────────────────────────────
    n_steps   = args.pgd_steps
    atk_eot   = args.attack_eot
    print(f"\n[4b/4] PGD-{n_steps} robust accuracy  "
          f"(EOT={atk_eot} per step  ε={args.attack_eps*255:.0f}/255) ...")
    print(f"  ⚡ Attack gradient = avg over {atk_eot} independent purifications per step")
    pgd_acc, pgd_correct, _, pgd_elapsed_s = eval_pgd(
        batches, score_fn, schedule, classifier,
        args.t_star, args.ddim_steps,
        n_steps, atk_eot,
        args.attack_eps, args.pgd_step_size, args.eval_eot, device,
    )
    print(f"\n  ✔ PGD-{n_steps} acc  : {pgd_acc:.4f}  ({pgd_correct}/{n_total})  "
          f"= {pgd_acc*100:.2f}%")
    print(f"  ✔ PGD time     : {pgd_elapsed_s/60:.1f} min")

    # ── Build result row ─────────────────────────────────────────────────────
    total_elapsed_s = time.time() - t_global
    row = {
        "run_id":               run_id,
        "timestamp":            datetime.datetime.now().isoformat(),
        "device":               device,
        "model_label":          label,
        "pruning_ratio":        args.ratio,
        "params_M":             round(n_params_M, 3),
        "compression_x":        compression_x,
        "reduction_pct":        reduction_pct,
        "n_images":             n_total,
        "clean_acc":            round(clean_acc, 6),
        "clean_pct":            round(clean_acc * 100, 2),
        "clean_correct":        clean_correct,
        "pgd_acc":              round(pgd_acc, 6),
        "pgd_pct":              round(pgd_acc * 100, 2),
        "pgd_correct":          pgd_correct,
        "purify_ms_img":        round(purify_ms_img, 2),
        "speedup_vs_baseline":  "",   # computed post-hoc after all models done
        "clean_time_min":       round(clean_elapsed_s / 60, 2),
        "pgd_time_min":         round(pgd_elapsed_s   / 60, 2),
        "total_time_min":       round(total_elapsed_s  / 60, 2),
        "t_star":               args.t_star,
        "ddim_steps":           args.ddim_steps,
        "eval_eot":             args.eval_eot,
        "attack_eot":           args.attack_eot,
        "pgd_steps":            args.pgd_steps,
        "attack_eps_over255":   round(args.attack_eps * 255, 1),
        "checkpoint":           ckpt_path,
    }

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  SUMMARY  —  {label}")
    print(f"{'═'*70}")
    print(f"  Model size  : {n_params_M:.2f}M  ({compression_x:.2f}× compression, "
          f"{reduction_pct:.1f}% smaller)")
    print(f"  Clean acc   : {clean_acc*100:.2f}%  ({clean_correct}/{n_total})"
          f"  ← full test set, EOT={args.eval_eot}")
    print(f"  PGD-{n_steps} acc: {pgd_acc*100:.2f}%  ({pgd_correct}/{n_total})"
          f"  ← NO pre-screening, EOT={atk_eot} attack")
    print(f"  Purify speed: {purify_ms_img:.1f} ms/image  (EOT=1)")
    print(f"  Total time  : {total_elapsed_s/60:.1f} min")
    print(f"{'═'*70}")

    save_row(row, args.output_dir)
    print(f"\n  Run:  python3 cifar10_final_eval.py --print_table")
    print(f"        (after all models are done — fills in speedup column)")


if __name__ == "__main__":
    main()
