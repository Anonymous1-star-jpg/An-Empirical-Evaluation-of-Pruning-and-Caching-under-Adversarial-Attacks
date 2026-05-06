"""
cifar10_prune_score_sde.py  (v2 — manual group pruning, no torch-pruning MetaPruner)
======================================================================================
Structural pruning of the CIFAR-10 Score SDE (NCSN++ deep, VP-SDE) model.

Why manual pruning instead of torch-pruning MetaPruner?
  The NCSN++ forward uses `torch.cat([h, hs.pop()], dim=1)` for skip connections,
  where `hs` is a dynamic Python list.  torch-pruning's JIT tracer assigns None to
  these concat sizes → TypeError at dependency-graph build time.

  Since the NCSN++ architecture is fixed and we know ALL channel dimensions, we
  implement group-based pruning directly:
    • Group-128 : all layers carrying nf=128 channels (encoder level 0 + decoder level 0)
    • Group-256 : all layers carrying nf×2=256 channels (levels 1-3, bottleneck)

  We correctly handle all concat patterns in the decoder (see prune_resblock_biggan).

Strategy:
  1. Load pretrained checkpoint_8.pth  (EMA weights)
  2. Compute per-channel importance (weight-gradient Taylor OR L1 magnitude)
  3. Select keep_128 / keep_256 channel indices (round to multiples of 32)
  4. Apply pruning to every module in model.all_modules
  5. Sanity check: forward pass on random input
  6. Save pruned model

Usage:
  python3 cifar10_prune_score_sde.py \\
      --ckpt checkpoints/score_sde/checkpoint_8.pth \\
      --pruning_ratio 0.25 \\
      --calib_images 1024 --calib_batch_size 16 \\
      --output_dir checkpoints/score_sde/pruned/ \\
      --device cuda:7

  # Quick pruning without calibration (L1 magnitude)
  python3 cifar10_prune_score_sde.py --pruning_ratio 0.25 --no_importance --device cuda:7
"""

import os, sys, math, json, time, argparse, warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent))

try:
    from score_sde.models import utils as mutils
    from score_sde.models.ema import ExponentialMovingAverage
    from score_sde import sde_lib
    from score_sde.models.layerspp import ResnetBlockBigGANpp, AttnBlockpp
    from score_sde.models.layers import NIN
except ImportError as e:
    print(f"[ERROR] score_sde not found: {e}\nRun: bash setup_cifar10.sh")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# 1.  CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt",             default="checkpoints/score_sde/checkpoint_8.pth")
    p.add_argument("--output_dir",       default="checkpoints/score_sde/pruned/")
    p.add_argument("--pruning_ratio",    type=float, default=0.25,
                   help="Fraction of channels to remove (0.10 / 0.25 / 0.50)")
    p.add_argument("--calib_images",     type=int,   default=1024)
    p.add_argument("--calib_batch_size", type=int,   default=16)
    p.add_argument("--data_dir",         default="./data")
    p.add_argument("--device",           default="cuda:7")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--no_importance",    action="store_true",
                   help="Use L1 magnitude instead of Taylor importance (faster, no data)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
# 2.  CONFIG
# ═══════════════════════════════════════════════════════════════

class _Bunch(dict):
    def __getattr__(self, k):
        try:
            v = self[k]; return _Bunch(v) if isinstance(v, dict) else v
        except KeyError: raise AttributeError(k)
    def __setattr__(self, k, v): self[k] = v

CIFAR10_CONFIG = _Bunch({
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


# ═══════════════════════════════════════════════════════════════
# 3.  MODEL LOADING
# ═══════════════════════════════════════════════════════════════

def load_score_sde_model(ckpt_path, device):
    config = CIFAR10_CONFIG
    print(f"[Model] Creating NCSN++ ...")
    model = mutils.create_model(config)

    from score_sde.losses import get_optimizer
    optimizer = get_optimizer(config, model.parameters())
    ema       = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)

    print(f"[Model] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    ema.load_state_dict(ckpt["ema"])

    ema.copy_to(model.parameters())          # use EMA weights for pruning
    model = model.to(device).eval()

    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model] Loaded — {n:.1f}M params  step={ckpt.get('step', '?')}")
    return model, config


# ═══════════════════════════════════════════════════════════════
# 4.  CALIBRATION DATA
# ═══════════════════════════════════════════════════════════════

def get_calib_loader(data_dir, n_images, batch_size, seed=42):
    tf = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tf)
    g  = torch.Generator(); g.manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n_images].tolist()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idx),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"[Calib] {n_images} CIFAR-10 images  batch={batch_size}  ({len(loader)} batches)")
    return loader


# ═══════════════════════════════════════════════════════════════
# 5.  DSM LOSS  (for Taylor importance)
# ═══════════════════════════════════════════════════════════════

def build_sde(config):
    return sde_lib.VPSDE(beta_min=config.model.beta_min,
                         beta_max=config.model.beta_max,
                         N=config.model.num_scales)

def dsm_loss(model, sde, x, eps=1e-5):
    B  = x.shape[0]
    t  = torch.rand(B, device=x.device) * (1.0 - eps) + eps
    mean, std = sde.marginal_prob(x, t)
    z  = torch.randn_like(x)
    xt = mean + std[:, None, None, None] * z
    score_fn = mutils.get_score_fn(sde, model, train=True, continuous=True)
    score    = score_fn(xt, t)
    return torch.mean(torch.sum((score * std[:, None, None, None] + z) ** 2, dim=(1,2,3)))


# ═══════════════════════════════════════════════════════════════
# 6.  IMPORTANCE ESTIMATION
# ═══════════════════════════════════════════════════════════════

def compute_l1_importance(model):
    """L1 magnitude per output channel, averaged across ResBlock Conv_1 layers."""
    imp_128, imp_256 = [], []
    for m in model.all_modules:
        if isinstance(m, ResnetBlockBigGANpp):
            # Conv_1 is the block's output conv — best representative
            w   = m.Conv_1.weight.data                   # (out_ch, in_ch, kH, kW)
            imp = w.abs().sum(dim=[1, 2, 3])             # (out_ch,)
            (imp_128 if m.out_ch == 128 else imp_256).append(imp)
    avg128 = torch.stack(imp_128).mean(0) if imp_128 else torch.ones(128)
    avg256 = torch.stack(imp_256).mean(0) if imp_256 else torch.ones(256)
    print(f"[Importance] L1 magnitude: found {len(imp_128)} 128-ch blocks, "
          f"{len(imp_256)} 256-ch blocks")
    return avg128, avg256


def compute_taylor_importance(model, sde, calib_loader, device):
    """
    Weight-gradient Taylor importance: I(c) = |∂L/∂W_c × W_c| per output channel c.

    Collected AFTER each loss.backward() — avoids the bug of checking m.weight.grad
    inside a register_hook callback (where it may not yet be populated).
    """
    model.train()
    accum = {}   # name → accumulated (out_ch,) tensor

    n_batches = len(calib_loader)
    t0 = time.time()

    for bi, (x, _) in enumerate(calib_loader):
        x = x.to(device)
        model.zero_grad()
        loss = dsm_loss(model, sde, x)
        loss.backward()

        # Collect AFTER backward — m.weight.grad is now filled
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d) and m.weight.grad is not None:
                imp = (m.weight.data * m.weight.grad).abs().sum(dim=[1, 2, 3])
                if name in accum:
                    accum[name] = accum[name] + imp.detach()
                else:
                    accum[name] = imp.detach().clone()

        if (bi + 1) % 8 == 0 or bi == n_batches - 1:
            print(f"  [Taylor] batch {bi+1}/{n_batches}  loss={loss.item():.4f}  "
                  f"layers={len(accum)}  [{time.time()-t0:.1f}s]")

    for k in accum:
        accum[k] = accum[k] / n_batches

    model.eval()
    model.zero_grad()
    print(f"[Taylor] Done — {len(accum)} Conv2d layers with importance scores")

    # Map per-layer importance back to 128-ch / 256-ch group averages
    imp_128, imp_256 = [], []
    for m in model.all_modules:
        if isinstance(m, ResnetBlockBigGANpp):
            # Find name of Conv_1 in the flat named_modules dict
            for name, sub in model.named_modules():
                if sub is m.Conv_1 and name in accum:
                    (imp_128 if m.out_ch == 128 else imp_256).append(accum[name])
                    break

    avg128 = torch.stack(imp_128).mean(0) if imp_128 else None
    avg256 = torch.stack(imp_256).mean(0) if imp_256 else None

    if avg128 is None:
        print("[WARN] No 128-ch Taylor importance found — falling back to L1")
        avg128, avg256 = compute_l1_importance(model)
    elif avg256 is None:
        print("[WARN] No 256-ch Taylor importance found — falling back to L1 for 256-ch")
        _, avg256 = compute_l1_importance(model)

    return avg128, avg256


def select_keep_indices(importance, n_total, pruning_ratio, round_to=32):
    """Select top-(1-ratio) channels by importance, rounded to multiple of round_to."""
    n_keep = int(n_total * (1 - pruning_ratio))
    n_keep = max(round_to, round(n_keep / round_to) * round_to)
    n_keep = min(n_keep, n_total)
    keep   = importance.argsort(descending=True)[:n_keep].sort().values
    print(f"  channels: {n_total} → {n_keep}  (keep {n_keep/n_total:.0%})")
    return keep


# ═══════════════════════════════════════════════════════════════
# 7.  LOW-LEVEL PRUNING PRIMITIVES
# ═══════════════════════════════════════════════════════════════

def _k(t, dev): return t.to(dev)  # short helper

def prune_conv2d_output(conv, keep_idx):
    k = _k(keep_idx, conv.weight.device)
    conv.weight = nn.Parameter(conv.weight.data[k])
    if conv.bias is not None:
        conv.bias = nn.Parameter(conv.bias.data[k])
    conv.out_channels = len(k)

def prune_conv2d_input(conv, keep_idx):
    k = _k(keep_idx, conv.weight.device)
    conv.weight = nn.Parameter(conv.weight.data[:, k])
    conv.in_channels = len(k)

def prune_linear_output(lin, keep_idx):
    k = _k(keep_idx, lin.weight.device)
    lin.weight = nn.Parameter(lin.weight.data[k])
    if lin.bias is not None:
        lin.bias = nn.Parameter(lin.bias.data[k])
    lin.out_features = len(k)

def prune_nin(nin, keep_in, keep_out):
    """NIN.W: (in_dim, num_units). prune axes 0 (in) and 1 (out)."""
    dev = nin.W.device
    ki, ko = _k(keep_in, dev), _k(keep_out, dev)
    nin.W = nn.Parameter(nin.W.data[ki][:, ko])
    nin.b = nn.Parameter(nin.b.data[ko])

def prune_groupnorm(gn, keep_idx):
    """Update GroupNorm weight/bias/num_channels/num_groups after pruning."""
    k   = _k(keep_idx, gn.weight.device)
    gn.weight = nn.Parameter(gn.weight.data[k])
    gn.bias   = nn.Parameter(gn.bias.data[k])
    n_ch      = len(k)
    gn.num_channels = n_ch
    # Largest divisor of n_ch that is ≤ 32  (ensures valid GroupNorm)
    best_g = max(g for g in range(1, min(33, n_ch + 1)) if n_ch % g == 0)
    gn.num_groups = best_g


# ═══════════════════════════════════════════════════════════════
# 8.  MODULE-LEVEL PRUNING  (BigGAN ResBlock & AttnBlock)
# ═══════════════════════════════════════════════════════════════

def prune_resblock_biggan(m: ResnetBlockBigGANpp, keep_128: torch.Tensor,
                          keep_256: torch.Tensor):
    """
    Prune a single ResnetBlockBigGANpp in-place.

    The NCSN++ decoder uses torch.cat([h, hs.pop()]) for skip connections.
    After pruning, the concatenated input channels change shape.  We compute
    the correct keep_idx for the combined input depending on which channels
    were pruned from each source:

    Block (in_ch, out_ch) → source pattern:
      (128, 128)  — same-level 128-ch block        (encoder, bottleneck, decoder)
      (128, 128)  — same as above but with up/down flag
      (256, 256)  — same-level 256-ch block
      (128, 256)  — level 1 first encoder block (128→256 channel transition)
      (512, 256)  — decoder levels 1-3: concat[256, 256] → out 256
      (384, 256)  — decoder level 1 LAST block: concat[256_dec, 128_level0_skip] → 256
      (384, 128)  — decoder level 0 block 0: concat[256(dec upsample), 128(skip)] → 128
      (256, 128)  — decoder level 0 blocks 1-8: concat[128, 128] → 128
    """
    in_ch  = m.in_ch
    out_ch = m.out_ch
    n128   = len(keep_128)
    n256   = len(keep_256)
    dev    = next(m.parameters()).device

    # ── Determine OUTPUT keep indices ──────────────────────────────────────
    if out_ch == 128:
        keep_out = keep_128
    elif out_ch == 256:
        keep_out = keep_256
    else:
        print(f"  [WARN] Unknown out_ch={out_ch} — skipping block"); return

    # ── Determine INPUT keep indices (handles concat patterns) ─────────────
    if in_ch == 128:
        # Simple same-level 128-ch block
        keep_in = keep_128

    elif in_ch == 256 and out_ch == 256:
        # Same-level 256-ch block (encoder levels 1-3 blocks 2-8)
        keep_in = keep_256

    elif in_ch == 128 and out_ch == 256:
        # Level 1 FIRST encoder block: 128-ch input → 256-ch output
        keep_in = keep_128

    elif in_ch == 512 and out_ch == 256:
        # Decoder levels 1-3: concat[256_dec, 256_skip]
        # After pruning: [keep_256, keep_256+256] → both halves shrink equally
        keep_a  = keep_256
        keep_b  = keep_256 + 256   # second half offset
        keep_in = torch.cat([keep_a, keep_b]).sort().values

    elif in_ch == 384 and out_ch == 256:
        # Decoder level 1 last block: asymmetric concat[256_dec, 128_level0_down_skip]
        # The hs_c stack pops the level-0 downsampler's 128-ch output here.
        keep_a  = keep_256
        keep_b  = keep_128 + 256   # second half offset
        keep_in = torch.cat([keep_a, keep_b]).sort().values

    elif in_ch == 384 and out_ch == 128:
        # Decoder level 0 block 0: concat[256_dec_upsample, 128_skip]
        # Asymmetric: first 256 use keep_256, last 128 use keep_128
        keep_a  = keep_256
        keep_b  = keep_128 + 256   # second half offset
        keep_in = torch.cat([keep_a, keep_b]).sort().values

    elif in_ch == 256 and out_ch == 128:
        # Decoder level 0 blocks 1-8: concat[128_dec, 128_skip]
        # Symmetric: both halves use keep_128
        keep_a  = keep_128
        keep_b  = keep_128 + 128   # second half offset
        keep_in = torch.cat([keep_a, keep_b]).sort().values

    else:
        print(f"  [WARN] Unhandled block in_ch={in_ch} out_ch={out_ch} — skipping")
        return

    keep_in  = keep_in.to(dev)
    keep_out = keep_out.to(dev)

    # ── Apply pruning ──────────────────────────────────────────────────────
    prune_groupnorm(m.GroupNorm_0, keep_in)           # GroupNorm on input
    prune_conv2d_input(m.Conv_0, keep_in)             # Conv_0 input channels
    prune_conv2d_output(m.Conv_0, keep_out)           # Conv_0 output channels
    if hasattr(m, 'Dense_0'):
        prune_linear_output(m.Dense_0, keep_out)      # time-conditioning proj
    prune_groupnorm(m.GroupNorm_1, keep_out)          # GroupNorm on Conv_0 output
    prune_conv2d_input(m.Conv_1, keep_out)            # Conv_1 input = Conv_0 output
    prune_conv2d_output(m.Conv_1, keep_out)           # Conv_1 output = block output

    # Shortcut (Conv_2) — present when in_ch != out_ch OR up OR down
    if hasattr(m, 'Conv_2'):
        prune_conv2d_input(m.Conv_2, keep_in)
        prune_conv2d_output(m.Conv_2, keep_out)

    # Update stored dimensions (used for sanity in forward's assert)
    m.in_ch  = len(keep_in)
    m.out_ch = len(keep_out)


def prune_attn_block(m: AttnBlockpp, keep: torch.Tensor):
    """Prune AttnBlockpp — all NIN(ch,ch) and GroupNorm use the same keep set."""
    prune_groupnorm(m.GroupNorm_0, keep)
    prune_nin(m.NIN_0, keep, keep)
    prune_nin(m.NIN_1, keep, keep)
    prune_nin(m.NIN_2, keep, keep)
    prune_nin(m.NIN_3, keep, keep)


# ═══════════════════════════════════════════════════════════════
# 9.  FULL MODEL PRUNING
# ═══════════════════════════════════════════════════════════════

def prune_ncsnpp(model, keep_128: torch.Tensor, keep_256: torch.Tensor):
    """
    Prune the full NCSN++ model in-place using pre-computed keep indices.

    Architecture overview (CIFAR-10 deep, biggan, positional, progressive=none):
      all_modules[0]:  Linear(nf,   nf*4)      ← time emb MLP, SKIP
      all_modules[1]:  Linear(nf*4, nf*4)      ← time emb MLP, SKIP
      all_modules[2]:  Conv2d(3, nf=128)        ← input conv, prune OUTPUT only
      all_modules[3…]: encoder ResBlocks + AttnBlocks + downsampler ResBlocks
      ...              bottleneck ResBlocks + AttnBlock
      ...              decoder ResBlocks + AttnBlocks + upsampler ResBlocks
      all_modules[-2]: GroupNorm(128)           ← final norm, prune with keep_128
      all_modules[-1]: Conv2d(128, 3)           ← output conv, prune INPUT only
    """
    mods = list(model.all_modules)
    n    = len(mods)
    n_biggan = 0
    n_attn   = 0
    n_skip   = 0

    for i, m in enumerate(mods):
        # ── Time embedding MLPs: skip entirely ─────────────────────────────
        if i < 2:
            continue

        # ── Input conv: Conv2d(3 → 128), prune output only ─────────────────
        if i == 2:
            assert isinstance(m, nn.Conv2d), f"Expected Conv2d at idx 2, got {type(m)}"
            prune_conv2d_output(m, keep_128)
            continue

        # ── Final GroupNorm (idx -2) ────────────────────────────────────────
        if i == n - 2:
            assert isinstance(m, nn.GroupNorm)
            prune_groupnorm(m, keep_128)
            continue

        # ── Final output Conv (idx -1): prune input only, output MUST stay 3 ─
        if i == n - 1:
            assert isinstance(m, nn.Conv2d)
            prune_conv2d_input(m, keep_128)
            continue

        # ── BigGAN ResBlock ─────────────────────────────────────────────────
        if isinstance(m, ResnetBlockBigGANpp):
            prune_resblock_biggan(m, keep_128, keep_256)
            n_biggan += 1
            continue

        # ── AttnBlock (all at 256-ch: level 1 encoder/decoder + bottleneck) ─
        if isinstance(m, AttnBlockpp):
            prune_attn_block(m, keep_256)
            n_attn += 1
            continue

        # ── Everything else (Upsample interpolation, etc.) — no params ─────
        n_skip += 1

    print(f"[Pruned] {n_biggan} BigGAN ResBlocks  {n_attn} AttnBlocks  "
          f"{n_skip} skipped (no-param modules)")


# ═══════════════════════════════════════════════════════════════
# 10. SANITY CHECK
# ═══════════════════════════════════════════════════════════════

def sanity_check(model, device):
    model.eval()
    with torch.no_grad():
        x = torch.randn(2, 3, 32, 32, device=device)
        t = torch.tensor([0.3, 0.7], device=device)
        try:
            out = model(x, t)
            assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
            print(f"[Sanity]  output shape {out.shape}  "
                  f"mean={out.mean():.4f}  std={out.std():.4f}")
            return True
        except Exception as e:
            print(f"[Sanity]  {e}")
            import traceback; traceback.print_exc()
            return False


# ═══════════════════════════════════════════════════════════════
# 11. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 60)
    print(f"  Score SDE Pruning  (ratio={args.pruning_ratio:.0%})")
    print("=" * 60)
    if "cuda" in device and torch.cuda.is_available():
        idx = int(device.split(":")[-1]) if ":" in device else 0
        print(f"  GPU : {torch.cuda.get_device_name(idx)}")

    # ── Load model ─────────────────────────────────────────────
    model, config = load_score_sde_model(args.ckpt, device)
    sde = build_sde(config)
    params_before = sum(p.numel() for p in model.parameters())

    # ── Compute importance ─────────────────────────────────────
    if args.no_importance:
        print("\n[Importance] Using L1 magnitude (no calibration data) ...")
        imp128, imp256 = compute_l1_importance(model)
    else:
        print(f"\n[Importance] Computing Taylor importance on {args.calib_images} images ...")
        loader = get_calib_loader(args.data_dir, args.calib_images,
                                  args.calib_batch_size, args.seed)
        imp128, imp256 = compute_taylor_importance(model, sde, loader, device)

    # ── Select keep channels ────────────────────────────────────
    print(f"\n[Select] Selecting channels to keep ...")
    print(f"  128-ch group:", end=" ")
    keep_128 = select_keep_indices(imp128, 128, args.pruning_ratio)
    print(f"  256-ch group:", end=" ")
    keep_256 = select_keep_indices(imp256, 256, args.pruning_ratio)

    # ── Apply pruning ───────────────────────────────────────────
    print(f"\n[Prune] Applying structural pruning ...")
    t0 = time.time()
    prune_ncsnpp(model, keep_128.to(device), keep_256.to(device))
    print(f"[Prune] Done in {time.time()-t0:.1f}s")

    # ── Sanity check ────────────────────────────────────────────
    ok = sanity_check(model, device)
    if not ok:
        print("[ERROR] Sanity check failed — do not proceed with fine-tuning!")
        return

    # ── Report stats ────────────────────────────────────────────
    params_after = sum(p.numel() for p in model.parameters())
    actual_red   = 1 - params_after / params_before
    compression  = params_before / max(params_after, 1)
    print(f"\n{'─'*55}")
    print(f"  Target ratio     : {args.pruning_ratio:.0%}")
    print(f"  Actual reduction : {actual_red:.1%}")
    print(f"  Params before    : {params_before/1e6:.2f}M")
    print(f"  Params after     : {params_after/1e6:.2f}M")
    print(f"  Compression      : {compression:.2f}x")
    print(f"{'─'*55}")

    stats = {
        "pruning_ratio": args.pruning_ratio, "actual_reduction": actual_red,
        "params_before_M": params_before/1e6, "params_after_M": params_after/1e6,
        "compression_x": compression,
        "keep_128": keep_128.tolist(), "keep_256": keep_256.tolist(),
    }

    # ── Save ────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    tag  = f"pruned_{int(args.pruning_ratio*100):02d}pct"
    path = os.path.join(args.output_dir, f"{tag}.pth")
    meta = os.path.join(args.output_dir, f"{tag}_meta.json")

    torch.save({
        "model":         model.state_dict(),
        "pruning_ratio": args.pruning_ratio,
        "keep_128":      keep_128.cpu(),
        "keep_256":      keep_256.cpu(),
        "stats":         stats,
    }, path)
    with open(meta, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"\n[Save] {path}")
    print(f"[Save] {meta}")
    print(f"\n Next: fine-tune the pruned model:")
    print(f"   python3 cifar10_finetune_score_sde.py \\")
    print(f"       --ckpt {path} --steps 20000 --device {device}")


if __name__ == "__main__":
    main()
