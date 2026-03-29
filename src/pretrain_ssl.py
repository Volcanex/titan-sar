"""SimCLR self-supervised pretraining for the SAR tile encoder.

Trains an EfficientNet-B4 encoder on all unlabelled SAR tiles using
contrastive learning (NT-Xent / InfoNCE), then saves encoder weights
compatible with segmentation_models_pytorch.

Usage
-----
    python -m src.pretrain_ssl \
        --data-dir /workspace/data \
        --epochs 150 --batch-size 128 --lr 3e-4 \
        --output-path models/ssl_encoder_effb4.pth

Designed for ~10k 256x256 single-channel float32 .npy tiles.
On a single A100/4090 this finishes in roughly 1-2 hours at 150 epochs.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm


# ── Augmentation helpers (pure torch, no albumentations needed) ────────

class SARPairTransform:
    """Produce two differently-augmented views of a single SAR tile.

    All ops work on (1, H, W) float tensors.
    """

    def __init__(self, crop_size=224):
        self.crop_size = crop_size

    def _augment_one(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to a (1, H, W) tensor."""
        _, h, w = x.shape

        # Random crop
        cs = self.crop_size
        if h > cs and w > cs:
            top = torch.randint(0, h - cs, (1,)).item()
            left = torch.randint(0, w - cs, (1,)).item()
            x = x[:, top:top + cs, left:left + cs]
        elif h != cs or w != cs:
            x = F.interpolate(x.unsqueeze(0), size=(cs, cs), mode="bilinear",
                              align_corners=False).squeeze(0)

        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            x = x.flip(-1)

        # Random vertical flip
        if torch.rand(1).item() > 0.5:
            x = x.flip(-2)

        # Random 90-degree rotation (0, 1, 2, or 3 times)
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            x = torch.rot90(x, k, dims=(-2, -1))

        # Gaussian noise
        if torch.rand(1).item() > 0.5:
            std = torch.rand(1).item() * 0.1
            x = x + torch.randn_like(x) * std

        # Brightness jitter (multiplicative + additive)
        if torch.rand(1).item() > 0.5:
            gain = 0.8 + torch.rand(1).item() * 0.4   # [0.8, 1.2]
            bias = (torch.rand(1).item() - 0.5) * 0.2  # [-0.1, 0.1]
            x = x * gain + bias

        return x

    def __call__(self, x: torch.Tensor):
        return self._augment_one(x.clone()), self._augment_one(x.clone())


# ── Dataset ───────────────────────────────────────────────────────────

class SARTileDataset(Dataset):
    """Loads ALL .npy SAR tiles from a directory (no labels needed)."""

    def __init__(self, sar_dir: str | Path, transform=None):
        self.sar_dir = Path(sar_dir)
        self.files = sorted(self.sar_dir.glob("*.npy"))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npy files found in {self.sar_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx]).astype(np.float32)
        x = torch.from_numpy(arr)
        if x.ndim == 2:
            x = x.unsqueeze(0)  # (1, H, W)

        if self.transform is not None:
            v1, v2 = self.transform(x)
            # Replicate to 3 channels for EfficientNet
            v1 = v1.expand(3, -1, -1).contiguous()
            v2 = v2.expand(3, -1, -1).contiguous()
            return v1, v2

        x = x.expand(3, -1, -1).contiguous()
        return x, x


# ── NT-Xent (InfoNCE) loss ──────────────────────────────────────────

class NTXentLoss(nn.Module):
    """Normalised Temperature-scaled Cross-Entropy Loss (SimCLR)."""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i, z_j: (B, D) L2-normalised embeddings for the two views.
        """
        B = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
        sim = torch.mm(z, z.T) / self.temperature  # (2B, 2B)

        # Mask out self-similarity on the diagonal
        mask = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=sim.device),
            torch.arange(0, B, device=sim.device),
        ])  # (2B,)

        return F.cross_entropy(sim, labels)


# ── Projection head ─────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """Two-layer MLP projection head (SimCLR v2 style)."""

    def __init__(self, in_dim, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SimCLR pretraining for SAR encoder")
    parser.add_argument("--data-dir", type=str, default="/workspace/data",
                        help="Root data directory")
    parser.add_argument("--sar-dir-name", type=str, default="sar_tiles",
                        help="Subdirectory under data-dir/processed/ with .npy tiles")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--crop-size", type=int, default=224,
                        help="Random crop size for augmented views")
    parser.add_argument("--proj-dim", type=int, default=128,
                        help="Projection head output dimension")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-path", type=str,
                        default="models/ssl_encoder_effb4.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Print loss every N epochs")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──
    sar_dir = Path(args.data_dir) / "processed" / args.sar_dir_name
    transform = SARPairTransform(crop_size=args.crop_size)
    dataset = SARTileDataset(sar_dir, transform=transform)
    print(f"Loaded {len(dataset)} SAR tiles from {sar_dir}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    steps_per_epoch = len(loader)
    print(f"Batch size: {args.batch_size}, Steps/epoch: {steps_per_epoch}")

    # ── Encoder ──
    # Use timm to create the same EfficientNet-B4 that smp uses internally
    encoder = timm.create_model(
        "tf_efficientnet_b4",
        pretrained=False,
        in_chans=3,
        num_classes=0,       # remove classifier, get features
        global_pool="avg",
    ).to(device)

    encoder_dim = encoder.num_features  # 1792 for EfficientNet-B4
    print(f"Encoder output dim: {encoder_dim}")

    # ── Projection head ──
    proj_head = ProjectionHead(encoder_dim, hidden_dim=2048,
                               out_dim=args.proj_dim).to(device)

    # ── Optimiser + scheduler ──
    params = list(encoder.parameters()) + list(proj_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    criterion = NTXentLoss(temperature=args.temperature).to(device)

    total_params = sum(p.numel() for p in params)
    print(f"Trainable parameters: {total_params:,}")
    print(f"Starting SimCLR pretraining for {args.epochs} epochs\n")

    # ── Training loop ──
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        proj_head.train()
        epoch_loss = 0.0

        for v1, v2 in loader:
            v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)

            # Forward
            h1 = encoder(v1)  # (B, encoder_dim)
            h2 = encoder(v2)
            z1 = proj_head(h1)
            z2 = proj_head(h2)

            loss = criterion(z1, z2)

            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / steps_per_epoch

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            elapsed = time.time() - t_start
            lr_now = optimizer.param_groups[0]["lr"]
            eta = elapsed / epoch * (args.epochs - epoch)
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"Loss: {avg_loss:.4f} | LR: {lr_now:.6f} | "
                  f"Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")

    total_time = time.time() - t_start
    print(f"\nPretraining complete in {total_time/60:.1f} minutes.")

    # ── Save encoder weights ──
    # Convert the timm state dict to the format smp expects.
    # smp's EfficientNet encoder wraps timm, prefixing keys with "model."
    # (for smp >= 0.3.x) or using them directly. We save the raw timm
    # state dict, and the fine-tuning config will load it via a custom
    # loading function in train.py. The simplest approach: save with a
    # known prefix convention so we can remap at load time.
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save raw timm encoder weights (no projection head)
    torch.save(encoder.state_dict(), output_path)
    print(f"Encoder weights saved to {output_path}")
    print(f"  State dict keys: {len(encoder.state_dict())}")

    # Also save a metadata sidecar
    import json
    meta = {
        "method": "simclr",
        "encoder": "tf_efficientnet_b4",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "temperature": args.temperature,
        "crop_size": args.crop_size,
        "proj_dim": args.proj_dim,
        "num_tiles": len(dataset),
        "training_time_minutes": round(total_time / 60, 1),
    }
    meta_path = output_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
