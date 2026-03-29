"""Standalone training script for Titan SAR segmentation — runs on GPU."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp
import yaml
import albumentations as A


# ── Dataset ─────────────────────────────────────────────────────────────
class TitanDataset(torch.utils.data.Dataset):
    # Normalization stats computed from full tile sets
    NORM_STATS = {
        "sar_tiles":    (0.5946, 0.2249),
        "nldsar_tiles": (0.2671, 0.2348),
    }

    def __init__(self, tile_ids, sar_dir, label_dir, transform=None,
                 nldsar_dir=None):
        """
        If nldsar_dir is given, prefer NLDSAR tiles where available,
        fall back to sar_dir otherwise.
        """
        self.sar_dir = Path(sar_dir)
        self.label_dir = Path(label_dir)
        self.nldsar_dir = Path(nldsar_dir) if nldsar_dir else None
        self.transform = transform

        # Determine per-tile source (NLDSAR preferred, SAR fallback)
        self.tile_ids = []
        self.tile_sources = {}  # tid -> ("sar" | "nldsar")
        for tid in sorted(tile_ids):
            if self.nldsar_dir and (self.nldsar_dir / f"{tid}.npy").exists():
                self.tile_ids.append(tid)
                self.tile_sources[tid] = "nldsar"
            elif (self.sar_dir / f"{tid}.npy").exists():
                self.tile_ids.append(tid)
                self.tile_sources[tid] = "sar"

        # Stats lookup
        sar_dir_name = Path(sar_dir).name
        self.sar_mean, self.sar_std = self.NORM_STATS.get(sar_dir_name, (0.5946, 0.2249))
        self.nldsar_mean, self.nldsar_std = self.NORM_STATS.get("nldsar_tiles", (0.2671, 0.2348))

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, idx):
        tid = self.tile_ids[idx]
        source = self.tile_sources[tid]

        if source == "nldsar":
            sar = np.load(self.nldsar_dir / f"{tid}.npy").astype(np.float32)
            sar = (sar - self.nldsar_mean) / self.nldsar_std
        else:
            sar = np.load(self.sar_dir / f"{tid}.npy").astype(np.float32)
            sar = (sar - self.sar_mean) / self.sar_std

        label = np.load(self.label_dir / f"{tid}.npy").astype(np.int64)

        if self.transform:
            t = self.transform(image=sar, mask=label)
            sar, label = t["image"], t["mask"]

        if isinstance(sar, np.ndarray):
            sar = torch.from_numpy(sar)
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)

        if sar.ndim == 2:
            sar = sar.unsqueeze(0)

        # Replicate single channel to 3 for pretrained encoders
        sar = sar.expand(3, -1, -1).contiguous()

        return sar, label


# ── Loss functions ──────────────────────────────────────────────────────
def get_loss(loss_name, class_weights=None, device="cuda", num_classes=6):
    if loss_name == "focal":
        return smp.losses.FocalLoss(mode="multiclass", ignore_index=255)
    elif loss_name == "dice":
        return smp.losses.DiceLoss(mode="multiclass", ignore_index=255)
    elif loss_name == "ce":
        w = torch.tensor(class_weights, dtype=torch.float32).to(device) if class_weights else None
        return nn.CrossEntropyLoss(weight=w, ignore_index=255)
    elif loss_name == "focal+dice":
        focal = smp.losses.FocalLoss(mode="multiclass", ignore_index=255)
        dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=255)

        class CombinedLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.focal = focal
                self.dice = dice

            def forward(self, pred, target):
                return self.focal(pred, target) + self.dice(pred, target)

        return CombinedLoss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


# ── Metrics ─────────────────────────────────────────────────────────────
def compute_iou(pred, target, num_classes, ignore_index=255):
    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        target_c = (target == c) & (target != ignore_index)
        inter = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union > 0:
            ious.append(inter / union)
    return np.mean(ious) if ious else 0.0


def per_class_iou(pred, target, num_classes, ignore_index=255):
    results = {}
    class_names = ["plains", "dunes", "hummocky", "lakes_seas", "labyrinth", "craters"]
    for c in range(num_classes):
        pred_c = pred == c
        target_c = (target == c) & (target != ignore_index)
        inter = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        name = class_names[c] if c < len(class_names) else f"class_{c}"
        results[name] = inter / union if union > 0 else 0.0
    return results


# ── Training loop ──────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    miou = compute_iou(all_preds, all_targets, num_classes)
    cls_iou = per_class_iou(all_preds, all_targets, num_classes)
    return total_loss / len(loader.dataset), miou, cls_iou, all_preds


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="/workspace/data")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--encoder-weights", type=str, default=None,
                        help="Override encoder weights (e.g. 'imagenet', 'None' for random)")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze encoder weights (linear probing)")
    parser.add_argument("--sar-dir-name", type=str, default="sar_tiles",
                        help="Subdirectory name for SAR tiles (e.g. 'nldsar_tiles')")
    parser.add_argument("--nldsar", action="store_true",
                        help="Use NLDSAR tiles where available, fall back to SAR")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(args.data_dir)
    run_name = args.run_name or Path(args.config).stem
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {cfg}")
    print(f"Run name: {run_name}")

    # Override encoder weights if specified
    if args.encoder_weights is not None:
        if args.encoder_weights.lower() == "none":
            cfg["encoder_weights"] = None
        else:
            cfg["encoder_weights"] = args.encoder_weights

    # Load splits
    split_path = data_dir / "splits" / "split_v1.json"
    with open(split_path) as f:
        split_map = json.load(f)

    sar_dir = data_dir / "processed" / args.sar_dir_name
    label_dir = data_dir / "processed" / "label_tiles"

    train_ids = [k for k, v in split_map.items() if v == "train"]
    val_ids = [k for k, v in split_map.items() if v == "val"]
    test_ids = [k for k, v in split_map.items() if v == "test"]

    print(f"Data: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
    print(f"SAR dir: {sar_dir}")

    # Augmentations
    train_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GaussNoise(p=0.3),
    ])

    nldsar_dir = data_dir / "processed" / "nldsar_tiles" if args.nldsar else None
    if nldsar_dir:
        print(f"NLDSAR dir: {nldsar_dir}")

    train_ds = TitanDataset(train_ids, sar_dir, label_dir, transform=train_aug,
                            nldsar_dir=nldsar_dir)
    val_ds = TitanDataset(val_ids, sar_dir, label_dir, nldsar_dir=nldsar_dir)
    test_ds = TitanDataset(test_ids, sar_dir, label_dir, nldsar_dir=nldsar_dir)

    if nldsar_dir:
        nldsar_count = sum(1 for v in train_ds.tile_sources.values() if v == "nldsar")
        print(f"  Train: {nldsar_count}/{len(train_ds)} tiles from NLDSAR")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=4, pin_memory=True)

    # Model
    encoder_weights = cfg.get("encoder_weights", "imagenet")
    print(f"Encoder weights: {encoder_weights}")

    model = smp.create_model(
        cfg["architecture"],
        encoder_name=cfg["encoder"],
        encoder_weights=encoder_weights,
        in_channels=3,  # replicated single channel
        classes=cfg["classes"],
    ).to(device)

    # Load SSL-pretrained encoder weights if configured
    ssl_path = cfg.get("ssl_weights_path")
    if ssl_path and Path(ssl_path).exists():
        print(f"Loading SSL encoder weights from {ssl_path}")
        ssl_state = torch.load(ssl_path, map_location=device, weights_only=True)
        # Map timm keys → smp encoder keys.  smp wraps timm models and
        # stores them under model.encoder with a consistent prefix
        # structure.  We try a direct load first; if that fails we
        # do a filtered/partial load matching by suffix.
        enc_state = model.encoder.state_dict()
        # Build mapping: strip common timm prefixes and match to smp keys
        mapped, skipped = 0, 0
        new_state = {}
        for smp_key in enc_state:
            # smp keys look like "model.blocks.0.0.conv_pw.weight" or
            # "conv_stem.weight" depending on version.  timm keys are
            # the same or prefixed differently.  Try exact match first.
            if smp_key in ssl_state:
                new_state[smp_key] = ssl_state[smp_key]
                mapped += 1
            else:
                # Try matching by the tail of the key (after first dot)
                tail = smp_key.split(".", 1)[-1] if "." in smp_key else smp_key
                found = False
                for ssl_key, ssl_val in ssl_state.items():
                    ssl_tail = ssl_key.split(".", 1)[-1] if "." in ssl_key else ssl_key
                    if ssl_tail == tail and ssl_val.shape == enc_state[smp_key].shape:
                        new_state[smp_key] = ssl_val
                        mapped += 1
                        found = True
                        break
                if not found:
                    skipped += 1
        model.encoder.load_state_dict(new_state, strict=False)
        print(f"  SSL weights loaded: {mapped} mapped, {skipped} skipped "
              f"(of {len(enc_state)} encoder params)")

    if args.freeze_encoder:
        print("FREEZING encoder weights (linear probing mode)")
        for p in model.encoder.parameters():
            p.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Loss
    weights_path = data_dir / "processed" / "class_weights.json"
    class_weights = None
    if weights_path.exists():
        with open(weights_path) as f:
            class_weights = json.load(f).get("weights_list")

    criterion = get_loss(cfg["loss"], class_weights, device)

    # Optimizer (only trainable params)
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    if cfg.get("optimizer", "adamw") == "adamw":
        optimizer = torch.optim.AdamW(params_to_optimize, lr=cfg["lr"], weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(params_to_optimize, lr=cfg["lr"])

    # Scheduler
    if cfg.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    else:
        scheduler = None

    # TensorBoard
    writer = SummaryWriter(f"runs/{run_name}")

    # Training
    best_val_iou = 0
    start_epoch = 0
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    predictions_dir = Path("data/predictions")
    predictions_dir.mkdir(parents=True, exist_ok=True)

    epoch_log = []

    # Resume from checkpoint if available
    ckpt_path = models_dir / f"{run_name}_checkpoint.pth"
    if ckpt_path.exists():
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        best_val_iou = ckpt.get("best_val_iou", 0)
        epoch_log = ckpt.get("epoch_log", [])
        print(f"  Resumed at epoch {start_epoch}, best mIoU={best_val_iou:.4f}")

    for epoch in range(start_epoch, cfg["epochs"]):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou, val_cls_iou, _ = evaluate(model, val_loader, criterion, device, cfg["classes"])

        if scheduler:
            scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d}/{cfg['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val mIoU: {val_iou:.4f} | LR: {lr:.6f} | Time: {elapsed:.1f}s")

        epoch_log.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_miou": val_iou,
            "lr": lr,
        })

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("mIoU/val", val_iou, epoch)
        writer.add_scalar("LR", lr, epoch)
        for cls_name, cls_iou in val_cls_iou.items():
            writer.add_scalar(f"IoU_val/{cls_name}", cls_iou, epoch)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), models_dir / f"{run_name}_best.pth")
            print(f"  -> New best model saved (mIoU={val_iou:.4f})")

        # Checkpoint every 10 epochs (full state for resume)
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "best_val_iou": best_val_iou,
                "epoch_log": epoch_log,
            }, models_dir / f"{run_name}_checkpoint.pth")
            print(f"  -> Checkpoint saved (epoch {epoch+1})")

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    model.load_state_dict(torch.load(models_dir / f"{run_name}_best.pth", weights_only=True))
    test_loss, test_iou, test_cls_iou, test_preds = evaluate(
        model, test_loader, criterion, device, cfg["classes"]
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test mIoU: {test_iou:.4f}")
    print("Per-class IoU:")
    for cls_name, cls_iou in test_cls_iou.items():
        print(f"  {cls_name:>12s}: {cls_iou:.4f}")

    # Save metrics
    metrics = {
        "run_name": run_name,
        "config": cfg,
        "best_val_iou": float(best_val_iou),
        "test_iou": float(test_iou),
        "test_loss": float(test_loss),
        "test_per_class_iou": test_cls_iou,
        "epoch_log": epoch_log,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "freeze_encoder": args.freeze_encoder,
    }
    with open(models_dir / f"{run_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save test predictions
    np.save(predictions_dir / f"{run_name}_test_predictions.npy", test_preds.numpy())

    writer.close()
    print(f"\nDone. Results saved to models/{run_name}_*")


if __name__ == "__main__":
    main()
