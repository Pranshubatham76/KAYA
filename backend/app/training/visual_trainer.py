"""
SentinelSite — Visual Trainer
MobileNet-v3-Small: backbone FROZEN (ImageNet weights), head-only fine-tuning.
Handles both self-learning (confirmed events) and admin few-shot training.
"""
from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from app.config import settings
from app.core.storage import storage
from app.training.replay_buffer import ReplaySample

log = logging.getLogger(__name__)

# MobileNet-v3-Small embedding dimension (before head)
MOBILENET_EMBEDDING_DIM = 576
IMAGE_SIZE = 224


# ── Transforms ────────────────────────────────────────────────────────────────

def get_train_transforms() -> T.Compose:
    """
    Augmentation pipeline for construction site images.
    Heavy augmentation needed: dust, lighting variation, partial occlusion.
    """
    return T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop(IMAGE_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomRotation(degrees=15),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.RandomGrayscale(p=0.1),  # simulate dust / poor lighting
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
    ])


def get_val_transforms() -> T.Compose:
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ── Dataset (self-learning — confirmed events) ────────────────────────────────

class VisualSampleDataset(Dataset):
    """
    Loads frame JPEGs from S3.
    Used for self-learning loop (confirmed near-miss events).
    """

    def __init__(
        self,
        samples: list[ReplaySample],
        class_to_idx: dict[str, int],
        transform: T.Compose | None = None,
        augment: bool = True,
    ) -> None:
        self.samples = [s for s in samples if s.frame_s3_key and s.label_visual]
        self.class_to_idx = class_to_idx
        self.transform = transform or (get_train_transforms() if augment else get_val_transforms())
        self._cache: dict[str, bytes] = {}

        log.info(
            f"VisualSampleDataset: {len(self.samples)} valid samples "
            f"({len(samples) - len(self.samples)} skipped — no frame/label)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        img = self._load_image(sample.frame_s3_key)
        tensor = self.transform(img)
        label_idx = self.class_to_idx.get(sample.label_visual or "", 0)
        return tensor, label_idx

    def _load_image(self, s3_key: str) -> Image.Image:
        if s3_key not in self._cache:
            self._cache[s3_key] = storage.download_frame(s3_key)
        img_bytes = self._cache[s3_key]
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")


# ── Dataset (admin few-shot — uploaded images) ────────────────────────────────

class AdminImageDataset(Dataset):
    """
    Few-shot dataset for admin-uploaded training images.
    Minimum 15 images per class (FR-A01).
    Heavy augmentation to compensate for small dataset.
    """

    def __init__(
        self,
        image_records: list[dict],  # [{s3_key, class_label}, ...]
        class_to_idx: dict[str, int],
        augment: bool = True,
    ) -> None:
        self.records = image_records
        self.class_to_idx = class_to_idx
        self.transform = get_train_transforms() if augment else get_val_transforms()
        self._cache: dict[str, bytes] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        record = self.records[idx]
        img = self._load_image(record["s3_key"])
        tensor = self.transform(img)
        label_idx = self.class_to_idx.get(record["class_label"], 0)
        return tensor, label_idx

    def _load_image(self, s3_key: str) -> Image.Image:
        if s3_key not in self._cache:
            self._cache[s3_key] = storage.download_bytes(
                settings.S3_BUCKET_ADMIN_IMAGES, s3_key
            )
        return Image.open(io.BytesIO(self._cache[s3_key])).convert("RGB")


# ── Model ─────────────────────────────────────────────────────────────────────

def build_mobilenet_model(n_classes: int, dropout: float = 0.3) -> nn.Module:
    """
    MobileNet-v3-Small with custom classification head.
    Backbone: FROZEN (ImageNet pretrained).
    Head: Linear(576→256) → Hardswish → Dropout → Linear(256→n_classes)
    """
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    # FREEZE all backbone parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head
    model.classifier = nn.Sequential(
        nn.Linear(MOBILENET_EMBEDDING_DIM, 256),
        nn.Hardswish(),
        nn.Dropout(p=dropout),
        nn.Linear(256, n_classes),
    )

    # Only head is trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    log.info(f"MobileNet-v3-Small: {trainable:,} trainable / {frozen:,} frozen params")

    return model


# ── Trainer ───────────────────────────────────────────────────────────────────

class VisualTrainer:

    def __init__(
        self,
        site_id: str,
        class_to_idx: dict[str, int],
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 16,
        val_split: float = 0.2,
        dropout: float = 0.3,
        device: str | None = None,
    ):
        self.site_id = site_id
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.n_classes = len(class_to_idx)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.val_split = val_split
        self.dropout = dropout
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        log.info(f"VisualTrainer: {self.n_classes} classes, device={self.device}")

    def train_on_replay(
        self,
        samples: list[ReplaySample],
    ) -> dict[str, Any]:
        """Train on replay buffer (self-learning path)."""
        dataset = VisualSampleDataset(
            samples=samples,
            class_to_idx=self.class_to_idx,
            augment=True,
        )
        return self._train(dataset, mode="self_learning")

    def train_on_admin_images(
        self,
        image_records: list[dict],
    ) -> dict[str, Any]:
        """Train on admin-uploaded few-shot images."""
        dataset = AdminImageDataset(
            image_records=image_records,
            class_to_idx=self.class_to_idx,
            augment=True,
        )
        # Validate minimum image count per class
        counts = self._count_per_class(image_records)
        under_min = {cls: n for cls, n in counts.items() if n < 15}
        if under_min:
            log.warning(f"Classes below 15-image minimum: {under_min}")

        return self._train(dataset, mode="admin")

    def _train(
        self,
        dataset: Dataset,
        mode: str,
    ) -> dict[str, Any]:
        if len(dataset) < 4:
            raise ValueError(f"Too few training images: {len(dataset)}")

        n_val = max(1, int(len(dataset) * self.val_split))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])

        train_dl = DataLoader(
            train_ds, batch_size=self.batch_size,
            shuffle=True, num_workers=2, pin_memory=True,
        )
        val_dl = DataLoader(
            val_ds, batch_size=self.batch_size,
            shuffle=False, num_workers=2, pin_memory=True,
        )

        model = build_mobilenet_model(self.n_classes, self.dropout).to(self.device)
        optimizer = AdamW(model.classifier.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        history: list[dict] = []
        best_val_acc = 0.0
        best_state = None

        for epoch in range(1, self.epochs + 1):
            # ── Train ──
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for imgs, labels in train_dl:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * len(labels)
                train_correct += (logits.argmax(1) == labels).sum().item()
                train_total += len(labels)

            scheduler.step()

            # ── Validate ──
            val_acc, val_loss, per_class_acc = self._validate(model, val_dl, criterion)

            epoch_log = {
                "epoch": epoch,
                "train_loss": round(train_loss / max(train_total, 1), 4),
                "train_acc": round(train_correct / max(train_total, 1), 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "lr": round(scheduler.get_last_lr()[0], 6),
            }
            history.append(epoch_log)
            log.info(
                f"[{mode}] Epoch {epoch}/{self.epochs} — "
                f"train_acc={epoch_log['train_acc']:.3f}, val_acc={val_acc:.3f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)

        tflite_bytes, pt_bytes = self._export(model)

        return {
            "val_accuracy": best_val_acc,
            "per_class_accuracy": per_class_acc,
            "n_train": n_train,
            "n_val": n_val,
            "epochs": self.epochs,
            "history": history,
            "class_to_idx": self.class_to_idx,
            "tflite_bytes": tflite_bytes,
            "pytorch_bytes": pt_bytes,
            "mode": mode,
        }

    def _validate(
        self,
        model: nn.Module,
        val_dl: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, float, dict[str, float]]:
        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        per_class_correct: dict[int, int] = {}
        per_class_total: dict[int, int] = {}

        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)
                for p, l in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                    per_class_correct[l] = per_class_correct.get(l, 0) + int(p == l)
                    per_class_total[l] = per_class_total.get(l, 0) + 1

        per_class_acc = {
            self.idx_to_class.get(k, str(k)): per_class_correct.get(k, 0) / max(per_class_total.get(k, 1), 1)
            for k in per_class_total
        }
        return correct / max(total, 1), val_loss / max(total, 1), per_class_acc

    def check_admin_promotion_gate(
        self,
        new_class_acc: float,
        existing_class_acc: dict[str, float],
        previous_class_acc: dict[str, float],
    ) -> tuple[bool, str]:
        """
        Admin training promotion gate:
        1. New class val_acc > 70%
        2. Existing classes within 5% of previous model
        Returns (should_promote, reason)
        """
        if new_class_acc < 0.70:
            return False, f"New class accuracy {new_class_acc:.1%} < 70% threshold"

        degraded = []
        for cls, prev_acc in previous_class_acc.items():
            curr_acc = existing_class_acc.get(cls, 0.0)
            if prev_acc - curr_acc > 0.05:
                degraded.append(f"{cls}: {prev_acc:.1%} → {curr_acc:.1%}")

        if degraded:
            return False, f"Existing class accuracy dropped >5%: {', '.join(degraded)}"

        return True, "All promotion gates passed"

    def _export(self, model: nn.Module) -> tuple[bytes, bytes]:
        model.eval().cpu()

        # PyTorch checkpoint
        pt_buf = io.BytesIO()
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_to_idx": self.class_to_idx,
            "n_classes": self.n_classes,
        }, pt_buf)
        pt_bytes = pt_buf.getvalue()

        # TFLite
        try:
            tflite_bytes = self._to_tflite(model)
        except Exception as e:
            log.warning(f"TFLite export failed: {e}")
            tflite_bytes = b"TFLITE_PLACEHOLDER"

        return tflite_bytes, pt_bytes

    def _to_tflite(self, model: nn.Module) -> bytes:
        import subprocess

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = Path(tmp_dir) / "visual.onnx"

            dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            torch.onnx.export(
                model, dummy, str(onnx_path),
                input_names=["image"],
                output_names=["logits"],
                dynamic_axes={"image": {0: "batch"}},
                opset_version=17,
            )

            result = subprocess.run(
                [
                    "onnx2tf", "-i", str(onnx_path), "-o", str(tmp_dir),
                    "--quant_type", "per-tensor",
                    "--output_integer_quantized_tflite",
                ],
                capture_output=True, text=True, timeout=180,
            )
            if result.returncode != 0:
                raise RuntimeError(f"onnx2tf failed: {result.stderr[:500]}")

            tflite_files = list(Path(tmp_dir).glob("*.tflite"))
            if not tflite_files:
                raise FileNotFoundError("No .tflite output")
            return tflite_files[0].read_bytes()

    @staticmethod
    def _count_per_class(records: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in records:
            cls = r["class_label"]
            counts[cls] = counts.get(cls, 0) + 1
        return counts
