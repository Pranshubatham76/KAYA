"""
SentinelSite — Acoustic Trainer
Fine-tunes YAMNet classification head on confirmed near-miss audio events.
Backbone FROZEN. Head only. PyTorch → ONNX → TFLite INT8.
Called by Celery training task.
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split

from app.config import settings
from app.core.storage import storage
from app.training.replay_buffer import ReplaySample

log = logging.getLogger(__name__)

# ── YAMNet AudioSet classes we care about ─────────────────────────────────────
# These are the 8 near-miss-relevant AudioSet class IDs.
# Workers confirmed events that triggered these → our training targets.

YAMNET_TARGET_CLASSES = {
    "impact_heavy": 375,
    "crash": 373,
    "bang": 374,
    "thud": 376,
    "shout": 44,
    "screaming": 45,
    "breaking": 378,
    "alarm_bell": 388,
}

# Map class name → local label index (0-based)
CLASS_TO_IDX: dict[str, int] = {cls: i for i, cls in enumerate(YAMNET_TARGET_CLASSES)}
IDX_TO_CLASS: dict[int, str] = {i: cls for cls, i in CLASS_TO_IDX.items()}
N_CLASSES = len(YAMNET_TARGET_CLASSES)

# Audio params (must match on-device YAMNet input)
SAMPLE_RATE = 16000
CLIP_DURATION_S = 0.96
N_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION_S)  # 15360 samples
N_MEL_BINS = 64
HOP_SIZE = 160  # 10ms hop


# ── Dataset ───────────────────────────────────────────────────────────────────

class AudioSampleDataset(Dataset):
    """
    Downloads audio clips from S3, extracts mel spectrograms.
    Augmentation: noise injection, time-shift, pitch-shift (lightweight).
    """

    def __init__(
        self,
        samples: list[ReplaySample],
        augment: bool = True,
    ) -> None:
        self.samples = [s for s in samples if s.audio_s3_key and s.label_acoustic]
        self.augment = augment
        self._audio_cache: dict[str, np.ndarray] = {}
        log.info(
            f"AudioSampleDataset: {len(self.samples)} samples "
            f"(from {len(samples)} total, {len(samples) - len(self.samples)} skipped — no audio/label)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        waveform = self._load_waveform(sample.audio_s3_key)
        mel = self._to_mel_spectrogram(waveform)
        if self.augment:
            mel = self._augment(mel)
        label_idx = CLASS_TO_IDX.get(sample.label_acoustic or "", 0)
        return torch.tensor(mel, dtype=torch.float32), label_idx

    def _load_waveform(self, s3_key: str) -> np.ndarray:
        if s3_key in self._audio_cache:
            return self._audio_cache[s3_key]
        audio_bytes = storage.download_audio(s3_key)
        waveform = self._decode_audio(audio_bytes)
        self._audio_cache[s3_key] = waveform
        return waveform

    def _decode_audio(self, audio_bytes: bytes) -> np.ndarray:
        """Decode WAV to mono float32 numpy array at 16kHz."""
        try:
            import soundfile as sf
            waveform, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception:
            import scipy.io.wavfile as wav
            sr, waveform = wav.read(io.BytesIO(audio_bytes))
            waveform = waveform.astype(np.float32) / 32768.0

        # Mono
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        # Resample if needed (shouldn't happen — all clips are 16kHz)
        if sr != SAMPLE_RATE:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)
        # Pad/trim to fixed length (0.96s = 15360 samples)
        if len(waveform) < N_SAMPLES:
            waveform = np.pad(waveform, (0, N_SAMPLES - len(waveform)))
        else:
            waveform = waveform[:N_SAMPLES]
        return waveform

    def _to_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """Convert waveform to log mel spectrogram (H×W = N_MEL_BINS × T)."""
        try:
            import librosa
            mel = librosa.feature.melspectrogram(
                y=waveform,
                sr=SAMPLE_RATE,
                n_mels=N_MEL_BINS,
                hop_length=HOP_SIZE,
                n_fft=512,
            )
            log_mel = librosa.power_to_db(mel, ref=np.max)
            # Normalize to [0, 1]
            log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
            return log_mel[np.newaxis, :, :]  # (1, N_MEL_BINS, T) — channel first
        except ImportError:
            # Minimal fallback without librosa
            log.warning("librosa not installed — using raw waveform reshape as spectrogram approximation")
            spec = waveform.reshape(N_MEL_BINS, -1)[:, :96]  # crude approximation
            return spec[np.newaxis, :, :]

    def _augment(self, mel: np.ndarray) -> np.ndarray:
        """Lightweight mel spectrogram augmentation."""
        # Gaussian noise
        if np.random.rand() < 0.5:
            mel = mel + np.random.normal(0, 0.01, mel.shape).astype(np.float32)
            mel = np.clip(mel, 0, 1)
        # Time masking (SpecAugment-lite)
        if np.random.rand() < 0.4:
            T = mel.shape[2]
            mask_w = np.random.randint(1, max(2, T // 8))
            mask_start = np.random.randint(0, T - mask_w)
            mel[:, :, mask_start : mask_start + mask_w] = 0
        # Frequency masking
        if np.random.rand() < 0.4:
            F = mel.shape[1]
            mask_h = np.random.randint(1, max(2, F // 8))
            mask_start = np.random.randint(0, F - mask_h)
            mel[:, mask_start : mask_start + mask_h, :] = 0
        return mel


# ── Model head ────────────────────────────────────────────────────────────────

class YAMNetHead(nn.Module):
    """
    Classification head on top of frozen YAMNet embeddings.
    YAMNet produces 1024-dim embeddings per 0.96s window.
    We attach this trainable head.
    """

    def __init__(self, n_classes: int, embedding_dim: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)


class YAMNetWithHead(nn.Module):
    """
    Full model: frozen YAMNet backbone + trainable head.
    Backbone is loaded from TFLite via tf.lite or as a PyTorch port.
    """

    def __init__(self, n_classes: int):
        super().__init__()
        # Try to load TorchYAMNet or use conv backbone approximation
        self.backbone = self._load_backbone()
        self.head = YAMNetHead(n_classes=n_classes)

        # FREEZE backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        log.info(f"YAMNet backbone frozen. Trainable params: {self._count_trainable()}")

    def _load_backbone(self) -> nn.Module:
        """Load YAMNet backbone. Falls back to Conv approximation."""
        try:
            import torch_yamnet
            backbone = torch_yamnet.YAMNet()
            backbone.eval()
            return backbone
        except ImportError:
            log.warning("torch_yamnet not installed — using Conv approximation backbone")
            return _ConvBackboneApprox()

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self.backbone(mel_spec)  # (B, 1024)
        return self.head(embeddings)

    def _count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _ConvBackboneApprox(nn.Module):
    """
    Lightweight Conv backbone approximating YAMNet's feature extraction.
    Used when torch_yamnet is unavailable (CI, testing).
    Output: (B, 1024) embeddings.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.proj = nn.Linear(256 * 4, 1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = f.view(f.size(0), -1)
        return self.proj(f)


# ── Training ──────────────────────────────────────────────────────────────────

class AcousticTrainer:

    def __init__(
        self,
        site_id: str,
        n_classes: int = N_CLASSES,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 16,
        val_split: float = 0.2,
        device: str | None = None,
    ):
        self.site_id = site_id
        self.n_classes = n_classes
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.val_split = val_split
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        log.info(f"AcousticTrainer on device: {self.device}")

    def train(
        self,
        samples: list[ReplaySample],
    ) -> dict[str, Any]:
        """
        Train YAMNet head on replay batch.
        Returns training metrics dict.
        """
        dataset = AudioSampleDataset(samples, augment=True)
        if len(dataset) < 4:
            raise ValueError(f"Too few valid audio samples: {len(dataset)}")

        # Train/val split
        n_val = max(1, int(len(dataset) * self.val_split))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])

        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2)
        val_dl = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=2)

        # Model
        model = YAMNetWithHead(n_classes=self.n_classes).to(self.device)

        # Only head params in optimizer
        optimizer = AdamW(model.head.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        history: list[dict[str, float]] = []
        best_val_acc = 0.0
        best_state = None

        for epoch in range(1, self.epochs + 1):
            # Train
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for mel, labels in train_dl:
                mel, labels = mel.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = model(mel)
                loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += len(labels)

            scheduler.step()

            # Validate
            val_acc, val_loss, per_class = self._validate(model, val_dl, criterion)

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_loss / max(train_total, 1),
                "train_acc": train_correct / max(train_total, 1),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": scheduler.get_last_lr()[0],
            }
            history.append(epoch_metrics)

            log.info(
                f"Epoch {epoch}/{self.epochs} — "
                f"train_acc={epoch_metrics['train_acc']:.3f}, "
                f"val_acc={val_acc:.3f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Restore best weights
        if best_state:
            model.load_state_dict(best_state)

        # Export
        tflite_bytes, pt_bytes = self._export(model)

        return {
            "val_accuracy": best_val_acc,
            "n_train": n_train,
            "n_val": n_val,
            "epochs": self.epochs,
            "history": history,
            "per_class_accuracy": per_class,
            "tflite_bytes": tflite_bytes,
            "pytorch_bytes": pt_bytes,
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
            for mel, labels in val_dl:
                mel, labels = mel.to(self.device), labels.to(self.device)
                logits = model(mel)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)
                for p, l in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                    per_class_correct[l] = per_class_correct.get(l, 0) + int(p == l)
                    per_class_total[l] = per_class_total.get(l, 0) + 1

        per_class_acc = {
            IDX_TO_CLASS.get(k, str(k)): per_class_correct.get(k, 0) / max(per_class_total.get(k, 1), 1)
            for k in per_class_total
        }
        return correct / max(total, 1), val_loss / max(total, 1), per_class_acc

    def _export(self, model: nn.Module) -> tuple[bytes, bytes]:
        """Export trained model to TFLite INT8 and PyTorch bytes."""
        model.eval().cpu()

        # PyTorch checkpoint
        pt_buf = io.BytesIO()
        torch.save({"model_state_dict": model.state_dict(), "n_classes": self.n_classes}, pt_buf)
        pt_bytes = pt_buf.getvalue()

        # TFLite (via ONNX → TFLite)
        try:
            tflite_bytes = self._to_tflite(model)
        except Exception as e:
            log.warning(f"TFLite export failed: {e} — returning placeholder")
            tflite_bytes = b"TFLITE_PLACEHOLDER"

        return tflite_bytes, pt_bytes

    def _to_tflite(self, model: nn.Module) -> bytes:
        """PyTorch → ONNX → TFLite INT8 quantized."""
        import onnx
        import subprocess

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = Path(tmp_dir) / "acoustic.onnx"
            tflite_path = Path(tmp_dir) / "acoustic.tflite"

            dummy = torch.zeros(1, 1, N_MEL_BINS, 96)  # (B, C, H, W)
            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                input_names=["mel_spec"],
                output_names=["logits"],
                dynamic_axes={"mel_spec": {0: "batch"}},
                opset_version=17,
            )

            # onnx2tf for TFLite conversion
            result = subprocess.run(
                [
                    "onnx2tf",
                    "-i", str(onnx_path),
                    "-o", str(tmp_dir),
                    "--quant_type", "per-tensor",
                    "--output_integer_quantized_tflite",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(f"onnx2tf failed: {result.stderr}")

            tflite_files = list(Path(tmp_dir).glob("*.tflite"))
            if not tflite_files:
                raise FileNotFoundError("TFLite export produced no .tflite file")

            return tflite_files[0].read_bytes()
