# SentinelSite — ML Setup Guide
**Version:** 1.0 | Focus: Self-Learning Pipeline Optimization  
**Goal:** End-to-end guide to build, train, evaluate, and deploy every ML component

---

## Overview of ML Components

| Component | Type | Where It Runs | Primary Goal |
|---|---|---|---|
| YAMNet (acoustic) | Pretrained TFLite + fine-tuned head | On-device (phone) | Detect near-miss acoustic events |
| MobileNet-v3 (visual) | Pretrained TFLite + fine-tuned head | On-device (phone) | Classify visual context at event time |
| DistilBERT (intent router) | Fine-tuned classifier | On-device (phone) | Route voice query to correct document type |
| RAG Pipeline | Retrieval + LLM chain | Cloud (FastAPI) | Answer site-specific questions from documents |
| Self-Learning System | Continual learning with replay | Cloud (training job) | Improve acoustic/visual models from confirmed events |
| Admin Training Pipeline | Few-shot fine-tuning | Cloud (training job) | Adapt visual model to site-specific objects |

---

## Part 1 — Acoustic Model (YAMNet)

### 1.1 What YAMNet Is

YAMNet is a pretrained deep neural net (MobileNet-v1 backbone) that classifies audio into 521 classes from the AudioSet ontology. It is available as a TFLite model from TensorFlow Hub.

Key properties:
- Input: 0.96-second waveform segment at 16kHz mono
- Output: 521-class probability distribution
- Model size: ~3.7MB (quantized ~1MB)
- Inference: ~100ms on CPU, ~15ms with hardware delegation

### 1.2 Installation

```bash
# Download YAMNet TFLite from TF Hub
wget https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1 \
  -O android/app/src/main/assets/yamnet.tflite

# Also download the class map
wget https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv \
  -O android/app/src/main/assets/yamnet_class_map.csv
```

### 1.3 Construction-Relevant AudioSet Classes

Map these class indices for your anomaly detector. These are the ones that matter:

```python
# ml/acoustic/construction_sound_classes.json
CONSTRUCTION_NEAR_MISS_CLASSES = {
    # High priority — strong near-miss signal
    "Crash": 373,
    "Bang": 374,
    "Thud": 376,
    "Shout": 44,
    "Screaming": 45,
    "Breaking": 378,
    "Alarm": 388,
    "Impact sounds": 375,
    "Explosion": 414,

    # Medium priority — context dependent
    "Fell": 392,           # sound of falling
    "Clatter": 377,
    "Mechanical failure": 379,
    "Warning signal": 389,

    # Exclude these common site sounds from anomaly scoring
    # (they should be part of baseline, not anomalies)
    "Drill": 474,
    "Jackhammer": 476,
    "Sawing": 479,
    "Engine": 355,
    "Compressor": 472,
}
```

### 1.4 Baseline Calibration Algorithm

```python
# ml/acoustic/acoustic_baseline_calibrator.py
import numpy as np
import librosa

class AcousticBaselineCalibrator:
    """
    Records site ambient audio and computes a per-site noise baseline.
    This baseline defines the 'normal' for this site — anything significantly
    above it is an anomaly candidate.
    """

    def __init__(self, sample_rate=16000, duration_seconds=60):
        self.sample_rate = sample_rate
        self.duration = duration_seconds
        self.n_fft = 512
        self.hop_length = 256

    def compute_baseline(self, audio_array: np.ndarray) -> dict:
        """
        Input: 60s of ambient audio as float32 array
        Output: baseline parameters dict to store on device
        """
        # 1. Compute Short-Time Fourier Transform
        stft = librosa.stft(audio_array,
                           n_fft=self.n_fft,
                           hop_length=self.hop_length)
        magnitude = np.abs(stft)

        # 2. Per-frequency-band statistics
        mean_per_band = np.mean(magnitude, axis=1)   # shape: [n_fft//2 + 1]
        std_per_band = np.std(magnitude, axis=1)

        # 3. Overall RMS energy baseline
        rms_baseline = np.sqrt(np.mean(audio_array**2))
        rms_std = np.std([
            np.sqrt(np.mean(audio_array[i:i+self.sample_rate]**2))
            for i in range(0, len(audio_array), self.sample_rate)
        ])

        # 4. YAMNet class distribution baseline
        # Run YAMNet on 30 windows of baseline audio
        # Store the average class probability distribution
        # This becomes the "normal" distribution to compare against

        return {
            "mean_per_band": mean_per_band.tolist(),
            "std_per_band": std_per_band.tolist(),
            "rms_baseline": float(rms_baseline),
            "rms_std": float(rms_std),
            "threshold_theta1": float(rms_baseline + 3.0 * rms_std),
            # 3σ above baseline = anomaly candidate
            # This is tunable: lower N = more sensitive, higher N = fewer FP
        }

    def compute_anomaly_score(self,
                               window_audio: np.ndarray,
                               baseline: dict) -> float:
        """
        Returns 0.0–1.0 anomaly score for a 1-second audio window.
        Compares current RMS against baseline distribution.
        """
        current_rms = np.sqrt(np.mean(window_audio**2))
        baseline_rms = baseline["rms_baseline"]
        baseline_std = baseline["rms_std"]

        # Z-score of current window vs baseline
        z_score = (current_rms - baseline_rms) / (baseline_std + 1e-8)

        # Sigmoid to normalize to 0–1
        anomaly_score = 1.0 / (1.0 + np.exp(-z_score + 2.0))
        return float(anomaly_score)
```

### 1.5 Threshold Selection Strategy

```python
# ml/acoustic/threshold_sweep.py
"""
Run this BEFORE the hackathon demo on recorded construction audio.
Find optimal θ₁ that maximizes TPR while keeping FPR < 5%.

You need:
- ~10 minutes of recorded "normal" construction site audio
- ~20 synthetic near-miss events (drop objects, shout loudly near mic)
- Label them: 0 = normal, 1 = near-miss
"""

def sweep_thresholds(audio_segments, labels, theta_range):
    results = []
    for theta in theta_range:
        predictions = [1 if score > theta else 0
                      for score in compute_scores(audio_segments)]
        tp = sum(p==1 and l==1 for p,l in zip(predictions, labels))
        fp = sum(p==1 and l==0 for p,l in zip(predictions, labels))
        fn = sum(p==0 and l==1 for p,l in zip(predictions, labels))
        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (sum(l==0 for l in labels) + 1e-8)
        results.append({"theta": theta, "tpr": tpr, "fpr": fpr})

    # Find θ₁ where TPR > 0.80 and FPR < 0.05
    # That's your default threshold
    return results
```

---

## Part 2 — Visual Model (MobileNet-v3)

### 2.1 Base Model Setup

```python
# ml/visual/mobilenet_fine_tuning.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class SiteVisualClassifier(nn.Module):
    """
    MobileNet-v3-Small with frozen backbone.
    Only the classification head is trainable.
    This is the key design: backbone preserves ImageNet knowledge,
    head adapts to site-specific classes.
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # Load pretrained backbone
        base = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        # FREEZE backbone — critical for few-shot generalization
        for param in base.features.parameters():
            param.requires_grad = False

        # Extract backbone (everything except classifier)
        self.backbone = base.features
        self.avgpool = base.avgpool

        # Replace head with our custom head
        # MobileNet-v3-Small backbone output: 576 channels
        self.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def build_augmentation_pipeline(is_training: bool):
    """
    Aggressive augmentation for small datasets (15-30 images per class).
    Without this, you will overfit immediately.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3,    # ±30% brightness (site lighting varies)
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            # Simulate dust/blur common on construction sites
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],   # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def train_visual_head(
    model: SiteVisualClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    learning_rate: float = 1e-3
) -> dict:
    """
    Trains only the classification head.
    Backbone gradients are disabled.
    Returns validation accuracy per epoch.
    """
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # label_smoothing: prevents overconfidence on small datasets

    history = {"train_loss": [], "val_accuracy": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        scheduler.step()
        history["train_loss"].append(train_loss / len(train_loader))
        history["val_accuracy"].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Acc: {val_acc:.4f}")

    return history
```

---

## Part 3 — The Self-Learning Pipeline (Core Goal)

### 3.1 Philosophy and Design Constraints

The self-learning pipeline has one optimization goal:
> **Improve near-miss detection accuracy for each specific site over time, using only the supervisor's confirm/dismiss decisions as supervision signals, without degrading general detection capability, and without ever affecting real-time inference latency.**

Three hard constraints flow from this:
1. Training NEVER runs on-device
2. New data ALWAYS mixed with historical data (replay buffer)
3. New model ONLY deployed after validation accuracy ≥ previous model

### 3.2 Experience Replay Buffer — Optimized Implementation

This is the most important component to get right. Naive sequential fine-tuning causes catastrophic forgetting — accuracy on earlier learned patterns drops drastically. Experience replay substantially reduces catastrophic forgetting in continual learning by complementing new training data with samples representative of previous tasks.

The key question is **how to sample** from the replay buffer. Random sampling is the baseline. The optimized approach uses **class-balanced + uncertainty-weighted sampling**.

```python
# backend/app/training/replay_buffer.py
import numpy as np
from sqlalchemy.orm import Session
from collections import defaultdict
from app.db.models import TrainingSample

class ExperienceReplayBuffer:
    """
    Optimized replay buffer with class-balanced + uncertainty-weighted sampling.

    Why this matters:
    - Naive random sampling underrepresents rare event classes
      (e.g., "Electrocution near-miss" is rare but critical)
    - Class-balanced sampling ensures all learned classes are reviewed
    - Uncertainty weighting prioritizes samples the model is least confident on
      (these are the most informative for preventing forgetting)

    Research basis: Chaudhry et al. (2019) and Aljundi et al. (2019) showed
    gradient-aware and diversity-based buffer sampling outperforms random.
    We use a simplified version appropriate for our scale.
    """

    def __init__(self, db: Session):
        self.db = db

    def sample(
        self,
        n_new: int,
        ratio_historical: float = 0.70,
        strategy: str = "class_balanced"
    ) -> tuple[list, list]:
        """
        Returns (new_samples, historical_samples) for a training batch.

        Args:
            n_new: Number of new (unprocessed) samples to include
            ratio_historical: Fraction of batch that should be historical
                              Default: 70% historical, 30% new
                              Rationale: Prevents catastrophic forgetting while
                              still learning from new site patterns
            strategy: "random" | "class_balanced" | "uncertainty_weighted"
        """
        # Fetch new samples (not yet used in training)
        new_samples = (
            self.db.query(TrainingSample)
            .filter(TrainingSample.is_used_in_training == False)
            .order_by(TrainingSample.created_at.asc())
            .limit(n_new)
            .all()
        )

        # Compute how many historical samples to fetch
        n_historical = int(n_new * (ratio_historical / (1 - ratio_historical)))

        # Sample historical based on strategy
        if strategy == "class_balanced":
            historical = self._sample_class_balanced(n_historical)
        elif strategy == "uncertainty_weighted":
            historical = self._sample_uncertainty_weighted(n_historical)
        else:
            historical = self._sample_random(n_historical)

        return new_samples, historical

    def _sample_class_balanced(self, n: int) -> list:
        """
        Ensures each known class is equally represented in historical batch.
        Prevents model from forgetting rare event types.

        Example: if we have 300 "crash" events and 10 "electrical arc" events,
        random sampling would never show electrical arc. Class-balanced
        sampling guarantees electrical arc appears proportionally.
        """
        # Get all historical samples grouped by acoustic class
        all_historical = (
            self.db.query(TrainingSample)
            .filter(TrainingSample.is_used_in_training == True)
            .all()
        )

        # Group by label
        by_class = defaultdict(list)
        for sample in all_historical:
            by_class[sample.label_acoustic].append(sample)

        n_classes = len(by_class)
        if n_classes == 0:
            return []

        n_per_class = max(1, n // n_classes)
        selected = []

        for class_name, class_samples in by_class.items():
            # For each class, sample with replacement if needed
            sampled = np.random.choice(
                class_samples,
                size=min(n_per_class, len(class_samples)),
                replace=len(class_samples) < n_per_class
            ).tolist()
            selected.extend(sampled)

        # Trim or pad to exactly n
        np.random.shuffle(selected)
        return selected[:n]

    def _sample_uncertainty_weighted(self, n: int) -> list:
        """
        Prioritizes samples where the CURRENT MODEL was least confident.
        These samples are the most informative for preventing forgetting —
        they represent patterns the model is struggling with.

        Requires: model confidence scores stored per sample at last evaluation.
        This is stored in the training_samples.last_model_confidence column.
        """
        all_historical = (
            self.db.query(TrainingSample)
            .filter(TrainingSample.is_used_in_training == True)
            .filter(TrainingSample.last_model_confidence.isnot(None))
            .all()
        )

        if not all_historical:
            return self._sample_class_balanced(n)

        # Weight by inverse confidence (lower confidence = higher weight)
        confidences = np.array([s.last_model_confidence for s in all_historical])
        weights = 1.0 - confidences  # invert: uncertain samples get high weight
        weights = weights / weights.sum()  # normalize

        selected_indices = np.random.choice(
            len(all_historical),
            size=min(n, len(all_historical)),
            replace=False,
            p=weights
        )
        return [all_historical[i] for i in selected_indices]

    def _sample_random(self, n: int) -> list:
        """Baseline: pure random sampling from historical buffer."""
        all_historical = (
            self.db.query(TrainingSample)
            .filter(TrainingSample.is_used_in_training == True)
            .order_by(TrainingSample.created_at.desc())
            .all()
        )
        if len(all_historical) <= n:
            return all_historical
        indices = np.random.choice(len(all_historical), size=n, replace=False)
        return [all_historical[i] for i in indices]

    def get_buffer_statistics(self) -> dict:
        """
        Diagnostic: understand buffer composition before training.
        Always call this before a training job and log the output.
        """
        all_samples = (
            self.db.query(TrainingSample)
            .filter(TrainingSample.is_used_in_training == True)
            .all()
        )
        by_class = defaultdict(int)
        by_site = defaultdict(int)
        for s in all_samples:
            by_class[s.label_acoustic] += 1
            by_site[s.site_id] += 1

        return {
            "total_historical": len(all_samples),
            "class_distribution": dict(by_class),
            "site_distribution": dict(by_site),
            "most_underrepresented_class": min(by_class, key=by_class.get)
                                           if by_class else None,
        }
```

### 3.3 Training Job Scheduler — Trigger Conditions

```python
# backend/app/training/scheduler.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime, timedelta
from app.db.models import TrainingSample, ModelVersion
from app.training.acoustic_trainer import AcousticTrainer
import psutil, GPUtil

class TrainingScheduler:
    """
    Checks every 6 hours whether conditions are right for a training job.

    Design rationale for each condition:
    - n_new >= 20: Below 20 samples, statistical variance dominates.
      You'd be training on noise, not signal.
    - days_since >= 7: Prevents thrashing — training too frequently on
      small batches is worse than batching weekly.
    - gpu_util < 0.5: Don't compete with inference/API serving for GPU.
      Training is scheduled, not urgent.
    """

    def __init__(self, db_session_factory, trainer: AcousticTrainer):
        self.db_factory = db_session_factory
        self.trainer = trainer
        self.scheduler = AsyncIOScheduler()

    def start(self):
        self.scheduler.add_job(
            self.check_and_trigger,
            "interval",
            hours=6,
            id="training_check"
        )
        self.scheduler.start()

    async def check_and_trigger(self):
        db = self.db_factory()
        try:
            # Condition 1: Enough new samples?
            n_new = db.query(TrainingSample).filter(
                TrainingSample.is_used_in_training == False
            ).count()

            if n_new < 20:
                print(f"Training skip: only {n_new} new samples (need 20)")
                return

            # Condition 2: Enough time since last training?
            last_model = db.query(ModelVersion).filter(
                ModelVersion.status == "active",
                ModelVersion.model_type == "acoustic"
            ).order_by(ModelVersion.created_at.desc()).first()

            if last_model:
                days_since = (datetime.utcnow() - last_model.created_at).days
                if days_since < 7:
                    print(f"Training skip: only {days_since} days since last training")
                    return

            # Condition 3: GPU not overloaded?
            try:
                gpus = GPUtil.getGPUs()
                if gpus and gpus[0].load > 0.5:
                    print(f"Training skip: GPU at {gpus[0].load:.0%} utilization")
                    return
            except Exception:
                pass  # If no GPU info available, proceed

            # All conditions met — dispatch training job
            print(f"Training conditions met: {n_new} new samples. Dispatching job.")
            from app.training.acoustic_trainer import run_acoustic_training
            run_acoustic_training.delay()  # Celery async dispatch

        finally:
            db.close()
```

### 3.4 Acoustic Trainer — Fine-Tuning with Replay

```python
# backend/app/training/acoustic_trainer.py
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from celery import shared_task
from app.training.replay_buffer import ExperienceReplayBuffer
from app.training.quantizer import export_to_tflite
from app.training.model_pusher import push_model_to_devices

class YAMNetHeadTrainer:
    """
    Fine-tunes only the YAMNet classification head.

    BACKBONE IS ALWAYS FROZEN.
    This is non-negotiable. Fine-tuning the backbone with 20-100 samples
    would destroy the pretrained audio representations in 1-2 epochs.
    The head is a 2-layer MLP: 1024 → 512 → N_classes.
    """

    def __init__(self, n_classes: int, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_classes = n_classes

        # Load YAMNet backbone (frozen)
        # Using torchaudio.pipelines or custom loader
        self.backbone = self._load_yamnet_backbone()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Classification head (trainable)
        self.head = nn.Sequential(
            nn.Linear(1024, 512),    # YAMNet embedding is 1024-dim
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes)
        ).to(self.device)

    def _load_yamnet_backbone(self):
        """
        Load YAMNet backbone weights.
        Using the embeddings output (1024-dim), not the classification scores.
        """
        # Implementation depends on your YAMNet source
        # Options:
        # 1. TF Hub → export embeddings layer → ONNX → PyTorch
        # 2. torchaudio's VGGish (similar architecture, easier PyTorch integration)
        # 3. torch.hub with community YAMNet port
        # For hackathon: use torchaudio.models.get_pretrained_pipeline
        pass

    def preprocess_audio(self, audio_s3_url: str) -> torch.Tensor:
        """
        Download audio from S3 and convert to mel spectrogram for YAMNet.
        """
        # Download 30s clip
        # Segment into 0.96s windows
        # For training, use the 2s window around the trigger point
        # Returns: mel spectrogram tensor [batch, 1, 96, 64]
        pass

    def train_epoch(self, dataloader, optimizer, criterion):
        self.backbone.eval()  # backbone always in eval mode
        self.head.train()
        total_loss = 0
        for audio_tensors, labels in dataloader:
            audio_tensors = audio_tensors.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                embeddings = self.backbone(audio_tensors)  # frozen

            logits = self.head(embeddings)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, val_dataloader) -> float:
        self.backbone.eval()
        self.head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for audio_tensors, labels in val_dataloader:
                audio_tensors = audio_tensors.to(self.device)
                labels = labels.to(self.device)
                embeddings = self.backbone(audio_tensors)
                logits = self.head(embeddings)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total


@shared_task
def run_acoustic_training():
    """
    Celery task: runs async, does not block API.
    """
    from app.db.session import SessionLocal
    db = SessionLocal()

    try:
        # 1. Sample from replay buffer
        buffer = ExperienceReplayBuffer(db)
        buffer_stats = buffer.get_buffer_statistics()
        print(f"Buffer stats: {buffer_stats}")

        new_samples, historical_samples = buffer.sample(
            n_new=20,
            ratio_historical=0.70,
            strategy="class_balanced"  # or "uncertainty_weighted" for V2
        )
        all_samples = new_samples + historical_samples
        print(f"Training on {len(all_samples)} samples "
              f"({len(new_samples)} new, {len(historical_samples)} historical)")

        # 2. Build dataset
        # Download audio from S3, preprocess, create DataLoader
        train_set, val_set = build_acoustic_dataset(all_samples, split=0.8)
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32)

        # 3. Get current model for comparison
        current_model = get_current_model_version(db, "acoustic")
        current_accuracy = current_model.val_accuracy if current_model else 0.0

        # 4. Train
        n_classes = len(set(s.label_acoustic for s in all_samples))
        trainer = YAMNetHeadTrainer(n_classes=n_classes)

        optimizer = torch.optim.AdamW(
            trainer.head.parameters(),
            lr=1e-3,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=30
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_val_acc = 0.0
        for epoch in range(30):
            train_loss = trainer.train_epoch(train_loader, optimizer, criterion)
            val_acc = trainer.evaluate(val_loader)
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save best checkpoint
                torch.save(trainer.head.state_dict(), "/tmp/best_head.pt")

            print(f"Epoch {epoch+1}/30 | Loss: {train_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

        # 5. Promotion gate
        if best_val_acc >= current_accuracy:
            print(f"New model ({best_val_acc:.4f}) >= current ({current_accuracy:.4f}). Promoting.")

            # Load best checkpoint
            trainer.head.load_state_dict(torch.load("/tmp/best_head.pt"))

            # Export to TFLite INT8
            tflite_path = export_to_tflite(
                trainer.backbone,
                trainer.head,
                model_type="acoustic",
                quantize=True
            )

            # Register new version
            new_version = register_model_version(
                db,
                model_type="acoustic",
                val_accuracy=best_val_acc,
                training_sample_count=len(all_samples),
                tflite_path=tflite_path
            )

            # Mark samples as used
            for sample in new_samples:
                sample.is_used_in_training = True
            db.commit()

            # Push to device fleet
            push_model_to_devices(new_version.id, model_type="acoustic")
            print(f"Model v{new_version.version_tag} pushed to devices.")

        else:
            print(f"New model ({best_val_acc:.4f}) < current ({current_accuracy:.4f}). Rejected.")
            # Notify admin dashboard via WebSocket

    finally:
        db.close()
```

### 3.5 TFLite Export Pipeline

```python
# backend/app/training/quantizer.py
"""
PyTorch model → ONNX → TFLite INT8

This is the most finicky part of the pipeline.
Test this FIRST before the hackathon — it has dependency hell.
"""

import torch
import onnx
import subprocess
import numpy as np

def export_to_tflite(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    model_type: str,
    quantize: bool = True
) -> str:
    """
    Exports combined backbone+head to TFLite INT8.

    Steps:
    1. Combine backbone + head into single nn.Sequential
    2. Export to ONNX
    3. Convert ONNX → TFLite via onnx2tf
    4. Apply INT8 quantization via TFLite converter
    5. Save to /tmp, upload to S3, return S3 URL
    """

    # Combine models
    class CombinedModel(torch.nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
        def forward(self, x):
            return self.head(self.backbone(x))

    combined = CombinedModel(backbone, head)
    combined.eval()

    # Step 1: Export to ONNX
    dummy_input = torch.randn(1, 1, 96, 64)   # YAMNet input shape
    onnx_path = f"/tmp/sentinel_{model_type}.onnx"

    torch.onnx.export(
        combined,
        dummy_input,
        onnx_path,
        opset_version=17,
        input_names=["audio_input"],
        output_names=["class_scores"],
        dynamic_axes={"audio_input": {0: "batch"}}
    )

    # Step 2: ONNX → TFLite via onnx2tf
    tflite_dir = f"/tmp/sentinel_{model_type}_tflite"
    subprocess.run([
        "onnx2tf",
        "-i", onnx_path,
        "-o", tflite_dir,
        "--non_verbose"
    ], check=True)

    tflite_path_fp32 = f"{tflite_dir}/sentinel_{model_type}_float32.tflite"

    if not quantize:
        return tflite_path_fp32

    # Step 3: INT8 quantization
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(tflite_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    # Representative dataset for calibration
    # Use 100 samples from validation set
    def representative_dataset():
        for _ in range(100):
            yield [np.random.randn(1, 1, 96, 64).astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant = converter.convert()
    tflite_int8_path = f"/tmp/sentinel_{model_type}_int8.tflite"
    with open(tflite_int8_path, "wb") as f:
        f.write(tflite_quant)

    print(f"Exported INT8 TFLite: {tflite_int8_path}")
    print(f"Size: {len(tflite_quant) / 1024:.1f} KB")

    return tflite_int8_path
```

---

## Part 4 — RAG Pipeline

### 4.1 Document Ingestion

```python
# backend/app/rag/ingestion.py
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

COLLECTION_TEMPLATE = "site_{site_id}_{doc_type}"

# Chunk sizes by document type
# Structural drawings: smaller chunks (numbers/specs are dense)
# SOPs: larger chunks (procedural steps need context)
CHUNK_CONFIG = {
    "STRUCTURAL": {"chunk_size": 600, "overlap": 100},
    "SAFETY":     {"chunk_size": 800, "overlap": 150},
    "SCHEDULE":   {"chunk_size": 500, "overlap": 80},
    "MATERIAL":   {"chunk_size": 700, "overlap": 100},
    "ELECTRICAL": {"chunk_size": 600, "overlap": 100},
    "DEFAULT":    {"chunk_size": 800, "overlap": 150},
}

def ingest_document(
    pdf_path: str,
    site_id: str,
    doc_type: str,
    filename: str,
    qdrant_client: QdrantClient
):
    """
    Ingest a PDF into the vector store for a specific site.

    Uses Unstructured.io for layout-aware parsing — handles:
    - Tables (converted to structured text)
    - Multi-column layouts (construction spec sheets)
    - Scanned PDFs (OCR via Tesseract)
    - Annotation layers in technical drawings
    """

    # 1. Parse with Unstructured (layout-aware)
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",          # Use hi_res for technical drawings
        infer_table_structure=True,  # Tables as structured text
        languages=["eng"],
        extract_images_in_pdf=False  # Skip embedded images for now
    )

    # 2. Convert elements to text with metadata
    chunks_with_meta = []
    for element in elements:
        text = str(element)
        if len(text.strip()) < 20:  # Skip very short fragments
            continue

        meta = {
            "site_id": site_id,
            "doc_type": doc_type,
            "filename": filename,
            "page_number": element.metadata.page_number,
            "element_type": type(element).__name__,  # Title, NarrativeText, Table, etc.
        }
        chunks_with_meta.append({"text": text, "metadata": meta})

    # 3. Apply chunking per doc type
    config = CHUNK_CONFIG.get(doc_type, CHUNK_CONFIG["DEFAULT"])
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["overlap"],
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    final_chunks = []
    for item in chunks_with_meta:
        sub_chunks = splitter.split_text(item["text"])
        for chunk in sub_chunks:
            final_chunks.append({"text": chunk, "metadata": item["metadata"]})

    # 4. Create/get Qdrant collection for this site + doc type
    collection_name = COLLECTION_TEMPLATE.format(
        site_id=site_id, doc_type=doc_type.lower()
    )

    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            # 1536 = text-embedding-3-small dimension
        )

    # 5. Embed and upsert
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    texts = [c["text"] for c in final_chunks]
    metas = [c["metadata"] for c in final_chunks]

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings
    )
    vector_store.add_texts(texts=texts, metadatas=metas)

    return {"chunks_ingested": len(final_chunks), "collection": collection_name}
```

### 4.2 Intent Router

```python
# backend/app/rag/intent_classifier.py
"""
Routes voice queries to the correct document collection.
Uses a fine-tuned DistilBERT classifier.

Training data: 300 synthetic Q&A pairs, labeled by doc type.
Generated using GPT-4o from sample construction document text.

Classes:
0 = STRUCTURAL    (rebar, beams, columns, dimensions, loads)
1 = SAFETY        (PPE, OSHA, emergency, hazard, restricted)
2 = SCHEDULE      (trades, timeline, zone assignment, today's work)
3 = MATERIAL      (specs, grades, suppliers, MDS, quantities)
4 = ELECTRICAL    (conduit, panel, voltage, grounding)
5 = PLUMBING      (pipes, fixtures, drainage, water supply)
6 = INSPECTION    (QC, tolerances, sign-off, punch list)
7 = GENERAL       (project info, contacts, logistics)
"""

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

DOC_TYPES = [
    "STRUCTURAL", "SAFETY", "SCHEDULE", "MATERIAL",
    "ELECTRICAL", "PLUMBING", "INSPECTION", "GENERAL"
]

class IntentRouter:

    def __init__(self, model_path: str = "models/intent_router"):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def classify(self, query: str) -> str:
        """Returns document type string."""
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=True
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        class_idx = logits.argmax(dim=1).item()
        return DOC_TYPES[class_idx]


def generate_training_data(site_document_texts: list[str]) -> list[dict]:
    """
    Use GPT-4o to generate synthetic Q&A pairs from real site documents.
    Run this once per document set to create intent classifier training data.

    Generates 300 pairs: 300/8 ≈ 37-38 per class.
    """
    from openai import OpenAI
    client = OpenAI()

    prompt_template = """
    Given this construction document text:
    {text}

    Generate 5 natural questions a construction worker might ask verbally,
    along with the document type each question belongs to.

    Document types: STRUCTURAL, SAFETY, SCHEDULE, MATERIAL, ELECTRICAL,
                   PLUMBING, INSPECTION, GENERAL

    Format as JSON array:
    [{"question": "...", "doc_type": "..."}]

    Make questions sound natural and spoken, not formal.
    Include variations in phrasing.
    """
    # ... implementation
```

### 4.3 LLM Chain

```python
# backend/app/rag/llm_chain.py
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate

CONSTRUCTION_SYSTEM_PROMPT = """You are SentinelSite Voice, an AI assistant for construction workers.
You answer questions about this specific construction project using only the provided document context.

Rules (non-negotiable):
1. Answer ONLY from the provided context. Never use general knowledge.
2. Always cite your source: "Per [Document Name], Section [X]..."
3. If the context doesn't contain the answer, say exactly:
   "I couldn't find that in the site documents. Please check with your supervisor."
4. Be concise — your answer will be spoken aloud. Maximum 3 sentences.
5. Use simple, clear language. Avoid technical jargon unless it's in the spec.
6. For safety-critical answers (load limits, electrical, PPE), always add:
   "Verify this with your supervisor before proceeding."

Context from site documents:
{context}

Worker's question: {question}

Answer (spoken format, cite source):"""

def build_rag_chain(llm_provider: str = "anthropic"):
    if llm_provider == "anthropic":
        llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=256)
    else:
        llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=256)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=CONSTRUCTION_SYSTEM_PROMPT
    )
    return prompt | llm
```

---

## Part 5 — Evaluation Framework

### 5.1 Acoustic Model Evaluation

```python
# ml/evaluation/end_to_end_latency_test.py
"""
Run this on an actual Android device (adb shell) before the demo.
Measures real latency including BT overhead.
"""

LATENCY_TARGETS = {
    "yamnet_inference_1s_window": 100,     # ms, on CPU
    "imu_jerk_computation": 1,             # ms
    "fusion_gate_check": 1,                # ms
    "frame_capture_bt": 500,               # ms
    "payload_build": 50,                   # ms
    "total_event_to_queue": 600,           # ms
    "server_to_dashboard_alert": 5000,     # ms (includes upload + analysis)
}

# Run 20 controlled events, measure each stage
# Document results. If any stage exceeds target, optimize before demo.
```

### 5.2 RAG Evaluation

```python
# ml/rag/retrieval_eval.py
"""
Evaluate RAG accuracy on 50 ground-truth Q&A pairs.
Generate these pairs from your ingested site documents using GPT-4o.

Metrics:
- Hit Rate @ 3: Is the correct chunk in top 3 results? Target: > 85%
- MRR (Mean Reciprocal Rank): How highly ranked is the correct chunk?
- Answer Faithfulness: Is the LLM answer grounded in retrieved context?
  (Use Ragas for automated faithfulness scoring)
"""

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

# ragas gives you automated evaluation of:
# 1. Faithfulness: answer is grounded in context (no hallucination)
# 2. Answer Relevancy: answer addresses the question
# 3. Context Recall: retrieved context contains the answer
```

### 5.3 Self-Learning Benchmark

```python
# ml/training/forgetting_benchmark.py
"""
CRITICAL EXPERIMENT — run this before the hackathon.
Proves that your replay buffer actually prevents catastrophic forgetting.

Procedure:
1. Train head on Classes A, B, C (50 samples each) → record accuracy
2. Train on Class D only (no replay) → measure accuracy on A, B, C
3. Train on Class D with replay buffer → measure accuracy on A, B, C
4. Show the difference. This is your proof that replay works.

Expected result without replay: A, B, C accuracy drops 20-40%
Expected result with replay: A, B, C accuracy drops < 5%
"""

def run_forgetting_experiment():
    # Phase 1: Train on initial classes
    # Phase 2a: Fine-tune without replay → measure forgetting
    # Phase 2b: Fine-tune with replay → measure forgetting
    # Plot: accuracy per class, before and after, with vs without replay
    # Save plot to docs/ — use in demo to prove your approach works
    pass
```

---

## Part 6 — Setup Sequence

### Before Writing Any App Code

```bash
# Step 1: Verify YAMNet works
cd ml/acoustic
python yamnet_baseline_eval.py
# → Should classify: crash, shout, impact correctly on test audio

# Step 2: Run forgetting benchmark
python ../training/forgetting_benchmark.py
# → Save the plot. This is your ML proof slide.

# Step 3: Test TFLite export pipeline
python ../training/export_to_tflite.py --model-type acoustic
# → Verify .tflite file generated, correct size

# Step 4: Verify MobileNet fine-tuning on 15 images
python ../visual/few_shot_eval.py --n-images 15
# → Expect > 70% val accuracy with augmentation

# Step 5: Test RAG pipeline on sample construction PDF
cd ../../backend
python -c "from app.rag.ingestion import ingest_document; ..."
# → Verify chunks ingested, retrieval returns relevant chunks

# Only after all 5 steps pass → start building the Android app
```

### Environment Setup

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Environment variables (copy .env.example)
cp .env.example .env
# Fill in: OPENAI_API_KEY, ANTHROPIC_API_KEY, AWS_* keys, DB_URL

# Start infrastructure
docker-compose up -d postgres qdrant redis

# Run migrations
alembic upgrade head

# Start API
uvicorn app.main:app --reload --port 8000

# Start Celery worker (needs GPU for training jobs)
celery -A app.celery_app worker --loglevel=info --concurrency=2

# Start Celery beat (scheduler)
celery -A app.celery_app beat --loglevel=info
```

---

## Critical Warnings

**1. Test YAMNet at your demo noise level FIRST.**  
Record 5 minutes of audio at the same ambient noise level as your demo location. Run the threshold sweep. If false positive rate > 10%, your demo will embarrass you.

**2. The TFLite export is the most failure-prone step.**  
PyTorch → ONNX → TFLite has known compatibility issues with certain ops. Test this pipeline on Day 0 of the hackathon, not Day 2.

**3. The replay buffer is only as good as your supervisor labels.**  
If supervisors don't click Confirm/Dismiss, self-training cannot happen. Design the dashboard to make review so fast (< 5 seconds) that it happens reflexively.

**4. Don't claim sub-500ms voice response without measuring it.**  
STT + network + RAG + LLM + TTS adds up. Measure end-to-end latency on real LTE (not your office WiFi) before stating numbers to judges.

**5. The forgetting benchmark is not optional.**  
Run it. Get the plot. Show it to judges. It's the one piece of evidence that proves your self-learning approach is architecturally sound, not just claimed.