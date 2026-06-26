"""
SentinelSite — Acoustic Fine Tuning (Dev Workspace)
Test harness for running acoustic fine-tuning locally with sample data.
Validates the YAMNet training pipeline without needing full cloud infrastructure.
"""
import logging
import sys
from pathlib import Path
import numpy as np
import torch

# Add backend to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "backend"))

from app.training.acoustic_trainer import AcousticTrainer, N_CLASSES, CLASS_TO_IDX
from app.training.replay_buffer import ReplaySample

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def generate_dummy_samples(n_samples=30):
    """Generate dummy ReplaySamples for testing the training loop."""
    samples = []
    labels = list(CLASS_TO_IDX.keys())
    for i in range(n_samples):
        # We assign realistic acoustic labels
        label = labels[i % len(labels)]
        sample = ReplaySample(
            id=f"mock_{i}",
            site_id="dev_site",
            audio_s3_key=f"mock/audio_{i}.wav",
            frame_s3_key=None,
            label_acoustic=label,
            label_visual=None,
            confidence_score=0.9
        )
        samples.append(sample)
    return samples

def run_local_fine_tuning():
    log.info("Starting local Acoustic Fine-Tuning test...")
    trainer = AcousticTrainer(site_id="dev_site", epochs=2, batch_size=4)
    
    samples = generate_dummy_samples(30)
    
    # Mock the dataset's audio loading to avoid S3 calls
    from app.training.acoustic_trainer import AudioSampleDataset
    original_load = AudioSampleDataset._load_waveform
    
    def mock_load_waveform(self, s3_key: str) -> np.ndarray:
        # Return dummy 16kHz audio (0.96s)
        return np.random.randn(int(16000 * 0.96)).astype(np.float32) * 0.01
        
    AudioSampleDataset._load_waveform = mock_load_waveform
    
    try:
        # Run training
        results = trainer.train(samples)
        
        log.info("Training completed successfully.")
        log.info(f"Validation Accuracy: {results.get('val_accuracy'):.2f}")
        log.info(f"TFLite model size: {len(results.get('tflite_bytes', b''))} bytes")
        log.info(f"PyTorch model size: {len(results.get('pytorch_bytes', b''))} bytes")
    except Exception as e:
        log.error(f"Training failed: {e}")
        raise
    finally:
        # Restore original method
        AudioSampleDataset._load_waveform = original_load

if __name__ == "__main__":
    run_local_fine_tuning()
