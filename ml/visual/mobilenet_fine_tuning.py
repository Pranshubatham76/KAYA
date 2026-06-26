"""
SentinelSite — MobileNet Fine Tuning (Dev Workspace)
Test harness for running visual fine-tuning locally with sample data.
Validates the MobileNet-v3-Small training pipeline and augmentations.
"""
import logging
import sys
from pathlib import Path
import torch
from PIL import Image
import numpy as np

# Add backend to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "backend"))

from app.training.visual_trainer import VisualTrainer
from app.training.replay_buffer import ReplaySample

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CLASS_TO_IDX = {
    "no_hard_hat": 0,
    "unsafe_posture": 1,
    "spill": 2,
    "clear": 3
}

def generate_dummy_samples(n_samples=30):
    """Generate dummy ReplaySamples for testing the visual training loop."""
    samples = []
    labels = list(CLASS_TO_IDX.keys())
    for i in range(n_samples):
        label = labels[i % len(labels)]
        sample = ReplaySample(
            id=f"mock_{i}",
            site_id="dev_site",
            audio_s3_key=None,
            frame_s3_key=f"mock/frame_{i}.jpg",
            label_acoustic=None,
            label_visual=label,
            confidence_score=0.9
        )
        samples.append(sample)
    return samples

def run_local_fine_tuning():
    log.info("Starting local MobileNet Fine-Tuning test...")
    trainer = VisualTrainer(site_id="dev_site", class_to_idx=CLASS_TO_IDX, epochs=2, batch_size=4)
    
    samples = generate_dummy_samples(30)
    
    # Mock the dataset's image loading to avoid S3 calls
    from app.training.visual_trainer import VisualSampleDataset
    original_load = VisualSampleDataset._load_image
    
    def mock_load_image(self, s3_key: str) -> Image.Image:
        # Return dummy 224x224 RGB image
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        return Image.fromarray(arr)
        
    VisualSampleDataset._load_image = mock_load_image
    
    try:
        # Run training
        results = trainer.train_on_replay(samples)
        
        log.info("Training completed successfully.")
        log.info(f"Validation Accuracy: {results.get('val_accuracy'):.2f}")
        log.info(f"TFLite model size: {len(results.get('tflite_bytes', b''))} bytes")
        log.info(f"PyTorch model size: {len(results.get('pytorch_bytes', b''))} bytes")
    except Exception as e:
        log.error(f"Training failed: {e}")
        raise
    finally:
        # Restore original method
        VisualSampleDataset._load_image = original_load

if __name__ == "__main__":
    run_local_fine_tuning()
