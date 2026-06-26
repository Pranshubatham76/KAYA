"""
SentinelSite — Acoustic Baseline Calibrator (Dev Workspace)
Algorithm: RMS Z-score -> sigmoid -> anomaly_score (0-1)
Calibration: 3σ above baseline RMS = θ₁ default
"""
import numpy as np

class AcousticBaselineCalibrator:
    """
    Calibrates the anomaly threshold for a construction site based on a 60-second
    baseline audio recording.
    """
    def __init__(self, sample_rate: int = 16000, window_size_s: float = 0.96):
        self.sample_rate = sample_rate
        self.window_size_s = window_size_s
        self.window_size_samples = int(self.sample_rate * self.window_size_s)

    def compute_baseline(self, waveform: np.ndarray) -> dict:
        """
        Input: 60s baseline recording (mono float32 at 16kHz)
        Computes RMS per 0.96s window, then calculates the distribution statistics.
        Returns baseline mean, std, and suggested threshold θ₁ (mean + 3*std).
        """
        if len(waveform) < self.window_size_samples:
            raise ValueError(f"Audio must be at least {self.window_size_s}s long.")
        
        num_windows = len(waveform) // self.window_size_samples
        rms_values = []
        for i in range(num_windows):
            chunk = waveform[i * self.window_size_samples : (i + 1) * self.window_size_samples]
            rms = np.sqrt(np.mean(chunk ** 2) + 1e-10) # avoid zero
            rms_values.append(rms)
            
        mean_rms = float(np.mean(rms_values))
        std_rms = float(np.std(rms_values))
        
        # θ₁ default formula: baseline_rms + 3σ
        theta_1 = mean_rms + 3 * std_rms
        
        return {
            "mean_rms": mean_rms,
            "std_rms": std_rms,
            "theta_1_default": theta_1
        }
    
    def score_anomaly(self, chunk_rms: float, baseline_mean: float, baseline_std: float) -> float:
        """
        Calculates the anomaly score for a given audio chunk's RMS.
        Uses RMS Z-score mapped through a sigmoid to yield an anomaly_score in (0, 1).
        
        Calibration shifts the sigmoid such that 3σ gives a score of exactly 0.5.
        """
        if baseline_std <= 0:
            return 0.0
            
        z_score = (chunk_rms - baseline_mean) / baseline_std
        
        # Sigmoid shifted so that z_score = 3 corresponds to anomaly_score = 0.5
        # Higher z_scores asymptotically reach 1.0.
        anomaly_score = 1.0 / (1.0 + np.exp(-(z_score - 3.0)))
        return float(anomaly_score)

if __name__ == "__main__":
    # Edge case testing
    calibrator = AcousticBaselineCalibrator()
    dummy_waveform = np.random.normal(0, 0.05, 16000 * 60) # 60 seconds of background noise
    
    stats = calibrator.compute_baseline(dummy_waveform)
    print("Calibration Stats:", stats)
    
    # Test a loud noise chunk
    loud_chunk = np.random.normal(0, 0.8, 16000)
    loud_rms = np.sqrt(np.mean(loud_chunk ** 2))
    score = calibrator.score_anomaly(loud_rms, stats["mean_rms"], stats["std_rms"])
    print(f"Anomaly Score for loud chunk: {score:.4f} (Expected > 0.9)")
    
    # Test background chunk
    bg_rms = stats["mean_rms"]
    score_bg = calibrator.score_anomaly(bg_rms, stats["mean_rms"], stats["std_rms"])
    print(f"Anomaly Score for bg chunk: {score_bg:.4f} (Expected near 0.047)")
