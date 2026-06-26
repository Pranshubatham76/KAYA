import numpy as np
import json

def generate_roc_curve():
    print("Sweeping thresholds for anomaly detection...")
    thresholds = np.linspace(0.1, 0.9, 9)
    results = {}
    
    for t in thresholds:
        # Simulate FP / TP rates
        tp = 1.0 - (t ** 2)
        fp = (1.0 - t) ** 3
        results[f"thresh_{t:.2f}"] = {"tp": round(tp, 3), "fp": round(fp, 3)}
        
    with open("threshold_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Threshold sweep complete.")

if __name__ == "__main__":
    generate_roc_curve()
