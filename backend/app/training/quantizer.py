"""
SentinelSite — Quantizer
Utility for exporting PyTorch models to TFLite INT8 format via ONNX.
Centralizes the quantization logic used by AcousticTrainer and VisualTrainer.
"""
import io
import logging
import tempfile
import subprocess
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

class ModelQuantizer:
    """
    Handles the pipeline: PyTorch Model -> ONNX -> TFLite INT8 Quantized.
    """
    
    @staticmethod
    def export_tflite_int8(
        model: nn.Module,
        dummy_input: torch.Tensor,
        input_names: list[str],
        output_names: list[str],
        dynamic_axes: dict,
        model_name: str = "model"
    ) -> bytes:
        """
        Exports a PyTorch model to TFLite using integer quantization.
        Requires `onnx2tf` to be available in the system PATH.
        """
        model.eval().cpu()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = Path(tmp_dir) / f"{model_name}.onnx"
            
            # 1. Export to ONNX
            try:
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=17,
                )
                log.info(f"ONNX export successful: {onnx_path}")
            except Exception as e:
                log.error(f"PyTorch to ONNX export failed: {e}")
                raise RuntimeError(f"ONNX export failed: {e}")

            # 2. Convert ONNX to TFLite INT8
            try:
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
                    timeout=180,
                )
                
                if result.returncode != 0:
                    log.error(f"onnx2tf stderr: {result.stderr}")
                    raise RuntimeError(f"onnx2tf failed with exit code {result.returncode}")
                
            except subprocess.TimeoutExpired:
                log.error("onnx2tf conversion timed out after 180 seconds.")
                raise RuntimeError("TFLite conversion timed out")
            except FileNotFoundError:
                log.error("onnx2tf command not found. Ensure it is installed via pip.")
                raise RuntimeError("onnx2tf not found")

            # 3. Read the output TFLite file
            tflite_files = list(Path(tmp_dir).glob("*.tflite"))
            if not tflite_files:
                raise FileNotFoundError("TFLite export produced no .tflite file")

            tflite_bytes = tflite_files[0].read_bytes()
            log.info(f"TFLite INT8 export successful. Size: {len(tflite_bytes)} bytes.")
            return tflite_bytes

    @staticmethod
    def get_pytorch_bytes(model: nn.Module, metadata: dict = None) -> bytes:
        """
        Serializes the PyTorch model state_dict alongside optional metadata.
        """
        pt_buf = io.BytesIO()
        save_dict = {"model_state_dict": model.state_dict()}
        if metadata:
            save_dict.update(metadata)
            
        torch.save(save_dict, pt_buf)
        return pt_buf.getvalue()
