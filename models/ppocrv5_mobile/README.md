# PP-OCRv5 Mobile ONNX quantized models

This directory contains PP-OCRv5 mobile detection and recognition ONNX variants prepared for RapidOCR / OnnxOCR-style pipelines. The input and output tensors remain `float32`; FP16 changes internal floating-point weights/tensors, and INT8 uses ONNX Runtime static QDQ quantization for `Conv` and `MatMul` nodes.

| File | Task | Precision | Size | Notes |
| --- | --- | --- | ---: | --- |
| `ppocrv5_mobile_det_fp16.onnx` | detection | FP16 | 2.4 MiB | Internal FP16 conversion with Resize and normalization-style ops kept in FP32 for ONNX Runtime compatibility. |
| `ppocrv5_mobile_det_int8.onnx` | detection | 8-bit QDQ | 1.4 MiB | Static QDQ calibration with PP-OCR detection preprocessing on `test_paper.PNG`; uses U8/U8 QDQ for stable CPU execution. |
| `ppocrv5_mobile_rec_fp16.onnx` | recognition | FP16 | 8.0 MiB | Internal FP16 conversion with ReduceMean/Pow/Sqrt/Div kept in FP32 to avoid LayerNorm fusion load issues. |
| `ppocrv5_mobile_rec_int8.onnx` | recognition | 8-bit QDQ | 4.2 MiB | Static QDQ calibration with PP-OCR recognition preprocessing on `test_paper.PNG`; uses U8/U8 QDQ for stable CPU execution. |

The files were generated with:

```bash
uv run --extra quantize python -m pp2onnx.quantize \
  --model det=source_onnx/ppocrv5_det.onnx \
  --model rec=source_onnx/ppocrv5_rec.onnx \
  --output-dir models/ppocrv5_mobile \
  --calibration-image test_paper.PNG
```

ONNX Runtime CPU load and single-image parity against the source FP32 ONNX were checked on `test_paper.PNG`:

| File | Output shape | Max abs diff | Mean abs diff | Cosine |
| --- | --- | ---: | ---: | ---: |
| `ppocrv5_mobile_det_fp16.onnx` | `1x1x896x672` | 0.053399 | 0.000123 | 0.999994 |
| `ppocrv5_mobile_det_int8.onnx` | `1x1x896x672` | 1.000000 | 0.027009 | 0.911641 |
| `ppocrv5_mobile_rec_fp16.onnx` | `1x40x18385` | 0.083090 | 0.00000151 | 0.999735 |
| `ppocrv5_mobile_rec_int8.onnx` | `1x40x18385` | 0.407478 | 0.0000166 | 0.992306 |

INT8 output drift is expected to be larger than FP16, especially for DB detector probability maps. Recalibrate with more representative document images before using the INT8 models as production drop-in replacements.
