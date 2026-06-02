"""FP16 and INT8 quantization helpers for PP-OCRv5 ONNX models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

from .convert import ConversionError, preprocess_detection_image, preprocess_recognition_image

TASK_CHOICES = ("det", "rec")
DEFAULT_OUTPUT_DIR = Path("models/ppocrv5_mobile")
DEFAULT_CALIBRATION_IMAGES = (Path("test_paper.PNG"),)
FP16_OP_BLOCK_LIST = ("ReduceMean", "Pow", "Sqrt", "Div")


class TensorCalibrationDataReader:
    """Small ONNX Runtime calibration reader backed by precomputed tensors."""

    def __init__(self, input_name: str, tensors: Iterable[object]):
        self.input_name = input_name
        self._items = [{input_name: tensor} for tensor in tensors]

    def get_next(self) -> dict[str, object] | None:
        """Return one calibration sample at a time for ONNX Runtime."""
        if not self._items:
            return None
        return self._items.pop(0)


def _input_name(onnx_model_path: Path) -> str:
    import onnx

    model = onnx.load(onnx_model_path)
    if not model.graph.input:
        raise ConversionError(f"ONNX model has no inputs: {onnx_model_path}")
    return model.graph.input[0].name


def calibration_tensors(task: str, image_paths: Sequence[Path]) -> list[object]:
    """Build PP-OCR-style calibration tensors for detection or recognition."""
    if not image_paths:
        raise ConversionError("At least one --calibration-image is required for INT8 quantization")
    tensors = []
    for image_path in image_paths:
        if task == "det":
            tensors.append(preprocess_detection_image(image_path))
        elif task == "rec":
            tensors.append(preprocess_recognition_image(image_path))
        else:
            raise ConversionError(f"Unsupported quantization task: {task}")
    return tensors


def quantize_fp16(source_path: Path, output_path: Path, keep_io_types: bool = True) -> Path:
    """Convert float initializers and internal tensors in *source_path* to FP16."""
    import onnx
    from onnxconverter_common.float16 import DEFAULT_OP_BLOCK_LIST, convert_float_to_float16

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(source_path)
    fp16_model = convert_float_to_float16(
        model,
        keep_io_types=keep_io_types,
        disable_shape_infer=True,
        max_finite_val=65504.0,
        min_positive_val=1e-7,
        op_block_list=list(dict.fromkeys([*DEFAULT_OP_BLOCK_LIST, *FP16_OP_BLOCK_LIST])),
    )
    onnx.save_model(fp16_model, output_path)
    onnx.checker.check_model(str(output_path))
    return output_path


def quantize_int8(
    source_path: Path,
    output_path: Path,
    task: str,
    calibration_images: Sequence[Path],
    per_channel: bool = False,
) -> Path:
    """Static QDQ INT8 quantization for PP-OCRv5 detection or recognition ONNX models."""
    import onnx
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reader = TensorCalibrationDataReader(_input_name(source_path), calibration_tensors(task, calibration_images))
    quantize_static(
        str(source_path),
        str(output_path),
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        per_channel=per_channel,
        op_types_to_quantize=["Conv", "MatMul"],
    )
    onnx.checker.check_model(str(output_path))
    return output_path


def quantized_output_path(output_dir: Path, source_path: Path, task: str, precision: str) -> Path:
    """Return the repository naming convention for generated PP-OCRv5 mobile ONNX files."""
    source_stem = source_path.stem.lower()
    task_name = task
    if "det" in source_stem:
        task_name = "det"
    elif "rec" in source_stem:
        task_name = "rec"
    return output_dir / f"ppocrv5_mobile_{task_name}_{precision}.onnx"


def _parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model entries must use TASK=PATH, for example det=ppocrv5_det.onnx")
    task, raw_path = value.split("=", 1)
    task = task.strip().lower()
    if task not in TASK_CHOICES:
        raise argparse.ArgumentTypeError(f"task must be one of: {', '.join(TASK_CHOICES)}")
    return task, Path(raw_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate FP16 and static INT8 PP-OCRv5 ONNX variants.")
    parser.add_argument(
        "--model",
        action="append",
        type=_parse_model,
        required=True,
        metavar="TASK=ONNX",
        help="Source ONNX model tagged with its PP-OCR task. Repeat for det and rec.",
    )
    parser.add_argument(
        "--precision",
        nargs="+",
        choices=("fp16", "int8"),
        default=["fp16", "int8"],
        help="Quantized precisions to emit. Default: fp16 int8.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated ONNX files.")
    parser.add_argument(
        "--calibration-image",
        type=Path,
        action="append",
        default=None,
        help="Image used for INT8 static calibration. Repeat for more samples. Default: test_paper.PNG",
    )
    parser.add_argument(
        "--int8-per-channel",
        action="store_true",
        help="Enable per-channel INT8 weights. Use only with source models whose opset supports the emitted QDQ attributes.",
    )
    parser.add_argument(
        "--allow-missing-calibration",
        action="store_true",
        help="Allow FP16-only runs when calibration images are absent and INT8 was not requested.",
    )
    return parser


def quantize_models(args: argparse.Namespace) -> list[dict[str, object]]:
    calibration_images = args.calibration_image or list(DEFAULT_CALIBRATION_IMAGES)
    if "int8" in args.precision:
        missing = [image_path for image_path in calibration_images if not image_path.exists()]
        if missing:
            raise ConversionError(f"Missing calibration image(s): {', '.join(map(str, missing))}")
    elif not args.allow_missing_calibration:
        calibration_images = [image_path for image_path in calibration_images if image_path.exists()]

    results: list[dict[str, object]] = []
    for task, source_path in args.model:
        if not source_path.exists():
            raise ConversionError(f"Missing source ONNX model: {source_path}")
        model_result: dict[str, object] = {"task": task, "source": str(source_path), "outputs": {}}
        outputs = model_result["outputs"]
        if "fp16" in args.precision:
            fp16_path = quantized_output_path(args.output_dir, source_path, task, "fp16")
            quantize_fp16(source_path, fp16_path)
            outputs["fp16"] = str(fp16_path)
        if "int8" in args.precision:
            int8_path = quantized_output_path(args.output_dir, source_path, task, "int8")
            quantize_int8(source_path, int8_path, task, calibration_images, per_channel=args.int8_per_channel)
            outputs["int8"] = str(int8_path)
        results.append(model_result)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        results = quantize_models(args)
    except ConversionError as exc:
        print(f"pp2onnx-quantize: error: {exc}")
        return 1
    print(json.dumps({"models": results}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
