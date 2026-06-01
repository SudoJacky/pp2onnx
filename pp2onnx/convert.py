"""Command-line helpers for PP-OCRv5 Paddle-to-ONNX conversion and parity checks."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

from .models import MODEL_SPECS, ModelSpec

DEFAULT_OUTPUT_ROOT = Path("artifacts")
DEFAULT_MODELS = ("mobile_det", "mobile_rec")
TASK_CHOICES = ("auto", "det", "rec")


def safe_name(name: str) -> str:
    """Return a filesystem-safe stem for generated result artifacts."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "model"


def make_json_safe(value):
    """Convert common runtime result objects into JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return make_json_safe(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json(data: object, path: Path) -> Path:
    """Write JSON data to *path* and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_json_safe(data), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


class ConversionError(RuntimeError):
    """Raised when conversion or validation cannot complete."""


def _require_module(module_name: str, install_hint: str) -> None:
    try:
        __import__(module_name)
    except ImportError as exc:  # pragma: no cover - message path only
        raise ConversionError(f"Missing dependency '{module_name}'. Install it with: {install_hint}") from exc


def download_file(url: str, destination: Path) -> Path:
    """Download *url* to *destination* unless the file already exists."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:  # nosec B310 - trusted model URL or user supplied CLI URL
            shutil.copyfileobj(response, handle)
    except (OSError, urllib.error.URLError) as exc:
        tmp_path.unlink(missing_ok=True)
        raise ConversionError(f"Unable to download model from {url}: {exc}") from exc
    tmp_path.replace(destination)
    return destination


def _safe_members(tar: tarfile.TarFile, target_dir: Path) -> Iterable[tarfile.TarInfo]:
    """Yield tar members that stay inside *target_dir* after extraction."""
    target_root = target_dir.resolve()
    for member in tar.getmembers():
        if member.issym() or member.islnk():
            raise ConversionError(f"Unsafe link in archive: {member.name}")
        member_path = (target_dir / member.name).resolve()
        if os.path.commonpath([target_root, member_path]) != str(target_root):
            raise ConversionError(f"Unsafe path in archive: {member.name}")
        yield member


def extract_tar(archive_path: Path, target_dir: Path) -> Path:
    """Extract a Paddle inference tarball and return the inferred model directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    before = {path.resolve() for path in target_dir.iterdir()} if target_dir.exists() else set()
    with tarfile.open(archive_path) as tar:
        tar.extractall(target_dir, members=_safe_members(tar, target_dir))
    after = {path.resolve() for path in target_dir.iterdir()}
    new_dirs = [path for path in after - before if path.is_dir()]
    if len(new_dirs) == 1:
        return new_dirs[0]

    model_dirs = sorted(path for path in target_dir.iterdir() if path.is_dir() and has_paddle_model_files(path))
    if not model_dirs:
        raise ConversionError(f"No Paddle inference model files found after extracting {archive_path}")
    return model_dirs[-1]


def _find_paddle_model_files(model_dir: Path) -> tuple[Path | None, Path | None]:
    model_candidates = [model_dir / "inference.json", model_dir / "inference.pdmodel"]
    params_candidates = [model_dir / "inference.pdiparams"]
    model_file = next((path for path in model_candidates if path.exists()), None)
    params_file = next((path for path in params_candidates if path.exists()), None)
    if model_file is None or params_file is None:
        return None, None
    return model_file, params_file


def has_paddle_model_files(model_dir: Path) -> bool:
    """Return True when *model_dir* looks like a Paddle inference model directory."""
    model_file, params_file = _find_paddle_model_files(model_dir)
    return model_file is not None and params_file is not None


def resolve_model(spec_or_dir: str, download_dir: Path) -> Path:
    """Resolve a known model key, URL, or local Paddle model directory."""
    candidate = Path(spec_or_dir)
    if candidate.exists():
        if not candidate.is_dir():
            raise ConversionError(f"Model path must be a directory: {candidate}")
        return candidate

    spec = MODEL_SPECS.get(spec_or_dir)
    if spec is None and spec_or_dir.startswith(("http://", "https://")):
        name = Path(spec_or_dir).name.removesuffix(".tar")
        spec = ModelSpec(name=name, task="unknown", url=spec_or_dir)
    if spec is None:
        choices = ", ".join(sorted(MODEL_SPECS))
        raise ConversionError(f"Unknown model '{spec_or_dir}'. Use one of: {choices}; or pass a local directory/URL.")

    extracted_dir = download_dir / spec.name
    if has_paddle_model_files(extracted_dir):
        return extracted_dir
    archive_path = download_file(spec.url, download_dir / f"{spec.name}.tar")
    return extract_tar(archive_path, download_dir)


def infer_task(model_arg: str, paddle_model_dir: Path, task_override: str = "auto") -> str:
    """Infer whether a model is a detection or recognition model."""
    if task_override != "auto":
        return task_override
    spec = MODEL_SPECS.get(model_arg)
    if spec is not None and spec.task in {"det", "rec"}:
        return spec.task

    text = f"{model_arg} {paddle_model_dir.name}".lower()
    if "_rec" in text or "rec_" in text or "recognition" in text:
        return "rec"
    if "_det" in text or "det_" in text or "detection" in text:
        return "det"
    raise ConversionError(
        f"Unable to infer task for '{model_arg}' from {paddle_model_dir}. "
        "Pass --task det or --task rec for local/custom models."
    )


def convert_with_paddlex(paddle_model_dir: Path, onnx_model_dir: Path, opset_version: int = 7) -> Path:
    """Run the official PaddleX Paddle2ONNX plugin."""
    if shutil.which("paddlex") is None:
        raise ConversionError(
            "The 'paddlex' command was not found. Install dependencies, then run: paddlex --install paddle2onnx"
        )
    onnx_model_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "paddlex",
        "--paddle2onnx",
        "--paddle_model_dir",
        str(paddle_model_dir),
        "--onnx_model_dir",
        str(onnx_model_dir),
        "--opset_version",
        str(opset_version),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise ConversionError(
            f"PaddleX paddle2onnx conversion failed for {paddle_model_dir}. "
            "If the Paddle2ONNX plugin is missing, run: paddlex --install paddle2onnx"
        ) from exc

    onnx_files = sorted(onnx_model_dir.glob("*.onnx"))
    if not onnx_files:
        raise ConversionError(f"PaddleX finished but no .onnx file was written to {onnx_model_dir}")
    return onnx_files[0]


def preprocess_detection_image(image_path: Path, limit_side_len: int = 960):
    """Preprocess an image for PaddleOCR DB-style text detection."""
    _require_module("cv2", "python -m pip install opencv-contrib-python")
    _require_module("numpy", "python -m pip install numpy")
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    if image is None:
        raise ConversionError(f"Unable to read image: {image_path}")
    image = image.astype("float32")
    height, width = image.shape[:2]
    ratio = min(1.0, float(limit_side_len) / max(height, width))
    resize_h = max(32, int(round(height * ratio / 32) * 32))
    resize_w = max(32, int(round(width * ratio / 32) * 32))
    image = cv2.resize(image, (resize_w, resize_h))
    image = image[:, :, ::-1] / 255.0  # BGR to RGB
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    image = (image - mean) / std
    return np.expand_dims(image.transpose(2, 0, 1), axis=0).astype("float32")


def preprocess_recognition_image(image_path: Path, image_shape: tuple[int, int, int] = (3, 48, 320)):
    """Preprocess an image for PP-OCR recognition model tensor parity checks."""
    _require_module("cv2", "python -m pip install opencv-contrib-python")
    _require_module("numpy", "python -m pip install numpy")
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    if image is None:
        raise ConversionError(f"Unable to read image: {image_path}")

    channels, image_h, image_w = image_shape
    if channels != image.shape[2]:
        raise ConversionError(f"Recognition preprocessing expects {channels} channels, got {image.shape[2]}")

    h, w = image.shape[:2]
    resized_w = min(image_w, max(1, int(round(image_h * (w / float(h))))))
    resized = cv2.resize(image, (resized_w, image_h)).astype("float32")
    resized = resized.transpose((2, 0, 1)) / 255.0
    resized = (resized - 0.5) / 0.5

    padded = np.zeros((channels, image_h, image_w), dtype="float32")
    padded[:, :, :resized_w] = resized
    return np.expand_dims(padded, axis=0).astype("float32")


def preprocess_image_for_task(image_path: Path, task: str):
    """Build the model input tensor used for Paddle-vs-ONNX parity validation."""
    if task == "det":
        return preprocess_detection_image(image_path)
    if task == "rec":
        return preprocess_recognition_image(image_path)
    raise ConversionError(f"Unsupported validation task: {task}")


def run_paddle_inference(paddle_model_dir: Path, input_tensor) -> list:
    """Run a Paddle static inference model and return all outputs."""
    _require_module("paddle", "python -m pip install paddlepaddle")
    import paddle.inference as paddle_infer

    model_file, params_file = _find_paddle_model_files(paddle_model_dir)
    if not model_file or not params_file:
        raise ConversionError(f"Missing inference model files in {paddle_model_dir}")
    config = paddle_infer.Config(str(model_file), str(params_file))
    config.disable_gpu()
    if hasattr(config, "disable_onednn"):
        config.disable_onednn()
    if hasattr(config, "disable_mkldnn"):
        config.disable_mkldnn()
    config.switch_ir_optim(False)
    predictor = paddle_infer.create_predictor(config)
    input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
    input_handle.reshape(input_tensor.shape)
    input_handle.copy_from_cpu(input_tensor)
    predictor.run()
    return [predictor.get_output_handle(name).copy_to_cpu() for name in predictor.get_output_names()]


def run_onnx_inference(onnx_model_path: Path, input_tensor) -> list:
    """Run an ONNX model and return all outputs."""
    _require_module("onnxruntime", "python -m pip install onnxruntime")
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: input_tensor})


def array_metadata(array, file_path: Path) -> dict[str, object]:
    """Return serializable metadata for a saved tensor."""
    return {
        "file": str(file_path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min(initial=0.0)),
        "max": float(array.max(initial=0.0)),
        "mean": float(array.mean()) if array.size else 0.0,
    }


def export_native_onnx_outputs(
    export_root: Path,
    model_name: str,
    task: str,
    input_tensor,
    onnx_outputs: Sequence,
) -> dict[str, object]:
    """Save native ONNX Runtime input/output tensors and a JSON manifest."""
    _require_module("numpy", "python -m pip install numpy")
    import numpy as np

    model_dir = export_root / "native_onnx" / safe_name(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    input_path = model_dir / "input.npy"
    np.save(input_path, input_tensor)
    output_metadata = []
    for index, output in enumerate(onnx_outputs):
        output_path = model_dir / f"output_{index}.npy"
        np.save(output_path, output)
        output_metadata.append(array_metadata(output, output_path))

    manifest = {
        "model": model_name,
        "task": task,
        "runtime": "onnxruntime",
        "input": array_metadata(input_tensor, input_path),
        "outputs": output_metadata,
    }
    manifest_path = write_json(manifest, model_dir / "manifest.json")
    manifest["manifest_file"] = str(manifest_path)
    return manifest


def compare_outputs(paddle_outputs: Sequence, onnx_outputs: Sequence) -> dict[str, float | int | bool]:
    """Compare Paddle and ONNX outputs with numeric metrics."""
    _require_module("numpy", "python -m pip install numpy")
    import numpy as np
    if len(paddle_outputs) != len(onnx_outputs):
        raise ConversionError(f"Output count mismatch: Paddle={len(paddle_outputs)} ONNX={len(onnx_outputs)}")
    max_abs = 0.0
    mean_abs_values = []
    cosine_values = []
    for paddle_out, onnx_out in zip(paddle_outputs, onnx_outputs, strict=True):
        if paddle_out.shape != onnx_out.shape:
            raise ConversionError(f"Output shape mismatch: Paddle={paddle_out.shape} ONNX={onnx_out.shape}")
        diff = np.abs(paddle_out.astype("float64") - onnx_out.astype("float64"))
        max_abs = max(max_abs, float(diff.max(initial=0.0)))
        mean_abs_values.append(float(diff.mean()))
        paddle_flat = paddle_out.reshape(-1).astype("float64")
        onnx_flat = onnx_out.reshape(-1).astype("float64")
        denom = float(np.linalg.norm(paddle_flat) * np.linalg.norm(onnx_flat))
        cosine_values.append(float(np.dot(paddle_flat, onnx_flat) / denom) if denom else 1.0)
    return {
        "outputs": len(paddle_outputs),
        "max_abs_diff": max_abs,
        "mean_abs_diff": float(np.mean(mean_abs_values)),
        "min_cosine_similarity": float(np.min(cosine_values)),
    }


def validate_parity(
    paddle_model_dir: Path,
    onnx_model_path: Path,
    image_path: Path,
    task: str,
    max_abs_diff: float,
    min_cosine: float,
    export_root: Path | None = None,
    model_name: str | None = None,
) -> dict[str, object]:
    """Validate ONNX parity against Paddle on one image for a detection or recognition model."""
    input_tensor = preprocess_image_for_task(image_path, task)
    paddle_outputs = run_paddle_inference(paddle_model_dir, input_tensor)
    onnx_outputs = run_onnx_inference(onnx_model_path, input_tensor)
    metrics = compare_outputs(paddle_outputs, onnx_outputs)
    metrics["task"] = task
    metrics["input_shape"] = list(input_tensor.shape)
    metrics["passed"] = bool(metrics["max_abs_diff"] <= max_abs_diff and metrics["min_cosine_similarity"] >= min_cosine)
    metrics["max_abs_threshold"] = max_abs_diff
    metrics["min_cosine_threshold"] = min_cosine
    if export_root is not None:
        metrics["native_onnx_result"] = export_native_onnx_outputs(
            export_root,
            model_name or paddle_model_dir.name,
            task,
            input_tensor,
            onnx_outputs,
        )
    return metrics


def validate_detection(paddle_model_dir: Path, onnx_model_path: Path, image_path: Path, max_abs_diff: float, min_cosine: float) -> dict[str, object]:
    """Validate ONNX detection parity against Paddle on one image."""
    return validate_parity(paddle_model_dir, onnx_model_path, image_path, "det", max_abs_diff, min_cosine)


def infer_paddleocr_model_name(model_dir: Path, task: str) -> str | None:
    """Infer the PaddleOCR pipeline model name from a PP-OCR inference directory."""
    name = model_dir.name.lower()
    if "pp-ocrv5" not in name:
        return None
    size = "mobile" if "mobile" in name else "server" if "server" in name else None
    if size is None:
        return None
    return f"PP-OCRv5_{size}_{task}"


def export_paddleocr_result(
    det_model_dir: Path,
    rec_model_dir: Path,
    image_path: Path,
    export_root: Path,
) -> dict[str, object]:
    """Run the PaddleOCR pipeline and save its structured JSON result."""
    _require_module(
        "paddleocr",
        "python -m pip install paddleocr 'paddlex[ocr-core]'",
    )
    from paddleocr import PaddleOCR

    try:
        ocr = PaddleOCR(
            text_detection_model_name=infer_paddleocr_model_name(det_model_dir, "det"),
            text_detection_model_dir=str(det_model_dir),
            text_recognition_model_name=infer_paddleocr_model_name(rec_model_dir, "rec"),
            text_recognition_model_dir=str(rec_model_dir),
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        results = ocr.predict(str(image_path))
    except Exception as exc:  # pragma: no cover - depends on optional PaddleOCR runtime extras
        raise ConversionError(f"PaddleOCR result export failed: {exc}") from exc

    output_dir = export_root / "paddleocr"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_name(image_path.stem)
    json_files = []
    for index, result in enumerate(results):
        suffix = f"_{index}" if len(results) > 1 else ""
        result_path = output_dir / f"{stem}{suffix}.json"
        if hasattr(result, "json"):
            payload = result.json
        else:
            payload = {"result": str(result)}
        write_json(payload, result_path)
        json_files.append(str(result_path))

    manifest = {
        "runtime": "paddleocr",
        "image": str(image_path),
        "text_detection_model_dir": str(det_model_dir),
        "text_recognition_model_dir": str(rec_model_dir),
        "result_count": len(results),
        "json_files": json_files,
    }
    manifest_path = write_json(manifest, output_dir / "manifest.json")
    manifest["manifest_file"] = str(manifest_path)
    return manifest


def export_paddleocr_result_from_models(results: Sequence[dict[str, object]], image_path: Path, export_root: Path) -> dict[str, object]:
    """Export PaddleOCR output when both a det and rec model are available."""
    det_model_dir = next((Path(str(item["paddle_model_dir"])) for item in results if item.get("task") == "det"), None)
    rec_model_dir = next((Path(str(item["paddle_model_dir"])) for item in results if item.get("task") == "rec"), None)
    if det_model_dir is None or rec_model_dir is None:
        return {
            "skipped": True,
            "reason": "PaddleOCR export requires both one detection model and one recognition model.",
        }
    return export_paddleocr_result(det_model_dir, rec_model_dir, image_path, export_root)


def convert_and_validate_model(model_arg: str, args: argparse.Namespace) -> dict[str, object]:
    """Resolve, convert, and optionally validate one model from CLI arguments."""
    paddle_dir = resolve_model(model_arg, args.output_root / "paddle")
    task = infer_task(model_arg, paddle_dir, args.task)
    onnx_dir = args.output_root / "onnx" / paddle_dir.name
    if args.skip_convert:
        onnx_files = sorted(onnx_dir.glob("*.onnx"))
        if not onnx_files:
            raise ConversionError(f"No existing .onnx file found in {onnx_dir}")
        onnx_path = onnx_files[0]
    else:
        onnx_path = convert_with_paddlex(paddle_dir, onnx_dir, args.opset)

    result: dict[str, object] = {
        "model": model_arg,
        "task": task,
        "paddle_model_dir": str(paddle_dir),
        "onnx_model_path": str(onnx_path),
    }
    if not args.skip_validate:
        metrics = validate_parity(
            paddle_dir,
            onnx_path,
            args.image,
            task,
            args.max_abs_diff,
            args.min_cosine,
            args.results_dir if args.export_results else None,
            model_arg,
        )
        result["validation"] = metrics
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert PP-OCRv5 Paddle inference models to ONNX and validate parity.")
    known_keys = ", ".join(sorted(MODEL_SPECS))
    parser.add_argument(
        "--model",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help=f"One or more known keys, URLs, or local Paddle model directories. Default: {' '.join(DEFAULT_MODELS)}. Known keys: {known_keys}",
    )
    parser.add_argument("--task", choices=TASK_CHOICES, default="auto", help="Validation task for local/custom models; known keys are inferred automatically.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory for downloads and ONNX output.")
    parser.add_argument("--opset", type=int, default=7, help="ONNX opset version passed to PaddleX.")
    parser.add_argument("--image", type=Path, default=Path("test_paper.PNG"), help="Image used for parity validation.")
    parser.add_argument("--skip-convert", action="store_true", help="Use existing ONNX files under output-root/onnx instead of running PaddleX.")
    parser.add_argument("--skip-validate", action="store_true", help="Only download/convert; do not run Paddle vs ONNX parity validation.")
    parser.add_argument("--export-results", action="store_true", help="Export PaddleOCR JSON output plus native ONNX Runtime input/output tensors.")
    parser.add_argument("--results-dir", type=Path, default=None, help="Directory for exported PaddleOCR and native ONNX results. Defaults to output-root/results.")
    parser.add_argument("--max-abs-diff", type=float, default=1e-3, help="Maximum allowed absolute output difference.")
    parser.add_argument("--min-cosine", type=float, default=0.99999, help="Minimum allowed output cosine similarity.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.results_dir is None:
        args.results_dir = args.output_root / "results"
    try:
        results = [convert_and_validate_model(model_arg, args) for model_arg in args.model]
    except ConversionError as exc:
        print(f"pp2onnx: error: {exc}", file=sys.stderr)
        return 1

    failed = [result for result in results if result.get("validation", {}).get("passed") is False]
    payload: dict[str, object]
    if len(results) == 1:
        payload = results[0]
    else:
        payload = {"models": results}
    if args.export_results:
        try:
            payload["exports"] = {
                "results_dir": str(args.results_dir),
                "paddleocr": export_paddleocr_result_from_models(results, args.image, args.results_dir),
            }
        except ConversionError as exc:
            print(f"pp2onnx: error: {exc}", file=sys.stderr)
            return 1
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 2 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
