from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from pp2onnx.convert import (
    ConversionError,
    build_parser,
    compare_outputs,
    extract_tar,
    infer_task,
    preprocess_recognition_image,
    resolve_model,
)


def _write_tar(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_extract_tar_returns_model_dir(tmp_path: Path) -> None:
    archive = tmp_path / "model.tar"
    _write_tar(
        archive,
        {
            "demo/inference.json": b"{}",
            "demo/inference.pdiparams": b"params",
        },
    )

    model_dir = extract_tar(archive, tmp_path / "out")

    assert model_dir.name == "demo"
    assert (model_dir / "inference.json").exists()


def test_extract_tar_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar"
    _write_tar(archive, {"../evil": b"nope"})

    with pytest.raises(ConversionError, match="Unsafe path"):
        extract_tar(archive, tmp_path / "out")


def test_extract_tar_rejects_links(tmp_path: Path) -> None:
    archive = tmp_path / "bad_link.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("demo/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../evil"
        tar.addfile(info)

    with pytest.raises(ConversionError, match="Unsafe link"):
        extract_tar(archive, tmp_path / "out")


def test_resolve_model_accepts_local_dir(tmp_path: Path) -> None:
    model_dir = tmp_path / "local_model"
    model_dir.mkdir()

    assert resolve_model(str(model_dir), tmp_path / "downloads") == model_dir


def test_infer_task_prefers_known_model_key(tmp_path: Path) -> None:
    assert infer_task("mobile_det", tmp_path / "anything") == "det"
    assert infer_task("mobile_rec", tmp_path / "anything") == "rec"


def test_infer_task_uses_local_model_name(tmp_path: Path) -> None:
    assert infer_task(str(tmp_path / "custom_rec_dir"), tmp_path / "custom_rec_dir") == "rec"
    assert infer_task(str(tmp_path / "custom_det_dir"), tmp_path / "custom_det_dir") == "det"


def test_default_cli_models_are_mobile_det_and_rec() -> None:
    args = build_parser().parse_args([])

    assert args.model == ["mobile_det", "mobile_rec"]


def test_compare_outputs_metrics() -> None:
    np = pytest.importorskip("numpy")
    paddle = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
    onnx = [np.array([1.0, 2.001, 2.999], dtype=np.float32)]

    metrics = compare_outputs(paddle, onnx)

    assert metrics["outputs"] == 1
    assert metrics["max_abs_diff"] == pytest.approx(0.001, rel=1e-3)
    assert metrics["min_cosine_similarity"] > 0.999999


def test_compare_outputs_rejects_shape_mismatch() -> None:
    np = pytest.importorskip("numpy")
    with pytest.raises(ConversionError, match="Output shape mismatch"):
        compare_outputs([np.zeros((1, 2))], [np.zeros((2, 1))])


def test_preprocess_recognition_image_pads_to_ppocr_shape(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image_path = tmp_path / "word.png"
    cv2.imwrite(str(image_path), np.full((20, 80, 3), 127, dtype=np.uint8))

    tensor = preprocess_recognition_image(image_path)

    assert tensor.shape == (1, 3, 48, 320)
    assert tensor.dtype == np.float32


def test_quantize_model_parser_accepts_task_paths() -> None:
    from pp2onnx.quantize import build_parser

    args = build_parser().parse_args(["--model", "det=det.onnx", "--model", "rec=rec.onnx"])

    assert args.model == [("det", Path("det.onnx")), ("rec", Path("rec.onnx"))]
    assert args.precision == ["fp16", "int8"]


def test_quantized_output_path_uses_ppocrv5_mobile_names(tmp_path: Path) -> None:
    from pp2onnx.quantize import quantized_output_path

    assert quantized_output_path(tmp_path, Path("ppocrv5_det.onnx"), "det", "fp16") == tmp_path / "ppocrv5_mobile_det_fp16.onnx"
    assert quantized_output_path(tmp_path, Path("ppocrv5_rec.onnx"), "rec", "int8") == tmp_path / "ppocrv5_mobile_rec_int8.onnx"


def test_tensor_calibration_data_reader_yields_once() -> None:
    np = pytest.importorskip("numpy")
    from pp2onnx.quantize import TensorCalibrationDataReader

    tensor = np.zeros((1, 3, 48, 320), dtype=np.float32)
    reader = TensorCalibrationDataReader("x", [tensor])

    first = reader.get_next()
    assert first is not None
    assert first["x"].shape == (1, 3, 48, 320)
    assert reader.get_next() is None
