from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from pp2onnx.convert import (
    ConversionError,
    build_parser,
    compare_outputs,
    export_native_onnx_outputs,
    extract_tar,
    infer_paddleocr_model_name,
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


def test_infer_paddleocr_model_name_for_known_ppocr_dirs() -> None:
    assert infer_paddleocr_model_name(Path("PP-OCRv5_mobile_det_infer"), "det") == "PP-OCRv5_mobile_det"
    assert infer_paddleocr_model_name(Path("PP-OCRv5_server_rec_infer"), "rec") == "PP-OCRv5_server_rec"
    assert infer_paddleocr_model_name(Path("custom_model"), "det") is None


def test_default_cli_models_are_mobile_det_and_rec() -> None:
    args = build_parser().parse_args([])

    assert args.model == ["mobile_det", "mobile_rec"]
    assert args.export_results is False
    assert args.results_dir is None


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


def test_export_native_onnx_outputs_writes_manifest_and_tensors(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    input_tensor = np.zeros((1, 3, 4, 4), dtype=np.float32)
    output_tensor = np.ones((1, 1, 2, 2), dtype=np.float32)

    manifest = export_native_onnx_outputs(tmp_path, "mobile/det", "det", input_tensor, [output_tensor])

    manifest_path = Path(manifest["manifest_file"])
    assert manifest_path.exists()
    assert (tmp_path / "native_onnx" / "mobile_det" / "input.npy").exists()
    assert (tmp_path / "native_onnx" / "mobile_det" / "output_0.npy").exists()
    assert manifest["runtime"] == "onnxruntime"
    assert manifest["outputs"][0]["shape"] == [1, 1, 2, 2]
