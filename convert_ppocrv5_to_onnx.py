#!/usr/bin/env python3
"""Standalone PP-OCRv5 Paddle inference model to ONNX converter.

The script intentionally has no dependency on this repository's package modules.  It
accepts either a local Paddle inference directory containing ``inference.json`` or
``inference.pdmodel`` plus ``inference.pdiparams``, or one of the built-in PP-OCRv5
model aliases and then writes a single ONNX file.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

BASE_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0"
MODEL_URLS = {
    "mobile_det": f"{BASE_URL}/PP-OCRv5_mobile_det_infer.tar",
    "mobile_rec": f"{BASE_URL}/PP-OCRv5_mobile_rec_infer.tar",
    "server_det": f"{BASE_URL}/PP-OCRv5_server_det_infer.tar",
    "server_rec": f"{BASE_URL}/PP-OCRv5_server_rec_infer.tar",
}
MODEL_FILENAMES = ("inference.json", "inference.pdmodel")
PARAMS_FILENAME = "inference.pdiparams"


class ConvertError(RuntimeError):
    """Raised when a model cannot be downloaded, resolved, or converted."""


def find_model_files(model_dir: Path) -> tuple[Path, Path]:
    model_file = next((model_dir / name for name in MODEL_FILENAMES if (model_dir / name).exists()), None)
    params_file = model_dir / PARAMS_FILENAME
    if model_file is None or not params_file.exists():
        expected = " or ".join(MODEL_FILENAMES)
        raise ConvertError(f"{model_dir} must contain {expected} and {PARAMS_FILENAME}")
    return model_file, params_file


def has_model_files(model_dir: Path) -> bool:
    try:
        find_model_files(model_dir)
    except ConvertError:
        return False
    return True


def download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url) as response, tmp.open("wb") as f:  # nosec B310 - user-selected model URL
            shutil.copyfileobj(response, f)
    except (OSError, urllib.error.URLError) as exc:
        tmp.unlink(missing_ok=True)
        raise ConvertError(f"failed to download {url}: {exc}") from exc
    tmp.replace(target)
    return target


def safe_tar_members(tar: tarfile.TarFile, target_dir: Path) -> Iterable[tarfile.TarInfo]:
    root = target_dir.resolve()
    for member in tar.getmembers():
        if member.issym() or member.islnk():
            raise ConvertError(f"unsafe link in archive: {member.name}")
        member_path = (target_dir / member.name).resolve()
        if os.path.commonpath([root, member_path]) != str(root):
            raise ConvertError(f"unsafe path in archive: {member.name}")
        yield member


def extract_tar(archive: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(target_dir, members=safe_tar_members(tar, target_dir))
    candidates = sorted(path for path in target_dir.iterdir() if path.is_dir() and has_model_files(path))
    if not candidates:
        raise ConvertError(f"no Paddle inference model found after extracting {archive}")
    return candidates[-1]


def resolve_model(model: str, download_dir: Path) -> Path:
    local = Path(model).expanduser()
    if local.exists():
        if not local.is_dir():
            raise ConvertError(f"local model path must be a directory: {local}")
        find_model_files(local)
        return local

    url = MODEL_URLS.get(model, model if model.startswith(("http://", "https://")) else None)
    if url is None:
        known = ", ".join(sorted(MODEL_URLS))
        raise ConvertError(f"unknown model '{model}'. Use one of: {known}; a local directory; or a tar URL")

    name = Path(url).name.removesuffix(".tar")
    extracted = download_dir / name
    if has_model_files(extracted):
        return extracted
    archive = download(url, download_dir / f"{name}.tar")
    return extract_tar(archive, download_dir)


def run_command(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def convert_with_paddle2onnx(model_dir: Path, output_path: Path, opset: int, enable_onnx_checker: bool) -> None:
    model_file, params_file = find_model_files(model_dir)
    if shutil.which("paddle2onnx") is None:
        raise ConvertError("paddle2onnx command was not found. Install with: python -m pip install paddle2onnx")
    command = [
        "paddle2onnx",
        "--model_dir",
        str(model_dir),
        "--model_filename",
        model_file.name,
        "--params_filename",
        params_file.name,
        "--save_file",
        str(output_path),
        "--opset_version",
        str(opset),
    ]
    if enable_onnx_checker:
        command += ["--enable_onnx_checker", "True"]
    run_command(command)


def convert_with_paddlex(model_dir: Path, output_dir: Path, opset: int) -> Path:
    if shutil.which("paddlex") is None:
        raise ConvertError("paddlex command was not found. Install PaddleX and run: paddlex --install paddle2onnx")
    run_command([
        "paddlex",
        "--paddle2onnx",
        "--paddle_model_dir",
        str(model_dir),
        "--onnx_model_dir",
        str(output_dir),
        "--opset_version",
        str(opset),
    ])
    onnx_files = sorted(output_dir.glob("*.onnx"))
    if not onnx_files:
        raise ConvertError(f"PaddleX finished but wrote no .onnx file under {output_dir}")
    return onnx_files[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a PP-OCRv5 Paddle inference model (.pdiparams + graph) to ONNX.")
    parser.add_argument("--model", default="mobile_det", help="Built-in alias, local inference dir, or tar URL. Default: mobile_det")
    parser.add_argument("--output", type=Path, default=None, help="Output .onnx path. Default: <output-dir>/<model_dir_name>.onnx")
    parser.add_argument("--output-dir", type=Path, default=Path("onnx_models"), help="Directory used when --output is omitted.")
    parser.add_argument("--download-dir", type=Path, default=Path("paddle_models"), help="Directory for downloaded/extracted Paddle models.")
    parser.add_argument("--opset", type=int, default=7, help="ONNX opset version. PP-OCR deployment commonly uses opset 7.")
    parser.add_argument("--backend", choices=("paddle2onnx", "paddlex"), default="paddle2onnx", help="Conversion command to run.")
    parser.add_argument("--enable-onnx-checker", action="store_true", help="Pass --enable_onnx_checker True to paddle2onnx.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        model_dir = resolve_model(args.model, args.download_dir)
        output_path = args.output or (args.output_dir / f"{model_dir.name}.onnx")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.backend == "paddlex":
            converted = convert_with_paddlex(model_dir, output_path.parent, args.opset)
            if converted != output_path:
                shutil.copy2(converted, output_path)
        else:
            convert_with_paddle2onnx(model_dir, output_path, args.opset, args.enable_onnx_checker)
        if not output_path.exists() or output_path.stat().st_size <= 0:
            raise ConvertError(f"conversion finished but output file is missing or empty: {output_path}")
    except (ConvertError, subprocess.CalledProcessError) as exc:
        print(f"convert_ppocrv5_to_onnx.py: error: {exc}", file=sys.stderr)
        return 1

    print(f"Paddle model: {model_dir}")
    print(f"ONNX model:   {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
