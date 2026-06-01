#!/usr/bin/env python3
"""Standalone PP-OCRv5 ONNX Runtime text detector for images and PDFs.

The preprocessing and DB postprocessing defaults mirror RapidOCR's PP-OCR text
Detector path: resize to a multiple of 32, normalize with mean/std 0.5, threshold
DB maps, dilate, unclip boxes, filter tiny boxes, and sort boxes top-to-bottom then
left-to-right.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import cv2
except ImportError:  # pragma: no cover - dependency hint path
    cv2 = None
try:
    import numpy as np
except ImportError:  # pragma: no cover - dependency hint path
    np = None
try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - dependency hint path
    ort = None
try:
    import pyclipper
except ImportError:  # pragma: no cover - dependency hint path
    pyclipper = None
try:
    from shapely.geometry import Polygon
except ImportError:  # pragma: no cover - dependency hint path
    Polygon = None

_BOX_SORT_Y_THRESHOLD = 10


def require_runtime_dependencies() -> None:
    missing = []
    if cv2 is None:
        missing.append("opencv-python-headless")
    if np is None:
        missing.append("numpy")
    if ort is None:
        missing.append("onnxruntime")
    if pyclipper is None:
        missing.append("pyclipper")
    if Polygon is None:
        missing.append("shapely")
    if missing:
        raise RuntimeError("missing dependencies: " + ", ".join(missing) + ". Install with: python -m pip install " + " ".join(missing))


IMAGE_SUFFIXES = {".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".webp", ".pbm", ".pgm", ".ppm", ".pxm", ".pnm", ".tif", ".tiff"}
PDF_SUFFIX = ".pdf"


@dataclass
class PageResult:
    source: str
    page_index: int | None
    width: int
    height: int
    boxes: np.ndarray
    scores: list[float]
    elapsed: float
    visualization: str | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "source": self.source,
            "page_index": self.page_index,
            "width": self.width,
            "height": self.height,
            "elapsed": self.elapsed,
            "boxes": self.boxes.astype(int).tolist(),
            "scores": [float(score) for score in self.scores],
            "visualization": self.visualization,
        }


class DetPreProcess:
    def __init__(self, limit_side_len: int = 736, limit_type: str = "min", mean=None, std=None):
        self.limit_side_len = limit_side_len
        self.limit_type = limit_type
        self.mean = np.array([0.5, 0.5, 0.5] if mean is None else mean, dtype=np.float32)
        self.std = np.array([0.5, 0.5, 0.5] if std is None else std, dtype=np.float32)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        resized = self.resize(image)
        normalized = (resized.astype(np.float32) / 255.0 - self.mean) / self.std
        return np.expand_dims(normalized.transpose(2, 0, 1), axis=0).astype(np.float32)

    def resize(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if self.limit_type == "max":
            ratio = min(1.0, float(self.limit_side_len) / max(h, w))
        else:
            ratio = max(1.0, float(self.limit_side_len) / min(h, w))
        resize_h = max(32, int(round((h * ratio) / 32) * 32))
        resize_w = max(32, int(round((w * ratio) / 32) * 32))
        return cv2.resize(image, (resize_w, resize_h))


class DBPostProcess:
    def __init__(
        self,
        thresh: float = 0.3,
        box_thresh: float = 0.5,
        max_candidates: int = 1000,
        unclip_ratio: float = 1.6,
        score_mode: str = "fast",
        use_dilation: bool = True,
    ):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.score_mode = score_mode
        self.min_size = 3
        self.dilation_kernel = np.array([[1, 1], [1, 1]], dtype=np.uint8) if use_dilation else None

    def __call__(self, pred: np.ndarray, original_shape: tuple[int, int]) -> tuple[np.ndarray, list[float]]:
        pred = normalize_prediction(pred)
        src_h, src_w = original_shape
        bitmap = pred > self.thresh
        mask = bitmap[0]
        if self.dilation_kernel is not None:
            mask = cv2.dilate(mask.astype(np.uint8), self.dilation_kernel)
        boxes, scores = self.boxes_from_bitmap(pred[0], mask, src_w, src_h)
        return self.filter_det_res(boxes, scores, src_h, src_w)

    def boxes_from_bitmap(self, pred: np.ndarray, bitmap: np.ndarray, dest_width: int, dest_height: int) -> tuple[np.ndarray, list[float]]:
        height, width = bitmap.shape
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes, scores = [], []
        for contour in contours[: self.max_candidates]:
            points, small_side = self.get_mini_boxes(contour)
            if small_side < self.min_size:
                continue
            score = self.box_score_fast(pred, points.reshape(-1, 2)) if self.score_mode == "fast" else self.box_score_slow(pred, contour)
            if score < self.box_thresh:
                continue
            expanded = self.unclip(points)
            if expanded.size == 0:
                continue
            box, small_side = self.get_mini_boxes(expanded)
            if small_side < self.min_size + 2:
                continue
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int32))
            scores.append(float(score))
        return np.array(boxes, dtype=np.int32), scores

    @staticmethod
    def get_mini_boxes(contour: np.ndarray) -> tuple[np.ndarray, float]:
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        if points[1][1] > points[0][1]:
            index_1, index_4 = 0, 1
        else:
            index_1, index_4 = 1, 0
        if points[3][1] > points[2][1]:
            index_2, index_3 = 2, 3
        else:
            index_2, index_3 = 3, 2
        return np.array([points[index_1], points[index_2], points[index_3], points[index_4]]), min(bounding_box[1])

    @staticmethod
    def box_score_fast(bitmap: np.ndarray, box: np.ndarray) -> float:
        h, w = bitmap.shape[:2]
        box = box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] -= xmin
        box[:, 1] -= ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return float(cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0])

    @staticmethod
    def box_score_slow(bitmap: np.ndarray, contour: np.ndarray) -> float:
        h, w = bitmap.shape[:2]
        contour = contour.reshape(-1, 2).copy()
        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)
        mask = np.zeros((int(ymax - ymin) + 1, int(xmax - xmin) + 1), dtype=np.uint8)
        contour[:, 0] -= xmin
        contour[:, 1] -= ymin
        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return float(cv2.mean(bitmap[int(ymin) : int(ymax) + 1, int(xmin) : int(xmax) + 1], mask)[0])

    def unclip(self, box: np.ndarray) -> np.ndarray:
        polygon = Polygon(box)
        if polygon.length <= 0:
            return np.empty((0, 1, 2), dtype=np.float32)
        distance = polygon.area * self.unclip_ratio / polygon.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        if not expanded:
            return np.empty((0, 1, 2), dtype=np.float32)
        return np.array(expanded, dtype=np.float32).reshape((-1, 1, 2))

    def filter_det_res(self, boxes: np.ndarray, scores: list[float], img_height: int, img_width: int) -> tuple[np.ndarray, list[float]]:
        filtered_boxes, filtered_scores = [], []
        for box, score in zip(boxes, scores):
            box = order_points_clockwise(box)
            box = clip_box(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            filtered_boxes.append(box.astype(np.int32))
            filtered_scores.append(score)
        return np.array(filtered_boxes, dtype=np.int32), filtered_scores


def normalize_prediction(pred: np.ndarray) -> np.ndarray:
    """Return a DB probability map shaped [N, H, W] from common ONNX output layouts."""
    pred = np.asarray(pred)
    if pred.ndim == 4:
        return pred[:, 0, :, :]
    if pred.ndim == 3:
        return pred
    if pred.ndim == 2:
        return pred[np.newaxis, :, :]
    raise ValueError(f"unsupported detection output shape: {pred.shape}")


def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    tl, bl = left_most
    right_most = right_most[np.argsort(right_most[:, 1]), :]
    tr, br = right_most
    return np.array([tl, tr, br, bl], dtype=np.float32)


def clip_box(points: np.ndarray, img_height: int, img_width: int) -> np.ndarray:
    points[:, 0] = np.clip(points[:, 0], 0, img_width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, img_height - 1)
    return points


def sorted_box_indices(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    y_coords = boxes[:, 0, 1]
    y_order = np.argsort(y_coords, kind="stable")
    boxes_y_sorted = boxes[y_order]
    y_sorted = y_coords[y_order]
    line_ids = np.concatenate([[0], np.cumsum((np.diff(y_sorted) >= _BOX_SORT_Y_THRESHOLD).astype(np.int32))])
    x_coords = boxes_y_sorted[:, 0, 0]
    return y_order[np.lexsort((x_coords, line_ids))]


def load_pdf_pages(path: Path, dpi: int) -> Iterable[tuple[int, np.ndarray]]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError("PDF input requires PyMuPDF. Install with: python -m pip install pymupdf") from exc
    doc = fitz.open(path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for page_index in range(doc.page_count):
        pix = doc[page_index].get_pixmap(matrix=matrix, alpha=False)
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
        yield page_index, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def iter_inputs(paths: list[Path], dpi: int) -> Iterable[tuple[Path, int | None, np.ndarray]]:
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == PDF_SUFFIX:
            for page_index, image in load_pdf_pages(path, dpi):
                yield path, page_index, image
        elif suffix in IMAGE_SUFFIXES:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"failed to read image: {path}")
            yield path, None, image
        else:
            raise RuntimeError(f"unsupported input type: {path}")


def draw_boxes(image: np.ndarray, boxes: np.ndarray, scores: list[float]) -> np.ndarray:
    canvas = image.copy()
    for box, score in zip(boxes, scores):
        pts = box.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        x, y = box[0]
        cv2.putText(canvas, f"{score:.2f}", (int(x), max(0, int(y) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return canvas


def detect_one(session: ort.InferenceSession, preprocess: DetPreProcess, postprocess: DBPostProcess, image: np.ndarray) -> tuple[np.ndarray, list[float], float]:
    input_name = session.get_inputs()[0].name
    tensor = preprocess(image)
    start = time.perf_counter()
    outputs = session.run(None, {input_name: tensor})
    boxes, scores = postprocess(outputs[0], image.shape[:2])
    order = sorted_box_indices(boxes)
    boxes = boxes[order]
    scores = [scores[int(idx)] for idx in order]
    elapsed = time.perf_counter() - start
    return boxes, scores, elapsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PP-OCRv5 ONNX text detection on images and PDFs.")
    parser.add_argument("--model", type=Path, required=True, help="Path to converted PP-OCRv5 detection ONNX model.")
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="One or more image/PDF files.")
    parser.add_argument("--output-json", type=Path, default=Path("det_results.json"), help="JSON result path.")
    parser.add_argument("--vis-dir", type=Path, default=None, help="Optional directory for visualization images.")
    parser.add_argument("--dpi", type=int, default=200, help="PDF rendering DPI.")
    parser.add_argument("--limit-side-len", type=int, default=736, help="RapidOCR-style detection resize side limit.")
    parser.add_argument("--limit-type", choices=("min", "max"), default="min", help="Resize by minimum or maximum side.")
    parser.add_argument("--thresh", type=float, default=0.3, help="DB binary map threshold.")
    parser.add_argument("--box-thresh", type=float, default=0.5, help="Minimum box score.")
    parser.add_argument("--unclip-ratio", type=float, default=1.6, help="DB box expansion ratio.")
    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"], help="ONNX Runtime providers.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        require_runtime_dependencies()
        session = ort.InferenceSession(str(args.model), providers=args.providers)
        preprocess = DetPreProcess(args.limit_side_len, args.limit_type)
        postprocess = DBPostProcess(thresh=args.thresh, box_thresh=args.box_thresh, unclip_ratio=args.unclip_ratio)
        if args.vis_dir is not None:
            args.vis_dir.mkdir(parents=True, exist_ok=True)

        results: list[PageResult] = []
        for source, page_index, image in iter_inputs(args.input, args.dpi):
            boxes, scores, elapsed = detect_one(session, preprocess, postprocess, image)
            vis_path = None
            if args.vis_dir is not None:
                stem = source.stem if page_index is None else f"{source.stem}_page_{page_index + 1:04d}"
                vis_path = str(args.vis_dir / f"{stem}_det.jpg")
                cv2.imwrite(vis_path, draw_boxes(image, boxes, scores))
            results.append(PageResult(str(source), page_index, image.shape[1], image.shape[0], boxes, scores, elapsed, vis_path))

        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps([result.to_json() for result in results], ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"detect_ppocrv5_onnx.py: error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(results)} page/image result(s) to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
