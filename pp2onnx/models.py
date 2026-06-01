"""Known PP-OCRv5 Paddle inference model URLs."""
from __future__ import annotations

from dataclasses import dataclass

BASE_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0"


@dataclass(frozen=True)
class ModelSpec:
    """Download metadata for one PaddleOCR inference model."""

    name: str
    task: str
    url: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "mobile_det": ModelSpec(
        name="PP-OCRv5_mobile_det_infer",
        task="det",
        url=f"{BASE_URL}/PP-OCRv5_mobile_det_infer.tar",
    ),
    "server_det": ModelSpec(
        name="PP-OCRv5_server_det_infer",
        task="det",
        url=f"{BASE_URL}/PP-OCRv5_server_det_infer.tar",
    ),
    "mobile_rec": ModelSpec(
        name="PP-OCRv5_mobile_rec_infer",
        task="rec",
        url=f"{BASE_URL}/PP-OCRv5_mobile_rec_infer.tar",
    ),
    "server_rec": ModelSpec(
        name="PP-OCRv5_server_rec_infer",
        task="rec",
        url=f"{BASE_URL}/PP-OCRv5_server_rec_infer.tar",
    ),
}
