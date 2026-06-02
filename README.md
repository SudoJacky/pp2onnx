# PP-OCRv5 PaddleOCR 转 ONNX

这个仓库提供一个最小可运行的转换工具，用于把 PaddleOCR / PP-OCRv5 的 Paddle 静态图推理模型转换为 ONNX，并用仓库中的 `test_paper.PNG` 做 Paddle-vs-ONNX 数值一致性验证，降低转换后检测和识别输出漂移的风险。

## 设计要点

- 使用 PaddleOCR 官方推荐的 PaddleX `paddle2onnx` 插件完成转换。
- 默认使用 PP-OCRv5 **移动端模型**：`mobile_det` 和 `mobile_rec`，也内置 `server_det` / `server_rec` 官方推理模型下载地址。
- 支持一次传入多个模型 key、本地 Paddle 推理模型目录或自定义 tar URL。
- 对文本检测模型执行同一张图片、同一套 DB 检测预处理下的 Paddle 推理与 ONNX Runtime 推理，并比较输出张量。
- 对文本识别模型执行 PP-OCR 识别预处理（默认 `3x48x320` 归一化并右侧补零），再比较 Paddle 与 ONNX 输出张量。
- 默认验收阈值：
  - `max_abs_diff` 不超过 `1e-3`；
  - `min_cosine_similarity` 不低于 `0.99999`。

> 注意：ONNX 转换只能保证模型图和权重的数值等价。最终 OCR 效果还依赖 PaddleOCR 的预处理、后处理、阈值、字典和部署端实现。检测精度验收建议优先比较检测模型的原始输出张量，再在业务数据集上做端到端 OCR 回归测试；识别模型建议额外使用真实文本裁剪图做回归。

## 环境安装

建议使用 Python 3.10 或 3.11 的独立虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[convert]'
paddlex --install paddle2onnx
```

如果 PaddlePaddle 安装受平台影响，请先按 PaddlePaddle 官网选择 CPU/GPU 对应安装命令，然后再安装本项目依赖。

## 一键转换并验证 PP-OCRv5 移动端 det + rec 模型

默认转换 `mobile_det` 和 `mobile_rec`，并使用 `test_paper.PNG` 做输出一致性验证：

```bash
pp2onnx --image test_paper.PNG
```

等价的显式写法：

```bash
pp2onnx --model mobile_det mobile_rec --image test_paper.PNG
```

常用参数：

```bash
# 只转换移动端检测模型
pp2onnx --model mobile_det --image test_paper.PNG

# 只转换移动端识别模型；建议把 --image 换成真实文本裁剪图做识别回归
pp2onnx --model mobile_rec --image test_paper.PNG

# 使用服务端检测和识别模型
pp2onnx --model server_det server_rec --image test_paper.PNG

# 使用本地 Paddle 推理模型目录，并手动声明任务类型
pp2onnx --model /path/to/PP-OCRv5_mobile_rec_infer --task rec --image test_paper.PNG

# 只转换，不验证
pp2onnx --model mobile_det mobile_rec --skip-validate

# 已经有 ONNX 文件时，只重新跑验证
pp2onnx --model mobile_det mobile_rec --skip-convert --image test_paper.PNG
```

输出示例：

```json
{
  "models": [
    {
      "model": "mobile_det",
      "task": "det",
      "paddle_model_dir": "artifacts/paddle/PP-OCRv5_mobile_det_infer",
      "onnx_model_path": "artifacts/onnx/PP-OCRv5_mobile_det_infer/inference.onnx",
      "validation": {
        "outputs": 1,
        "max_abs_diff": 0.000123,
        "mean_abs_diff": 0.000001,
        "min_cosine_similarity": 0.999999,
        "task": "det",
        "input_shape": [1, 3, 960, 960],
        "passed": true,
        "max_abs_threshold": 0.001,
        "min_cosine_threshold": 0.99999
      }
    },
    {
      "model": "mobile_rec",
      "task": "rec",
      "paddle_model_dir": "artifacts/paddle/PP-OCRv5_mobile_rec_infer",
      "onnx_model_path": "artifacts/onnx/PP-OCRv5_mobile_rec_infer/inference.onnx",
      "validation": {
        "outputs": 1,
        "max_abs_diff": 0.000123,
        "mean_abs_diff": 0.000001,
        "min_cosine_similarity": 0.999999,
        "task": "rec",
        "input_shape": [1, 3, 48, 320],
        "passed": true,
        "max_abs_threshold": 0.001,
        "min_cosine_threshold": 0.99999
      }
    }
  ]
}
```

如果 `passed` 为 `false`，建议：

1. 提高 ONNX opset 后重新转换，例如 `--opset 11` 或 `--opset 13`；
2. 确认 PaddlePaddle、PaddleOCR、PaddleX、paddle2onnx 和 onnxruntime 版本兼容；
3. 使用 `test_paper.PNG` 之外的业务样本重复验证，避免只对单张图片过拟合验收；
4. 对 `mobile_rec` 使用真实文本框裁剪图验证识别 logits 和后处理文本结果。


## PP-OCRv5 Mobile FP16 / INT8 ONNX 模型

仓库现在内置了两组 PP-OCRv5 Mobile ONNX 量化模型，命名和输入保持 RapidOCR / OnnxOCR 常见部署方式：检测模型使用动态 `NCHW` 输入，识别模型使用 `N x 3 x 48 x W` 输入，部署端继续复用 PP-OCR 的 det/rec 前后处理。

| 文件 | 任务 | 精度 | 说明 |
| --- | --- | --- | --- |
| `models/ppocrv5_mobile/ppocrv5_mobile_det_fp16.onnx` | 文本检测 | FP16 | 保留 float32 输入/输出，图内部权重和中间张量转 FP16。 |
| `models/ppocrv5_mobile/ppocrv5_mobile_det_int8.onnx` | 文本检测 | INT8 | ONNX Runtime static QDQ，量化 `Conv` / `MatMul`，使用 PP-OCR 检测预处理校准。 |
| `models/ppocrv5_mobile/ppocrv5_mobile_rec_fp16.onnx` | 文本识别 | FP16 | 保留 float32 输入/输出，图内部权重和中间张量转 FP16。 |
| `models/ppocrv5_mobile/ppocrv5_mobile_rec_int8.onnx` | 文本识别 | INT8 | ONNX Runtime static QDQ，量化 `Conv` / `MatMul`，使用 PP-OCR 识别预处理校准。 |

生成这些模型的命令如下；源 ONNX 可以来自本仓库转换出的 FP32 ONNX，也可以使用其他 PP-OCRv5 mobile det/rec ONNX：

```bash
uv run --extra quantize python -m pp2onnx.quantize \
  --model det=/path/to/ppocrv5_det.onnx \
  --model rec=/path/to/ppocrv5_rec.onnx \
  --output-dir models/ppocrv5_mobile \
  --calibration-image test_paper.PNG
```

INT8 模型默认使用 per-tensor QDQ，是为了兼容常见的 PP-OCRv5 opset 11 ONNX；如果你的源模型 opset 和运行端支持 per-channel QDQ，可以额外加 `--int8-per-channel`。量化只改变 ONNX 图精度，不改变 OCR pipeline：检测后处理阈值、unclip、文本框排序、识别字典和解码仍需由 RapidOCR / OnnxOCR / 业务部署端提供。

## 项目结构

```text
pp2onnx/
  convert.py   # 下载、转换、det/rec 模型数值验证 CLI
  models.py    # PP-OCRv5 官方推理模型 URL

tests/
  test_convert.py
```

## 开发测试

```bash
python -m pip install -e '.[test]'
pytest
```

## 两个独立脚本

如果只需要独立脚本，可以直接使用仓库根目录下的两个文件，不需要安装 `pp2onnx` 包本身。

### 1. `convert_ppocrv5_to_onnx.py`

将 PP-OCRv5 Paddle 推理模型转换成 ONNX。脚本接受：

- 内置模型别名：`mobile_det`、`mobile_rec`、`server_det`、`server_rec`；
- 本地 Paddle 推理模型目录，目录内需要包含 `inference.json` 或 `inference.pdmodel`，以及 `inference.pdiparams`；
- 自定义 Paddle 推理模型 tar URL。

```bash
python convert_ppocrv5_to_onnx.py \
  --model mobile_det \
  --output onnx_models/PP-OCRv5_mobile_det.onnx
```

也可以转换本地目录：

```bash
python convert_ppocrv5_to_onnx.py \
  --model /path/to/PP-OCRv5_mobile_det_infer \
  --output onnx_models/PP-OCRv5_mobile_det.onnx
```

默认后端是 `paddle2onnx` 命令行；如需复用 PaddleX 插件，可以加 `--backend paddlex`。

### 2. `detect_ppocrv5_onnx.py`

使用 ONNX Runtime 加载转换后的 PP-OCRv5 检测模型，并对图片或 PDF 页面执行文本框检测。检测前处理、DB 后处理阈值、膨胀、unclip 和文本框排序默认参考 RapidOCR 对 PP-OCR 系列检测模型的处理方式。

```bash
python -m pip install '.[detect]'
python detect_ppocrv5_onnx.py \
  --model onnx_models/PP-OCRv5_mobile_det.onnx \
  --input test_paper.PNG sample.pdf \
  --output-json det_results.json \
  --vis-dir det_vis
```

输出 JSON 中每个图片或 PDF 页面包含原始宽高、检测框四点坐标、置信度、耗时和可视化图片路径。
