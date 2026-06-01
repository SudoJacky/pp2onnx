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
