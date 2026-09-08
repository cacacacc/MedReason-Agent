# 模型接入说明

本文档说明如何把 Direct VLM baseline 从 `mock` backend 切换到真实 Qwen2.5-VL backend。

## 核心思想

实验管线不应该因为换模型而重写。我们只替换这一层：

```text
mock backend -> qwen2_5_vl backend
```

其他部分保持不变：

- 读取同一个 VQA-RAD split。
- 使用同一个 prompt version。
- 保存同一种 prediction record。
- 使用同一套 evaluator。

这样后续才能保证 controlled comparison。

## 推荐接入顺序

### 1. 先确认 mock pipeline 能跑

```powershell
.\.venv\Scripts\python.exe scripts\run_direct_vlm.py
```

这个结果不能写进论文，只证明实验管线可运行。

### 2. 先安装 PyTorch

Qwen2.5-VL 本地推理需要 PyTorch。这个依赖要根据你的机器选择 CPU 或 CUDA 版本，因此不要盲目安装。

请先到 PyTorch 官方安装页选择适合你电脑的命令：

```text
https://pytorch.org/get-started/locally/
```

如果你有 NVIDIA GPU，通常要选择与你 CUDA 版本匹配的 pip 命令。如果没有可用 GPU，可以先装 CPU 版本，但推理会很慢。

### 3. 安装 Qwen2.5-VL 依赖

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements\vlm.txt
```

你的电脑是 CPU 环境，所以真实 Qwen2.5-VL 推理会很慢。开发阶段只建议用 `Qwen/Qwen2.5-VL-3B-Instruct`，并且先跑 1 条样本 smoke test。

### 4. 运行 CPU Qwen smoke test

```powershell
.\.venv\Scripts\python.exe scripts\run_direct_vlm.py --config configs\experiments\exp01_direct_vlm_qwen_smoke.yaml
```

第一次运行会下载模型权重，耗时和磁盘占用都比较大。CPU 上生成 1 条样本也可能需要几分钟。

### 5. 后续再扩大样本数

先不要直接跑完整 test。建议顺序：

```text
1 sample -> 5 samples -> 20 samples -> 100 samples -> 451 samples
```

确认输出格式、速度和显存都正常后，再跑 full test。

## 当前 Qwen 配置

```yaml
method:
  backend: qwen2_5_vl
  model_id: Qwen/Qwen2.5-VL-3B-Instruct
  torch_dtype: float32
  device_map: cpu
```

后续如果有足够算力，可以改成：

```yaml
model_id: /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct
```

但你的 CPU 环境不建议本地运行 7B。主实验如果坚持使用 7B，更现实的方式是改用云 GPU 或 API；否则本地项目主实验应统一固定为 3B。

## 常见问题

`ModuleNotFoundError: qwen_vl_utils`

说明 VLM 依赖没装。运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements\vlm.txt
```

`PackageNotFoundError` 或 `ImportError: torch`

说明 PyTorch 没装。先按 PyTorch 官方安装页选择适合你的 CPU / CUDA 版本。

`CUDA out of memory`

说明显存不足。先用 3B、减少 `max_samples`，或改用云 GPU / API。

模型下载很慢

这是正常情况。第一次运行需要从 Hugging Face 下载模型文件。

## 结果判断

只有当 `metrics.json` 中：

```json
"is_valid_main_result": true
```

并且 backend 是真实模型时，结果才可以进入主实验表。
