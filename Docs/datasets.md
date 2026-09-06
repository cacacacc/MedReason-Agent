# 数据集说明

本文档记录 MedReason-Agent 当前使用的数据来源、导入方式、划分方式和注意事项。

## 当前主数据集：VQA-RAD

VQA-RAD 是医学视觉问答数据集，任务形式是：

```text
医学图像 + 自然语言问题 -> 答案
```

当前项目优先使用 VQA-RAD，因为它规模较小、适合早期 pipeline 开发，也适合做 Direct VLM、CoT、RAG、Multi-Agent 和 Supervisor 的 controlled comparison。

## 当前导入来源

当前导入脚本使用 OSF 上的 VQA-RAD 公共文件：

- `VQA_RAD Dataset Public.json`
- `VQA_RAD Image Folder.zip`

注意：公开 JSON 常见版本约为 2,248 条 QA、315 张图像，其中 test 样本可通过 `phrase_type` 中的 `test_*` 标记识别。你之前项目草案里的 3,064 / 464 数量可能来自其他整理版本或不同划分。为了保证可复现，当前先以 OSF 公共版为准，并把实际导入统计写入 `Data/Processed/vqa_rad/stats.json`。

## 本次实际导入统计

当前本地导入结果：

```text
总 QA 数量：2248
被 QA 引用的图像数量：314
缺失图片数量：0
官方 train pool：1797
官方 test：451
内部 train：1527
内部 validation：270
最终 test：451
```

答案类型：

```text
CLOSED: 1299
OPEN: 949
```

说明：VQA-RAD 论文和不同整理版本中可能出现不同统计口径。后续报告必须明确写出我们使用的是 OSF 公共 JSON 文件，并引用 `stats.json` 中的实际数量。

## 导入后目录

```text
Data/
├── Raw/
│   └── vqa_rad/
│       ├── VQA_RAD Dataset Public.json
│       └── VQA_RAD Image Folder/
└── Processed/
    └── vqa_rad/
        ├── all.jsonl
        ├── all.csv
        ├── train.jsonl
        ├── train.csv
        ├── validation.jsonl
        ├── validation.csv
        ├── test.jsonl
        ├── test.csv
        └── stats.json
```

## 划分策略

优先保留官方 test：

- `phrase_type` 以 `test` 开头的样本 -> `test`
- 其他样本 -> 官方 train pool

为了后续 prompt tuning 和 dev experiment，需要从官方 train pool 中固定抽取 validation：

- 默认 validation ratio: `0.15`
- 默认 seed: `42`
- 剩余样本作为 train

这不是重新定义官方 test，而是在官方 train 内部切出开发验证集。

## 为什么不把数据提交到 Git

`Data/Raw` 和 `Data/Processed` 默认被 `.gitignore` 忽略。原因：

- 数据文件可能较大。
- 数据许可证和再分发权限需要谨慎处理。
- GitHub repo 应提交导入脚本、配置和文档，而不是直接提交数据本体。

本地实验时重新运行导入脚本即可恢复数据。
