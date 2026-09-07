# VQA-RAD 数据版本审计

本文档固定 MedReason-Agent 使用的 VQA-RAD 数据统计口径，避免后续实验报告里混用不同版本的数量。

## 当前决定

MedReason-Agent 使用 `scripts/import_vqa_rad.py` 下载的 OSF VQA-RAD 公共发布版本。

主要原始文件：

- `VQA_RAD Dataset Public.json`
- `VQA_RAD Image Folder.zip`

项目使用的来源 URL：

- `https://osf.io/download/6qdas/`
- `https://files.osf.io/v1/resources/89kps/providers/osfstorage/5b21453986d8510011c277bc/?zip=`

## 本地审计结果

当前本地导入结果：

```text
Raw JSON QA records: 2248
Image files in zip: 315
Images referenced by QA records: 314
Missing referenced images: 0
Official raw train pool: 1797
Official raw test split: 451
Project train split: 1527
Project validation split: 270
Project test split: 451
```

项目的 train / validation split 从 official raw train pool 中生成：

```text
validation_ratio: 0.15
seed: 42
```

官方 test split 保持不变。

## 为什么会出现其他数量

不同 VQA-RAD 论文、镜像和清洗版本会报告不同数量，因为统计口径不同。

- `2248` 是本仓库使用的 OSF 公共 JSON 中的 QA record 数量。
- `315` 是 OSF image zip 中的图像文件数量。
- `314` 是 QA records 实际引用到的图像数量。
- `1797 + 451 = 2248` 是根据 `phrase_type` 得到的官方 raw train / test split。
- `1793 + 451 = 2244` 常见于删除了少量重复或泄漏样本的清洗镜像。
- `3515` 是论文层面的 visual-question 数量，包含 free-form、rephrased、framed 等问题变体，不等于 OSF JSON 里的 QA record 数量。
- `3064` 和 `464` 不应该用于本项目主实验，除非我们明确切换到另一个数据发布版本，并记录来源和处理脚本。

## 报告写法

论文或 README 中建议使用下面这段表述：

```text
We use the official OSF VQA-RAD public release, which contains 2,248 QA records
and 315 radiology images, with 314 images referenced by QA pairs. Following the
official phrase_type split, the raw train/test split contains 1,797 train QA
records and 451 test QA records. We further split the official train portion
into train/validation with validation_ratio=0.15 and seed=42, resulting in
1,527 train, 270 validation, and 451 test records.
```

## 实验规则

本仓库中所有 VQA-RAD 实验统一使用：

```text
Smoke: 1-20 test samples
Dev: 100-200 test samples
Full: 451 test samples
```

不要把本仓库的 VQA-RAD full-test 实验报告成 464 samples。
