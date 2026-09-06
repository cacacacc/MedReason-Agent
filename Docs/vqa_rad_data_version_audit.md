# VQA-RAD Data Version Audit

This document fixes the dataset-count policy used by MedReason-Agent.

## Decision

MedReason-Agent uses the official OSF VQA-RAD public release downloaded by
`scripts/import_vqa_rad.py`.

Primary source files:

- `VQA_RAD Dataset Public.json`
- `VQA_RAD Image Folder.zip`

Project source URLs:

- `https://osf.io/download/6qdas/`
- `https://files.osf.io/v1/resources/89kps/providers/osfstorage/5b21453986d8510011c277bc/?zip=`

## Local Audit Result

The current local import produced:

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

The project train/validation split is created from the official raw train pool
with `validation_ratio=0.15` and `seed=42`. The official test split is preserved.

## Why Other Counts Appear

Different VQA-RAD papers and dataset mirrors report different counts because
they use different counting policies.

- `2248` is the number of QA records in the OSF public JSON used by this repo.
- `315` is the number of image files in the OSF image zip.
- `314` is the number of images actually referenced by QA records.
- `1797 + 451 = 2248` is the official raw train/test split derived from
  `phrase_type`.
- `1793 + 451 = 2244` appears in cleaned mirrors that remove four duplicate or
  leaked triplets.
- `3515` is a paper-level visual-question count that includes generated
  question variants such as free-form, rephrased, and framed questions. It is
  not the same as the OSF JSON QA-record count.
- `3064` and `464` should not be used in this project unless we intentionally
  switch to a different dataset release and document the source.

## Reporting Policy

Use this sentence in reports:

```text
We use the official OSF VQA-RAD public release, which contains 2,248 QA records
and 315 radiology images, with 314 images referenced by QA pairs. Following the
official phrase_type split, the raw train/test split contains 1,797 train QA
records and 451 test QA records. We further split the official train portion
into train/validation with validation_ratio=0.15 and seed=42, resulting in
1,527 train, 270 validation, and 451 test records.
```

## Experiment Rule

All VQA-RAD experiments in this repository must use:

```text
Smoke: 1-20 test samples
Dev: 100-200 test samples
Full: 451 test samples
```

Do not report VQA-RAD full-test experiments as 464 samples for this repository.
