# Moondream 2 — Facial Emotion Recognition on FER+

**Author:** Marco Macrì (ID: 276608)  
**Model:** `vikhyatk/moondream2`  
**Dataset:** FER+ (8-class standard)

---
 
## Approach

Moondream 2 is a small, efficient Vision-Language Model designed to run on resource-constrained devices. It features a vision encoder and a language model (Phhi-2 based) that can be fine-tuned for specific tasks. 

Moondream 2 demonstrated high receptivity to task-specific supervision on the FER+ dataset. By freezing the vision encoder and training the language model, we achieved significant performance gains over the base model's zero-shot capabilities.

### Fine-Tuning Strategy

The fine-tuning process utilized the following configuration:
- **Optimization:** 8-bit Adam (`bitsandbytes`) to minimize VRAM footprint.
- **Precision:** Mixed precision (FP16).
- **Architecture:** Vision encoder frozen; text model fine-tuned.
- **Schedule:** Cosine learning rate schedule with warmup.
- **Data:** Stratified sampling from FER+ to handle class imbalance.

---

## Results

### Run 1: 8-Class Evaluation (Group Schema)

**Status:** completed  
**Model directory:** `./moondream-ferplus-emotion-full-fp16`  
**Target schema:** `group`  
**Test set:** `/workspace/datasets/jsonl/test.jsonl`  
**Created at:** `2026-06-07T11:19:31.750701+00:00`

**Config:** 1 Epoch, 1/3 Sample Fraction, Learning Rate 1e-5, `max_new_tokens=96`, `temperature=0.0`.

| Metric | Value |
|--------|-------|
| Samples | 3,573 |
| Correct predictions | 2,936 |
| Incorrect predictions | 637 |
| Accuracy | 82.17% |
| Balanced accuracy | 62.06% |
| Macro precision | 66.98% |
| Macro recall | 62.06% |
| Macro F1 | 63.24% |
| Weighted precision | 81.77% |
| Weighted recall | 82.17% |
| Weighted F1 | 81.65% |
| Micro precision | 82.17% |
| Micro recall | 82.17% |
| Micro F1 | 82.17% |
| Invalid predictions | 0 |
| Invalid prediction rate | 0.00% |

**Prediction parsing:** all 3,573 predictions were parsed as valid labels.

**Per-Class Metrics:**

| Emotion | Precision | Recall | F1 | Support |
|---------|-----------|--------|----|---------|
| neutral | 83.68% | 88.15% | 85.86% | 1,274 |
| happiness | 92.22% | 91.82% | 92.02% | 929 |
| surprise | 77.80% | 84.89% | 81.19% | 450 |
| sadness | 74.04% | 60.36% | 66.50% | 449 |
| anger | 73.88% | 81.68% | 77.58% | 322 |
| disgust | 25.00% | 33.33% | 28.57% | 21 |
| fear | 59.18% | 29.59% | 39.46% | 98 |
| contempt | 50.00% | 26.67% | 34.78% | 30 |

**Confusion Matrix:** rows are true labels, columns are predicted labels.

| True \ Predicted | neutral | happiness | surprise | sadness | anger | disgust | fear | contempt |
|------------------|---------|-----------|----------|---------|-------|---------|------|----------|
| neutral | 1123 | 26 | 28 | 71 | 15 | 4 | 4 | 3 |
| happiness | 26 | 853 | 22 | 3 | 20 | 2 | 1 | 2 |
| surprise | 28 | 14 | 382 | 2 | 14 | 0 | 9 | 1 |
| sadness | 117 | 15 | 7 | 271 | 26 | 7 | 5 | 1 |
| anger | 25 | 9 | 12 | 7 | 263 | 5 | 1 | 0 |
| disgust | 1 | 1 | 1 | 2 | 8 | 7 | 0 | 1 |
| fear | 12 | 6 | 38 | 6 | 7 | 0 | 29 | 0 |
| contempt | 10 | 1 | 1 | 4 | 3 | 3 | 0 | 8 |

### Run 2: 8-Class Evaluation (Primary Schema - Stopped Early)
**Config:** 2 Epochs (Partial), 1.0 Sample Fraction, Learning Rate 5e-6.

| Metric | Value |
|--------|-------|
| Accuracy | 73.22% |
| Samples | 743 |
| Invalid predictions | 0.00% |

---

## Qualitative Evaluation

Evaluation on the sample images in the `img/` folder yielded highly accurate results, demonstrating the model's more than acceptable capabilities on the samples.

| Image | Predicted Emotion | Image | Predicted Emotion |
|-------|-------------------|-------|-------------------|
| 1.jpg | happiness | 11.jpg | neutral |
| 2.jpg | happiness | 12.jpg | happiness |
| 3.jpg | happiness | 13.jpg | happiness |
| 4.jpg | surprise | 14.jpg | neutral |
| 5.png | neutral | 15.jpg | neutral |
| 6.jpeg | happiness | 16.jpg | happiness |
| 7.png | neutral | 17.jpg | neutral |
| 8.png | happiness | 18.jpg | neutral |
| 9.png | surprise | 19.jpg | neutral |
| 10.jpg | happiness | 20.jpeg | neutral |

**Note:** The predictions are "seem" to align well with human perception of the expressions.

---

## Confusion Matrix Analysis

- **High-Performance Classes:** Happiness and Neutral remain the strongest categories, with Moondream 2 achieving nearly 92% and 88% accuracy respectively.
- **Minority Classes:** While performance on contempt, disgust, and fear is lower than the majority classes.
- **Reliability:** The model maintains a 0% invalid prediction rate, consistently outputting valid JSON or well-formatted text that maps to the target labels.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `finetune.py` | Main script for fine-tuning Moondream 2 on FER+ data. |
| `evaluate_VLM.py` | Evaluate the fine-tuned adapter on the FER+ test set. |
| `folder_evaluation.py` | Run inference on a folder of raw images. |
| `sumarize_predictions.py` | Generate readable accuracy metrics from JSONL prediction files. |

---

## How to Run

### Prerequisites

```bash
pip install -r reqiurements.txt
```

### Fine-Tuning (Run 1)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python finetune.py \
  --train_json /workspace/datasets/jsonl/train.jsonl \
  --eval_json /workspace/datasets/jsonl/val.jsonl \
  --output_dir ./moondream-ferplus-emotion-full-fp16 \
  --training_summary_json ./moondream-ferplus-emotion-full-fp16/training_summary.json \
  --num_train_epochs 1 \
  --sample_fraction 0.3333333333 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.05 \
  --max_grad_norm 0.3 \
  --max_length 1024 \
  --torch_dtype float16 \
  --attn_implementation eager
```

### Evaluation (Run 1)

```bash
python evaluate_VLM.py \
  --test_json /workspace/datasets/jsonl/test.jsonl \
  --adapter_dir ./moondream-ferplus-emotion-full-fp16 \
  --metrics_json ./moondream-ferplus-emotion-full-fp16/test_metrics.json \
  --predictions_jsonl ./moondream-ferplus-emotion-full-fp16/test_predictions.jsonl \
  --checkpoint_jsonl ./moondream-ferplus-emotion-full-fp16/test_predictions.checkpoint.jsonl \
  --partial_metrics_json ./moondream-ferplus-emotion-full-fp16/test_metrics.partial.json \
  --checkpoint_every 25 \
  --target_schema group \
  --torch_dtype float16
```








# Model Queue Service

`model_queue_service.py` is a standalone helper module for FastAPI apps that need
to run Moondream inference on base64 images.

It loads the model once at application startup, then accepts requests from two
separate endpoint queues:

- `endpoint_one`
- `endpoint_two`

Both queues share the same model instance. A single priority worker chooses the
next image to process, so the model is not called concurrently from multiple
tasks. By default, endpoint 2 has higher priority and can process up to 3 jobs
before endpoint 1 gets a turn when both queues are busy.

## Main Functions

- `startup_model_service(config)`: loads the model and starts the queue worker.
- `shutdown_model_service()`: stops the worker during FastAPI shutdown.
- `predict_endpoint_one(image_base64)`: sends an image to endpoint 1's queue.
- `predict_endpoint_two(image_base64)`: sends an image to endpoint 2's queue.
- `get_model_service().queue_sizes()`: returns the current queue sizes.

## FastAPI Example

```python
from fastapi import FastAPI
from pydantic import BaseModel

from model_queue_service import (
    ModelServiceConfig,
    get_model_service,
    predict_endpoint_one,
    predict_endpoint_two,
    shutdown_model_service,
    startup_model_service,
)


class ImageRequest(BaseModel):
    image_base64: str


app = FastAPI()


@app.on_event("startup")
async def startup():
    await startup_model_service(
        ModelServiceConfig(
            model_dir="./moondream-ferplus-emotion-primary-full",
            base_model_dir="./moondream2-base",
            endpoint_one_schema="primary",
            endpoint_two_schema="primary",
            endpoint_two_priority_weight=3,
            device="auto",
            torch_dtype="auto",
        )
    )


@app.on_event("shutdown")
async def shutdown():
    await shutdown_model_service()


@app.post("/endpoint-one")
async def endpoint_one(payload: ImageRequest):
    return await predict_endpoint_one(payload.image_base64)


@app.post("/endpoint-two")
async def endpoint_two(payload: ImageRequest):
    return await predict_endpoint_two(payload.image_base64)


@app.get("/queues")
async def queues():
    return get_model_service().queue_sizes()
```

## Priority Behavior

The priority is controlled by `endpoint_two_priority_weight`.

With the default value:

```python
endpoint_two_priority_weight=3
```

and both queues full, processing is roughly:

```text
endpoint_two
endpoint_two
endpoint_two
endpoint_one
endpoint_two
endpoint_two
endpoint_two
endpoint_one
```

Increase the value if endpoint 2 should be favored more strongly. Set it to `1`
for near round-robin behavior.

## Expected Request Body

Each endpoint expects a JSON body containing a base64 image:

```json
{
  "image_base64": "..."
}
```

The image can be plain base64 or a data URL such as:

```text
data:image/png;base64,...
```

## Response Shape

Each prediction returns a dictionary like:

```json
{
  "predicted_emotion": "happiness",
  "parse_status": "valid_label",
  "raw_model_output": "{\"primary_emotion\":\"happiness\"}",
  "parsed_model_output": {"primary_emotion": "happiness"},
  "model_source": "./moondream-ferplus-emotion-primary-full",
  "queue": "endpoint_two",
  "queue_wait_seconds": 0.003214,
  "processing_seconds": 1.48291
}
```
