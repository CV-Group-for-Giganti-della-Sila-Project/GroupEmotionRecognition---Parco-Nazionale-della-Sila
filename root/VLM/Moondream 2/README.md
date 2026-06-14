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
**Config:** 1 Epoch, 1/3 Sample Fraction, Learning Rate 1e-5.

| Metric | Value |
|--------|-------|
| Accuracy | 82.17% |
| Samples | 3,573 |
| Invalid predictions | 0.00% |

**Per-Class Accuracy:**

| Emotion | Accuracy |
|---------|----------|
| happiness | 91.82% |
| neutral | 88.15% |
| surprise | 84.89% |
| anger | 81.68% |
| sadness | 60.36% |
| disgust | 33.33% |
| fear | 29.59% |
| contempt | 26.67% |

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
