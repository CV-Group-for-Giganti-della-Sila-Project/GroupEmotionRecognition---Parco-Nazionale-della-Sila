# MiniCPM-V 4.6 — Facial Emotion Recognition on FER+

**Author:** Orazio Mattia Ruberto  
**Model:** `openbmb/MiniCPM-V-4.6`  
**Dataset:** FER+ (8-class standard)  
**Target schema:** `group`

---

## Approach

MiniCPM-V 4.6 was fine-tuned for facial emotion recognition on FER+ using a strict machine-readable output format. Each image is presented as a multimodal chat example, and the assistant is trained to return JSON containing both a per-subject emotion and an aggregate `group_emotion`.

The default output schema is:

```json
{"subjects":[{"id":1,"emotion":"<emotion>"}],"group_emotion":"<emotion>"}
```

Although FER+ is mostly a single-face dataset, the `group` schema makes the model output compatible with downstream multi-person or group-analysis pipelines. For a single visible subject, `group_emotion` is the same as the detected subject emotion.

The supported FER+ emotions are:

```text
neutral, happiness, surprise, sadness, anger, disgust, fear, contempt
```

---

## Fine-Tuning Strategy

The MiniCPM-V training workflow uses QLoRA to adapt the model efficiently on limited GPU memory while keeping the base model quantized.

Key configuration details:

- **Base model:** `openbmb/MiniCPM-V-4.6`
- **Optimization:** QLoRA with 4-bit NF4 quantization through `bitsandbytes`
- **Adapter targets:** language-model attention and MLP projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Vision modules:** avoided when resolving LoRA target layers
- **Precision:** configured for T4-friendly training; the final evaluated run uses an `fp32` adapter directory
- **Data handling:** JSON or JSONL input, with labels read from `output.primary_emotion`; if missing, the parent image folder is used as fallback
- **Sampling:** stratified class-wise sampling is supported so reduced runs preserve minority classes
- **Stability:** NaN/Inf loss guardrails, learning-rate reduction on loss explosion, gradient clipping, early stopping, and best-checkpoint reload
- **Image settings used at evaluation:** `downsample_mode=16x`, `max_slice_nums=2`
- **Generation settings used at evaluation:** `temperature=0.0`, `max_new_tokens=96`

---

## Results

### 8-Class Evaluation — FER+ Standard Labels

**Evaluation timestamp:** 2026-06-02 15:00:33 UTC  
**Samples:** 3,573  
**Correct predictions:** 2,672  
**Incorrect predictions:** 901  
**Invalid predictions:** 0 / 3,573

| Metric | Value |
|--------|------:|
| Accuracy | 74.78% |
| Balanced accuracy | 63.83% |
| Macro precision | 55.11% |
| Macro recall | 63.83% |
| Macro F1 | 56.66% |
| Weighted precision | 79.36% |
| Weighted recall | 74.78% |
| Weighted F1 | 76.66% |
| Micro F1 | 74.78% |
| Invalid prediction rate | 0.00% |

### Per-Class Performance

| Emotion | Precision | Recall | F1 | Support |
|---------|----------:|-------:|---:|--------:|
| neutral | 85.70% | 72.92% | 78.80% | 1,274 |
| happiness | 93.54% | 90.42% | 91.95% | 929 |
| surprise | 79.45% | 76.44% | 77.92% | 450 |
| sadness | 56.98% | 57.24% | 57.11% | 449 |
| anger | 69.85% | 70.50% | 70.17% | 322 |
| disgust | 16.67% | 52.38% | 25.29% | 21 |
| fear | 31.18% | 54.08% | 39.55% | 98 |
| contempt | 7.53% | 36.67% | 12.50% | 30 |

---

## Confusion Matrix

Rows are true labels and columns are predicted labels.

| True \\ Predicted | neutral | happiness | surprise | sadness | anger | disgust | fear | contempt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neutral | 929 | 24 | 20 | 155 | 23 | 15 | 24 | 84 |
| happiness | 14 | 840 | 29 | 9 | 20 | 3 | 6 | 8 |
| surprise | 22 | 11 | 344 | 2 | 17 | 0 | 50 | 4 |
| sadness | 93 | 17 | 5 | 257 | 24 | 14 | 24 | 15 |
| anger | 18 | 4 | 8 | 15 | 227 | 17 | 12 | 21 |
| disgust | 0 | 1 | 2 | 2 | 4 | 11 | 0 | 1 |
| fear | 3 | 1 | 24 | 6 | 8 | 1 | 53 | 2 |
| contempt | 5 | 0 | 1 | 5 | 2 | 5 | 1 | 11 |

---

## Confusion Matrix Analysis

- **Strongest class:** Happiness is the most reliable class, with 840 correct predictions out of 929 samples and an F1 score of 91.95%.
- **Strong high-support classes:** Neutral and surprise also perform well, with F1 scores of 78.80% and 77.92% respectively.
- **Moderate negative-emotion performance:** Anger reaches a 70.17% F1 score, while sadness is lower at 57.11%.
- **Minority-class difficulty:** Disgust, fear, and contempt have much lower F1 scores. This is partly due to small support counts and overlap among subtle facial action patterns.
- **Neutral vs. sadness:** The largest neutral error is neutral being predicted as sadness 155 times. Sadness is also confused with neutral 93 times.
- **Surprise vs. fear:** Surprise is confused with fear 50 times, and fear is confused with surprise 24 times, suggesting that open-mouth or widened-eye expressions can overlap.
- **Contempt precision issue:** Contempt has very low precision because many non-contempt images, especially neutral images, are predicted as contempt.
- **Parsing reliability:** The parser resolved all 3,573 model outputs as valid labels, producing a 0.00% invalid prediction rate.

---

## Qualitative Evaluation

No separate folder-level qualitative image predictions were included in the provided MiniCPM-V files. Based on the test metrics, the model is most dependable on visually distinctive or high-support expressions such as happiness, neutral, surprise, and anger.

The main qualitative weakness is fine-grained negative-emotion separation. Disgust, contempt, fear, and sadness share subtle visual cues, and the class distribution is highly imbalanced. In practice, this means the model is likely suitable for broad emotion recognition, but caution is needed if the downstream application requires precise separation among rare negative emotions.

---

## Scripts and Files

| File | Purpose |
|------|---------|
| `finetune_minicpmv_emotion_qlora.py` | Main QLoRA fine-tuning script for MiniCPM-V on FERPlus-style emotion data. |
| `evaluate_minicpmv_emotion.py` | Evaluation script referenced by the training guide for testing the trained adapter. |
| `TRAINING_AND_EVALUATION.md` | Full setup, training, evaluation, and troubleshooting guide. |
| `test_metrics.json` | Final evaluation metrics for the MiniCPM-V adapter. |
| `moondream.md` / `paligemma.md` | Formatting/style references for this report. |

---

## How to Run

### Prerequisites

Create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch with CUDA support:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
```

Install the remaining dependencies:

```bash
pip install -U \
  "transformers>=4.57.0" \
  accelerate \
  peft \
  bitsandbytes \
  pillow \
  sentencepiece \
  protobuf \
  safetensors \
  huggingface_hub
```

Check that CUDA is visible:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output on the target machine should include:

```text
True
Tesla T4
```

---

### Fine-Tuning

Example command:

```bash
python finetune_minicpmv_emotion_qlora.py \
  --model_id openbmb/MiniCPM-V-4.6 \
  --train_json /workspace/datasets/jsonl/train.jsonl \
  --eval_json /workspace/datasets/jsonl/val.jsonl \
  --output_dir /workspace/orazio/minicpmv-ferplus-emotion-full-fp32 \
  --training_summary_json /workspace/orazio/minicpmv-ferplus-emotion-full-fp32/training_summary.json \
  --target_schema group \
  --downsample_mode 16x \
  --max_slice_nums 2
```

For lower-memory or diagnostic runs, reduce the sample fraction:

```bash
--sample_fraction 0.3333333333
```

For the original simple output schema instead of the group schema:

```bash
--target_schema primary
```

---

### Evaluation

Command matching the final reported evaluation setup:

```bash
python evaluate_minicpmv_emotion.py \
  --test_json /workspace/datasets/jsonl/test.jsonl \
  --adapter_dir /workspace/orazio/minicpmv-ferplus-emotion-full-fp32/checkpoint-1383 \
  --metrics_json /workspace/orazio/minicpmv-ferplus-emotion-full-fp32/test_metrics.json \
  --predictions_jsonl /workspace/orazio/minicpmv-ferplus-emotion-full-fp32/test_predictions.jsonl \
  --checkpoint_jsonl /workspace/orazio/minicpmv-ferplus-emotion-full-fp32/test_predictions.checkpoint.jsonl \
  --partial_metrics_json /workspace/orazio/minicpmv-ferplus-emotion-full-fp32/test_metrics.partial.json \
  --checkpoint_every 25 \
  --target_schema group \
  --downsample_mode 16x \
  --max_slice_nums 2 \
  --max_new_tokens 96 \
  --temperature 0.0
```

---

## Final Notes

MiniCPM-V 4.6 reaches a strong overall accuracy of 74.78% on the FER+ 8-class test set while maintaining a fully parseable output format. Compared with a plain classification-only setup, the JSON group schema is more useful for downstream applications because it can represent subject-level predictions and a global group emotion in one response.

The main limitation is the same one seen in many FER+ experiments: minority negative emotions are difficult to separate. Future improvements should focus on class balancing, additional data for disgust/fear/contempt, targeted prompt experiments, and possibly a 4-class grouped evaluation where low-support negative emotions are merged into a broader distress category.
