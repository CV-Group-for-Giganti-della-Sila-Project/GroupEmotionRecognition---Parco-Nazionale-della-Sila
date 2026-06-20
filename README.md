# Smart Emotion Recognition System — Parco Nazionale della Sila

Academic project for the joint course **IoT Device Programming** + **Distributed Systems, Cloud and Edge Computing**  
University of Calabria (UNICAL) — A.Y. 2025/2026

---

## Project Overview

The system captures facial images from visitor groups at Parco Nazionale della Sila using a Raspberry Pi 4 edge device and classifies the group emotion in real-time through a cloud inference pipeline powered by Vision-Language Models (VLMs). Classified emotions are persisted to MySQL and surfaced through a Flutter mobile application used by park staff.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  EDGE                                                                                │
│                                                                                      │
│  ┌───────────────────────────────────────┐                                           │
│  │  Raspberry Pi 4                       │                                           │
│  │  USB camera (capture + compress)      │                                           │
│  └───────────────────────────────────────┘                                           │
└──────────────────────────────────────────────────────────────────────────────────────┘
        │  JPEG frame (HTTPS POST) + Cognito M2M token           ▲  200 OK
        ▼                                                        │
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  CLOUD (AWS eu-west-1)                                                               │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐     │
│  │  EC2 g4dn.xlarge (NVIDIA Tesla T4) — reachable via Tailscale VPN            │     │
│  │                                                                             │     │
│  │  ┌───────────────────────┐    queue (normal)    ┌──────────────────────┐    │     │
│  │  │  FastAPI (port 8080)  │─────────────────────►│  VLM Moondream       │    │     │
│  │  │                       │◄──── emotion ─────── │  (predict_endpoint_  │    │     │
│  │  │                       │                      │   one / two)         │    │     │
│  │  │                       │    queue (priority)  └──────────────────────┘    │     │
│  │  │                       │─────────────────────►  (app photo analysis)      │     │
│  │  │                       │                                                  │     │
│  │  │                       │─── forward message ─►┌──────────────────────┐    │     │
│  │  │                       │◄─── NL response ──── │  AI Agent            │    │     │
│  │  │                       │                      │  (Silvan + llama3.2) │    │     │
│  │  └──────────┬────────────┘                      └──────────┬───────────┘    │     │
│  │             │ save result                                  │ SQL queries    │     │
│  │             └──────────────────┐  ┌────────────────────────┘                │     │
│  │                                ▼  ▼                                         │     │
│  │                   ┌──────────────────────────┐                              │     │
│  │                   │  MySQL (localhost:3306)  │                              │     │
│  │                   └──────────────────────────┘                              │     │
│  │                                                                             │     │
│  │  EC2 ── pulls scripts, datasets, model artefacts ──► S3                     │     │
│  └─────────────────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────────────┘
        │  data + agent replies                         ▲  queries + chat + photos
        ▼                                               │
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  CLIENT                                                                              │
│                                                                                      │
│  ┌───────────────────────────────────────┐                                           │
│  │  Flutter Mobile App                   │                                           │
│  │  (group emotion dashboard + chat)     │                                           │
│  └───────────────────────────────────────┘                                           │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Team

| Name | Student ID | Contribution |
|------|------------|--------------|
| Gianluca Perrotta | 277091 | PaliGemma 2 — fine-tuning & evaluation, system architecture, AWS infrastructure, Backend |
| Marco Macrì | 276608 | Moondream2 — fine-tuning & evaluation, system architecture, AWS infrastructure, Frontend |
| Orazio Ruberto | 276576 | MiniCPM-V — fine-tuning & evaluation, system architecture, AWS infrastructure, Agent |
| Asrar Jemal Mohammed | 284598 | Edge integration — Raspberry Pi 5, GoPro camera, face-detection-triggered frame capture \& transmission to cloud |

---

## Repository structure

```
├── docs/                          # Project report and documentation
│   └── GER_Report_final.docx
└── root/
    ├── Backend/                   # FastAPI backend (EC2)
    │   ├── main.py
    │   ├── database.py
    │   ├── models.py
    │   ├── schemas.py
    │   ├── auth.py
    │   ├── predict_folder.py      # VLM helper module
    │   ├── requirements.txt
    │   ├── .env.example
    │   ├── routers/
    │   │   ├── app_routes.py
    │   │   └── emonodes.py
    │   ├── services/
    │   │   ├── agent.py           # AI Agent integration
    │   │   └── vlm.py             # VLM queue integration
    │   └── Silvan_Agent/          # AI Agent (text-to-SQL, Ollama)
    ├── Edge/                      # Raspberry Pi 4 capture script
    ├── Frontend/                  # Flutter mobile app
    └── VLM/
        ├── Dataset/               # FER+ JSONL builders and dataset split files
        ├── Paligemma 2/           # PaliGemma 2 3B — fine-tuning & evaluation & queue service (imported from Moondream 2)
        ├── MiniCPM-V/             # MiniCPM-V — fine-tuning & evaluation
        └── Moondream 2/           # Moondream2 — fine-tuning, evaluation & queue service
            ├── finetune.py
            ├── evaluate_VLM.py
            ├── folder_evaluation.py
            ├── model_queue_service.py
            └── sumarize_predictions.py
```
---

## Dataset — FER+

| Property | Value |
|----------|-------|
| Source | FER+ (Barsoum et al., 2016) |
| Classes | 8: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt |
| Total images | ~78,000 facial images |
| Original resolution | 48×48 px grayscale (resized to 224×224 RGB for VLM input) |
| Splits | train / validation / test |
| Distress variant | 4 classes — sadness + anger + disgust + fear + contempt → `distress` |

JSONL files in `dataset/` map each image path (as it appears on the EC2 instance at `/workspace/datasets/images/`) to its ground-truth emotion label.

---

## Results Summary

### 8-Class Evaluation (FER+ standard)

| Model | Strategy | Accuracy | Macro F1 | Invalid % |
|-------|----------|----------|----------|-----------|
| **PaliGemma 2 3B** | Base (no fine-tuning) + simple prompt | **0.691** | **0.436** | **0.00%** |
| **Moondream2** | Fine-tuning (1 epoch, 1/3 data, frozen vision encoder) | **0.822** | — | **0.00%** |
| MiniCPM-V 2.6 | QLoRA fine-tuning (1 epoch, 1/3 data) | — | — | — |

### PaliGemma 2 — 4-Class Distress Evaluation

Grouping sadness / anger / disgust / fear / contempt into a single `distress` class eliminates the severe class imbalance caused by low-support minority emotions in FER+, substantially improving macro F1.

| Accuracy | Macro F1 |
|----------|----------|
| 0.695 | 0.691 |

**Per-class F1:**

| Emotion | F1 |
|---------|----|
| happiness | 0.880 |
| neutral | 0.665 |
| distress | 0.621 |
| surprise | 0.598 |

### Moondream2 — Per-Class Accuracy (8 classes, after fine-tuning)

Config: 1 epoch, 1/3 sample fraction, learning rate 1e-5. Vision encoder frozen, text model fine-tuned.

| Emotion | Accuracy | Comment |
|---------|----------|---------|
| happiness | 91.82% | Excellent |
| neutral | 88.15% | Excellent |
| surprise | 84.89% | Good |
| anger | 81.68% | Good |
| sadness | 60.36% | Moderate |
| disgust | 33.33% | Weak |
| fear | 29.59% | Weak |
| contempt | 26.67% | Very weak |

### MiniCPM-V 2.6 — Per-Class F1 (8 classes, after QLoRA fine-tuning)

| Emotion | F1 | Comment |
|---------|----|---------|
| happiness | 0.920 | Excellent |
| neutral | 0.788 | Good |
| surprise | 0.779 | Good |
| anger | 0.702 | Good |
| sadness | 0.571 | Moderate |
| fear | 0.396 | Weak |
| disgust | 0.253 | Very weak |
| contempt | 0.125 | Very weak |

---

## API Endpoints

| Method | Path | Caller | Auth | Description |
|--------|------|--------|------|-------------|
| `POST` | `/emonodes/sendmessage` | Raspberry Pi | Cognito M2M JWT | Submit a frame for emotion classification |
| `GET` | `/app/data/getbetweendates` | App | Cognito User JWT | Query emotion percentages for a node within a time range |
| `POST` | `/app/analyzephoto` | App | Cognito User JWT | Send a photo for immediate emotion classification (high priority queue) |
| `POST` | `/app/askagent` | App | Cognito User JWT | Send a natural language question to the AI agent |

### POST `/emonodes/sendmessage`

- Body fields: `foto` (file), `node_name` (string), `num_persone` (integer), `timestamp` (unix integer)
- The frame is queued via `predict_endpoint_one` (normal priority) and processed by the VLM Moondream model
- Response: `200 OK` (empty body)

### GET `/app/data/getbetweendates`

Returns emotion percentages (integers, sum ≈ 100) for the 8 emotions: `happiness`, `neutral`, `surprise`, `sadness`, `fear`, `disgust`, `contempt`, `anger`.

### POST `/app/analyzephoto`

- Body: `{"image_base64": "..."}` — base64 encoded image (JPG, PNG, JPEG supported)
- The image is queued via `predict_endpoint_two` (high priority, processed before Raspberry frames when both queues are busy)
- Response: `{"emotion": "happiness"}`

### POST `/app/askagent`

- Body: `{"message": "What emotions were detected on Tree1?"}`
- Single message per request — no conversation history
- The AI agent (Silvan + llama3.2 via Ollama) builds a SQL query, queries MySQL, and returns a natural language response
- Response: `{"response": "Here are the emotions detected on Tree1..."}`

---

## Authentication

All endpoints require a valid AWS Cognito JWT token in the Authorization header:

```
Authorization: Bearer <token>
```

- Raspberry Pi nodes authenticate using Cognito Client Credentials (M2M flow)
- App users authenticate via Cognito User Pool (email/password)
- Both client types connect to the EC2 instance via Tailscale VPN
- FastAPI validates tokens directly — no API Gateway involved

---

## AWS Infrastructure

| Component | Specification |
|-----------|---------------|
| Instance type | `g4dn.xlarge` |
| GPU | NVIDIA Tesla T4 — 16 GB VRAM |
| Storage | EBS gp3 + S3 (scripts, datasets, model artefacts) |
| Database | MySQL (on EC2, localhost:3306) |
| Connectivity | Tailscale VPN (replaces API Gateway) |
| Auth | AWS Cognito (User Pool + M2M Client Credentials) |
| Backend | FastAPI (port 8080) |
| VLM | Moondream2 fine-tuned (two priority queues) |
| AI Agent | Silvan + llama3.2 via Ollama |
| Region | `eu-west-1` |

---

## How to Run

See the model-specific READMEs for full instructions:

- **PaliGemma 2:** [`vlm/paligemma2/README.md`](vlm/paligemma2/README.md)
- **MiniCPM-V:** `vlm/minicpmv/`
- **Moondream2:** `vlm/moondream2/`
- **Backend:** [`root/Backend/README.md`](root/Backend/README.md) — runs on port `8080`

On startup FastAPI automatically:

- Loads the VLM Moondream model into memory
- Warms up the AI agent (Ollama) with a dummy request
- Prints `VLM model service ready.` and `AI agent ready.` when both are operational

For dataset preparation: [`dataset/`](dataset/)

---

## Academic Context

- **University:** University of Calabria (UNICAL)
- **Courses:** IoT Device Programming + Distributed Systems, Cloud and Edge Computing
- **Academic Year:** 2025/2026
- **Report:** [`docs/GER_Report_final.docx`](docs/GER_Report_final.docx)
