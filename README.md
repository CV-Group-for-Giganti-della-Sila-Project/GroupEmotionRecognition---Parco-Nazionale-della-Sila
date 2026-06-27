# Smart Emotion Recognition System — Parco Nazionale della Sila

Academic project for the joint course **IoT Device Programming** + **Distributed Systems, Cloud and Edge Computing**  
University of Calabria (UNICAL) — A.Y. 2025/2026

---

## Project Overview

The system captures facial images from visitor groups at Parco Nazionale della Sila using a Raspberry Pi 5 edge device and classifies the group emotion in real-time through a cloud inference pipeline powered by Vision-Language Models (VLMs). Classified emotions are persisted to MySQL and surfaced through a Flutter mobile application used by park staff.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  TAILSCALE VPN — private mesh network                                                   │
│                                                                                         │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │  EDGE                                                                            │   │
│  │                                                                                  │   │
│  │  ┌───────────────────────────────────────┐                                       │   │
│  │  │  Raspberry Pi 5                       │                                       │   │
│  │  │  GoPro camera    (capture + compress) │                                       │   │ 
│  │  │  OpenCV DNN SSD                       │                                       │   │
│  │  └───────────────────────────────────────┘                                       │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│          │  JPEG frame (HTTPS POST) + Cognito M2M token           ▲  200 OK             │
│          ▼                                                        │                     │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │  CLOUD (AWS eu-west-1)                                                           │   │
│  │                                                                                  │   │
│  │  ┌───────────────────────────────────────────────────────────────────────────┐   │   │
│  │  │  EC2 g4dn.xlarge (NVIDIA Tesla T4) — reachable via Tailscale VPN          │   │   │
│  │  │                                                                           │   │   │
│  │  │  ┌───────────────────────┐    queue (normal)    ┌──────────────────────┐  │   │   │
│  │  │  │  FastAPI (port 8080)  │─────────────────────►│  VLM Moondream       │  │   │   │
│  │  │  │                       │◄──── emotion ─────── │  (predict_endpoint_  │  │   │   │
│  │  │  │                       │                      │   one / two)         │  │   │   │
│  │  │  │                       │    queue (priority)  └──────────────────────┘  │   │   │
│  │  │  │                       │─────────────────────►  (app photo analysis)    │   │   │
│  │  │  │                       │                                                │   │   │
│  │  │  │                       │─── forward message ─►┌──────────────────────┐  │   │   │
│  │  │  │                       │◄─── NL response ──── │  AI Agent            │  │   │   │
│  │  │  │                       │                      │  (Silvan + llama3.2) │  │   │   │
│  │  │  └──────────┬────────────┘                      └──────────┬───────────┘  │   │   │
│  │  │             │ save result                                  │ SQL queries  │   │   │
│  │  │             └──────────────────┐  ┌────────────────────────┘              │   │   │
│  │  │                                ▼  ▼                                       │   │   │
│  │  │                   ┌──────────────────────────┐                            │   │   │
│  │  │                   │  MySQL (localhost:3306)  │                            │   │   │
│  │  │                   └──────────────────────────┘                            │   │   │
│  │  │                                                                           │   │   │
│  │  │  EC2 ── pulls scripts, datasets, model artefacts ──► S3                   │   │   │
│  │  └───────────────────────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│          │  data + agent replies                         ▲  queries + chat + photos     │
│          ▼                                               │  + Cognito User JWT          │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │  CLIENT                                                                          │   │
│  │                                                                                  │   │
│  │  ┌───────────────────────────────────────┐                                       │   │
│  │  │  Flutter Mobile App                   │                                       │   │
│  │  │  (group emotion dashboard + chat)     │                                       │   │
│  │  └───────────────────────────────────────┘                                       │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
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
├── docs/                          # Project report and documentation + .html
│   ├── GER_Report_final.docx
|   └── Architecture diagram/
|       ├── architecture_diagram.html
└── root/
    ├── Backend/                   # FastAPI backend (EC2)
    ├── Edge/                      # Raspberry Pi 5 capture script
    ├── EdgeCloudSim/  
    ├── MobileApp/                  
    └── VLM/
        ├── Dataset/               # FER+ JSONL builders and dataset split files
        ├── Paligemma 2/           # PaliGemma 2 3B — fine-tuning & evaluation + project scripts
        ├── MiniCPM-V/             # MiniCPM-V — fine-tuning & evaluation + project scripts
        └── Moondream 2/           # Moondream2 — fine-tuning & evaluation + project scripts
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
| **PaliGemma 2 3B** | Base (no fine-tuning) + simple prompt | 0.691 | 0.436 | 0.00% |
| **Moondream2** | QLoRA fine-tuning (1 epoch, 1/3 data, frozen vision encoder) | **0.822** | 0.632 | 0.00% |
| **MiniCPM-V 4.6** | QLoRA fine-tuning (1 epoch, 1/3 data, FP32 adapter) | 0.748 | 0.567 | 0.00% |

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

### Moondream2 — Per-Class F1 (8 classes, after fine-tuning)

Config: 1 epoch, 1/3 sample fraction, learning rate 1e-5. Vision encoder frozen, text model fine-tuned.

| Emotion | F1 | Comment |
|---------|-----|---------|
| happiness | 92.02% | Excellent |
| neutral | 85.86% | Excellent |
| surprise | 81.19% | Good |
| anger | 77.58% | Good |
| sadness | 66.50% | Moderate |
| disgust | 28.57% | Weak |
| fear | 39.46% | Weak |
| contempt | 34.78% | Very weak |

### MiniCPM-V 4.6 — Per-Class F1 (8 classes, after QLoRA fine-tuning)

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
| Backend | FastAPI (on EC2, localhost:8080) |
| VLM | Moondream2 fine-tuned (two priority queues) |
| AI Agent | Silvan + llama3.2 via Ollama |
| Region | `eu-west-1` |

---

## How to Run

See the model-specific READMEs for full instructions:

- **PaliGemma 2:** [`root/VLM/Paligemma 2/README.md`](root/VLM/Paligemma%202/README.md)
- **MiniCPM-V:** [`root/VLM/MiniCPM-V/MiniCPM-V_readme.md`](root/VLM/MiniCPM-V/MiniCPM-V_readme.md)
- **Moondream2:** [`root/VLM/Moondream 2/README.md`](root/VLM/Moondream%202/README.md)
- **Backend:** [`root/Backend/README.md`](root/Backend/README.md) — runs on port `8080`

On startup FastAPI automatically:

- Loads the VLM Moondream2 model into GPU memory and starts the priority-queue worker
- Warms up the AI agent (Ollama) with a dummy request to pre-load llama3.2
- Configures the MQTT publisher with broker host, port and topic from `.env`
- Prints `VLM model service ready.`, `AI agent ready.` and `MQTT publisher configured.` when all three are operational

For dataset preparation: [`root/VLM/Dataset/`](root/VLM/Dataset/)

---

## Academic Context

- **University:** University of Calabria (UNICAL)
- **Courses:** IoT Device Programming + Distributed Systems, Cloud and Edge Computing
- **Academic Year:** 2025/2026
- **Report:** [`docs/GER_Report_final.docx`](docs/GER_Report_final.docx)