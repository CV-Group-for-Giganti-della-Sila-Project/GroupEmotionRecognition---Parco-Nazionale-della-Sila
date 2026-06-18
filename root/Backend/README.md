# Group Emotion Recognition — Backend API

FastAPI backend for the Group Emotion Recognition system. Receives image frames from Raspberry Pi nodes, classifies group emotions via an external VLM server, persists results in MySQL, and serves aggregated data to the mobile/web application.

## Endpoints

| Method | Path | Caller | Auth | Description |
|--------|------|--------|------|-------------|
| `POST` | `/emonodes/sendmessage` | Raspberry Pi | Cognito M2M JWT | Submit a frame for emotion classification |
| `GET` | `/app/data/getbetweendates` | App | Cognito User JWT | Query emotion percentages for a node within a time range |
| `POST` | `/app/askagent` | App | Cognito User JWT | Send a message to the chatbot |

---

### POST `/emonodes/sendmessage`

**Content-Type:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `foto` | file | Captured image from the node |
| `node_name` | string | Identifier of the sending node (e.g. `raspi-01`) |
| `num_persone` | integer | Number of people detected in the frame |
| `timestamp` | integer | Unix timestamp of the capture |

**Response:** `200 OK` (empty body)

The row is written immediately with `emotion = NULL`. The VLM is then called asynchronously; if it succeeds the row is updated with the predicted emotion. If the VLM is unreachable the row keeps `emotion = NULL` and the endpoint still returns `200 OK`.

---

### GET `/app/data/getbetweendates`

**Query parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `start` | integer | Start unix timestamp (inclusive) |
| `end` | integer | End unix timestamp (inclusive) |
| `nodename` | string | Node identifier |

**Response:**

```json
{
  "raspi-01": {
    "happiness": 40,
    "neutral": 20,
    "surprise": 10,
    "sadness": 5,
    "fear": 5,
    "disgust": 5,
    "contempt": 3,
    "anger": 2
  }
}
```

Returns emotion percentages (integers, sum ≈ 100) for the 8 emotions: `happiness`, `neutral`, `surprise`, `sadness`, `fear`, `disgust`, `contempt`, `anger`.

---

### POST `/app/askagent`

**Content-Type:** `application/json`

```json
{
  "message": "What emotions were detected today?",
  "foto": "optional — base64 string or https:// URL"
}
```

No conversation history — each request is a single independent message.

**Response:**

```json
{
  "response": "Based on the detected emotions..."
}
```

Returns `500` if the language model call fails.

---

## Authentication

All endpoints require a valid AWS Cognito JWT token in the Authorization header:

```
Authorization: Bearer <token>
```

- Raspberry Pi nodes authenticate using Cognito Client Credentials (M2M flow) — token obtained via OAuth2 `client_credentials` grant
- App users authenticate via Cognito User Pool — token obtained after login in the mobile app

---

## Setup

### Prerequisites

- Python 3.11+
- MySQL running on `localhost:3306` with the `emotion_db` database and `detections` table already provisioned

### Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Then install dependencies:

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `DB_HOST` | MySQL host |
| `DB_PORT` | MySQL port |
| `DB_NAME` | Database name |
| `DB_USER` | Database user |
| `DB_PASSWORD` | Database password |
| `VLM_URL` | Full URL of the VLM inference endpoint |
| `AGENT_URL` | Full URL of the AI agent endpoint |
| `COGNITO_REGION` | AWS region (eu-west-1) |
| `COGNITO_USER_POOL_ID` | Cognito User Pool ID |
| `COGNITO_CLIENT_ID` | Cognito App Client ID |

### Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

For development with auto-reload:

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

Interactive API documentation is available at `http://localhost:8080/docs` once the server is running.
