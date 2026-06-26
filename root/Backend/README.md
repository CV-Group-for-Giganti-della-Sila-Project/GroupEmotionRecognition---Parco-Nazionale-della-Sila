# Group Emotion Recognition — Backend API

FastAPI backend for the Group Emotion Recognition system deployed on AWS EC2 `g4dn.xlarge` (NVIDIA Tesla T4, 16 GB VRAM). It is the central hub of the system: receives JPEG frames from Raspberry Pi 5 edge nodes over Tailscale VPN, classifies group emotions via a fine-tuned Moondream2 VLM loaded directly in-process, persists every detection to MySQL, serves aggregated emotion statistics and natural-language queries to the Flutter mobile app, and forwards each classified detection to a digital twin broker via MQTT.

All traffic is authenticated via AWS Cognito — machine-to-machine (M2M) JWT for edge nodes, User Pool JWT for app users. The backend never exposes a public endpoint: all clients reach it exclusively through the private Tailscale mesh network.

---

**Key design decisions:**
- **No API Gateway** — replaced by Tailscale VPN to stay within the $150 project budget (~$24/month saved)
- **No DynamoDB** — replaced by MySQL on the same EC2 host (zero extra cost, better fit for tabular relational data)
- **Direct in-process integration** — both the VLM queue service (Marco) and the Silvan agent (Orazio) run inside the FastAPI process, eliminating HTTP round-trips and extra process management
- **Full async/await stack** — GPU inference is blocking; without async a single VLM call would stall the entire server for 1–2 s

---

## Endpoints

| Method | Path | Caller | Auth | Description |
|--------|------|--------|------|-------------|
| `POST` | `/emonodes/sendmessage` | Raspberry Pi | Cognito M2M JWT | Submit a frame → VLM → MySQL → MQTT |
| `GET` | `/app/data/getbetweendates` | App | Cognito User JWT | Emotion percentages for a node and time range |
| `POST` | `/app/analyzephoto` | App | Cognito User JWT | On-demand photo classification (high-priority queue) |
| `POST` | `/app/askagent` | App | Cognito User JWT | Natural-language query → Silvan agent → NL answer |

---

### POST `/emonodes/sendmessage`

Called by the Raspberry Pi nodes every 20 seconds (or immediately on new-person detection). Uses a **two-phase write** pattern to guarantee no frame is ever lost: the row is inserted with `emotion = NULL` before VLM inference starts, then updated in place once the prediction arrives.

**Content-Type:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `foto` | file | Captured frame (JPG, PNG, JPEG) |
| `node_name` | string | Node identifier (e.g. `Tree1`) |
| `num_persone` | integer | Number of people detected on-edge |
| `timestamp` | integer | Unix timestamp of the capture |

**Internal flow:**
1. Token validation — M2M JWT verified locally via cached JWKS keys (no Cognito round-trip)
2. DB insert — row saved immediately with `emotion = NULL`
3. Base64 encoding — raw image bytes encoded for the VLM queue service
4. VLM inference — image submitted to `endpoint_one` (normal-priority queue), `await`ed
5. DB update — emotion label written to the row
6. MQTT publish — detection forwarded to the digital twin broker (fire-and-forget, wrapped in `try/except`)

**Response:** `200 OK` (empty body). MQTT failures never affect this response.

---

### GET `/app/data/getbetweendates`

Called by the Flutter statistics dashboard to populate per-node emotion charts.

**Query parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `start` | integer | Start unix timestamp (inclusive) |
| `end` | integer | End unix timestamp (inclusive) |
| `nodename` | string | Node identifier |

**Internal flow:**
1. Token validation — User Pool JWT verified online via Cognito `GetUser`
2. SQL query — `SELECT` on `detections` filtered by `node_name` and timestamp range
3. Aggregation — occurrences of each of the 8 emotions counted
4. Percentage computation — integer truncation (sum ≈ 100)

**Response:**
```json
{
  "Tree1": {
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

---

### POST `/app/analyzephoto`

Called by the Flutter app when the user submits a photo for immediate classification. Routed to the **high-priority** VLM queue (weight = 3), so it is processed before any pending Raspberry Pi frames — ensuring the interactive user experience stays responsive even under continuous edge load.

**Content-Type:** `application/json`

```json
{
  "image_base64": "<base64-encoded image>"
}
```

Accepts JPG, PNG, and JPEG. The VLM decodes the image from base64 regardless of original format.

**Response:**
```json
{
  "emotion": "happiness"
}
```

---

### POST `/app/askagent`

Called by the Flutter chat interface. Forwards the user's message to Orazio's Silvan agent, which runs a deterministic three-stage pipeline: JSON query planning via llama3.2, parameterized SQL compilation (with regex fallbacks), and natural-language answer synthesis from the fetched rows — designed to prevent hallucinations by forcing the model to rely exclusively on database results.

No conversation history is maintained — each request is fully independent.

**Content-Type:** `application/json`

```json
{
  "message": "What emotions were detected on Tree1?",
  "foto": null
}
```

Because `handle_request` is synchronous and blocking (MySQL connection + two sequential Ollama calls), it is offloaded via `asyncio.to_thread` to keep the FastAPI event loop free.

**Response:**
```json
{
  "response": "Here are the emotions detected on Tree1: Happiness 18 people (avg 3.67 per observation)..."
}
```

Returns `500` if the agent call fails.

---

## Authentication

All endpoints require a valid AWS Cognito JWT in the `Authorization` header:
Authorization: Bearer <token>

Two separate validators are implemented in `auth.py`:

- **`verify_node_token`** — validates M2M Client Credentials tokens for Raspberry Pi nodes. Verification is done **locally** using RSA public keys fetched once from the Cognito JWKS endpoint and cached in memory. Checks RS256 signature, issuer claim, and client ID. No network round-trip on every request — critical because nodes transmit every 20 seconds.

- **`verify_app_token`** — validates User Pool tokens for Flutter app users. Calls Cognito's `GetUser` API **online**, confirming both validity and revocation status. Suitable for human sessions that may be terminated at any time.

---

## VLM Priority Queue

Moondream2 is loaded once into GPU memory at startup and shared across all requests through a two-queue priority system implemented in Marco's `model_queue_service.py`. A single background worker processes one image at a time, preventing concurrent GPU access.

| Queue | Endpoint | Priority | Used by |
|-------|----------|----------|---------|
| `endpoint_one` | normal | 1× | `/emonodes/sendmessage` — continuous Raspberry Pi stream |
| `endpoint_two` | high | 3× | `/app/analyzephoto` — user-triggered on-demand analysis |

With `endpoint_two_priority_weight = 3`, when both queues have pending work the app request is served three times before a Raspberry Pi frame is processed.

---

## AI Agent Integration

The Silvan agent (`Silvan_Agent/silvan_agent.py`, by Orazio Ruberto) is integrated as a **direct Python import** — no HTTP proxy, no separate process. The `Silvan_Agent/` directory is inserted onto `sys.path` at startup and `handle_request` is imported directly into the FastAPI process.

Agent pipeline:
1. **Query planning** — llama3.2 converts the natural-language question into a structured JSON query plan
2. **SQL compilation** — the plan is compiled into a safe, parameterized SQL `SELECT` statement (with regex fallbacks to bypass LLM errors)
3. **DB execution** — query executed against the `detections` table
4. **Answer synthesis** — llama3.2 generates a natural-language answer from the result rows, forced to use only fetched data

---

## MQTT Digital Twin Integration

After each successful VLM classification, the detection is published to an MQTT broker in **Eclipse Ditto** format:

```json
{
  "thingId": "Tree1",
  "features": {
    "sensors": {
      "properties": {
        "numPeople": 3,
        "gEmotion": "happiness",
        "timestamp": 1782401888
      }
    }
  }
}
```

The publisher (`services/MQTTService.py`, by Marco Macrì) maintains a persistent background connection via `paho-mqtt`'s `loop_start()`. Broker host, port, and topic are configured at startup from `.env` via `publisher.set_info()`, so the ngrok tunnel address can be updated without any code changes. The publish call is wrapped in `try/except` — broker failures are logged but never propagate to the HTTP response.

---

## Database

Single MySQL table `detections` on `localhost:3306`:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INT, PK, autoincrement | Unique detection identifier |
| `node_name` | VARCHAR(50) | Node identifier (e.g. `Tree1`) |
| `num_persone` | INT | People count from on-edge detector |
| `emotion` | VARCHAR(50), **nullable** | Predicted emotion — `NULL` until VLM responds |
| `timestamp` | BIGINT | Unix timestamp (Europe/Rome) |

The `emotion` column is nullable by design — rows are inserted immediately on frame arrival, guaranteeing no detection is lost even if VLM inference fails or times out.

---

## Startup Sequence

On `uvicorn main:app` three initialization steps run sequentially before traffic is accepted:

1. **VLM startup** — `startup_model_service` loads Moondream2 onto the GPU and starts the priority-queue worker → prints `VLM model service ready.`
2. **Agent warmup** — a dummy question is sent via `asyncio.to_thread`, forcing Ollama to load `llama3.2` into memory → prints `AI agent ready.`
3. **MQTT configuration** — broker host, port and topic read from `.env` via `publisher.set_info()` → prints `MQTT publisher configured.`

On shutdown, `shutdown_model_service` releases the GPU and stops the queue worker cleanly.

---

## File Structure
root/Backend/

├── main.py                    # Entry point, startup/shutdown lifecycle

├── database.py                # Async SQLAlchemy engine and session factory

├── models.py                  # ORM model for the detections table

├── schemas.py                 # Pydantic request/response schemas

├── auth.py                    # Cognito JWT validation (two validators)

├── predict_folder.py          # VLM helper module (direct import from Marco)

├── requirements.txt

├── .env.example

├── routers/

│   ├── app_routes.py          # /app/* endpoints

│   └── emonodes.py            # /emonodes/* endpoints

├── services/

│   ├── agent.py               # Silvan agent integration

│   ├── vlm.py                 # Legacy HTTP VLM proxy (no longer active)

│   └── MQTTService.py         # MQTT publisher service (by Marco Macrì)

└── Silvan_Agent/

└── silvan_agent.py        # AI agent (by Orazio Ruberto)

---

## Setup

### Prerequisites

- Python 3.11+
- MySQL running on `localhost:3306` with the `emotion_db` database and `detections` table already provisioned
- Ollama running locally with `llama3.2` pulled
- Moondream2 fine-tuned checkpoint available at the path set in `VLM_MODEL_DIR`

### Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

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
| `VLM_MODEL_DIR` | Path to the fine-tuned Moondream2 checkpoint |
| `VLM_BASE_MODEL_DIR` | Path to the base Moondream2 weights (fallback) |
| `COGNITO_REGION` | AWS region (e.g. `eu-west-1`) |
| `COGNITO_USER_POOL_ID` | Cognito User Pool ID |
| `COGNITO_CLIENT_ID` | Cognito App Client ID (M2M) |
| `OLLAMA_MODEL` | Ollama model name (e.g. `llama3.2`) |
| `MQTT_HOST` | MQTT broker host |
| `MQTT_PORT` | MQTT broker port |
| `MQTT_TOPIC` | MQTT topic for digital twin publications |

### Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

For development with auto-reload:

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

Interactive API documentation is available at `http://localhost:8080/docs` once the server is running.