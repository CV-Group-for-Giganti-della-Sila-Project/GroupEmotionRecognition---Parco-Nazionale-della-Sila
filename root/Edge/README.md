# Edge Node — Raspberry Pi 5 + GoPro

Edge component of the **Smart Emotion Recognition System** (Parco Nazionale della Sila).

University of Calabria (UNICAL) — A.Y. 2025/2026

---

## Overview

This is the **edge node**: a **Raspberry Pi 5** with a **GoPro camera** that captures frames of visitor groups and transmits them to the cloud for emotion classification.

The node is deliberately lightweight and performs **no emotion inference** — the Raspberry Pi has no GPU, so all model inference happens in the cloud. The edge node is responsible only for:

- **Frame acquisition** from the GoPro WiFi live stream
- **Transmission triggering** — deciding *when* a frame is worth sending
- **Authentication** with a Cognito machine-to-machine (M2M) token
- **Secure transmission** of the frame to the cloud endpoint

A small on-device face detector + tracker is used **only** to decide when to send (a new person/group entering, or a periodic heartbeat). It produces no emotion labels and stores no identities.

---

## Hardware

| Component | Role |
|-----------|------|
| Raspberry Pi 5 | Runs the capture-and-transmit application |
| GoPro camera | Image source, via its WiFi live stream |
| microSD (Pi) | Boot + storage: OS, script, model files, libraries |
| microSD (GoPro) | Required for the GoPro to enable its video / live stream |
| USB WiFi adapter (`wlan1`) | Dedicated interface that joins the GoPro hotspot |
| Onboard WiFi (`wlan0`) | Internet connectivity for the cloud transmission |

The Pi uses **two wireless interfaces at once**: `wlan1` for the GoPro stream and `wlan0` for internet.

> **GoPro stream note:** the GoPro WiFi live stream is a compressed preview feed capped at ~**432×240 px**, below the camera's recording resolution. This is a limitation of the GoPro's wireless streaming, not of this code.

---

## Software & Dependencies

Written in **Python 3** for Raspberry Pi OS. No heavy ML frameworks — only a compact OpenCV DNN face detector that runs on the CPU.

| Library | Purpose |
|---------|---------|
| `opencv-python` (cv2) | Stream capture, frame resize, JPEG encode, DNN face detector |
| `numpy` | Array math for detection and the tracker |
| `requests` | Cognito token request + frame POST |
| `datetime` / `time` | Timestamps and loop pacing |
| `os` | Reads the Cognito client secret from the environment |

Install:

```bash
pip install opencv-python numpy requests --break-system-packages
```

### Model files (required, in `/home/pi/emotion/`)

| File | Size | Description |
|------|------|-------------|
| `deploy.prototxt` | ~28 KB | Network structure (res10 SSD) |
| `res10_300x300_ssd_iter_140000.caffemodel` | ~10 MB | Trained weights |

---

## Files

```
/home/pi/emotion/
├── edge_node_dnn.py                          # the edge application
├── deploy.prototxt                           # DNN model structure
└── res10_300x300_ssd_iter_140000.caffemodel  # DNN model weights
```

---

## Configuration

Set at the top of `edge_node_dnn.py`:

| Constant | Description |
|----------|-------------|
| `API_URL` | Cloud endpoint the frame is POSTed to |
| `NODE_NAME` | This node's name / location |
| `TOKEN_URL`, `CLIENT_ID`, `SCOPE` | Cognito M2M settings |
| `GOPRO_STREAM` | GoPro live stream URL (`http://10.5.5.9:8080/live/amba.m3u8`) |
| `HEARTBEAT_INTERVAL` | Seconds between sends while the group is unchanged (default 20) |
| `CONFIDENCE_THRESHOLD` | DNN face confidence (default 0.6; lower to 0.5 if faces are missed) |
| `JPEG_QUALITY` | JPEG quality of the transmitted frame (default 80) |

The **Cognito client secret is not stored in the file** — it is read from an environment variable:

```bash
export COGNITO_CLIENT_SECRET="<your-client-secret>"
```

---

## How It Works

1. Capture a frame from the GoPro stream.
2. Run the DNN face detector on a downscaled copy of the frame.
3. Track faces across frames (position-based) to know the current set of people.
4. Decide whether to send:
   - **New group / new person** — the set of tracked people changed → send immediately.
   - **Heartbeat** — same group still present → send every `HEARTBEAT_INTERVAL` seconds.
5. If sending: request a Cognito token (cached/auto-refreshed), JPEG-encode the original frame, and `POST` it with the metadata.

A person leaving does not trigger a send. Frames are never written to disk — each is encoded in memory and transmitted directly.

---

## Transmission Request

The node sends each selected frame as an HTTP `POST`:

**`POST <API_URL>`** — Header: `Authorization: Bearer <Cognito M2M token>`

| Field | Type | Description |
|-------|------|-------------|
| `foto` | file (JPEG) | The captured frame |
| `node_name` | string | This node's name / location |
| `num_persone` | integer | People count from the on-edge tracker |
| `timestamp` | integer | Capture time (Unix timestamp) |

A successful send returns `200 OK`.

---

## Setup & Run

### 1. Prepare the Pi (headless)

Flash Raspberry Pi OS to the microSD using the Raspberry Pi imaging tool, with **SSH enabled** and WiFi configured. Then connect:

```bash
ssh pi@<pi-ip-address>
```

### 2. Copy the files

```bash
scp edge_node_dnn.py pi@<pi-ip>:/home/pi/emotion/
scp deploy.prototxt pi@<pi-ip>:/home/pi/emotion/
scp res10_300x300_ssd_iter_140000.caffemodel pi@<pi-ip>:/home/pi/emotion/
```

### 3. Connect the GoPro

Turn on the GoPro WiFi (app/streaming mode), then bring up `wlan1` on the GoPro hotspot:

```bash
sudo nmcli connection up <gopro-connection-uuid>
iwconfig wlan1          # should show the GoPro ESSID
```

### 4. Connect to the cloud network (Tailscale)

The cloud endpoint is reachable over the project's private Tailscale network:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
ping -c 4 <ec2-tailscale-ip>   # confirm the cloud host is reachable
```

### 5. Set the secret and run

```bash
export COGNITO_CLIENT_SECRET="<your-client-secret>"
python3 /home/pi/emotion/edge_node_dnn.py
```

---

## Expected Output

```
Loading DNN face detector...
  Detector loaded
Requesting Cognito token...
  Cognito token obtained (valid ~86400s)
  Auth ready

Opening GoPro stream...
  GoPro stream opened

Ready.
New group / new person -> send immediately
Same group             -> send every 20.0s

NEW GROUP   | group 1 | people: 1 | 200 OK
SAME GROUP  | group 1 | people: 1 | 200 OK
```

A `200 OK` confirms the frame was accepted by the cloud.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `cannot open GoPro stream` | GoPro WiFi off, or `wlan1` not on the GoPro hotspot — check `iwconfig wlan1` |
| `COGNITO_CLIENT_SECRET not set` | Run the `export` command before the script |
| `network error` on send | Cloud host unreachable — check Tailscale and `ping` |
| `HTTP 401 / 403` | Token rejected — check Cognito client ID / secret / scope |
| `HTTP 422` | Field mismatch — verify the POST field names with the backend |
| Count reads low for multiple people | Lower `CONFIDENCE_THRESHOLD` to `0.5` |
| Stream stalls after a few frames | Slow uplink — reduce `JPEG_QUALITY` or frame size |

---

## Notes

- The client secret is never committed — it is read from `COGNITO_CLIENT_SECRET` at runtime.
- No frames are stored on the device; each is encoded in memory and sent directly.
