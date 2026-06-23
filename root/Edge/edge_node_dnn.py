import cv2
import time
import os
import numpy as np
import requests
from datetime import datetime, timezone

# ============================================================
#  EDGE NODE (DNN detection + tracking) - GER, Parco della Sila
#  Hardware: Raspberry Pi 5 + GoPro (WiFi live stream)
#
#  Auth: Cognito machine-to-machine (M2M) client_credentials.
#    - The Pi requests a JWT access token from Cognito using
#      its client_id + client_secret.
#    - It sends that token as  Authorization: Bearer <token>
#      with every frame POST. The token is cached and auto-
#      refreshed before it expires.
#
#  On-edge perception (DNN + tracker) decides WHEN to send;
#  the cloud VLM does the authoritative counting + emotion.
# ============================================================

# ===== Cognito M2M auth (from colleague) =====
TOKEN_URL = "https://eu-west-1gpyk7h1zy.auth.eu-west-1.amazoncognito.com/oauth2/token"
CLIENT_ID = "1dp6ih7fbcbm9d70sivbbdkitt"
# SECURITY: do NOT hardcode the secret in the committed file.
# Set it once on the Pi:  export COGNITO_CLIENT_SECRET="....."
CLIENT_SECRET = os.environ.get("COGNITO_CLIENT_SECRET", "")
SCOPE = "default-m2m-resource-server-uyhoou/read"

# ===== Cloud endpoint (STILL NEEDED from colleague) =====
# Ask: exact URL the Pi POSTs the frame to (e.g. direct FastAPI on EC2),
# and the exact field names that endpoint expects.
API_URL = "http://100.85.74.87:8080/emonodes/sendmessage"   # <-- put EC2 Tailscale IP here
NODE_NAME = "Tree1"   # this node's name (README field: node_name); confirm with team

# ===== Settings =====
HEARTBEAT_INTERVAL = 20.0
FRAME_PROCESS_INTERVAL = 0.3
NEW_GROUP_MIN_GAP = 2.0
CONFIDENCE_THRESHOLD = 0.6
JPEG_QUALITY = 80

POSITION_THRESHOLD = 150
MIN_HITS = 3
MAX_MISSES = 20
PRESENT_GRACE = 5
EDGE_MARGIN = 10

PROTOTXT = "/home/pi/emotion/deploy.prototxt"
MODEL = "/home/pi/emotion/res10_300x300_ssd_iter_140000.caffemodel"
GOPRO_STREAM = "http://10.5.5.9:8080/live/amba.m3u8"

# ===== Token management (Cognito M2M) =====
_token_cache = {"token": None, "expires_at": 0}

def get_token():
    """Return a valid Cognito access token, refreshing if needed."""
    now = time.time()
    # reuse cached token until 60s before expiry
    if _token_cache["token"] and now < _token_cache["expires_at"] - 60:
        return _token_cache["token"]

    if not CLIENT_SECRET:
        print("  ERROR: COGNITO_CLIENT_SECRET not set in environment.")
        return None

    try:
        resp = requests.post(
            TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "client_credentials",
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "scope": SCOPE,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            _token_cache["token"] = data["access_token"]
            _token_cache["expires_at"] = now + int(data.get("expires_in", 3600))
            print("  Cognito token obtained (valid ~%ds)" % int(data.get("expires_in", 3600)))
            return _token_cache["token"]
        print(f"  Token request failed: HTTP {resp.status_code} {resp.text[:120]}")
    except requests.exceptions.RequestException as e:
        print(f"  Token request error: {e}")
    return None

# ===== Load DNN face detector =====
print("Loading DNN face detector...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
print("  Detector loaded")

# ===== Get first token before streaming =====
print("Requesting Cognito token...")
if get_token():
    print("  Auth ready\n")
else:
    print("  WARNING: could not get token; sends will fail until fixed\n")

# ===== Open GoPro stream =====
print("Opening GoPro stream...")
cap = cv2.VideoCapture(GOPRO_STREAM, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    print("ERROR: cannot open GoPro stream")
    print("  Check: GoPro WiFi on, wlan1 connected to GoPro")
    exit()
print("  GoPro stream opened\n")

print("Ready.")
print("New group / new person -> send immediately")
print(f"Same group             -> send every {HEARTBEAT_INTERVAL}s\n")

# ===== Helpers =====
def position_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def detect_faces(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        if x1 < EDGE_MARGIN or y1 < EDGE_MARGIN or \
           x2 > w - EDGE_MARGIN or y2 > h - EDGE_MARGIN:
            continue
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        faces.append(((cx, cy), (x1, y1, x2 - x1, y2 - y1)))
    return faces

def send_frame(frame, reason, group_id, people_count):
    """POST the JPEG to the cloud endpoint with a Bearer token. Expect 200 OK."""
    token = get_token()
    if not token:
        print(f"{reason:11s} | no token, cannot send")
        return False
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        print(f"{reason:11s} | encode failed")
        return False
    try:
        resp = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {token}"},
            files={"foto": ("frame.jpg", buf.tobytes(), "image/jpeg")},
            data={
                "node_name": NODE_NAME,
                "num_persone": str(people_count),
                "timestamp": str(int(time.time())),
            },
            timeout=10,
        )
        if resp.status_code == 200:
            print(f"{reason:11s} | group {group_id} | people: {people_count} | 200 OK")
            return True
        print(f"{reason:11s} | group {group_id} | people: {people_count} | HTTP {resp.status_code} {resp.text[:80]}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"{reason:11s} | network error: {e}")
        return False

# ===== State =====
tracks = {}
next_track_id = 1
current_group = set()
group_id = 0
last_send_time = 0
last_process_time = 0
stop_flag = False

# ===== Main loop =====
try:
    while not stop_flag:
        ok, frame = cap.read()
        if not ok:
            print("  Stream lost, reconnecting...")
            time.sleep(2)
            cap = cv2.VideoCapture(GOPRO_STREAM, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        send_full = frame.copy()
        frame = cv2.resize(frame, (640, 480))

        now = time.time()
        if now - last_process_time < FRAME_PROCESS_INTERVAL:
            time.sleep(0.02)
            continue
        last_process_time = now

        detections = detect_faces(frame)
        for t in tracks.values():
            t["matched"] = False

        used = set()
        for center, bbox in detections:
            best_tid, best_dist = None, POSITION_THRESHOLD
            for tid, t in tracks.items():
                if tid in used:
                    continue
                d = position_distance(center, t["center"])
                if d < best_dist:
                    best_dist, best_tid = d, tid
            if best_tid is None:
                tid = next_track_id
                next_track_id += 1
                tracks[tid] = {"center": center, "bbox": bbox, "hits": 1,
                               "misses": 0, "confirmed": False, "matched": True}
                used.add(tid)
            else:
                t = tracks[best_tid]
                t["center"], t["bbox"] = center, bbox
                t["hits"] += 1
                t["misses"] = 0
                t["matched"] = True
                used.add(best_tid)

        dead = []
        for tid, t in tracks.items():
            if not t["matched"]:
                t["misses"] += 1
                t["hits"] = 0
                if t["misses"] > MAX_MISSES:
                    dead.append(tid)
            if t["hits"] >= MIN_HITS:
                t["confirmed"] = True
        for tid in dead:
            del tracks[tid]
            current_group.discard(tid)

        present = {tid for tid, t in tracks.items()
                   if t["confirmed"] and t["misses"] <= PRESENT_GRACE}
        people_count = len(present)
        new_members = present - current_group

        reason = None
        if new_members and (now - last_send_time >= NEW_GROUP_MIN_GAP):
            group_id += 1
            current_group = set(present)
            reason = "NEW GROUP"
        elif present and (now - last_send_time >= HEARTBEAT_INTERVAL):
            current_group = set(present)
            reason = "SAME GROUP"

        if reason:
            send_frame(send_full, reason, group_id, people_count)
            last_send_time = now

except KeyboardInterrupt:
    stop_flag = True
    print("\nStopped by user.")
finally:
    cap.release()
    print("Camera released.")