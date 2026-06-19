import base64
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../VLM/Moondream 2'))

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated

from database import get_db
from models import Detection
from model_queue_service import predict_endpoint_one
from auth import verify_node_token

router = APIRouter(prefix="/emonodes", tags=["emonodes"])

DbDep = Annotated[AsyncSession, Depends(get_db)]


@router.post("/sendmessage", status_code=200)
async def send_message(
    db: DbDep,
    foto: UploadFile = File(...),
    node_name: str = Form(...),
    num_persone: int = Form(...),
    timestamp: int = Form(...),
    _token: dict = Depends(verify_node_token),
) -> Response:
    """Receive a frame from a Raspberry Pi node, store it, and update it with the VLM emotion result."""
    image_bytes = await foto.read()

    detection = Detection(
        node_name=node_name,
        num_persone=num_persone,
        emotion=None,
        timestamp=timestamp,
    )
    db.add(detection)
    await db.commit()
    await db.refresh(detection)

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    result = await predict_endpoint_one(image_base64)
    emotion = result.get("predicted_emotion")

    if emotion is not None:
        detection.emotion = emotion
        await db.commit()

    return Response(status_code=200)
