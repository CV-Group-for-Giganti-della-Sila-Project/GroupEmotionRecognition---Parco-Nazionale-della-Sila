import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../VLM/Moondream 2'))

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Annotated

from database import get_db
from models import Detection
from schemas import AskAgentRequest, AskAgentResponse, AnalyzePhotoRequest, AnalyzePhotoResponse
from services.agent import run_agent
from model_queue_service import predict_endpoint_two
from auth import verify_app_token

router = APIRouter(prefix="/app", tags=["app"])

DbDep = Annotated[AsyncSession, Depends(get_db)]

_EMOTIONS = ["happiness", "neutral", "surprise", "sadness", "fear", "disgust", "contempt", "anger"]


@router.get("/data/getbetweendates")
async def get_between_dates(
    db: DbDep,
    start: int = Query(...),
    end: int = Query(...),
    nodename: str = Query(...),
    _token: dict = Depends(verify_app_token),
) -> dict:
    result = await db.execute(
        select(Detection).where(
            Detection.node_name == nodename,
            Detection.timestamp >= start,
            Detection.timestamp <= end,
        )
    )
    detections = result.scalars().all()

    counts: dict[str, int] = {emotion: 0 for emotion in _EMOTIONS}
    for detection in detections:
        if detection.emotion in counts:
            counts[detection.emotion] += 1

    total = len(detections)
    if total == 0:
        percentages = {emotion: 0 for emotion in _EMOTIONS}
    else:
        percentages = {emotion: int(counts[emotion] / total * 100) for emotion in _EMOTIONS}

    return {nodename: percentages}


@router.post("/askagent", response_model=AskAgentResponse)
async def ask_agent(
    payload: AskAgentRequest,
    _token: dict = Depends(verify_app_token),
) -> AskAgentResponse:
    try:
        reply = await run_agent(payload.message, payload.foto)
        return AskAgentResponse(response=reply)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/analyzephoto", response_model=AnalyzePhotoResponse)
async def analyze_photo(
    payload: AnalyzePhotoRequest,
    _token: dict = Depends(verify_app_token),
) -> AnalyzePhotoResponse:
    try:
        result = await predict_endpoint_two(payload.image_base64)
        return AnalyzePhotoResponse(emotion=result.get("predicted_emotion", "unknown"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
