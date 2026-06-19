import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../VLM/Moondream 2'))

from fastapi import FastAPI
from routers import emonodes, app_routes
from model_queue_service import ModelServiceConfig, startup_model_service, shutdown_model_service

app = FastAPI(title="Group Emotion Recognition API", version="1.0.0")

app.include_router(emonodes.router)
app.include_router(app_routes.router)


@app.on_event("startup")
async def startup():
    await startup_model_service(
        ModelServiceConfig(
            model_dir=os.getenv("VLM_MODEL_DIR", "/workspace/marco/moondream-ferplus-emotion-full-fp16"),
            endpoint_one_schema="primary",
            endpoint_two_schema="primary",
            endpoint_two_priority_weight=3,
            device="auto",
            torch_dtype="auto",
        )
    )
    print("VLM model service ready.")

    try:
        import asyncio
        _sys_path = os.path.join(os.path.dirname(__file__), 'Silvan_Agent')
        if _sys_path not in sys.path:
            sys.path.insert(0, _sys_path)
        from silvan_agent import handle_request
        await asyncio.to_thread(handle_request, {"question": "warmup"})
        print("AI agent ready.")
    except Exception:
        print("AI agent warmup failed - will retry on first request.")


@app.on_event("shutdown")
async def shutdown():
    await shutdown_model_service()
