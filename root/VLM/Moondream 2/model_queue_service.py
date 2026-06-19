#!/usr/bin/env python3
"""Async queued inference helpers for FastAPI endpoints.

Import this module from your FastAPI app, call ``startup_model_service`` once
on application startup, then await one of the endpoint-specific prediction
functions for every incoming base64 image.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import time
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence

import torch
from PIL import Image, UnidentifiedImageError

from predict_folder import (
    DEFAULT_EMOTIONS,
    DEFAULT_LOCAL_MODEL_DIR,
    build_user_prompt,
    load_moondream,
    normalize_emotion,
    parse_emotion,
)


ENDPOINT_ONE = "endpoint_one"
ENDPOINT_TWO = "endpoint_two"


@dataclass(frozen=True)
class ModelServiceConfig:
    model_dir: str
    base_model_dir: str = DEFAULT_LOCAL_MODEL_DIR
    endpoint_one_schema: str = "primary"
    endpoint_two_schema: str = "primary"
    emotion_labels: Sequence[str] = DEFAULT_EMOTIONS
    max_new_tokens: int = 96
    temperature: float = 0.0
    device: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = True
    endpoint_one_queue_size: int = 0
    endpoint_two_queue_size: int = 0
    endpoint_two_priority_weight: int = 3


@dataclass
class PredictionJob:
    image_base64: str
    future: asyncio.Future
    received_at: float


class QueuedMoondreamService:
    def __init__(self, config: ModelServiceConfig) -> None:
        self.config = config
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.model_source: Optional[str] = None
        self.device: Optional[torch.device] = None
        self._args = self._build_args(config)
        self._emotion_labels = [normalize_emotion(label) for label in config.emotion_labels if label.strip()]
        if not self._emotion_labels:
            raise ValueError("emotion_labels cannot be empty.")

        self._prompts = {
            ENDPOINT_ONE: build_user_prompt(self._emotion_labels, config.endpoint_one_schema),
            ENDPOINT_TWO: build_user_prompt(self._emotion_labels, config.endpoint_two_schema),
        }
        self._schemas = {
            ENDPOINT_ONE: config.endpoint_one_schema,
            ENDPOINT_TWO: config.endpoint_two_schema,
        }
        self._queues: Dict[str, asyncio.Queue[PredictionJob]] = {
            ENDPOINT_ONE: asyncio.Queue(maxsize=config.endpoint_one_queue_size),
            ENDPOINT_TWO: asyncio.Queue(maxsize=config.endpoint_two_queue_size),
        }
        self._prefetched_jobs: Dict[str, deque[PredictionJob]] = {
            ENDPOINT_ONE: deque(),
            ENDPOINT_TWO: deque(),
        }
        self._workers: list[asyncio.Task] = []
        self._endpoint_two_credit = max(1, config.endpoint_two_priority_weight)
        self._started = False

    @staticmethod
    def _build_args(config: ModelServiceConfig) -> SimpleNamespace:
        return SimpleNamespace(
            model_dir=config.model_dir,
            base_model_dir=config.base_model_dir,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            device=config.device,
            torch_dtype=config.torch_dtype,
            trust_remote_code=config.trust_remote_code,
        )

    async def start(self) -> None:
        if self._started:
            return
        self.model, self.tokenizer, self.model_source, self.device = await asyncio.to_thread(load_moondream, self._args)
        self._workers = [asyncio.create_task(self._worker(), name="priority_inference_worker")]
        self._started = True

    async def stop(self) -> None:
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._started = False

    async def predict(self, endpoint_name: str, image_base64: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        if not self._started:
            raise RuntimeError("Model service is not started. Call startup_model_service from FastAPI startup.")
        if endpoint_name not in self._queues:
            raise ValueError(f"Unknown endpoint queue: {endpoint_name}")

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        job = PredictionJob(image_base64=image_base64, future=future, received_at=time.perf_counter())
        await self._queues[endpoint_name].put(job)
        return await asyncio.wait_for(future, timeout=timeout)

    def queue_sizes(self) -> Dict[str, int]:
        return {
            name: queue.qsize() + len(self._prefetched_jobs[name])
            for name, queue in self._queues.items()
        }

    async def _worker(self) -> None:
        while True:
            endpoint_name, job = await self._get_next_job()
            try:
                if job.future.cancelled():
                    continue
                result = await self._run_prediction(endpoint_name, job.image_base64, job.received_at)
                job.future.set_result(result)
            except Exception as exc:
                if not job.future.cancelled():
                    job.future.set_exception(exc)
            finally:
                self._queues[endpoint_name].task_done()

    async def _get_next_job(self) -> tuple[str, PredictionJob]:
        while True:
            queued_job = self._get_queued_job_nowait()
            if queued_job is not None:
                return queued_job

            get_tasks = {
                asyncio.create_task(self._queues[ENDPOINT_ONE].get()): ENDPOINT_ONE,
                asyncio.create_task(self._queues[ENDPOINT_TWO].get()): ENDPOINT_TWO,
            }
            done, pending = await asyncio.wait(get_tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

            endpoint_name = self._choose_completed_endpoint(done, get_tasks)
            job = None
            for task in done:
                task_endpoint = get_tasks[task]
                task_job = task.result()
                if task_endpoint == endpoint_name and job is None:
                    job = task_job
                else:
                    self._prefetched_jobs[task_endpoint].append(task_job)
            if job is None:
                raise RuntimeError("Priority scheduler did not select a completed job.")
            self._record_priority_choice(endpoint_name)
            return endpoint_name, job

    def _get_queued_job_nowait(self) -> Optional[tuple[str, PredictionJob]]:
        endpoint_one_waiting = bool(self._prefetched_jobs[ENDPOINT_ONE]) or not self._queues[ENDPOINT_ONE].empty()
        endpoint_two_waiting = bool(self._prefetched_jobs[ENDPOINT_TWO]) or not self._queues[ENDPOINT_TWO].empty()

        if endpoint_two_waiting and (self._endpoint_two_credit > 0 or not endpoint_one_waiting):
            return self._get_job_nowait(ENDPOINT_TWO)
        if endpoint_one_waiting:
            return self._get_job_nowait(ENDPOINT_ONE)
        if endpoint_two_waiting:
            return self._get_job_nowait(ENDPOINT_TWO)
        return None

    def _get_job_nowait(self, endpoint_name: str) -> tuple[str, PredictionJob]:
        if self._prefetched_jobs[endpoint_name]:
            job = self._prefetched_jobs[endpoint_name].popleft()
        else:
            job = self._queues[endpoint_name].get_nowait()
        self._record_priority_choice(endpoint_name)
        return endpoint_name, job

    def _choose_completed_endpoint(
        self,
        done: set[asyncio.Task],
        get_tasks: Dict[asyncio.Task, str],
    ) -> str:
        completed_endpoints = {get_tasks[task] for task in done}
        if ENDPOINT_TWO in completed_endpoints and (
            self._endpoint_two_credit > 0 or ENDPOINT_ONE not in completed_endpoints
        ):
            return ENDPOINT_TWO
        if ENDPOINT_ONE in completed_endpoints:
            return ENDPOINT_ONE
        return ENDPOINT_TWO

    def _record_priority_choice(self, endpoint_name: str) -> None:
        if endpoint_name == ENDPOINT_TWO:
            self._endpoint_two_credit = max(0, self._endpoint_two_credit - 1)
        else:
            self._endpoint_two_credit = max(1, self.config.endpoint_two_priority_weight)

    async def _run_prediction(self, endpoint_name: str, image_base64: str, received_at: float) -> Dict[str, Any]:
        started_at = time.perf_counter()
        image = decode_base64_image(image_base64)
        prompt = self._prompts[endpoint_name]
        schema = self._schemas[endpoint_name]

        raw_output = await asyncio.to_thread(self._predict_image, image, prompt)

        finished_at = time.perf_counter()
        predicted_emotion, info = parse_emotion(raw_output, self._emotion_labels, schema)
        return {
            "predicted_emotion": predicted_emotion,
            "parse_status": info["parse_status"],
            "raw_model_output": raw_output,
            "parsed_model_output": info.get("parsed_json"),
            "model_source": self.model_source,
            "queue": endpoint_name,
            "queue_wait_seconds": round(started_at - received_at, 6),
            "processing_seconds": round(finished_at - started_at, 6),
        }

    @torch.inference_mode()
    def _predict_image(self, image: Image.Image, prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model service is not started.")
        if not hasattr(self.model, "encode_image") or not hasattr(self.model, "answer_question"):
            raise RuntimeError("Loaded model does not expose Moondream encode_image/answer_question methods.")

        encoded_image = self.model.encode_image(image)
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.temperature > 0.0,
            "temperature": self.config.temperature if self.config.temperature > 0.0 else None,
        }
        generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}
        try:
            return self.model.answer_question(encoded_image, prompt, self.tokenizer, **generation_kwargs).strip()
        except TypeError:
            return self.model.answer_question(encoded_image, prompt, self.tokenizer).strip()


def decode_base64_image(image_base64: str) -> Image.Image:
    payload = image_base64.strip()
    if "," in payload and payload.lower().startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Request image is not valid base64.") from exc
    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Decoded base64 payload is not a supported image.") from exc


_service: Optional[QueuedMoondreamService] = None


async def startup_model_service(config: ModelServiceConfig) -> QueuedMoondreamService:
    global _service
    if _service is None:
        _service = QueuedMoondreamService(config)
        await _service.start()
    return _service


async def shutdown_model_service() -> None:
    global _service
    if _service is not None:
        await _service.stop()
        _service = None


def get_model_service() -> QueuedMoondreamService:
    if _service is None:
        raise RuntimeError("Model service is not started.")
    return _service


async def predict_endpoint_one(image_base64: str, timeout: Optional[float] = None) -> Dict[str, Any]:
    return await get_model_service().predict(ENDPOINT_ONE, image_base64, timeout=timeout)


async def predict_endpoint_two(image_base64: str, timeout: Optional[float] = None) -> Dict[str, Any]:
    return await get_model_service().predict(ENDPOINT_TWO, image_base64, timeout=timeout)
