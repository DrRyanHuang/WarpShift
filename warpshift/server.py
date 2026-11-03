"""Request draining and model switching on SGLang's existing HTTP server."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from warpshift.weights import ModelWeightClient, WeightSource


class ModelSwitcher:
    failure: str | None

    def __init__(
        self,
        client: ModelWeightClient,
        models: dict[str, str],
        source: WeightSource,
        initial: str,
        drain_timeout: float = 1200,
    ) -> None:
        """Coordinate one engine's switches.

        client: SGLang weight-control client for this engine.
        models: Model ID -> local checkpoint directory.
        source: disk, pageable, pinned, or delta; fixed for this server.
        initial: Model ID selected after prepare, not a checkpoint path.
        drain_timeout: Maximum wait for active requests to finish, in seconds.
        """
        self.client, self.models, self.source, self.initial = client, models, source, initial
        self.drain_timeout = drain_timeout
        self.state, self.failure, self.active = "unprepared", None, 0
        self.condition, self.writer = asyncio.Condition(), asyncio.Lock()

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "ready": self.state == "ready",
            "active_requests": self.active,
            "model": self.client.current,
            "models": list(self.models),
            "source": self.source,
            "failure": self.failure,
        }

    @asynccontextmanager
    async def inference(self, expected: str | None) -> AsyncIterator[str | None]:
        async with self.condition:
            if self.state != "ready":
                raise HTTPException(503, f"WarpShift is {self.state}; retry after switching completes")
            current = self.client.current
            if expected is not None and expected != current:
                raise HTTPException(409, f"Requested model {expected}, current model is {current}")
            self.active += 1
        try:
            yield current
        finally:
            async with self.condition:
                self.active -= 1
                if self.active == 0:
                    self.condition.notify_all()

    async def _pause_and_update(self, operation: str, target: str) -> dict[str, Any]:
        """Pause the engine, update weights, then resume only after success."""
        started, previous_model = time.perf_counter(), self.client.current
        try:
            # Native abort also drains GPU work left by disconnected HTTP clients.
            await asyncio.to_thread(self.client._request, "/pause_generation", {"mode": "abort"})
            if operation == "prepare":
                result = await asyncio.to_thread(self.client.prepare, self.models, self.source, target)
            else:
                result = await asyncio.to_thread(self.client.restore, target)
            await asyncio.to_thread(self.client._request, "/continue_generation", {})
            self.state = "ready"
            return {
                **result,
                "from_model": previous_model,
                "operation": operation,
                "transaction_seconds": time.perf_counter() - started,
            }
        except Exception as exc:
            self.failure, self.state = f"{type(exc).__name__}: {exc}", "failed"
            raise  # Partial writes must not reopen inference.

    async def switch_or_prepare(self, operation: str, target: str | None = None) -> dict[str, Any]:
        """Serialize updates and drain active requests before pausing the engine."""
        target = self.initial if target is None else target
        if operation not in ("prepare", "switch") or target not in self.models:
            raise HTTPException(400, "Unknown operation or model")
        async with self.writer:
            if self.state == "failed":
                raise HTTPException(503, "Weight update failed; restart WarpShift")
            if operation != "prepare" and self.state != "ready":
                raise HTTPException(409, "Prepare models before switching")
            previous_state = self.state
            async with self.condition:
                self.state = "draining"
                try:
                    await asyncio.wait_for(self.condition.wait_for(lambda: self.active == 0), self.drain_timeout)
                except BaseException:
                    self.state = previous_state
                    raise
            self.state = "switching"
            task = asyncio.create_task(self._pause_and_update(operation, target))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # Keep the writer lock until the update finishes, even if its caller disconnects.
                try:
                    await task
                finally:
                    raise


class AdmissionMiddleware:
    INFERENCE = {"/generate", "/v1/completions", "/v1/chat/completions"}

    def __init__(self, app: ASGIApp, switcher: ModelSwitcher) -> None:
        self.app, self.switcher = app, switcher

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in self.INFERENCE:
            await self.app(scope, receive, send)
            return
        expected = dict(scope.get("headers", [])).get(b"x-warpshift-model")
        try:
            async with self.switcher.inference(expected.decode() if expected else None) as model:

                async def tagged_send(message: Message) -> None:
                    if message["type"] == "http.response.start":
                        message = {
                            **message,
                            "headers": [*message.get("headers", []), (b"x-warpshift-model", model.encode())],
                        }
                    await send(message)

                await self.app(scope, receive, tagged_send)
        except HTTPException as exc:
            await JSONResponse({"error": exc.detail}, status_code=exc.status_code)(scope, receive, send)


def install(app: FastAPI, switcher: ModelSwitcher) -> None:
    # Reuse SGLang's native admin/API-key policy, including on our control endpoints.
    from sglang.srt.utils.auth import AuthLevel, auth_level

    @app.get("/warpshift/status")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def status() -> dict[str, Any]:
        return switcher.status()

    @app.post("/warpshift/prepare")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def prepare(body: dict[str, Any] | None = None) -> dict[str, Any]:
        if body:
            raise HTTPException(400, "prepare takes no options; choose --source at launch")
        return await switcher.switch_or_prepare("prepare")

    @app.post("/warpshift/switch")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def switch(body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(body.get("model"), str):
            raise HTTPException(400, "model is required")
        return await switcher.switch_or_prepare("switch", body["model"])

    app.add_middleware(AdmissionMiddleware, switcher=switcher)  # 我们这个是不是没啥用啊？
