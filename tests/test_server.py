from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from starlette.responses import StreamingResponse

from warpshift.server import ModelSwitcher, install
from warpshift.weights import WeightSource


class FakeWeightClient:
    def __init__(self) -> None:
        self.headers = {}
        self.current: str | None = None
        self.events: list[str] = []
        self.fail = False
        self.block: threading.Event | None = None
        self.started = threading.Event()

    def prepare(self, models: dict[str, str], source: WeightSource, target: str) -> dict[str, Any]:
        self.events.append("prepare")
        self.current = target
        return {"model": target, "source": source}

    def restore(self, target: str) -> dict[str, Any]:
        self.events.append("restore:" + target)
        self.started.set()
        if self.block is not None:
            assert self.block.wait(5)
        if self.fail:
            raise RuntimeError("partial write")
        self.current = target
        return {"model": target, "changed": True}

    def _request(self, route: str, body: dict[str, Any]) -> dict[str, Any]:
        self.events.append(route)
        return {"status": "ok"}


def make_switcher() -> ModelSwitcher:
    client = FakeWeightClient()
    return ModelSwitcher(client, {"a": "/a", "b": "/b"}, "pinned", "a", drain_timeout=1)


def test_drains_old_requests_and_blocks_new_before_switch() -> None:
    async def scenario() -> None:
        switcher = make_switcher()
        await switcher.switch_or_prepare("prepare")
        async with switcher.inference("a"):
            before = len(switcher.client.events)
            switch = asyncio.create_task(switcher.switch_or_prepare("switch", "b"))
            await asyncio.sleep(0)
            assert switcher.state == "draining"
            assert len(switcher.client.events) == before
            with pytest.raises(HTTPException) as exc:
                async with switcher.inference(None):
                    pass
            assert exc.value.status_code == 503
        result = await switch
        assert result["from_model"] == "a" and result["model"] == "b"
        assert switcher.client.events[-3:] == ["/pause_generation", "restore:b", "/continue_generation"]
        with pytest.raises(HTTPException) as exc:
            async with switcher.inference("a"):
                pass
        assert exc.value.status_code == 409

    asyncio.run(scenario())


def test_failed_switch_never_reopens_admission_or_resumes_engine() -> None:
    async def scenario() -> None:
        switcher = make_switcher()
        await switcher.switch_or_prepare("prepare")
        switcher.client.fail = True
        with pytest.raises(RuntimeError, match="partial write"):
            await switcher.switch_or_prepare("switch", "b")
        assert switcher.state == "failed"
        assert switcher.client.events[-1] == "restore:b"
        with pytest.raises(HTTPException):
            await switcher.switch_or_prepare("prepare")
        with pytest.raises(HTTPException):
            async with switcher.inference(None):
                pass

    asyncio.run(scenario())


def test_cancelled_control_waits_for_gpu_operation_before_releasing_gate() -> None:
    async def scenario() -> None:
        switcher = make_switcher()
        await switcher.switch_or_prepare("prepare")
        switcher.client.block = threading.Event()
        task = asyncio.create_task(switcher.switch_or_prepare("switch", "b"))
        assert await asyncio.to_thread(switcher.client.started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and switcher.state == "switching"
        switcher.client.block.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert switcher.state == "ready" and switcher.client.current == "b"
        assert switcher.client.events[-1] == "/continue_generation"

    asyncio.run(scenario())


def test_cancelled_inference_releases_reader_and_next_switch_uses_native_abort() -> None:
    async def scenario() -> None:
        switcher = make_switcher()
        await switcher.switch_or_prepare("prepare")
        entered = asyncio.Event()

        async def request() -> None:
            async with switcher.inference("a"):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(request())
        await entered.wait()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        assert switcher.state == "ready" and switcher.active == 0
        await switcher.switch_or_prepare("switch", "b")
        assert switcher.client.events[-3:] == ["/pause_generation", "restore:b", "/continue_generation"]

    asyncio.run(scenario())


def test_drain_timeout_keeps_existing_model_and_does_not_touch_weights() -> None:
    async def scenario() -> None:
        switcher = make_switcher()
        await switcher.switch_or_prepare("prepare")
        switcher.drain_timeout = 0.01
        before = list(switcher.client.events)
        async with switcher.inference("a"):
            with pytest.raises(TimeoutError):
                await switcher.switch_or_prepare("switch", "b")
            assert switcher.state == "ready"
        assert switcher.client.events == before and switcher.client.current == "a"

    asyncio.run(scenario())


def test_http_routes_tag_models_and_leave_native_routes_alone() -> None:
    async def scenario() -> None:
        switcher = make_switcher()
        app = FastAPI()

        @app.post("/generate")
        async def generate() -> dict[str, str]:
            return {"text": "done"}

        @app.post("/update_weights_from_disk")
        async def native() -> dict[str, bool]:
            return {"success": True}

        install(app, switcher)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.post("/generate", json={})).status_code == 503
            assert (await client.post("/warpshift/prepare", json={})).status_code == 200
            response = await client.post("/generate", json={}, headers={"x-warpshift-model": "a"})
            assert response.status_code == 200 and response.headers["x-warpshift-model"] == "a"
            assert (await client.post("/warpshift/switch", json={"model": "b"})).status_code == 200
            assert (await client.post("/generate", json={}, headers={"x-warpshift-model": "a"})).status_code == 409
            assert (await client.post("/update_weights_from_disk", json={})).status_code == 200
            assert switcher.state == "ready"

    asyncio.run(scenario())


def test_stream_remains_active_until_last_body_chunk() -> None:
    async def scenario() -> None:
        switcher = make_switcher()
        await switcher.switch_or_prepare("prepare")
        app = FastAPI()
        first, finish = asyncio.Event(), asyncio.Event()

        @app.post("/generate")
        async def generate() -> StreamingResponse:
            async def tokens() -> AsyncIterator[bytes]:
                yield b"first"
                first.set()
                await finish.wait()
                yield b"last"

            return StreamingResponse(tokens())

        install(app, switcher)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            request = asyncio.create_task(client.post("/generate", json={}))
            await first.wait()
            assert switcher.active == 1
            switching = asyncio.create_task(switcher.switch_or_prepare("switch", "b"))
            await asyncio.sleep(0)
            assert switcher.state == "draining" and not switching.done()
            finish.set()
            response = await request
            assert response.content == b"firstlast"
            assert response.headers["x-warpshift-model"] == "a"
            await switching
            assert switcher.active == 0 and switcher.client.current == "b"

    asyncio.run(scenario())


def test_invalid_target_does_not_poison_or_switch() -> None:
    async def scenario() -> None:
        switcher = make_switcher()
        await switcher.switch_or_prepare("prepare")
        before = list(switcher.client.events)
        with pytest.raises(HTTPException):
            await switcher.switch_or_prepare("switch", "missing")
        assert switcher.client.events == before and switcher.state == "ready"

    asyncio.run(scenario())


def test_control_endpoints_reuse_native_admin_key_auth() -> None:
    from sglang.srt.utils.auth import add_api_key_middleware

    async def scenario() -> None:
        switcher = make_switcher()
        app = FastAPI()
        install(app, switcher)
        add_api_key_middleware(app, api_key="inference", admin_api_key="admin")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.post("/warpshift/prepare")).status_code == 401
            assert (
                await client.post("/warpshift/prepare", headers={"Authorization": "Bearer inference"})
            ).status_code == 401
            assert (
                await client.post("/warpshift/prepare", headers={"Authorization": "Bearer admin"})
            ).status_code == 200

    asyncio.run(scenario())


def test_removed_prepare_options_and_empty_model_are_rejected() -> None:
    async def scenario() -> None:
        switcher = make_switcher()
        app = FastAPI()
        install(app, switcher)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/warpshift/prepare", json={"source": "pageable"})
            assert response.status_code == 400 and not switcher.client.events
            assert (await client.post("/warpshift/prepare", json={})).status_code == 200
            before = list(switcher.client.events)
            assert (await client.post("/warpshift/switch", json={"model": ""})).status_code == 400
            assert switcher.client.events == before and switcher.client.current == "a"

    asyncio.run(scenario())
