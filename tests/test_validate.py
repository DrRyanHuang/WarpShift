"""CPU checks for the GPU validator's server lifetime; no server or GPU is started."""

from __future__ import annotations

import argparse
import signal
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock, call

import pytest

from scripts import validate


@pytest.mark.parametrize("failure", [None, "validation", "startup", "exited", "termination"])
def test_validation_server_releases_process_and_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    args = argparse.Namespace(models=tmp_path / "models.json", output_dir=tmp_path, cuda_devices="0", tp_size=1)
    process = MagicMock(pid=12345, returncode=-signal.SIGTERM)
    process.poll.return_value = 1 if failure == "exited" else None
    if failure == "termination":
        process.wait.side_effect = [subprocess.TimeoutExpired("test server", 30), 0]
    launch = Mock(return_value=process)
    kill_group = Mock()
    request = Mock(return_value={"ready": True})
    if failure == "startup":
        request.side_effect = ConnectionError("not ready")
    monkeypatch.setattr(validate.subprocess, "Popen", launch)
    monkeypatch.setattr(validate.os, "killpg", kill_group)
    monkeypatch.setattr(validate, "request_json", request)
    monkeypatch.setattr(validate, "read_gpu_status", lambda _: "0, 5 MiB, 0 %")
    monkeypatch.setattr(validate, "time", SimpleNamespace(monotonic=Mock(side_effect=[0, 601]), sleep=Mock()))

    errors = {"validation": ValueError, "startup": TimeoutError, "exited": RuntimeError}
    expected_error = pytest.raises(errors[failure]) if failure in errors else nullcontext()
    result: dict[str, Any] = {}
    progress: list[str] = []
    entered = False
    with expected_error:
        with validate.validation_server(args, "pinned", "test-key", result, progress.append) as base_url:
            entered = True
            assert base_url.startswith("http://127.0.0.1:")
            if failure == "validation":
                raise ValueError("weight verification failed")

    assert entered == (failure not in ("startup", "exited"))
    assert launch.call_args.kwargs["stdout"].closed
    assert launch.call_args.kwargs["start_new_session"] is True
    assert result["gpu_after"] == "0, 5 MiB, 0 %"
    assert result["server_returncode"] == process.returncode
    assert progress == ["Starting pinned", "Released pinned"]
    if failure == "exited":
        kill_group.assert_not_called()
        process.wait.assert_not_called()
    else:
        expected_signals = [call(process.pid, signal.SIGTERM)]
        expected_waits = [call(timeout=30)]
        if failure == "termination":
            expected_signals.append(call(process.pid, signal.SIGKILL))
            expected_waits.append(call(timeout=10))
        assert kill_group.call_args_list == expected_signals
        assert process.wait.call_args_list == expected_waits
