"""Two-model GPU acceptance. Expensive hashing lives here, never in the serving path."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import time
import traceback
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from http.client import HTTPResponse
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    import torch

ROOT = Path(__file__).resolve().parents[1]
CHECK_LOADER = "scripts.validate.check_weights"


def check_weights(model: torch.nn.Module, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> None:
    """Test-only custom loader: full byte comparison and stable-address checks."""
    import torch
    import torch.distributed as dist
    from sglang.srt.distributed import get_tp_group
    from warpshift.weights import _STATE, _collect_tensors

    ((payload, _),) = named_tensors
    command = json.loads(payload)
    tp_group = get_tp_group().cpu_group
    rank, error = dist.get_rank(tp_group), None
    try:
        torch.cuda.synchronize()
        tensors = _collect_tensors(model)
        pointers = {name: tensor.data_ptr() for name, tensor in tensors.items()}
        previous_pointers = getattr(model, "_validation_pointers", pointers)
        assert pointers == previous_pointers, "Runtime addresses changed; captured CUDA graphs may be stale"
        model._validation_pointers = pointers
        tensor_hashes = {}
        for name, tensor in tensors.items():
            tensor_bytes = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
            digest = hashlib.sha256()
            for offset in range(0, tensor_bytes.numel(), 64 * 1024**2):
                digest.update(memoryview(tensor_bytes[offset : offset + 64 * 1024**2].cpu().numpy()))
            tensor_hashes[name] = digest.hexdigest()
        result = {
            "model": command["model"],
            "rank": rank,
            "tensors": len(tensors),
            "runtime_bytes": sum(tensor.numel() * tensor.element_size() for tensor in tensors.values()),
            "hashes": tensor_hashes,
            "pointers_stable": True,
        }
        delta = getattr(model, _STATE, {}).get("delta")
        if delta is not None:
            result["delta"] = delta.stats
            result["full_cpu_backups_retained"] = bool(getattr(model, _STATE)["cpu_backups"])
        filename = f"{command['model']}.rank{rank}.json"
        reference_path = Path(command["reference_dir"]) / filename
        if command["record"]:
            reference_path.write_text(json.dumps(result, indent=2) + "\n")
        else:
            assert tensor_hashes == json.loads(reference_path.read_text())["hashes"], (
                "Runtime bytes differ from native loading"
            )
        (Path(command["output_dir"]) / filename).write_text(json.dumps(result, indent=2) + "\n")
    except Exception as exc:
        error = f"rank {rank}: {type(exc).__name__}: {exc}"
    errors: list[str | None] = [None] * dist.get_world_size(tp_group)
    dist.all_gather_object(errors, error, group=tp_group)
    if any(errors):
        raise RuntimeError("; ".join(rank_error for rank_error in errors if rank_error))


def request_json(
    base_url: str,
    route: str,
    body: dict[str, Any] | None = None,
    expected_model: str | None = None,
    api_key: str | None = None,
    timeout_s: float = 1200,
) -> dict[str, Any]:
    """Request JSON; optionally verify which model served the response."""
    headers = {"Content-Type": "application/json"}
    if expected_model:
        headers["x-warpshift-model"] = expected_model
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    http_request = Request(base_url + route, data=None if body is None else json.dumps(body).encode(), headers=headers)
    with urlopen(http_request, timeout=timeout_s) as response:
        result = json.load(response)
        if expected_model:
            assert response.headers.get("x-warpshift-model") == expected_model
        return result


def verify_weights(
    base_url: str,
    model_id: str,
    reference_dir: Path,
    output_dir: Path,
    record_reference: bool,
    tp_size: int,
    api_key: str,
) -> None:
    """Run the TP worker check; record native-load hashes or compare against them."""
    import torch
    from sglang.srt.utils import MultiprocessingSerializer

    command = dict(
        model=model_id, reference_dir=str(reference_dir), output_dir=str(output_dir), record=record_reference
    )
    serialized_command = MultiprocessingSerializer.serialize(
        [(json.dumps(command), torch.empty(0, dtype=torch.uint8))],
        output_str=True,
    )
    # No concurrent inference during test-only inspection.
    response = request_json(
        base_url,
        "/update_weights_from_tensor",
        {
            "serialized_named_tensors": [serialized_command] * tp_size,
            "load_format": CHECK_LOADER,
            "flush_cache": False,
        },
        api_key=api_key,
    )
    assert response["success"], response


def compare_generation(actual: dict[str, Any], reference: dict[str, Any], atol: float) -> dict[str, float | bool]:
    """Require equal greedy text/token IDs and logprobs within absolute tolerance."""
    assert actual["text"] == reference["text"], "Greedy decode differs"
    max_abs_error = 0
    for field in ("input_token_logprobs", "output_token_logprobs"):
        actual_logprobs, reference_logprobs = actual["meta_info"][field], reference["meta_info"][field]
        assert len(actual_logprobs) == len(reference_logprobs)
        for actual_token, reference_token in zip(actual_logprobs, reference_logprobs):
            assert actual_token[1] == reference_token[1] and (actual_token[0] is None) == (reference_token[0] is None)
            if actual_token[0] is not None:
                assert math.isfinite(actual_token[0]) and math.isfinite(reference_token[0])
                max_abs_error = max(max_abs_error, abs(actual_token[0] - reference_token[0]))
    assert max_abs_error <= atol, (max_abs_error, atol)
    return {"max_abs_error": max_abs_error, "greedy_decode_equal": True, "atol": atol}


def read_gpu_status(device_ids: str) -> str:
    """Read memory and utilization for comma-separated physical GPU IDs."""
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--id=" + device_ids,
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        text=True,
        timeout=20,
    ).strip()


@contextmanager
def validation_server(
    args: argparse.Namespace,
    source: str,
    api_key: str,
    source_result: dict[str, Any],
    save_progress: Callable[[str], None],
) -> Iterator[str]:
    """Start one test server, wait for readiness, and release it even on failure."""
    output_dir = args.output_dir.resolve()
    sglang_python_path = str(ROOT / "sglang" / "python")
    source_dir = output_dir / source
    source_dir.mkdir()
    with socket.socket() as port_probe:
        port_probe.bind(("0.0.0.0", 0))
        port = port_probe.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    server_command = [
        "bash",
        str(ROOT / "scripts/serve.sh"),
        "--models",
        str(args.models.resolve()),
        "--source",
        source,
        "--",
        "--host",
        "0.0.0.0",
        "--api-key",
        api_key,
        "--port",
        str(port),
        "--tp-size",
        str(args.tp_size),
        "--context-length",
        "1024",
        "--max-total-tokens",
        "2048",
        "--max-running-requests",
        "8",
        "--chunked-prefill-size",
        "512",
        "--mem-fraction-static",
        "0.75",
        "--attention-backend",
        "triton",
        "--cuda-graph-bs",
        "1",
        "2",
        "4",
        "8",
        "--custom-weight-loader",
        "warpshift.weights.model_weight_loader",
        CHECK_LOADER,
    ]
    server_env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=args.cuda_devices,
        PYTHONPATH=os.pathsep.join([sglang_python_path, str(ROOT), os.environ.get("PYTHONPATH", "")]),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        SGLANG_JIT_DEEPGEMM_PRECOMPILE="false",
        SGL_JIT_DEEPGEMM_PRECOMPILE="false",
    )
    source_result.update(status="running", command=server_command)
    save_progress("Starting " + source)
    with (source_dir / "server.log").open("w") as server_log:
        server_process = subprocess.Popen(
            server_command,
            cwd=ROOT,
            env=server_env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            startup_deadline = time.monotonic() + 600
            while True:
                if server_process.poll() is not None:
                    raise RuntimeError(f"Server exited {server_process.returncode}; see {source_dir}/server.log")
                try:
                    request_json(base_url, "/warpshift/status", api_key=api_key, timeout_s=2)
                    break
                except Exception:
                    if time.monotonic() > startup_deadline:
                        raise TimeoutError("Server startup exceeded 600s")
                    time.sleep(2)
            yield base_url
        finally:
            if server_process.poll() is None:
                os.killpg(server_process.pid, signal.SIGTERM)
                try:
                    server_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server_process.pid, signal.SIGKILL)
                    server_process.wait(timeout=10)
            source_result["server_returncode"] = server_process.returncode
            source_result["gpu_after"] = read_gpu_status(args.cuda_devices)
            save_progress("Released " + source)


def benchmark_switches(base_url: str, model_ids: list[str], repeats: int, api_key: str) -> dict[str, Any]:
    """Measure both directions after warmup; include the complete HTTP switch."""
    timings: dict[str, list[float]] = {
        f"{model_ids[0]}->{model_ids[1]}": [],
        f"{model_ids[1]}->{model_ids[0]}": [],
    }
    for _ in range(repeats):
        for from_model, to_model in ((model_ids[0], model_ids[1]), (model_ids[1], model_ids[0])):
            started = time.perf_counter()
            response = request_json(base_url, "/warpshift/switch", {"model": to_model}, api_key=api_key)
            timings[f"{from_model}->{to_model}"].append(time.perf_counter() - started)
            assert response["from_model"] == from_model and response["model"] == to_model
    summary = {
        direction: {
            "count": len(samples),
            "median_seconds": statistics.median(samples),
            "min_seconds": min(samples),
            "max_seconds": max(samples),
        }
        for direction, samples in timings.items()
    }
    return {"timings": timings, "summary": summary}


def check_streaming_switches(
    base_url: str,
    model_ids: list[str],
    reference_outputs: dict[str, dict[str, Any]],
    generation_request: dict[str, Any],
    atol: float,
    api_key: str,
    source_result: dict[str, Any],
) -> None:
    """Check switching after disconnect and while a complete stream is still active."""
    stream_request = {
        **generation_request,
        "stream": True,
        "sampling_params": {"max_new_tokens": 256, "temperature": 0, "ignore_eos": True},
    }

    def open_generation_stream() -> HTTPResponse:
        http_request = Request(
            base_url + "/generate",
            data=json.dumps(stream_request).encode(),
            headers={
                "Content-Type": "application/json",
                "x-warpshift-model": model_ids[0],
                "Authorization": f"Bearer {api_key}",
            },
        )
        return urlopen(http_request, timeout=1200)

    # Disconnect during decode; service must remain usable and switch safely.
    with open_generation_stream() as response:
        response.readline()
    request_json(base_url, "/warpshift/switch", {"model": model_ids[1]}, api_key=api_key)
    generation_output = request_json(
        base_url, "/generate", generation_request, expected_model=model_ids[1], api_key=api_key
    )
    source_result["disconnect"] = compare_generation(generation_output, reference_outputs[model_ids[1]], atol)
    request_json(base_url, "/warpshift/switch", {"model": model_ids[0]}, api_key=api_key)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        with open_generation_stream() as response:
            first_chunk = response.readline()
            switch_future = executor.submit(
                request_json,
                base_url,
                "/warpshift/switch",
                {"model": model_ids[1]},
                api_key=api_key,
            )
            remaining_chunks = response.read()
            assert response.headers["x-warpshift-model"] == model_ids[0]
        switch_future.result()
    assert b"[DONE]" in first_chunk + remaining_chunks
    source_result["stream_completed_before_switch"] = True


def validate_source(
    args: argparse.Namespace,
    source: str,
    base_url: str,
    model_ids: list[str],
    reference_outputs: dict[str, dict[str, Any]],
    api_key: str,
    source_result: dict[str, Any],
    save_progress: Callable[[str], None],
) -> None:
    """Prepare and warm models, benchmark switching, then check correctness."""
    reference_dir = args.output_dir.resolve() / "references"
    source_dir = args.output_dir.resolve() / source
    generation_request = {
        "text": "Explain why the sky looks blue during the day. The atmosphere scatters short wavelengths.",
        "sampling_params": {"max_new_tokens": 16, "temperature": 0, "ignore_eos": True},
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    try:
        request_json(base_url, "/warpshift/status")
        raise AssertionError("Unauthenticated control accepted")
    except HTTPError as exc:
        assert exc.code == 401
    source_result["authentication_checked"] = True
    save_progress(f"Checking startup weights {source}: {model_ids[0]}")
    verify_weights(base_url, model_ids[0], reference_dir, source_dir, source == "disk", args.tp_size, api_key)
    source_result["prepare"] = request_json(base_url, "/warpshift/prepare", {}, api_key=api_key)
    save_progress(f"Prepared {source}: {source_result['prepare']['seconds']:.3f}s")
    for model_id in model_ids[::-1]:
        request_json(base_url, "/warpshift/switch", {"model": model_id}, api_key=api_key)
        generation_output = request_json(
            base_url, "/generate", generation_request, expected_model=model_id, api_key=api_key
        )
        if source == "disk":
            reference_outputs[model_id] = generation_output
            verify_weights(base_url, model_id, reference_dir, source_dir, True, args.tp_size, api_key)
        else:
            compare_generation(generation_output, reference_outputs[model_id], args.atol)
    source_result.update(benchmark_switches(base_url, model_ids, args.repeats, api_key))
    save_progress(source + " timings " + json.dumps(source_result["summary"]))
    source_result["correctness"] = {}
    for model_id in model_ids[::-1]:
        request_json(base_url, "/warpshift/switch", {"model": model_id}, api_key=api_key)
        verify_weights(base_url, model_id, reference_dir, source_dir, False, args.tp_size, api_key)
        generation_output = request_json(
            base_url, "/generate", generation_request, expected_model=model_id, api_key=api_key
        )
        source_result["correctness"][model_id] = {
            "weight_bytes_equal": True,
            **compare_generation(generation_output, reference_outputs[model_id], args.atol),
        }
    if source == "delta":
        source_result["packages"] = [
            json.loads((source_dir / f"{model_ids[0]}.rank{rank}.json").read_text())["delta"]
            for rank in range(args.tp_size)
        ]
        assert all(
            not json.loads((source_dir / f"{model_ids[0]}.rank{rank}.json").read_text())["full_cpu_backups_retained"]
            for rank in range(args.tp_size)
        )
    if source in ("pinned", "delta"):
        check_streaming_switches(
            base_url, model_ids, reference_outputs, generation_request, args.atol, api_key, source_result
        )
    source_result["final_status"] = request_json(base_url, "/warpshift/status", api_key=api_key)
    assert source_result["final_status"]["ready"] and source_result["final_status"]["active_requests"] == 0
    source_result["status"] = "passed"
    save_progress("Passed " + source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--cuda-devices", required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("disk", "pageable", "pinned", "delta"),
        default=["disk", "pageable", "pinned"],
        help="Sources to test; disk reference always runs first",
    )
    args = parser.parse_args()
    if args.repeats < 5 or len(args.cuda_devices.split(",")) != args.tp_size:
        parser.error("Require >=5 repetitions and exactly tp-size devices")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    models: dict[str, str] = json.loads(args.models.read_text())
    if len(models) != 2:
        parser.error("Requires two existing local checkpoints")
    sglang_python_path = str(ROOT / "sglang" / "python")
    import sys

    sys.path[:0] = [sglang_python_path, str(ROOT)]
    model_ids = list(models)
    reference_outputs: dict[str, dict[str, Any]] = {}
    (output_dir / "references").mkdir()
    api_key = "warpshift-local-acceptance"  # Temporary test server, not a user credential.
    report: dict[str, Any] = {
        "status": "running",
        "models": models,
        "tp_size": args.tp_size,
        "sources": {},
        "timing": "Full HTTP switch; hashing excluded; disk uses warm OS page cache, not cold storage",
        "cuda_graph_enabled": True,
        "gpu_before": read_gpu_status(args.cuda_devices),
        "sglang_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / "sglang", text=True).strip(),
        "source_sha256": {
            str(source_file.relative_to(ROOT)): hashlib.sha256(source_file.read_bytes()).hexdigest()
            for pattern in ("warpshift/*.py", "scripts/*.py", "tests/*.py", "patches/*.patch")
            for source_file in ROOT.glob(pattern)
        },
    }
    report_path = output_dir / "results.json"

    def save_progress(message: str) -> None:
        print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + message, flush=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    try:
        for source in dict.fromkeys(["disk", *args.sources]):
            source_result: dict[str, Any] = {}
            report["sources"][source] = source_result
            with validation_server(args, source, api_key, source_result, save_progress) as base_url:
                validate_source(
                    args, source, base_url, model_ids, reference_outputs, api_key, source_result, save_progress
                )
        report["status"] = "passed"
    except BaseException:
        report["status"], report["error"] = "failed", traceback.format_exc()
        raise
    finally:
        save_progress("Validation " + report["status"])


if __name__ == "__main__":
    main()
