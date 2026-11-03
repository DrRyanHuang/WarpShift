# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU weight backups and reversible delta packages for a drained SGLang instance."""
# 这个文件被分成了上下两半，分别跑在两个不同的进程里！
# ModelWeightClient 跑在 HTTP 主进程里，负责发号施令。
# model_weight_loader / _update_worker_weights 跑在 GPU 子进程（TP Worker）里，直接操作显卡里的 PyTorch model 对象。

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    import torch

WeightSource = Literal["disk", "pageable", "pinned", "delta"]

LOADER = "warpshift.weights.model_weight_loader"
_STATE = "_warpshift_weights"


def _collect_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Registered parameters and buffers, including non-persistent derived weights."""
    tensors, seen = {}, set()
    for prefix, named_tensors in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, tensor in named_tensors:
            tensor_key = (tensor.device, tensor.data_ptr(), tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()))
            if tensor.numel() and tensor_key not in seen:
                seen.add(tensor_key)
                tensors[f"{prefix}:{name}"] = tensor
    return tensors


def _update_worker_weights(model: torch.nn.Module, command: dict[str, Any]) -> None:
    import torch

    target = command["model"]
    if command["operation"] == "backup":
        if command.get("reset"):
            setattr(model, _STATE, {"cpu_backups": {}})
        state = getattr(model, _STATE)
        gpu_tensors = _collect_tensors(model)
        layout = {name: (tuple(tensor.shape), tensor.dtype) for name, tensor in gpu_tensors.items()}
        if not gpu_tensors or layout != state.get("layout", layout):
            raise ValueError("Model tensor layout changed")
        state["gpu_tensors"], state["layout"] = gpu_tensors, layout
        state["pinned"] = command["source"] == "pinned"
        if command["source"] == "delta" and state["cpu_backups"]:
            from warpshift.delta import DeltaWeights

            base_model, cpu_base = next(iter(state["cpu_backups"].items()))
            if base_model == target or "delta" in state:
                raise ValueError("Delta prepare requires exactly two distinct models")
            logging.getLogger(__name__).info("Building delta package: %s <-> %s", base_model, target)
            state["delta"] = DeltaWeights(cpu_base, gpu_tensors)
            state["delta_models"] = (base_model, target)
            state["cpu_backups"].clear()
            logging.getLogger(__name__).info("Delta package ready: %s", json.dumps(state["delta"].stats))
            return
        cpu_backup = {}
        for name, gpu_tensor in gpu_tensors.items():
            cpu_tensor = torch.empty_like(gpu_tensor, device="cpu", pin_memory=state["pinned"])
            cpu_tensor.copy_(gpu_tensor.detach())
            cpu_backup[name] = cpu_tensor
        state["cpu_backups"][target] = cpu_backup
    elif command["operation"] == "restore":
        state = getattr(model, _STATE)
        if command["source"] == "delta":
            base_model = state["delta_models"][0]
            state["delta"].apply(reverse=target == base_model)
            return  # Commit the source-model identity only after every TP rank succeeds.
        cpu_backup = state["cpu_backups"][target]
        with torch.no_grad():
            for name, gpu_tensor in state["gpu_tensors"].items():
                gpu_tensor.copy_(cpu_backup[name], non_blocking=state["pinned"])
        for device in {tensor.device for tensor in state["gpu_tensors"].values() if tensor.is_cuda}:
            torch.cuda.synchronize(device)
    else:
        raise ValueError(f"Unknown operation: {command['operation']}")


def _validate_delta_restore(model: torch.nn.Module, command: dict[str, Any]) -> None:
    """Reject a stale source or changed GPU storage before any TP rank writes."""
    if command["source"] != "delta" or command["operation"] != "restore":
        return
    state = getattr(model, _STATE)
    if "delta" not in state:
        raise ValueError("Prepare the delta package before restoring")
    if command.get("from_model") != state["current"]:
        raise ValueError("Delta source model does not match resident weights")
    if command["model"] not in state["delta_models"] or command["model"] == state["current"]:
        raise ValueError("Delta target must be the other model in the prepared pair")
    current_tensors = _collect_tensors(model)
    if current_tensors.keys() != state["gpu_tensors"].keys() or any(
        tensor is not state["gpu_tensors"][name] for name, tensor in current_tensors.items()
    ):
        raise ValueError("Delta model tensors were replaced; prepare again")
    state["delta"].validate_bindings()


def model_weight_loader(model: torch.nn.Module, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> None:
    """Agree before writes and acknowledge GPU completion on every TP rank."""
    import torch.distributed as dist
    from sglang.srt.distributed import get_tp_group

    tp_group = get_tp_group().cpu_group
    rank, size = dist.get_rank(tp_group), dist.get_world_size(tp_group)
    command, error = None, None
    try:
        ((payload, _),) = named_tensors
        command = json.loads(payload)
        _validate_delta_restore(model, command)
    except Exception as exc:
        error = f"rank {rank}: {type(exc).__name__}: {exc}"
    proposals: list[tuple[dict[str, Any] | None, str | None]] = [(None, None)] * size
    dist.all_gather_object(proposals, (command, error), group=tp_group)
    errors = [failure for _, failure in proposals if failure]
    if errors:
        raise RuntimeError("; ".join(errors))
    if any(proposal != command for proposal, _ in proposals):
        raise RuntimeError("TP ranks received different weight commands")

    try:
        _update_worker_weights(model, command)
    except Exception as exc:
        error = f"rank {rank}: {type(exc).__name__}: {exc}"
    completions: list[str | None] = [None] * size
    dist.all_gather_object(completions, error, group=tp_group)
    if any(completions):
        raise RuntimeError("; ".join(failure for failure in completions if failure))
    if command["source"] == "delta":
        getattr(model, _STATE)["current"] = command["model"]


def _normalized_config(value: Any) -> Any:
    if isinstance(value, dict):
        ignored = {"_name_or_path", "_commit_hash", "transformers_version", "dtype", "torch_dtype"}
        return {key: _normalized_config(child) for key, child in value.items() if key not in ignored}
    if isinstance(value, list):
        return [_normalized_config(child) for child in value]
    return value


def _validate_models(models: dict[str, str]) -> dict[str, str]:
    """Check local paths and structure; callers supply checkpoints with the same tokenizer."""
    if not models:
        raise ValueError("At least one model is required")
    reference_config, resolved_paths = None, {}
    for model, path in models.items():
        if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", model):
            raise ValueError("Model IDs must contain only ASCII letters, digits, dot, underscore or hyphen")
        directory = Path(path).resolve(strict=True)
        config = _normalized_config(json.loads((directory / "config.json").read_text()))
        if config.get("quantization_config"):
            raise ValueError("Quantized models are unsupported")
        if reference_config is not None and config != reference_config:
            raise ValueError(f"Model config mismatch: {model}")
        reference_config, resolved_paths[model] = config, str(directory)
    return resolved_paths


class ModelWeightClient:
    """Control one engine. The server serializes calls; no second lock or state machine."""

    models: dict[str, str]
    source: WeightSource | None
    current: str | None

    def __init__(self, base_url: str, tp_size: int, timeout_s: float = 1200, api_key: str | None = None) -> None:
        self.base_url, self.tp_size, self.timeout_s = base_url.rstrip("/"), tp_size, timeout_s
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.models, self.source, self.current = {}, None, None

    def _request(self, route: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            self.base_url + route,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", **self.headers},
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            result = json.load(response)
        if result.get("success", result.get("status") == "ok") is not True:
            raise RuntimeError(f"SGLang {route} failed: {result}")
        return result

    def _control(self, operation: Literal["backup", "restore"], model: str, reset: bool = False) -> dict[str, Any]:
        import torch
        from sglang.srt.utils import MultiprocessingSerializer

        command = {
            "operation": operation,
            "model": model,
            "from_model": self.current,
            "source": self.source,
            "reset": reset,
        }
        serialized = MultiprocessingSerializer.serialize(
            [(json.dumps(command), torch.empty(0, dtype=torch.uint8))],
            output_str=True,
        )
        return self._request(
            "/update_weights_from_tensor",
            {
                "serialized_named_tensors": [serialized] * self.tp_size,
                "load_format": LOADER,
                "flush_cache": operation == "restore",
                "weight_version": model,
            },
        )

    def _load_from_disk(self, model: str) -> None:
        self._request(
            "/update_weights_from_disk",
            {
                "model_path": self.models[model],
                "load_format": "auto",
                "flush_cache": True,
                "weight_version": model,
            },
        )
        self.current = model

    def prepare(self, models: dict[str, str], source: WeightSource, initial: str) -> dict[str, Any]:
        if self.source is not None and source != self.source:
            raise ValueError("Choose the weight source at launch; restart to change it")
        if source == "delta" and len(models) != 2:
            raise ValueError("Delta source currently requires exactly two models")
        self.models, self.source = models, source
        started, records = time.perf_counter(), []
        for index, model in enumerate(models):
            load_started = time.perf_counter()
            if self.current != model:
                self._load_from_disk(model)
            load_finished = time.perf_counter()
            if source != "disk":
                self._control("backup", model, reset=index == 0)  # 通知 GPU worker 将当前权重备份到 CPU。
            records.append(
                {
                    "model": model,
                    "load_seconds": load_finished - load_started,
                    "backup_seconds": time.perf_counter() - load_finished,
                }
            )
        self.restore(initial)
        return {"source": source, "model": self.current, "seconds": time.perf_counter() - started, "backups": records}

    def restore(self, model: str) -> dict[str, Any]:
        """Switch using the configured checkpoint, CPU backup, or delta package."""
        started, changed = time.perf_counter(), self.current != model
        if changed:
            if self.source == "disk":
                self._load_from_disk(model)
            else:
                self._control("restore", model)
            self.current = model
        return {"model": model, "source": self.source, "changed": changed, "seconds": time.perf_counter() - started}
