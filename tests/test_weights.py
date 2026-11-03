# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU contracts for weight backups, delta preparation, and TP update agreement."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path
from typing import Any, Literal

import pytest

from warpshift import weights as backend

torch = pytest.importorskip("torch")


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(8, dtype=torch.bfloat16).reshape(2, 4))
        self.tied = self.weight
        self.A_log = torch.nn.Parameter(torch.tensor([0.001], dtype=torch.float32))
        self.register_buffer("derived", torch.tensor([0.25, 0.5]), persistent=False)
        # Same start pointer, different shape: do not deduplicate away the full tensor.
        self.register_buffer("view", self.weight.detach()[:, :2])


def weight_command(
    operation: Literal["backup", "restore"],
    model: str,
    reset: bool = False,
    source: backend.WeightSource | None = "pageable",
    from_model: str | None = None,
) -> dict[str, Any]:
    return {"operation": operation, "model": model, "reset": reset, "source": source, "from_model": from_model}


def test_round_trip_restores_bf16_fp32_buffers_and_preserves_pointers() -> None:
    model = TinyModel()
    tensors = backend._collect_tensors(model)
    pointers = {name: tensor.data_ptr() for name, tensor in tensors.items()}
    expected = {"a": {name: tensor.detach().clone() for name, tensor in tensors.items()}}
    backend._update_worker_weights(model, weight_command("backup", "a", reset=True))
    with torch.no_grad():
        model.weight.add_(10)
        model.A_log.add_(1)
        model.derived.add_(2)
    expected["b"] = {name: tensor.detach().clone() for name, tensor in tensors.items()}
    backend._update_worker_weights(model, weight_command("backup", "b"))
    for target in ("a", "b", "a", "b"):
        backend._update_worker_weights(model, weight_command("restore", target))
        for name, tensor in tensors.items():
            assert torch.equal(tensor, expected[target][name])
            assert tensor.data_ptr() == pointers[name]
        assert model.tied is model.weight
    with torch.no_grad():
        model.weight.zero_()
    assert torch.equal(
        getattr(model, backend._STATE)["cpu_backups"]["b"]["parameter:weight"], expected["b"]["parameter:weight"]
    )


def test_restore_uses_cached_tensors_and_reset_drops_old_cpu_backups(monkeypatch: pytest.MonkeyPatch) -> None:
    model = TinyModel()
    backend._update_worker_weights(model, weight_command("backup", "a", reset=True))
    backend._update_worker_weights(model, weight_command("backup", "b"))
    collect_tensors = backend._collect_tensors
    monkeypatch.setattr(backend, "_collect_tensors", lambda _: pytest.fail("Hot path rescanned model"))
    backend._update_worker_weights(model, weight_command("restore", "a"))
    monkeypatch.setattr(backend, "_collect_tensors", collect_tensors)
    backend._update_worker_weights(model, weight_command("backup", "c", reset=True))
    assert list(getattr(model, backend._STATE)["cpu_backups"]) == ["c"]


def test_backup_rejects_changed_layout_before_storing_it() -> None:
    model = TinyModel()
    backend._update_worker_weights(model, weight_command("backup", "a", reset=True))
    model.register_buffer("extra", torch.ones(1))
    with pytest.raises(ValueError, match="layout changed"):
        backend._update_worker_weights(model, weight_command("backup", "b"))
    assert list(getattr(model, backend._STATE)["cpu_backups"]) == ["a"]


def test_pinned_backup_uses_pinned_allocator(monkeypatch: pytest.MonkeyPatch) -> None:
    original_empty_like, pin_memory_flags = torch.empty_like, []

    def allocate(tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        pin_memory_flags.append(kwargs["pin_memory"])
        return original_empty_like(tensor, **{**kwargs, "pin_memory": False})

    monkeypatch.setattr(torch, "empty_like", allocate)
    backend._update_worker_weights(TinyModel(), weight_command("backup", "a", reset=True, source="pinned"))
    assert pin_memory_flags and all(pin_memory_flags)


class LocalClient(backend.ModelWeightClient):
    def __init__(self) -> None:
        super().__init__("http://unused", 1)
        self.model, self.events = TinyModel(), []
        self.current = "a"

    def _control(self, operation: Literal["backup", "restore"], model: str, reset: bool = False) -> None:
        self.events.append(operation)
        backend._update_worker_weights(self.model, weight_command(operation, model, reset, self.source))

    def _load_from_disk(self, model: str) -> None:
        self.events.append("disk:" + model)
        with torch.no_grad():
            self.model.weight.fill_(1 if model == "a" else 2)
            self.model.derived.fill_(1 if model == "a" else 2)
        self.current = model


@pytest.mark.parametrize("source", ["disk", "pageable"])
def test_prepare_and_switch_without_probe_warm_commit_or_duplicate_initial_load(source: backend.WeightSource) -> None:
    client = LocalClient()
    result = client.prepare({"a": "/a", "b": "/b"}, source, "a")
    assert result["model"] == "a" and len(result["backups"]) == 2
    events_before_restore = list(client.events)
    assert not client.restore("a")["changed"]
    assert client.events == events_before_restore
    client.restore("b")
    assert torch.all(client.model.weight == 2)
    if source == "disk":
        assert client.events == ["disk:b", "disk:a", "disk:b"]
        assert not hasattr(client.model, backend._STATE)
    else:
        assert client.events == ["backup", "disk:b", "backup", "restore", "restore"]
    with pytest.raises(ValueError, match="restart"):
        client.prepare({"a": "/a", "b": "/b"}, "pinned", "a")


def test_config_check_needs_no_tokenizer_load(tmp_path: Path) -> None:
    models = {}
    for name in ("a", "b"):
        path = tmp_path / name
        path.mkdir()
        (path / "config.json").write_text(json.dumps({"hidden_size": 8, "_name_or_path": name}))
        models[name] = str(path)
    assert backend._validate_models(models) == models
    (tmp_path / "b" / "config.json").write_text(json.dumps({"hidden_size": 16}))
    with pytest.raises(ValueError, match="config mismatch"):
        backend._validate_models(models)


class FakeDelta:
    """CPU stand-in for the codec; the worker protocol owns model identity."""

    def __init__(self, cpu_base: dict[str, torch.Tensor], gpu_tensors: dict[str, torch.Tensor]) -> None:
        self.base = {name: tensor.clone() for name, tensor in cpu_base.items()}
        self.target = {name: tensor.detach().clone() for name, tensor in gpu_tensors.items()}
        self.gpu_tensors = gpu_tensors
        self.pointers = {name: tensor.data_ptr() for name, tensor in gpu_tensors.items()}
        self.reverse_calls: list[bool] = []
        self.stats = {"payload_bytes": 1}

    def validate_bindings(self) -> None:
        if any(tensor.data_ptr() != self.pointers[name] for name, tensor in self.gpu_tensors.items()):
            raise ValueError("Delta tensor binding changed")

    def apply(self, *, reverse: bool) -> None:
        self.reverse_calls.append(reverse)
        backup = self.base if reverse else self.target
        with torch.no_grad():
            for name, tensor in self.gpu_tensors.items():
                tensor.copy_(backup[name])


@pytest.fixture
def delta_packages(monkeypatch: pytest.MonkeyPatch) -> list[FakeDelta]:
    packages: list[FakeDelta] = []

    def build_delta(cpu_base: dict[str, torch.Tensor], gpu_tensors: dict[str, torch.Tensor]) -> FakeDelta:
        package = FakeDelta(cpu_base, gpu_tensors)
        packages.append(package)
        return package

    module = ModuleType("warpshift.delta")
    module.DeltaWeights = build_delta
    monkeypatch.setitem(sys.modules, "warpshift.delta", module)
    tp_group = object()

    def get_tp_group() -> SimpleNamespace:
        return SimpleNamespace(cpu_group=tp_group)

    def get_rank(group: object) -> int:
        assert group is tp_group
        return 0

    def get_world_size(group: object) -> int:
        assert group is tp_group
        return 1

    def all_gather_object(results: list[Any], local: Any, group: object) -> None:
        assert group is tp_group
        results[:] = [local]

    distributed = ModuleType("sglang.srt.distributed")
    distributed.get_tp_group = get_tp_group
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed", distributed)
    monkeypatch.setattr(torch.distributed, "get_rank", get_rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", get_world_size)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    return packages


def prepare_delta_pair(model: TinyModel) -> None:
    command = weight_command("backup", "a", reset=True, source="delta")
    backend.model_weight_loader(model, [(json.dumps(command), torch.empty(0, dtype=torch.uint8))])
    with torch.no_grad():
        model.weight.add_(10)
        model.A_log.add_(1)
        model.derived.add_(2)
    command = weight_command("backup", "b", source="delta")
    backend.model_weight_loader(model, [(json.dumps(command), torch.empty(0, dtype=torch.uint8))])


def test_delta_prepare_builds_one_package_releases_backups_and_keeps_b(delta_packages: list[FakeDelta]) -> None:
    model = TinyModel()
    original = {name: tensor.detach().clone() for name, tensor in backend._collect_tensors(model).items()}
    prepare_delta_pair(model)
    state = getattr(model, backend._STATE)
    assert len(delta_packages) == 1
    package = delta_packages[0]
    assert state["delta"] is package
    assert state["delta_models"] == ("a", "b") and state["current"] == "b"
    assert state["cpu_backups"] == {} and package.reverse_calls == []
    for name, tensor in state["gpu_tensors"].items():
        assert torch.equal(package.base[name], original[name])
        assert torch.equal(tensor, package.target[name])
    assert torch.equal(model.weight, torch.arange(8, dtype=torch.bfloat16).reshape(2, 4) + 10)


@pytest.mark.parametrize(
    ("from_model", "target", "message"),
    [(None, "a", "source model"), ("a", "a", "source model"), ("b", "b", "other model"), ("b", "c", "other model")],
)
def test_delta_restore_requires_current_source_and_other_target(
    delta_packages: list[FakeDelta], from_model: str | None, target: str, message: str
) -> None:
    model = TinyModel()
    prepare_delta_pair(model)
    with pytest.raises(ValueError, match=message):
        backend._validate_delta_restore(
            model, weight_command("restore", target, source="delta", from_model=from_model)
        )
    assert delta_packages[0].reverse_calls == []


@pytest.mark.parametrize("replacement", ["storage", "parameter"])
def test_delta_restore_rejects_changed_bindings(delta_packages: list[FakeDelta], replacement: str) -> None:
    model = TinyModel()
    prepare_delta_pair(model)
    if replacement == "storage":
        model.weight.data = model.weight.detach().clone()
    else:
        model.weight = torch.nn.Parameter(model.weight.detach().clone())
    with pytest.raises(ValueError, match="(?i)(binding|storage|tensor)"):
        backend._validate_delta_restore(model, weight_command("restore", "a", source="delta", from_model="b"))
    assert delta_packages[0].reverse_calls == []


def test_delta_same_model_restore_does_not_send_a_worker_command() -> None:
    client = LocalClient()
    client.source = "delta"
    assert client.restore("a")["changed"] is False
    assert client.events == []


@pytest.mark.parametrize("model_count", [0, 1, 3])
def test_delta_prepare_rejects_model_counts_other_than_two(model_count: int) -> None:
    client = LocalClient()
    with pytest.raises(ValueError, match="exactly two"):
        client.prepare({str(index): f"/model/{index}" for index in range(model_count)}, "delta", "a")
    assert client.events == [] and client.source is None


@pytest.mark.parametrize("outcome", ["preflight_error", "different_commands", "apply_error", "success"])
def test_tp4_agrees_before_writes_and_commits_only_after_all_ranks_succeed(
    monkeypatch: pytest.MonkeyPatch, delta_packages: list[FakeDelta], outcome: str
) -> None:
    model = TinyModel()
    prepare_delta_pair(model)
    command = weight_command("restore", "a", source="delta", from_model="b")
    tp_group = object()
    gathers: list[Any] = []
    updates: list[dict[str, Any]] = []

    def get_tp_group() -> SimpleNamespace:
        return SimpleNamespace(cpu_group=tp_group)

    def get_rank(group: object) -> int:
        assert group is tp_group
        return 0

    def get_world_size(group: object) -> int:
        assert group is tp_group
        return 4

    def all_gather_object(results: list[Any], local: Any, group: object) -> None:
        assert group is tp_group and len(results) == 4
        gathers.append(local)
        results[:] = [local] * 4
        if len(gathers) == 1:
            if outcome == "preflight_error":
                results[3] = (command, "rank 3: ValueError: stale source")
            elif outcome == "different_commands":
                results[2] = ({**command, "model": "b"}, None)
        elif outcome == "apply_error":
            results[2] = "rank 2: RuntimeError: decode failed"

    update_worker = backend._update_worker_weights

    def update_weights(target_model: torch.nn.Module, received_command: dict[str, Any]) -> None:
        updates.append(received_command)
        update_worker(target_model, received_command)
        assert getattr(target_model, backend._STATE)["current"] == "b"

    distributed = ModuleType("sglang.srt.distributed")
    distributed.get_tp_group = get_tp_group
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed", distributed)
    monkeypatch.setattr(torch.distributed, "get_rank", get_rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", get_world_size)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(backend, "_update_worker_weights", update_weights)
    payload = [(json.dumps(command), torch.empty(0, dtype=torch.uint8))]
    if outcome == "success":
        backend.model_weight_loader(model, payload)
    else:
        message = {
            "preflight_error": "rank 3",
            "different_commands": "different weight commands",
            "apply_error": "rank 2",
        }
        with pytest.raises(RuntimeError, match=message[outcome]):
            backend.model_weight_loader(model, payload)
    assert updates == ([] if outcome in {"preflight_error", "different_commands"} else [command])
    assert len(gathers) == (1 if not updates else 2)
    assert getattr(model, backend._STATE)["current"] == ("a" if outcome == "success" else "b")
    assert delta_packages[0].reverse_calls == ([] if not updates else [True])
