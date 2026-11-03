"""CPU contracts for the SGLang TP disk-update completion patch."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch.distributed as dist
from sglang.srt import platforms
from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqInput
from sglang.srt.managers.scheduler_components.weight_updater import SchedulerWeightUpdaterManager


@pytest.mark.parametrize(
    ("tp_size", "failure"),
    [(4, None), (4, "remote"), (4, "load"), (4, "load_result"), (4, "flush"), (4, "sync"), (1, None)],
)
def test_disk_update_reports_all_tp_ranks_after_completion(
    monkeypatch: pytest.MonkeyPatch, tp_size: int, failure: str | None
) -> None:
    events: list[str] = []
    tp_group = object()

    def load_weights(request: UpdateWeightFromDiskReqInput) -> tuple[bool, str]:
        events.append("load")
        assert request.model_path == "/local/Teacher2"
        if failure == "load":
            raise RuntimeError("load failed")
        return (False, "load failed") if failure == "load_result" else (True, "loaded")

    def flush_cache(*, empty_cache: bool) -> bool:
        events.append("flush")
        if failure == "flush":
            raise RuntimeError("flush failed")
        return True

    def synchronize() -> None:
        events.append("sync")
        if failure == "sync":
            raise RuntimeError("sync failed")

    def gather_results(
        results: list[tuple[bool, str] | None], local_result: tuple[bool, str], *, group: object
    ) -> None:
        events.append("gather")
        assert group is tp_group and len(results) == tp_size
        results[:] = [local_result] + [(True, "loaded")] * (tp_size - 1)
        if failure == "remote":
            results[3] = (False, "remote load failed")

    gather = Mock(side_effect=gather_results)
    monkeypatch.setattr(dist, "get_world_size", Mock(return_value=tp_size))
    monkeypatch.setattr(dist, "all_gather_object", gather)
    monkeypatch.setattr(platforms, "current_platform", SimpleNamespace(synchronize=synchronize))
    updater = SchedulerWeightUpdaterManager(
        tp_worker=SimpleNamespace(update_weights_from_disk=load_weights),
        draft_worker=None,
        tp_cpu_group=tp_group,
        memory_saver_adapter=None,
        flush_cache=flush_cache,
        is_fully_idle=lambda: True,
    )
    request = UpdateWeightFromDiskReqInput(model_path="/local/Teacher2", flush_cache=True)
    response = updater.update_weights_from_disk(request)

    assert response.success == (failure is None)
    assert response.num_paused_requests == 0
    if tp_size == 1:
        gather.assert_not_called()
        assert events == ["load", "flush", "sync"]
    else:
        gather.assert_called_once()
        assert events[-1] == "gather"
        if failure in (None, "remote", "sync"):
            assert events == ["load", "flush", "sync", "gather"]
        elif failure == "load":
            assert events == ["load", "gather"]
        elif failure == "load_result":
            assert events == ["load", "sync", "gather"]
        else:
            assert events == ["load", "flush", "gather"]
    if failure is not None:
        failed_rank = 3 if failure == "remote" else 0
        assert f"TP rank {failed_rank}:" in response.message
        assert "failed" in response.message
