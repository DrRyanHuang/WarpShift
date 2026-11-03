"""CPU invariants for reversible integer residuals and bounded GPU batches."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from warpshift.delta import DeltaWeights, _batch_layout, _u16_residual


@pytest.mark.parametrize("base_word", [0, 0x3F80, 0x8000, 0xFFFF])
def test_u16_residual_round_trip_covers_every_bit_pattern(base_word: int) -> None:
    # Covers signed zero, infinities and every NaN payload without float conversion.
    target = np.arange(65536, dtype="<u2")
    base = np.full(target.shape, base_word, dtype="<u2")
    grouped = _u16_residual(base.view(np.uint8), target.view(np.uint8))
    words = grouped.reshape(2, -1).T.copy().reshape(-1).view("<u2").astype(np.int32)
    signed = np.where(words % 2, -(words // 2) - 1, words // 2)
    restored_target = ((base.astype(np.int32) + signed) % 65536).astype("<u2")
    restored_base = ((target.astype(np.int32) - signed) % 65536).astype("<u2")
    np.testing.assert_array_equal(restored_target, target)
    np.testing.assert_array_equal(restored_base, base)
    assert grouped.size == base.nbytes


def test_u16_residual_known_values_and_byte_grouping() -> None:
    base = np.array([0, 0, 0xFFFF, 0, 0x8000], dtype="<u2")
    target = np.array([0, 1, 0, 0xFFFF, 0], dtype="<u2")
    # Zigzag residual words: 0, +1, +1 (wrap), -1, -32768 (half-range).
    expected = np.array([0, 2, 2, 1, 255, 0, 0, 0, 0, 255], dtype=np.uint8)
    np.testing.assert_array_equal(_u16_residual(base.view(np.uint8), target.view(np.uint8)), expected)


def test_batch_layout_bounds_both_arenas_and_preserves_packet_order() -> None:
    sizes = [(300, 900), (300, 900), (700, 0), (400, 500), (1, 0)]
    batches = _batch_layout(sizes, limit=2048)
    indices = []
    expected_payload_offset = 0
    for batch in batches:
        assert batch["payload_offset"] == expected_payload_offset
        expected_payload_offset += batch["input_bytes"]
        assert 0 < batch["input_bytes"] <= 2048
        assert 0 <= batch["decoded_bytes"] <= 2048
        input_end = output_end = 0
        for index, input_offset, output_offset in batch["packets"]:
            encoded, decoded = sizes[index]
            assert input_offset % 256 == output_offset % 256 == 0
            assert input_offset >= input_end and output_offset >= output_end
            input_end, output_end = input_offset + encoded, output_offset + decoded
            assert input_end <= batch["input_bytes"] and output_end <= batch["decoded_bytes"]
            indices.append(index)
    assert indices == list(range(len(sizes)))
    assert len(batches) > 1
    assert _batch_layout([]) == []


@pytest.mark.parametrize("size", [(0, 1), (-1, 1), (1, -1), (2049, 1), (1, 2049)])
def test_batch_layout_rejects_frames_that_cannot_fit(size: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="frame size"):
        _batch_layout([size], limit=2048)


def test_bindings_reject_changed_storage_without_requiring_cuda() -> None:
    tensor = torch.arange(8, dtype=torch.float32)
    delta = DeltaWeights.__new__(DeltaWeights)
    delta.device = tensor.device
    delta.bindings = [(tensor, tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)]
    delta.validate_bindings()
    tensor.data = tensor.clone()
    with pytest.raises(ValueError, match="storage changed"):
        delta.validate_bindings()
