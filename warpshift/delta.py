# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Reversible U16/XOR residuals in pinned memory, with bounded GPU decoding.

Adapted from the validated ADP U16 transform and batched executor. Preparation
chooses the smallest representation per chunk. No floating-point subtraction,
checkpoint format, extra RPC, or persistent full-weight CPU backup is needed.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

import numpy as np
import torch
import triton
import triton.language as tl

_CHUNK_BYTES = 16 * 1024**2
_BATCH_BYTES = 64 * 1024**2
_TILE_BYTES = 4096
_MODES = {0: "XOR_RAW", 1: "XOR_GROUP2_ANS", 2: "XOR_ANS", 3: "U16_ANS"}


def _u16_residual(base: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Grouped low/high bytes of zigzag((B - A) mod 65536), without float math."""
    base_words, target_words = (
        base.view("<u2").astype(np.int32),
        target.view("<u2").astype(np.int32),
    )  # uint16, 0-65535
    delta = ((target_words - base_words + 32768) & 65535) - 32768
    zigzag = ((delta << 1) ^ (delta >> 15)).astype("<u2")
    return zigzag.view(np.uint8).reshape(-1, 2).T.copy().reshape(-1)


def _batch_layout(sizes: list[tuple[int, int]], limit: int = _BATCH_BYTES) -> list[dict[str, Any]]:
    """Bound both encoded and decoded arenas; align independent nvCOMP frames."""
    batches, packets = [], []
    used_input = used_output = payload_offset = 0
    for index, (encoded, decoded) in enumerate(sizes):
        if min(encoded, decoded) < 0 or encoded == 0 or max(encoded, decoded) > limit:
            raise ValueError("Invalid delta frame size")
        input_offset, output_offset = (used_input + 255) // 256 * 256, (used_output + 255) // 256 * 256
        if packets and max(input_offset + encoded, output_offset + decoded) > limit:
            batches.append(
                dict(payload_offset=payload_offset, input_bytes=used_input, decoded_bytes=used_output, packets=packets)
            )
            payload_offset += used_input
            packets, input_offset, output_offset = [], 0, 0
        packets.append((index, input_offset, output_offset))
        used_input, used_output = input_offset + encoded, output_offset + decoded
    if packets:
        batches.append(
            dict(payload_offset=payload_offset, input_bytes=used_input, decoded_bytes=used_output, packets=packets)
        )
    return batches


@triton.jit
def _restore_batch(
    input_payload, decoded_residuals, descriptors, tile_packets, REVERSE: tl.constexpr, BLOCK: tl.constexpr
):
    # Input payload holds ANS frames or raw XOR bytes; only ANS frames are decoded.
    # One packet descriptor plus one int32 packet index per tile.
    tile_index = tl.program_id(0)
    descriptor_offset = tl.load(tile_packets + tile_index).to(tl.int64) * 5
    mode = tl.load(descriptors + descriptor_offset)
    destination = tl.load(descriptors + descriptor_offset + 1).to(tl.pointer_type(tl.uint8))
    num_bytes = tl.load(descriptors + descriptor_offset + 2)
    residual_offset = tl.load(descriptors + descriptor_offset + 3)
    byte_offset = (tile_index - tl.load(descriptors + descriptor_offset + 4)) * BLOCK
    residual_source = input_payload + residual_offset if mode == 0 else decoded_residuals + residual_offset
    if mode == 3:
        # One lane owns the whole uint16, avoiding read/write races between bytes.
        word_index = byte_offset // 2 + tl.arange(0, BLOCK // 2)
        word_valid = word_index < num_bytes // 2
        low_byte = tl.load(residual_source + word_index, word_valid, other=0).to(tl.int32)
        high_byte = tl.load(residual_source + num_bytes // 2 + word_index, word_valid, other=0).to(tl.int32)
        zigzag = low_byte | (high_byte << 8)
        word_delta = (zigzag >> 1) ^ -(zigzag & 1)
        destination_words = destination.to(tl.pointer_type(tl.uint16))
        old_word = tl.load(destination_words + word_index, word_valid, other=0).to(tl.int32)
        restored_word = old_word - word_delta if REVERSE else old_word + word_delta
        tl.store(destination_words + word_index, restored_word.to(tl.uint16), word_valid)
    else:
        byte_index = byte_offset + tl.arange(0, BLOCK)
        byte_valid = byte_index < num_bytes
        residual_index = byte_index // 2 + (byte_index % 2) * (num_bytes // 2) if mode == 1 else byte_index
        byte_delta = tl.load(residual_source + residual_index, byte_valid, other=0)
        old_byte = tl.load(destination + byte_index, byte_valid, other=0)
        tl.store(destination + byte_index, old_byte ^ byte_delta, byte_valid)


class DeltaWeights:
    """One rank's A↔B package; caller owns model identity and update serialization.

    cpu_base contains A and gpu_target currently contains B. Construction never
    changes GPU weights. The complete CPU A copy is not retained by this object.
    """

    def __init__(self, cpu_base: dict[str, torch.Tensor], gpu_target: dict[str, torch.Tensor]) -> None:
        started = time.perf_counter()
        self._bind_tensors(cpu_base, gpu_target)
        from nvidia import nvcomp

        self.nvcomp = nvcomp
        self.compute_stream = torch.cuda.current_stream(self.device)
        self.copy_stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.device(self.device), torch.cuda.stream(self.compute_stream):
            self.codec = nvcomp.Codec(
                algorithm="ANS", cuda_stream=self.compute_stream.cuda_stream, uncomp_chunk_size=65536
            )
            packets, counts, skipped_bytes = self._encode_packets(cpu_base, gpu_target)
            self._prepare_batches(packets)
        self.compute_stream.synchronize()
        self.stats = {
            "codec": "U16_XOR_ANS",
            "reversible": True,
            "chunk_bytes": _CHUNK_BYTES,
            "batch_bytes": _BATCH_BYTES,
            "runtime_bytes": sum(t.numel() * t.element_size() for t in gpu_target.values()),
            "skipped_bytes": skipped_bytes,
            "payload_bytes": sum(p["payload"].numel() for p in packets),
            "transferred_bytes": self.pinned_payload.numel(),
            "pinned_bytes": self.pinned_payload.numel(),
            "mode_counts": dict(counts),
            "batches": len(self.batches),
            "gpu_scratch_bytes": sum(t.numel() for t in self.gpu_input_buffers) + self.gpu_decoded_buffer.numel(),
            "gpu_metadata_bytes": sum(
                batch["descriptors"].numel() * 8 + batch["tile_packets"].numel() * 4 for batch in self.batches
            ),
            "memory_scope": "Buffers/descriptors only; excludes nvCOMP workspace and allocator caching",
            "prepare_seconds": time.perf_counter() - started,
        }

    def _bind_tensors(self, cpu_base: dict[str, torch.Tensor], gpu_target: dict[str, torch.Tensor]) -> None:
        """Validate the A/B layout and retain GPU storage bindings for later restores."""
        if not cpu_base or cpu_base.keys() != gpu_target.keys():
            raise ValueError("Delta tensor coverage differs")
        self.bindings = []
        spans = []
        devices = {tensor.device for tensor in gpu_target.values()}
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            raise ValueError("Delta tensors must reside on one CUDA device")
        self.device = next(iter(devices))
        for name, target in gpu_target.items():
            base = cpu_base[name]
            if base.device.type != "cpu" or base.shape != target.shape or base.dtype != target.dtype:
                raise ValueError(f"Delta layout differs: {name}")
            if not base.is_contiguous() or not target.is_contiguous():
                raise ValueError(f"Delta requires contiguous tensors: {name}")
            size = target.numel() * target.element_size()
            if size:
                spans.append((target.data_ptr(), target.data_ptr() + size, name))
            self.bindings.append(
                (target, target.data_ptr(), tuple(target.shape), tuple(target.stride()), target.dtype)
            )
        spans.sort()
        for previous, current in zip(spans, spans[1:]):
            if current[0] < previous[1]:
                raise ValueError(f"Delta cannot update overlapping tensors: {previous[2]}, {current[2]}")

    def _compress(self, data: np.ndarray, configs: dict[int, Any]) -> torch.Tensor:
        gpu_source = torch.from_numpy(data).to(self.device)
        num_bytes = gpu_source.numel()
        if num_bytes not in configs:
            configs[num_bytes] = self.codec.compression_config(num_bytes)
        encoded = self.codec.encode(self.nvcomp.as_array(gpu_source), compression_config=configs[num_bytes])
        # Keep encoded alive until the synchronous CPU copy; its DLPack view
        # alone does not retain nvCOMP's allocation owner.
        cpu_encoded = torch.from_dlpack(encoded).view(torch.uint8)[: encoded.buffer_size].cpu()
        self.compute_stream.synchronize()
        return cpu_encoded

    def _encode_chunk(
        self, base_chunk: np.ndarray, target_chunk: np.ndarray, is_bf16: bool, configs: dict[int, Any]
    ) -> tuple[int, torch.Tensor] | None:
        """Return the smallest reversible encoding, or None for an unchanged chunk."""
        xor = np.bitwise_xor(base_chunk, target_chunk)
        if not np.any(xor):
            return None
        # Default to uncompressed raw XOR; replace only when compression is smaller.
        mode, payload = 0, torch.from_numpy(xor)
        if xor.size >= 16384:
            grouped = xor.reshape(-1, 2).T.copy().reshape(-1) if xor.size % 2 == 0 else xor
            candidate = self._compress(grouped, configs)
            if candidate.numel() < payload.numel():
                mode, payload = (1 if xor.size % 2 == 0 else 2), candidate
            if is_bf16:
                candidate = self._compress(_u16_residual(base_chunk, target_chunk), configs)
                if candidate.numel() < payload.numel():
                    mode, payload = 3, candidate
        return mode, payload

    def _encode_packets(
        self, cpu_base: dict[str, torch.Tensor], gpu_target: dict[str, torch.Tensor]
    ) -> tuple[list[dict[str, Any]], Counter[str], int]:
        packets, counts, configs = [], Counter(), {}
        skipped_bytes = 0
        for name, target in gpu_target.items():
            base_bytes = cpu_base[name].detach().reshape(-1).view(torch.uint8)
            target_bytes = target.detach().reshape(-1).view(torch.uint8)
            for offset in range(0, target_bytes.numel(), _CHUNK_BYTES):
                gpu_chunk = target_bytes[offset : offset + _CHUNK_BYTES]
                base_chunk = base_bytes[offset : offset + gpu_chunk.numel()].numpy()
                target_chunk = gpu_chunk.cpu().numpy()
                encoded_chunk = self._encode_chunk(base_chunk, target_chunk, target.dtype == torch.bfloat16, configs)
                if encoded_chunk is None:
                    counts["SKIP"] += 1
                    skipped_bytes += gpu_chunk.numel()
                    continue
                mode, payload = encoded_chunk
                packets.append(dict(mode=mode, payload=payload, dst=gpu_chunk.data_ptr(), num_bytes=gpu_chunk.numel()))
                counts[_MODES[mode]] += 1
        return packets, counts, skipped_bytes

    def _prepare_batches(self, packets: list[dict[str, Any]]) -> None:
        """Allocate shared CPU/GPU buffers, then prepare each bounded batch."""
        # (传输字节数, 解压字节数)；裸 XOR 直接读取输入缓冲，不占解压缓冲。
        sizes = [(packet["payload"].numel(), packet["num_bytes"] if packet["mode"] else 0) for packet in packets]
        self.batches = _batch_layout(sizes)
        self.pinned_payload = torch.zeros(
            sum(batch["input_bytes"] for batch in self.batches), dtype=torch.uint8, device="cpu", pin_memory=True
        )
        # _batch_layout 按 _BATCH_BYTES（64 MiB）同时限制输入和解压容量。
        input_capacity = max((batch["input_bytes"] for batch in self.batches), default=1)
        decoded_capacity = max(1, max((batch["decoded_bytes"] for batch in self.batches), default=0))
        self.gpu_input_buffers = [
            torch.empty(input_capacity, dtype=torch.uint8, device=self.device)
            for _ in range(min(2, max(1, len(self.batches))))
        ]
        self.gpu_decoded_buffer = torch.empty(decoded_capacity, dtype=torch.uint8, device=self.device)
        self.copy_done = [torch.cuda.Event() for _ in self.gpu_input_buffers]
        self.compute_done = [torch.cuda.Event() for _ in self.gpu_input_buffers]
        for batch in self.batches:
            self._pack_batch(batch, packets)

    def _pack_batch(self, batch: dict[str, Any], packets: list[dict[str, Any]]) -> None:
        """Copy a batch into pinned memory and prepare its GPU descriptors and decoder."""
        rows, tile_packets = [], []
        input_offsets, input_sizes, output_offsets, output_sizes = [], [], [], []
        payload_offset = batch["payload_offset"]
        for row_index, (packet_index, input_offset, output_offset) in enumerate(batch.pop("packets")):
            packet = packets[packet_index]
            payload, mode, num_bytes = packet["payload"], packet["mode"], packet["num_bytes"]
            start = payload_offset + input_offset
            self.pinned_payload[start : start + payload.numel()].copy_(payload)
            # 压缩帧读取解压缓冲，裸 XOR 直接读取输入缓冲。
            residual_offset = output_offset if mode else input_offset
            first_tile = len(tile_packets)
            num_tiles = triton.cdiv(num_bytes, _TILE_BYTES)
            rows.append((mode, packet["dst"], num_bytes, residual_offset, first_tile))
            tile_packets.extend([row_index] * num_tiles)
            if mode:
                input_offsets.append(input_offset)
                input_sizes.append(payload.numel())
                output_offsets.append(output_offset)
                output_sizes.append(num_bytes)
        batch["descriptors"] = torch.tensor(rows, dtype=torch.int64, device=self.device)
        batch["tile_packets"] = torch.tensor(tile_packets, dtype=torch.int32, device=self.device)
        batch["tiles"] = len(tile_packets)
        batch["config"] = None
        if input_offsets:
            # 两个输入缓冲使用相同的压缩帧偏移和长度。
            batch["inputs"] = [
                self.nvcomp.as_arrays(
                    gpu_input_buffer, input_offsets, input_sizes, cuda_stream=self.compute_stream.cuda_stream
                )
                for gpu_input_buffer in self.gpu_input_buffers
            ]
            batch["outputs"] = self.nvcomp.as_arrays(
                self.gpu_decoded_buffer, output_offsets, output_sizes, cuda_stream=self.compute_stream.cuda_stream
            )
            # 准备时复制一次压缩帧，让 nvCOMP 读取帧头并缓存解压配置。
            input_bytes = batch["input_bytes"]
            self.gpu_input_buffers[0][:input_bytes].copy_(
                self.pinned_payload[payload_offset : payload_offset + input_bytes], non_blocking=True
            )
            batch["config"] = self.codec.decompression_config(batch["inputs"][0])

    def validate_bindings(self) -> None:
        """Reject changed GPU storage before dereferencing prepared kernel pointers."""
        for tensor, pointer, shape, stride, dtype in self.bindings:
            if (
                tensor.data_ptr() != pointer
                or tuple(tensor.shape) != shape
                or tuple(tensor.stride()) != stride
                or tensor.dtype != dtype
                or tensor.device != self.device
            ):
                raise ValueError("Delta GPU tensor storage changed; prepare again")

    @torch.no_grad()
    def apply(self, reverse: bool) -> None:
        """A→B when false, B→A when true; return only after GPU completion."""
        self.validate_bindings()
        caller_stream = torch.cuda.current_stream(self.device)
        self.compute_stream.wait_stream(caller_stream)
        with torch.cuda.device(self.device), torch.cuda.stream(self.compute_stream):
            self.copy_stream.wait_stream(self.compute_stream)
            for index, batch in enumerate(self.batches):
                slot = index % len(self.gpu_input_buffers)
                payload_offset, input_bytes = batch["payload_offset"], batch["input_bytes"]
                with torch.cuda.stream(self.copy_stream):
                    if index >= len(self.gpu_input_buffers):
                        self.copy_stream.wait_event(self.compute_done[slot])
                    self.gpu_input_buffers[slot][:input_bytes].copy_(
                        self.pinned_payload[payload_offset : payload_offset + input_bytes], non_blocking=True
                    )
                    self.copy_done[slot].record(self.copy_stream)
                self.compute_stream.wait_event(self.copy_done[slot])
                if batch["config"] is not None:
                    self.codec.decode(
                        batch["inputs"][slot], out=batch["outputs"], decompression_config=batch["config"]
                    )
                _restore_batch[(batch["tiles"],)](
                    self.gpu_input_buffers[slot],
                    self.gpu_decoded_buffer,
                    batch["descriptors"],
                    batch["tile_packets"],
                    reverse,
                    _TILE_BYTES,
                )
                self.compute_done[slot].record(self.compute_stream)
            self.compute_stream.wait_stream(self.copy_stream)
        self.compute_stream.synchronize()
