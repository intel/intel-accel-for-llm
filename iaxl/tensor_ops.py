# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import torch

from . import torch_ext as _iqt
from .envs import envs
from .torch_ext import Context, GpuTransferDirection

__all__ = ["SliceCopier", "copy_slices"]


class SliceCopier:
    """Reusable IAXL copy: gpu.select(dim, index[i]) <-> cpu[cpu_index[i]].

    Exactly one of `src` / `out` is a CUDA tensor; the other is a CPU tensor whose
    dim 0 is the block dimension. Both are registered as KV pools for the lifetime
    of the copier; call close() (or let it be garbage collected) to unregister.
    """

    def __init__(
        self,
        src: torch.Tensor,
        dim: int,
        index: Sequence[int] | torch.Tensor,
        *,
        out: torch.Tensor,
        cpu_index: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        if src.is_cuda and not out.is_cuda:
            gpu, cpu, direction = src, out, GpuTransferDirection.D2H
        elif out.is_cuda and not src.is_cuda:
            gpu, cpu, direction = out, src, GpuTransferDirection.H2D
        else:
            raise TypeError("exactly one of src and out must be a CUDA tensor")

        index = index.tolist() if isinstance(index, torch.Tensor) else list(index)
        if cpu_index is None:
            cpu_index = list(range(len(index)))
        elif isinstance(cpu_index, torch.Tensor):
            cpu_index = cpu_index.tolist()
        else:
            cpu_index = list(cpu_index)
        if len(index) != len(cpu_index):
            raise ValueError("index and cpu_index must have the same length")

        if dim < -gpu.ndim or dim >= gpu.ndim:
            raise IndexError("dimension out of range")
        if dim < 0:
            dim += gpu.ndim
        if not gpu.is_contiguous() or not cpu.is_contiguous():
            raise ValueError("src and out must be contiguous")
        if cpu.dtype != gpu.dtype or cpu.shape[1:] != gpu.shape[:dim] + gpu.shape[dim + 1 :]:
            raise ValueError("cpu rows must match the CUDA tensor slice")

        self._index, self._cpu_index = index, cpu_index
        self._gpu_name = f"slice.gpu.{gpu.data_ptr():#x}"
        self._cpu_name = f"slice.cpu.{cpu.data_ptr():#x}"
        _iqt.register_kv_pool_gpu(self._gpu_name, gpu, dim)
        _iqt.register_kv_pool_cpu(self._cpu_name, cpu)
        self._work_stream = torch.cuda.Stream(device=gpu.device)
        backend = "dsa" if envs.IAXL_DSA_GD_ENABLE else "cuda"
        self._context = Context.create(
            self._gpu_name, self._cpu_name, direction, f"copy_slices_{backend}", self._work_stream
        )

    def copy(self) -> None:
        if not self._index:
            return
        self._context.xfer_chunks_batch(self._index, self._cpu_index)
        self._context.xfer_finish()
        self._context.xfer_wait()

    def close(self) -> None:
        if self._context is None:
            return
        self._context = None
        _iqt.unregister_kv_pool(self._gpu_name)
        _iqt.unregister_kv_pool(self._cpu_name)

    def __del__(self) -> None:
        if getattr(self, "_context", None) is not None:
            self.close()


def copy_slices(
    src: torch.Tensor,
    dim: int,
    index: Sequence[int] | torch.Tensor,
    *,
    out: torch.Tensor,
    cpu_index: Sequence[int] | torch.Tensor | None = None,
) -> None:
    """Copy gpu.select(dim, index[i]) <-> cpu[cpu_index[i]] using IAXL (see SliceCopier)."""
    copier = SliceCopier(src, dim, index, out=out, cpu_index=cpu_index)
    try:
        copier.copy()
    finally:
        copier.close()
