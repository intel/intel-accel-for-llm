# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import torch

from .envs import envs
from .torch_ext import Context, GpuTransferDirection

__all__ = ["SliceCopier", "copy_slices"]


class SliceCopier:
    """Reusable IAXL copy between one GPU tensor and CPU tensor slices."""

    def __init__(
        self,
        src: torch.Tensor | Sequence[torch.Tensor],
        dim: int,
        index: Sequence[int] | torch.Tensor,
        *,
        out: torch.Tensor | Sequence[torch.Tensor],
    ) -> None:
        indices = index.tolist() if isinstance(index, torch.Tensor) else list(index)

        if isinstance(src, torch.Tensor) and src.is_cuda:
            gpu_tensor = src
            cpu_tensors = [out] if isinstance(out, torch.Tensor) else list(out)
            direction = GpuTransferDirection.D2H
        elif isinstance(out, torch.Tensor) and out.is_cuda:
            gpu_tensor = out
            cpu_tensors = [src] * len(indices) if isinstance(src, torch.Tensor) else list(src)
            direction = GpuTransferDirection.H2D
        else:
            raise TypeError("exactly one of src and out must be a CUDA tensor")

        if len(indices) != len(cpu_tensors):
            raise ValueError("index and CPU tensor counts must match")

        if dim < -gpu_tensor.ndim or dim >= gpu_tensor.ndim:
            raise IndexError("dimension out of range")
        if dim < 0:
            dim += gpu_tensor.ndim
        if not gpu_tensor.is_contiguous():
            raise ValueError("the CUDA tensor must be contiguous")
        if any(i < 0 or i >= gpu_tensor.shape[dim] for i in indices):
            raise IndexError("slice index out of range")

        slice_shape = gpu_tensor.shape[:dim] + gpu_tensor.shape[dim + 1 :]
        for tensor in cpu_tensors:
            if tensor.device.type != "cpu":
                raise ValueError("slice tensors must be on CPU")
            if tensor.dtype != gpu_tensor.dtype or tensor.shape != slice_shape:
                raise ValueError("slice tensors must match the CUDA tensor slice")
            if not tensor.is_contiguous():
                raise ValueError("slice tensors must be contiguous")

        self._indices = indices
        # Kept alive so the addresses in self._cpu_ptrs stay valid.
        self._cpu_tensors = cpu_tensors
        # Built once so copy() hands the C++ side ready-to-use arrays instead of Python lists.
        self._chunk_indices = torch.tensor(indices, dtype=torch.int64)
        self._cpu_ptrs = torch.tensor(
            [tensor.data_ptr() for tensor in cpu_tensors], dtype=torch.int64
        )
        self._work_stream = torch.cuda.Stream(device=gpu_tensor.device)
        if envs.IAXL_DSA_V1_ENABLE:
            create, backend = Context.create_dsa_v1, "dsa_v1"
        else:
            create, backend = Context.create, "dsa" if envs.IAXL_DSA_GD_ENABLE else "cuda"
        self._context = create(
            gpu_tensor, dim, direction, f"copy_slices_{backend}", self._work_stream
        )

    def copy(self) -> None:
        if not self._indices:
            return
        self._context.xfer_chunks_batch_fast(self._chunk_indices, self._cpu_ptrs)
        self._context.xfer_finish()
        self._context.xfer_wait()


def copy_slices(
    src: torch.Tensor | Sequence[torch.Tensor],
    dim: int,
    index: Sequence[int] | torch.Tensor,
    *,
    out: torch.Tensor | Sequence[torch.Tensor],
) -> None:
    """Copy slices between one GPU tensor and CPU tensors using IAXL.

    A single CPU source is copied to every selected GPU slice. A sequence
    supplies one CPU tensor per index. GPU-to-CPU copies require one output
    tensor per selected GPU slice.
    """
    SliceCopier(src, dim, index, out=out).copy()
