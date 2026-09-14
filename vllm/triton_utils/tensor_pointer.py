# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor view bounds for Triton kernels with view-bounded memory accesses."""

import torch

_MAX_POINTER_RANGE = 2**31 - 1


class TensorView:
    """Keep the tensor's address and lifetime, reporting its strided byte span.

    Only use for kernels whose loads and stores are bounded by the tensor view.
    Triton HIP uses ptr_range() instead of the size of the backing allocation.
    """

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)

    def ptr_range(self) -> int:
        tensor = self.tensor
        if not tensor.numel():
            return 0
        span = 1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
        return span * tensor.element_size()


def bounded_tensor_view(tensor: torch.Tensor) -> torch.Tensor | TensorView:
    """Expose small view bounds when a large allocation hides them from Triton.

    Small allocations, genuinely large views and compile-only mock tensors
    keep their existing specialization. No allocation or tensor copy occurs.
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.untyped_storage().nbytes() <= _MAX_POINTER_RANGE:
        return tensor
    view = TensorView(tensor)
    if not tensor.numel() or view.ptr_range() > _MAX_POINTER_RANGE:
        return tensor
    return view
