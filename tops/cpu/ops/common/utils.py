"""Low-level utilities shared across CPU reference kernels."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax


def cdiv(x: int, y: int) -> int:
    """Ceiling division of x by y."""
    return (x + y - 1) // y


def pad_to_multiple(x: jnp.ndarray, multiple: int, axis: int) -> jnp.ndarray:
    """Zero-pad array along the given axis so its length becomes a multiple of `multiple`.

    Args:
        x: Input array to pad.
        multiple: Target multiple for the axis length.
        axis: Axis along which to pad.

    Returns:
        Padded array (or original if already aligned).
    """
    length = x.shape[axis]
    remainder = length % multiple
    if remainder == 0:
        return x
    pad_len = multiple - remainder
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[axis] = (0, pad_len)
    return jnp.pad(x, pad_widths)


def pad_varlen_seqs(
    tensors: list[jnp.ndarray],
    cu_seqlens: jnp.ndarray,
    chunk_size: int,
) -> tuple[list[jnp.ndarray], jnp.ndarray, list[int], list[int]]:
    """Pad each variable-length segment along dim=1 to a multiple of chunk_size.

    Works with tensors of any trailing dimensions (3D [B,T,H], 4D [B,T,H,K], etc.)
    since padding is only applied along axis=1.

    Args:
        tensors: list of [1, T_total, ...] arrays sharing the same sequence layout.
        cu_seqlens: [N+1] cumulative sequence lengths, e.g. [0, 10, 30].
        chunk_size: block size.

    Returns:
        (padded_tensors, new_cu_seqlens, orig_seqlens, padded_seqlens)
    """
    N = len(cu_seqlens) - 1
    orig_seqlens = [int(cu_seqlens[i + 1] - cu_seqlens[i]) for i in range(N)]
    padded_seqlens = [cdiv(L, chunk_size) * chunk_size for L in orig_seqlens]

    if orig_seqlens == padded_seqlens:
        return tensors, cu_seqlens, orig_seqlens, padded_seqlens

    padded = [[] for _ in tensors]
    for i in range(N):
        bos = int(cu_seqlens[i])
        L = orig_seqlens[i]
        pad = padded_seqlens[i] - L
        for j, t in enumerate(tensors):
            seg = t[:, bos:bos + L]
            if pad > 0:
                pad_widths = [(0, 0)] * seg.ndim
                pad_widths[1] = (0, pad)
                seg = jnp.pad(seg, pad_widths)
            padded[j].append(seg)

    padded_tensors = [jnp.concatenate(p, axis=1) for p in padded]
    offsets = [0]
    for pl in padded_seqlens:
        offsets.append(offsets[-1] + pl)
    new_cu_seqlens = jnp.array(offsets, dtype=jnp.int32)

    return padded_tensors, new_cu_seqlens, orig_seqlens, padded_seqlens


def unpad_varlen_seqs(
    tensor: jnp.ndarray,
    orig_seqlens: list[int],
    padded_seqlens: list[int],
) -> jnp.ndarray:
    """Remove per-segment padding from a variable-length tensor along dim=1.

    Args:
        tensor: [1, T_padded_total, ...] padded tensor.
        orig_seqlens: original segment lengths.
        padded_seqlens: padded segment lengths.

    Returns:
        [1, T_total, ...] tensor with padding removed.
    """
    parts = []
    offset = 0
    for L, PL in zip(orig_seqlens, padded_seqlens):
        parts.append(tensor[:, offset:offset + L])
        offset += PL
    return jnp.concatenate(parts, axis=1)


def acc_dtype(input_dtype) -> jnp.dtype:
    """Accumulator dtype: fp64 for fp64 inputs, fp32 otherwise."""
    return jnp.float64 if input_dtype == jnp.float64 else jnp.float32


def dot(subscripts: str, a: jnp.ndarray, b: jnp.ndarray, acc: jnp.dtype) -> jnp.ndarray:
    """Einsum simulating Triton's tl.dot with fp32 accumulation.

    XLA CPU's DotThunk does not support bf16×bf16→fp32 for certain einsum
    contraction patterns with asymmetric non-contracting dims (e.g.
    nchk,nhkv->nchv). Pre-casting half-precision inputs to fp32 is
    numerically equivalent to Triton tensor core behavior:
    bf16→fp32 cast is exact, and the product of two bf16 values fits in fp32
    (8+8=16 < 24 mantissa bits). fp16 is also covered (11+11=22 < 24).
    """
    if a.dtype in (jnp.bfloat16, jnp.float16) or b.dtype in (jnp.bfloat16, jnp.float16):
        a, b = a.astype(acc), b.astype(acc)
    return jnp.einsum(subscripts, a, b,
                      precision=lax.Precision.HIGHEST,
                      preferred_element_type=acc)
