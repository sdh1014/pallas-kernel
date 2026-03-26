"""Tests for pad_varlen_seqs / unpad_varlen_seqs."""
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest
from tops.cpu.ops.common.utils import pad_varlen_seqs, unpad_varlen_seqs


def test_pad_no_padding_needed():
    """Chunk-aligned segments should return original tensors."""
    q = jnp.ones((1, 32, 4, 16))
    cu = jnp.array([0, 16, 32])
    padded, new_cu, orig, pad_lens = pad_varlen_seqs([q], cu, chunk_size=16)
    assert padded[0].shape == (1, 32, 4, 16)
    assert jnp.array_equal(new_cu, cu)
    assert orig == [16, 16]
    assert pad_lens == [16, 16]


def test_pad_unequal_segments():
    """Non-aligned segments get zero-padded per segment."""
    q = jnp.ones((1, 30, 2, 8))
    cu = jnp.array([0, 10, 30])
    padded, new_cu, orig, pad_lens = pad_varlen_seqs([q], cu, chunk_size=16)
    assert padded[0].shape == (1, 48, 2, 8)
    assert jnp.array_equal(new_cu, jnp.array([0, 16, 48]))
    assert orig == [10, 20]
    assert pad_lens == [16, 32]
    assert jnp.all(padded[0][0, 10:16] == 0)


def test_pad_3d_tensor():
    """Works for 3D [B, T, H] tensors (Simple GLA g)."""
    g = jnp.ones((1, 30, 4))
    cu = jnp.array([0, 10, 30])
    padded, new_cu, orig, pad_lens = pad_varlen_seqs([g], cu, chunk_size=16)
    assert padded[0].shape == (1, 48, 4)


def test_pad_multiple_tensors():
    """Pads all tensors identically."""
    q = jnp.ones((1, 30, 2, 8))
    k = jnp.ones((1, 30, 2, 8)) * 2
    cu = jnp.array([0, 10, 30])
    [pq, pk], new_cu, orig, pad_lens = pad_varlen_seqs([q, k], cu, chunk_size=16)
    assert pq.shape == pk.shape == (1, 48, 2, 8)
    assert jnp.all(pk[0, :10] == 2)


def test_unpad_roundtrip():
    """pad then unpad recovers original."""
    q = jax.random.normal(jax.random.PRNGKey(0), (1, 30, 2, 8))
    cu = jnp.array([0, 10, 30])
    [pq], new_cu, orig, pad_lens = pad_varlen_seqs([q], cu, chunk_size=16)
    recovered = unpad_varlen_seqs(pq, orig, pad_lens)
    assert recovered.shape == q.shape
    assert jnp.allclose(recovered, q)


def test_unpad_3d():
    """unpad works for 3D tensors."""
    g = jax.random.normal(jax.random.PRNGKey(1), (1, 30, 4))
    cu = jnp.array([0, 10, 30])
    [pg], _, orig, pad_lens = pad_varlen_seqs([g], cu, chunk_size=16)
    recovered = unpad_varlen_seqs(pg, orig, pad_lens)
    assert jnp.allclose(recovered, g)


def test_single_token_segment():
    """Segment of length 1 should pad to chunk_size."""
    q = jnp.ones((1, 17, 2, 8))
    cu = jnp.array([0, 1, 17])
    [pq], new_cu, orig, pad_lens = pad_varlen_seqs([q], cu, chunk_size=16)
    assert orig == [1, 16]
    assert pad_lens == [16, 16]
    assert pq.shape == (1, 32, 2, 8)
