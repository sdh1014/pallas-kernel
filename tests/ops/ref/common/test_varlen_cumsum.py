"""Tests for chunk_local_cumsum with cu_seqlens."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from tops.cpu.ops.common.chunk_o import chunk_local_cumsum


def test_cumsum_varlen_independent():
  """Each segment's cumsum should be independent (not bleed across segments)."""
  C = 16
  g = jnp.concatenate(
    [
      jnp.ones((1, 16, 2, 8)),
      jnp.ones((1, 16, 2, 8)) * 2,
    ],
    axis=1,
  )
  cu = jnp.array([0, 16, 32])
  result = chunk_local_cumsum(g, C, cu_seqlens=cu)
  seg0 = result[0, :16, 0, 0]
  assert jnp.allclose(seg0, jnp.arange(1, 17, dtype=jnp.float32))
  seg1 = result[0, 16:32, 0, 0]
  assert jnp.allclose(seg1, jnp.arange(1, 17, dtype=jnp.float32) * 2)


def test_cumsum_varlen_matches_independent_calls():
  """Varlen cumsum should match calling cumsum on each segment independently."""
  C = 16
  key = jax.random.PRNGKey(42)
  g = jax.random.normal(key, (1, 48, 2, 8))
  cu = jnp.array([0, 16, 48])
  result = chunk_local_cumsum(g, C, cu_seqlens=cu)
  seg0 = chunk_local_cumsum(g[:, :16], C)
  seg1 = chunk_local_cumsum(g[:, 16:48], C)
  expected = jnp.concatenate([seg0, seg1], axis=1)
  assert jnp.allclose(result, expected)


def test_cumsum_varlen_3d():
  """Works for 3D [B, T, H] tensors (Simple GLA scalar g)."""
  C = 16
  g = jnp.ones((1, 32, 4))
  cu = jnp.array([0, 16, 32])
  result = chunk_local_cumsum(g, C, cu_seqlens=cu)
  assert result.shape == (1, 32, 4)
  assert jnp.allclose(result[0, 0, 0], 1.0)
  assert jnp.allclose(result[0, 16, 0], 1.0)


def test_cumsum_varlen_reverse():
  """Reverse cumsum with varlen should be per-segment independent."""
  C = 16
  key = jax.random.PRNGKey(7)
  g = jax.random.normal(key, (1, 32, 2, 8))
  cu = jnp.array([0, 16, 32])
  result = chunk_local_cumsum(g, C, reverse=True, cu_seqlens=cu)
  seg0 = chunk_local_cumsum(g[:, :16], C, reverse=True)
  seg1 = chunk_local_cumsum(g[:, 16:], C, reverse=True)
  expected = jnp.concatenate([seg0, seg1], axis=1)
  assert jnp.allclose(result, expected)


def test_cumsum_none_cu_seqlens_unchanged():
  """cu_seqlens=None should behave identically to original."""
  C = 16
  key = jax.random.PRNGKey(99)
  g = jax.random.normal(key, (2, 32, 4, 8))
  result_none = chunk_local_cumsum(g, C, cu_seqlens=None)
  result_orig = chunk_local_cumsum(g, C)
  assert jnp.allclose(result_none, result_orig)
