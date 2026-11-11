"""Sharding the baseline axis with ``shard_map``.

Each device is handed the index arrays of its own baseline list and calls
:func:`vis_eval_with_indices` or :func:`analytic_eval_with_indices` on them.
The concatenated per-device rows, with the padding dropped, must equal the
unsharded operator, and so must the gradients with respect to the replicated
inputs (amp and phase, and the delay for the analytic operator).

The padding follows the rule the two functions document: duplicates of a real
pair for :class:`RFIVisOp`, and pairs distinct within each device's list for
:class:`RFIAnalyticVisOp`.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P

from ri_kernels.jax_api import (
    RFIAnalyticVisOp,
    RFIVisOp,
    analytic_eval_with_indices,
    analytic_indices,
    vis_eval_with_indices,
)
from ri_kernels.jax_api import rfi_analytic_vis_op as analytic_module
from ri_kernels.jax_api import rfi_vis_op as vis_module
from test_rfi_analytic_vis_op import make_inputs

try:
    from jax import shard_map
except ImportError:  # pragma: no cover - jax < 0.7
    from jax.experimental.shard_map import shard_map

AXIS = "bl"


def _meshes():
    """A mesh over the devices of each platform that has the kernels and >= 2 devices."""
    meshes = []
    platforms = []
    if vis_module._TAB_LIB is not None:
        platforms.append("cpu")
    if vis_module._TAB_LIB_GPU is not None:
        platforms += ["cuda", "rocm"]
    for platform in platforms:
        try:
            devices = jax.devices(platform)
        except RuntimeError:
            continue
        if len(devices) >= 2:
            meshes.append(Mesh(np.array(devices[:4]), (AXIS,)))
    return meshes


MESHES = _meshes()

if not MESHES:
    pytest.skip("no platform with the kernels and two or more devices", allow_module_level=True)


@pytest.fixture(params=MESHES, ids=lambda m: m.devices.flat[0].platform)
def mesh(request):
    with jax.default_device(request.param.devices.flat[0]):
        yield request.param


@pytest.fixture(params=[(jnp.float32, jnp.complex64), (jnp.float64, jnp.complex128)],
                ids=["f32", "f64"])
def precision(request):
    return request.param


def map_baselines(fn, mesh, n_replicated):
    """shard_map ``fn(indices, *replicated)`` with the indices split across devices.

    Every index array arrives stacked along a leading device axis; each device
    unstacks its own. The ``n_replicated`` other arguments are replicated.
    """
    def body(stacked, *args):
        return fn(tuple(x[0] for x in stacked), *args)

    kwargs = dict(mesh=mesh, in_specs=(P(AXIS),) + (P(),) * n_replicated, out_specs=P(AXIS))
    try:
        return shard_map(body, check_vma=False, **kwargs)
    except TypeError:  # pragma: no cover - jax < 0.7 spells it check_rep
        return shard_map(body, check_rep=False, **kwargs)


def split(pairs, n_dev, pad):
    """Contiguous per-device chunks of ``pairs``, each padded by ``pad(chunk, n)``.

    Returns the padded per-device lists and the positions of the real rows in
    the concatenated output, in the order of ``pairs``.
    """
    per_device = -(-len(pairs) // n_dev)
    lists, keep = [], []
    for d in range(n_dev):
        chunk = pairs[d * per_device:(d + 1) * per_device]
        keep.extend(range(d * per_device, d * per_device + len(chunk)))
        n_pad = per_device - len(chunk)
        lists.append(np.concatenate([chunk, pad(chunk, n_pad)]) if n_pad else chunk)
    return lists, np.asarray(keep)


def pad_with_duplicates(pairs):
    """RFIVisOp padding: any valid pair, here repeats of the first real one."""
    return lambda chunk, n: np.repeat(pairs[:1], n, axis=0)


def pad_with_unused_pairs(n_ant):
    """RFIAnalyticVisOp padding: ordered pairs that are not in the device's list."""
    def pad(chunk, n):
        used = {tuple(p) for p in chunk}
        free = (p for p in itertools.product(range(n_ant), repeat=2) if p not in used)
        return np.asarray(list(itertools.islice(free, n)), dtype=np.int32).reshape(-1, 2)
    return pad


def all_pairs(n_ant, autocorr=True, seed=None):
    lo = 0 if autocorr else 1
    pairs = np.asarray([(i, j) for i in range(n_ant) for j in range(i + lo, n_ant)], np.int32)
    if seed is not None:
        pairs = pairs[np.random.default_rng(seed).permutation(len(pairs))]
    return pairs


def needing_padding(pairs, n_dev):
    """Drop the last pair if that is what it takes for the split to need padding."""
    return pairs[:-1] if len(pairs) % n_dev == 0 else pairs


def stack(indices_per_device):
    return tuple(jnp.stack(x) for x in zip(*indices_per_device))


def assert_close(actual, expected, real, what=""):
    tolerance = 2e-5 if real == jnp.float32 else 1e-12
    expected = np.asarray(expected)
    scale = max(float(np.max(np.abs(expected), initial=0)), 1.)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=tolerance,
                               atol=tolerance * scale, err_msg=what)


# --- RFIVisOp -----------------------------------------------------------------


@pytest.mark.parametrize("n_ant,autocorr,seed", [(5, False, None), (7, True, 3)],
                         ids=["cross", "auto_shuffled"])
def test_vis_sharded_matches_unsharded(mesh, precision, n_ant, autocorr, seed):
    real, complex_ = precision
    n_dev = mesh.devices.size
    pairs = needing_padding(all_pairs(n_ant, autocorr, seed), n_dev)
    lists, keep = split(pairs, n_dev, pad_with_duplicates(pairs))
    indices = stack([RFIVisOp(n_ant, jnp.asarray(p[:, 0]), jnp.asarray(p[:, 1])).indices
                     for p in lists])

    rng = np.random.default_rng(0)
    shape = (n_ant, 3, 2, 2, 2, 3)
    amp = jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape), complex_)
    phase = jnp.asarray(rng.uniform(-np.pi, np.pi, size=shape), real)
    g = rng.normal(size=(len(pairs), 3, 2)) + 1j * rng.normal(size=(len(pairs), 3, 2))
    g = jnp.asarray(g, complex_)

    sharded_eval = map_baselines(vis_eval_with_indices, mesh, 2)
    sharded = jax.jit(lambda a, p: sharded_eval(indices, a, p)[keep])
    op = RFIVisOp(n_ant, jnp.asarray(pairs[:, 0]), jnp.asarray(pairs[:, 1]))

    value, pullback = jax.vjp(sharded, amp, phase)
    ref_value, ref_pullback = jax.vjp(op.eval, amp, phase)
    assert_close(value, ref_value, real, "value")
    for name, x, y in zip(("amp", "phase"), pullback(g), ref_pullback(g)):
        assert_close(x, y, real, f"{name} cotangent")


# --- RFIAnalyticVisOp ---------------------------------------------------------

requires_analytic = pytest.mark.skipif(
    analytic_module._TAB_LIB_ANALYTIC is None and analytic_module._TAB_LIB_ANALYTIC_GPU is None,
    reason="the installed library has no analytic kernels",
)


@requires_analytic
@pytest.mark.parametrize("n_ant,autocorr,seed", [(5, True, None), (34, False, 2)],
                         ids=["one_tile", "two_tiles_shuffled"])
def test_analytic_sharded_matches_unsharded(mesh, precision, n_ant, autocorr, seed):
    real, complex_ = precision
    n_dev = mesh.devices.size
    pairs = needing_padding(all_pairs(n_ant, autocorr, seed), n_dev)
    lists, keep = split(pairs, n_dev, pad_with_unused_pairs(n_ant))
    indices = stack([analytic_indices(n_ant, p[:, 0], p[:, 1]) for p in lists])

    args = make_inputs(real, complex_, n_ant=n_ant)
    tables = args[3:]
    rng = np.random.default_rng(1)
    out = (len(pairs),) + args[0].shape[2:]
    g = jnp.asarray(rng.normal(size=out) + 1j * rng.normal(size=out), complex_)

    sharded_eval = map_baselines(
        lambda idx, *a: analytic_eval_with_indices(idx, *a, scratch_mb=64), mesh, len(args))
    sharded = jax.jit(lambda a, p, d: sharded_eval(indices, a, p, d, *tables)[keep])
    op = RFIAnalyticVisOp(n_ant, pairs[:, 0], pairs[:, 1])
    unsharded = jax.jit(lambda a, p, d: op.eval(a, p, d, *tables, scratch_mb=64))

    value, pullback = jax.vjp(sharded, *args[:3])
    ref_value, ref_pullback = jax.vjp(unsharded, *args[:3])
    assert_close(value, ref_value, real, "value")
    for name, x, y in zip(("amp", "phase", "delay"), pullback(g), ref_pullback(g)):
        assert_close(x, y, real, f"{name} cotangent")


# --- the index tuples and their checks -----------------------------------------


@requires_analytic
def test_analytic_indices_match_the_op():
    pairs = all_pairs(6, seed=1)
    op = RFIAnalyticVisOp(6, pairs[:, 0], pairs[:, 1])
    built = analytic_indices(6, pairs[:, 0], pairs[:, 1])
    assert len(built) == len(op.indices) == analytic_module.N_IDX
    for x, y in zip(built, op.indices):
        assert x.dtype == y.dtype == jnp.int32
        np.testing.assert_array_equal(x, y)


def test_analytic_indices_reject_duplicate_padding():
    a1, a2 = np.array([0, 1, 0], np.int32), np.array([1, 2, 1], np.int32)
    with pytest.raises(ValueError, match="at most once"):
        analytic_indices(3, a1, a2)


def test_vis_indices_match_the_op_fields():
    pairs = all_pairs(5, seed=0)
    op = RFIVisOp(5, jnp.asarray(pairs[:, 0]), jnp.asarray(pairs[:, 1]))
    assert op.indices == (op.a1, op.a1_sorter, op.a1_start, op.a2, op.a2_sorter, op.a2_start)


def test_index_tuples_of_the_wrong_length_are_rejected():
    pairs = all_pairs(4)
    vis_indices = RFIVisOp(4, jnp.asarray(pairs[:, 0]), jnp.asarray(pairs[:, 1])).indices
    with pytest.raises(ValueError, match="6 RFIVisOp index arrays"):
        vis_eval_with_indices(vis_indices[:5], None, None)
    with pytest.raises(ValueError, match="8 RFIAnalyticVisOp index arrays"):
        analytic_eval_with_indices(vis_indices, *([None] * 10))


@pytest.mark.skipif(not hasattr(jax, "typeof"), reason="jax < 0.7 tracks no varying axes")
def test_check_vma_is_refused(mesh):
    """With check_vma=True the replicated inputs' gradients would be wrong; refuse."""
    n_dev = mesh.devices.size
    pairs = all_pairs(4)[: 2 * n_dev]
    lists, _ = split(pairs, n_dev, pad_with_duplicates(pairs))
    indices = stack([RFIVisOp(4, jnp.asarray(p[:, 0]), jnp.asarray(p[:, 1])).indices
                     for p in lists])
    shape = (4, 1, 1, 1, 1, 1)
    amp, phase = jnp.ones(shape, jnp.complex64), jnp.zeros(shape, jnp.float32)

    def body(stacked, a, p):
        return vis_eval_with_indices(tuple(x[0] for x in stacked), a, p)

    fn = shard_map(body, mesh=mesh, in_specs=(P(AXIS), P(), P()), out_specs=P(AXIS),
                   check_vma=True)
    with pytest.raises(ValueError, match="check_vma=False"):
        jax.jit(fn)(indices, amp, phase)
