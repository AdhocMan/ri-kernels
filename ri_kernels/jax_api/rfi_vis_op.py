"""
Custom JAX primitives for RFI visibility calculation using FFI.
"""

import ctypes
import importlib.util
import os
import site
import sysconfig
from functools import partial

import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.extend import core
from jax.interpreters import ad, mlir, xla

class RFIVisOp:
    """
    Operator for computing RFI visibility using JAX FFI with precomputed indices.
    
    This class encapsulates the antenna baseline indexing logic required for efficient
    RFI visibility calculations. It precomputes sorted indices and search positions
    for both antenna arrays (a1 and a2) to enable fast lookups in the FFI kernel.
    """
    
    def __init__(self, n_ant, a1, a2):
        """
        Initialize the RFI visibility operator with antenna baseline information.
        
        Args:
            n_ant: Number of antennas in the array.
            a1: Array of first antenna indices for each baseline (shape: [n_baselines]).
            a2: Array of second antenna indices for each baseline (shape: [n_baselines]).
        
        The initialization precomputes:
            - a1_sorter, a2_sorter: Indices that would sort the a1 and a2 arrays
            - a1_start, a2_start: Starting positions for each antenna in the sorted arrays
        """
        self.a1 = a1
        self.a2 = a2
        (
            self.a1_sorter,
            self.a1_start,
            self.a2_sorter,
            self.a2_start,
        ) = prepare_indices(n_ant, a1, a2)

    @property
    def indices(self):
        """The index arrays the primitive takes before the data arrays.

        ``(a1, a1_sorter, a1_start, a2, a2_sorter, a2_start)``, the tuple
        :func:`vis_eval_with_indices` expects.
        """
        return (
            self.a1, self.a1_sorter, self.a1_start,
            self.a2, self.a2_sorter, self.a2_start,
        )

    def eval(self, rfi_amp_fine, rfi_phase):
        """
        Evaluate the RFI visibility for given RFI amplitudes and phases.

        Args:
            rfi_amp_fine: Fine-grained RFI amplitude array with shape
                         (n_ant, n_freq, n_time, n_rfi, n_int_freq, n_int_time).
            rfi_phase: RFI phase array with shape matching rfi_amp_fine.

        Returns:
            Array of RFI visibilities with shape (n_baselines, n_freq, n_time).
        """
        return vis_eval_with_indices(self.indices, rfi_amp_fine, rfi_phase)


def vis_eval_with_indices(indices, rfi_amp_fine, rfi_phase):
    """:meth:`RFIVisOp.eval` with the index arrays passed in, not stored on an op.

    An :class:`RFIVisOp` stores the index arrays of one fixed baseline list.
    Inside a function that ``shard_map`` maps over the baseline axis, those
    stored arrays are constants that every device sees whole, so every device
    would compute every baseline. Pass each device its own indices as a mapped
    argument and call this function in the mapped body instead. The output
    has one row per entry of ``a1``, so a device computes exactly the rows for
    its own baselines.

    Args:
        indices: ``(a1, a1_sorter, a1_start, a2, a2_sorter, a2_start)``, as
            returned by :attr:`RFIVisOp.indices` for the device's baseline list.
            ``a1_start`` and ``a2_start`` have length ``n_ant`` whatever the
            number of baselines.
        rfi_amp_fine, rfi_phase: As in :meth:`RFIVisOp.eval`.

    Padding: when the baseline count does not divide evenly across devices,
    each device's list has to be padded to the same length. Any valid antenna
    pair works as padding, a pair the list already holds included. Discard
    the padded rows of the output. They then get a zero cotangent and add
    nothing to the gradients.
    """
    if len(indices) != 6:
        raise ValueError(
            f"Expected the 6 RFIVisOp index arrays (a1, a1_sorter, a1_start, a2, "
            f"a2_sorter, a2_start); got {len(indices)}"
        )
    return rfi_vis_op.bind(*indices, rfi_amp_fine, rfi_phase)


def prepare_indices(n_ant, a1, a2):
    """
    Prepare indices for sorted usage in the C++ kernel.

    Args:
        n_ant: Number of antennas.
        a1: Antenna 1 indices.
        a2: Antenna 2 indices.

    Returns:
        A tuple (a1_sorter, a1_start, a2_sorter, a2_start) suitable for the kernel.
    """
    a1_sorter = jnp.argsort(a1).astype(a1.dtype)
    a2_sorter = jnp.argsort(a2).astype(a1.dtype)

    v = jnp.arange(0, n_ant, dtype=a1.dtype)
    a1_start = jnp.searchsorted(a1, v, sorter=a1_sorter)
    a2_start = jnp.searchsorted(a2, v, sorter=a2_sorter)

    return (a1_sorter, a1_start, a2_sorter, a2_start)


_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# The CPU library ships inside ri_kernels itself, while the GPU libraries come
# from optional add-on distributions (ri_kernels_cuda12, ri_kernels_cuda13,
# ri_kernels_rocm) that install a top-level package of their own containing
# nothing but the shared library. Search all of them. cuda13 is probed before
# cuda12 so that a machine with both installed picks the newer toolkit.
_PKG_NAMES = (
    "ri_kernels",
    "ri_kernels_cuda13",
    "ri_kernels_cuda12",
    "ri_kernels_rocm",
)


def _site_dirs():
    dirs = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        dirs.append(user_site)
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        dirs.append(purelib)
    return dirs


def _candidate_dirs():
    dirs = [_DIR_PATH]
    site_dirs = _site_dirs()
    for pkg in _PKG_NAMES:
        try:
            spec = importlib.util.find_spec(pkg)
        except (ImportError, ValueError):
            spec = None
        if spec and spec.submodule_search_locations:
            dirs.extend(spec.submodule_search_locations)
        for sp in site_dirs:
            dirs.append(os.path.join(sp, pkg))
    return dirs


def _load_library(name):
    seen = set()
    for d in _candidate_dirs():
        lib_path = os.path.join(d, name)
        if lib_path in seen:
            continue
        seen.add(lib_path)
        if os.path.isfile(lib_path):
            return ctypes.cdll.LoadLibrary(lib_path)
    return None



_TAB_LIB_GPU_NAME = "libri_kernels_cuda.so"
_TAB_PLATFORM_NAME = "CUDA"

if any("rocm" in str(s) for s in list(jax.devices())):
    _TAB_LIB_GPU_NAME = "libri_kernels_hip.so"
    _TAB_PLATFORM_NAME = "ROCM"


_TAB_LIB = _load_library("libri_kernels.so")
_TAB_LIB_GPU = _load_library(_TAB_LIB_GPU_NAME)


def _ffi_target(op, pass_, platform, dtype):
    """Name of an FFI handler: both its exported symbol and its XLA target."""
    return f"ri_{op}_{pass_}_{platform}_{dtype}"


def _register_targets(lib, op, passes, platform, xla_platform):
    for dtype in ("f32", "f64"):
        for pass_ in passes:
            name = _ffi_target(op, pass_, platform, dtype)
            jax.ffi.register_ffi_target(
                name, jax.ffi.pycapsule(getattr(lib, name)), platform=xla_platform
            )


if _TAB_LIB:
    _register_targets(_TAB_LIB, "rfi_vis", ("fwd", "jvp", "transpose"), "cpu", "cpu")

if _TAB_LIB_GPU:
    _register_targets(
        _TAB_LIB_GPU, "rfi_vis", ("fwd", "jvp", "transpose"), "gpu", _TAB_PLATFORM_NAME
    )


def _dtype_suffix(amp_dtype, phase_dtype):
    """Pick the FFI symbol suffix for a (rfi_amp_fine, rfi_phase) dtype pair.

    The kernels accept matched precision: complex64+float32 or complex128+float64.
    Mixed precision is rejected at the Python boundary.
    """
    amp = jnp.dtype(amp_dtype)
    phase = jnp.dtype(phase_dtype)
    if amp == jnp.complex64 and phase == jnp.float32:
        return "f32"
    if amp == jnp.complex128 and phase == jnp.float64:
        return "f64"
    raise TypeError(
        f"RFI FFI kernels require matched precision: (complex64, float32) "
        f"or (complex128, float64). Got (amp={amp}, phase={phase})."
    )


def _varying_axes(aval):
    """The manual mesh axes ``aval`` varies over inside ``shard_map``.

    Non-empty only under ``shard_map(..., check_vma=True)``. JAX spells this
    ``aval.mat.varying`` from 0.11 and ``aval.vma`` before that.
    """
    mat = getattr(aval, "mat", None)
    if mat is not None:
        return frozenset(getattr(mat, "varying", ()))
    return frozenset(getattr(aval, "vma", ()))


def _check_no_vma(name, *avals):
    """Refuse to trace under ``shard_map`` with ``check_vma=True``.

    The primitives' abstract evaluation does not propagate the varying manual
    axes, so with checking on JAX would treat a per-device result as
    replicated. The values still come out right, but the transpose then omits
    the psum of the cotangents of replicated inputs, and their gradients are
    silently wrong. Fail here instead.
    """
    axes = frozenset().union(*(_varying_axes(a) for a in avals))
    if axes:
        raise ValueError(
            f"{name} cannot track the varying manual axes {sorted(axes)}: call it "
            f"inside shard_map(..., check_vma=False) (check_rep=False before "
            f"jax 0.7), as for jax.ffi.ffi_call."
        )


def _check_tab_lib():
    if _TAB_LIB is None:
        raise RuntimeError(
            f"FFI selected, but libri_kernels.so not found! "
            f"Install the wheel built via scikit-build-core or build from "
            f"{_DIR_PATH}"
        )


def _check_tab_lib_gpu():
    if _TAB_LIB_GPU is None:
        raise RuntimeError(
            f"FFI selected, but {_TAB_LIB_GPU_NAME} not found! "
            f"Install the matching add-on package (ri_kernels_cuda12 or "
            f"ri_kernels_cuda13), or rebuild the wheel with "
            f"-Ccmake.define.RI_KERNELS_CUDA=ON (or RI_KERNELS_ROCM=ON)"
        )


# --- Transpose Op ---

rfi_transpose_op = core.Primitive("rfi_transpose_op")
rfi_transpose_op.def_impl(partial(xla.apply_primitive, rfi_transpose_op))
rfi_transpose_op.multiple_results = True


def rfi_transpose_abstract(
    a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase, g
):
    """
    Abstract evaluation for the RFI transpose operation.
    """
    _check_no_vma(
        "rfi_transpose_op",
        a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase, g,
    )
    _dtype_suffix(rfi_amp_fine.dtype, rfi_phase.dtype)
    # Shapes are implicitly checked by JAX mechanisms or assumed correct from lowering
    t1 = ShapedArray(rfi_amp_fine.shape, rfi_amp_fine.dtype)
    t2 = ShapedArray(rfi_phase.shape, rfi_phase.dtype)
    return (t1, t2)


rfi_transpose_op.def_abstract_eval(rfi_transpose_abstract)


def rfi_transpose_lowering_cpu(
    ctx, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase, g
):
    _check_tab_lib()
    suffix = _dtype_suffix(ctx.avals_in[6].dtype, ctx.avals_in[7].dtype)
    res = jax.ffi.ffi_lowering(_ffi_target("rfi_vis", "transpose", "cpu", suffix))
    return res(
        ctx,
        a1,
        a1_sorter,
        a1_start,
        a2,
        a2_sorter,
        a2_start,
        rfi_amp_fine,
        rfi_phase,
        g,
    )


mlir.register_lowering(rfi_transpose_op, rfi_transpose_lowering_cpu, platform="cpu")


def rfi_transpose_lowering_gpu(
    ctx, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase, g
):
    _check_tab_lib_gpu()
    suffix = _dtype_suffix(ctx.avals_in[6].dtype, ctx.avals_in[7].dtype)
    res = jax.ffi.ffi_lowering(_ffi_target("rfi_vis", "transpose", "gpu", suffix))
    return res(
        ctx,
        a1,
        a1_sorter,
        a1_start,
        a2,
        a2_sorter,
        a2_start,
        rfi_amp_fine,
        rfi_phase,
        g,
    )


mlir.register_lowering(rfi_transpose_op, rfi_transpose_lowering_gpu, platform="gpu")


# --- JVP Op ---

rfi_jvp_op = core.Primitive("rfi_jvp_op")
rfi_jvp_op.def_impl(partial(xla.apply_primitive, rfi_jvp_op))


def rfi_jvp_abstract(
    a1,
    a1_sorter,
    a1_start,
    a2,
    a2_sorter,
    a2_start,
    rfi_amp_fine,
    rfi_amp_fine_grad,
    rfi_phase,
    rfi_phase_grad,
):
    """
    Abstract evaluation for the RFI JVP operation.
    """
    _check_no_vma(
        "rfi_jvp_op",
        a1, a1_sorter, a1_start, a2, a2_sorter, a2_start,
        rfi_amp_fine, rfi_amp_fine_grad, rfi_phase, rfi_phase_grad,
    )
    _dtype_suffix(rfi_amp_fine.dtype, rfi_phase.dtype)
    # rfi_amp_fine and rfi_phase shape is
    # (n_ant, n_freq, n_time, n_rfi, n_int_freq, n_int_time)
    n_freq = rfi_amp_fine.shape[1]
    n_time = rfi_amp_fine.shape[2]
    return ShapedArray([a1.shape[0], n_freq, n_time], rfi_amp_fine.dtype)


rfi_jvp_op.def_abstract_eval(rfi_jvp_abstract)


def rfi_jvp_lowering_cpu(
    ctx,
    a1,
    a1_sorter,
    a1_start,
    a2,
    a2_sorter,
    a2_start,
    rfi_amp_fine,
    rfi_amp_fine_grad,
    rfi_phase,
    rfi_phase_grad,
):
    _check_tab_lib()
    suffix = _dtype_suffix(ctx.avals_in[6].dtype, ctx.avals_in[8].dtype)
    res = jax.ffi.ffi_lowering(_ffi_target("rfi_vis", "jvp", "cpu", suffix))
    return res(
            ctx,
            a1,
            a1_sorter,
            a1_start,
            a2,
            a2_sorter,
            a2_start,
            rfi_amp_fine,
            rfi_amp_fine_grad,
            rfi_phase,
            rfi_phase_grad,
        )


mlir.register_lowering(rfi_jvp_op, rfi_jvp_lowering_cpu, platform="cpu")


def rfi_jvp_lowering_gpu(
    ctx,
    a1,
    a1_sorter,
    a1_start,
    a2,
    a2_sorter,
    a2_start,
    rfi_amp_fine,
    rfi_amp_fine_grad,
    rfi_phase,
    rfi_phase_grad,
):
    _check_tab_lib_gpu()
    suffix = _dtype_suffix(ctx.avals_in[6].dtype, ctx.avals_in[8].dtype)
    res = jax.ffi.ffi_lowering(_ffi_target("rfi_vis", "jvp", "gpu", suffix))
    return res(
            ctx,
            a1,
            a1_sorter,
            a1_start,
            a2,
            a2_sorter,
            a2_start,
            rfi_amp_fine,
            rfi_amp_fine_grad,
            rfi_phase,
            rfi_phase_grad,
        )


mlir.register_lowering(rfi_jvp_op, rfi_jvp_lowering_gpu, platform="gpu")


def rfi_jvp_transpose(
    g,
    a1,
    a1_sorter,
    a1_start,
    a2,
    a2_sorter,
    a2_start,
    rfi_amp_fine,
    rfi_amp_fine_grad,
    rfi_phase,
    rfi_phase_grad,
):
    """
    Transpose rule for the RFI JVP operation.
    """
    t1, t2 = rfi_transpose_op.bind(
        a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase, g
    )

    return None, None, None, None, None, None, t1, t1, t2, t2


ad.primitive_transposes[rfi_jvp_op] = rfi_jvp_transpose


# --- Vis Op ---

rfi_vis_op = core.Primitive("rfi_vis_op")
rfi_vis_op.def_impl(partial(xla.apply_primitive, rfi_vis_op))


def rfi_vis_abstract(
    a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase
):
    """
    Abstract evaluation for the RFI visibility operation.
    """
    _check_no_vma(
        "rfi_vis_op",
        a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase,
    )
    _dtype_suffix(rfi_amp_fine.dtype, rfi_phase.dtype)
    # rfi_amp_fine and rfi_phase shape is
    # (n_ant, n_freq, n_time, n_rfi, n_int_freq, n_int_time)
    n_freq = rfi_amp_fine.shape[1]
    n_time = rfi_amp_fine.shape[2]
    return ShapedArray([a1.shape[0], n_freq, n_time], rfi_amp_fine.dtype)


rfi_vis_op.def_abstract_eval(rfi_vis_abstract)


def rfi_vis_lowering_cpu(
    ctx, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase
):
    _check_tab_lib()
    suffix = _dtype_suffix(ctx.avals_in[6].dtype, ctx.avals_in[7].dtype)
    res = jax.ffi.ffi_lowering(_ffi_target("rfi_vis", "fwd", "cpu", suffix))
    return res(
            ctx,
            a1,
            a1_sorter,
            a1_start,
            a2,
            a2_sorter,
            a2_start,
            rfi_amp_fine,
            rfi_phase,
        )


mlir.register_lowering(rfi_vis_op, rfi_vis_lowering_cpu, platform="cpu")


def rfi_vis_lowering_gpu(
    ctx, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase
):
    _check_tab_lib_gpu()
    suffix = _dtype_suffix(ctx.avals_in[6].dtype, ctx.avals_in[7].dtype)
    res = jax.ffi.ffi_lowering(_ffi_target("rfi_vis", "fwd", "gpu", suffix))
    return res(
            ctx,
            a1,
            a1_sorter,
            a1_start,
            a2,
            a2_sorter,
            a2_start,
            rfi_amp_fine,
            rfi_phase,
        )


mlir.register_lowering(rfi_vis_op, rfi_vis_lowering_gpu, platform="gpu")


def rfi_vis_jvp(args, tangents):
    """
    JVP rule for the RFI visibility operation.
    """
    a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase = args
    _, _, _, _, _, _, rfi_amp_fine_dot, rfi_phase_dot = tangents

    if isinstance(rfi_amp_fine_dot, ad.Zero):
        rfi_amp_fine_dot = jnp.zeros(rfi_amp_fine.shape, rfi_amp_fine.dtype)
    if isinstance(rfi_phase_dot, ad.Zero):
        rfi_phase_dot = jnp.zeros(rfi_phase.shape, rfi_phase.dtype)

    grad = rfi_jvp_op.bind(
        a1,
        a1_sorter,
        a1_start,
        a2,
        a2_sorter,
        a2_start,
        rfi_amp_fine,
        rfi_amp_fine_dot,
        rfi_phase,
        rfi_phase_dot,
    )

    return (
        rfi_vis_op.bind(
            a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase
        ),
        grad,
    )


ad.primitive_jvps[rfi_vis_op] = rfi_vis_jvp
