import os
import sys

import jax

# The f64 FFI kernels are only reachable with x64 enabled, so turn it on before
# any jax computation happens. Everything in the tests is explicitly typed, so
# the change of default dtypes does not affect them.
jax.config.update("jax_enable_x64", True)

# Several host devices, so that test_sharding.py can shard over the CPU. This
# has to happen before the CPU backend is first initialised.
N_CPU_DEVICES = 4
try:
    jax.config.update("jax_num_cpu_devices", N_CPU_DEVICES)
except AttributeError:  # pragma: no cover - jax without the option
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + f" --xla_force_host_platform_device_count={N_CPU_DEVICES}"
    )

# Allow running the tests straight from a checkout (pytest only puts the tests
# directory on sys.path), while still preferring an installed ri_kernels.
try:  # pragma: no cover - trivial import plumbing
    import ri_kernels  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
