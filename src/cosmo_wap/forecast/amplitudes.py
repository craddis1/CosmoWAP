"""Spectrum amplitudes: analytic terms or full-minus-without-kernel templates."""

from cosmo_wap.numeric_mu.kernels import K1, IntK1

KERNEL_NAMES = frozenset(
    name for cls in (K1, IntK1) for name in vars(cls) if not name.startswith("_") and callable(getattr(cls, name))
)


def as_list(value):
    return [] if value is None else ([value] if isinstance(value, str) else list(value))


def is_kernel_amplitude(param, kernels, analytic_terms):
    """An active kernel wins a shared name (Loc/Eq/Orth); otherwise keep the analytic meaning."""
    return param in as_list(kernels) or (param in KERNEL_NAMES and param not in analytic_terms)


def validate_kernel_amplitudes(params, kernel_sets, analytic_terms):
    """Allow a kernel absent from one probe, but catch an amplitude absent from every probe."""
    active = {name for kernels in kernel_sets for name in as_list(kernels)}
    for param in params:
        for name in as_list(param):
            if name in KERNEL_NAMES and name not in analytic_terms and name not in active:
                raise ValueError(
                    f"Kernel amplitude '{name}' must be included in kernels or bk_kernels for an active probe."
                )


def kernel_signal(kernels, evaluate, cache):
    """Share the full/reduced numerical spectra within one fixed cosmology, bias and grid evaluation."""
    key = tuple(as_list(kernels))
    if not key:
        return 0
    if key not in cache:
        cache[key] = evaluate(list(key))
    return cache[key]


def kernel_template(param, kernels, evaluate, cache):
    """All contributions containing param, including its cross terms with the other active kernels."""
    kernels = as_list(kernels)
    if param not in kernels:
        return 0
    return kernel_signal(kernels, evaluate, cache) - kernel_signal(
        [name for name in kernels if name != param], evaluate, cache
    )
