"""Spectrum amplitudes: analytic terms or full-minus-without-kernel templates."""

import logging

from cosmo_wap.numeric_mu.kernels import K1, K2, IntK1

logger = logging.getLogger(__name__)

KERNEL_NAMES = frozenset(
    name for cls in (K1, IntK1) for name in vars(cls) if not name.startswith("_") and callable(getattr(cls, name))
)
BK_KERNEL_NAMES = frozenset(name for name in KERNEL_NAMES if hasattr(K2, name))  # as numeric_mu.bk.check_kernels
INT_KERNEL_NAMES = frozenset(name for name in KERNEL_NAMES if hasattr(IntK1, name))

# the kernels each analytic term (or composite - see pk/combined.py) duplicates. Any analytic integrated term
# clashes with any integrated kernel, as mixing the two drops their cross terms
_INT_TERMS = ["IntNPP", "IntInt", "GRI", "ISW", "LxNPP", "ISWxNPP", "TDxNPP"]
_INT_TERMS += ["LxL", "LxTD", "LxISW", "ISWxISW", "ISWxTD", "TDxISW", "TDxTD"]
TERM_KERNELS = {
    "NPP": {"N"},
    **dict.fromkeys(["GR1", "GR2", "GRL", "Full"], {"LP"}),
    "GR": {"LP"} | INT_KERNEL_NAMES,
    **dict.fromkeys(_INT_TERMS, INT_KERNEL_NAMES),
    **{shape: {shape, "PNG"} for shape in ("Loc", "Eq", "Orth")},
}


def as_list(value):
    return [] if value is None else ([value] if isinstance(value, str) else list(value))


def resolve_signal(kernels, bk_kernels, terms, bk_terms, pkln=None, bkln=None):
    """The signal model of get_fish/Sampler: kernels (None -> ['N']) for both probes - bk_kernels (None) keeps
    those with a second order part - and terms (bk_terms None -> terms) the analytic contributions on top"""
    hint = {
        "": " (kernels defaults to ['N'])" if kernels is None else "",
        "bk_": " (bk_kernels follows kernels)" if bk_kernels is None else "",
    }
    kernels = ["N"] if kernels is None else as_list(kernels)
    if bk_kernels is None:
        bk_kernels = [name for name in kernels if name in BK_KERNEL_NAMES]
        if bkln and len(bk_kernels) < len(kernels):
            dropped = sorted(set(kernels) - BK_KERNEL_NAMES)
            logger.info("bk_kernels: %s have no second order kernel - left out of the bispectrum", dropped)
    else:
        bk_kernels = as_list(bk_kernels)
    if bk_terms is None:
        bk_terms = terms

    for probe, ln, model, extra, name in (("pk", pkln, kernels, terms, ""), ("bk", bkln, bk_kernels, bk_terms, "bk_")):
        if not ln:
            continue
        if not model and not as_list(extra):
            raise ValueError(f"No {probe} signal - pass {name}kernels or {name}terms, or set {probe}ln=None.")
        for term in as_list(extra):
            clash = TERM_KERNELS.get(term, set()) & set(model)
            if clash:
                raise ValueError(
                    f"{name}terms entry '{term}' duplicates {name}kernels {sorted(clash)}{hint[name]} - drop it from"
                    f" {name}terms, or pass {name}kernels=[] for an analytic {probe} signal."
                )
    return kernels, bk_kernels, terms, bk_terms


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
