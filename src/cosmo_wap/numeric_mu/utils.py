from .kernels import IntK1


def merge_terms(terms):
    """Group kernel terms by (mu_pow, q_pow), summing the radial arrays element-wise."""
    merged = {}
    for i, j, arr in terms:
        merged[(i, j)] = merged[(i, j)] + arr if (i, j) in merged else arr
    return merged


def split_kernels(kernels):
    """Separate integrated and normal kernels - integrated ones are defined in IntK1"""
    if not isinstance(kernels, list):
        kernels = [kernels]
    return [s for s in kernels if hasattr(IntK1, s)], [s for s in kernels if not hasattr(IntK1, s)]
