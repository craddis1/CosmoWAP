from .kernels import IntK1


def freeze_weights(weights):
    """Hashable form of a term's survey-weight dict (for grouping in merge_terms)."""
    return tuple(sorted(weights.items(), key=lambda kv: repr(kv[0])))


def merge_terms(terms):
    """Group kernel terms by (mu_pow, q_pow, weights), summing the radial arrays element-wise.
    Terms with different survey-weight dicts stay separate so the weights can be applied
    after the line-of-sight integral (which is linear in the radial array)."""
    merged = {}
    for i, j, arr, wt in terms:
        key = (i, j, freeze_weights(wt))
        merged[key] = merged[key] + arr if key in merged else arr
    return merged


def split_kernels(kernels):
    """Separate integrated and normal kernels - integrated ones are defined in IntK1"""
    if not isinstance(kernels, list):
        kernels = [kernels]
    return [s for s in kernels if hasattr(IntK1, s)], [s for s in kernels if not hasattr(IntK1, s)]
