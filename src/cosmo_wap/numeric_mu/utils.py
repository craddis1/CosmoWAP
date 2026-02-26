def merge_kernel_dicts(arr_dict, tmp_dict):
    """Merge tmp_dict into arr_dict, summing overlapping entries element-wise."""
    for i in tmp_dict:
        if i in arr_dict:
            for j in tmp_dict[i]:
                if j in arr_dict[i]:
                    arr_dict[i][j] = arr_dict[i][j] + tmp_dict[i][j]
                else:
                    arr_dict[i][j] = tmp_dict[i][j]
        else:
            arr_dict[i] = tmp_dict[i].copy()


def split_kernels(kernels):
    """Separate integrated and normal kernels"""
    int_kernels = set(
        ["I", "L", "TD", "ISW", "L1", "kappa_g"]
    )  # these are the integrated kernels - will need updating if add more
    if not isinstance(kernels, list):
        kernels = [kernels]
    return [s for s in kernels if s in int_kernels], [s for s in kernels if s not in int_kernels]
