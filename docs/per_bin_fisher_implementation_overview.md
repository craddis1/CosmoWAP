# Per-Bin Fisher Forecast with Global fNL — Implementation Overview

This document describes the planned extension to `CosmoWAP`'s Fisher forecasting system
to support **per-bin parameters** (e.g. bias parameters that are independent in each
redshift bin) alongside **global parameters** (e.g. `fNL`) that are constrained by
summing information across all bins.  It also explains the two equivalent marginalization
strategies and how they map onto the existing code.

---

## 1. Existing Code Architecture

### Key files

| File | Purpose |
|---|---|
| `src/cosmo_wap/forecast/forecast.py` | `FullForecast` — main entry point; builds Fisher matrix |
| `src/cosmo_wap/forecast/core.py` | `PkForecast`, `BkForecast` — single-bin objects; numerical derivatives |
| `src/cosmo_wap/forecast/fisher.py` | `FisherMat` — stores and analyses the result |
| `src/cosmo_wap/forecast/fisher_list.py` | `FisherList` — manages collections of `FisherMat` objects |

### How the Fisher matrix is currently built (`forecast.py:422-543`)

**Step 1 — Pre-compute** (`_precompute_derivatives_and_covariances`, lines 352–411):

For every redshift bin `k` and every parameter `p` in `param_list`:

```
derivs[p][k] = {'pk': dD/dp|_k,   'bk': dB/dp|_k}
inv_covs[k]  = {'pk': C_pk^{-1},  'bk': C_bk^{-1}}
```

Derivatives are computed via a 5-point stencil in `core.py:115-303`
(`five_point_stencil`).  Cosmological parameters reuse pre-computed `ClassWAP`
objects (`_precompute_cache`, lines 312–350) so CLASS is only called once.

**Step 2 — Assemble** (lines 491–514):

```python
for i, j in upper_triangle(param_list):
    f_ij = 0
    for k in range(N_bins):          # sum over ALL bins unconditionally
        f_ij += einsum(derivs[i][k]['pk'], inv_covs[k]['pk'], derivs[j][k]['pk'])
        f_ij += einsum(derivs[i][k]['bk'], inv_covs[k]['bk'], derivs[j][k]['bk'])
    fish_mat[i,j] = f_ij
```

**Key point**: every parameter is currently treated as *global* — the same parameter
appears in all bins and all bins contribute to every Fisher element.

### `FisherMat` (fisher.py)

`FisherMat.__init__` immediately inverts the full matrix:

```python
self.covariance = np.linalg.inv(fisher_matrix)   # line 50
self.errors     = np.sqrt(np.diag(self.covariance))
```

Analysis helpers (`get_error`, `get_correlation`, `add_chain`, `plot_1D`, etc.) all
work on `self.covariance` / `self.errors` and index by position in `param_list`.

---

## 2. What We Want to Add

### Two use cases

**A — Per-bin constraints (no marginalization)**:
"What is the constraint on `b_1` in each redshift bin independently?" — treat each
bin's bias as a separate parameter and read off the per-bin errors.

**B — Marginalized global constraint**:
"What is the constraint on `fNL` after marginalizing over independent per-bin biases?"

Both emerge from the same extended Fisher matrix; the difference is only in what you
read out at the end.

### The extended parameter vector

Instead of a flat `param_list = ["b_1", "fNL"]`, we need to distinguish:

- **Global parameters** `θ_A`: `["fNL", "n_s", ...]` — same value across all bins.
- **Per-bin parameters** `θ_B^k` (`k = 0 … N_bins-1`): e.g. `["b_1_bin0",
  "b_1_bin1", ..., "b_1_binN"]`.

The extended parameter vector has size `N_A + N_B × N_bins`.

### Block structure of the extended Fisher matrix

Because per-bin parameters from different bins are independent (they only affect the
observable in their own bin), the extended Fisher matrix has a natural block structure:

```
F = | F_AA      F_AB^0   F_AB^1  ...  F_AB^{K-1} |
    | F_AB^0 T  F_BB^0   0       ...  0           |
    | F_AB^1 T  0        F_BB^1  ...  0           |
    | ...                                          |
    | F_AB^K-1T 0        0       ...  F_BB^{K-1}  |
```

where:
- `F_AA` = (N_A × N_A) — Fisher among global params, summed over all bins.
- `F_BB^k` = (N_B × N_B) — Fisher among per-bin params, contribution from bin k only.
- `F_AB^k` = (N_A × N_B) — cross Fisher between global and per-bin, from bin k only.
- Off-diagonal per-bin blocks are **exactly zero** by construction.

The per-bin contributions are:

```
F_AA         += sum_k  d_A^k · C_k^{-1} · d_A^k          (current behaviour)
F_BB^k        =        d_B^k · C_k^{-1} · d_B^k          (NEW — only bin k)
F_AB^k        =        d_A^k · C_k^{-1} · d_B^k          (NEW — only bin k)
```

---

## 3. Two Equivalent Marginalization Strategies

### Strategy 1 — Build full matrix, invert once

Build the full `(N_A + N_B × N_bins) × (N_A + N_B × N_bins)` Fisher matrix and invert
it. The top-left `N_A × N_A` block of the covariance gives the marginalized constraints
on global parameters.

- **Pro**: conceptually simple, handles all use-cases uniformly.
- **Con**: matrix grows quickly — 10 bins × 5 per-bin params + 3 global = 53×53.
  Numerically fine at these sizes but slower and requires storing the big matrix.

### Strategy 2 — Schur complement (bin-by-bin marginalization)

Because `F_BB` is block-diagonal, the Schur complement of `F_BB` in `F` simplifies to
a sum over bins:

```
F_marg = F_AA  -  sum_k  F_AB^k · (F_BB^k)^{-1} · (F_AB^k)^T
```

This is an `(N_A × N_A)` matrix.  Inverting it gives the marginalized covariance on
global parameters directly.

- **Pro**: never builds the large matrix; only inverts small `(N_B × N_B)` blocks per
  bin and one small `(N_A × N_A)` matrix at the end. More numerically stable.
  Integrates naturally into the existing bin loop.
- **Byproduct**: `(F_BB^k)^{-1}` computed as a side-effect gives per-bin errors on
  the bias parameters (conditional on global params fixed).

**Both strategies are mathematically identical** — the Schur complement formula is
exactly what full block-matrix inversion computes for the top-left block.

---

## 4. Mapping onto the Existing Code

### 4.1 New `get_fish` signature (forecast.py)

Add two new optional arguments:

```python
def get_fish(
    self,
    param_list,          # global parameters (as today)
    ...
    per_bin_params=None, # NEW: list of parameter names that are per-bin
    marginalize_per_bin=True,  # NEW: if True use Schur, if False build full matrix
    ...
) -> FisherMat:
```

`per_bin_params` is a list of parameter names drawn from the same vocabulary as
`param_list` (e.g. `["b_1", "b_2"]`). Internally the code expands these to
`b_1_bin0`, `b_1_bin1`, … for labelling purposes.

### 4.2 Derivative pre-computation

No changes needed to `_precompute_derivatives_and_covariances`.  The derivatives
`derivs[p][k]` are already organised as (parameter, bin).  A per-bin parameter `b_1`
in bin `k` simply uses `derivs[b_1][k]` and ignores all other bins — the existing
precomputation covers this automatically.

### 4.3 Assembly loop changes (forecast.py:491-514)

#### Strategy 2 (Schur, recommended)

Replace the inner bin loop with logic that accumulates three accumulators:

```python
F_AA  = np.zeros((N_A, N_A))
F_AB  = np.zeros((N_A, N_B, N_bins))   # or list of N_bins (N_A x N_B) blocks
F_BB  = np.zeros((N_B, N_B, N_bins))   # block diagonal

for k in range(N_bins):
    for spec in ['pk', 'bk']:
        if not active(spec): continue
        C_inv = inv_covs[k][spec]

        # --- global-global ---
        for i in range(N_A):
            for j in range(i, N_A):
                F_AA[i,j] += einsum(d_A[i][k], C_inv, d_A[j][k])

        # --- per-bin self (only bin k contributes to bin k's block) ---
        for i in range(N_B):
            for j in range(i, N_B):
                F_BB[i,j,k] += einsum(d_B[i][k], C_inv, d_B[j][k])

        # --- cross global x per-bin ---
        for i in range(N_A):
            for j in range(N_B):
                F_AB[i,j,k] += einsum(d_A[i][k], C_inv, d_B[j][k])

# symmetrize F_AA, F_BB blocks

# Schur complement marginalization (if requested)
if marginalize_per_bin:
    for k in range(N_bins):
        Fbb_inv_k = np.linalg.inv(F_BB[:,:,k])           # N_B x N_B inversion
        F_AA -= F_AB[:,:,k] @ Fbb_inv_k @ F_AB[:,:,k].T  # rank-N_B update

    return FisherMat(F_AA, ..., param_list=global_params, per_bin_errors=Fbb_inv_k_list)
else:
    # build full block matrix and return as large FisherMat
    full_F = assemble_block_matrix(F_AA, F_AB, F_BB, N_bins)
    return FisherMat(full_F, ..., param_list=expanded_param_list)
```

#### Strategy 1 (full matrix)

Expand `param_list` to `["fNL", "b_1_bin0", "b_1_bin1", ...]` before the existing
loop, then modify the sum: for per-bin parameter `b_1_binK` paired with any other
parameter, only add the contribution from bin `K`.  The existing einsum calls are
unchanged; only the bin-index loop gains an `if` guard.

### 4.4 `FisherMat` changes (fisher.py)

The Schur path returns an `(N_A × N_A)` matrix — `FisherMat` needs no structural
changes for this case.  Optionally store the per-bin errors as an extra attribute:

```python
self.per_bin_errors = per_bin_errors   # list of N_bins arrays, shape (N_B,)
```

The full-matrix path returns a larger matrix; `FisherMat` already handles arbitrary
sizes. The only ergonomic addition needed is a helper to extract the per-bin block for
a named parameter:

```python
def get_per_bin_error(self, param):
    """Return array of per-bin 1-sigma errors for a per-bin parameter."""
    ...
```

---

## 5. Summary of Changes Required

| Location | Change | Size |
|---|---|---|
| `forecast.py:get_fish` | Add `per_bin_params` and `marginalize_per_bin` args | ~5 lines |
| `forecast.py:get_fish` | Split derivs into `d_A` / `d_B` based on param classification | ~10 lines |
| `forecast.py:get_fish` (assembly) | Replace inner loop with Schur accumulation logic | ~40 lines |
| `fisher.py:FisherMat.__init__` | Store optional `per_bin_errors` attribute | ~5 lines |
| `fisher.py:FisherMat` | Add `get_per_bin_error(param)` helper | ~10 lines |

No changes needed to:
- `_precompute_derivatives_and_covariances` (derivatives already per-bin)
- `five_point_stencil` (unaware of global vs per-bin)
- `FisherList`, `Sampler`, `BasePosterior`
- Any power spectrum / bispectrum computation code

Total new code: approximately 70–100 lines.

---

## 6. Example Usage (proposed API)

```python
forecast = FullForecast(cosmo_funcs, N_bins=10)

# Global fNL constraint marginalised over per-bin b_1, b_2
fish = forecast.get_fish(
    param_list    = ["fNL", "n_s"],
    per_bin_params= ["b_1", "b_2"],
    pkln=[0],
    terms="NPP",
    marginalize_per_bin=True,   # Schur complement — returns (2x2) FisherMat
)
print(fish.get_error("fNL"))

# Per-bin b_1 errors (byproduct of Schur inversion)
print(fish.per_bin_errors["b_1"])   # array of length N_bins

# Or: keep full matrix to examine all correlations
fish_full = forecast.get_fish(
    param_list    = ["fNL", "n_s"],
    per_bin_params= ["b_1", "b_2"],
    pkln=[0],
    terms="NPP",
    marginalize_per_bin=False,  # returns (22x22) FisherMat for 10 bins
)
print(fish_full.get_error("b_1[3]"))
```
