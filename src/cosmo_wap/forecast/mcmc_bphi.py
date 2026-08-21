"""
P(k) + B(k) MCMC - numeric-mu integrated kernels, per-bin
nuisance parameters and fast/slow dragging.

Run uses compiled bk tables, CosmoPower emulator for the cosmology rebuild, and cobaya's dragging to keep the fast block cheap.

Example run:
mpirun -n 20 python3 mcmc_bphi.py
"""

import os

# both are read once at import time, so they have to be set before cosmo_wap comes in.
os.environ.setdefault("COSMOWAP_BK_TABLE", "1")  # compiled redshift-monomial bk tables
os.environ.setdefault("COSMOWAP_BK_TABLE_MB", "128")  # per-process budget - n_ranks x this has to fit in RAM

from mpi4py import MPI

import cosmo_wap as cw
from cosmo_wap.lib import utils

# --- MPI setup ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # The process ID (e.g., 0, 1, 2, ...)

# get planck cosmology
cosmo = utils.get_cosmo(k_max=1)
# load preset surveys
survey_params = cw.SurveyParams()

# emulator=True keeps the cosmology rebuild (the slow block) off CLASS
cosmo_funcs = cw.ClassWAP(cosmo, survey_params.MegaMapper(cosmo), compute_bias=True, emulator=True, verbose=False)

# nonlinear scale for the power spectrum, tree-level cut for the bispectrum
kmax_func = 0.15
bkmax_func = 0.1
forecast = cw.forecast.FullForecast(cosmo_funcs, kmax_func=kmax_func, bkmax_func=bkmax_func, s_k=4, N_bins=5)

params = ["fNL_loc", "Omega_m", "ln_A_s", "n_s", "Omega_b", "h", "A_loc_b_11"]

sampler = forecast.sampler(
    params,
    terms=["WAGR", "RRGR"],  # analytic WS terms
    bk_terms=["NPP", "GR1", "GR2", "WAGR", "RRGR", "Loc"],
    pkln=[0, 2, 4],
    bkln=[0, 1, 2],
    kernels=["N", "LP", "I", "Loc"],  # numeric-mu pk kernels summed onto `terms`
    mu_grid=None,  # [n_mu, GL, los_n, deg, n_p]; None -> [48, True, 8, 8] and the sampler's n_p=1000
    per_bin_params=["b_1", "b_phi_e", "Q"],  # one marginalised amplitude per redshift bin
    R_stop=0.005,
    planck_prior=True,
    max_tries=10000,
    fisher_covmat=True,
)
# fisher_covmat=True and drag=True are the defaults: the Fisher gives cobaya its proposal
# covmat, and dragging splits Omega_m/ln_A_s (cosmology rebuild) from the rest.
sampler.run(output="chains/bphi_MM")

# ONLY rank 0 has the full results and is allowed to write to the disk
if rank == 0:
    sampler.save("sampler_bphi_MM.pkl")
