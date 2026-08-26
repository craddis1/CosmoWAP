"""
Example to forecast local-type fnl constraints for a given survey

Example run:
mpirun -n 10 python3 mpi_mcmc.py - to run on 10 cores
"""

from mpi4py import MPI

import cosmo_wap as cw
from cosmo_wap.lib import utils

# --- MPI setup ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # The process ID (e.g., 0, 1, 2, ...)
size = comm.Get_size()  # The total number of processes

# --- Each process does some work ---

# get planck comsology
cosmo = utils.get_cosmo(k_max=10)
# load preset surveys
survey_params = cw.SurveyParams(cosmo)

# get main class object - this is dependent on cosmology and survey so we will gather for different surveys
cosmo_funcs = cw.ClassWAP(cosmo, survey_params.Euclid, compute_bias=False, emulator=True)

# define k-cutoff scale
kmax_func = 0.15
bkmax_func = 0.1
forecast = cw.forecast.FullForecast(cosmo_funcs, kmax_func=kmax_func, bkmax_func=bkmax_func, s_k=4, N_bins=5)

params = ["fNL_loc", "Omega_m", "ln_A_s", "n_s", "Omega_b", "h"]

sampler = forecast.sampler(
    params,
    terms=["WAGR", "RRGR"],  # analytic WS terms
    bk_terms=["NPP", "GR1", "GR2", "WAGR", "RRGR", "Loc"],
    pkln=[0, 2, 4],
    bkln=[0, 1, 2],
    kernels=["N", "LP", "I", "Loc"],  # numeric-mu pk kernels summed onto `terms`
    mu_grid=None,  # [n_mu, GL, los_n, deg, n_p]; None -> [48, True, 8, 8] and the sampler's n_p=1000
    per_bin_params=["b_1", "b_e", "Q"],  # one marginalised amplitude per redshift bin
    R_stop=0.005,
    planck_prior=True,  # use planck 2018 constraints as a prior on cosmological parameters
    max_tries=10000,
    fisher_covmat=True,  # seed MCMC proposals with fisher run
)
# fisher_covmat=True and drag=True are the defaults: the Fisher gives cobaya its proposal
# covmat, and dragging splits Omega_m/ln_A_s (cosmology rebuild) from the rest.
sampler.run(output="chains/bphi")

# ONLY rank 0 has the full results and is allowed to write to the disk
if rank == 0:
    sampler.save("sampler.pkl")
