"""
Example run:
mpirun -n 20 python3 mpi_mcmc.py
"""

from mpi4py import MPI

import cosmo_wap as cw
from cosmo_wap.lib import utils

# --- MPI setup ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # The process ID (e.g., 0, 1, 2, ...)
size = comm.Get_size() # The total number of processes

# --- Each process does some work ---

#get planck comsology
cosmo = utils.get_cosmo(k_max=10)
#load preset surveys
survey_params = cw.SurveyParams(cosmo)

# get main class object - this is dependent on cosmology and survey so we will gather for different surveys
cosmo_funcs = cw.ClassWAP(cosmo,survey_params.Euclid,compute_bias=False,emulator=True)
# define k_max_func for forecasts
kmax_func = lambda zz: 0.1 *cosmo_funcs.h*(1+zz)**(2/(2+cosmo_funcs.n_s))
pkln=[0,2]

forecast = cw.forecast.FullForecast(cosmo_funcs,kmax_func=kmax_func,N_bins=10)
sampler = forecast.sampler(['A_s','n_s','Omega_cdm','Omega_b','fNL'],terms=['Loc','NPP'],bias_list=['IntInt'],pkln=pkln,R_stop=0.005,planck_prior=True)
sampler.run()

# ONLY rank 0 has the full results and is allowed to write to the disk
if rank == 0:
    sampler.save("sampler.pkl")
