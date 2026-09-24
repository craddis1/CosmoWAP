"""
Compute the Planck 2018 parameter covariances hardcoded in BasePosterior.planck_cov.

Source: COM_CosmoParams_base-plikHM_R3.01.zip from the Planck Legacy Archive
(https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/), 471 MB.
Only base/plikHM_TTTEEE_lowl_lowE_lensing/ is needed:
    base_plikHM_TTTEEE_lowl_lowE_lensing            - CMB only (default prior)
    base_plikHM_TTTEEE_lowl_lowE_lensing_post_BAO   - CMB + BAO (bao=True)

Usage:
    python scripts/planck_prior_cov.py <path to base/plikHM_TTTEEE_lowl_lowE_lensing>

Omega_m is taken as Omega_b + Omega_cdm (our CLASS setup has massless neutrinos) rather than
Planck's omegam, which includes the 0.06 eV neutrino - so Omega_m = Omega_b + Omega_cdm holds
exactly in the matrix. Only the covariance is used; the prior is centred on the fiducial.
"""

import sys

import numpy as np
from getdist import loadMCSamples

PARAMS = ["Omega_m", "Omega_b", "Omega_cdm", "h", "ln_A_s", "n_s", "sigma8"]


def planck_cov(root):
    s = loadMCSamples(root, settings={"ignore_rows": 0.3})  # PLA chains include burn-in
    p = s.getParams()
    h = p.H0 / 100
    Omega_b = p.omegabh2 / h**2
    Omega_cdm = p.omegach2 / h**2
    x = np.array([Omega_b + Omega_cdm, Omega_b, Omega_cdm, h, p.logA, p.ns, p.sigma8])
    return np.cov(x, aweights=s.weights)


if __name__ == "__main__":
    base = sys.argv[1].rstrip("/")
    np.set_printoptions(precision=8, linewidth=200)
    for name, root in [
        ("cmb", f"{base}/base_plikHM_TTTEEE_lowl_lowE_lensing"),
        ("cmb+bao", f"{base}/base_plikHM_TTTEEE_lowl_lowE_lensing_post_BAO"),
    ]:
        cov = planck_cov(root)
        print(f"# {name}: {PARAMS}")
        print("sigma:", np.sqrt(np.diag(cov)))
        print(repr(cov), "\n")
