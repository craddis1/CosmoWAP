

So conda is stricter with dependencies clashes than pip.
And there is a clash with chainconsumer and cosmopower - hopefully should be solved in newer cosmopower update - it needs an older version of tensorflow is the root of the problem. So the solution is to remove chainconsumer from the build and just install it with pip later.

so inside the conda-recipe/ folder (classy has a local build as it does not have a conda install). So need to get up to date .yaml files. For PyPI packages
Use grayskull: grayskull pypi --package-name--

Then after that build classy first: conda build classy/

then build cosmowap with conda-forge channel:

conda build . -c conda-forge




### Ok so just to to install in a conda environment 

Create python3.11 environment

inside:

conda install -c conda-forge c-compiler cxx-compiler
pip install --extra-index-url https://pypi.anaconda.org/craddis1/simple cosmowap
python -m pip install "mpi4py>=3" --upgrade --no-binary :all:
pip install cosmopower # so go for this rather than the conda version


pip install -e .



mpirun -n 20 python3 mpi_mcmc.py