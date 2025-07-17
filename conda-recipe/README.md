

So conda is stricter with dependencies clashes than pip.
And there is a clash with chainconsumer and cosmopower - hopefully should be solved in newer cosmopower update - it needs an older version of tensorflow is the root of the problem. So the solution is to remove chainconsumer from the build and just install it with pip later.

so inside the conda-recipe/ folder (classy has a local build as it does not have a conda install). So need to get up to date .yaml files. For PyPI packages
Use grayskull: grayskull pypi --package-name--

Then after that build classy first: conda build classy/

then build cosmowap with conda-forge channel:

conda build . -c conda-forge