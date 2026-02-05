Installation Guide
==================

Dependencies
------------

**Core Requirements:**

* ``numpy``
* ``scipy``
* ``classy``
* ``matplotlib``
* ``tqdm``
* ``cython``
* ``ChainConsumer``
* ``cobaya``

**Optional:**

* ``CosmoPower``

.. note::
   To use the ``CosmoPower`` integration, it is recommended to use Python 3.10 or 3.11. You will need to install it separately. Please ensure you are using version ``0.2.0``.

   .. code-block:: bash

      pip install cosmopower==0.2.0


Installing via `pip`
--------------------

Requires at least Python >=3.10 for functionality.

To install the "stable" version of CosmoWAP, simply run:

.. code-block:: bash

    python -m pip install cosmowap


Install development version
---------------------------


1. Clone the repository:

   .. code-block:: bash

       git clone https://github.com/craddis1/CosmoWAP.git

2. Navigate into the cloned directory:

   .. code-block:: bash

       cd CosmoWAP

3. Install the package using `pip`:

   .. code-block:: bash

       pip install -e .




