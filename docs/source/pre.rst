------------
Installation
------------

1. Install Conda and create a virtual environment with *python==3.11*:

   .. code-block:: bash

      conda create -n scProca python==3.11
      conda activate scProca

2. Install `PyTorch <https://pytorch.org>`_ in the virtual environment. If you have an NVIDIA GPU, make sure to install a version of PyTorch that supports it. PyTorch performs much faster with an NVIDIA GPU. For maximum compatibility, we currently recommend installing *pytorch==2.3.1*.

3. Install scProca from GitHub:

   .. code-block:: bash

      git clone git://github.com/xiongbiolab/scProca.git
      cd scProca
      pip install .

---------------
Reproducibility
---------------

Replication code for the research paper is available at https://github.com/ZzzsHuqiaAao/scProca-reproducibility .
