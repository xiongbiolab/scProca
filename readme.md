# scProca - Integrate and generate single-cell proteomics from transcriptomics with cross-attention

[![Documentation Status](https://readthedocs.org/projects/scproca/badge/?version=latest)](https://scproca.readthedocs.io/en/latest/?badge=latest)

`scProca` is a package designed to integrate and generate single-cell proteomics from transcriptomics, implemented in PyTorch.

![scProca](docs/source/scProca.png)

(A) Schematic representation of scProca within the framework of deep generative models.  
(B) The variational auto-encoder with cross-attention introduced in scProca.

---

## Installation

1. Install Conda and create a virtual environment with `python==3.11`:

   ```bash
   conda create -n scProca python==3.11
   conda activate scProca
   ```

2. Install [PyTorch](https://pytorch.org) in the virtual environment. If you have an NVIDIA GPU, make sure to install a version of PyTorch that supports it. PyTorch performs much faster with an NVIDIA GPU. For maximum compatibility, we currently recommend installing `pytorch==2.3.1`.

3. Install scProca from GitHub:

   ```bash
   git clone git://github.com/xiongbiolab/scProca.git
   cd scProca
   pip install .
   ```

---

## Documentation

Detailed usage documentation is available at [https://scProca.readthedocs.io/](https://scProca.readthedocs.io/).

---

## Reproducibility

Replication code for the research paper is available at [https://github.com/ZzzsHuqiaAao/scProca-reproducibility](https://github.com/ZzzsHuqiaAao/scProca-reproducibility).
