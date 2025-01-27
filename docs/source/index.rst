===================================
Welcome to scProca's documentation!
===================================
.. image:: https://readthedocs.org/projects/scproca/badge/?version=latest
    :target: https://scproca.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

**scProca** is a package designed to integrate and generate single-cell proteomics from transcriptomics, implemented in PyTorch.

.. image:: scProca.png
   :align: center

(A) Schematic representation of scProca within the framework of deep generative models. scProca is capable of inferring batch-corrected, integrated latent variables from scRNA-seq and CITE-seq data, and generating the expression profiles of ADT for scRNA-seq cells.
(B) The variational auto-encoder with cross-attention introduced in scProca. Cross-attention is used to incorporate CITE-seq cells as references, completing representation of scRNA-seq cells in the ADT embedding space.

.. toctree::
   :maxdepth: 1
   :caption: Installation and Reproducibility:

   pre

.. toctree::
   :maxdepth: 1
   :caption: Jupyter notebooks for examples:

   Integrate and generate CITE-seq PBMC Datasets from scRNA-seq PBMC Datasets
   Integrate and generate CITE-seq SLN Datasets from scRNA-seq SLN Datasets
   Integrate and generate CITE-seq SLN Datasets from scRNA-seq SLN Datasets using experimental batches

.. toctree::
   :maxdepth: 1
   :caption: Main functions and parameters:

   api

