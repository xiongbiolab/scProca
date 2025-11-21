# scProca - A Cross-Attention-Enhanced Deep Generative Model for Single-Cell Transcriptomics and Proteomics Integration and Imputation

[![Documentation Status](https://readthedocs.org/projects/scproca/badge/?version=latest)](https://scproca.readthedocs.io/en/latest/?badge=latest)

`scProca` is a package designed to integrate and generate single-cell proteomics from transcriptomics, implemented in PyTorch.

---

## News


<span style="color:red;font-size:20px; font-weight:bold;">2025-9-29: Now published in IEEE Journal of Biomedical and Health Informatics. https://doi.org/10.1109/JBHI.2025.3615771</span>


---

## Introduction

![scProca](docs/source/scProca.png)
Overview of scProca.  
(A) Schematic representation of scProca within the framework of deep generative models.  
(B) The variational auto-encoder with cross-attention introduced in scProca.

![examples](docs/source/examples.png)
UMAP visualization of the low-dimensional representations obtained by scProca on the SLN dataset, colored by batch annotation and cell types.   
(A-C) Only SLN111-D1 serves as CITE-seq data, while the others serve as scRNA-seq data.  
(B-D) Both SLN111-D1 and SLN111-D2 serve as CITE-seq data, while both SLN208-D1 and SLN208-D2 serve as scRNA-seq data. 

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

Detailed usage documentation is available at https://scProca.readthedocs.io.

---

## Reproducibility

Replication code for the research paper is available at https://github.com/ZzzsHuqiaAao/scProca-reproducibility.

---


## Citation

```bibtex
@article{xiong2025scproca,
  author={Xiong, Jiankang and Zheng, Shuqiao and Gong, Fuzhou and Ma, Liang and Wan, Lin},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={{scProca}: A Cross-Attention-Enhanced Deep Generative Model for Single-Cell Transcriptomics and Proteomics Integration and Imputation}, 
  year={2025},
  pages={1-11},
  keywords={Proteomics;Proteins;Transcriptomics;RNA;Mathematical models;Biomedical measurement;Sequential analysis;Imputation;Data models;Training;Multi-omics integration;Single-cell imputation;Deep generative model;Attention mechanisms;Transcriptomics and proteomics},
  doi={10.1109/JBHI.2025.3615771}
}
