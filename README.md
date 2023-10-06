# HyperGALE

HyperGALE is the open source implementation of ICASSP submitted paper [HyperGALE: ASD Classifcation via Hypergraph Gated Attention with Learnable Hyperedges](link) 

![language](https://img.shields.io/github/languages/top/mehular0ra/HyperGALE?color=lightgrey)
![lines](https://img.shields.io/tokei/lines/github/mehular0ra/HyperGALE?color=red)
![license](https://img.shields.io/github/license/mehular0ra/HyperGALE)


## Dataset

Download the ABIDE-2 dataset from [here](http://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html).

Resting state fMRI data is preprocessed using Version 1 of Schaefer2018 parecellation with #ROIs=400. Details about about parcellations can be found [here](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal).


## Usage

1. Change the *path* attribute in file *source/conf/dataset/fc_abide2.yaml* to the path of your dataset.

2. Run the following command to train the model.

```bash
python -m source --multirun model=hypergale,hypergraphgcnv2,hypergraphgcn,gcn,gat,graphsage dataset=fc_abide2 repeat_time=5
```

- **model**, default=(hypergale,hypergraphgcnv2,hypergraphgcn,gcn, gat, graphsage). Which model to use. The value is a list of model names. 

- **repeat_time**, default=5. How many times to repeat the experiment. The value is an integer. For example, 5 means repeat 5 times.

## Installation

```bash
conda create --name hypergel python=3.11
pip install torch
pip install pytorch-lightning
pip install torch-sparse torch-cluster torch-geometric

pip install pandas
pip install scikit-learn
pip install sicipy
pip install sympy
pip install matplotlib
pip install seaborn

pip install nilearn
pip install hydra-core
pip install omegaconf
pip install wandb
pip install ipdb

```


## Dependencies

  - python=3.11
  - cudatoolkit=11.10
  - torch==2.0.1
  - torch-cluster==1.6.1
  - torch-geometric==2.3.1
  - torch-sparse==0.6.17
  - torchmetrics==1.1.0
  - torchvision==0.15.2
  - pytorch-lightning==2.0.7
  - numpy==1.24.2
  - pandas==2.0.1
  - scikit-learn==1.2.2
  - scipy==1.10.1
  - seaborn==0.12.2
  - sympy==1.11.1
  - nilearn==0.10.1
  - hydra-core==1.3.2
  - omegaconf==2.3.0
  - wandb==0.15.0
  - ipdb==0.13.13




