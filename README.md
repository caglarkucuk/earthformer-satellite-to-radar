# Earthformer for satellite-to-radar nowcasting (EF Sat2Rad)

## Introduction

Welcome to the Earthformer for satellite-to-radar nowcasting (EF Sat2Rad) repository, which is accompanying the paper [Transformer-based nowcasting of radar composites from satellite images for severe weather](https://doi.org/10.48550/arXiv.2310.19515)

EF Sat2Rad is a modified version of the [Earthformer](https://github.com/amazon-science/earth-forecasting-transformer) package, tailored for radar nowcasting from satellite data by using [SEVIR](https://github.com/MIT-AI-Accelerator/neurips-2020-sevir) dataset. 

## Installation and Setup

Installation and setting up the data involves a couple of steps:

### 0) Requirements and Versions
- CUDA: To use GPU. CUDA 11.6 was available in the machines we ran the experiments with. 
- AWS CLI: To download data from SEVIR buckets comfortably.

### 1) Clone the repository and jump into the main directory
```bash
cd
git clone https://github.com/caglarkucuk/earthformer-satellite-to-radar
cd earthformer-satellite-to-radar
```

### 2) Create the environment and start using it via:
Once you're in the main directory:
```bash
conda create --name ef_sat2rad --file ef-sat2rad/Preprocess/ef_sat2rad.txt
conda activate ef_sat2rad
```

### 3) Download and preprocess the data:
- It is possible to use sample data provided in [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10033640), just unzip the archive and copy the content to `data/train` and `data/test`.

- For the full dataset, follow the steps below:
```bash
# Download the bulk data from SEVIR buckets (careful with disk space)
aws s3 sync --no-sign-request s3://sevir/data/lght bulk_data/lght # 1.7 GB
aws s3 sync --no-sign-request s3://sevir/data/ir069 bulk_data/ir069 # 46 GB
aws s3 sync --no-sign-request s3://sevir/data/ir107 bulk_data/ir107 # 46 GB
aws s3 sync --no-sign-request s3://sevir/data/vil bulk_data/vil # 110 GB
aws s3 cp --no-sign-request s3://sevir/CATALOG.csv bulk_data/CATALOG.csv

# Process the bulk data and save events separately (~90 GB) via (will take take some time)
python ef-sat2rad/Preprocess/save_oneByOne.py -dirBase bulk_data/ -dirOut data/ -dataMod test  # 23 GB
python ef-sat2rad/Preprocess/save_oneByOne.py -dirBase bulk_data/ -dirOut data/ -dataMod train  # 64 GB

# Consider removing the bulk data after preprocessing ;)
# rm -r bulk_data/{lght,ir069,ir107,vil}
```

## Running the model
In order to run the trained model right away, download the pretrained weights provided in [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10033640) and copy the unzipped file `ef_sevir_sat2rad.pt` into `trained_ckpt`. 

Afterwards run the chunk below and it'll make prediction on the test samples available in `data/test`:
```bash
cd ef-sat2rad
python train_cuboid_sevir_invLinear.py --pretrained
```
It is possible to use the model for inference on machines without a GPU. It takes ~2 seconds to predict one sample on a CPU with 32 threads! Though, it is not advised to train the model without an accelarator, e.g., GPU.

In order to train the model from scratch, run:
```bash
cd ef-sat2rad
python train_cuboid_sevir_invLinear.py
```
to train the model with default parameters and original structure described in the [paper](PAPER_URL). Alternating model structure by creating new config files is the best way to experiment further.

## Credits
This repository is built on top of the great repositories:
- [Earthformer](https://github.com/amazon-science/earth-forecasting-transformer) for the original Space-Time Transformer architecture. Most of the content in the `earthformer` directory is a simplified version of the original repository. New additions are highlighted with comments.  
- [SEVIR](https://github.com/MIT-AI-Accelerator/neurips-2020-sevir) for downloading and preprocessing the data.

## Cite
Please cite us if this repo helps your work!
```
@article{Kucuk2023,
   title = {Transformer-based nowcasting of radar composites from satellite images for severe weather},
   author = {K\"u\c{c}\"uk, {\c{C}}a\u{g}lar and Giannakos, Apostolos and Schneider, Stefan and Jann, Alexander},
   month = {10},
   doi = {10.48550/arXiv.2310.19515},
   year = {2023},
}
``` 

## Licence
GNU General Public License v3.0
