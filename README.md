
# Classification is a Strong Baseline for Deep Metric Learning (BMVC '19)
Andrew Zhai, Hao-Yu Wu

## Paper ([https://arxiv.org/abs/1811.12649](https://arxiv.org/abs/1811.12649))
This repo contains the source code for our paper (WIP)

## Setup Repo
```
git clone https://github.com/azgo14/classification_metric_learning.git
```

The repo assumes that all data files exist under `/data1` directory locally as that local directly will be mounted ot `/data1` in the container.

## Running Commands
We provide a simple utility to make running commands in a docker container easier. This tool will automatically download the expected Docker image, mount the expected directories, and make running commands simple. To use the command, add `scripts/bin/pdoc` to your PATH variable as so:
```
export PATH=$PATH:<PATH_TO_REPO>/scripts/bin
```

### Example
You can then see that commands such as `pdoc nvidia-smi` will run `nvidia-smi` inside the docker container.

## Build Docker
To rebuild the docker image:

1) Initialize all submodules recursively via:
```
git submodule update --init --recursive
```

2) Build the image with
```
./docker/docker-build.sh ./docker/Dockerfile
```

## Datasets
Download the datasets with the following scripts. We assume data will live in the /data1 directory throughput our code
### CUB
```
./scripts/get_cub200_dataset.sh
```

### CARS
```
./scripts/get_cars196_dataset.sh
```

### Stanford Online Products
```
./scripts/get_stanford_products_dataset.sh
```

### In-Shop
Manual download raw data from http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html

Expected the following raw data files to exist
/data1/data/inshop/img.zip
/data1/data/inshop/list_eval_partition.txt

```
./scripts/get_inshop_dataset.sh
```

## Reproduction
### CUB
```
./scripts/bin/pdoc CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID python metric_learning/train_classification.py --dataset Cub200 --dim 2048 --model_name resnet50 --epochs_per_step 15 --num_steps 2 --test_every_n_epochs 5 --lr 0.001 --lr_mult 1 --class_balancing --images_per_class 25 --batch_size 75
```
(May differ slightly because of random seed)\
Raw Features: R@1, R@2, R@4, R@8: 65.36 & 76.76 & 85.42 & 91.51\
Binary Features: R@1, R@2, R@4, R@8: 63.67 & 75.37 & 84.54 & 90.99


### CARS
```
./scripts/bin/pdoc CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID python metric_learning/train_classification.py --dataset Cars196 --dim 2048 --model_name resnet50 --epochs_per_step 15 --num_steps 2 --test_every_n_epochs 5 --lr 0.01 --lr_mult 1 --class_balancing --images_per_class 25 --batch_size 75
```
(May differ slightly because of random seed)\
Raw Features: R@1, R@2, R@4, R@8: 89.50 & 94.18 & 96.84 & 98.41\
Binary Features: R@1, R@2, R@4, R@8: 89.29 & 93.95 & 96.61 & 98.14


### Stanford Online Products
```
./scripts/bin/pdoc CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID python metric_learning/train_classification.py --dataset StanfordOnlineProducts --dim 2048 --model_name resnet50 --epochs_per_step 15 --num_steps 2 --test_every_n_epochs 5 --lr 0.01 --lr_mult 1 --class_balancing --images_per_class 5 --batch_size 75
```
(May differ slightly because of random seed)\
Raw Features: R@1, R@10, R@100, R@1000: 79.55 & 91.54 & 96.66 & 98.95\
Binary Features: R@1, R@10, R@100, R@1000: 78.03 & 90.71 & 96.24 & 98.72


### In-Shop
```
./scripts/bin/pdoc CUDA_VISIBLE_DEVICES=5 CUDA_DEVICE_ORDER=PCI_BUS_ID python metric_learning/train_classification.py --dataset InShop --dim 2048 --model_name resnet50 --epochs_per_step 15 --num_steps 2 --test_every_n_epochs 5 --lr 0.01 --lr_mult 1 --class_balancing --images_per_class 5 --batch_size 75
```
(May differ slightly because of random seed)\
Raw Features: R@1, R@10, R@20, R@30, R@40, R@50: 89.35 & 97.81 & 98.61 & 98.87 & 99.05 & 99.13\
Binary Features: R@1, R@10, R@20, R@30, R@40, R@50: 88.76 & 97.65 & 98.47 & 98.73 & 98.94 & 99.05

## References / re-implementations
- The Computer Vision Best Practices repository: [02_state_of_the_art.ipynb](https://github.com/microsoft/computervision-recipes/blob/master/scenarios/similarity/02_state_of_the_art.ipynb)
