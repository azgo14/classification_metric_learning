#!/bin/bash -ex
#
# Get the CARS196 dataset
#

DATA_DIR=/data1/data/cars196

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist, will create one.";
  mkdir -p ${DATA_DIR}
fi

wget -P ${DATA_DIR} http://imagenet.stanford.edu/internal/car196/car_ims.tgz
wget -P ${DATA_DIR} http://imagenet.stanford.edu/internal/car196/cars_annos.mat

cd ${DATA_DIR};
tar -xzf car_ims.tgz
