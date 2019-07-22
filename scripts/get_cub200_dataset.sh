#!/bin/bash -ex
#
# Get the CUB200_2011 dataset
#

DATA_DIR=/data1/data/cub200

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist, will create one.";
  mkdir -p ${DATA_DIR}
fi

wget -P ${DATA_DIR} http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
cd ${DATA_DIR}; tar -xf CUB_200_2011.tgz
