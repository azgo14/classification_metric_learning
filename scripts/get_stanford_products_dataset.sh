#!/bin/bash -ex
#
# Get the Stanford Online Product dataset
#

DATA_DIR=/data1/data/stanford_products

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist, will create one.";
  mkdir -p ${DATA_DIR}
fi

wget -P ${DATA_DIR} ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip
cd ${DATA_DIR}; unzip Stanford_Online_Products.zip