#!/bin/bash -ex
#
# Get the In-Shop dataset
#

DATA_DIR=/data1/data/inshop

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist, will create one.";
  mkdir -p ${DATA_DIR}
fi

# Pretty annoying but you have to download the datasets manually and put them into /data1/data/inshop
# There's no direct download link to the dataset from what we could find.

# Expected files to exist
#   /data1/data/inshop/img.zip
#   /data1/data/inshop/list_eval_partition.txt

cd ${DATA_DIR};
unzip img.zip
