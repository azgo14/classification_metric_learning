FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

MAINTAINER Andrew Zhai <andrew@pinterest.com>

ARG BUILD_JOBS=10
ENV BUILD_JOBS ${BUILD_JOBS}


RUN apt-get -q -y update && apt-get install --force-yes -q -y \
        build-essential \
        curl \
        vim \
        git \
        wget \
        libatlas-base-dev \
        libgtest-dev \
        libmysqlclient-dev \
        libssl-dev \
        libxml2-dev \
        libxslt1-dev \
        libjpeg8-dev \
        libexiv2-dev \
        zlib1g-dev \
        libffi-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        liblz4-dev \
        gfortran \
        libopencv-dev \
        libsnappy-dev \
        unzip \
        python-dev \
        python-tk \
        python-setuptools \
        libiomp-dev \
        libopenmpi-dev \
        openmpi-bin \
        openmpi-doc \
        libmagickwand-dev \
        unzip \
        dh-autoreconf \
        cmake \
        libnccl2=2.2.12-1+cuda9.1 \
        libnccl-dev=2.2.12-1+cuda9.1

RUN easy_install pip
RUN pip install \
    pip==9.0.1 \
    h5py==2.7.0

# libprotobuf. We do this instead of apt-get to ensure python bindings and binaries are same version
RUN cd /opt && \
    wget https://github.com/google/protobuf/archive/v3.5.2.tar.gz && \
    tar -xzf v3.5.2.tar.gz && \
    (cd protobuf-3.5.2 && \
      ./autogen.sh && \
      CPPFLAGS="-fPIC" ./configure && \
      make -j ${BUILD_JOBS} && \
      make install && \
      ldconfig) && \
    (cd protobuf-3.5.2/python/ && \
      python setup.py install) && \
    rm -rf protobuf-3.5.2

# Caffe2 + Pytorch
ENV PYTORCH_ROOT=/opt/pytorch
COPY third_party/pytorch $PYTORCH_ROOT
ENV LD_PRELOAD=/usr/lib/libmpi_cxx.so

RUN cd $PYTORCH_ROOT \
    && pip install -r requirements.txt \
    && TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    FULL_CAFFE2=1 python setup.py install

# torchvision
ENV VISION_ROOT=/opt/vision
COPY third_party/vision $VISION_ROOT

RUN cd $VISION_ROOT && \
    python setup.py install

# faiss
RUN apt-get install --force-yes -q -y libopenblas-dev

ENV BLASLDFLAGS=/usr/lib/libopenblas.so.0

RUN cd /opt && \
    git clone https://github.com/facebookresearch/faiss.git && \
    (cd faiss && \
      git reset --hard 87721af1294c0dc2008d0537d9082198a477ac3a && \
      ./configure && \
      make -j ${BUILD_JOBS} && \
      make -j ${BUILD_JOBS} install && \
      make -C gpu -j ${BUILD_JOBS} && \
      make -C gpu/test -j ${BUILD_JOBS} && \
      make -C python gpu -j ${BUILD_JOBS} && \
      make -C python build -j ${BUILD_JOBS} && \
      make -C python install -j ${BUILD_JOBS})

ENV PYTHONPATH=$PYTHONPATH:/opt/faiss/python

# requirements
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt && rm requirements.txt
