#!/bin/bash

set -e


ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )

GCC_COMPILER=${ROOT_PWD}/gcc-arm-12.3-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu
# build rockx
BUILD_DIR=${ROOT_PWD}/build

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake .. \
    -DCMAKE_C_COMPILER=${GCC_COMPILER}-gcc \
    -DCMAKE_CXX_COMPILER=${GCC_COMPILER}-g++ \
    -DSEETA_AUTHORIZE=ON
make -j4

