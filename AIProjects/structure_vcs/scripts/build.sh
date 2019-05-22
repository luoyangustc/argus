#!/bin/bash
set -ex

cur_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd ${cur_dir}/.. && pwd)"

if [ ! -d "${root_dir}/build" ]; then
    mkdir -p "${root_dir}/build"
fi

cd "${root_dir}/build"
cmake_args=""
if ! ${USE_MOCK_VSS}; then
    cmake_args="${cmake_args} -DUSE_MOCK_VSS=OFF "
fi
cmake .. ${cmake_args}
make

if ${USE_MOCK_VSS}; then
    # some test cases based on mock lib logic
    make test ARGS="-V"
fi

if ! ${USE_MOCK_VSS}; then
    make install
fi