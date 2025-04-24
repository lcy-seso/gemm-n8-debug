#!/bin/bash

if [ ! -d "build" ]; then
    mkdir build
fi

cd build

if [ -f "CMakeCache.txt" ]; then
    rm CMakeCache.txt
fi

if [ -d "CMakeFiles" ]; then
    rm -rf CMakeFiles
fi

if [ -f "gemm-n8" ]; then
    rm gemm-n8
fi

cmake ..

make 2>&1 | tee ../build.log

if [ -f "gemm-n8" ]; then
    echo "build success"
    ./gemm-n8 2>&1 | tee ../run.log
else
    echo "build failed"
fi

cd ..
