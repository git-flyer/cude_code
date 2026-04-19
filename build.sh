#!/bin/bash
set -e

# Load CUDA module
module load compiler/cuda/12.9.1

# Set CUDA paths
export CUDA_HOME=/public/software/compiler/cuda/cuda-12.9.1
export PATH=$CUDA_HOME/bin:$PATH

# Build directory
BUILD_DIR="build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure and build
cmake .. -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86
make -j$(nproc)
