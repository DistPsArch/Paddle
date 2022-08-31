export PYTHONPATH=/opt/_internal/cpython-3.7.0/lib/python3.7/:${PATH}
export LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib:$LD_LIBRARY_PATH

#export http_proxy=http://10.197.123.16:8128
#export https_proxy=http://10.197.123.16:8128

export https_proxy=http://172.19.57.45:3128/
export http_proxy=http://172.19.57.45:3128/

#unset http_proxy && unset https_proxy

echo $1
if [[ "$1" == "cmake" ]]; then
        #-DCMAKE_CXX_FLAGS=-Wl,-rpath=/opt/compiler/gcc-8.2/lib64:/usr/lib64 \
        #-DCUDNN_ROOT=$CUDNN_ROOT \
        #-DNCCL_ROOT=$NCCL_ROOT \
        #-DNCCL_INCLUDE_DIR=$NCCL_ROOT/include \
        #-DCMAKE_PREFIX_PATH=/usr/lib64 \
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=./output/ \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_PYTHON=ON \
        -DWITH_MKL=OFF -DWITH_GPU=ON -DWITH_FLUID_ONLY=ON \
        -DPY_VERSION=3.7 \
        -DPYTHON_INCLUDE_DIR=/opt/_internal/cpython-3.7.0/include/python3.7m \
        -DPYTHON_LIBRARY=/opt/_internal/cpython-3.7.0/lib/libpython3.7m.so \
        -DPYTHON_EXECUTABLE=/opt/_internal/cpython-3.7.0/bin/python \
        -DWITH_DISTRIBUTE=ON \
        -DWITH_GLOO=ON \
        -DWITH_PSLIB=ON \
        -DWITH_PSLIB_BRPC=OFF \
        -DWITH_PSCORE=OFF \
        -DWITH_HETERPS=ON \
        -DCUDA_ARCH_NAME=All
fi
make -j128
#make
