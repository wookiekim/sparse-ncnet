source /root/.zshrc && conda activate me

mkdir /root/dep

# Install MinkowskiEngine v0.5
cd /root/dep \
     && git clone https://github.com/NVIDIA/MinkowskiEngine.git \
     && cd MinkowskiEngine \
     && MAX_JOBS=8 python setup.py install --cuda_home=/usr/local/cuda/ --force_cuda

# Install Faiss
cd /root/dep \
     && git clone https://github.com/facebookresearch/faiss.git \
     && cd faiss \
     && git checkout 81b1aee \
     && cmake -B build . \
     && make -C build -j \
     && cd build/faiss/python \
     && python setup.py install
