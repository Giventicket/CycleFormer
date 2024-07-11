# Use the official CUDA image as a parent image
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# Set the working directory
WORKDIR /workspace

# Install Python, pip and git
RUN apt-get update && \
    apt-get install -y python3 python3-pip git

# Install Python packages
RUN pip3 install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install tensorboard==2.16.2 && \
    pip3 install pytorch-lightning==2.1.2 && \
    pip3 install tsplib95 && \
    pip3 install 'pyconcorde @ git+https://github.com/jvkersch/pyconcorde' && \
    pip3 install flash-attn --no-build-isolation

# install LKH solver, some errors in repository currently
# RUN wget http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.6.tgz && \
#     tar xvfz LKH-3.0.6.tgz && \
#     cd LKH-3.0.6 && \
#     make && \
#     cp LKH /usr/local/bin && \
#     cd .. && \
#     rm -rf LKH-3.0.6

# Clone CycleFormer repository
RUN git clone https://github.com/Giventicket/CycleFormer.git

# Set the default command to bash
CMD ["bash"]
