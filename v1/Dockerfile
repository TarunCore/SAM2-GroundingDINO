# Use an official Python runtime as a parent image
# FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/app
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y wget


# Set the working directory
WORKDIR $HOME

# Install system dependencies
# After the initial FROM and ENV statements, but before the WORKDIR command:
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    && wget https://bootstrap.pypa.io/get-pip.py \
    && python3.11 get-pip.py \
    && rm get-pip.py \
    && ln -s /usr/bin/python3.11 /usr/local/bin/python3 \
    && ln -s /usr/bin/python3.11 /usr/local/bin/python

# Then make sure pip commands use python3.11
RUN python3.11 -m pip install --upgrade pip

WORKDIR $HOME
# Install Python dependencies
COPY pytorch_requirements.txt $HOME/
RUN python3.11 -m pip install --no-cache-dir -r pytorch_requirements.txt
# COPY requirements.txt $HOME/
# RUN pip install --no-cache-dir -r requirements.txt

# ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX;8.9;9.0"
ENV CUDA_HOME=/usr/local/cuda
# Install GroundingDINO and SAM
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git $HOME/GroundingDINO
WORKDIR $HOME/GroundingDINO
# RUN git checkout -q 57535c5a79791cb76e36fdb64975271354f10251
RUN python3.11 -m pip install -v -e .

# Download model weights
RUN mkdir -p $HOME/weights
WORKDIR $HOME/weights
RUN wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
# COPY groundingdino_swint_ogc.pth $HOME/weights
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# COPY sam_vit_h_4b8939.pth $HOME/weights


WORKDIR $HOME
RUN pip install 'git+https://github.com/facebookresearch/segment-anything.git'
# RUN pip install supervision==0.6.0
RUN pip install roboflow


# Set up data directory
RUN mkdir -p $HOME/data
WORKDIR $HOME/data
RUN wget -q https://media.roboflow.com/notebooks/examples/dog.jpeg
RUN wget -q https://media.roboflow.com/notebooks/examples/dog-2.jpeg
RUN wget -q https://media.roboflow.com/notebooks/examples/dog-3.jpeg
RUN wget -q https://media.roboflow.com/notebooks/examples/dog-4.jpeg
RUN wget -q https://media.roboflow.com/notebooks/examples/dog-5.jpeg
RUN wget -q https://media.roboflow.com/notebooks/examples/dog-6.jpeg
RUN wget -q https://media.roboflow.com/notebooks/examples/dog-7.jpeg
RUN wget -q https://media.roboflow.com/notebooks/examples/dog-8.jpeg


# Copy the script into the container
COPY completebuild.ipynb $HOME/
WORKDIR $HOME
EXPOSE 8888
# Set the default command to run the script
# CMD ["jupyter", "nbconvert", "--to", "notebook", "--execute", "completebuild.ipynb"]
