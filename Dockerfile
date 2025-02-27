# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV PATH=/opt/conda/bin:/opt/go/bin:$PATH
ENV DEBIAN_FRONTEND=noninteractive

# Configure MS fonts and install system dependencies
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections &&
    apt-get update && apt-get install -yq --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    g++ \
    git \
    libc6 \
    libc6-dev \
    locales \
    make \
    nasm \
    openssh-client \
    pkg-config \
    ttf-mscorefonts-installer \
    unzip \
    unar \
    wget &&
    apt-get clean &&
    rm -rf /var/lib/apt/lists/* &&
    locale-gen "en_US.UTF-8"

# Install Miniconda
ENV CONDA_PATH="/opt/conda"
ENV MINICONDA_RELEASE="Miniconda3-py312_25.1.1-2-Linux-x86_64.sh"
ENV MINICONDA_CHECKSUM="4766d85b5f7d235ce250e998ebb5a8a8210cbd4f2b0fea4d2177b3ed9ea87884"

RUN wget --quiet https://repo.anaconda.com/miniconda/$MINICONDA_RELEASE &&
    echo "$MINICONDA_CHECKSUM  $MINICONDA_RELEASE" | sha256sum --check --status &&
    /bin/bash $MINICONDA_RELEASE -f -b -p $CONDA_PATH &&
    rm $MINICONDA_RELEASE

# Install FFmpeg with CUDA support
RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git &&
    cd nv-codec-headers && make install && cd - &&
    git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg/ &&
    cd ffmpeg &&
    ./configure \
        --enable-nonfree \
        --enable-cuda-nvcc \
        --enable-libnpp \
        --extra-cflags=-I/usr/local/cuda/include \
        --extra-ldflags=-L/usr/local/cuda/lib64 \
        --disable-static \
        --enable-shared &&
    make -j 8 &&
    make install &&
    cd - &&
    ldconfig &&
    rm -rf nv-codec-headers ffmpeg

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Set working directory and install OpenRetina
WORKDIR /openretina
COPY . /openretina/
RUN pip install -e ".[dev]"

# Set default command
CMD ["/bin/bash"]
