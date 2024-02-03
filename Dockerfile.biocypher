FROM python:3.11-bullseye

WORKDIR /app

COPY env/requirements-tc-graph.txt /app/
COPY env/tc-graph-docker.yaml /app/


RUN apt-get update && \
    apt-get install -y curl libcurl4-openssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda for ARM64 architecture
RUN curl -sLo /miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh && \
    sh /miniconda.sh -b -p /miniconda && \
    rm /miniconda.sh && \
    /miniconda/bin/conda clean -t -p -i -y && \
    ln -s /miniconda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /miniconda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
