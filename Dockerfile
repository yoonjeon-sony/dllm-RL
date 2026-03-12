FROM nvidia/cuda:12.6-cudnn9-devel-ubuntu22.04

# install uv
RUN pip install uv

# install kubectl
RUN apt-get update && apt-get install -y curl --no-install-recommends && \
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && \
    rm kubectl && \
    rm -rf /var/lib/apt/lists/*

# install jupyter
RUN pip install notebook==6.0.2 jupyterlab==1.2.4 jupyterlab-server==1.0.6
EXPOSE 8888
# user configuration
ENV NB_USER=irteam
ENV NB_UID=500

RUN useradd -m -s /bin/bash -U -u ${NB_UID} ${NB_USER}

USER ${NB_USER}
WORKDIR /home/${NB_USER}

# copy dependencies
COPY pyproject.toml uv.lock ./

# install python environment
RUN uv sync --frozen

# copy project
COPY . .

# activate venv automatically
ENV PATH="/home/${NB_USER}/.venv/bin:$PATH"

# run jupyter
CMD ["sh", "-c", "jupyter lab --notebook-dir=/home/${NB_USER} --ip=0.0.0.0 --no-browser --port=8888 --LabApp.token='' --LabApp.password=''"]