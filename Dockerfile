# ---- Base image ----
# If you do NOT need CUDA, replace this with:
#   FROM python:3.10-slim
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev python3-pip \
    build-essential \
    git curl ca-certificates \
    ffmpeg libsm6 libxext6 \
    zip unzip \
    && rm -rf /var/lib/apt/lists/*

# Make `python` → `python3`
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# ---- Virtual environment ----
RUN python -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

# ---- Python tooling ----
RUN pip install --upgrade pip setuptools wheel

# ---- Install your repo dependencies ----
# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt


# ---- Workspace ----
WORKDIR /workspace
COPY . /workspace

CMD ["/bin/bash"]
