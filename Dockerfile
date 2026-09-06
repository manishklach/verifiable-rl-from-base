FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/verifiable-rl-from-base
COPY . .
RUN python3 -m pip install --break-system-packages ".[dev,wandb,demo]"

CMD ["bash", "scripts/run_experiment.sh"]

