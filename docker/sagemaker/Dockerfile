# syntax=docker/dockerfile:1
# Python 3.12 Debian bookworm manifest resolved on 2026-10-14.
# Build for linux/amd64. CUDA runtime libraries come from the locked torch wheel;
# the SageMaker host must provide a compatible NVIDIA driver.
# For endpoint hosting, export Docker V2: --provenance=false and
# --output type=image,name=<local-image>,oci-mediatypes=false (see AWS guide).
ARG BASE_IMAGE=python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
FROM ${BASE_IMAGE}

ADD --checksum=sha256:cf9f55b8343e1ef6e6dcccdb09b99c70c0430a22ed7d0079286eccddeaf72670 https://github.com/SocketDev/sfw-free/releases/download/v1.15.4/sfw-free-linux-x86_64 /usr/local/bin/sfw
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && chmod +x /usr/local/bin/sfw \
    && sfw apt-get update \
    && sfw apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && sfw python -m pip install --no-cache-dir uv==0.11.24

WORKDIR /opt/smithtune
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
COPY pyproject.toml uv.lock overrides.txt README.md ./
RUN --mount=type=cache,target=/root/.cache/uv sfw uv sync --locked --no-dev --no-install-project --extra aws-training --python /usr/local/bin/python
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv sfw uv sync --locked --no-dev --no-editable --extra aws-training --python /usr/local/bin/python

ENV PYTHONUNBUFFERED=1
# PyTorch 2.14's eager Triton overrides require a host compiler. This slim
# worker uses precompiled ATen CUDA kernels for those operations instead.
ENV TORCH_DISABLE_NATIVE_JIT=1
ENTRYPOINT ["/opt/venv/bin/python", "-m", "smithtune.providers.sagemaker.worker"]
