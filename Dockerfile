FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /uvx /usr/local/bin/
ENV HOME=/tmp UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy
WORKDIR /workspace
CMD ["bash"]
