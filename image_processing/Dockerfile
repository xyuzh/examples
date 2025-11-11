FROM anyscale/ray:2.51.1-slim-py312-cu128

# By default, Anyscale swaps in an optimized version of Ray, so we need to avoid that
# in order to use our specific patched version of Ray.
ENV ANYSCALE_DISABLE_OPTIMIZED_RAY=1

RUN pip uninstall -y ray
# Install a patched version of Ray. In this case, we use the nightly wheels that Ray publishes.
# Learn more about the nightly wheels at https://docs.ray.io/en/latest/ray-overview/installation.html#daily-releases-nightlies
RUN pip install https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp312-cp312-manylinux2014_x86_64.whl

# C compiler for Triton’s runtime build step (vLLM V1 engine)
# https://github.com/vllm-project/vllm/issues/2997
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends build-essential

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN uv pip install --system huggingface_hub boto3

RUN uv pip install --system vllm==0.11.0

RUN uv pip install --system transformers==4.57.1