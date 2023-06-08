FROM ubuntu:latest as base

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        g++ \
        gcc \
        python3 \
        python3-dev \
        python3-pip

WORKDIR /app

COPY . ./Detection

RUN --mount=type=cache,target=/root/.cache python3 -m pip install -r Detection/requirements.txt