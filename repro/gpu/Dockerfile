FROM python:3.11.11-slim-bookworm@sha256:081075da77b2b55c23c088251026fb69a7b2bf92471e491ff5fd75c192fd38e5

ARG SOURCE_REVISION=8b9045b9549b776c7ebfdf163e77cde714b00d83
LABEL org.opencontainers.image.source="https://github.com/richard880502/AICUP2026_GO" \
      org.opencontainers.image.description="GPU training environment for AI CUP 2026 Go competitions 1 and 2" \
      org.opencontainers.image.revision="${SOURCE_REVISION}"

ENV VIRTUAL_ENV=/opt/aicup-venv \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PATH="/opt/aicup-venv/bin:${PATH}" \
    PYTHONPATH=/workspace:/workspace/competition1_player_identification

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libgomp1 libgl1 libglib2.0-0 unzip \
    && rm -rf /var/lib/apt/lists/*

COPY repro/gpu/requirements.lock /opt/requirements.lock
RUN /usr/local/bin/python -m venv "$VIRTUAL_ENV" \
    && pip install --no-cache-dir --only-binary=:all: -r /opt/requirements.lock \
    && pip check

# This release contains the byte-identical KataGo AppImage inspected on w77.
# Unpack it at build time so the runtime does not require a FUSE mount.
RUN mkdir -p /opt/katago /opt/humansl-vendor \
    && curl --fail --silent --show-error --location --retry 3 \
        https://github.com/lightvector/KataGo/releases/download/v1.18.2/katago-v1.18.2-cuda12.1-cudnn9.8.0-linux-x64.zip \
        -o /tmp/katago.zip \
    && echo '3e30b486b7bc38287eeead350aa832ff8ca50d1f0c0a83b5409cabf8a3bc9c5d  /tmp/katago.zip' | sha256sum -c - \
    && unzip -q /tmp/katago.zip katago -d /tmp \
    && chmod +x /tmp/katago \
    && cd /opt/katago \
    && /tmp/katago --appimage-extract >/dev/null \
    && curl --fail --silent --show-error --location --retry 3 \
        https://codeload.github.com/lightvector/KataGo/tar.gz/c560d38585b446bc93b0a62c77364bc8e7d266a4 \
        -o /tmp/humansl-source.tar.gz \
    && tar -xzf /tmp/humansl-source.tar.gz -C /opt/humansl-vendor --strip-components=1 \
    && rm /tmp/katago.zip /tmp/katago /tmp/humansl-source.tar.gz
COPY repro/gpu/run-katago /usr/local/bin/katago

WORKDIR /workspace
COPY competition1_player_identification/ competition1_player_identification/
COPY competition2_rank_prediction/ competition2_rank_prediction/
COPY repro/gpu/check_gpu.py /opt/check_gpu.py

# competition2_rank_prediction/c1 is a repository symlink to Competition 1.
RUN mkdir -p competition1_player_identification/models competition1_player_identification/third_party/katago \
    && ln -s "$VIRTUAL_ENV" competition1_player_identification/.venv \
    && ln -s /usr/local/bin/katago competition1_player_identification/third_party/katago/katago \
    && ln -s /assets/katago competition1_player_identification/models/katago \
    && mkdir -p runtime/competition2_research/vendor runtime/competition2_research/models \
        data production /assets /tmp2/richard/aicup_tutorial_review \
    && ln -s /opt/humansl-vendor runtime/competition2_research/vendor/KataGo-human-v1.15.0 \
    && ln -s /assets/human_trunk runtime/competition2_research/models/human_trunk \
    && chmod +x /usr/local/bin/katago \
    && katago version

ENV PYTHONPATH=/workspace:/workspace/competition1_player_identification:/workspace/competition2_rank_prediction

CMD ["python", "/opt/check_gpu.py"]
