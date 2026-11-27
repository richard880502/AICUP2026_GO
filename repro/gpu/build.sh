#!/usr/bin/env bash
# Use Git's tracked files as the build context.
set -euo pipefail

repo=$(git rev-parse --show-toplevel)
revision=$(git -C "$repo" rev-parse HEAD)
image=${1:-aicup2026-go:gpu-local}

git -C "$repo" archive --format=tar "$revision" |
    docker build --platform linux/amd64 \
        --build-arg "SOURCE_REVISION=$revision" \
        -f repro/gpu/Dockerfile -t "$image" -
