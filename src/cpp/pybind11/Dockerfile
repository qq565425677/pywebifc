FROM python:3.13-slim-trixie

ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ARG UID=
ARG GID=

ENV http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=4

RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates git build-essential cmake ninja-build \
      gcc-14 g++-14 pkg-config; \
    python -m pip install --no-cache-dir -U pip setuptools wheel pybind11; \
    apt-get clean; rm -rf /var/lib/apt/lists/*

COPY . /repo

RUN set -eux; \
    python -m pip install /repo/pybind11 --no-cache-dir --no-build-isolation

RUN set -eux; \
    groupadd -g ${GID} app || true; \
    useradd -m -u ${UID} -g ${GID} app || true

USER app
WORKDIR /work

ENTRYPOINT ["pywebifc-export-glb"]

# build:
# cd .. 
# docker build \
#  --build-arg UID=$(id -u) \
#  --build-arg GID=$(id -g) \
#  --build-arg HTTP_PROXY="on demand" \
#  --build-arg HTTPS_PROXY="on demand" \
#  -f pybind11/Dockerfile \
#  -t pywebifc:gcc14 .

# use:
# IFC_FILE="$HOME/your_ifc_file.ifc"
# docker run --rm \
#   -v "$(dirname "$IFC_FILE"):/work/" \
#   pywebifc:gcc14 \
#   "/work/$(basename $IFC_FILE)" "/work/$(basename "$IFC_FILE" .ifc).glb"

# SRC_DIR="$HOME/ifc_batch_dir"
# docker run --rm \
#   -v "$SRC_DIR:/work/" \
#   pywebifc:gcc14 \
#   bash -lc 'set -eux; shopt -s nullglob; for f in /work/*.ifc; do b=$(basename "$f" .ifc); pywebifc-export-glb "$f" "/work/${b}.glb"; done'
