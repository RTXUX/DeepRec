#!/bin/bash

set -ex

TMPDIR="$(mktemp -d)"

cleanup() {
    if [[ -n "${TMPDIR}" ]]; then
        rm -rf "${TMPDIR}"
    fi
}

trap cleanup EXIT

bazel build -c opt --config=opt //tensorflow/tools/pip_package:build_pip_package

./bazel-bin/tensorflow/tools/pip_package/build_pip_package "${TMPDIR}"

pip install --no-deps --force-reinstall "${TMPDIR}"/*.whl
