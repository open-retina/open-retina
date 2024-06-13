#!/bin/bash

set -ex

here=$(dirname $0)
OPENRETINA_HOME="$here/../.."

ruff check --config ${here}/ruff.toml \
    ${OPENRETINA_HOME}/openretina \
    ${OPENRETINA_HOME}/scripts

