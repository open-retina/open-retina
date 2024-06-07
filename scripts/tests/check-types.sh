#!/bin/bash

set -ex

here=$(dirname $0)
OPENRETINA_HOME="$here/../.."

export MPYPATH="$OPENRETINA_HOME"
mypy --config $here/mypy.ini \
    ${OPENRETINA_HOME}/openretina \
    ${OPENRETINA_HOME}/scripts

