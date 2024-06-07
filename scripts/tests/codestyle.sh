#!/bin/bash

set -ex

here=$(dirname $0)
OPENRETINA_HOME="$here/../.."

pycodestyle --max-line-length 120 \
    ${OPENRETINA_HOME}/openretina \
    ${OPENRETINA_HOME}/scripts

