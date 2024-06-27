#!/bin/bash

set -ex

here=$(dirname $0)
OPENRETINA_HOME="$here/../.."

pytest ${OPENRETINA_HOME}/tests

