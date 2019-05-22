#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

gofmt=$(which gofmt)

find_files() {
  find . -not \( \
      \( \
        -wholename './output' \
        -o -wholename '*/vendor/*' \
      \) -prune \
    \) -name '*.go'
}

diff=$(find_files | xargs ${gofmt} -d 2>&1)
if [[ -n "${diff}" ]]; then
  echo "${diff}"
  exit 1
fi

diff=$(find_files | xargs grep "golang.org/x/net/context" 2>&1 || true)
if [[ -n "${diff}" ]]; then
  echo "${diff} => \"context\" "
  exit 1
fi
