#!/bin/bash -e
#
# Build Docker image, and optionally publish.
#
# Usage:
#
#   $ ./docker-build.sh DOCKERFILE [publish]
#
# Examples:
#
#   Build docker image locally. Useful to test Dockerfile changes before publishing.
#   $ ./docker/docker-build.sh docker/Dockerfile
#   Build + publish
#   $ ./docker/docker-build.sh docker/Dockerfile publish
#
# Note: only tested on linux
#

export VERSION=`git rev-parse --short HEAD`

DOCKERFILE=$1

docker build -f $DOCKERFILE -t pinterestdocker/visualembedding:$VERSION .

# If the publish command is given AND we have no code changes not checked in.
# The second condition is to prevent folks from overwriting a production
# commit hash. If you publish, you only publish committed changes. Unfortunately
# this can be circumvented as currently implemented but the current solution
# is better than none
if [[ "$2" == "publish" ]]; then
  if [[ -z $(git status -s) ]]; then
    docker push pinterestdocker/visualembedding:$VERSION
  else
    echo "[PUSH ERROR] Cannot push with outstanding changes:"
    git status -s
  fi
fi

cat <<EOL > DOCKER_TAG
$VERSION
EOL
