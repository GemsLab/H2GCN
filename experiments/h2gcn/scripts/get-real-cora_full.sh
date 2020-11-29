#!/usr/bin/env bash
LINK=https://umich.box.com/shared/static/k8nl55pcjum7ke6sffbcs8dqhvddgujs.gz
TARGET=archives/real-cora_full.tar.gz
SHA256SUM=b04a3db58aee34ddec4e24970665a3ef094125f39e2051c6e5024f124caa5053

cd "$(dirname ${BASH_SOURCE[0]})/.."
mkdir -p archives
wget -L $LINK -O $TARGET
echo "$SHA256SUM  $TARGET" | sha256sum -c
test $? -eq 0 || read -p "Failed to verify SHA256 checksum. Press any key to continue anyway." -n 1 -r
tar -xvzf $TARGET
