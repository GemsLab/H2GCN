#!/usr/bin/env bash
LINK=https://umich.box.com/shared/static/oerjpreqd1u1cn481mk1m788fd4ahcvv.gz
TARGET=archives/syn-cora.tar.gz
SHA256SUM=93a5329054bc36d742f394589a4de9b6239f8a19ccb6b7b894228841887a413b

cd "$(dirname ${BASH_SOURCE[0]})/.."
mkdir -p archives
wget -L $LINK -O $TARGET
echo "$SHA256SUM  $TARGET" | sha256sum -c
test $? -eq 0 || read -p "Failed to verify SHA256 checksum. Press any key to continue anyway." -n 1 -r
tar -xvzf $TARGET