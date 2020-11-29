#!/usr/bin/env bash
LINK=https://umich.box.com/shared/static/sxxqp4zv9ui7ilzcoml6tgswcwj8p2is.gz
TARGET=archives/syn-products.tar.gz
SHA256SUM=ee92199881159dbb259c9b1c580984e3a9a7681b0b5da35ac2d2e36c9e240f26

cd "$(dirname ${BASH_SOURCE[0]})/.."
mkdir -p archives
wget -L $LINK -O $TARGET
echo "$SHA256SUM  $TARGET" | sha256sum -c
test $? -eq 0 || read -p "Failed to verify SHA256 checksum. Press any key to continue anyway." -n 1 -r
tar -xvzf $TARGET
