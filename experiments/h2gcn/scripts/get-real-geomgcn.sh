#!/usr/bin/env bash
LINK=https://umich.box.com/shared/static/k3baoomusdzvrhi58nmgneibxuhunrkz.gz
TARGET=archives/real-geomgcn.tar.gz
SHA256SUM=06bf9a52cb272b3b25227530eafc2a40681fa7c548641ec00ca2427812fbe39f

cd "$(dirname ${BASH_SOURCE[0]})/.."
mkdir -p archives
wget -L $LINK -O $TARGET
echo "$SHA256SUM  $TARGET" | sha256sum -c
test $? -eq 0 || read -p "Failed to verify SHA256 checksum. Press any key to continue anyway." -n 1 -r
tar -xvzf $TARGET
