#!/usr/bin/env bash
# LINK=https://drive.google.com/file/d/1gbAE4GwjWQz60e_kBeXe0ID_ODC6dzkE/view?usp=sharing
# If automatic downloading isn't working, the file can be downloaded manually with the above link.
ggID='1gbAE4GwjWQz60e_kBeXe0ID_ODC6dzkE'
ggURL='https://drive.google.com/uc?export=download'
TARGET=archives/syn-products-npz.tar.gz
SHA256SUM=f1500ce1b342c361897230249cf41ebc26f3d1df09e89118c5f691eecd16262e

cd "$(dirname ${BASH_SOURCE[0]})/.."
mkdir -p archives

# Automatic downloading script adopted from https://stackoverflow.com/a/38937732
filename="$(curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"  
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"
curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${TARGET}"

echo "$SHA256SUM  $TARGET" | sha256sum -c
test $? -eq 0 || read -p "Failed to verify SHA256 checksum. Press any key to continue anyway." -n 1 -r
tar -xvzf $TARGET
