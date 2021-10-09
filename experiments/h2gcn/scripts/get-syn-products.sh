#!/usr/bin/env bash
# LINK=https://drive.google.com/file/d/19wUu-fVhXbgqDFdkmkBAgTA7JKSOYNDm/view?usp=sharing
# If automatic downloading isn't working, the file can be downloaded manually with the above link.
ggID='19wUu-fVhXbgqDFdkmkBAgTA7JKSOYNDm'
ggURL='https://drive.google.com/uc?export=download'
TARGET=archives/syn-products.tar.gz
SHA256SUM=ee92199881159dbb259c9b1c580984e3a9a7681b0b5da35ac2d2e36c9e240f26

cd "$(dirname ${BASH_SOURCE[0]})/.."
mkdir -p archives

# Automatic downloading script adopted from https://stackoverflow.com/a/38937732
filename="$(curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"  
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"
curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${TARGET}"

echo "$SHA256SUM  $TARGET" | sha256sum -c
test $? -eq 0 || read -p "Failed to verify SHA256 checksum. Press any key to continue anyway." -n 1 -r
tar -xvzf $TARGET
