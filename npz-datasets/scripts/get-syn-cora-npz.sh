#!/usr/bin/env bash
# LINK=https://drive.google.com/file/d/1TbC-10pF2WlfbYLmu_gPV8TcXEBYeyOl/view?usp=sharing
# If automatic downloading isn't working, the file can be downloaded manually with the above link.
ggID='1TbC-10pF2WlfbYLmu_gPV8TcXEBYeyOl'
ggURL='https://drive.google.com/uc?export=download'
TARGET=archives/syn-cora-npz.tar.gz
SHA256SUM=7609527ece3dbc3eadb84350754404a37d5fc6b2dc3ff74f0e4fda3922fb28fa

cd "$(dirname ${BASH_SOURCE[0]})/.."
mkdir -p archives

# Automatic downloading script adopted from https://stackoverflow.com/a/38937732
filename="$(curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"  
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"
curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${TARGET}"

echo "$SHA256SUM  $TARGET" | sha256sum -c
test $? -eq 0 || read -p "Failed to verify SHA256 checksum. Press any key to continue anyway." -n 1 -r
tar -xvzf $TARGET