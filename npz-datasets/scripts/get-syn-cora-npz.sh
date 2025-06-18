#!/usr/bin/env bash
set -e

URL="https://public-files.jiongzhu.net/syn-cora-npz.tar.gz" # You can download this with browser too
FILE="syn-cora-npz.tar.gz"
EXPECTED_SUM="7609527ece3dbc3eadb84350754404a37d5fc6b2dc3ff74f0e4fda3922fb28fa"

echo "Downloading $URL..."
curl -L -o "$FILE" "$URL"

echo "Verifying SHA256 checksum..."
ACTUAL_SUM=$(sha256sum "$FILE" | awk '{print $1}')

if [ "$ACTUAL_SUM" = "$EXPECTED_SUM" ]; then
    echo "Checksum OK. Extracting archive..."
    tar -xzvf "$FILE"
    echo "Extraction complete."
else
    echo "Checksum FAILED!"
    echo "Expected: $EXPECTED_SUM"
    echo "Actual:   $ACTUAL_SUM"
    exit 1
fi