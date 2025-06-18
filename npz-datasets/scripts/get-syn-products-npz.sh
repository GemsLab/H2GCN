#!/usr/bin/env bash
set -e

URL="https://public-files.jiongzhu.net/syn-products-npz.tar.gz" # You can download this with browser too
FILE="syn-products-npz.tar.gz"
EXPECTED_SUM="f1500ce1b342c361897230249cf41ebc26f3d1df09e89118c5f691eecd16262e"

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
