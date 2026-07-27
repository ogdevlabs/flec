#!/usr/bin/env bash
# Sign model_checksums.json with the Flec maintainer keypair.
# Run this after updating SHA-256 checksums with real values.
#
# Prerequisites: brew install minisign
# First time: minisign -G -s flec-models.key -p flec-models.pub
# Then: minisign -S -s flec-models.key -m scripts/model_checksums.json
# Verify: minisign -V -p flec-models.pub -m scripts/model_checksums.json
#
# Never commit flec-models.key — add it to .gitignore.
# Commit flec-models.pub and model_checksums.json.minisig.
set -euo pipefail
minisign -S -s flec-models.key -m scripts/model_checksums.json
echo "Signed. Commit scripts/model_checksums.json.minisig."
