# Model Signing Keys

Flec uses [minisign](https://jedisct1.github.io/minisign/) to sign `scripts/model_checksums.json`, which lists SHA-256 hashes of all downloaded model files. This prevents a compromised download mirror from substituting malicious model weights.

## Setup (first time)

```bash
brew install minisign
# Generate keypair (run once per maintainer workstation):
minisign -G -s flec-models.key -p flec-models.pub
# Never commit flec-models.key — it is gitignored.
```

## Signing after a model update

```bash
# 1. Update SHA-256 checksums in scripts/model_checksums.json
# 2. Sign:
bash scripts/sign_checksums.sh
# 3. Commit the updated json + the new .minisig:
git add scripts/model_checksums.json scripts/model_checksums.json.minisig
git commit -m "chore: update model checksums for <model-name> <version>"
```

## Key rotation

1. Generate a new keypair on a new machine (same command as Setup).
2. Update `model_checksums.json` with the new `public_key` value.
3. Re-sign the checksums file with the new key.
4. Commit and push the new pub key + re-signed minisig.

## Embedded public key

The public key lives in `scripts/model_checksums.json` under the `public_key` field. `download_models.py` will read it at runtime to verify the signature before trusting any hash. (Verification is a stub until minisign Python bindings are available on ARM64.)
