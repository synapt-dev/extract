# Security Policy

## Reporting vulnerabilities

Report security vulnerabilities privately via email to **security@synapt.dev**. Do not open a public GitHub issue.

We will acknowledge receipt within **4 hours** during business hours (US Eastern). For confirmed vulnerabilities in published releases, we target a patch release within **48 hours** of confirmation.

## Threat model

### What extract does

extract is a **pure computation library**. It builds prompts from text, validates JSON documents against the SynaptExtraction IL schema, and runs a deterministic finalization pipeline. It does not:

- Make network requests
- Access the filesystem (beyond reading bundled prompt fragments at import time)
- Execute user-supplied code
- Store or transmit credentials

### What extract does NOT do

The proposed `extract()` callback architecture (target: v0.4.0, see `docs/callback-signature.md`) will delegate all network operations to the caller. The design ensures synapt never sees API keys, auth tokens, or user credentials. The caller will own:

- LLM API calls (via `callLlm` callback)
- Embedding API calls (via `getEmbedding` callback)
- Retry logic, rate limiting, and fallback providers
- Credential management and rotation

> **Note:** The callback API is not exported in v0.3.x. The types and `extract()` function described in `docs/callback-signature.md` are proposed; implementation ships in v0.4.0.

### Forbidden APIs

The following APIs MUST NOT appear in extract's source code. CI enforces this via AST-aware scanning of both source and packed artifacts.

- `fetch`, `XMLHttpRequest`, `WebSocket`
- `node:net`, `node:http`, `node:https`, `node:http2`
- `Deno.connect`, `Deno.dial`, `Deno.listen`
- Dynamic `import()` of network-capable modules
- `child_process`, `node:child_process`
- `eval`, `new Function()`

Any PR introducing a forbidden API is a security-relevant change and requires explicit review.

## Supply chain verification

### npm (Sigstore provenance)

Every npm release is published with [Sigstore provenance](https://docs.npmjs.com/generating-provenance-statements) via GitHub Actions OIDC. Verify provenance:

```bash
npm audit signatures
```

This confirms the published package was built from the source commit in this repository by the CI workflow, not by a human.

### PyPI (trusted publishing)

PyPI releases use [trusted publishing](https://docs.pypi.org/trusted-publishers/) via GitHub Actions OIDC. The publishing workflow is the only entity authorized to upload releases.

### SBOM

Each GitHub Release includes a CycloneDX SBOM (`sbom.cdx.json`) listing all dependencies included in the npm package.

### Reproducible builds

The CI pipeline verifies build determinism on every push:

- **npm**: `npm pack` is run twice and SHA256 checksums are compared (byte-identical)
- **Python wheel**: `python -m build` is run twice with a fixed `SOURCE_DATE_EPOCH` and wheel checksums are compared (byte-identical)
- **Python sdist**: content-equivalent only. Setuptools embeds wall-clock timestamps in gzip/PAX headers, a known upstream limitation. The unpacked contents are identical; the `.tar.gz` wrapper bytes may differ.

You can reproduce a release locally:

```bash
# npm
cd packages/ts && npm ci && npm run build && npm pack
sha256sum *.tgz

# Python (wheel)
cd packages/python
cp -r ../../prompts src/synapt_extract/prompts
cp -r ../../schemas src/synapt_extract/schemas
SOURCE_DATE_EPOCH=1704067200 python -m build
sha256sum dist/*.whl
```

## Incident response

If a compromised release is confirmed:

1. **Unpublish** the affected version from npm and PyPI within 1 hour
2. **Post a GitHub Security Advisory** with affected versions and remediation
3. **Publish a patch release** with the fix
4. **Notify known consumers** (Conversa and any registered downstream integrators)

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.3.x   | Yes       |
| 0.2.x   | Security fixes only |
| < 0.2   | No        |
