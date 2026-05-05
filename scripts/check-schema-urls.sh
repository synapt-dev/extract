#!/usr/bin/env bash
# Verify every $id URL in schemas/*/v1.json follows the expected pattern
# and that schema files are internally consistent.
#
# In CI, live CDN verification is not possible because Cloudflare blocks
# GitHub Actions datacenter IPs and the site repo is private. This script
# validates URL structure and $id consistency instead.
#
# Live CDN verification is a manual release-checklist step:
#   curl -fsSL https://synapt.dev/schemas/extract/v1.json | jq .
#
# Full live-CDN CI verification deferred to v0.3.2 (requires Cloudflare
# allowlist for GitHub Actions IP ranges).
#
# Exits non-zero on malformed $id or content inconsistency.
set -euo pipefail

SCHEMA_DIR="${1:-schemas}"
FAILURES=0
EXPECTED_BASE="https://synapt.dev/schemas"

for f in "$SCHEMA_DIR"/*/v1.json; do
  id=$(jq -r '."$id"' "$f")
  if [ -z "$id" ] || [ "$id" = "null" ]; then
    echo "SKIP: $f has no \$id"
    continue
  fi

  # Extract expected schema name from file path (e.g., schemas/action/v1.json -> action)
  schema_name=$(basename "$(dirname "$f")")
  expected_id="$EXPECTED_BASE/$schema_name/v1.json"

  echo -n "Checking $f ... "

  # 1. Verify $id follows expected URL pattern
  if [ "$id" != "$expected_id" ]; then
    echo "FAIL: \$id is '$id', expected '$expected_id'"
    FAILURES=$((FAILURES + 1))
    continue
  fi

  # 2. Verify the file is valid JSON
  if ! jq -e . "$f" > /dev/null 2>&1; then
    echo "FAIL: invalid JSON"
    FAILURES=$((FAILURES + 1))
    continue
  fi

  # 3. Verify required JSON Schema fields
  schema_field=$(jq -r '."$schema" // empty' "$f")
  if [ -z "$schema_field" ]; then
    echo "FAIL: missing \$schema field"
    FAILURES=$((FAILURES + 1))
    continue
  fi

  echo "OK ($id)"
done

# 4. Try live CDN verification if not in CI (best-effort, non-blocking)
if [ -z "${CI:-}" ] && [ -z "${GITHUB_ACTIONS:-}" ]; then
  echo ""
  echo "=== Live CDN verification (local only) ==="
  for f in "$SCHEMA_DIR"/*/v1.json; do
    id=$(jq -r '."$id"' "$f")
    if [ -z "$id" ] || [ "$id" = "null" ]; then
      continue
    fi
    echo -n "  $id ... "
    if ! remote=$(curl -fsSL --max-time 10 "$id" 2>/dev/null); then
      echo "SKIP (unreachable)"
      continue
    fi
    local_norm=$(jq -S . "$f")
    remote_norm=$(echo "$remote" | jq -S .)
    if [ "$local_norm" != "$remote_norm" ]; then
      echo "MISMATCH"
      diff <(echo "$local_norm") <(echo "$remote_norm") || true
      FAILURES=$((FAILURES + 1))
      continue
    fi
    echo "OK"
  done
fi

if [ "$FAILURES" -gt 0 ]; then
  echo ""
  echo "❌ $FAILURES schema URL(s) failed verification."
  exit 1
else
  echo ""
  echo "✅ All schema URLs verified."
fi
