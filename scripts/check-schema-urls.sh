#!/usr/bin/env bash
# Verify every $id URL in schemas/*/v1.json is live and matches local content.
# Exits non-zero on 404 or content mismatch.
set -euo pipefail

SCHEMA_DIR="${1:-schemas}"
FAILURES=0

for f in "$SCHEMA_DIR"/*/v1.json; do
  id=$(jq -r '."$id"' "$f")
  if [ -z "$id" ] || [ "$id" = "null" ]; then
    echo "SKIP: $f has no \$id"
    continue
  fi

  echo -n "Checking $id ... "
  if ! remote=$(curl -fsSL "$id" 2>/dev/null); then
    echo "FAIL: 404 or unreachable"
    FAILURES=$((FAILURES + 1))
    continue
  fi

  local_norm=$(jq -S . "$f")
  remote_norm=$(echo "$remote" | jq -S .)
  if [ "$local_norm" != "$remote_norm" ]; then
    echo "FAIL: content mismatch"
    diff <(echo "$local_norm") <(echo "$remote_norm") || true
    FAILURES=$((FAILURES + 1))
    continue
  fi

  echo "OK"
done

if [ "$FAILURES" -gt 0 ]; then
  echo ""
  echo "❌ $FAILURES schema URL(s) failed verification."
  exit 1
else
  echo ""
  echo "✅ All schema URLs verified."
fi
