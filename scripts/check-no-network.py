#!/usr/bin/env python3
"""No-network guard for Python source. Scans for forbidden API usage."""

import re
import sys
from pathlib import Path

FORBIDDEN_MODULES = {
    "socket", "http", "http.client", "http.server",
    "urllib", "urllib.request", "urllib.parse",
    "requests", "httpx", "aiohttp",
    "subprocess", "os.system",
}

FORBIDDEN_PATTERNS = [
    (re.compile(r"\bimport\s+(" + "|".join(re.escape(m) for m in FORBIDDEN_MODULES) + r")\b"), "forbidden module import"),
    (re.compile(r"\bfrom\s+(" + "|".join(re.escape(m) for m in FORBIDDEN_MODULES) + r")\b"), "forbidden from-import"),
    (re.compile(r"\beval\s*\("), "eval() call"),
    (re.compile(r"\bexec\s*\("), "exec() call"),
    (re.compile(r"\b__import__\s*\("), "__import__() call"),
    (re.compile(r"\bsubprocess\b"), "subprocess reference"),
    (re.compile(r"\bos\.system\s*\("), "os.system() call"),
    (re.compile(r"\bos\.popen\s*\("), "os.popen() call"),
]


def scan_file(path: Path) -> list[dict]:
    violations = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pattern, kind in FORBIDDEN_PATTERNS:
            if pattern.search(line):
                violations.append({"file": str(path), "line": i, "kind": kind, "match": stripped[:60]})
    return violations


def main():
    if len(sys.argv) < 2:
        print("Usage: check-no-network.py <dir> [<dir> ...]", file=sys.stderr)
        sys.exit(1)

    all_violations = []
    for d in sys.argv[1:]:
        for p in Path(d).rglob("*.py"):
            all_violations.extend(scan_file(p))

    if all_violations:
        print(f"\n❌ {len(all_violations)} forbidden API violation(s) found:\n", file=sys.stderr)
        for v in all_violations:
            print(f"  {v['file']}:{v['line']} [{v['kind']}] {v['match']}", file=sys.stderr)
        print(file=sys.stderr)
        sys.exit(1)
    else:
        print("✅ No forbidden API usage detected.")


if __name__ == "__main__":
    main()
