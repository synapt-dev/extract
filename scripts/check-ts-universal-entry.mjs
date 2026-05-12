#!/usr/bin/env node
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

const FORBIDDEN_BUILTINS = new Set([
  "assert",
  "async_hooks",
  "buffer",
  "child_process",
  "cluster",
  "console",
  "constants",
  "crypto",
  "dgram",
  "diagnostics_channel",
  "dns",
  "domain",
  "events",
  "fs",
  "http",
  "http2",
  "https",
  "module",
  "net",
  "os",
  "path",
  "perf_hooks",
  "process",
  "punycode",
  "querystring",
  "readline",
  "repl",
  "stream",
  "string_decoder",
  "timers",
  "tls",
  "trace_events",
  "tty",
  "url",
  "util",
  "v8",
  "vm",
  "worker_threads",
  "zlib",
]);

const entryArg = process.argv[2];
if (!entryArg) {
  console.error("Usage: check-ts-universal-entry.mjs <dist-entry.js>");
  process.exit(2);
}

const entry = resolve(entryArg);
const visited = new Set();
const violations = [];

function forbiddenBuiltin(specifier) {
  const normalized = specifier.startsWith("node:") ? specifier.slice("node:".length) : specifier;
  const root = normalized.split("/")[0];
  return FORBIDDEN_BUILTINS.has(root) ? specifier : undefined;
}

function resolveRelativeImport(fromFile, specifier) {
  const base = resolve(dirname(fromFile), specifier);
  const candidates = [
    base,
    `${base}.js`,
    resolve(base, "index.js"),
  ];
  return candidates.find((candidate) => existsSync(candidate));
}

function importSpecifiers(code) {
  const specs = [];
  const staticImport = /\b(?:import|export)\s+(?:[^'"]*?\s+from\s*)?["']([^"']+)["']/g;
  const dynamicImport = /\bimport\s*\(\s*["']([^"']+)["']\s*\)/g;
  for (const match of code.matchAll(staticImport)) specs.push(match[1]);
  for (const match of code.matchAll(dynamicImport)) specs.push(match[1]);
  return specs;
}

function scan(file) {
  if (visited.has(file)) return;
  visited.add(file);

  const code = readFileSync(file, "utf-8");
  for (const specifier of importSpecifiers(code)) {
    const builtin = forbiddenBuiltin(specifier);
    if (builtin !== undefined) {
      violations.push(`${file} imports ${builtin}`);
      continue;
    }

    if (specifier.startsWith(".")) {
      const resolved = resolveRelativeImport(file, specifier);
      if (resolved === undefined) {
        violations.push(`${file} imports missing local module ${specifier}`);
        continue;
      }
      scan(resolved);
    }
  }
}

scan(entry);

if (violations.length > 0) {
  console.error("Default TypeScript entry is not universal-safe:");
  for (const violation of violations) {
    console.error(`- ${violation}`);
  }
  process.exit(1);
}

console.log(`Default TypeScript entry is universal-safe (${visited.size} modules scanned).`);
