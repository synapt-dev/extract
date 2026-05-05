#!/usr/bin/env node
// Best-effort regex no-network guard for extract.
// Scans JS/TS source files for forbidden API usage including obfuscated forms.
// Exit 0 = clean, Exit 1 = violations found.

import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, extname } from "node:path";

const FORBIDDEN_GLOBALS = new Set([
  "fetch", "XMLHttpRequest", "WebSocket",
  "eval",
]);

const FORBIDDEN_MODULES = new Set([
  "node:net", "node:http", "node:https", "node:http2",
  "node:child_process", "child_process",
  "net", "http", "https", "http2",
]);

const FORBIDDEN_DENO = new Set([
  "connect", "dial", "listen",
]);

const FORBIDDEN_CONSTRUCTORS = new Set([
  "Function",
]);

function collectFiles(dir, exts) {
  const results = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (entry === "node_modules" || entry === ".git") continue;
    const st = statSync(full);
    if (st.isDirectory()) {
      results.push(...collectFiles(full, exts));
    } else if (exts.some((e) => full.endsWith(e))) {
      results.push(full);
    }
  }
  return results;
}

function scanFile(filePath) {
  const code = readFileSync(filePath, "utf-8");
  const lines = code.split("\n");
  const violations = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const lineNum = i + 1;

    // Skip comments (rough heuristic: full-line comments and inline //)
    const trimmed = line.trim();
    if (trimmed.startsWith("//") || trimmed.startsWith("*") || trimmed.startsWith("/*")) continue;

    // 1. Direct forbidden global references (word boundary)
    for (const name of FORBIDDEN_GLOBALS) {
      const re = new RegExp(`\\b${name}\\b`, "g");
      if (re.test(line)) {
        violations.push({ file: filePath, line: lineNum, match: name, kind: "forbidden global" });
      }
    }

    // 2. Function() constructor (with or without new)
    if (/\bFunction\s*\(/.test(line) && !trimmed.startsWith("//")) {
      violations.push({ file: filePath, line: lineNum, match: "Function()", kind: "forbidden constructor" });
    }

    // 3. Forbidden module imports/requires
    for (const mod of FORBIDDEN_MODULES) {
      const importRe = new RegExp(`(?:import|require)\\s*\\(?\\s*["'\`]${mod.replace("/", "\\/")}["'\`]`);
      if (importRe.test(line)) {
        violations.push({ file: filePath, line: lineNum, match: mod, kind: "forbidden module" });
      }
      // Dynamic import
      const dynRe = new RegExp(`import\\s*\\(\\s*["'\`]${mod.replace("/", "\\/")}["'\`]`);
      if (dynRe.test(line)) {
        violations.push({ file: filePath, line: lineNum, match: `import("${mod}")`, kind: "forbidden dynamic import" });
      }
    }

    // 4. Deno forbidden APIs
    for (const api of FORBIDDEN_DENO) {
      const re = new RegExp(`\\bDeno\\.${api}\\b`);
      if (re.test(line)) {
        violations.push({ file: filePath, line: lineNum, match: `Deno.${api}`, kind: "forbidden Deno API" });
      }
    }

    // 5. Computed property access on globalThis/window/global/self (obfuscation vector)
    if (/\b(globalThis|window|global|self)\s*\[/.test(line)) {
      violations.push({ file: filePath, line: lineNum, match: line.trim().slice(0, 60), kind: "computed property on global object" });
    }

    // 6. Dynamic import() with non-literal argument
    if (/\bimport\s*\(\s*[^"'`\s]/.test(line)) {
      // import( followed by something that isn't a string literal
      if (!/\bimport\s*\(\s*["'`]/.test(line)) {
        violations.push({ file: filePath, line: lineNum, match: line.trim().slice(0, 60), kind: "dynamic import with non-literal" });
      }
    }

    // 7. atob / Buffer.from with base64 (potential decoded forbidden name)
    if (/\batob\s*\(/.test(line) || /Buffer\.from\s*\([^)]*,\s*["']base64["']/.test(line)) {
      violations.push({ file: filePath, line: lineNum, match: line.trim().slice(0, 60), kind: "base64 decode (potential obfuscation)" });
    }

    // 8. String concatenation that could spell forbidden names
    // Detect patterns like "fe"+"tch", 'ev'+'al', etc.
    const concatMatch = line.match(/["'][a-zA-Z]{1,6}["']\s*\+\s*["'][a-zA-Z]{1,12}["']/g);
    if (concatMatch) {
      for (const m of concatMatch) {
        const assembled = m.replace(/["']\s*\+\s*["']/g, "").replace(/["']/g, "");
        if (FORBIDDEN_GLOBALS.has(assembled) || FORBIDDEN_CONSTRUCTORS.has(assembled)) {
          violations.push({ file: filePath, line: lineNum, match: m, kind: `string concat assembles "${assembled}"` });
        }
      }
    }

    // 9. Reflect.get on global objects (obfuscation vector for accessing forbidden globals)
    if (/\bReflect\.get\s*\(\s*(globalThis|window|global|self)\b/.test(line)) {
      violations.push({ file: filePath, line: lineNum, match: line.trim().slice(0, 60), kind: "Reflect.get on global object" });
    }

    // 10. Array .join() assembling forbidden names (e.g. ["fe","tch"].join(""))
    const joinMatch = line.match(/\[([^\]]*)\]\s*\.\s*join\s*\(\s*["'`]\s*["'`]\s*\)/g);
    if (joinMatch) {
      for (const m of joinMatch) {
        const inner = m.match(/\[([^\]]*)\]/)?.[1] ?? "";
        const assembled = inner.replace(/["'`\s,]/g, "");
        if (FORBIDDEN_GLOBALS.has(assembled) || FORBIDDEN_CONSTRUCTORS.has(assembled)) {
          violations.push({ file: filePath, line: lineNum, match: m, kind: `array join assembles "${assembled}"` });
        }
      }
    }
  }

  return violations;
}

// Main
const args = process.argv.slice(2);
if (args.length === 0) {
  console.error("Usage: check-no-network.mjs <dir> [<dir> ...]");
  process.exit(1);
}

const exts = [".ts", ".js", ".mts", ".mjs", ".cjs", ".cts"];
let allViolations = [];

for (const dir of args) {
  const files = collectFiles(dir, exts);
  for (const f of files) {
    allViolations.push(...scanFile(f));
  }
}

if (allViolations.length > 0) {
  console.error(`\n❌ ${allViolations.length} forbidden API violation(s) found:\n`);
  for (const v of allViolations) {
    console.error(`  ${v.file}:${v.line} [${v.kind}] ${v.match}`);
  }
  console.error("");
  process.exit(1);
} else {
  console.log("✅ No forbidden API usage detected.");
  process.exit(0);
}
