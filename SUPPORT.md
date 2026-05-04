# Support

This document describes how to report bugs against `@synapt-dev/extract` (npm) and `synapt-extract` (PyPI), what response timing and triage to expect from synapt, what reproducibility we ask for in bug reports, and what synapt commits to in return.

## Bug-report channels

Pick the channel that matches your issue's nature. We monitor all three; volume is highest on GitHub.

### GitHub Issues (default)

https://github.com/synapt-dev/extract/issues

Use this for:
- Bugs in validation, finalization, prompt generation, or schema correctness
- Feature requests (new fields, capabilities, sub-schemas)
- Documentation issues
- Questions about expected behavior

GitHub Issues are public. Do not include sensitive data (real transcripts, user data, API keys, internal customer information). For sanitized reproductions, see "Reproducibility expectations" below.

We provide three issue templates (bug report, feature request, security disclosure pointer) to keep reports structured. Blank issues are disabled to encourage template use.

### Dedicated Slack channel (sev-1 production escalation)

For partners with an active integration and a sev-1 production issue, we maintain a dedicated Slack channel for real-time coordination. This is not for general support; it's for "production is broken right now" cases where the synchronous channel is faster than ticket triage.

Contact `support@synapt.dev` to establish a Slack channel as part of integration setup.

The Slack channel is **not** the right place for non-urgent bugs, feature requests, or general questions. Those go to GitHub Issues for proper triage and public visibility.

### Security disclosures (private)

`security@synapt.dev`

Use this for security-sensitive issues: vulnerabilities, supply-chain concerns, anything that should not be disclosed publicly until a fix is shipped. See `SECURITY.md` for the full vulnerability disclosure process.

Do not file security issues as public GitHub issues.

## Severity tiers and SLA

We classify reported issues into three tiers based on production impact. SLAs are best-effort commitments based on our current team size; we'll communicate any expected slip transparently.

### Sev-1: production-down or data-integrity at risk

| Step | Commitment |
|------|------------|
| Acknowledgement | Same business day (within ~4 business hours during 9-5 CDT weekdays; best-effort outside business hours) |
| Triage decision (severity confirmation, root-cause direction) | Within 24 hours |
| Fix or rollback recommendation | Within 48 hours where feasible |

Sev-1 examples:
- A production deployment using `@synapt-dev/extract` is broken (validation rejecting valid documents, finalize pipeline crashing, etc.)
- A schema regression causes data loss or corruption
- A supply-chain compromise is detected (per `SECURITY.md` incident protocol)
- A behavioral shift (output changes for same input) appears in a release without a CHANGELOG entry

The "fix or rollback recommendation" language is deliberate. Real fixes can take longer than 48 hours; rollbacks are usually a same-day operation (revert to previous package version, pin in `package.json` / `pyproject.toml`, redeploy). We commit to either fixing within 48 hours or providing the partner with a clean rollback path.

### Sev-2: degraded but workaround available

| Step | Commitment |
|------|------------|
| Acknowledgement | 1 business day |
| Triage decision | 3 business days |
| Fix | Next minor release (or sooner if blocking) |

Sev-2 examples:
- A capability under-extracts compared to documented behavior
- A validator emits an unhelpful error message
- A performance regression that doubles extraction latency without breaking correctness
- A documentation mismatch

### Sev-3: cosmetic, edge case, or future-version concern

| Step | Commitment |
|------|------------|
| Acknowledgement | 3 business days |
| Triage decision | 2 weeks |
| Fix | Tracked; no commitment beyond "in the queue" |

Sev-3 examples:
- Typos in documentation
- Edge cases that affect <1% of expected use
- Feature requests for future schema additions

Sev-3 issues stay open until resolved. We don't auto-close stale issues.

## Reproducibility expectations

For us to investigate a reported bug, we need enough context to reproduce locally.

| Required | Why we need it |
|----------|----------------|
| Package version (`@synapt-dev/extract@x.y.z` or `synapt-extract` version) | Bugs are version-specific |
| Input that triggers the bug | The transcript text or LLM output passed to extract |
| Observed output | What the library returned (or the error it threw) |
| Expected output | What you expected instead, with reasoning |
| Reproduction script (if possible) | Minimal code that demonstrates the issue |

| Helpful but not required | Why it accelerates triage |
|---------------------------|---------------------------|
| Schema version expectation | If you're targeting a specific schema version |
| Capabilities used | The capability set passed to `buildExtractionPrompt` or detected by `detectCapabilities` |
| Profile used | If using a named profile (`minimal` / `standard` / `full`) |
| LLM provider details | Which provider's output triggered the issue |
| Stack trace | For uncaught errors, the full stack |

### Sanitization

For partners with sensitive source data, please sanitize before filing public GitHub issues. Replace identifying details with generic placeholders that preserve the structural shape of the input. If a sanitized reproduction doesn't reproduce the bug, that itself is useful information.

For sensitive reproductions you cannot sanitize, contact `support@synapt.dev` for a private channel.

### Our reproducibility commitment back

When you provide a complete reproduction, we commit to **reproducing the issue locally before requesting more information**. We will not bounce you with "can you provide more context?" if the report already contains a working reproduction.

If we cannot reproduce, we say so explicitly and walk through what we tried.

## What we do not commit to

- **24/7 oncall**: We are a small team. Sev-1 SLA is best-effort during business hours (9-5 CDT weekdays). We do not have round-the-clock pager rotation.
- **Custom feature work for individual partners**: We prioritize features that benefit the broader ecosystem. Custom modifications are a paid engagement; contact `partnerships@synapt.dev`.
- **LLM provider issues**: If the bug is in your LLM provider's output (a model that hallucinates schema-violating output), we can help diagnose but the fix is in your prompt or model choice, not in `@synapt-dev/extract`.
- **Third-party tooling integration**: We test against Node 18+, Deno, and Python 3.10+; other environments are best-effort.
- **Performance guarantees**: We aim for reasonable performance but don't commit to specific latency or throughput numbers. Performance regressions are sev-2 issues.

## Communication norms

When you file an issue, expect:

- **Plain language**: We avoid jargon where possible
- **Honest assessment**: If we can't reproduce, we say so. If a fix will take longer than the SLA, we say why. If the issue is by design, we explain the rationale.
- **Visible progress**: Issues get labels for tracking: `triage`, `confirmed`, `fix-in-progress`
- **Closing rationale**: We don't close issues without explanation

## Accountability

We publish a quarterly support-metrics summary (mean time to acknowledgement, mean time to fix, sev-1 incident count) in our public CHANGELOG so partners can hold us accountable to our SLA.

## Cross-references

- `SECURITY.md` -- vulnerability disclosure process, supply-chain commitments, verification commands
- `VERSIONING.md` -- version policy, support windows, behavioral-shift commitments
- Issue templates: https://github.com/synapt-dev/extract/tree/main/.github/ISSUE_TEMPLATE
