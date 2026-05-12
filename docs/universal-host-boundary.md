# Universal Host Boundary

**Status:** 0.5.0 groundwork.

The default TypeScript entry point, `@synapt-dev/extract`, is intended to run in hosts that do not provide Node APIs: browsers, edge workers, and a future WASM component host. The core package now treats model calls, embeddings, logging, clocks, hashing, and artifact persistence as host imports instead of ambient runtime capabilities.

Node-specific helpers remain available through explicit subpaths:

- `@synapt-dev/extract/openai`
- `@synapt-dev/extract/artifacts`

Those subpaths may use Node APIs. The default entry point must not.

## Host Interface

```typescript
interface SynaptHost {
  callLlm: LlmCallback;
  getEmbedding?: EmbeddingCallback;
  log?: LogCallback;
  now?: () => string;
  randomId?: () => string;
  hash?: (request: HostHashRequest) => string | Promise<string>;
  writeArtifact?: (artifact: HostArtifact) => HostArtifactWriteResult | Promise<HostArtifactWriteResult>;
}
```

`extract()` still accepts `ExtractCallbacks` directly. `callbacksFromHost(host)` adapts a `SynaptHost` to the existing callback object:

```typescript
import { callbacksFromHost, extract } from "@synapt-dev/extract";

const result = await extract(text, callbacksFromHost(host), {
  profile: "full",
  embeddingInputs: "all",
});
```

## Required Features

Callers can use `unsupportedHostFeatures(host, requirements)` to fail early when a requested pipeline needs host features the current runtime does not provide:

```typescript
const missing = unsupportedHostFeatures(host, {
  callLlm: true,
  getEmbedding: true,
  hash: true,
});

if (missing.length > 0) {
  throw new Error(`Unsupported host features: ${missing.join(", ")}`);
}
```

This is separate from extraction capabilities such as `entities`, `goals`, and `relations`. Host features describe runtime services; extraction capabilities describe prompt/schema output shape.

## WASM Direction

The WASM boundary should map these host fields to imports without changing extraction semantics:

- `callLlm`: caller-owned model routing and credentials
- `getEmbedding`: caller-owned embedding routing and credentials
- `log`: host observability sink
- `now`: deterministic timestamp source
- `randomId`: host-generated IDs when future packets need them
- `hash`: host crypto, starting with SHA-256
- `writeArtifact`: host persistence for audit bundles, fixtures, or traces

Core extraction should remain deterministic and side-effect-free except for explicit host calls.

## Guardrail

The TypeScript package has a universal-entry guard:

```bash
cd packages/ts
npm run check:universal
```

The guard builds `dist/index.js`, follows local static imports, and fails if the default entry graph imports Node built-in modules. CI also runs the guard against the packed npm artifact.
