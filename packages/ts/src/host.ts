import type {
  EmbeddingCallback,
  ExtractCallbacks,
  LlmCallback,
  LogCallback,
} from "./extract.js";

export type HostMaybePromise<T> = T | Promise<T>;

export type HostHashAlgorithm = "sha256";
export type HostHashEncoding = "hex" | "base64";

export interface HostHashRequest {
  algorithm: HostHashAlgorithm;
  data: string | Uint8Array;
  encoding?: HostHashEncoding;
}

export interface HostArtifact {
  name: string;
  content: string | Uint8Array;
  mediaType?: string;
  metadata?: Record<string, unknown>;
}

export interface HostArtifactWriteResult {
  uri?: string;
  path?: string;
  digest?: string;
  bytes?: number;
}

export interface SynaptHost {
  callLlm: LlmCallback;
  getEmbedding?: EmbeddingCallback;
  log?: LogCallback;
  now?: () => string;
  randomId?: () => string;
  hash?: (request: HostHashRequest) => HostMaybePromise<string>;
  writeArtifact?: (artifact: HostArtifact) => HostMaybePromise<HostArtifactWriteResult>;
}

export type SynaptHostFeature =
  | "callLlm"
  | "getEmbedding"
  | "log"
  | "now"
  | "randomId"
  | "hash"
  | "writeArtifact";

export type SynaptHostRequirements = Partial<Record<SynaptHostFeature, boolean>>;

const HOST_FEATURES: SynaptHostFeature[] = [
  "callLlm",
  "getEmbedding",
  "log",
  "now",
  "randomId",
  "hash",
  "writeArtifact",
];

export function callbacksFromHost(host: SynaptHost): ExtractCallbacks {
  return {
    callLlm: host.callLlm,
    ...(host.getEmbedding !== undefined ? { getEmbedding: host.getEmbedding } : {}),
    ...(host.log !== undefined ? { log: host.log } : {}),
  };
}

export function supportedHostFeatures(host: SynaptHost): SynaptHostFeature[] {
  const features: SynaptHostFeature[] = [];
  for (const feature of HOST_FEATURES) {
    if (host[feature] !== undefined) {
      features.push(feature);
    }
  }
  return features;
}

export function unsupportedHostFeatures(
  host: SynaptHost,
  requirements: SynaptHostRequirements,
): SynaptHostFeature[] {
  const supported = new Set(supportedHostFeatures(host));
  return (Object.entries(requirements) as [SynaptHostFeature, boolean][])
    .filter(([, required]) => required)
    .map(([feature]) => feature)
    .filter((feature) => !supported.has(feature));
}
