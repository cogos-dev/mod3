import type {
  SpeechProviderConfig,
  SpeechProviderPlugin,
  SpeechVoiceOption,
} from "openclaw/plugin-sdk/speech";

type Mod3ProviderConfig = {
  enabled: boolean;
  baseUrl: string;
  voice: string;
  speed: number;
};

const DEFAULT_BASE_URL = "http://localhost:7860";
const DEFAULT_VOICE = "bm_lewis";
const DEFAULT_SPEED = 1.25;

function readConfig(config: SpeechProviderConfig): Mod3ProviderConfig {
  const raw =
    (config.mod3 as Record<string, unknown> | undefined) ??
    (config.providers as Record<string, unknown> | undefined)?.mod3 as Record<string, unknown> | undefined ??
    config;
  return {
    enabled: typeof raw?.enabled === "boolean" ? raw.enabled : true,
    baseUrl:
      typeof raw?.baseUrl === "string" && raw.baseUrl.trim()
        ? raw.baseUrl.trim()
        : DEFAULT_BASE_URL,
    voice:
      typeof raw?.voice === "string" && raw.voice.trim()
        ? raw.voice.trim()
        : DEFAULT_VOICE,
    speed:
      typeof raw?.speed === "number" && Number.isFinite(raw.speed)
        ? raw.speed
        : DEFAULT_SPEED,
  };
}

export function buildMod3SpeechProvider(): SpeechProviderPlugin {
  return {
    id: "mod3",
    label: "Mod³ (Local)",
    aliases: ["mod3-tts"],
    autoSelectOrder: 5,

    resolveConfig: ({ rawConfig }) => readConfig(rawConfig),

    isConfigured: ({ providerConfig }) => {
      const cfg = readConfig(providerConfig);
      return cfg.enabled;
    },

    synthesize: async (req) => {
      const cfg = readConfig(req.providerConfig);
      const voice =
        (typeof req.providerOverrides?.voice === "string" && req.providerOverrides.voice.trim()) ||
        cfg.voice;
      const speed =
        typeof req.providerOverrides?.speed === "number"
          ? req.providerOverrides.speed
          : cfg.speed;

      const res = await fetch(`${cfg.baseUrl}/v1/synthesize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: req.text,
          voice,
          speed,
          format: "wav",
        }),
        signal: AbortSignal.timeout(req.timeoutMs),
      });

      if (!res.ok) {
        const body = await res.text().catch(() => "");
        throw new Error(`Mod³ synthesis error (${res.status}): ${body}`);
      }

      const audioBuffer = Buffer.from(await res.arrayBuffer());

      return {
        audioBuffer,
        outputFormat: "audio-wav",
        fileExtension: ".wav",
        voiceCompatible: false,
      };
    },

    listVoices: async (req) => {
      const cfg = readConfig(req?.providerConfig ?? {});
      const res = await fetch(`${cfg.baseUrl}/v1/voices`, {
        signal: AbortSignal.timeout(5000),
      });

      if (!res.ok) {
        throw new Error(`Mod³ voices error (${res.status})`);
      }

      const data = (await res.json()) as {
        engines: Record<
          string,
          { voices: string[]; default_voice: string; supports: string[] }
        >;
      };

      const voices: SpeechVoiceOption[] = [];
      for (const [engine, info] of Object.entries(data.engines)) {
        for (const v of info.voices) {
          voices.push({
            id: v,
            name: v,
            category: engine,
            description: `${engine} voice${info.supports.length ? ` (${info.supports.join(", ")})` : ""}`,
          });
        }
      }
      return voices;
    },
  };
}
