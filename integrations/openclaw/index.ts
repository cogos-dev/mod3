import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { buildMod3SpeechProvider } from "./speech-provider.js";

export default definePluginEntry({
  id: "mod3",
  name: "Mod³ Local TTS",
  description: "Local multi-model TTS on Apple Silicon via Mod³",
  register(api) {
    api.registerSpeechProvider(buildMod3SpeechProvider());
  },
});
