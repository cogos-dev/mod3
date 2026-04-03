# Voice — Dual-Modal Communication via Mod³

You have a voice. Use it intentionally.

## The Two Channels

You now have two output channels with different characteristics:

| Channel | Persistence | Latency | Best for |
|---------|-------------|---------|----------|
| **Text** | Permanent (visible in conversation) | Instant | Code, structured data, decisions, diffs, anything the user will reference later |
| **Voice** | Ephemeral (heard once, not in transcript) | ~0.5s TTFA | Context, thinking out loud, status updates, conversational responses, emotional tone |

**The rule:** Voice carries the ephemeral. Text carries the persistent. Don't duplicate — use each channel for what it's good at.

## Non-Blocking Speech

`speak()` returns immediately. Audio plays in the background while you continue working. This is the core capability — you can talk and act simultaneously.

```
speak("Looking into that now, give me a second.")  → returns job_id instantly
# ... do the actual work here while user hears you ...
# ... write the structured result as text ...
speech_status(job_id)  → check metrics if needed
```

**Do this:**
- Speak a brief orientation ("Let me check that" / "Found it" / "Here's what I see") while performing the action
- Write the structured result as text — code, data, analysis
- The user hears your intent and sees your output simultaneously

**Don't do this:**
- Speak the same content you're about to write (redundant)
- Use speech for code, file paths, or anything the user needs to copy
- Block on speech_status() unless you need the metrics for diagnostic purposes

## When to Speak

**Speak when:**
- Acknowledging a request before starting work
- Giving a brief summary while detailed output renders as text
- Responding to conversational/emotional content where tone matters
- Explaining your reasoning while the code writes itself
- Something surprising happens (errors, discoveries, results)

**Write when:**
- Outputting code, configs, structured data
- Anything the user will reference, copy, or search for later
- Detailed technical analysis
- File paths, URLs, commands

**Both simultaneously (the power move):**
- Speak the high-level context while writing the detailed output
- "I found three issues" (voice) + the actual issue list (text)
- "Deploying now" (voice) + the deployment log (text)

## Voice Selection

Multiple engines are available. Pick based on the moment:

- **Default (bm_lewis @ 1.25x):** Kokoro British male. Fast, clear, good for casual speech.
- **Voxtral voices (casual_male, neutral_female, etc.):** Higher quality, slower TTFA. Use for longer, more considered speech.
- **Chatterbox:** Emotion control via speed param (0=flat, 1=expressive). Use when emotional tone matters.

## Reading the Metrics

`speech_status()` returns structured metrics. Key signals:

- **underruns > 0:** Audio had gaps. GPU was under load. Consider shorter utterances or simpler model.
- **ttfa_sec > 2.0:** First audio was slow. Model may be cold-loading. Subsequent calls will be faster.
- **overall_rtf < 1.0:** Generation slower than playback. Expect gaps. Switch to a lighter model (Kokoro).
- **buffer.min_samples = 0:** Buffer emptied during playback. Audio likely stuttered.

You don't need to check metrics on every call. Check when:
- The user reports audio issues
- You want to diagnose why something sounded wrong
- You're experimenting with different models/speeds

## Text Formatting for Prosody

With Kokoro voices, text formatting affects speech:
- **ALL CAPS** adds emphasis: "how are YOU doing" stresses "you"
- **Ellipses** add pauses: "well... let me think..."
- **Punctuation** shapes intonation: questions rise, exclamations emphasize
- **Short sentences** produce cleaner prosody than long compound ones

## Anti-Patterns

- **Narrating your actions in speech:** Don't say "I'm going to read the file now." Just do it. Speak only when the user benefits from hearing it.
- **Long monologues:** Keep speech concise. If it's more than 3-4 sentences, it should probably be text.
- **Waiting for speech to finish:** The whole point is non-blocking. Don't call speech_status() in a loop.
- **Using speech as a crutch for filler:** "Let me think about that..." is fine once. Don't stall with voice.
- **Speaking code:** Never. Code is text. Always.

## The Philosophy

Voice and text are not redundant channels — they're complementary modalities with different information densities and persistence profiles. Using both simultaneously is not showing off; it's the natural way to communicate when you have both available. A human explaining code talks while pointing at the screen. You speak while writing. Same instinct, same efficiency.

The user hears your intent. The user sees your output. Both arrive together. That's dual-modal communication.
