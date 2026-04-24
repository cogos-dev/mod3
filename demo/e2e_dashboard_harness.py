"""Headless end-to-end harness for the Mod³ dashboard pipeline.

Simulates a dashboard client without a browser or a microphone:
  1. Connects to ws://localhost:7860/ws/chat
  2. Sends a text_message, collects response_text frames + audio frames
  3. Writes captured TTS audio to /tmp/mod3-harness-tts.wav
  4. Sends an interrupt frame mid-response (if TTS still playing) to exercise barge-in
  5. In parallel, subscribes to kernel trace events via /v1/events/stream?bus_id=bus_cycle_trace
  6. Emits a summary

Exits 0 on success, 1 on any missing path.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
import wave
from pathlib import Path

from websockets.client import connect as ws_connect

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bus_bridge import KernelBusSubscriber  # noqa: E402

WS_URL = "ws://localhost:7860/ws/chat"
KERNEL_STREAM = "http://localhost:6931/v1/events/stream"
TTS_OUT = Path("/tmp/mod3-harness-tts.wav")
RESULT: dict = {
    "response_text": [],
    "audio_frames": 0,
    "audio_bytes": 0,
    "trace_events": [],
    "interrupt_sent": False,
    "started": time.time(),
}


async def subscribe_trace(seconds: float) -> None:
    sub = KernelBusSubscriber(url=KERNEL_STREAM, bus_filter="*")
    try:
        deadline = time.time() + seconds
        async for env in sub.stream():
            kind = env.kind or "?"
            RESULT["trace_events"].append({"kind": kind, "ts": env.ts})
            if time.time() > deadline:
                break
    finally:
        await sub.close()


def _pcm_frames_to_wav(frames: list[bytes], sample_rate: int, path: Path) -> int:
    pcm = b"".join(frames)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return len(pcm)


async def run_dashboard_session(interrupt_after_s: float, text: str) -> None:
    audio_pcm: list[bytes] = []
    audio_sr: int | None = None
    interrupt_task: asyncio.Task | None = None

    async with ws_connect(WS_URL, max_size=32 * 1024 * 1024) as ws:
        # 1. config
        await ws.send(json.dumps({"type": "config", "voice": "bm_lewis", "model": None}))
        # 2. user turn
        await ws.send(json.dumps({"type": "text_message", "text": text}))

        async def deferred_interrupt() -> None:
            await asyncio.sleep(interrupt_after_s)
            RESULT["interrupt_sent"] = True
            await ws.send(json.dumps({"type": "interrupt"}))
            await asyncio.sleep(0.2)
            await ws.send(json.dumps({"type": "text_message", "text": "wait hold on"}))

        interrupt_task = asyncio.create_task(deferred_interrupt())

        end_deadline = time.time() + 45
        got_done = False
        done_ts: float | None = None
        while time.time() < end_deadline:
            if got_done and done_ts and (time.time() - done_ts) > 4.0:
                break
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            if isinstance(msg, bytes):
                continue
            try:
                frame = json.loads(msg)
            except Exception:
                continue
            t = frame.get("type")
            print(f"  [frame] type={t} keys={sorted(frame.keys())}", flush=True)
            if t == "response_text":
                RESULT["response_text"].append(frame.get("text", ""))
            elif t == "audio":
                RESULT["audio_frames"] += 1
                RESULT["audio_bytes"] += len(frame.get("data", "")) // 4 * 3
                try:
                    wav_b64 = frame.get("data", "")
                    wav_bytes = base64.b64decode(wav_b64)
                    if wav_bytes.startswith(b"RIFF"):
                        with wave.open(__import__("io").BytesIO(wav_bytes), "rb") as r:
                            audio_sr = r.getframerate()
                            audio_pcm.append(r.readframes(r.getnframes()))
                    else:
                        sr = int(frame.get("sample_rate", 24000))
                        audio_sr = audio_sr or sr
                        audio_pcm.append(wav_bytes)
                except Exception as e:
                    print(f"  (audio decode err: {e})", flush=True)
            elif t == "response_complete":
                got_done = True
                done_ts = time.time()
            elif t == "trace_event":
                RESULT["trace_events"].append(
                    {
                        "kind": (frame.get("event") or {}).get("kind", "?"),
                        "ts": (frame.get("event") or {}).get("ts"),
                    }
                )
            if RESULT["audio_frames"] >= 2 and RESULT["interrupt_sent"] is False:
                await deferred_interrupt()
                break

        if interrupt_task and not interrupt_task.done():
            interrupt_task.cancel()

    if audio_pcm and audio_sr:
        size = _pcm_frames_to_wav(audio_pcm, audio_sr, TTS_OUT)
        RESULT["tts_wav_path"] = str(TTS_OUT)
        RESULT["tts_wav_bytes"] = size
        RESULT["tts_wav_sr"] = audio_sr


async def main() -> int:
    text = os.environ.get("HARNESS_PROMPT", "In one short sentence, describe the planet Jupiter.")
    skip_interrupt = os.environ.get("HARNESS_SKIP_INTERRUPT") == "1"
    trace_task = asyncio.create_task(subscribe_trace(seconds=45.0))
    try:
        await asyncio.wait_for(
            run_dashboard_session(interrupt_after_s=9999 if skip_interrupt else 2.5, text=text), timeout=90.0
        )
    except asyncio.TimeoutError:
        print("[warn] dashboard session timed out")
    await asyncio.sleep(1.5)
    trace_task.cancel()
    try:
        await trace_task
    except asyncio.CancelledError:
        pass

    print("\n=== HARNESS SUMMARY ===")
    print(f"prompt: {text!r}")
    print(f"response_text frames: {len(RESULT['response_text'])}")
    if RESULT["response_text"]:
        joined = " ".join(RESULT["response_text"])[:400]
        print(f"  preview: {joined!r}")
    print(
        f"audio frames: {RESULT['audio_frames']}  wav path: {RESULT.get('tts_wav_path', '-')}  sr={RESULT.get('tts_wav_sr', '-')}"
    )
    print(f"interrupt_sent: {RESULT['interrupt_sent']}")
    print(
        f"trace events observed: {len(RESULT['trace_events'])}  kinds: {sorted({e['kind'] for e in RESULT['trace_events']})}"
    )

    ok = len(RESULT["response_text"]) > 0 and RESULT["audio_frames"] > 0 and len(RESULT["trace_events"]) > 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
