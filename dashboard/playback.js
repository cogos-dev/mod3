/**
 * Streaming audio playback engine.
 * Receives Int16 PCM chunks and plays them seamlessly via Web Audio API.
 */
class AudioPlayback {
  constructor(sampleRate = 24000) {
    this.sampleRate = sampleRate;
    this.queue = [];
    this.isPlaying = false;
    this.audioContext = null;
    this.currentSource = null;
    this.nextStartTime = 0;
    this.onPlaybackStart = null;
    this.onPlaybackEnd = null;
    this.sinkId = undefined; // output device ID
  }

  _ensureContext() {
    if (!this.audioContext || this.audioContext.state === "closed") {
      this.audioContext = new AudioContext({ sampleRate: this.sampleRate });
      // Apply selected output device if set
      if (this.sinkId !== undefined && this.audioContext.setSinkId) {
        this.audioContext.setSinkId(this.sinkId).catch(() => {});
      }
    }
    if (this.audioContext.state === "suspended") {
      this.audioContext.resume();
    }
  }

  async setOutputDevice(deviceId) {
    this.sinkId = deviceId;
    if (this.audioContext && this.audioContext.setSinkId) {
      await this.audioContext.setSinkId(deviceId);
    }
  }

  /** Enqueue raw Int16 PCM for playback. */
  enqueue(pcmArrayBuffer) {
    this._ensureContext();

    const int16 = new Int16Array(pcmArrayBuffer);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768;
    }

    const buffer = this.audioContext.createBuffer(1, float32.length, this.sampleRate);
    buffer.getChannelData(0).set(float32);
    this.queue.push(buffer);

    if (!this.isPlaying) this._playNext();
  }

  /** Enqueue a full WAV file (with header) for playback. */
  async enqueueWav(wavArrayBuffer) {
    this._ensureContext();
    try {
      const audioBuffer = await this.audioContext.decodeAudioData(wavArrayBuffer.slice(0));
      this.queue.push(audioBuffer);
      if (!this.isPlaying) this._playNext();
    } catch (err) {
      console.error("[AudioPlayback] Failed to decode WAV:", err);
    }
  }

  _playNext() {
    if (this.queue.length === 0) {
      this.isPlaying = false;
      if (this.onPlaybackEnd) this.onPlaybackEnd();
      return;
    }

    if (!this.isPlaying) {
      this.isPlaying = true;
      this.nextStartTime = this.audioContext.currentTime;
      if (this.onPlaybackStart) this.onPlaybackStart();
    }

    const buffer = this.queue.shift();
    const source = this.audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(this.audioContext.destination);
    source.onended = () => this._playNext();

    // Schedule this chunk right after the previous one for gapless playback
    const startTime = Math.max(this.nextStartTime, this.audioContext.currentTime);
    source.start(startTime);
    this.nextStartTime = startTime + buffer.duration;
    this.currentSource = source;
  }

  flush() {
    this.queue = [];
    if (this.currentSource) {
      try { this.currentSource.stop(); } catch {}
    }
    this.isPlaying = false;
    this.nextStartTime = 0;
  }

  setSampleRate(rate) {
    if (rate !== this.sampleRate) {
      this.sampleRate = rate;
      // Close old context so next enqueue creates one with the right rate
      if (this.audioContext) {
        this.audioContext.close();
        this.audioContext = null;
      }
    }
  }
}
