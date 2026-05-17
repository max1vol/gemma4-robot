# iPhone Gemma Inference Server Bridge

This is a private-device iPhone app scaffold for running Gemma 4 E2B on the phone and letting the Raspberry Pi send inference jobs over a connection initiated by the iPhone.

## Current Platform Notes

- The current default model is `gemma-4-E2B-it-Q4_K_M.gguf`, downloaded from Hugging Face into iPhone application storage.
- The current default multimodal projector is `mmproj-F16.gguf`, downloaded from the same Hugging Face model family into iPhone application storage.
- The current default runtime is `llama.cpp` on Metal GPU with `libmtmd` for image/audio input.
- Do not commit `.gguf`, `.litertlm`, `.task`, local runtime builds, certificates, provisioning profiles, or signing keys to this repo.
- The iPhone connects outbound to the Pi with WebSocket: `ws://pi3:8765/worker`.
- Direct LAN works when the phone and Pi are on the same network. The verified LAN bridge URL is `ws://192.168.1.174:8765/worker`, with the iPhone observed as `192.168.1.72`.
- Tailscale remains useful for discovery, SSH, and fallback routing. The app can use the Pi's Tailscale IP if LAN is not available.
- Text-only Pi clients can call `POST /generate-stream`, which returns newline-delimited JSON token events.
- Audio/image Pi clients call `POST /generate-media-stream` with the binary `G4GEN01` frame. This keeps media binary end to end and avoids base64.
- Pi TTS clients should call `POST /tts-stream`, which streams raw 24 kHz mono `S16_LE` PCM back from the iPhone. This path is binary chunked audio, not base64 and not a WAV file.
- Pi pose clients should call `POST /pose-frame` with a binary body. Use `deflate_rgb24` for RGB frames and `deflate_yuv420` for live `rpicam-vid` YUV frames; do not send pose frames as base64 JSON for real-time use.
- iOS can listen for incoming TCP connections while the app is active, but it is not a reliable always-on background server. The outbound iPhone-to-Pi connection is the right default for this setup.
- The checked-in app refuses to generate unless a real runtime is linked and the model loads successfully. There are no fake model responses in the fallback path.
- The iPhone Gemma bridge accepts text, image, and audio prompt parts when the projector is loaded.
- The current Pi voice bot records from the USB webcam microphone, sends the WAV to iPhone Gemma as raw audio input, streams text back to the full-screen terminal display, then streams iPhone Kokoro TTS audio back to Pi HDMI playback.

## Runtime Dependency

The source of truth for the iOS project is `ios/GemmaPi/project.yml`. Generated Xcode files and DerivedData are ignored.

Build the local llama.cpp framework before building the app on a fresh machine:

```sh
ios/GemmaPi/scripts/build_llama_ios.sh
```

The framework is written under ignored `ios/GemmaPi/LocalLlama/`. The app target embeds and signs `LocalLlama/llama.framework`.

The LiteRT-LM Swift package is still present as a fallback dependency, but the app prefers `llama.cpp` when `llama.framework` is linked.

The iPhone TTS runtime uses FluidAudio. `fluid-kokoro-ane` is the default and the only selectable app UI backend right now because it was validated with Gemma loaded. `fluid-pocket` remains in code for comparison, but it is not exposed in the app until it is revalidated on the same device with Gemma loaded.

Benchmark report:

```text
docs/IPHONE_LLAMA_BENCHMARK_REPORT.md
```

Pose offload benchmark report:

```text
docs/IPHONE_POSE_BENCHMARK_REPORT.md
```

## Build From CLI

Generate the Xcode project:

```sh
ios/GemmaPi/scripts/generate_project.sh
```

Build for the simulator:

```sh
ios/GemmaPi/scripts/build_simulator.sh
```

Install on the paired iPhone:

```sh
ios/GemmaPi/scripts/install_device.sh
```

If Xcode says the Developer Disk Image is not mounted, unlock the iPhone, keep it on the Home Screen, and rerun the same command.

Optional overrides:

```sh
IOS_DEVICE_ID=<device-uuid> IOS_DEVELOPMENT_TEAM=<team-id> ios/GemmaPi/scripts/install_device.sh
```

Generated Xcode files and DerivedData are ignored. The source of truth is `ios/GemmaPi/project.yml` plus the Swift files under `ios/GemmaPi/GemmaPiApp/`.

## Pi Bridge

Copy the bridge to the Pi:

```sh
tailscale ssh max@pi3 'mkdir -p ~/gemma4-robot/scripts'
tailscale ssh max@pi3 'cat > ~/gemma4-robot/scripts/iphone_llm_bridge.py' < scripts/iphone_llm_bridge.py
```

Run it on the Pi:

```sh
tailscale ssh max@pi3 'python3 ~/gemma4-robot/scripts/iphone_llm_bridge.py serve --host 0.0.0.0 --port 8765'
```

Current Pi autostart:

```cron
# gemma iphone bridge autostart
@reboot /usr/bin/python3 /home/max/gemma4-robot/scripts/iphone_llm_bridge.py serve --host 0.0.0.0 --port 8765 >> /home/max/gemma4-robot/iphone_llm_bridge.log 2>&1
```

The voice bot and kiosk are systemd services. The bridge is started by the `max` user's crontab.

In the iPhone app, keep the bridge URL as:

```text
ws://pi3:8765/worker
```

If MagicDNS does not resolve inside the app, use the Pi's Tailscale IP:

```text
ws://100.x.y.z:8765/worker
```

When LAN routing is available, use the direct Pi address:

```text
ws://192.168.1.174:8765/worker
```

Send a prompt from another Pi shell:

```sh
python3 ~/gemma4-robot/scripts/iphone_llm_bridge.py prompt 'Say hello from the iPhone worker.'
```

The Rust harness uses the streaming endpoint automatically:

```sh
GEMMA_AGENT_PROVIDER=ios-bridge GEMMA_IOS_BRIDGE_URL=http://127.0.0.1:8765 bin/gemma-agent-harness prompt 'Say hello from the iPhone worker.'
```

Raw stream smoke test:

```sh
curl -N -X POST http://127.0.0.1:8765/generate-stream \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"hi","max_tokens":64,"timeout":300}'
```

Raw audio media smoke test:

```sh
GEMMA_AGENT_PROVIDER=ios-bridge GEMMA_IOS_BRIDGE_URL=http://127.0.0.1:8765 \
  bin/gemma-agent-harness prompt 'Answer the spoken request. Do not use emojis.' \
  --audio /tmp/request.wav
```

The app can also exercise its own microphone recording path without touching
the screen:

```sh
ios/GemmaPi/scripts/device_app.sh logs --audio-recording-smoke
```

The 2026-05-17 iPhone smoke run recorded a 1.283 second WAV from the app, sent
that WAV to native Gemma audio input through `llama.cpp`/`mtmd`, generated a
text response, and produced no new crash report. This specifically validates
the app microphone path; the separate OpenAI TTS audio benchmark validates a
known spoken phrase and measured the text-vs-audio overhead.

Check state:

```sh
python3 ~/gemma4-robot/scripts/iphone_llm_bridge.py health
```

## iPhone TTS

The bridge exposes iPhone-side TTS through `POST /tts-stream`:

```sh
curl -sS --max-time 600 \
  -o /tmp/iphone-tts.raw \
  -X POST http://127.0.0.1:8765/tts-stream \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from iPhone text to speech.","tts_backend":"fluid-kokoro-ane","timeout":600}'
```

The response body is raw PCM with headers:

```text
X-Audio-Format: s16le
X-Audio-Sample-Rate: 24000
X-Audio-Channels: 1
```

Play the stream or saved raw file on the Pi HDMI device:

```sh
aplay -q -D plughw:vc4hdmi,0 -t raw -f S16_LE -r 24000 -c 1 /tmp/iphone-tts.raw
```

Run the iPhone-side TTS benchmark:

```sh
curl -sS --max-time 900 \
  -X POST http://127.0.0.1:8765/tts-benchmark \
  -H 'Content-Type: application/json' \
  -d '{"text":"The robot is ready. Hold the button and speak.","timeout":900}'
```

Measured on the connected iPhone with Gemma already loaded:

```text
fluid-kokoro-ane / af_heart:
  audio: 2.975 s
  elapsed: 1.046 s
  first audio: 1.038 s
  realtime factor: 2.84x
  chunks: 15

fluid-pocket / alba:
  audio: 3.360 s
  elapsed: 4.484 s
  first audio: 3.261 s
  realtime factor: 0.75x
  chunks: 42
```

The first cold `fluid-kokoro-ane` request can take much longer because FluidAudio downloads and loads model assets; one observed cold bridge call took about 56 seconds wall-clock while the actual iPhone synthesis metric was 4.54 seconds for 2.52 seconds of audio. Warm requests are the useful latency number for the voice loop.

## Pi Voice Bot

The Pi systemd unit starts the Rust harness directly at boot.

Current command shape:

```sh
~/gemma4-robot/bin/gemma-agent-harness \
  --env-file ~/gemma4-robot/.env \
  voice-bot \
  --button-source microbit-serial \
  --microbit-device auto \
  --led-source none \
  --playback-device plughw:vc4hdmi,0 \
  --capture-device plughw:Camera,0 \
  --sample-rate 48000 \
  --channels 2 \
  --transcription-provider none \
  --tts-provider iphone \
  --iphone-tts-backend fluid-kokoro-ane \
  --fullscreen-terminal \
  --startup-greeting=
```

The USB webcam microphone currently works as `plughw:Camera,0` at 48 kHz stereo. Mono capture failed on this device with an ALSA input/output error, so keep `--sample-rate 48000 --channels 2` for the voice bot.

The voice bot writes a JSON status file and also renders a full-screen terminal
view. It shows recording, sending, receiving, and playback states, the audio
byte count, audio duration, generated text, output token estimate, generation
speed, and elapsed time.

The current button input is the micro:bit A button over USB serial. Flash `scripts/microbit/microbit_a_button_serial.py` to the micro:bit in the Nezha Pro board; it emits `A:down` while held and `A:up` on release. The removed Voice HAT GPIO button path is still available with `--button-source gpio`, but it is no longer the default service path.

Binary pose smoke test from the Pi:

```sh
python3 scripts/pose_iphone_transport_bench.py \
  --bridge-url http://127.0.0.1:8765 \
  --rgb-file kiosk/exercise_frame.rgb \
  --rgb-size 320x240 \
  --sizes 320x240 \
  --formats deflate_rgb24 \
  --backend gpu \
  --model full \
  --warmup 1 \
  --repeats 3
```
