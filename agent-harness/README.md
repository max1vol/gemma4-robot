# Gemma Agent Harness

This is the Rust LLM harness for the robot. It keeps the agent loop separate
from the model transport so the same loop can use Google-hosted Gemma or an
iPhone-hosted Gemma worker. Both provider paths are streaming-first; the CLI
prints text deltas as they arrive and still returns the final response to the
agent loop.

Build it on the Mac with Apple container, not on the Raspberry Pi:

```sh
scripts/build_agent_harness_container.sh
```

The generated Linux arm64 binary is copied to:

```text
bin/gemma-agent-harness
```

Run a Google-hosted Gemma prompt from this checkout:

```sh
bin/gemma-agent-harness prompt "Reply with one short sentence."
```

Send multimodal parts through the Google provider:

```sh
bin/gemma-agent-harness prompt "Describe this image." --image ./photo.jpg
bin/gemma-agent-harness prompt "Answer the spoken request." --audio ./request.wav
```

The iPhone bridge accepts text plus raw image/audio parts. Media requests use a
binary `G4GEN01` frame through `POST /generate-media-stream`, so the recorded
WAV is not base64 encoded on the Pi-to-iPhone path. The iPhone must have both
the Gemma GGUF and matching `mmproj` loaded for raw `--image` and `--audio`
requests.

Use the Pi-side iPhone worker bridge:

```sh
bin/gemma-agent-harness --provider ios-bridge --ios-bridge-url http://127.0.0.1:8765 prompt "Hello from the Pi."
```

The same configuration can live in `~/gemma4-robot/.env` on the Pi:

```sh
GEMMA_AGENT_PROVIDER=ios-bridge
GEMMA_IOS_BRIDGE_URL=http://127.0.0.1:8765
GEMMA_AGENT_MODEL=gemma-4-E2B-it
```

With that file in place:

```sh
bin/gemma-agent-harness prompt "Hello from the iPhone Gemma worker."
```

Run the Voice Kit button agent against the same iPhone Gemma bridge:

```sh
bin/gemma-agent-harness --env-file ~/gemma4-robot/.env voice-bot \
  --button-source microbit-serial \
  --microbit-device auto \
  --led-source none \
  --playback-device plughw:vc4hdmi,0 \
  --capture-device plughw:Camera,0 \
  --sample-rate 48000 \
  --channels 2 \
  --transcription-provider none \
  --tts-provider iphone \
  --iphone-tts-backend fluid-kokoro-ane
```

The current button source is micro:bit serial. Flash
`scripts/microbit/microbit_a_button_serial.py` to the micro:bit in the Nezha
Pro board; it writes `A:down` when button A is held and `A:up` when released.
The default voice path sends the recorded WAV to the iPhone as raw Gemma audio
input, not through speech-to-text. The terminal display is updated full-screen
while text deltas arrive, including audio bytes, audio duration, token estimate,
and token/sec, so the screen shows the response as it is generated. The current
Pi audio setup uses the USB webcam
microphone (`plughw:Camera,0`) at 48 kHz stereo because the camera rejects mono
ALSA capture requests, and HDMI output (`plughw:vc4hdmi,0`).

iPhone TTS is streamed as raw 24 kHz mono `S16_LE` PCM from the Pi bridge
`/tts-stream` endpoint and piped directly into `aplay`; it is not base64 and no
WAV file is written on the Pi for that path.

For Google-hosted Gemma, keep `GEMMA_AGENT_PROVIDER=google` or omit it and set
`GEMINI_API_KEY`. The default API is `streamGenerateContent` with SSE enabled.

Tools are declared with a JSON file and executed as local commands. The tool
receives `{"name": "...", "args": {...}}` on stdin and its stdout is returned to
the model as the function response:

```json
{
  "tools": [
    {
      "name": "example_status",
      "description": "Return a small robot status object.",
      "parameters": {
        "type": "object",
        "properties": {}
      },
      "command": ["python3", "scripts/tools/example_status.py"]
    }
  ]
}
```
