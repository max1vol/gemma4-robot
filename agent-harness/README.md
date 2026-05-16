# Gemma Agent Harness

This is the Rust LLM harness for the robot. It keeps the agent loop separate
from the model transport so the same loop can use Google-hosted Gemma now and an
iPhone-hosted Gemma worker later.

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

Send multimodal parts:

```sh
bin/gemma-agent-harness prompt "Describe this image." --image ./photo.jpg
bin/gemma-agent-harness prompt "Answer the spoken request." --audio ./request.wav
```

Use a future Pi-side iPhone worker bridge:

```sh
bin/gemma-agent-harness --provider ios-bridge --ios-bridge-url http://127.0.0.1:8765 prompt "Hello from the Pi."
```

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
