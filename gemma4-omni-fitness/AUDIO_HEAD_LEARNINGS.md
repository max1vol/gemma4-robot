# Audio Head Learnings

Date: 2026-05-17

## Stop Decision

Stop all further large audio-head training on the current experimental dataset.

Manual browsing found samples that are not good enough for serious supervision.
The existing data is useful as smoke-test evidence only. It should not be used
for another serious GPU run until it has passed the review gate.

Modal state was checked after this decision:

```text
uvx modal app list
```

All `gemma4-omni-*` apps were already `stopped` with `0` tasks. A local process
scan found no active Modal training process. No GPU work is intentionally left
running.

## Code Preserved

These files are kept for reuse, but should not be used for more training until
the dataset is rebuilt or curated:

```text
modal_llasa_xcodec2_prior_smoke.py
  Runs a pretrained LLaSA 1B -> XCodec2 speech-token prior on validation
  phrases, decodes generated codec tokens to WAV, and checks the WAV.

modal_llasa_lora_audio_head_smoke.py
  Draft LoRA fine-tuning path for LLaSA 1B as the speech-token prior. This is
  guarded and should remain off until the dataset gate passes.
```

Previously written XCodec2 scripts remain useful:

```text
modal_xcodec2_codec_smoke.py
  XCodec2 codec-only reconstruction gate.

modal_gemma_xcodec2_sequence_head_smoke.py
  Gemma-conditioned random GRU -> XCodec2 token head. Useful as a negative
  baseline, not as the next architecture.
```

## Evidence So Far

XCodec2 codec reconstruction is good enough for small transcript-level checks:

```text
GPU: L4 on Modal
validation reconstructions: 8/8 matched in the checked run
estimated cost: about $0.009
```

The random Gemma-conditioned XCodec2 GRU head is not good enough:

```text
GPU: L4 on Modal
train examples: 24
validation examples: 8
loss: 11.0954 -> 0.1721
teacher-forced token accuracy: 0.9912
actual held-out matches: 0/8 validation
```

The strongest narrow audio-conditioned style-head run used H100 on Modal:

```text
script: modal_gemma_audio_conditioned_style_head_smoke.py
GPU: H100
examples: 8
steps: 200
explicit style labels in conditioning: false
result: generated the expected short fitness-coach phrases in the checked run
```

Interpretation:

```text
Gemma-conditioned audio-output plumbing works in a small controlled setup.
XCodec2 is a viable codec target.
The simple random token head does not generalize well enough.
The current dataset must be curated before more training.
```

## Required Next Step

Before any more training:

```text
1. Build or curate a cleaner dataset.
2. Review it in the dataset browser.
3. Add a dataset review gate before upload/training.
4. Keep bad samples in a quarantine manifest rather than deleting evidence.
5. Re-run only small L4 smoke checks after curation.
```

Minimum dataset gate:

```text
Transcript must match expected text.
Manual sample review must pass for each style bucket.
Duration should be plausible for phrase length.
RMS/peak should be non-degenerate.
The browser/review UI should expose target text, requested style labels,
transcript, duration, RMS, peak, and the WAV.
```

Only after that gate should LoRA fine-tuning of a speech-token prior be run.

## Architecture Direction After Dataset Fix

Do not train another random codec-token GRU from scratch.

The next useful architecture is:

```text
Gemma 4 fitness/audio state
-> compact speech instruction / hidden-state adapter
-> pretrained speech-token prior
-> low-rate codec tokens
-> codec decoder
-> waveform
-> audio/transcript verification
```

This still moves toward the requested Gemma4 omni model, but uses a stronger
speech-token prior instead of making a tiny random head learn speech generation
from a small dataset.
