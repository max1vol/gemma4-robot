# Gemma4 Omni Fitness

This directory contains the Gemma4 omni audio-output research for the fitness
coach. The goal is a small model path that can speak short coaching responses
from Gemma conditioning, rather than only producing text and handing that text
to a separate deployed voice service.

The target runtime shape is:

```text
fitness state or spoken user request
-> Gemma 4 backbone / hidden-state conditioning
-> trained audio-output head
-> speech codec tokens
-> codec decoder
-> waveform for the robot speaker
```

The current robot still uses the iPhone/Pi bridge for deployed speech. The
trained omni model here is a research prototype for future lower-latency,
style-controllable coaching audio.

## Current Training Status

The useful runs were trained on Modal:

- The larger audio-conditioned style-head experiments used H100 GPUs.
- Smaller projection-head, codec reconstruction, packaging, and negative
  baseline checks used L4 GPUs when that was sufficient.

The strongest narrow proof is the audio-conditioned style-head run:

```text
script: modal_gemma_audio_conditioned_style_head_smoke.py
GPU: H100 on Modal
examples: 8
steps: 200
conditioning: Gemma 4 native audio frontend plus target text
explicit style labels in conditioning: false
result: generated the expected short fitness-coach phrases in the checked run
```

That run demonstrates that Gemma-conditioned audio heads can carry short
fitness-coaching phrases and vocal properties in a tightly bounded setup. It is
not yet a broad speech model and should not be presented as general-purpose
voice generation.

Other important evidence:

- SNAC projection-head smokes proved the end-to-end codec-token path and fast
  first-audio scheduling, but the simple multi-codebook head did not generalize
  reliably.
- XCodec2 reconstruction passed transcript-level checks on a small validation
  set, making it a better codec target for future work.
- A random Gemma-conditioned XCodec2 token head still failed held-out audio
  checks, so the next serious model should use a stronger pretrained
  speech-token prior or adapter rather than another small random GRU head.

See `AUDIO_HEAD_LEARNINGS.md` for the current stop rules and next architecture
direction.

## Dataset Browser

`dataset-browser/` is a local SvelteKit UI for reviewing the voice-example
manifests before any more training. It reads summaries from ignored `out/`
directories and streams local WAV files through a safe `/audio` route that only
serves files under the repo `out/` tree.

The UI supports:

- split, style, voice, loudness, status, and text filters,
- matched vs needs-review status,
- input/target pair inspection,
- transcript, duration, RMS, peak, model/source metadata, and path display,
- in-browser playback for manual listening.

Screenshot: `dataset-browser/docs/dataset-browser-screenshot.png`.

Run it from this repo with:

```sh
cd gemma4-omni-fitness/dataset-browser
npm install
GEMMA4_OMNI_REPO_ROOT=/Users/yaroslavvolovich/projects/gemma4-robot npm run dev
```

The browser expects these local manifests when present:

```text
out/gemma4_omni_train_200/summary.json
out/gemma4_omni_validation_40/summary.json
```

The audio data and trained artifacts stay out of git under `out/`.

## Files

Core runtime and training code:

- `fitness_omni_audio_runtime.py`: shared audio-head/runtime helpers.
- `modal_gemma_audio_conditioned_style_head_smoke.py`: H100 audio-conditioned
  style-head training smoke.
- `modal_gemma_audio_conditioned_sequence_head_smoke.py`: text/audio
  conditioned SNAC sequence-head experiments.
- `modal_gemma_xcodec2_sequence_head_smoke.py`: Gemma-conditioned XCodec2
  negative baseline.
- `modal_llasa_xcodec2_prior_smoke.py`: pretrained speech-token prior probe.
- `modal_llasa_lora_audio_head_smoke.py`: guarded LoRA path; do not run until
  the dataset gate passes.
- `modal_xcodec2_codec_smoke.py`: XCodec2 reconstruction gate.
- `tts_dataset_quality_gate.py`: manifest review and quarantine gate.

Supporting scripts:

- `audio_phrase_runtime.py`: phrase-bank playback helpers.
- `voice_control_probe.py`: local control-surface probe for voice/style/speed.
- `gemini_tts_style_dataset.py` and `openai_tts_style_dataset.py`: historical
  dataset-generation utilities kept for auditability and small controlled
  experiments.
- `gemini_verify_style_outputs.py`: style/speed/loudness verification helper.
- `modal_kittentts_audio_milestone.py`: small audio milestone probe.

## Training Gate

Do not launch more large GPU training from these scripts until the dataset gate
passes. The next dataset must be reviewed in the browser, checked for transcript
quality, and explicitly approved before running Modal training.

Minimum gate:

```text
1. Transcript check matches the expected phrase.
2. Manual listening passes for every style bucket.
3. Duration and loudness are plausible for the phrase.
4. Bad samples are kept in a quarantine manifest.
5. Only approved rows are used for training.
```

The next model step should keep Gemma as the conditioning backbone and train a
compact adapter into a stronger speech-token prior, then decode through XCodec2
or another low-rate codec. More steps on the same small random codec-token head
are not the best use of GPU time.
