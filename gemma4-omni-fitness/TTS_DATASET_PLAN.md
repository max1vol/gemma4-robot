# Voice Dataset Curation Plan For Gemma4 Omni Fitness

This file records the next dataset requirements for the Gemma4 omni fitness
audio-head work. The file name is historical; the active plan is dataset
curation and review, not another blind generation run.

## Recommendation

Do not scale training until the current small voice-example set is reviewed and
curated. The dataset browser now exists because automatic transcript checks were
not enough: a sample can pass a text check and still be a bad supervision
example.

The next useful dataset should be small enough to review carefully first, then
large enough to test generalization after the gate is proven.

## Data Shape

Generate or curate short home-fitness utterances only. Avoid broad chatbot
speech.

Each record should include:

```json
{
  "id": "count_000123",
  "split": "train",
  "exercise": "squat",
  "event": "rep_complete",
  "target_text": "Seven. Quiet.",
  "style": "whispered",
  "speed": "normal",
  "loudness": "quiet",
  "voice": "calm_home_coach",
  "target_wav": "target.wav",
  "input_text": "Please keep coaching quietly.",
  "input_wav": "input.wav",
  "approved_for_training": false,
  "review_notes": ""
}
```

## Suggested Dataset Mix

Start with a reviewed seed set, then expand only after the gate works:

```text
60% count phrases:
  "One."
  "Two."
  ...
  "Twenty."

20% count + short coaching:
  "Seven. Nice rhythm."
  "Eight. Strong rep."
  "Nine. Keep going."
  "Ten. Great set."

10% style-control pairs:
  whispered / quiet
  projected / loud
  calm / slow
  upbeat / fast
  warm / reassuring

10% silence/no-speak negatives:
  not audio targets, but labels for speech-action training
```

For audio-head training, prioritize the first three groups. For speech-action
training, include silence/no-speak rows separately.

## Splits

Use split rules that actually test generalization:

```text
train:
  most numbers, styles, voices, and common phrase suffixes

validation_seen_style:
  held-out phrase text but seen style and voice

validation_seen_text:
  seen phrase text but held-out voice/style

validation_unseen_combo:
  unseen number + unseen suffix + held-out style/voice combination

test_home_demo:
  manually selected phrases that match the one-person home demo
```

Do not claim success unless `validation_unseen_combo` passes transcript and
manual listening checks.

## Verification

Use three gates:

```text
Transcript gate:
  required for text correctness

Manual listening gate:
  required for training approval

Audio-style gate:
  useful for style/loudness/speed, but secondary
```

Treat evaluator disagreement as failure if the transcript or manual listening
review indicates the wrong phrase, bad audio, or mismatched style.

For any larger dataset:

```text
100% of validation/test examples transcript-checked
100% of validation/test examples manually spot-checked by style bucket
5-10% of training examples manually reviewed before the first run
all failures written to a quarantine manifest
```

## Dataset Browser

Use the browser before any more training:

```sh
cd gemma4-omni-fitness/dataset-browser
npm install
GEMMA4_OMNI_REPO_ROOT=/Users/yaroslavvolovich/projects/gemma4-robot npm run dev
```

The browser should expose:

```text
target text
input text when present
style/speed/loudness/voice labels
transcript check result
duration
RMS/peak
WAV playback
source path
review status
```

## Training Rule

Only rows with explicit training approval should be consumed by Modal training
scripts. Bad examples should stay in the manifest or quarantine file so that the
failure mode remains auditable.

After the curated seed set passes:

```text
1. Run a tiny L4 codec reconstruction check.
2. Run a tiny L4 audio-head smoke.
3. Only then run a bounded H100 audio-conditioned training job.
4. Re-check generated WAVs in the browser before claiming progress.
```
