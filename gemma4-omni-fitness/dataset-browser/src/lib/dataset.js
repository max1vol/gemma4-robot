import fs from 'node:fs';
import path from 'node:path';

const repoRoot = path.resolve(process.env.GEMMA4_OMNI_REPO_ROOT || path.join(process.cwd(), '../..'));

const datasetSummaries = [
  {
    split: 'train',
    label: 'Train',
    summaryPath: 'out/gemma4_omni_train_200/summary.json'
  },
  {
    split: 'validation',
    label: 'Validation',
    summaryPath: 'out/gemma4_omni_validation_40/summary.json'
  }
];

export function resolveRepoPath(relativePath) {
  if (!relativePath || typeof relativePath !== 'string') return null;
  const absolute = path.resolve(repoRoot, relativePath);
  const allowedRoot = path.resolve(repoRoot, 'out');
  if (!absolute.startsWith(allowedRoot + path.sep)) return null;
  return absolute;
}

function readRows(summaryPath) {
  const absoluteSummaryPath = path.resolve(repoRoot, summaryPath);
  const summary = JSON.parse(fs.readFileSync(absoluteSummaryPath, 'utf8'));
  if (Array.isArray(summary.pairs)) return summary.pairs;

  const rowsPath = path.join(path.dirname(absoluteSummaryPath), 'rows.jsonl');
  return fs
    .readFileSync(rowsPath, 'utf8')
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function audioUrl(audio) {
  if (!audio?.wav_path) return null;
  return `/audio?path=${encodeURIComponent(audio.wav_path)}`;
}

function transcriptMatch(verification, key) {
  return verification?.[key]?.matches_expected === true;
}

function rowView(row, index) {
  const expected = row.labels?.expected || {};
  const target = row.audio?.target || null;
  const input = row.audio?.input || null;
  const targetWhisper = row.verification?.target_whisper || null;
  const inputWhisper = row.verification?.input_whisper || null;

  return {
    index,
    id: row.id,
    split: row.split,
    exercise: row.exercise,
    event: row.event,
    trigger: row.home_demo_trigger || '',
    text: row.labels?.target_text || expected.transcript || '',
    inputText: row.labels?.input_text || '',
    style: expected.style || 'unknown',
    speed: expected.speed || 'unknown',
    loudness: expected.loudness || 'unknown',
    voiceRole: expected.voice || 'unknown',
    targetVoice: row.labels?.target_voice || target?.voice || 'unknown',
    inputVoice: row.labels?.input_voice || input?.voice || null,
    targetDirection: row.labels?.target_direction || target?.instructions || '',
    inputDirection: row.labels?.input_direction || input?.instructions || '',
    targetAudio: target
      ? {
          url: audioUrl(target),
          duration: target.duration_s,
          rms: target.rms_dbfs,
          peak: target.peak_dbfs,
          sampleRate: target.sample_rate,
          model: target.source_model,
          path: target.wav_path
        }
      : null,
    inputAudio: input
      ? {
          url: audioUrl(input),
          duration: input.duration_s,
          rms: input.rms_dbfs,
          peak: input.peak_dbfs,
          sampleRate: input.sample_rate,
          model: input.source_model,
          path: input.wav_path
        }
      : null,
    targetWhisper: targetWhisper
      ? {
          expected: targetWhisper.expected,
          heard: targetWhisper.heard,
          matches: transcriptMatch(row.verification, 'target_whisper')
        }
      : null,
    inputWhisper: inputWhisper
      ? {
          expected: inputWhisper.expected,
          heard: inputWhisper.heard,
          matches: transcriptMatch(row.verification, 'input_whisper')
        }
      : null,
    targetMatched: transcriptMatch(row.verification, 'target_whisper'),
    inputMatched: input ? transcriptMatch(row.verification, 'input_whisper') : null
  };
}

function optionCounts(rows, key) {
  return [...rows.reduce((map, row) => map.set(row[key], (map.get(row[key]) || 0) + 1), new Map())]
    .filter(([value]) => value)
    .map(([value, count]) => ({ value, count }))
    .sort((a, b) => String(a.value).localeCompare(String(b.value)));
}

export function loadDataset() {
  const rows = datasetSummaries.flatMap((source) =>
    readRows(source.summaryPath).map((row, index) => rowView({ ...row, split: source.split }, index))
  );

  const stats = {
    total: rows.length,
    train: rows.filter((row) => row.split === 'train').length,
    validation: rows.filter((row) => row.split === 'validation').length,
    targetMatches: rows.filter((row) => row.targetMatched).length,
    inputPairs: rows.filter((row) => row.inputAudio).length,
    inputMatches: rows.filter((row) => row.inputMatched).length,
    targetMinutes: rows.reduce((sum, row) => sum + (row.targetAudio?.duration || 0), 0) / 60,
    inputMinutes: rows.reduce((sum, row) => sum + (row.inputAudio?.duration || 0), 0) / 60
  };

  return {
    repoRoot,
    rows,
    stats,
    options: {
      styles: optionCounts(rows, 'style'),
      voices: optionCounts(rows, 'targetVoice'),
      loudness: optionCounts(rows, 'loudness'),
      speeds: optionCounts(rows, 'speed')
    }
  };
}
