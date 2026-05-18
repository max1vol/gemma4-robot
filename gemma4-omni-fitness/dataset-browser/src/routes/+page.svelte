<script>
  import {
    Activity,
    CheckCircle2,
    Database,
    Filter,
    ListMusic,
    Search,
    SlidersHorizontal,
    Volume2,
    XCircle
  } from '@lucide/svelte';

  let { data } = $props();

  let query = $state('');
  let split = $state('all');
  let style = $state('all');
  let voice = $state('all');
  let loudness = $state('all');
  let status = $state('all');
  let selectedId = $state('');

  const fmt = new Intl.NumberFormat('en-US', { maximumFractionDigits: 1 });

  const filteredRows = $derived(
    data.rows.filter((row) => {
      const haystack = [
        row.id,
        row.text,
        row.inputText,
        row.style,
        row.speed,
        row.loudness,
        row.targetVoice,
        row.voiceRole,
        row.trigger
      ]
        .join(' ')
        .toLowerCase();
      const q = query.trim().toLowerCase();
      if (q && !haystack.includes(q)) return false;
      if (split !== 'all' && row.split !== split) return false;
      if (style !== 'all' && row.style !== style) return false;
      if (voice !== 'all' && row.targetVoice !== voice) return false;
      if (loudness !== 'all' && row.loudness !== loudness) return false;
      if (status === 'matched' && !row.targetMatched) return false;
      if (status === 'mismatch' && row.targetMatched) return false;
      if (status === 'paired' && !row.inputAudio) return false;
      return true;
    })
  );

  const selected = $derived(
    data.rows.find((row) => row.id === selectedId) || filteredRows[0] || data.rows[0]
  );

  $effect(() => {
    if (!selectedId && data.rows[0]) selectedId = data.rows[0].id;
  });

  function choose(row) {
    selectedId = row.id;
  }

  function resetFilters() {
    query = '';
    split = 'all';
    style = 'all';
    voice = 'all';
    loudness = 'all';
    status = 'all';
  }

  function bars(seed, count = 42) {
    let state = 0;
    for (const char of seed) state = (state * 31 + char.charCodeAt(0)) >>> 0;
    return Array.from({ length: count }, (_, index) => {
      state = (1664525 * state + 1013904223 + index) >>> 0;
      return 18 + (state % 72);
    });
  }

  function db(value) {
    if (value === null || value === undefined) return 'n/a';
    return `${fmt.format(value)} dB`;
  }

  function duration(value) {
    if (!value) return '0.0s';
    return `${fmt.format(value)}s`;
  }
</script>

<svelte:head>
  <title>Gemma4 Omni Fitness Dataset</title>
</svelte:head>

<main class="shell">
  <header class="topbar">
    <div class="brand">
      <div class="brand-mark"><Database size={21} /></div>
      <div>
        <h1>Gemma4 Omni Fitness Dataset</h1>
      </div>
    </div>
    <div class="summary-strip" aria-label="Dataset summary">
      <div>
        <strong>{data.stats.total}</strong>
        <span>Total</span>
      </div>
      <div>
        <strong>{data.stats.train}</strong>
        <span>Train</span>
      </div>
      <div>
        <strong>{data.stats.validation}</strong>
        <span>Validation</span>
      </div>
      <div>
        <strong>{data.stats.targetMatches}</strong>
        <span>Whisper matched</span>
      </div>
      <div>
        <strong>{fmt.format(data.stats.targetMinutes)}</strong>
        <span>Target min</span>
      </div>
    </div>
  </header>

  <section class="workspace">
    <aside class="filters" aria-label="Dataset filters">
      <div class="filter-title">
        <Filter size={17} />
        <span>Filters</span>
      </div>

      <label class="search-box">
        <span class="search-icon" aria-hidden="true"><Search size={17} /></span>
        <input bind:value={query} placeholder="Search transcript, style, voice" />
      </label>

      <label>
        <span>Split</span>
        <select bind:value={split}>
          <option value="all">All splits</option>
          <option value="train">Train</option>
          <option value="validation">Validation</option>
        </select>
      </label>

      <label>
        <span>Style</span>
        <select bind:value={style}>
          <option value="all">All styles</option>
          {#each data.options.styles as item}
            <option value={item.value}>{item.value} · {item.count}</option>
          {/each}
        </select>
      </label>

      <label>
        <span>Voice</span>
        <select bind:value={voice}>
          <option value="all">All voices</option>
          {#each data.options.voices as item}
            <option value={item.value}>{item.value} · {item.count}</option>
          {/each}
        </select>
      </label>

      <label>
        <span>Loudness</span>
        <select bind:value={loudness}>
          <option value="all">All loudness</option>
          {#each data.options.loudness as item}
            <option value={item.value}>{item.value} · {item.count}</option>
          {/each}
        </select>
      </label>

      <label>
        <span>Status</span>
        <select bind:value={status}>
          <option value="all">All statuses</option>
          <option value="matched">Matched</option>
          <option value="mismatch">Needs review</option>
          <option value="paired">Input + target</option>
        </select>
      </label>

      <button class="reset" type="button" onclick={resetFilters}>
        <SlidersHorizontal size={16} />
        Reset
      </button>

      <div class="mini-stats">
        <div>
          <span>Input pairs</span>
          <strong>{data.stats.inputPairs}</strong>
        </div>
        <div>
          <span>Input matched</span>
          <strong>{data.stats.inputMatches}</strong>
        </div>
        <div>
          <span>Input min</span>
          <strong>{fmt.format(data.stats.inputMinutes)}</strong>
        </div>
      </div>
    </aside>

    <section class="sample-list" aria-label="Dataset rows">
      <div class="list-head">
        <div>
          <span class="eyebrow"><ListMusic size={15} /> Samples</span>
          <strong>{filteredRows.length}</strong>
        </div>
        <div class="legend">
          <span><i class="dot matched"></i>Matched</span>
          <span><i class="dot review"></i>Review</span>
        </div>
      </div>

      <div class="rows">
        {#each filteredRows as row}
          <button
            type="button"
            class:selected={selected?.id === row.id}
            class="row"
            onclick={() => choose(row)}
          >
            <span class="status" class:bad={!row.targetMatched}>
              {#if row.targetMatched}
                <CheckCircle2 size={17} />
              {:else}
                <XCircle size={17} />
              {/if}
            </span>
            <span class="row-main">
              <strong>{row.text}</strong>
              <small>{row.id}</small>
            </span>
            <span class="row-tags">
              <span>{row.split}</span>
              <span>{row.style}</span>
              <span>{row.targetVoice}</span>
            </span>
            <span class="row-metrics">
              <span>{duration(row.targetAudio?.duration)}</span>
              <span>{db(row.targetAudio?.rms)}</span>
            </span>
          </button>
        {/each}
      </div>
    </section>

    {#if selected}
      <aside class="detail" aria-label="Selected sample">
        <div class="detail-head">
          <span class="split-badge">{selected.split}</span>
          <h2>{selected.text}</h2>
          <p>{selected.id}</p>
        </div>

        <div class="waveform" aria-hidden="true">
          {#each bars(selected.id) as height}
            <i style={`height: ${height}%`}></i>
          {/each}
        </div>

        <div class="audio-block">
          <div class="audio-title">
            <span class="audio-icon" aria-hidden="true"><Volume2 size={17} /></span>
            <strong>Target</strong>
            <span>{duration(selected.targetAudio?.duration)}</span>
          </div>
          <audio controls src={selected.targetAudio?.url}></audio>
        </div>

        {#if selected.inputAudio}
          <div class="audio-block input">
            <div class="audio-title">
              <span class="audio-icon" aria-hidden="true"><Activity size={17} /></span>
              <strong>Input</strong>
              <span>{duration(selected.inputAudio.duration)}</span>
            </div>
            <audio controls src={selected.inputAudio.url}></audio>
          </div>
        {/if}

        <dl class="facts">
          <div>
            <dt>Style</dt>
            <dd>{selected.style}</dd>
          </div>
          <div>
            <dt>Speed</dt>
            <dd>{selected.speed}</dd>
          </div>
          <div>
            <dt>Loudness</dt>
            <dd>{selected.loudness}</dd>
          </div>
          <div>
            <dt>Voice</dt>
            <dd>{selected.targetVoice}</dd>
          </div>
          <div>
            <dt>RMS</dt>
            <dd>{db(selected.targetAudio?.rms)}</dd>
          </div>
          <div>
            <dt>Peak</dt>
            <dd>{db(selected.targetAudio?.peak)}</dd>
          </div>
        </dl>

        <section class="verify">
          <h3>Whisper</h3>
          <div class="verify-row">
            <span class:ok={selected.targetWhisper?.matches} class:review={!selected.targetWhisper?.matches}>
              {selected.targetWhisper?.matches ? 'Matched' : 'Review'}
            </span>
            <p>{selected.targetWhisper?.heard || 'No transcript'}</p>
          </div>
          {#if selected.inputWhisper}
            <div class="verify-row input-row">
              <span class:ok={selected.inputWhisper.matches} class:review={!selected.inputWhisper.matches}>
                Input {selected.inputWhisper.matches ? 'matched' : 'review'}
              </span>
              <p>{selected.inputWhisper.heard}</p>
            </div>
          {/if}
        </section>

        <section class="instructions">
          <h3>Target Instructions</h3>
          <p>{selected.targetDirection}</p>
        </section>

        <section class="path">
          <h3>File</h3>
          <p>{selected.targetAudio?.path}</p>
        </section>
      </aside>
    {/if}
  </section>
</main>

<style>
  :global(*) {
    box-sizing: border-box;
  }

  :global(html) {
    color-scheme: light;
    font-family:
      Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #f6f8f7;
    color: #17211f;
  }

  :global(body) {
    margin: 0;
    min-width: 320px;
  }

  button,
  input,
  select {
    font: inherit;
  }

  .shell {
    min-height: 100vh;
    padding: 20px;
  }

  .topbar {
    display: grid;
    grid-template-columns: minmax(260px, 1fr) auto;
    gap: 22px;
    align-items: center;
    max-width: 1680px;
    margin: 0 auto 18px;
  }

  .brand {
    display: flex;
    gap: 14px;
    align-items: center;
  }

  .brand-mark {
    display: grid;
    width: 44px;
    height: 44px;
    place-items: center;
    border: 1px solid #c7d2cf;
    border-radius: 8px;
    background: #ffffff;
    color: #0f766e;
  }

  h1,
  h2,
  h3,
  p,
  dl {
    margin: 0;
  }

  h1 {
    font-size: 24px;
    line-height: 1.1;
    letter-spacing: 0;
  }

  .summary-strip {
    display: grid;
    grid-template-columns: repeat(5, minmax(92px, 1fr));
    gap: 1px;
    overflow: hidden;
    border: 1px solid #cfd9d6;
    border-radius: 8px;
    background: #cfd9d6;
  }

  .summary-strip div {
    min-width: 0;
    padding: 12px 14px;
    background: #ffffff;
  }

  .summary-strip strong {
    display: block;
    font-size: 19px;
    line-height: 1.1;
  }

  .summary-strip span {
    display: block;
    margin-top: 4px;
    color: #65746f;
    font-size: 12px;
    white-space: nowrap;
  }

  .workspace {
    display: grid;
    grid-template-columns: 260px minmax(420px, 1fr) minmax(360px, 430px);
    gap: 18px;
    align-items: start;
    max-width: 1680px;
    margin: 0 auto;
  }

  .filters,
  .sample-list,
  .detail {
    border: 1px solid #cfd9d6;
    border-radius: 8px;
    background: #ffffff;
  }

  .filters {
    position: sticky;
    top: 20px;
    display: grid;
    gap: 14px;
    padding: 16px;
  }

  .filter-title,
  .eyebrow,
  .audio-title {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .filter-title {
    font-weight: 700;
  }

  label {
    display: grid;
    gap: 7px;
  }

  label > span {
    color: #5c6a66;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
  }

  .search-box {
    position: relative;
  }

  .search-icon {
    position: absolute;
    left: 11px;
    top: 50%;
    display: inline-flex;
    align-items: center;
    transform: translateY(-50%);
    color: #60706b;
    pointer-events: none;
  }

  .search-icon :global(svg) {
    display: block;
  }

  input,
  select {
    width: 100%;
    height: 40px;
    border: 1px solid #cfd9d6;
    border-radius: 7px;
    background: #fbfcfc;
    color: #17211f;
  }

  input {
    padding: 0 11px 0 36px;
  }

  select {
    padding: 0 10px;
  }

  .reset {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    height: 38px;
    border: 1px solid #b8c6c2;
    border-radius: 7px;
    background: #f7faf9;
    color: #20302c;
    cursor: pointer;
  }

  .mini-stats {
    display: grid;
    gap: 1px;
    overflow: hidden;
    border: 1px solid #d8e0de;
    border-radius: 7px;
    background: #d8e0de;
  }

  .mini-stats div {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    padding: 10px 11px;
    background: #fbfcfc;
    font-size: 13px;
  }

  .mini-stats span {
    color: #66746f;
  }

  .list-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    padding: 15px 16px;
    border-bottom: 1px solid #e0e6e4;
  }

  .eyebrow {
    color: #5c6a66;
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
  }

  .list-head strong {
    display: block;
    margin-top: 5px;
    font-size: 23px;
  }

  .legend {
    display: flex;
    gap: 14px;
    color: #64726e;
    font-size: 12px;
  }

  .dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    margin-right: 6px;
    border-radius: 99px;
  }

  .dot.matched {
    background: #0f766e;
  }

  .dot.review {
    background: #b45309;
  }

  .rows {
    max-height: calc(100vh - 134px);
    overflow: auto;
  }

  .row {
    display: grid;
    grid-template-columns: 28px minmax(170px, 1.2fr) minmax(180px, 0.9fr) 112px;
    gap: 12px;
    align-items: center;
    width: 100%;
    min-height: 76px;
    padding: 12px 16px;
    border: 0;
    border-bottom: 1px solid #edf1f0;
    background: #ffffff;
    color: inherit;
    text-align: left;
    cursor: pointer;
  }

  .row:hover,
  .row.selected {
    background: #f1f8f7;
  }

  .row.selected {
    box-shadow: inset 3px 0 0 #0f766e;
  }

  .status {
    color: #0f766e;
  }

  .status.bad {
    color: #b45309;
  }

  .row-main,
  .row-tags,
  .row-metrics {
    min-width: 0;
  }

  .row-main strong {
    display: block;
    overflow-wrap: anywhere;
    font-size: 15px;
  }

  .row-main small {
    display: block;
    margin-top: 5px;
    overflow: hidden;
    color: #65746f;
    font-size: 12px;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .row-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }

  .row-tags span {
    padding: 4px 7px;
    border: 1px solid #d7e0dd;
    border-radius: 6px;
    background: #fafcfb;
    color: #495955;
    font-size: 12px;
  }

  .row-metrics {
    display: grid;
    justify-items: end;
    gap: 5px;
    color: #60706b;
    font-size: 12px;
  }

  .detail {
    position: sticky;
    top: 20px;
    display: grid;
    gap: 16px;
    max-height: calc(100vh - 40px);
    overflow: auto;
    padding: 17px;
  }

  .detail-head {
    display: grid;
    gap: 7px;
  }

  .split-badge {
    width: max-content;
    padding: 5px 8px;
    border-radius: 6px;
    background: #fff6e8;
    color: #965b09;
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
  }

  h2 {
    font-size: 25px;
    line-height: 1.12;
    letter-spacing: 0;
    overflow-wrap: anywhere;
  }

  .detail-head p {
    color: #60706b;
    font-size: 12px;
    overflow-wrap: anywhere;
  }

  .waveform {
    display: flex;
    align-items: center;
    gap: 3px;
    height: 92px;
    padding: 13px;
    border: 1px solid #d9e2df;
    border-radius: 8px;
    background: linear-gradient(180deg, #fbfcfc, #f0f6f5);
  }

  .waveform i {
    flex: 1;
    min-width: 2px;
    border-radius: 3px;
    background: #0f766e;
    opacity: 0.72;
  }

  .audio-block {
    display: grid;
    gap: 8px;
  }

  .audio-title {
    justify-content: space-between;
  }

  .audio-icon {
    display: inline-flex;
    color: #0f766e;
  }

  .audio-title strong {
    margin-right: auto;
  }

  .audio-title span {
    color: #65746f;
    font-size: 12px;
  }

  audio {
    width: 100%;
    height: 38px;
  }

  .facts {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1px;
    overflow: hidden;
    border: 1px solid #d8e0de;
    border-radius: 8px;
    background: #d8e0de;
  }

  .facts div {
    min-width: 0;
    padding: 10px;
    background: #fbfcfc;
  }

  dt {
    color: #63716d;
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
  }

  dd {
    margin: 4px 0 0;
    overflow-wrap: anywhere;
    font-size: 14px;
    font-weight: 700;
  }

  .verify,
  .instructions,
  .path {
    display: grid;
    gap: 9px;
  }

  h3 {
    color: #354642;
    font-size: 13px;
    letter-spacing: 0;
  }

  .verify-row {
    display: grid;
    gap: 6px;
    padding: 11px;
    border: 1px solid #dbe4e1;
    border-radius: 8px;
    background: #fbfcfc;
  }

  .verify-row span {
    width: max-content;
    font-size: 12px;
    font-weight: 800;
  }

  .verify-row .ok,
  .verify-row span.ok {
    color: #0f766e;
  }

  .verify-row .review,
  .verify-row span.review {
    color: #b45309;
  }

  .verify-row p,
  .instructions p,
  .path p {
    color: #40514d;
    font-size: 13px;
    line-height: 1.45;
    overflow-wrap: anywhere;
  }

  .path p {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
  }

  @media (max-width: 1180px) {
    .topbar,
    .workspace {
      grid-template-columns: 1fr;
    }

    .filters,
    .detail {
      position: static;
      max-height: none;
    }

    .summary-strip {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .rows {
      max-height: none;
    }
  }

  @media (max-width: 760px) {
    .shell {
      padding: 12px;
    }

    .row {
      grid-template-columns: 24px 1fr;
    }

    .row-tags,
    .row-metrics {
      grid-column: 2;
      justify-items: start;
    }

    .facts {
      grid-template-columns: 1fr;
    }
  }
</style>
