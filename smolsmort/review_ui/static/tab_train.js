// train: load tiles and weights, run and watch a training job, save a checkpoint, sweep a recording

smolsmortTabs.register({id: 'train', label: 'train', mount: mountTrain});

// one row per number: label left, value and -/+ pinned right (the stepper-rows recipe)
const STEPPERS = [
  {key: 'epochs', label: 'epochs', step: 50, min: 10, max: 2000, value: 200},
  {key: 'batches', label: 'batches', step: 2, min: 1, max: 64, value: 8},
  {key: 'windows', label: 'windows', step: 32, min: 64, max: 1024, value: 256},
];

function stepperRow(row) {
  return `<div class="run-controls stepper-row" data-stepper="${row.key}">
    <span class="field-label">${row.label}</span>
    <span class="spacer"></span>
    <span class="field-value" id="train-${row.key}">${row.value}</span>
    <div class="toggle" data-step="-${row.step}">-</div>
    <div class="toggle" data-step="${row.step}">+</div>
  </div>`;
}

function mountTrain(panel) {
  panel.innerHTML = `
    <div class="train-pane">
      <div class="train-loaders">
        <div class="toggle dropdown-head grow" id="train-open"><span>tiles</span></div>
        <div class="toggle dropdown-head grow" id="train-load"><span>weights</span></div>
      </div>
      <div id="train-summary" class="train-facts"></div>
      <div class="h-divider"></div>

      <div class="field-label">run</div>
      <div id="train-steppers">${STEPPERS.map(stepperRow).join('')}</div>
      <span class="stat" id="train-crop-note"></span>
      <div id="train-buttons">
        <div class="toggle" id="train-config">config</div>
        <div class="toggle" id="train-start">train</div>
        <div class="toggle removes disabled" id="train-abort">abort</div>
        <div class="toggle adds disabled" id="train-save">save weights</div>
      </div>
      <div class="progress-row">
        <div class="bar"><div class="bar-fill" id="train-bar-fill"></div></div>
        <span class="stat" id="train-progress-text">idle</span>
      </div>
      <div class="strip" id="train-losses">
        <div class="cell"><span class="field-label">train</span><span class="field-value" id="loss-train">-</span></div>
        <div class="cell"><span class="field-label">val</span><span class="field-value" id="loss-val">-</span></div>
        <div class="cell"><span class="field-label">test</span><span class="field-value" id="loss-test">-</span></div>
      </div>
      <canvas id="train-loss" height="90"></canvas>
      <div class="h-divider"></div>

      <div class="field-label">separation</div>
      <div id="train-accuracy" class="train-facts"></div>
      <div class="field-label">val split</div>
      <div id="train-holdout" class="train-facts"></div>
      <div class="h-divider"></div>

      <div class="field-label">sweep</div>
      <div class="run-controls sweep-row">
        <div class="toggle dropdown-head grow" id="sweep-open"><span>recordings</span></div>
        <div class="toggle dropdown-head narrow" id="sweep-percent" data-value="100"><span>100%</span></div>
        <div class="toggle" id="sweep-start">sweep</div>
      </div>
      <div class="crop-row">
        <span class="field-label">min score</span>
        <input type="range" class="range-slider" id="sweep-score" min="0" max="100" value="50">
        <span class="field-value" id="sweep-score-value">0.50</span>
      </div>
      <div class="progress-row">
        <div class="bar"><div class="bar-fill" id="sweep-bar-fill"></div></div>
        <span class="stat" id="sweep-progress-text">idle</span>
      </div>
      <div class="toggle adds disabled" id="sweep-send">send above threshold to select</div>
    </div>`;
  const tab = new TrainTab();
  return {enter: () => tab.enter()};
}

const baseName = path => path.replace(/\/$/, '').split('/').pop();

// adapts dir-tree to what dirMenu wants; a leaf entry (no children, or marked a file) is pickable
function treeFetcher(root) {
  return async path => {
    const query = `root=${root}` + (path ? `&under=${encodeURIComponent(path)}` : '');
    const data = await api(`/api/dir-tree?${query}`);
    const isFile = entry => entry.is_file === true || entry.has_children === false;
    return {
      path: data.here, parent: data.parent,
      dirs: data.entries.filter(e => !isFile(e)).map(e => e.name),
      files: data.entries.filter(isFile).map(e => e.name),
    };
  };
}

class TrainTab {
  constructor() {
    this.values = Object.fromEntries(STEPPERS.map(row => [row.key, row.value]));
    this.floor = 0;
    this.downscale = 2;
    this.backend = 'heatmap';
    this.poll = null;
    this.sweepAbove = null;
    this.sweepRecording = null;
    this.sweepTimer = null;
    this.wire();
  }

  async enter() {
    await this.loadInfo();
    await this.loadFloor();
    await this.refreshStatus();
  }

  say(text) { setText('train-progress-text', text); }

  // ---- facts ----
  facts(id, pairs) {
    const el = document.getElementById(id);
    el.innerHTML = '';
    pairs.forEach(([k, v]) => {
      const key = document.createElement('span'); key.className = 'k'; key.textContent = k;
      const val = document.createElement('span'); val.className = 'v'; val.textContent = v;
      el.append(key, val);
    });
  }

  async loadInfo() {
    const info = await api('/api/train-info');
    if (info.backend) this.backend = info.backend;
    if (info.error) { this.facts('train-summary', [['problem', info.error]]); return; }
    this.facts('train-summary', [
      ['training on', info.set
        ? `${info.frames} frames - ${info.boxes} boxes, ${info.negatives} negatives`
        : 'nothing yet - load tiles or weights'],
      ['classes', info.thinnest
        ? `${info.classes}, thinnest is ${info.thinnest.label} with ${info.thinnest.count}`
        : (info.set ? 'none labelled' : '-')],
      ['device', info.device],
      ['model', info.model_exists ? 'trained, ready to sweep' : 'none yet - train or load weights'],
    ]);
  }

  async loadFloor() {
    const data = await api('/api/window-floor');
    if (data.error) return;
    this.floor = data.floor;
    this.downscale = data.downscale || this.downscale;
    if (data.default) this.values.windows = Math.max(this.values.windows, data.default);
    this.paintSteppers();
  }

  // ---- steppers ----
  paintSteppers() {
    STEPPERS.forEach(row => setText(`train-${row.key}`, String(this.values[row.key])));
    const note = document.getElementById('train-crop-note');
    const windows = this.values.windows;
    if (this.floor && windows < this.floor) {
      note.textContent = `too small - windows need ${this.floor}+`;
      note.classList.add('warn');
    } else {
      note.textContent = `${windows * this.downscale}px of capture`;
      note.classList.remove('warn');
    }
  }

  // ---- pickers ----
  wirePickers() {
    document.getElementById('train-open').onclick = evt => {
      dirMenu('tiles', treeFetcher('tiles'), async path => {
        const res = await api('/api/train-bind', {path, name: baseName(path)});
        if (res.error) { this.say(res.error); return; }
        document.getElementById('train-open').innerHTML = `<span>${baseName(path)}</span>`;
        await this.loadInfo();
        await this.loadFloor();
      }).openAt(evt.currentTarget);
    };
    document.getElementById('train-load').onclick = evt => {
      dirMenu('weights', treeFetcher('checkpoints'), async path => {
        const res = await api('/api/load-checkpoint', {path, name: baseName(path)});
        if (res.error) { this.say(res.error); return; }
        document.getElementById('train-load').innerHTML = `<span>${baseName(path)}</span>`;
        await this.loadInfo();
        this.say(`loaded ${res.name || baseName(path)}`);
      }).openAt(evt.currentTarget);
    };
  }

  // newest first as the server sends them, except a finished run's final entry leads
  async checkpointItems() {
    const data = await api('/api/saved-checkpoints');
    const rows = data.weights || data.checkpoints || [];
    return [...rows.filter(w => w.final), ...rows.filter(w => !w.final)].map(w => ({
      id: w.name, label: w.name, stats: w.final ? 'final' : (w.kb ? `${w.kb} kB` : ''),
    }));
  }

  // opens while a run goes on: it only reads the list and never touches the job
  async openSaveMenu() {
    const head = document.getElementById('train-save');
    if (head.classList.contains('disabled')) return;
    const items = await this.checkpointItems();
    listMenu('save weights', items, async item => {
      const res = await api('/api/save-checkpoint', {name: item.id});
      this.say(res.error ? res.error : `saved ${res.name || item.id}`);
    }, {empty: 'no checkpoints yet'}).openAt(head);
  }

  // ---- run ----
  async start() {
    if (this.floor && this.values.windows < this.floor) {
      this.say(`windows ${this.values.windows} is below the ${this.floor} floor`);
      return;
    }
    const hyper = window.hyperparams ? window.hyperparams.state() : {};
    const res = await api('/api/train-start', {
      ...hyper, backend: this.backend, epochs: this.values.epochs,
      batch: this.values.batches, crop: this.values.windows,
    });
    if (res.error) { this.say(res.error); return; }
    this.refreshStatus();
  }

  fmt(value) { return value === null || value === undefined ? '-' : Number(value).toFixed(3); }

  drawLoss(history) {
    const canvas = document.getElementById('train-loss');
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.clientWidth || 400;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (history.length < 2) return;
    // log scale, since a loss falls by orders of magnitude and a linear axis flattens it
    const ys = history.map(v => Math.log10(Math.max(v, 1e-3)));
    const lo = Math.min(...ys), hi = Math.max(...ys), span = (hi - lo) || 1;
    ctx.strokeStyle = '#e8ddc3';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ys.forEach((y, i) => {
      const px = (i / (ys.length - 1)) * (canvas.width - 4) + 2;
      const py = canvas.height - 4 - ((y - lo) / span) * (canvas.height - 8);
      if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py);
    });
    ctx.stroke();
  }

  async refreshStatus() {
    const job = await api('/api/train-status');
    document.getElementById('train-bar-fill').style.width =
      job.epochs ? `${(job.epoch / job.epochs) * 100}%` : '0';
    const loss = this.fmt(job.train_loss);
    if (job.error) this.say(job.error);
    else if (job.running) this.say(`epoch ${job.epoch} / ${job.epochs}  loss ${loss}`);
    else if (job.finished) this.say(`done - ${job.epochs} epochs, final loss ${loss}`);
    else if (job.aborted) this.say(`aborted at epoch ${job.aborted} of ${job.epochs}`);
    else this.say('idle');
    setText('loss-train', this.fmt(job.train_loss));
    setText('loss-val', this.fmt(job.val_loss));
    setText('loss-test', this.fmt(job.test_loss));
    document.getElementById('train-abort').classList.toggle('disabled', !job.running);
    // save weights waits for a first checkpoint, and stays usable while the run continues
    document.getElementById('train-save').classList.toggle('disabled', !((job.checkpoint_count || 0) >= 1));
    this.drawLoss(job.history || []);
    if (job.separation) {
      const s = job.separation;
      this.facts('train-accuracy', [
        ['on a box', s.on_box.toFixed(3)], ['background (99th pct)', s.background.toFixed(3)],
        ['ratio', s.ratio ? `${s.ratio.toFixed(2)}x brighter` : '-'], ['sampled', `${s.samples} boxes`],
      ]);
    } else if (!job.running) {
      this.facts('train-accuracy', [['not measured yet', 'train a model to see this']]);
    }
    if (job.holdout) {
      const h = job.holdout;
      const pct = x => `${Math.round(x * 100)}%`;
      this.facts('train-holdout', [
        ['recall', `${pct(h.recall)} of ${h.with_box} frames with a box`],
        ['false positives', `${pct(h.false_positive_rate)} of ${h.without_box} empty frames`],
        ['spurious boxes', `${h.spurious}`],
      ]);
    } else if (!job.running) {
      this.facts('train-holdout', [['not measured', 'needs a set with a val split']]);
    }
    if (job.running && !this.poll) this.poll = setInterval(() => this.refreshStatus(), 700);
    if (!job.running && this.poll) { clearInterval(this.poll); this.poll = null; this.loadInfo(); }
  }

  // ---- sweep ----
  sweepScore() { return Number(document.getElementById('sweep-score').value) / 100; }

  showSweepCount() {
    setText('sweep-score-value', this.sweepScore().toFixed(2));
    if (!this.sweepAbove) return;
    const n = this.sweepAbove[Math.round(this.sweepScore() * 100)] ?? 0;
    const total = this.sweepAbove[0] ?? 0;
    setText('sweep-progress-text', `${n} of ${total} proposals at or above ${this.sweepScore().toFixed(2)}`);
    document.getElementById('sweep-send').classList.toggle('disabled', !n);
  }

  async pollSweep() {
    const job = await api('/api/sweep-status');
    if (job.total) document.getElementById('sweep-bar-fill').style.width = `${Math.round(job.done / job.total * 100)}%`;
    if (job.running) { setText('sweep-progress-text', job.note || `${job.done} / ${job.total} frames`); return; }
    clearInterval(this.sweepTimer);
    this.sweepTimer = null;
    if (job.error) { setText('sweep-progress-text', job.error); return; }
    if (job.finished) {
      document.getElementById('sweep-bar-fill').style.width = '100%';
      this.sweepAbove = job.above || null;
      document.getElementById('sweep-send').classList.toggle('disabled', !job.found);
      if (job.found && this.sweepAbove) this.showSweepCount();
      else setText('sweep-progress-text', job.found ? `${job.found} proposals found` : 'nothing above the floor');
    }
  }

  wireSweep() {
    document.getElementById('sweep-open').onclick = async () => {
      const head = document.getElementById('sweep-open');
      const data = await api('/api/sweep-recordings');
      listMenu('sweep which recording',
        data.recordings.map(rec => ({
          id: rec.name, label: rec.name, on: rec.name === this.sweepRecording, stats: `${rec.frames} frames`})),
        item => {
          this.sweepRecording = item.id;
          document.getElementById('sweep-open').innerHTML = `<span>${item.id}</span>`;
        }, {empty: 'no recordings with frames'}).openAt(head);
    };
    document.getElementById('sweep-percent').onclick = evt => {
      const el = evt.currentTarget;
      listMenu('how much of it to sweep',
        [10, 25, 50, 100].map(n => ({id: String(n), label: `${n}%`, on: el.dataset.value === String(n)})),
        item => { el.dataset.value = item.id; el.innerHTML = `<span>${item.id}%</span>`; }
      ).openAt(el);
    };
    document.getElementById('sweep-score').oninput = () => this.showSweepCount();
    document.getElementById('sweep-start').onclick = async () => {
      if (!this.sweepRecording) { setText('sweep-progress-text', 'pick a recording first'); return; }
      const res = await api('/api/sweep-start', {
        recording: this.sweepRecording,
        percent: Number(document.getElementById('sweep-percent').dataset.value),
        min_score: this.sweepScore(),
      });
      if (res.error) { setText('sweep-progress-text', res.error); return; }
      document.getElementById('sweep-send').classList.add('disabled');
      clearInterval(this.sweepTimer);
      this.sweepTimer = setInterval(() => this.pollSweep(), 500);
    };
    document.getElementById('sweep-send').onclick = async evt => {
      if (evt.currentTarget.classList.contains('disabled')) return;
      const res = await api('/api/sweep-send', {min_score: this.sweepScore()});
      if (res.error) { setText('sweep-progress-text', res.error); return; }
      setText('sweep-progress-text', `sent ${res.sent} of ${res.of} proposals - ${res.tiles} tiles`);
      activateTab('select');
    };
  }

  wire() {
    this.wirePickers();
    this.wireSweep();
    // one handler for every stepper row, bound by data attributes and not by position
    document.getElementById('train-steppers').onclick = evt => {
      const button = evt.target.closest('[data-step]');
      if (!button) return;
      const key = button.closest('[data-stepper]').dataset.stepper;
      const row = STEPPERS.find(r => r.key === key);
      this.values[key] = Math.max(row.min, Math.min(row.max, this.values[key] + Number(button.dataset.step)));
      this.paintSteppers();
    };
    document.getElementById('train-config').onclick =
      evt => window.hyperparams && window.hyperparams.open(evt.currentTarget, this.backend);
    document.getElementById('train-start').onclick = () => this.start();
    document.getElementById('train-abort').onclick = async () => {
      if (document.getElementById('train-abort').classList.contains('disabled')) return;
      const res = await api('/api/train-abort', {});
      this.say(res.error || 'aborting after this epoch');
    };
    document.getElementById('train-save').onclick = () => this.openSaveMenu();
  }
}
