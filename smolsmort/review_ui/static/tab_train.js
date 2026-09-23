// train: load tiles and weights, run and watch a training job, save a checkpoint, sweep a recording

smolsmortTabs.register({id: 'train', label: 'train', mount: mountTrain});

// one row per number: label left, value and -/+ pinned right (the stepper-rows recipe)
const STEPPERS = [
  {key: 'epochs', label: 'epochs', step: 50, min: 10, max: 2000, value: 200},
  {key: 'batches', label: 'batches', step: 2, min: 1, max: 64, value: 8},
  {key: 'windows', label: 'windows', step: 32, min: 64, max: 1024, value: 256},
];

// the factors worth offering out of the server's 1..8 range - odd ones buy nothing here
const DOWNSCALES = [1, 2, 3, 4, 6, 8];

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
      <div class="run-controls" id="train-backend-row">
        <span class="field-label">backend</span>
        <div class="toggle dropdown-head grow disabled" id="train-backend"
             title="bind a set of tiles first"><span>-</span></div>
      </div>
      <div class="run-controls" id="train-capture-row">
        <span class="field-label">input</span>
        <div class="toggle dropdown-head grow disabled" id="train-capture"
             title="bind a set of tiles first"><span>-</span></div>
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
      <span class="stat warn hidden" id="sweep-warning"></span>
      <div class="toggle adds disabled" id="sweep-send">send above threshold to select</div>
    </div>`;
  const tab = new TrainTab();
  return {enter: () => tab.enter()};
}

const baseName = path => path.replace(/\/$/, '').split('/').pop();

// a dropdown head shows one line of text. as text: a set, recording or checkpoint name comes
// from a folder on disk, and a name is never markup
function setHead(el, text) {
  const span = document.createElement('span');
  span.textContent = text;
  el.replaceChildren(span);
}

class TrainTab {
  constructor() {
    this.values = Object.fromEntries(STEPPERS.map(row => [row.key, row.value]));
    this.floor = 0;
    this.downscale = 2;
    this.backend = 'heatmap';
    this.sizeMode = null;
    this.capture = {width: null, override: null, observed: null, downscale: 2, input: null};
    this.poll = null;
    this.sweepAbove = null;
    this.sweepRecording = null;
    this.sweepTimer = null;
    this.roots = {};
    this.hasWeights = false;
    this.expectRun = false;
    this.wire();
  }

  async enter() {
    await this.loadInfo();
    await this.loadFloor();
    await this.refreshStatus();
  }

  // save weights waits for weights to exist, and stays usable while a run continues
  paintSave() { document.getElementById('train-save').classList.toggle('disabled', !this.hasWeights); }

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
    this.setName = info.set || null;
    window.smolsmortActiveSet = this.setName;
    this.sizeMode = info.size_mode || null;
    this.capture = {
      width: info.capture_width ?? null, override: info.capture_override ?? null,
      observed: info.observed_width ?? null, downscale: info.downscale || this.capture.downscale,
      input: info.input_width ?? null,
    };
    this.paintBackend(info);
    this.paintCapture(info);
    await this.wireBackendFlavours();
    this.hasWeights = !!info.model_exists;
    this.paintSave();
    if (info.error) { this.facts('train-summary', [['problem', info.error]]); return; }
    this.facts('train-summary', [
      ['training on', info.set
        ? `${info.frames} frame${info.frames === 1 ? '' : 's'} - ${info.objects} objects, ${info.negatives} negatives`
        : 'nothing yet - load tiles or weights'],
      ['classes', info.thinnest
        ? `${info.classes}, thinnest is ${info.thinnest.label} with ${info.thinnest.count}`
        : (info.set ? 'none labelled' : '-')],
      ...(info.set ? [['backend', `${info.backend} - sizes ${info.size_mode}`]] : []),
      ...this.captureFacts(info),
      ['device', info.device],
      ['model', info.model_exists ? 'trained, ready to sweep' : 'none yet - train or load weights'],
    ]);
  }

  // the backend head: disabled with a hint until a set is bound
  paintBackend(info) {
    const head = document.getElementById('train-backend');
    head.classList.toggle('disabled', !info.set);
    head.title = info.set ? 'the backend this set trains with' : 'bind a set of tiles first';
    setHead(head, info.set ? info.backend : '-');
  }

  isBox(info) { return (info.backend || this.backend) === 'box'; }

  // what the net actually sees: capture width over downscale. a box model works at its own long
  // side instead, so the head reports that and the picker stays shut
  paintCapture(info) {
    const head = document.getElementById('train-capture');
    const box = this.isBox(info);
    head.classList.toggle('disabled', !info.set || box);
    if (box) {
      head.title = 'the box backend works at its own long side';
      setHead(head, info.working_size ? `${info.working_size}px long side` : '-');
      return;
    }
    head.title = info.set ? 'capture width and downscale for this set' : 'bind a set of tiles first';
    setHead(head, info.set && info.input_width ? this.inputLabel(info) : '-');
  }

  inputLabel(info) { return `${info.capture_width} / ${info.downscale} = ${info.input_width}px`; }

  captureFacts(info) {
    if (!info.set) return [];
    if (this.isBox(info)) {
      return info.working_size ? [['working size', `${info.working_size}px long side`]] : [];
    }
    const source = info.capture_override ? 'set here' : 'from the frames';
    return [
      ['input', info.input_width
        ? `${this.inputLabel(info)} (capture ${source})`
        : 'unknown - no frames read yet'],
      ...(info.resampled_from
        ? [['resampled', `frames are ${info.resampled_from}px, `
            + `resampled to the weights' ${info.weights_capture_width}px`]]
        : []),
    ];
  }

  // capture width and downscale live on the set, so applying writes them through the same route
  // the backend picker uses. an empty width clears the override and follows the frames again
  openCapturePicker(head) {
    if (!this.setName) { this.say('bind a set of tiles first'); return; }
    if (head.classList.contains('disabled')) return;
    const pending = {
      width: this.capture.override == null ? '' : String(this.capture.override),
      downscale: this.capture.downscale,
    };
    const effective = () => {
      const width = Number(pending.width) || this.capture.observed;
      return width ? `${Math.floor(width / pending.downscale)}px` : 'unknown';
    };
    // the width field is not redrawn as it is typed - a refresh would take the caret with it -
    // so the input size below follows the last downscale pick, not the half-typed number
    const sections = () => [
      {kind: 'field', label: 'capture width', value: pending.width,
       placeholder: this.capture.observed ? `${this.capture.observed} - what the frames are` : 'px',
       onInput: value => { pending.width = value; }},
      {kind: 'list', label: `downscale - input ${effective()}`,
       items: DOWNSCALES.map(n => ({id: String(n), label: `/ ${n}`, on: pending.downscale === n})),
       onPick: item => { pending.downscale = Number(item.id); menu.refresh(sections()); }},
      {kind: 'buttons', buttons: [
        {id: 'apply', label: 'apply', tone: 'adds', onClick: m => this.applyCapture(pending, m)}]},
    ];
    const menu = new Menu({title: 'capture size', persistent: true, sections: sections()});
    menu.openAt(head);
  }

  async applyCapture(pending, menu) {
    const text = String(pending.width).trim();
    const width = text === '' ? null : Number(text);
    if (width !== null && !Number.isInteger(width)) {
      this.say('capture width must be a whole number of pixels');
      return;
    }
    const res = await api('/api/train-set-backend', {
      name: this.setName, backend: this.backend, size_mode: this.sizeMode,
      capture_width: width, downscale: pending.downscale,
    });
    if (res.error) { this.say(res.error); return; }
    menu.close();
    await this.loadInfo();
    await this.loadFloor();
    this.say(this.capture.input ? `input ${this.capture.input}px` : 'capture size saved');
  }

  // the server names its backends in the error body of an unknown-backend request; nothing is written
  async backendNames() {
    if (this.backendList) return this.backendList;
    const res = await api('/api/train-set-backend', {name: '_probe', backend: ''});
    this.backendList = res.backends || [];
    return this.backendList;
  }

  // with topics and the first-row dropdown on the page, that dropdown picks the backend and this
  // row hides; an older core.js or a host page without the dropdown keeps this tab's own picker
  async wireBackendFlavours() {
    const topics = typeof smolsmortTabs.setFlavours === 'function';
    if (!topics || !document.getElementById('topic-flavour')) return;
    let names = [];
    try { names = await this.backendNames(); } catch (err) { return; }
    const items = names.filter(name => name !== 'xgboost').map(name => ({id: name, label: name}));
    smolsmortTabs.setFlavours('vision', items, this.backend);
    document.getElementById('train-backend-row').classList.add('hidden');
    if (this.flavourWired) return;
    this.flavourWired = true;
    // a pick that cannot apply snaps the dropdown back to the backend actually in use
    const snapBack = () => smolsmortTabs.setFlavours('vision', items, this.backend);
    smolsmortTabs.onFlavour('vision', async id => {
      if (id === this.backend) return;
      if (!this.setName) { this.say('bind a set of tiles first'); snapBack(); return; }
      const res = await api('/api/train-set-backend', {name: this.setName, backend: id});
      if (res.error) { this.say(res.error); snapBack(); return; }
      this.say(`${res.backend} - sizes ${res.size_mode}`);
      await this.loadInfo();
    });
  }

  async openBackendPicker(head) {
    if (!this.setName) { this.say('bind a set of tiles first'); return; }
    let names = [];
    try { names = await this.backendNames(); } catch (err) { this.say(err.message); return; }
    listMenu('backend', names.map(name => ({id: name, label: name, on: name === this.backend})),
      async item => {
        const res = await api('/api/train-set-backend', {name: this.setName, backend: item.id});
        if (res.error) { this.say(res.error); return; }
        this.say(`${res.backend} - sizes ${res.size_mode}`);
        await this.loadInfo();
      }).openAt(head);
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
    // two directory browsers, each rooted at its own base. a pick is sent as a name relative to
    // that root, which is what train-bind and load-checkpoint resolve
    document.getElementById('train-backend').onclick = evt => this.openBackendPicker(evt.currentTarget);
    document.getElementById('train-capture').onclick = evt => this.openCapturePicker(evt.currentTarget);
    document.getElementById('train-open').onclick = evt => {
      const head = evt.currentTarget;
      dirMenu('tiles', this.treeFetcher('tiles', () => this.setNames()), async path => {
        const name = this.relative('tiles', path);
        const res = await api('/api/train-bind', {name});
        if (res.error) { this.say(res.error); return; }
        setHead(head, baseName(name));
        await this.loadInfo();
        await this.loadFloor();
      }).openAt(head);
    };
    document.getElementById('train-load').onclick = evt => {
      const head = evt.currentTarget;
      dirMenu('weights', this.treeFetcher('checkpoints', () => this.weightNames()), async path => {
        const name = this.relative('checkpoints', path);
        const res = await api('/api/load-checkpoint', {name});
        if (res.error) { this.say(res.error); return; }
        setHead(head, baseName(name));
        await this.loadInfo();
        this.say(`loaded ${name}`);
      }).openAt(head);
    };
  }

  // dir-tree lists folders only, so the files a folder holds come from the server's own lists
  async setNames() {
    return (await api('/api/training-sets')).sets.map(set => set.name);
  }

  async weightNames() {
    return (await api('/api/saved-checkpoints')).weights.map(w => w.name);
  }

  // adapts dir-tree to what dirMenu wants: {path, parent, dirs, files}. `filesOf` names every
  // pickable entry as a path relative to the root; the ones directly under this folder are shown
  treeFetcher(root, filesOf) {
    return async path => {
      const query = `root=${root}` + (path ? `&under=${encodeURIComponent(path)}` : '');
      const data = await api(`/api/dir-tree?${query}`);
      this.roots[root] = data.root;
      const here = this.relative(root, data.here);
      const parentOf = name => (name.includes('/') ? name.slice(0, name.lastIndexOf('/')) : '');
      const files = (await filesOf()).filter(name => parentOf(name) === here).map(baseName);
      return {
        path: data.here, parent: data.parent,
        dirs: data.entries.map(e => e.name), files,
      };
    };
  }

  // an absolute path under a root, as the name relative to it
  relative(root, path) {
    const base = (this.roots[root] || '').replace(/\/$/, '');
    return path.startsWith(base) ? path.slice(base.length).replace(/^\//, '') : path;
  }

  // newest first as the server sends them, except a finished run's final entry leads
  async checkpointItems() {
    const data = await api('/api/saved-checkpoints');
    const rows = data.weights || data.checkpoints || [];
    return [...rows.filter(w => w.final), ...rows.filter(w => !w.final)].map(w => ({
      id: w.name, label: w.name, stats: w.final ? 'final' : (w.kb ? `${w.kb} kB` : ''),
    }));
  }

  // opens while a run goes on: it reads the list, and saving snapshots the weights as they are now
  async openSaveMenu() {
    const head = document.getElementById('train-save');
    if (head.classList.contains('disabled')) return;
    const items = await this.checkpointItems();
    new Menu({
      title: 'save weights', persistent: true,
      sections: [
        {kind: 'list', label: 'saved so far', empty: 'no checkpoints yet', items, onPick: () => {}},
        {kind: 'add', placeholder: 'a name for these weights', button: 'save',
          onAdd: async (value, menu) => {
            const name = (value || '').trim();
            const res = await api('/api/save-checkpoint', name ? {name} : {});
            this.say(res.error ? res.error : `saved ${res.name}`);
            if (!res.error) menu.close();
          }},
      ],
    }).openAt(head);
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
    // a short run can end before the first poll sees it running; info is reloaded either way
    this.expectRun = true;
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
    // a run reports its latest epoch loss as `loss`; train_loss is only set once it finishes
    const loss = this.fmt(job.train_loss ?? job.loss);
    if (job.error) this.say(job.error);
    else if (job.running) this.say(`epoch ${job.epoch} / ${job.epochs}  loss ${loss}`);
    else if (job.finished) this.say(`done - ${job.epochs} epochs, final loss ${loss}`);
    else if (job.aborted) this.say(`aborted at epoch ${job.aborted} of ${job.epochs}`);
    else this.say('idle');
    setText('loss-train', this.fmt(job.train_loss ?? job.loss));
    setText('loss-val', this.fmt(job.val_loss));
    setText('loss-test', this.fmt(job.test_loss));
    document.getElementById('train-abort').classList.toggle('disabled', !job.running);
    // save weights waits for a first checkpoint, and stays usable while the run continues
    this.hasWeights = this.hasWeights || job.finished || (job.checkpoint_count || 0) >= 1;
    this.paintSave();
    this.drawLoss(job.history || []);
    if (!job.running) await this.showSeparation(job);
    if (job.running && !this.poll) this.poll = setInterval(() => this.refreshStatus(), 700);
    if (!job.running && (this.poll || this.expectRun)) {
      clearInterval(this.poll);
      this.poll = null;
      this.expectRun = false;
      this.loadInfo();
    }
  }

  // the histogram's value at a share of its samples, as the centre of the bin it falls in
  quantile(bins, share) {
    const total = bins.reduce((sum, n) => sum + n, 0);
    let seen = 0;
    for (let i = 0; i < bins.length; i++) {
      seen += bins[i];
      if (total && seen >= share * total) return (i + 0.5) / bins.length;
    }
    return 0;
  }

  // separation comes from the peak distribution: response on a confirmed object against background
  async showSeparation(job) {
    if (!(job.finished || job.weights)) {
      this.facts('train-accuracy', [['not measured yet', 'train a model to see this']]);
      this.facts('train-holdout', [['val and test loss', 'shown above once a run finishes']]);
      return;
    }
    this.facts('train-holdout', [['val and test loss', 'shown above']]);
    const dist = await api('/api/peak-distribution');
    if (dist.error) { this.facts('train-accuracy', [['not measured', dist.error]]); return; }
    const onObject = this.quantile(dist.on_object, 0.5);
    const background = this.quantile(dist.elsewhere, 0.99);
    this.facts('train-accuracy', [
      ['on an object (median)', onObject.toFixed(3)], ['background (99th pct)', background.toFixed(3)],
      ['ratio', background ? `${(onObject / background).toFixed(2)}x brighter` : '-'],
      ['sampled', `${dist.counts.on_object} objects in ${dist.frames} frames`],
    ]);
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
    this.showSweepWarning(job);
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

  // frames wider or narrower than the weights were trained on are resampled, not refused
  showSweepWarning(job) {
    const el = document.getElementById('sweep-warning');
    el.textContent = job.warning || '';
    el.classList.toggle('hidden', !job.warning);
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
          setHead(document.getElementById('sweep-open'), item.id);
        }, {empty: 'no recordings with frames'}).openAt(head);
    };
    document.getElementById('sweep-percent').onclick = evt => {
      const el = evt.currentTarget;
      listMenu('how much of it to sweep',
        [10, 25, 50, 100].map(n => ({id: String(n), label: `${n}%`, on: el.dataset.value === String(n)})),
        item => { el.dataset.value = item.id; setHead(el, `${item.id}%`); }
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
      this.showSweepWarning({});
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
