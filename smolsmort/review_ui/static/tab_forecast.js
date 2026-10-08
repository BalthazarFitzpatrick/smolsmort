// forecast: the regression and classification topics share this one set of tabs, parameterised by
// topic/task. the flavour dropdown picks spec.mode; the topic itself picks spec.task

const FORECAST_ROW_ROLES = ['anchor', 'dimension', 'measure', 'target', 'ignore'];
const FORECAST_AGGREGATIONS = ['sum', 'mean', 'median', 'min', 'max'];
const FORECAST_UNITS = ['day', 'week', 'month', 'quarter', 'year'];
const FORECAST_PAGE_SIZE = 50;

function todayIso() {
  return new Date().toISOString().slice(0, 10);
}

function forecastNode(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function forecastFamilySummary(families) {
  if (!Array.isArray(families) || !families.length) return '-';
  const groups = new Map();
  const labels = {
    roll: 'rolling', cal: 'calendar', date: 'calendar', dim: 'dims',
    measure: 'measures', log: 'log measures', freq: 'frequencies', lead: 'measure lags',
  };
  families.forEach(family => {
    if (typeof family !== 'string') return;
    const colon = family.indexOf(':');
    const kind = family.startsWith('lag+') ? 'lags'
      : (labels[family.slice(0, colon)] || family.slice(0, colon));
    const label = colon < 0 && !family.startsWith('lag+') ? 'other' : kind;
    const value = family.startsWith('lag+') ? family.slice(4)
      : colon < 0 ? (family === 'season' ? 'seasonal lag' : family) : family.slice(colon + 1);
    if (!groups.has(label)) groups.set(label, new Set());
    groups.get(label).add(value);
  });
  const order = ['lags', 'rolling', 'ewm', 'calendar', 'dims', 'measures',
    'log measures', 'frequencies', 'measure lags', 'other'];
  const calendar = ['day', 'weekday', 'week', 'month', 'quarter', 'year', 'trend'];
  return [...groups].sort(([a], [b]) => {
    const index = label => order.includes(label) ? order.indexOf(label) : order.length;
    return index(a) - index(b) || a.localeCompare(b);
  }).map(([kind, items]) => {
    const values = [...items].sort((a, b) => {
      if (['lags', 'rolling', 'ewm'].includes(kind)) return Number(a) - Number(b);
      if (kind === 'calendar' && calendar.includes(a) && calendar.includes(b)) {
        return calendar.indexOf(a) - calendar.indexOf(b);
      }
      return a.localeCompare(b);
    });
    const detail = kind === 'ewm' && values.length > 1
      ? `${values[0]}-${values[values.length - 1]}` : values.join(',');
    return kind === 'other' ? detail : `${kind} ${detail}`;
  }).join(' · ') || '-';
}

function forecastDropdown(options, value, onPick, label) {
  const head = forecastNode('button', 'toggle dropdown-head');
  head.type = 'button';
  head.setAttribute('aria-label', label);
  head.append(forecastNode('span', null, value));
  head.onclick = () => listMenu(label, options.map(id => ({id, label: id, on: id === value})), item => {
    value = item.id;
    setHead(head, value);
    onPick(value);
  }).openAt(head);
  return head;
}

// choices open inside the parent menu: a second floating menu would dismiss its draft
function forecastInlineSelect(options, value, onPick, label) {
  const wrap = forecastNode('div', 'forecast-inline-select');
  const head = forecastNode('div', 'toggle dropdown-head menu-item');
  head.tabIndex = 0;
  head.setAttribute('role', 'button');
  head.setAttribute('aria-label', label);
  const text = forecastNode('span', null, value);
  const list = forecastNode('div', 'forecast-inline-list');
  head.append(text);
  const close = () => { list.replaceChildren(); head.classList.remove('open'); };
  head.onclick = () => {
    if (list.childElementCount) { close(); return; }
    head.classList.add('open');
    options.forEach(option => {
      const row = forecastNode('div', 'toggle menu-item' + (option === value ? ' on' : ''));
      row.append(forecastNode('span', 'name', option));
      row.onclick = () => {
        value = option;
        text.textContent = option;
        close();
        head.focus();
        onPick(option);
      };
      list.append(row);
    });
  };
  wrap.append(head, list);
  return wrap;
}

class ForecastSetupMenu extends Menu {
  constructor(key, options) {
    super(options);
    this.key = key;
  }

  _build() {
    const panel = super._build();
    panel.classList.add('forecast-setup-menu', `forecast-setup-${this.key}`);
    return panel;
  }
}

// one row per number, right-aligned buttons - the stepper-row recipe from the train tab
function forecastStepperRow(prefix, row) {
  return `<div class="run-controls stepper-row" data-stepper="${row.key}">
    <span class="field-label">${row.label}</span>
    <span class="spacer"></span>
    <span class="field-value" id="${prefix}-${row.key}">${row.value}</span>
    <div class="toggle" data-step="-${row.step}">-</div>
    <div class="toggle" data-step="${row.step}">+</div>
  </div>`;
}

function forecastWireSteppers(panel, prefix, rows, values, onChange) {
  panel.querySelectorAll(`[data-stepper]`).forEach(row => {
    const key = row.dataset.stepper;
    const spec = rows.find(r => r.key === key);
    if (!spec) return;
    row.querySelectorAll('[data-step]').forEach(btn => {
      btn.onclick = () => {
        values[key] = Math.max(spec.min, Math.min(spec.max, values[key] + Number(btn.dataset.step)));
        setText(`${prefix}-${key}`, String(values[key]));
        onChange();
      };
    });
  });
}

class ForecastTopic {
  constructor(topic, task) {
    this.topic = topic;
    this.task = task;
    this.storageKey = `smolsmort:forecast:${topic}`;
    this.source = null;
    this.encoding = 'auto';
    this.columnsInfo = [];
    this.columnState = {}; // name -> {role, aggregation, known}
    this.seriesSetup = null;
    this.step = 'week';
    this.horizon = 8;
    this.asOf = todayIso();
    this.where = '';
    this.predictWhere = '';
    this.censorOn = false;
    this.censorUnit = 'week';
    this.censorAsOf = todayIso();
    this.lastPrep = null;
    this.searchValues = {population: 16, plateau: 5, generations: 60, minutes: 20};
    this.currentRunId = null;
    this.pollTimer = null;
    this.searchChart = null;
    this.lossChart = null;
    this.lossCurves = [];
    this.lossGeneration = -1;
    this.resultsRunId = null;
    this.resultsSpec = null;
    this.chart = null;
    this.filters = {};
    this.breakdown = [];
    this.tableState = {page: 0, sort: null, desc: false, group: null};
    this.loadStored();
    if (!FORECAST_UNITS.includes(this.step)) this.step = 'week';
  }

  mode() {
    return smolsmortTabs.flavour(this.topic) || 'row';
  }

  // ---- persistence: the last spec per topic, wrapped so a private window never breaks the tab ----
  loadStored() {
    try {
      const raw = localStorage.getItem(this.storageKey);
      if (!raw) return;
      const saved = JSON.parse(raw);
      Object.assign(this, saved.state || {});
      // utf-8 was the old default, and auto still picks it for any file that decodes as utf-8
      if (this.encoding === 'utf-8') this.encoding = 'auto';
    } catch (err) { /* nothing saved, or it did not parse - start clean */ }
  }

  saveStored() {
    try {
      const state = {
        source: this.source, encoding: this.encoding, columnState: this.columnState,
        seriesSetup: this.seriesSetup,
        step: this.step, horizon: this.horizon, asOf: this.asOf, where: this.where,
        predictWhere: this.predictWhere, censorOn: this.censorOn, censorUnit: this.censorUnit,
        censorAsOf: this.censorAsOf,
      };
      localStorage.setItem(this.storageKey, JSON.stringify({state}));
    } catch (err) { /* private window, no matter */ }
  }

  onFlavourChange() {
    if (this.dataSay) this.dataSay('flavour changed - reloading columns for the new mode');
    if (this.source) this.loadColumns();
    this.paintDataMode();
  }

  // ================================================================== data tab

  mountData(panel) {
    const censorBlock = this.task === 'regression' ? `
      <div class="run-controls forecast-censor-row">
        <div class="toggle" id="${this.topic}-censor-toggle">censoring off</div>
        <span class="field-label">anchor</span>
        <span class="field-value" id="${this.topic}-censor-anchor">-</span>
        <span class="field-label">unit</span>
        <span id="${this.topic}-censor-unit"></span>
        <span class="field-label">as of</span>
        <input type="date" id="${this.topic}-censor-asof" class="text-field">
      </div>` : '';
    panel.innerHTML = `
      <div class="forecast-pane">
        <div class="run-controls">
          <div class="toggle dropdown-head grow" id="${this.topic}-data-source"><span>source</span></div>
          <span class="field-label">encoding</span>
          <input type="text" id="${this.topic}-data-encoding" class="text-field forecast-encoding">
          <span class="stat" id="${this.topic}-data-root"></span>
        </div>
        <div id="${this.topic}-data-columns"></div>
        <div class="h-divider"></div>
        <div id="${this.topic}-series-row">
          <div class="forecast-setup-heads" id="${this.topic}-setup-heads"></div>
          <div class="stat forecast-setup-readout" id="${this.topic}-setup-readout" aria-live="polite"></div>
          <div class="run-controls">
            <span class="field-label">as of</span>
            <input type="date" id="${this.topic}-asof" class="text-field">
          </div>
        </div>
        <div class="field-label" id="${this.topic}-row-heading">row settings</div>
        <div id="${this.topic}-row-row">
          ${censorBlock}
          <div class="run-controls">
            <span class="field-label">rows to predict</span>
            <input type="text" id="${this.topic}-predict-where"
                   placeholder="sql filter, optional" class="text-field grow">
          </div>
        </div>
        <div class="run-controls">
          <span class="field-label">filter</span>
          <input type="text" id="${this.topic}-where"
                 placeholder="sql filter over the source, optional" class="text-field grow">
        </div>
        <div class="run-controls">
          <div class="toggle adds" id="${this.topic}-prepare">prepare</div>
          <span class="stat" id="${this.topic}-data-status"></span>
        </div>
        <div id="${this.topic}-prep-summary" class="train-facts"></div>
        <details id="${this.topic}-sql-details">
          <summary>generated sql</summary>
          <pre id="${this.topic}-sql-text" class="forecast-sql"></pre>
        </details>
      </div>`;
    // typed text goes in through the property, never the markup: a sql filter carries quotes
    const typed = {
      'censor-asof': this.censorAsOf, 'data-encoding': this.encoding, asof: this.asOf,
      'predict-where': this.predictWhere, where: this.where,
    };
    for (const [suffix, value] of Object.entries(typed)) {
      const input = panel.querySelector(`#${this.topic}-${suffix}`);
      if (input) input.value = value || '';
    }
    this.dataSay = text => setText(`${this.topic}-data-status`, text);
    this.wireData(panel);
    this.renderSetupHeads();
    this.paintDataMode();
    return {enter: () => this.enterData()};
  }

  async enterData() {
    if (this.source) setHead(document.getElementById(`${this.topic}-data-source`), baseName(this.source));
    if (this.columnsInfo.length) this.renderColumnGrid();
  }

  paintDataMode() {
    const series = this.mode() === 'series';
    document.getElementById(`${this.topic}-series-row`).classList.toggle('hidden', !series);
    document.getElementById(`${this.topic}-row-row`).classList.toggle('hidden', series);
    const heading = document.getElementById(`${this.topic}-row-heading`);
    if (heading) heading.classList.toggle('hidden', series);
    this.renderColumnGrid();
  }

  wireData(panel) {
    document.getElementById(`${this.topic}-data-source`).onclick = evt => this.openSourcePicker(evt.currentTarget);
    document.getElementById(`${this.topic}-data-encoding`).onchange = evt => {
      this.encoding = evt.currentTarget.value.trim() || 'auto';
      evt.currentTarget.value = this.encoding;
      this.saveStored();
      if (this.source) this.loadColumns();
    };
    const asOf = document.getElementById(`${this.topic}-asof`);
    if (asOf) asOf.onchange = () => { this.asOf = asOf.value; this.saveStored(); };
    const predictWhere = document.getElementById(`${this.topic}-predict-where`);
    if (predictWhere) predictWhere.onchange = () => { this.predictWhere = predictWhere.value; this.saveStored(); };
    const where = document.getElementById(`${this.topic}-where`);
    where.onchange = () => { this.where = where.value; this.saveStored(); };
    if (this.task === 'regression') {
      const toggle = document.getElementById(`${this.topic}-censor-toggle`);
      toggle.onclick = () => {
        this.censorOn = !this.censorOn;
        toggle.textContent = this.censorOn ? 'censoring on' : 'censoring off';
        toggle.classList.toggle('on', this.censorOn);
        this.saveStored();
      };
      toggle.classList.toggle('on', this.censorOn);
      toggle.textContent = this.censorOn ? 'censoring on' : 'censoring off';
      const unit = document.getElementById(`${this.topic}-censor-unit`);
      unit.replaceChildren(forecastDropdown(FORECAST_UNITS, this.censorUnit, value => {
        this.censorUnit = value;
        this.saveStored();
      }, 'censor unit'));
      const asof2 = document.getElementById(`${this.topic}-censor-asof`);
      asof2.onchange = () => { this.censorAsOf = asof2.value; this.saveStored(); };
      this.updateCensorAnchor();
    }
    document.getElementById(`${this.topic}-prepare`).onclick = () => this.prepare();
  }

  updateCensorAnchor() {
    const el = document.getElementById(`${this.topic}-censor-anchor`);
    if (!el) return;
    const anchor = Object.entries(this.columnState).find(([, c]) => c.role === 'anchor');
    el.textContent = anchor ? anchor[0] : '-';
  }

  async openSourcePicker(head) {
    try {
      const data = await api('/api/forecast-sources');
      setText(`${this.topic}-data-root`, `root: ${data.root}`);
      listMenu('source - ' + data.root, data.files.map(f => ({
        id: f.path, label: f.path, on: f.path === this.source,
      })), item => {
        this.source = item.id;
        setHead(head, baseName(item.id));
        this.saveStored();
        this.loadColumns();
      }, {empty: 'no source files under the forecast root'}).openAt(head);
    } catch (err) {
      this.dataSay(err.message);
    }
  }

  async loadColumns() {
    if (!this.source) return;
    this.dataSay('reading columns...');
    try {
      const res = await api('/api/forecast-columns', {
        source: this.source, encoding: this.encoding, mode: this.mode(),
      });
      this.columnsInfo = res.columns;
      // row roles survive flavour changes; series picks have their own saved state
      const roles = FORECAST_ROW_ROLES;
      const next = {};
      this.columnsInfo.forEach(col => {
        const prior = this.columnState[col.name];
        const suggested = col.kind === 'date' ? 'anchor' : col.suggested_role;
        const role = prior && roles.includes(prior.role) ? prior.role : suggested;
        next[col.name] = {
          role,
          aggregation: (prior && prior.aggregation) || FORECAST_AGGREGATIONS[0],
          known: prior ? prior.known !== false : true,
        };
      });
      this.reconcileSeriesSetup();
      this.columnState = next;
      this.renderColumnGrid();
      this.updateCensorAnchor();
      const detected = this.encoding === 'auto' ? `, detected ${res.encoding}` : '';
      this.dataSay(`${this.columnsInfo.length} columns${detected}`);
      this.saveStored();
    } catch (err) {
      this.dataSay(err.message);
    }
  }

  reconcileSeriesSetup() {
    const fields = kind => this.columnsInfo.filter(col => col.kind === kind).map(col => col.name);
    const measures = fields('number');
    const dates = fields('date');
    const dimensions = fields('text');
    const prior = this.seriesSetup || {
      targets: measures.filter(name => this.columnState[name]?.role === 'target'),
      aggregations: Object.fromEntries(measures.map(name => [name, this.columnState[name]?.aggregation])),
      time: dates.find(name => this.columnState[name]?.role === 'time') || dates[0],
      dimensions: [],
    };
    const targets = prior.targets.filter(name => measures.includes(name));
    this.seriesSetup = {
      targets,
      aggregations: Object.fromEntries(targets.map(name => [name,
        FORECAST_AGGREGATIONS.includes(prior.aggregations[name]) ? prior.aggregations[name] : 'sum'])),
      time: dates.includes(prior.time) ? prior.time : (dates[0] || null),
      dimensions: prior.dimensions.filter(name => dimensions.includes(name)),
    };
    if (!FORECAST_UNITS.includes(this.step)) this.step = 'week';
    this.paintSetupHeads();
  }

  countSeries(dimensions) {
    return dimensions.reduce((product, name) => product *
      (this.columnsInfo.find(col => col.name === name)?.distinct || 0), 1);
  }

  renderSetupHeads() {
    const box = document.getElementById(`${this.topic}-setup-heads`);
    box.replaceChildren();
    [['predict', 'predict'], ['aggregate', 'aggregate'], ['time', 'extend in time'],
      ['scaffold', 'scaffold along']].forEach(([key, label]) => {
      const slot = forecastNode('div', 'forecast-setup-slot');
      const head = forecastNode('button', 'toggle dropdown-head');
      head.type = 'button';
      head.id = `${this.topic}-setup-${key}`;
      head.append(forecastNode('span'));
      head.onclick = () => this.openSetupPicker(key, head);
      slot.append(forecastNode('div', 'field-label', label), head);
      box.append(slot);
    });
    this.paintSetupHeads();
  }

  paintSetupHeads() {
    const setup = this.seriesSetup || {targets: [], aggregations: {}, time: null, dimensions: []};
    const aggregate = setup.targets.map(name => `${setup.aggregations[name]}(${name})`).join(', ');
    const series = this.countSeries(setup.dimensions).toLocaleString('en');
    const captions = {
      predict: setup.targets.join(', ') || 'choose variables',
      aggregate: aggregate || 'predict something first',
      time: `${setup.time || 'choose date'} · ${this.step} · +${this.horizon}`,
      scaffold: setup.dimensions.length ? `${setup.dimensions.join(' x ')} · ${series}` : 'none (one total series)',
    };
    Object.entries(captions).forEach(([key, text]) => {
      const head = document.getElementById(`${this.topic}-setup-${key}`);
      if (head) { setHead(head, text); head.title = text; }
    });
    setText(`${this.topic}-setup-readout`, `${aggregate || 'nothing yet'} · ${setup.time || 'no date'}, ` +
      `${this.horizon} ${this.step}${this.horizon === 1 ? '' : 's'} ahead · ` +
      `${setup.dimensions.join(' x ') || 'total'} = ${series} series`);
  }

  openSetupPicker(key, head) {
    const saved = this.seriesSetup || {targets: [], aggregations: {}, time: null, dimensions: []};
    const draft = {
      ...saved, targets: [...saved.targets], aggregations: {...saved.aggregations},
      dimensions: [...saved.dimensions], step: this.step === 'auto' ? 'week' : this.step,
      horizon: this.horizon,
    };
    const fields = kind => this.columnsInfo.filter(col => col.kind === kind);
    const footer = {kind: 'buttons', buttons: [
      {id: 'save', label: 'save', onClick: menu => {
        this.seriesSetup = {
          targets: draft.targets,
          aggregations: Object.fromEntries(draft.targets.map(name => [name, draft.aggregations[name] || 'sum'])),
          time: draft.time, dimensions: draft.dimensions,
        };
        this.step = draft.step;
        this.horizon = draft.horizon;
        this.saveStored();
        this.paintSetupHeads();
        menu.close();
      }},
      {id: 'close', label: 'close', onClick: menu => menu.close()},
    ]};
    let title, sections;
    if (key === 'predict') {
      title = 'variables to predict';
      sections = [{kind: 'list', multi: true, empty: 'no numeric fields',
        items: fields('number').map(col => ({id: col.name, label: col.name, on: draft.targets.includes(col.name)})),
        onPick: (item, on) => {
          const picks = on ? [...draft.targets, item.id] : draft.targets.filter(name => name !== item.id);
          draft.targets = fields('number').map(col => col.name).filter(name => picks.includes(name));
          if (on && !draft.aggregations[item.id]) draft.aggregations[item.id] = 'sum';
        },
      }];
    } else if (key === 'aggregate') {
      title = 'aggregation per variable';
      const pane = forecastNode('div', 'forecast-setup-pane');
      if (!draft.targets.length) pane.append(forecastNode('span', 'none', 'pick variables to predict in the first dropdown'));
      draft.targets.forEach(name => {
        const row = forecastNode('div', 'forecast-agg-row');
        const field = forecastNode('div', 'toggle forecast-field');
        field.append(forecastNode('span', 'name', name));
        row.append(field, forecastInlineSelect(FORECAST_AGGREGATIONS, draft.aggregations[name], value => {
          draft.aggregations[name] = value;
        }, `aggregation for ${name}`));
        pane.append(row);
      });
      sections = [{kind: 'node', node: pane}];
    } else if (key === 'time') {
      title = 'time to extend';
      const right = forecastNode('div', 'forecast-setup-pane forecast-time-side');
      const aheadLabel = forecastNode('div', 'field-label', `${draft.step}s ahead`);
      const granularity = forecastInlineSelect(FORECAST_UNITS, draft.step, value => {
        draft.step = value;
        aheadLabel.textContent = `${value}s ahead`;
      }, 'date granularity');
      const ahead = forecastNode('input', 'text-field');
      ahead.type = 'number';
      ahead.min = '1';
      ahead.max = '366';
      ahead.value = draft.horizon;
      ahead.setAttribute('aria-label', 'steps into the future');
      ahead.oninput = () => { draft.horizon = Math.max(1, Math.min(366, Math.round(+ahead.value) || 1)); };
      ahead.onkeydown = evt => { if (evt.key !== 'Escape') evt.stopPropagation(); };
      right.append(forecastNode('div', 'field-label', 'date granularity'), granularity, aheadLabel, ahead);
      sections = [{kind: 'list', multi: true, label: 'date field', empty: 'no date fields',
        items: fields('date').map(col => ({id: col.name, label: col.name, on: draft.time === col.name})),
        onPick: (item, on, menu) => {
          draft.time = item.id;
          menu.el.querySelectorAll('.menu-list .menu-item').forEach(row => row.classList.toggle('on', row.dataset.id === item.id));
        },
      }, {kind: 'node', node: right}];
    } else {
      title = 'dimensions to scaffold along';
      const note = forecastNode('div', 'stat forecast-series-product');
      const paint = () => {
        note.textContent = draft.dimensions.length ?
          `${draft.dimensions.map(name => this.columnsInfo.find(col => col.name === name).distinct).join(' x ')} = ` +
          `${this.countSeries(draft.dimensions).toLocaleString('en')} series` : '1 series (the total)';
      };
      paint();
      sections = [{kind: 'list', multi: true, empty: 'no dimensions',
        items: fields('text').map(col => ({id: col.name, label: col.name, stats: `[${col.distinct}]`, on: draft.dimensions.includes(col.name)})),
        onPick: (item, on) => {
          const picks = on ? [...draft.dimensions, item.id] : draft.dimensions.filter(name => name !== item.id);
          draft.dimensions = fields('text').map(col => col.name).filter(name => picks.includes(name));
          paint();
        },
      }, {kind: 'node', node: note}];
    }
    const menu = new ForecastSetupMenu(key, {
      title, columns: key === 'time', sections: [...sections, footer], onDismiss: () => head.focus(),
    });
    menu.openAt(head);
  }

  renderColumnGrid() {
    const box = document.getElementById(`${this.topic}-data-columns`);
    box.innerHTML = '';
    if (!this.columnsInfo.length || this.mode() === 'series') return;
    const table = document.createElement('table');
    table.className = 'forecast-column-grid';
    const head = document.createElement('tr');
    ['name', 'kind', 'samples', 'role', 'known when predicting']
      .forEach(label => {
        const th = document.createElement('th');
        th.textContent = label;
        head.appendChild(th);
      });
    table.appendChild(head);
    this.columnsInfo.forEach(col => {
      const state = this.columnState[col.name] || {role: col.suggested_role, aggregation: 'sum', known: true};
      const row = document.createElement('tr');
      const nameCell = document.createElement('td');
      nameCell.textContent = col.name;
      const kindCell = document.createElement('td');
      kindCell.textContent = col.kind;
      const samplesCell = document.createElement('td');
      samplesCell.textContent = col.sample.join(', ');
      const roleCell = document.createElement('td');
      const roleHead = forecastDropdown(FORECAST_ROW_ROLES, state.role, role => {
        this.columnState[col.name] = {...this.columnState[col.name], role};
        this.updateCensorAnchor();
        this.renderColumnGrid();
        this.saveStored();
      }, `role for ${col.name}`);
      roleCell.appendChild(roleHead);
      const lastCell = document.createElement('td');
      const known = state.known !== false;
      const enabled = ['dimension', 'measure'].includes(state.role);
      const check = forecastNode('button', 'toggle' + (known ? ' on' : '') +
        (enabled ? '' : ' disabled'), known ? 'known' : 'not known');
      check.type = 'button';
      check.setAttribute('role', 'checkbox');
      check.setAttribute('aria-checked', String(known));
      check.disabled = !enabled;
      check.onclick = () => {
        this.columnState[col.name] = {...this.columnState[col.name], known: !known};
        this.renderColumnGrid();
        this.saveStored();
      };
      lastCell.appendChild(check);
      row.append(nameCell, kindCell, samplesCell, roleCell, lastCell);
      table.appendChild(row);
    });
    box.appendChild(table);
  }

  buildSpec() {
    const mode = this.mode();
    const series = mode === 'series';
    const setup = this.seriesSetup;
    const columns = series ? this.columnsInfo.map(col => ({
      name: col.name,
      role: setup?.targets.includes(col.name) ? 'target' : setup?.time === col.name ? 'time' :
        setup?.dimensions.includes(col.name) ? 'dimension' : 'ignore',
      aggregation: setup?.targets.includes(col.name) ? setup.aggregations[col.name] : null,
      known: true,
    })) : Object.entries(this.columnState).map(([name, state]) => ({
      name,
      role: state.role,
      aggregation: null,
      known: state.known !== false,
    }));
    const anchorEntry = columns.find(c => c.role === 'anchor');
    return {
      source: this.source,
      mode,
      task: this.task,
      columns,
      encoding: this.encoding || 'auto',
      step: series ? (this.step === 'auto' ? null : this.step) : null,
      horizon: series ? this.horizon : 8,
      where: this.where || null,
      predict_where: !series ? (this.predictWhere || null) : null,
      censor: (!series && this.task === 'regression' && this.censorOn && anchorEntry) ? {
        anchor: anchorEntry.name, unit: this.censorUnit, as_of: this.censorAsOf || null,
      } : null,
      as_of: series ? (this.asOf || null) : null,
    };
  }

  async prepare() {
    if (!this.source) { this.dataSay('pick a source first'); return; }
    this.dataSay('preparing...');
    try {
      const spec = this.buildSpec();
      const res = await api('/api/forecast-prep', {spec});
      this.lastPrep = res;
      this.spec = spec;
      this.paintPrepSummary(res);
      this.dataSay(res.reused ? 'prepared (reused cache)' : 'prepared');
    } catch (err) {
      this.dataSay(err.message);
    }
  }

  paintPrepSummary(res) {
    const box = document.getElementById(`${this.topic}-prep-summary`);
    box.innerHTML = '';
    const summary = res.summary || {};
    Object.entries(summary).forEach(([k, v]) => {
      const key = document.createElement('span');
      key.className = 'k';
      key.textContent = k;
      const val = document.createElement('span');
      val.className = 'v';
      val.textContent = String(v);
      box.append(key, val);
    });
    setText(`${this.topic}-sql-text`, res.sql || '');
  }

  // ================================================================== search tab

  mountSearch(panel) {
    const budgetRows = [
      {key: 'population', label: 'population', step: 2, min: 2, max: 400, value: this.searchValues.population},
      {key: 'plateau', label: 'plateau generations', step: 1, min: 1, max: 100, value: this.searchValues.plateau},
      {key: 'generations', label: 'max generations', step: 1, min: 1, max: 500, value: this.searchValues.generations},
      {key: 'minutes', label: 'time cap (minutes)', step: 1, min: 1, max: 240, value: this.searchValues.minutes},
    ];
    panel.innerHTML = `
      <div class="forecast-pane">
        <div class="field-label">budget</div>
        <div id="${this.topic}-budget">${budgetRows.map(r => forecastStepperRow(`${this.topic}-budget`, r)).join('')}</div>
        <div class="run-controls">
          <div class="toggle adds" id="${this.topic}-search-start">start</div>
          <div class="toggle removes disabled" id="${this.topic}-search-cancel">cancel</div>
          <span class="stat" id="${this.topic}-search-status">idle</span>
        </div>
        <div id="${this.topic}-search-progress" class="hidden">
          <div class="field-label" id="${this.topic}-search-metric"></div>
          <div id="${this.topic}-search-chart" class="forecast-chart"></div>
          <div class="field-label">generation</div>
        </div>
        <div id="${this.topic}-search-loss" class="hidden">
          <div class="field-label" id="${this.topic}-loss-metric"></div>
          <div id="${this.topic}-loss-chart" class="forecast-chart"></div>
          <div class="field-label">boosting round</div>
        </div>
        <div id="${this.topic}-leaderboard" class="hidden">
          <div class="forecast-leaderboard-wrap">
            <table class="forecast-column-grid forecast-leaderboard" id="${this.topic}-leaderboard-table"></table>
          </div>
        </div>
        <div class="h-divider"></div>
        <div class="field-label">runs</div>
        <div id="${this.topic}-runs-list"></div>
      </div>`;
    forecastWireSteppers(panel, `${this.topic}-budget`, budgetRows, this.searchValues, () => {});
    document.getElementById(`${this.topic}-search-start`).onclick = () => this.startSearch();
    document.getElementById(`${this.topic}-search-cancel`).onclick = () => this.cancelSearch();
    return {enter: () => this.enterSearch()};
  }

  async enterSearch() {
    await this.loadRuns();
  }

  searchSay(text) { setText(`${this.topic}-search-status`, text); }

  async startSearch() {
    if (!this.spec) { this.searchSay('prepare the data first'); return; }
    const budget = {
      population: this.searchValues.population,
      plateau: this.searchValues.plateau,
      max_generations: this.searchValues.generations,
      time_cap: this.searchValues.minutes * 60,
    };
    try {
      const res = await api('/api/forecast-runs', {spec: this.spec, budget});
      this.currentRunId = res.run_id;
      this.resetSearchChart();
      this.searchSay('starting...');
      this.pollSearch();
    } catch (err) {
      this.searchSay(err.message);
    }
  }

  async cancelSearch() {
    if (!this.currentRunId) return;
    try {
      await api('/api/forecast-cancel', {id: this.currentRunId});
      this.searchSay('cancelling...');
    } catch (err) {
      this.searchSay(err.message);
    }
  }

  async pollSearch() {
    const runId = this.currentRunId;
    if (!runId) return;
    document.getElementById(`${this.topic}-search-cancel`).classList.remove('disabled');
    const state = await api(`/api/forecast-run?id=${encodeURIComponent(runId)}&since_generation=${this.lossGeneration}`);
    if (runId !== this.currentRunId) return;
    this.paintSearchChart(state.generations || []);
    this.paintLossChart(state.generation_curves || []);
    const latest = state.generations && state.generations.length
      ? state.generations[state.generations.length - 1] : null;
    if (latest) {
      this.searchSay(
        `generation ${latest.generation} - best ${Number(latest.best).toFixed(4)} - `
        + `evaluated ${latest.evaluated} - elapsed ${latest.elapsed}s`
      );
    } else {
      this.searchSay(state.state || 'starting');
    }
    if (state.state === 'done' || state.state === 'failed') {
      clearTimeout(this.pollTimer);
      this.pollTimer = null;
      document.getElementById(`${this.topic}-search-cancel`).classList.add('disabled');
      if (state.state === 'failed') {
        this.searchSay(`failed: ${state.reason || 'unknown error'}`);
      } else {
        this.searchSay(`done - ${state.verdict && state.verdict.trusted ? 'trusted' : 'not trusted'}`);
        this.paintLeaderboard(state.leaderboard || []);
      }
      await this.loadRuns();
      return;
    }
    this.pollTimer = setTimeout(() => this.pollSearch(), 2000);
  }

  resetSearchChart() {
    clearTimeout(this.pollTimer);
    this.pollTimer = null;
    if (this.searchChart) this.searchChart.destroy();
    this.searchChart = null;
    document.getElementById(`${this.topic}-search-progress`).classList.add('hidden');
    if (this.lossChart) this.lossChart.destroy();
    this.lossChart = null;
    this.lossCurves = [];
    this.lossGeneration = -1;
    document.getElementById(`${this.topic}-search-loss`).classList.add('hidden');
  }

  paintSearchChart(generations) {
    if (!generations.length) return;
    const metric = this.task === 'classification' ? 'log_loss' : 'wape';
    setText(`${this.topic}-search-metric`, `${metric} (lower is better)`);
    document.getElementById(`${this.topic}-search-progress`).classList.remove('hidden');
    const el = document.getElementById(`${this.topic}-search-chart`);
    if (!this.searchChart) this.searchChart = timeChart(el, {
      height: 220, yFormat: value => String(Number(value.toPrecision(4))),
    });
    this.searchChart.update({
      x: generations.map(entry => entry.generation),
      series: [{id: metric, label: `best ${metric} so far`,
        values: generations.map(entry => Number.isFinite(entry.best) ? entry.best : null)}],
    });
  }

  paintLossChart(curves) {
    const wrap = document.getElementById(`${this.topic}-search-loss`);
    const previousGeneration = this.lossGeneration;
    let added = false;
    // malformed records must not advance the cursor past later usable curves
    for (const curve of curves) {
      if (!Number.isInteger(curve.generation) || curve.generation <= previousGeneration) continue;
      const usable = curve.metric && Array.isArray(curve.validation) && curve.validation.length
        && curve.validation.every(Number.isFinite);
      if (!usable || this.lossCurves.some(entry => entry.generation === curve.generation)) continue;
      this.lossCurves.push(curve);
      this.lossGeneration = Math.max(this.lossGeneration, curve.generation);
      added = true;
    }
    if (!added) return;
    this.lossCurves.sort((a, b) => a.generation - b.generation);
    this.lossCurves = this.lossCurves.slice(-60);
    const history = this.lossCurves[this.lossCurves.length - 1];
    const bucket = history.bucket ? ` - horizon bucket ${history.bucket.join('-')}` : '';
    setText(`${this.topic}-loss-metric`, `${history.metric} (lower is better)${bucket}`);
    wrap.classList.remove('hidden');
    if (!this.lossChart) this.lossChart = timeChart(
      document.getElementById(`${this.topic}-loss-chart`), {
        height: 220, yFormat: value => String(Number(value.toPrecision(4))),
      }
    );
    const style = getComputedStyle(document.documentElement);
    const kingfisher = style.getPropertyValue('--kingfisher').trim();
    const cream = style.getPropertyValue('--cream').trim();
    const rounds = Math.max(...this.lossCurves.map(curve => curve.validation.length));
    this.lossChart.update({
      x: Array.from({length: rounds}, (_, index) => index + 1),
      series: this.lossCurves.map(curve => {
        const age = history.generation - curve.generation;
        return {
          id: `generation-${curve.generation}`, label: `generation ${curve.generation}`,
          values: Array.from({length: rounds}, (_, index) => curve.validation[index] ?? null),
          color: age === 0 ? kingfisher : cream,
          opacity: Math.max(0.2, 1 - age / 60),
        };
      }),
    });
  }

  paintLeaderboard(rows) {
    const wrap = document.getElementById(`${this.topic}-leaderboard`);
    const table = document.getElementById(`${this.topic}-leaderboard-table`);
    table.innerHTML = '';
    wrap.classList.toggle('hidden', !rows.length);
    if (!rows.length) return;
    const entries = rows.slice(0, 10);
    const metric = this.task === 'classification' ? 'log_loss' : 'wape';
    const errorOf = entry => Number.isFinite(entry.fitness) ? entry.fitness : entry.metrics?.[metric];
    const best = errorOf(entries[0]);
    const number = value => Number.isFinite(value) ? String(Number(value.toPrecision(4))) : '-';
    const integer = value => Number.isInteger(value) ? String(value) : '-';
    const objectives = {
      squared: 'squared error', absolute: 'absolute error', tweedie: 'tweedie loss',
      poisson: 'poisson loss', aft: 'survival loss', logistic: 'log loss',
      softprob: 'multiclass log loss',
    };
    const hasNote = entries.some(entry => typeof entry.note === 'string' && entry.note.trim());
    const columns = ['rank', 'generation found', `validation error (${metric})`, 'gap to best',
      'mae', 'objective', 'tree depth', 'learning rate (eta)', 'feature count (families)', 'feature families'];
    if (hasNote) columns.push('note');
    const caption = forecastNode('caption', 'field-label',
      `best ten recipes the search tried, ranked by validation error (${metric}); lower is better; the winner is the first row`);
    table.appendChild(caption);
    const thead = document.createElement('thead');
    const head = document.createElement('tr');
    columns.forEach(c => {
      const th = document.createElement('th');
      th.textContent = c;
      head.appendChild(th);
    });
    thead.appendChild(head);
    table.appendChild(thead);
    const tbody = document.createElement('tbody');
    entries.forEach((entry, index) => {
      const row = document.createElement('tr');
      row.classList.toggle('on', index === 0);
      const genome = entry.genome || {};
      const params = genome.params || {};
      const error = errorOf(entry);
      const gap = Number.isFinite(error) && Number.isFinite(best)
        ? best === 0 ? (error === 0 ? 0 : null) : (error - best) / Math.abs(best) * 100 : null;
      const gapText = Number.isFinite(gap) ? `${gap < 0 ? '-' : '+'}${Math.abs(gap).toFixed(1)}%` : '-';
      const count = Array.isArray(genome.families) ? new Set(genome.families).size : entry.families;
      const values = [String(index + 1), integer(entry.generation), number(error), gapText,
        number(entry.metrics?.mae), objectives[genome.objective] || '-',
        integer(params.max_depth), number(params.eta), integer(count), forecastFamilySummary(genome.families)];
      if (hasNote) values.push(typeof entry.note === 'string' && entry.note.trim() ? entry.note : '-');
      values.forEach(value => {
        const td = document.createElement('td');
        td.textContent = value;
        row.appendChild(td);
      });
      tbody.appendChild(row);
    });
    table.appendChild(tbody);
  }

  async loadRuns() {
    const data = await api('/api/forecast-runs');
    const box = document.getElementById(`${this.topic}-runs-list`);
    if (!box) return;
    box.innerHTML = '';
    if (!data.runs.length) {
      box.appendChild(textSpan('stat', 'no runs yet'));
      return;
    }
    data.runs.forEach(run => {
      const row = document.createElement('div');
      row.className = 'run-controls forecast-run-row';
      const label = document.createElement('span');
      label.className = 'field-value';
      label.textContent = `${run.id} - ${run.state}`;
      row.appendChild(label);
      row.appendChild(textSpan('spacer', ''));
      const open = document.createElement('div');
      open.className = 'toggle';
      open.textContent = 'open results';
      open.onclick = () => this.openResults(run.id);
      const refit = document.createElement('div');
      refit.className = 'toggle';
      refit.textContent = 'refit';
      refit.onclick = () => this.refitRun(run.id);
      const warm = document.createElement('div');
      warm.className = 'toggle';
      warm.textContent = 'warm-start';
      warm.onclick = () => this.warmStartRun(run.id);
      row.append(open, refit, warm);
      box.appendChild(row);
    });
  }

  async refitRun(runId) {
    if (!this.spec) { this.searchSay('prepare the data first'); return; }
    try {
      const res = await api('/api/forecast-refit', {spec: this.spec, run_id: runId});
      this.currentRunId = res.run_id;
      this.resetSearchChart();
      this.pollSearch();
    } catch (err) {
      this.searchSay(err.message);
    }
  }

  async warmStartRun(runId) {
    if (!this.spec) { this.searchSay('prepare the data first'); return; }
    const budget = {
      population: this.searchValues.population,
      plateau: this.searchValues.plateau,
      max_generations: this.searchValues.generations,
      time_cap: this.searchValues.minutes * 60,
    };
    try {
      const res = await api('/api/forecast-runs', {spec: this.spec, budget, warm_from: runId});
      this.currentRunId = res.run_id;
      this.resetSearchChart();
      this.pollSearch();
    } catch (err) {
      this.searchSay(err.message);
    }
  }

  openResults(runId) {
    this.resultsRunId = runId;
    if (typeof activateTab === 'function') activateTab(`${this.topic}-results`);
    this.loadResults();
  }

  // ================================================================== results tab

  mountResults(panel) {
    panel.innerHTML = `
      <div class="forecast-pane forecast-results">
        <div class="run-controls">
          <div class="toggle dropdown-head" id="${this.topic}-results-run"><span>pick a run</span></div>
          <span class="field-label">break down by</span>
          <div class="toggle dropdown-head narrow" id="${this.topic}-breakdown-head"><span>none</span></div>
          <span class="spacer"></span>
          <a class="toggle adds hidden" id="${this.topic}-export" download>export csv</a>
        </div>
        <div id="${this.topic}-verdict" class="forecast-verdict hidden"></div>
        <div id="${this.topic}-filter-chips" class="label-columns"></div>
        <div id="${this.topic}-chart" class="forecast-chart"></div>
        <div class="run-controls hidden" id="${this.topic}-group-row">
          <span class="field-label">group by</span>
          <div class="toggle dropdown-head narrow" id="${this.topic}-group-head"><span>none</span></div>
        </div>
        <div class="forecast-table-wrap">
          <table class="forecast-column-grid" id="${this.topic}-results-table"></table>
        </div>
        <div class="run-controls">
          <div class="toggle" id="${this.topic}-page-prev">prev</div>
          <span class="stat" id="${this.topic}-page-info"></span>
          <div class="toggle" id="${this.topic}-page-next">next</div>
        </div>
      </div>`;
    document.getElementById(`${this.topic}-results-run`).onclick = evt => this.openRunPicker(evt.currentTarget);
    document.getElementById(`${this.topic}-breakdown-head`).onclick = evt => this.openBreakdownPicker(evt.currentTarget);
    document.getElementById(`${this.topic}-group-head`).onclick = evt => this.openGroupPicker(evt.currentTarget);
    document.getElementById(`${this.topic}-page-prev`).onclick = () => this.changePage(-1);
    document.getElementById(`${this.topic}-page-next`).onclick = () => this.changePage(1);
    return {enter: () => this.enterResults()};
  }

  async enterResults() {
    if (this.resultsRunId) await this.loadResults();
  }

  async openRunPicker(head) {
    const data = await api('/api/forecast-runs');
    listMenu('runs', data.runs.map(r => ({
      id: r.id, label: `${r.id} - ${r.state}`, on: r.id === this.resultsRunId,
    })), item => {
      this.resultsRunId = item.id;
      setHead(head, item.id);
      this.loadResults();
    }, {empty: 'no runs yet'}).openAt(head);
  }

  async loadResults() {
    if (!this.resultsRunId) return;
    try {
      const state = await api(`/api/forecast-run?id=${encodeURIComponent(this.resultsRunId)}`);
      this.resultsSpec = state.spec;
      setHead(document.getElementById(`${this.topic}-results-run`), this.resultsRunId);
      const exportLink = document.getElementById(`${this.topic}-export`);
      exportLink.href = `/api/forecast-export/${encodeURIComponent(this.resultsRunId)}.csv`;
      exportLink.classList.remove('hidden');
      this.filters = {};
      this.breakdown = [];
      this.tableState = {page: 0, sort: null, desc: false, group: null};
      await this.loadFilterChoices();
      await this.refreshView();
      await this.refreshTable();
    } catch (err) {
      this.paintVerdict({trusted: false, summary: err.message, checks: []});
    }
  }

  dims() {
    if (!this.resultsSpec) return [];
    return (this.resultsSpec.columns || []).filter(c => c.role === 'dimension').map(c => c.name);
  }

  // one page of the table, unfiltered, just to learn what values each dimension carries
  async loadFilterChoices() {
    this.filterChoices = {};
    const dims = this.dims();
    if (!dims.length) { this.renderFilterChips(); return; }
    const data = await api(
      `/api/forecast-table?id=${encodeURIComponent(this.resultsRunId)}&page=0&size=2000`
    );
    dims.forEach(dim => {
      const idx = data.columns.indexOf(dim);
      if (idx < 0) return;
      this.filterChoices[dim] = [...new Set(data.rows.map(r => r[idx]))].filter(v => v != null);
    });
    this.renderFilterChips();
  }

  renderFilterChips() {
    const box = document.getElementById(`${this.topic}-filter-chips`);
    box.innerHTML = '';
    const dims = this.dims();
    dims.forEach((dim, index) => {
      if (index) box.appendChild(Object.assign(document.createElement('div'), {className: 'divider'}));
      const col = document.createElement('div');
      col.className = 'col';
      col.appendChild(Object.assign(document.createElement('div'), {className: 'field-label', textContent: dim}));
      (this.filterChoices[dim] || []).forEach(value => {
        const chip = document.createElement('div');
        chip.className = 'toggle';
        chip.textContent = String(value);
        chip.onclick = () => {
          chip.classList.toggle('on');
          const picked = [...col.querySelectorAll('.toggle.on')].map(el => el.textContent);
          if (picked.length) this.filters[dim] = picked; else delete this.filters[dim];
          this.refreshView();
          this.tableState.page = 0;
          this.refreshTable();
        };
        col.appendChild(chip);
      });
      box.appendChild(col);
    });
  }

  openBreakdownPicker(head) {
    const dims = this.dims();
    new Menu({
      title: 'break down by (up to 3)', persistent: true,
      sections: [{
        kind: 'list', multi: true, empty: 'no dimensions to break down by',
        items: dims.map(d => ({id: d, label: d, on: this.breakdown.includes(d)})),
        onPick: (item, on) => {
          if (on && this.breakdown.length >= 3) return; // capped, same as the server itself
          this.breakdown = on
            ? [...this.breakdown, item.id]
            : this.breakdown.filter(d => d !== item.id);
          setHead(head, this.breakdown.length ? this.breakdown.join(', ') : 'none');
          this.refreshView();
          this.refreshTable();
        },
      }],
    }).openAt(head);
  }

  openGroupPicker(head) {
    const dims = this.dims();
    listMenu('group by', dims.map(d => ({id: d, label: d, on: d === this.tableState.group})), item => {
      this.tableState.group = this.tableState.group === item.id ? null : item.id;
      setHead(head, this.tableState.group || 'none');
      this.tableState.page = 0;
      this.refreshTable();
    }, {empty: 'no dimensions'}).openAt(head);
  }

  paintVerdict(verdict) {
    const box = document.getElementById(`${this.topic}-verdict`);
    box.classList.remove('hidden');
    box.classList.toggle('trusted', !!verdict.trusted);
    box.classList.toggle('warn', !verdict.trusted);
    box.innerHTML = '';
    const summary = document.createElement('div');
    summary.className = 'forecast-verdict-summary';
    summary.textContent = verdict.summary || (verdict.trusted ? 'trusted' : 'not trusted');
    box.appendChild(summary);
    (verdict.checks || []).filter(c => !c.passed).forEach(check => {
      const line = document.createElement('div');
      line.className = 'forecast-verdict-check';
      line.textContent = `${check.check}: ${check.detail}`;
      box.appendChild(line);
    });
  }

  async refreshView() {
    if (!this.resultsRunId) return;
    const params = new URLSearchParams({id: this.resultsRunId});
    if (Object.keys(this.filters).length) params.set('filters', JSON.stringify(this.filters));
    if (this.breakdown.length) params.set('breakdown', this.breakdown.join(','));
    try {
      const view = await api(`/api/forecast-view?${params.toString()}`);
      this.paintVerdict(view.verdict);
      this.paintChart(view);
      document.getElementById(`${this.topic}-group-row`).classList.toggle('hidden', view.mode !== 'row');
    } catch (err) {
      this.paintVerdict({trusted: false, summary: err.message, checks: []});
    }
  }

  paintChart(view) {
    const el = document.getElementById(`${this.topic}-chart`);
    if (!this.chart) this.chart = timeChart(el, {height: el.clientHeight || 220});
    this.chart.update({x: view.x, series: view.series, bands: view.bands, markers: view.markers});
  }

  async refreshTable() {
    if (!this.resultsRunId) return;
    const params = new URLSearchParams({
      id: this.resultsRunId, page: String(this.tableState.page), size: String(FORECAST_PAGE_SIZE),
    });
    if (Object.keys(this.filters).length) params.set('filters', JSON.stringify(this.filters));
    if (this.breakdown.length) params.set('breakdown', this.breakdown.join(','));
    if (this.tableState.sort) {
      params.set('sort', this.tableState.sort);
      if (this.tableState.desc) params.set('desc', '1');
    }
    if (this.tableState.group) params.set('group', this.tableState.group);
    const data = await api(`/api/forecast-table?${params.toString()}`);
    this.paintTable(data);
  }

  paintTable(data) {
    const table = document.getElementById(`${this.topic}-results-table`);
    table.innerHTML = '';
    const head = document.createElement('tr');
    data.columns.forEach(col => {
      const th = document.createElement('th');
      th.textContent = col;
      th.classList.add('sortable');
      th.onclick = () => {
        this.tableState.desc = this.tableState.sort === col ? !this.tableState.desc : false;
        this.tableState.sort = col;
        this.refreshTable();
      };
      head.appendChild(th);
    });
    table.appendChild(head);
    data.rows.forEach(row => {
      const tr = document.createElement('tr');
      row.forEach(value => {
        const td = document.createElement('td');
        td.textContent = value === null || value === undefined ? '' : String(value);
        tr.appendChild(td);
      });
      table.appendChild(tr);
    });
    const pages = Math.max(1, Math.ceil(data.total / FORECAST_PAGE_SIZE));
    setText(`${this.topic}-page-info`, `page ${data.page + 1} / ${pages} - ${data.total} rows`);
  }

  changePage(delta) {
    this.tableState.page = Math.max(0, this.tableState.page + delta);
    this.refreshTable();
  }
}

function textSpan(className, text) {
  const span = document.createElement('span');
  span.className = className;
  span.textContent = text;
  return span;
}

['regression', 'classification'].forEach(topic => {
  const ctl = new ForecastTopic(topic, topic);
  smolsmortTabs.register({id: `${topic}-data`, label: 'data', topic, mount: p => ctl.mountData(p)});
  smolsmortTabs.register({id: `${topic}-search`, label: 'search', topic, mount: p => ctl.mountSearch(p)});
  smolsmortTabs.register({id: `${topic}-results`, label: 'results', topic, mount: p => ctl.mountResults(p)});
  const flavours = topic === 'regression'
    ? [{id: 'row', label: 'xgboost - per row'}, {id: 'series', label: 'xgboost - time series'}]
    : [{id: 'row', label: 'xgboost - per row'}];
  smolsmortTabs.setFlavours(topic, flavours, 'row');
  smolsmortTabs.onFlavour(topic, () => ctl.onFlavourChange());
});
