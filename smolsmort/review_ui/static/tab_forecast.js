// forecast: the regression and classification topics share this one set of tabs, parameterised by
// topic/task. the flavour dropdown picks spec.mode; the topic itself picks spec.task

const FORECAST_ROLES = {
  series: ['time', 'dimension', 'measure', 'target', 'ignore'],
  row: ['anchor', 'dimension', 'measure', 'target', 'ignore'],
};
const FORECAST_AGGREGATIONS = ['sum', 'mean', 'min', 'max', 'count', 'last'];
const FORECAST_STEPS = ['auto', 'day', 'week', 'month', 'quarter', 'year'];
const FORECAST_UNITS = ['day', 'week', 'month', 'quarter', 'year'];
const FORECAST_PAGE_SIZE = 50;

function todayIso() {
  return new Date().toISOString().slice(0, 10);
}

function forecastSelectEl(id, options, selected, extraClass) {
  const opts = options.map(o => {
    const value = typeof o === 'string' ? o : o.id;
    const label = typeof o === 'string' ? o : o.label;
    const sel = value === selected ? ' selected' : '';
    return `<option value="${value}"${sel}>${label}</option>`;
  }).join('');
  return `<select class="field-select ${extraClass || ''}" id="${id}">${opts}</select>`;
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
    this.encoding = 'utf-8';
    this.columnsInfo = [];
    this.columnState = {}; // name -> {role, aggregation, known}
    this.step = 'auto';
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
    this.resultsRunId = null;
    this.resultsSpec = null;
    this.chart = null;
    this.filters = {};
    this.breakdown = [];
    this.tableState = {page: 0, sort: null, desc: false, group: null};
    this.loadStored();
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
    } catch (err) { /* nothing saved, or it did not parse - start clean */ }
  }

  saveStored() {
    try {
      const state = {
        source: this.source, encoding: this.encoding, columnState: this.columnState,
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
        ${forecastSelectEl(`${this.topic}-censor-unit`, FORECAST_UNITS, this.censorUnit)}
        <span class="field-label">as of</span>
        <input type="date" id="${this.topic}-censor-asof">
      </div>` : '';
    panel.innerHTML = `
      <div class="forecast-pane">
        <div class="run-controls">
          <div class="toggle dropdown-head grow" id="${this.topic}-data-source"><span>source</span></div>
          <span class="field-label">encoding</span>
          <input type="text" id="${this.topic}-data-encoding" class="forecast-encoding">
          <span class="stat" id="${this.topic}-data-root"></span>
        </div>
        <div id="${this.topic}-data-columns"></div>
        <div class="h-divider"></div>
        <div class="field-label">series settings</div>
        <div class="run-controls" id="${this.topic}-series-row">
          <span class="field-label">time step</span>
          ${forecastSelectEl(`${this.topic}-step`, FORECAST_STEPS, this.step)}
          <span class="field-label">horizon</span>
          <span class="field-value" id="${this.topic}-horizon">${this.horizon}</span>
          <div class="toggle" data-step="-1" data-target="horizon">-</div>
          <div class="toggle" data-step="1" data-target="horizon">+</div>
          <span class="field-label">as of</span>
          <input type="date" id="${this.topic}-asof">
        </div>
        <div class="field-label" id="${this.topic}-row-heading">row settings</div>
        <div id="${this.topic}-row-row">
          ${censorBlock}
          <div class="run-controls">
            <span class="field-label">rows to predict</span>
            <input type="text" id="${this.topic}-predict-where"
                   placeholder="sql filter, optional" class="grow">
          </div>
        </div>
        <div class="run-controls">
          <span class="field-label">filter</span>
          <input type="text" id="${this.topic}-where"
                 placeholder="sql filter over the source, optional" class="grow">
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
    if (heading) heading.textContent = series ? 'series settings' : 'row settings';
  }

  wireData(panel) {
    document.getElementById(`${this.topic}-data-source`).onclick = evt => this.openSourcePicker(evt.currentTarget);
    document.getElementById(`${this.topic}-data-encoding`).onchange = evt => {
      this.encoding = evt.currentTarget.value || 'utf-8';
      this.saveStored();
    };
    const step = document.getElementById(`${this.topic}-step`);
    if (step) step.onchange = () => { this.step = step.value; this.saveStored(); };
    const asOf = document.getElementById(`${this.topic}-asof`);
    if (asOf) asOf.onchange = () => { this.asOf = asOf.value; this.saveStored(); };
    const predictWhere = document.getElementById(`${this.topic}-predict-where`);
    if (predictWhere) predictWhere.onchange = () => { this.predictWhere = predictWhere.value; this.saveStored(); };
    const where = document.getElementById(`${this.topic}-where`);
    where.onchange = () => { this.where = where.value; this.saveStored(); };
    panel.querySelectorAll('[data-target="horizon"]').forEach(btn => {
      btn.onclick = () => {
        this.horizon = Math.max(1, this.horizon + Number(btn.dataset.step));
        setText(`${this.topic}-horizon`, String(this.horizon));
        this.saveStored();
      };
    });
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
      unit.onchange = () => { this.censorUnit = unit.value; this.saveStored(); };
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
      // keep any per-column choice still valid for this mode; suggest for anything new
      const roles = FORECAST_ROLES[this.mode()];
      const next = {};
      this.columnsInfo.forEach(col => {
        const prior = this.columnState[col.name];
        const role = prior && roles.includes(prior.role) ? prior.role : col.suggested_role;
        next[col.name] = {
          role,
          aggregation: (prior && prior.aggregation) || FORECAST_AGGREGATIONS[0],
          known: prior ? prior.known !== false : true,
        };
      });
      this.columnState = next;
      this.renderColumnGrid();
      this.updateCensorAnchor();
      this.dataSay(`${this.columnsInfo.length} columns`);
      this.saveStored();
    } catch (err) {
      this.dataSay(err.message);
    }
  }

  renderColumnGrid() {
    const box = document.getElementById(`${this.topic}-data-columns`);
    box.innerHTML = '';
    if (!this.columnsInfo.length) return;
    const series = this.mode() === 'series';
    const table = document.createElement('table');
    table.className = 'forecast-column-grid';
    const head = document.createElement('tr');
    ['name', 'kind', 'samples', 'role', series ? 'aggregation' : 'known when predicting']
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
      const roleSelect = document.createElement('select');
      roleSelect.className = 'field-select';
      FORECAST_ROLES[this.mode()].forEach(role => {
        const opt = document.createElement('option');
        opt.value = role;
        opt.textContent = role;
        opt.selected = role === state.role;
        roleSelect.appendChild(opt);
      });
      roleSelect.onchange = () => {
        this.columnState[col.name] = {...this.columnState[col.name], role: roleSelect.value};
        this.updateCensorAnchor();
        this.saveStored();
      };
      roleCell.appendChild(roleSelect);
      const lastCell = document.createElement('td');
      if (series) {
        const aggSelect = document.createElement('select');
        aggSelect.className = 'field-select';
        const relevant = ['measure', 'target'].includes(state.role);
        aggSelect.disabled = !relevant;
        FORECAST_AGGREGATIONS.forEach(agg => {
          const opt = document.createElement('option');
          opt.value = agg;
          opt.textContent = agg;
          opt.selected = agg === state.aggregation;
          aggSelect.appendChild(opt);
        });
        aggSelect.onchange = () => {
          this.columnState[col.name] = {...this.columnState[col.name], aggregation: aggSelect.value};
          this.saveStored();
        };
        lastCell.appendChild(aggSelect);
      } else {
        const check = document.createElement('input');
        check.type = 'checkbox';
        check.checked = state.known !== false;
        check.disabled = !['dimension', 'measure'].includes(state.role);
        check.onchange = () => {
          this.columnState[col.name] = {...this.columnState[col.name], known: check.checked};
          this.saveStored();
        };
        lastCell.appendChild(check);
      }
      row.append(nameCell, kindCell, samplesCell, roleCell, lastCell);
      table.appendChild(row);
    });
    box.appendChild(table);
  }

  buildSpec() {
    const mode = this.mode();
    const series = mode === 'series';
    const columns = Object.entries(this.columnState).map(([name, state]) => ({
      name,
      role: state.role,
      aggregation: series && ['measure', 'target'].includes(state.role) ? state.aggregation : null,
      known: series ? true : state.known !== false,
    }));
    const anchorEntry = columns.find(c => c.role === 'anchor');
    return {
      source: this.source,
      mode,
      task: this.task,
      columns,
      encoding: this.encoding || 'utf-8',
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
        <div id="${this.topic}-leaderboard" class="hidden">
          <div class="field-label">leaderboard</div>
          <table class="forecast-column-grid" id="${this.topic}-leaderboard-table"></table>
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
    if (!this.currentRunId) return;
    document.getElementById(`${this.topic}-search-cancel`).classList.remove('disabled');
    const state = await api(`/api/forecast-run?id=${encodeURIComponent(this.currentRunId)}`);
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

  paintLeaderboard(rows) {
    const wrap = document.getElementById(`${this.topic}-leaderboard`);
    const table = document.getElementById(`${this.topic}-leaderboard-table`);
    table.innerHTML = '';
    wrap.classList.toggle('hidden', !rows.length);
    if (!rows.length) return;
    const columns = Object.keys(rows[0]);
    const head = document.createElement('tr');
    columns.forEach(c => {
      const th = document.createElement('th');
      th.textContent = c;
      head.appendChild(th);
    });
    table.appendChild(head);
    rows.slice(0, 10).forEach(entry => {
      const row = document.createElement('tr');
      columns.forEach(c => {
        const td = document.createElement('td');
        td.textContent = String(entry[c]);
        row.appendChild(td);
      });
      table.appendChild(row);
    });
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
