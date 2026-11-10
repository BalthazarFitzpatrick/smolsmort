// each family selection is built here so advanced space overrides can join the same request
function forecastBuildFamilySelection(family) {
  return {name: family.name, method: 'genetic'};
}

class ForecastTrainingConfig {
  constructor(ctl) {
    this.ctl = ctl;
    this.prefix = `${ctl.topic}-config`;
    this.catalog = [];
    this.catalogRequest = 0;
    this.catalogSpec = null;
    this.loading = false;
    this.starting = false;
    const saved = ctl.modelPlan || {};
    this.plan = ctl.modelPlan = {
      families: Array.isArray(saved.families)
        ? [...new Set(saved.families.filter(name => typeof name === 'string'))] : ['xgboost'],
      minutes: Number.isFinite(saved.minutes) && saved.minutes >= 1 && saved.minutes <= 1440
        ? saved.minutes : 20,
      atOnce: saved.parallel === false ? 1 : Number.isInteger(saved.atOnce) && saved.atOnce >= 1 ? saved.atOnce : 4,
      population: saved.population || 16,
      patience: saved.patience || 5,
      generations: saved.generations || 60,
    };
  }

  mount(panel) {
    this.panel = panel;
    this.prefix = `${this.ctl.topic}-config`;
    this.mode = 'new';
    this.heads = {};
    panel.id = this.prefix;
    panel.tabIndex = -1;
    const field = (key, label) => `<div class="forecast-config-field"><span class="field-label">${label}</span><span id="${this.prefix}-${key}-slot"></span></div>`;
    panel.innerHTML = `
      <div class="run-controls" role="group" aria-label="mode">
        <button class="toggle" id="${this.prefix}-mode-new">new run</button>
        <button class="toggle" id="${this.prefix}-mode-warm">from this run</button>
      </div>
      <div class="forecast-config-row">${field('families', 'families')}${field('minutes', 'time cap')}${field('atOnce', 'run at once')}</div>
      <div class="forecast-config-row">${field('population', 'population')}${field('patience', 'patience')}${field('generations', 'max generations')}</div>
      <div class="run-controls" id="${this.prefix}-new-actions"><button class="toggle adds" id="${this.prefix}-start">start run</button><span class="stat" id="${this.prefix}-summary"></span></div>
      <div class="run-controls hidden" id="${this.ctl.topic}-run-actions">
        <button class="toggle" id="${this.ctl.topic}-run-warm">warm start</button>
        <button class="toggle" id="${this.ctl.topic}-run-refit">refit</button>
        <button class="toggle" id="${this.ctl.topic}-run-open">open</button>
        <span class="stat">families come from the picked run</span>
      </div><div class="stat" id="${this.prefix}-status" aria-live="polite"></div>`;
    for (const key of ['families', ...Object.keys(this.numbers())]) {
      const head = forecastNode('button', 'toggle dropdown-head');
      head.type = 'button'; head.id = `${this.prefix}-${key}`;
      head.setAttribute('aria-expanded', 'false');
      head.setAttribute('aria-label', key === 'atOnce' ? 'run at once' : key);
      head.append(forecastNode('span'));
      head.onclick = () => key === 'families' ? this.openFamilies() : this.openNumber(key);
      document.getElementById(`${this.prefix}-${key}-slot`).append(head);
      this.heads[key] = head;
    }
    for (const mode of ['new', 'warm']) document.getElementById(`${this.prefix}-mode-${mode}`).onclick = () => this.setMode(mode);
    this.start = document.getElementById(`${this.prefix}-start`);
    this.start.onclick = () => this.startRun();
    this.paintControls();
  }

  numbers() {
    return {
      minutes: {presets: [5,10,20,30,60,120,240,480,1440], min: 1, max: 1440},
      atOnce: {presets: [1,2,4,8,16], min: 1, max: 16},
      population: {presets: [10,20,40,80,160,320], min: 2, max: 400},
      patience: {presets: [3,5,8,12,20,40], min: 1, max: 100},
      generations: {presets: [20,40,60,100,200,500], min: 1, max: 500},
    };
  }

  values() { return this.mode === 'warm' ? this.warmValues : this.plan; }

  numberLabel(key, value) {
    return key === 'minutes' ? `${value} min` : key === 'atOnce'
      ? value === 1 ? 'one at a time' : `${value} at once` : String(value);
  }

  openNumber(key) {
    const cfg = this.numbers()[key];
    const apply = value => {
      this.values()[key] = value;
      this.plan[key] = value;
      this.syncBudget(); this.ctl.saveStored(); this.paintControls();
    };
    const options = [...new Set([...cfg.presets, this.values()[key]])].sort((a, b) => a - b);
    const menu = forecastOpenMenu(listMenu(this.heads[key].getAttribute('aria-label'), options.map(value => ({
      id: value, label: this.numberLabel(key, value), on: value === this.values()[key],
    })), item => apply(item.id)), this.heads[key]);
    const row = forecastNode('label', 'run-controls forecast-config-custom', 'custom');
    const input = forecastNode('input', 'text-field');
    input.type = 'number'; input.min = cfg.min; input.max = cfg.max;
    input.setAttribute('aria-label', 'custom value');
    input.onkeydown = event => {
      if (event.key !== 'Enter') return;
      event.preventDefault(); event.stopPropagation();
      if (!input.value || !input.checkValidity()) return;
      apply(Math.round(input.valueAsNumber)); menu.close();
    };
    row.append(input); menu.el.append(row);
  }

  openFamilies() {
    if (this.mode === 'warm') return;
    const draft = new Set(this.plan.families);
    const sections = () => [{kind: 'list', multi: true,
      items: [{id: 'all-suitable', label: 'all suitable', on: this.catalog.filter(f => this.canSelect(f)).every(f => draft.has(f.name))},
        ...this.catalog.map(f => ({id: f.name, label: f.label || f.name, on: draft.has(f.name),
          disabled: !this.canSelect(f), stats: this.canSelect(f) ? '' : f.reason || f.install_command || 'unavailable'}))],
      onPick: (item, on, menu) => {
        if (item.id === 'all-suitable') this.catalog.filter(f => this.canSelect(f)).forEach(f => draft.add(f.name));
        else if (this.catalog.some(f => f.name === item.id && this.canSelect(f))) {
          if (on) draft.add(item.id); else draft.delete(item.id);
        }
        menu.refresh(sections());
      },
    }, {kind: 'buttons', buttons: [
      {id: 'save', label: 'save', tone: 'adds', primary: true, onClick: menu => {
        this.plan.families = [...draft]; this.ctl.saveStored(); menu.close(); this.paintControls();
      }},
      {id: 'close', label: 'close', onClick: menu => menu.close()},
    ]}];
    forecastOpenMenu(new Menu({title: 'model families', persistent: true, sections: sections()}), this.heads.families);
  }

  setMode(mode) {
    const run = this.ctl.runCache.get(this.ctl.selectedRunId);
    if (mode === 'warm' && !run) return;
    this.mode = mode;
    if (mode === 'warm') { this.loadRun(run); this.loadedRun = this.ctl.selectedRunId; }
    this.paintControls();
  }

  loadRun(run) {
    const budget = run.budget || {};
    this.loadedRunConfig = this.runConfigSignature(run);
    this.warmValues = {
      minutes: (run.time_budget_s || budget.time_cap || 1200) / 60,
      atOnce: run.max_concurrent_families || 1,
      population: run.population || budget.population || 16,
      patience: run.plateau || budget.plateau || 5,
      generations: run.max_generations || budget.max_generations || 60,
    };
    this.syncBudget();
  }

  runConfigSignature(run) {
    return JSON.stringify([run.budget, run.time_budget_s, run.max_concurrent_families,
      run.population, run.plateau, run.max_generations]);
  }

  syncBudget() {
    const values = this.values();
    this.ctl.searchValues = {population: values.population, plateau: values.patience,
      generations: values.generations, minutes: values.minutes};
  }

  paintFamilies() { this.paintControls(); }

  paintControls() {
    if (!this.heads) return;
    const run = this.ctl.runCache.get(this.ctl.selectedRunId);
    if (this.mode === 'warm' && !run) this.mode = 'new';
    if (this.mode === 'warm' && (this.loadedRun !== this.ctl.selectedRunId
        || this.loadedRunConfig !== this.runConfigSignature(run))) {
      this.loadRun(run); this.loadedRun = this.ctl.selectedRunId;
    }
    const warm = this.mode === 'warm';
    const runFamilies = Array.isArray(run?.families)
      ? run.families.map(f => typeof f === 'string' ? f : f.name) : Object.keys(run?.families || {});
    const families = warm ? runFamilies : this.selectedFamilies().map(f => f.name);
    if (warm && !families.length) families.push('xgboost');
    setHead(this.heads.families, families.length ? families.join(' + ') : 'none picked');
    this.heads.families.disabled = warm;
    for (const key of Object.keys(this.numbers())) setHead(this.heads[key], this.numberLabel(key, this.values()[key]));
    for (const mode of ['new', 'warm']) {
      const button = document.getElementById(`${this.prefix}-mode-${mode}`);
      button.classList.toggle('on', mode === this.mode);
      button.setAttribute('aria-pressed', String(mode === this.mode));
      button.disabled = mode === 'warm' && !run;
    }
    document.getElementById(`${this.prefix}-new-actions`).classList.toggle('hidden', warm);
    document.getElementById(`${this.ctl.topic}-run-actions`).classList.toggle('hidden', !warm);
    this.paintStart();
  }

  paintStart() {
    if (!this.start) return;
    const count = this.selectedFamilies().length;
    this.start.disabled = this.loading || this.starting || !count;
    setText(`${this.prefix}-summary`, count ? `${count} ${count === 1 ? 'family' : 'families'} · ${this.plan.minutes} min` : 'pick at least one family');
  }

  say(text) { setText(`${this.prefix}-status`, text); }

  canSelect(family) { return family.available && family.suitable === true; }

  selectedFamilies() {
    return this.catalog.filter(family => this.canSelect(family) && this.plan.families.includes(family.name));
  }

  clearCatalog() {
    this.catalogRequest += 1;
    this.catalog = [];
    this.catalogSpec = null;
    this.loading = true;
    this.paintFamilies();
  }

  async enter() {
    this.clearCatalog();
    const request = this.catalogRequest;
    const mode = this.ctl.mode();
    this.say('checking families...');
    try {
      if (this.ctl.source && !this.ctl.columnsInfo.length) await this.ctl.loadColumns();
      if (request !== this.catalogRequest) return;
      let spec = null, reason = '';
      try { spec = this.ctl.currentSpec(); } catch (err) { reason = err.message; }
      const params = new URLSearchParams(spec ? {spec: JSON.stringify(spec)} : {task: this.ctl.task});
      const res = await api(`/api/forecast-families?${params}`);
      // a slow catalog for an earlier flavour must never replace the current choices
      if (request !== this.catalogRequest) return;
      this.catalogSpec = spec ? JSON.stringify(spec) : null;
      this.catalog = res.families.filter(family => mode !== 'entity' || family.name === 'xgboost');
      this.say(reason || (mode === 'entity' ? 'entity runs support xgboost only' : ''));
    } catch (err) {
      if (request !== this.catalogRequest) return;
      this.say(err.message);
    } finally {
      if (request === this.catalogRequest) {
        this.loading = false;
        this.paintFamilies();
      }
    }
  }

  async startRun() {
    if (this.start.disabled || this.starting) return;
    this.starting = true;
    this.paintStart();
    this.say('starting...');
    try {
      const spec = this.ctl.currentSpec();
      if (JSON.stringify(spec) !== this.catalogSpec) throw new Error('data changed - reopen training to check families');
      const payload = {
        version: 2, spec,
        families: this.selectedFamilies().map(forecastBuildFamilySelection),
        time_budget_s: this.plan.minutes * 60,
        max_concurrent_families: this.plan.atOnce,
        plateau: this.plan.patience, population: this.plan.population,
        max_generations: this.plan.generations,
      };
      this.ctl.saveStored();
      const res = await api('/api/forecast-runs', payload);
      if (res.prepared) this.ctl.rememberPrepared(spec, res.prepared);
      this.ctl.attachRun(res.run_id, 'search');
      if (forecastActiveTopic === this.ctl.topic) activateTab(`${this.ctl.topic}-search`);
      this.say('run started');
    } catch (err) {
      this.say(err.message);
    } finally {
      this.starting = false;
      this.paintStart();
    }
  }
}

function registerForecastConfig(ctl) {
  ctl.models = new ForecastTrainingConfig(ctl);
}
