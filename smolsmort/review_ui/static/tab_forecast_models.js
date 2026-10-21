// each family selection is built here so advanced space overrides can join the same request
function forecastBuildFamilySelection(family) {
  return {name: family.name, method: 'genetic'};
}

class ForecastModels {
  constructor(ctl) {
    this.ctl = ctl;
    this.prefix = `${ctl.topic}-models`;
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
      // parallel is the default; only an explicit off is remembered as off
      parallel: saved.parallel === true,
      atOnce: Number.isInteger(saved.atOnce) && saved.atOnce >= 1 ? saved.atOnce : 4,
    };
  }

  mount(panel) {
    this.panel = panel;
    panel.innerHTML = `
      <div class="panel-inline forecast-pane forecast-models">
        <div class="run-controls">
          <span class="field-label">families</span>
          <button type="button" class="toggle" id="${this.prefix}-all-suitable">all suitable</button>
        </div>
        <div class="forecast-pane forecast-model-families" id="${this.prefix}-families"></div>
        <div class="run-controls">
          <label class="field-label" for="${this.prefix}-minutes">duration (minutes)</label>
          <input class="text-field" type="number" id="${this.prefix}-minutes" min="1" max="1440" step="any" required>
        </div>
        <div class="run-controls">
          <button type="button" class="toggle" role="checkbox" id="${this.prefix}-parallel">run families in parallel</button>
          <label class="field-label" for="${this.prefix}-at-once">at once</label>
          <input class="text-field" type="number" id="${this.prefix}-at-once" min="1" step="1" required>
        </div>
        <div class="run-controls">
          <button type="button" class="toggle adds" id="${this.prefix}-start">start</button>
          <span class="stat" id="${this.prefix}-summary"></span>
        </div>
        <div class="stat" id="${this.prefix}-status" aria-live="polite"></div>
      </div>`;
    this.minutes = document.getElementById(`${this.prefix}-minutes`);
    this.atOnce = document.getElementById(`${this.prefix}-at-once`);
    this.parallel = document.getElementById(`${this.prefix}-parallel`);
    this.start = document.getElementById(`${this.prefix}-start`);
    this.preset = document.getElementById(`${this.prefix}-all-suitable`);
    this.minutes.value = this.plan.minutes;
    this.minutes.oninput = () => {
      if (this.minutes.checkValidity()) {
        this.plan.minutes = this.minutes.valueAsNumber;
        this.ctl.saveStored();
      }
      this.paintStart();
    };
    this.atOnce.oninput = () => {
      if (this.atOnce.checkValidity()) {
        this.plan.atOnce = this.atOnce.valueAsNumber;
        this.ctl.saveStored();
      }
      this.paintStart();
    };
    this.parallel.onclick = () => {
      this.plan.parallel = !this.plan.parallel;
      this.ctl.saveStored();
      this.paintControls();
    };
    this.preset.onclick = () => {
      this.plan.families = this.catalog.filter(family => this.canSelect(family)).map(family => family.name);
      this.ctl.saveStored();
      this.paintFamilies();
    };
    this.start.onclick = () => this.startRun();
    this.paintControls();
    return {enter: () => this.enter()};
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
      this.catalog = res.families.filter(family => mode !== 'row' || family.name === 'xgboost');
      this.say(reason || (mode === 'row' ? 'per-row runs support xgboost only' : ''));
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

  paintFamilies() {
    const box = document.getElementById(`${this.prefix}-families`);
    box.replaceChildren();
    this.catalog.forEach(family => {
      const enabled = this.canSelect(family);
      const selected = enabled && this.plan.families.includes(family.name);
      const row = forecastNode('div', 'run-controls forecast-model-family');
      row.dataset.family = family.name;
      const check = forecastNode('button', `toggle${selected ? ' on' : ''}${enabled ? '' : ' disabled'}`, family.label);
      check.type = 'button';
      check.setAttribute('role', 'checkbox');
      check.setAttribute('aria-checked', String(selected));
      check.disabled = !enabled;
      check.onclick = () => {
        this.plan.families = selected ? this.plan.families.filter(name => name !== family.name)
          : [...this.plan.families, family.name];
        this.ctl.saveStored();
        this.paintFamilies();
        box.querySelector(`[data-family="${family.name}"] button`).focus();
      };
      row.append(check, forecastNode('span', 'field-value', family.available ? 'library ready' : 'missing library'));
      const reason = family.available ? family.reason : family.install_command || family.reason;
      const note = forecastNode('span', 'stat forecast-model-reason', reason || '');
      note.title = family.reason || reason || '';
      note.id = `${this.prefix}-${family.name}-reason`;
      check.setAttribute('aria-describedby', note.id);
      row.append(note);
      row.classList.toggle('unavailable', !family.available);
      box.append(row);
    });
    this.paintControls();
  }

  paintControls() {
    const count = this.selectedFamilies().length;
    this.atOnce.max = Math.max(1, count);
    this.atOnce.value = Math.min(this.plan.atOnce, Math.max(1, count));
    this.atOnce.disabled = !this.plan.parallel;
    this.parallel.classList.toggle('on', this.plan.parallel);
    this.parallel.setAttribute('aria-checked', String(this.plan.parallel));
    this.preset.disabled = this.loading || !this.catalog.some(family => this.canSelect(family));
    this.preset.classList.toggle('disabled', this.preset.disabled);
    this.paintStart();
  }

  paintStart() {
    const count = this.selectedFamilies().length;
    this.start.disabled = this.loading || this.starting || !count
      || !this.minutes.checkValidity() || !this.atOnce.checkValidity();
    this.start.classList.toggle('disabled', this.start.disabled);
    setText(`${this.prefix}-summary`, `${count} ${count === 1 ? 'family' : 'families'} selected`);
  }

  async startRun() {
    if (this.start.disabled || this.starting) return;
    this.starting = true;
    this.paintStart();
    this.say('starting...');
    try {
      const spec = this.ctl.currentSpec();
      if (JSON.stringify(spec) !== this.catalogSpec) throw new Error('data changed - reopen models to check families');
      const payload = {
        version: 2, spec,
        families: this.selectedFamilies().map(forecastBuildFamilySelection),
        time_budget_s: this.minutes.valueAsNumber * 60,
        max_concurrent_families: this.plan.parallel ? this.atOnce.valueAsNumber : 1,
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

function registerForecastModels(ctl) {
  ctl.models = new ForecastModels(ctl);
  smolsmortTabs.register({id: `${ctl.topic}-models`, label: 'models', topic: ctl.topic,
    mount: panel => ctl.models.mount(panel)});
}
