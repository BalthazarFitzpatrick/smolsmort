// select: judge tiles into classes. assignments are buffered on the server and written only when
// save is pressed; closing a box set with unsaved assignments asks first

smolsmortTabs.register({id: 'select', label: 'select', mount: mountSelect});

const DIM_SEP = ' / ';

function mountSelect(panel) {
  panel.innerHTML = `
    <div id="clusters-bar" class="run-controls">
      <div class="toggle dropdown-head" id="cluster-open-set"><span>boxes</span></div>
      <div class="toggle dropdown-head" id="cluster-classdef"><span>classes</span></div>
      <div class="toggle" id="cluster-select-all" title="select every tile shown (cmd/ctrl+A)">select all</div>
      <div class="toggle disabled" id="cluster-clear-sel" title="clear the selection (Escape)">clear</div>
      <div class="toggle" id="recluster" title="read the tiles again">reload</div>
      <span class="stat" id="boxes-loaded">0 boxes loaded</span>
      <span class="stat" id="classes-assigned">0 classes assigned</span>
      <span class="spacer"></span>
      <div class="toggle adds disabled" id="cluster-save">save</div>
      <div class="toggle adds" id="cluster-promote">promote</div>
    </div>
    <div class="run-controls select-status">
      <span class="field-label">showing</span>
      <span class="field-value" id="clusters-count">0 / 0</span>
      <span class="stat" id="cluster-selected"></span>
      <span class="stat" id="clusters-status"></span>
    </div>
    <div class="h-divider"></div>
    <div id="cluster-filter" class="label-columns">
      <div id="filter-dims" class="label-columns"></div>
      <div class="divider"></div>
      <div class="col" id="filter-judged"></div>
    </div>
    <div class="h-divider"></div>
    <div id="cluster-groups"></div>`;
  const tab = new SelectTab();
  return {enter: () => tab.enter()};
}

class SelectTab {
  constructor() {
    this.items = [];
    this.visible = [];
    this.classDef = null;
    this.pending = 0;
    this.pendingNames = new Map();   // tile name -> what it was given, until saved
    this.bust = Date.now();
    this.grid = document.getElementById('cluster-groups');
    this.sel = makeSelection(this.grid, {
      itemSelector: '.cluster-item',
      onChange: names => this.showSelection(names),
      onContext: (names, x, y) => this.openLabelPopup(names, x, y),
    });
    this.wire();
  }

  async enter() {
    await this.loadClassDef();
    await this.load();
  }

  status(text) { setText('clusters-status', text); }

  // ---- data ----
  async load() {
    this.bust = Date.now();
    const data = await api('/api/clusters');
    this.items = data.clusters.flatMap(cluster =>
      cluster.items.map(item => ({...item, clusterLabel: cluster.label})));
    setText('boxes-loaded', `${data.boxes_loaded ?? this.items.length} boxes loaded`);
    setText('classes-assigned', `${data.classes_assigned ?? 0} classes assigned`);
    await this.refreshPending();
    this.renderFilter();
    this.renderGrid();
  }

  async loadClassDef(name) {
    const list = await api('/api/classdefs');
    const wanted = name || list.active;
    this.classDef = null;
    if (wanted) {
      const def = await api(`/api/classdefs?name=${encodeURIComponent(wanted)}`);
      this.classDef = def.error ? null : def;
    }
    document.getElementById('cluster-classdef').innerHTML =
      `<span>${this.classDef ? '' : 'classes'}</span>`;
    if (this.classDef) document.querySelector('#cluster-classdef span').textContent = this.classDef.name;
  }

  async refreshPending() {
    const data = await api('/api/labels-buffer');
    this.setPending(data.pending || 0);
  }

  setPending(count) {
    this.pending = count;
    if (!count) this.pendingNames.clear();
    const save = document.getElementById('cluster-save');
    save.textContent = count ? `save (${count})` : 'save';
    save.classList.toggle('disabled', count === 0);
  }

  async saveLabels() {
    if (!this.pending) return;
    const res = await api('/api/save-labels', {});
    if (res.error) { this.status(res.error); return; }
    this.setPending(0);
    await this.load();
    this.status(`saved ${res.saved} class assignments`);
  }

  // ---- filter and grid ----
  parts(label) {
    return label ? label.replace(/\s*\?$/, '').split(DIM_SEP).map(p => p.trim()) : [];
  }

  toggleOn(selector) {
    return [...document.querySelectorAll(selector)]
      .filter(el => el.classList.contains('on')).map(el => el.dataset.name);
  }

  // no tick on a dimension means it does not filter; a partial tick restricts only that dimension
  matches(item) {
    const parts = this.parts(item.clusterLabel);
    const dims = this.classDef ? this.classDef.dimensions : [];
    for (const [index] of dims.entries()) {
      const picked = this.toggleOn(`#filter-dim-${index} .toggle`);
      if (picked.length && !picked.includes(parts[index])) return false;
    }
    if (this.toggleOn('#filter-judged .toggle').length && !item.excluded) return false;
    return true;
  }

  filterToggle(container, name) {
    const btn = document.createElement('div');
    btn.className = 'toggle label-option';
    btn.textContent = name;
    btn.dataset.name = name;
    btn.onclick = () => { btn.classList.toggle('on'); this.renderGrid(); };
    container.appendChild(btn);
  }

  renderFilter() {
    const dimsEl = document.getElementById('filter-dims');
    dimsEl.innerHTML = '';
    const judged = document.getElementById('filter-judged');
    judged.innerHTML = '';
    const dims = this.classDef ? this.classDef.dimensions : [];
    dims.forEach((dimension, index) => {
      if (index) dimsEl.insertAdjacentHTML('beforeend', '<div class="divider"></div>');
      const col = document.createElement('div');
      col.className = 'col';
      col.id = `filter-dim-${index}`;
      dimsEl.appendChild(col);
      dimension.members.forEach(member => this.filterToggle(col, member));
    });
    if (!dims.length) {
      dimsEl.innerHTML = '<span class="none">no class definition yet - make one under classes</span>';
    }
    this.filterToggle(judged, 'not a class');
  }

  renderGrid() {
    this.grid.innerHTML = '';
    if (!this.items.length) {
      this.status('no boxes loaded - open a set under boxes');
      setText('clusters-count', '0 / 0');
      return;
    }
    // stable partition: tiles already marked "not a class" sink below those still waiting
    this.visible = this.items.filter(item => this.matches(item))
      .sort((a, b) => (a.excluded ? 1 : 0) - (b.excluded ? 1 : 0));
    setText('clusters-count', `${this.visible.length} / ${this.items.length}`);
    this.status('');
    this.visible.forEach(item => {
      const cell = document.createElement('div');
      const given = this.pendingNames.get(item.name);
      const verdict = given ? 'is-pending'
        : item.excluded ? 'is-negative' : (item.assigned ? 'is-class' : 'is-unset');
      cell.className = 'cluster-item' + (item.excluded ? ' excluded' : '');
      cell.dataset.name = item.name;
      cell.innerHTML = '<div class="tile-viewport"><img loading="lazy" decoding="async"></div>'
        + `<span class="tile-dot ${verdict}"></span>`;
      cell.querySelector('img').src = `/unsorted-thumb/${item.name}?v=${this.bust}`;
      const stands = given || (item.excluded ? 'not a class' : (item.assigned || 'unjudged'));
      cell.title = `${item.name} - ${stands}\nclick to pick, cmd/ctrl+click to add\n`
        + 'shift+drag a net over several, right-click to tag every picked';
      this.grid.appendChild(cell);
    });
    this.sel.repaint();
  }

  showSelection(names) {
    setText('cluster-selected', names.length ? `${names.length} selected` : '');
    document.getElementById('cluster-clear-sel').classList.toggle('disabled', !names.length);
  }

  // ---- the class picker that opens on right-click ----
  openLabelPopup(names, x, y) {
    const dims = this.classDef ? this.classDef.dimensions : [];
    const picks = {};
    let menu = null;
    const sections = () => {
      if (!dims.length) {
        return [{kind: 'list', empty: 'no class definition yet - make one under classes', items: []}];
      }
      return [
        {kind: 'columns', columns: dims.map(dimension => ({
          label: dimension.name, empty: 'no members yet - writes n/a',
          items: dimension.members.map(member => ({
            id: member, label: member, on: picks[dimension.name] === member,
          })),
          onPick: item => { picks[dimension.name] = item.id; menu.refresh(sections()); },
        }))},
        {kind: 'buttons', buttons: [
          {id: 'negative', label: 'not a class', tone: 'removes', onClick: () => this.assign(names, {}, true, menu)},
          {id: 'apply', label: 'apply', tone: 'adds', enabled: Object.keys(picks).length > 0,
            onClick: () => this.assign(names, picks, false, menu)},
        ]},
      ];
    };
    menu = new Menu({
      title: `${names.length} tile${names.length === 1 ? '' : 's'}`, persistent: true,
      sections: sections(), onDismiss: () => this.sel.clear(),
    });
    menu.openAt({x, y});
  }

  // buffers only: the server answers with how many assignments now wait for save
  async assign(names, picked, excluded, menu) {
    try {
      const res = await api('/api/manual-label', {
        names, definition: this.classDef ? this.classDef.name : null, picked, excluded,
      });
      if (res.error) { this.status(res.error); return; }
      const given = excluded ? 'not a class' : Object.values(picked).join(DIM_SEP);
      names.forEach(name => this.pendingNames.set(name, given));
      this.setPending(res.pending);
      this.status(`${names.length} buffered - press save to write them`);
    } finally {
      menu.close();
      this.renderGrid();
    }
  }

  // ---- boxes: open sets into the pool, close them (asking first when work is unsaved) ----
  async closeOne(tag) {
    let res = await api('/api/close-source', {tag});
    if (res.unsaved > 0) {
      const answer = await askDialog({
        title: `close ${tag}`,
        lines: [`you have ${res.unsaved} unsaved class assignments`],
        buttons: [
          {id: 'save-close', label: 'save and close', tone: 'adds'},
          {id: 'discard-close', label: 'discard and close', tone: 'removes'},
          {id: 'cancel', label: 'cancel'},
        ],
      });
      if (answer === 'cancel') return;
      if (answer === 'save-close') {
        await api('/api/save-labels', {});
        res = await api('/api/close-source', {tag});
      } else {
        res = await api('/api/close-source', {tag, discard: true});
      }
    }
    await this.refreshPending();
    await this.load();
    this.status(`closed ${tag}`);
  }

  async openClusterSets() {
    const chosen = new Map();
    let menu = null;
    const build = async () => {
      const [openable, pool] = await Promise.all([api('/api/openable-datasets'), api('/api/pool-sources')]);
      const openTags = new Set(pool.sources.map(src => src.tag));
      const live = pool.sources.map(src => ({
        id: src.tag, label: src.tag, stats: `${src.tiles} tiles`, close: !src.closed,
        reopen: src.closed, state: src.closed ? {} : {opened: true}, source: src.source || 'drawn',
      }));
      const shut = openable.datasets.filter(d => !openTags.has(d.tag)).map(d => ({
        id: d.file, label: d.tag, stats: `${d.boxes} boxes`, source: d.source || 'drawn',
      }));
      const column = (label, source) => ({
        label, empty: 'none', multi: true,
        onPick: (item, on) => {
          if (on) chosen.set(item.id, item); else chosen.delete(item.id);
          menu.setButtonEnabled('open', [...chosen.values()].some(i => !i.close));
          menu.setButtonEnabled('close-set', [...chosen.values()].some(i => i.close));
        },
        items: [
          ...live.filter(i => i.source === source && !i.reopen),
          ...live.filter(i => i.source === source && i.reopen),
          ...shut.filter(i => i.source === source),
        ],
      });
      const sources = new Set([...live, ...shut].map(i => i.source));
      const columns = [column('drawn', 'drawn')];
      if ([...sources].some(s => s !== 'drawn')) {
        const other = [...sources].find(s => s !== 'drawn');
        columns.push(column('proposed', other));
      }
      return [
        {kind: 'columns', columns},
        {kind: 'buttons', buttons: [
          {id: 'open', label: 'open', tone: 'adds', enabled: false, onClick: async () => {
            for (const item of [...chosen.values()].filter(i => !i.close)) {
              const res = item.reopen ? await api('/api/open-source', {tag: item.id})
                : await api('/api/open-dataset', {name: item.id});
              if (res.error) this.status(res.error);
            }
            chosen.clear();
            await this.load();
            menu.refresh(await build());
          }},
          {id: 'close-set', label: 'close', tone: 'removes', enabled: false, onClick: async m => {
            const picked = [...chosen.values()].filter(i => i.close);
            chosen.clear();
            m.close();
            for (const item of picked) await this.closeOne(item.id);
          }},
        ]},
      ];
    };
    menu = await openPicker('cluster-open-set', build, {
      title: 'boxes - tick to open, tick an open one to close', status: 'clusters-status',
    });
  }

  // ---- class definitions ----
  async openClassPicker() {
    const build = async () => {
      const list = await api('/api/classdefs');
      return [
        {kind: 'list', empty: 'none yet', multi: false,
          items: (list.definitions || []).map(n => ({
            id: n, label: n, on: !!this.classDef && this.classDef.slug === n})),
          onPick: async item => { await this.loadClassDef(item.id); await this.load(); }},
        {kind: 'buttons', buttons: [
          {id: 'edit', label: this.classDef ? 'edit / new' : 'new', onClick: m => { m.close(); this.editClassDef(); }},
        ]},
      ];
    };
    await openPicker('cluster-classdef', build, {
      title: 'classes', persistent: false, status: 'clusters-status',
    });
  }

  editClassDef() {
    const draft = this.classDef
      ? {name: this.classDef.name, dimensions: this.classDef.dimensions.map(d => ({...d, members: [...d.members]}))}
      : {name: '', dimensions: []};
    let menu = null;
    const ask = text => (prompt(text) || '').trim();
    const rows = () => {
      const out = [];
      draft.dimensions.forEach((dimension, index) => {
        out.push({
          id: `dim:${index}`, label: dimension.name, dot: false, state: {removes: true},
          badges: ['remove'], title: `remove the "${dimension.name}" dimension`,
          onPick: () => { draft.dimensions.splice(index, 1); menu.refresh(sections()); },
        });
        dimension.members.forEach(member => out.push({
          id: `${index}:${member}`, label: member, dot: false, badges: ['remove'],
          title: `remove "${member}"`,
          onPick: () => {
            dimension.members = dimension.members.filter(m => m !== member);
            menu.refresh(sections());
          },
        }));
        out.push({
          id: `${index}:add`, label: '+ add member', dot: false, state: {adds: true},
          onPick: () => {
            const value = ask(`new member of ${dimension.name}`);
            if (value && !dimension.members.includes(value)) dimension.members.push(value);
            menu.refresh(sections());
          },
        });
      });
      if (draft.dimensions.length < 3) {
        out.push({
          id: 'add-dim', label: '+ add dimension', dot: false, state: {adds: true},
          onPick: () => {
            const value = ask('new dimension');
            if (value) draft.dimensions.push({name: value, members: []});
            menu.refresh(sections());
          },
        });
      }
      return out;
    };
    const sections = () => {
      const node = document.createElement('div');
      renderTree(node, rows(), {itemClass: 'classdef-item'});
      return [
        {kind: 'field', label: 'name', value: draft.name, placeholder: 'my classes',
          onInput: value => { draft.name = value; }},
        {kind: 'node', node},
        {kind: 'buttons', buttons: [{id: 'save', label: 'save', tone: 'adds', onClick: async () => {
          const res = await api('/api/classdef-save', draft);
          if (res.error) { this.status(res.error); return; }
          menu.close();
          await this.loadClassDef(res.name);
          await this.load();
        }}]},
      ];
    };
    menu = new Menu({title: 'class definition', sections: sections()});
    menu.openAt(headPoint('cluster-classdef'));
  }

  // ---- promote: every kept, labelled tile into one training set ----
  async promote(name, mode = 'new') {
    this.status(`promoting into ${name}...`);
    let res;
    try {
      res = await api('/api/promote-training', {name, mode});
    } catch (err) { this.status(`promote failed: ${err.message}`); return; }
    if (res.exists) {
      const answer = await askDialog({
        title: `${res.name} already exists`,
        lines: [`it holds ${res.existing_rows} rows, this promote has ${res.would_add} to add`],
        buttons: [
          {id: 'merge', label: 'merge into', tone: 'adds'},
          {id: 'overwrite', label: 'overwrite', tone: 'removes'},
          {id: 'cancel', label: 'cancel'},
        ],
      });
      if (answer === 'cancel') { this.status(`${res.name} left as it was`); return; }
      return this.promote(name, answer);
    }
    if (res.error) { this.status(res.error); return; }
    this.status(`${res.name}: ${res.rows} rows ${res.mode === 'merge' ? 'merged' : 'promoted'}`);
  }

  async openPromote() {
    let sets = [];
    try { sets = (await api('/api/training-sets')).sets || []; } catch (err) { this.status(err.message); }
    const go = (menu, name) => { if (!name) return; menu.close(); this.promote(name); };
    new Menu({
      title: 'promote to a set',
      sections: [
        {kind: 'list', empty: 'none yet - name one below', multi: false,
          items: sets.map(set => ({id: set.name, label: set.name, stats: `${set.rows} rows`})),
          onPick: (item, _on, menu) => go(menu, item.id)},
        {kind: 'add', placeholder: 'or a new set', button: 'promote',
          onAdd: (value, menu) => go(menu, (value || '').trim())},
      ],
    }).openAt(document.getElementById('cluster-promote'));
  }

  wire() {
    document.getElementById('cluster-open-set').onclick = () => this.openClusterSets();
    document.getElementById('cluster-classdef').onclick = () => this.openClassPicker();
    document.getElementById('cluster-select-all').onclick =
      () => this.sel.setKeys(this.visible.map(item => item.name), true);
    document.getElementById('cluster-clear-sel').onclick = () => this.sel.clear();
    document.getElementById('recluster').onclick = () => this.load();
    document.getElementById('cluster-save').onclick = () => this.saveLabels();
    document.getElementById('cluster-promote').onclick = () => this.openPromote();
    document.addEventListener('keydown', evt => {
      if (activeTab !== 'select') return;
      if (/^(INPUT|TEXTAREA)$/.test(document.activeElement?.tagName || '')) return;
      if (evt.key === 'Escape' && this.sel.size()) { this.sel.clear(); return; }
      if ((evt.metaKey || evt.ctrlKey) && evt.key.toLowerCase() === 'a') {
        evt.preventDefault();
        document.getElementById('cluster-select-all').onclick();
      }
    });
  }
}
