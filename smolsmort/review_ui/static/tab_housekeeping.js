// housekeeping: find and clear what a deleted recording left behind

smolsmortTabs.register({id: 'housekeeping', label: 'housekeeping', mount: mountHousekeeping});

const HK_KINDS = {
  boxes: 'boxes', tiles: 'tiles', tiles_synth: 'generated tiles', sets: 'sets', checkpoints: 'checkpoints',
};

function mountHousekeeping(panel) {
  panel.innerHTML = `
    <div class="row-label">recordings and everything made from them</div>
    <div class="h-divider"></div>
    <div class="label-columns" id="hk-columns">
      <div class="col" id="hk-recordings"></div>
      <div class="divider"></div>
      <div class="col" id="hk-assets-col">
        <div id="hk-assets-bar" class="run-controls">
          <div class="toggle" id="hk-select-all">select all for this recording</div>
          <span class="spacer"></span>
          <span class="stat" id="hk-selected-count">nothing selected</span>
          <div class="toggle adds" id="hk-archive">archive</div>
          <div class="toggle removes" id="hk-delete">delete</div>
        </div>
        <div id="hk-assets"></div>
      </div>
    </div>
    <div id="hk-status" class="stat"></div>`;
  const tab = new HousekeepingTab();
  return {enter: () => tab.load()};
}

class HousekeepingTab {
  constructor() {
    this.recordings = [];
    this.tag = null;
    this.groups = {};
    this.rowPaths = {};     // checkbox id -> the file paths it stands for
    this.checked = new Set();
    this.wire();
  }

  say(text) { setText('hk-status', text); }

  async load() {
    this.say('');
    const data = await api('/api/housekeeping-recordings');
    this.recordings = data.recordings;
    this.renderRecordings();
    if (this.tag && !this.recordings.some(r => r.tag === this.tag)) {
      this.tag = null;
      document.getElementById('hk-assets').innerHTML = '';
    }
  }

  renderRecordings() {
    const col = document.getElementById('hk-recordings');
    col.innerHTML = '';
    const groups = {};
    this.recordings.forEach(r => { (groups[r.group] ||= []).push(r); });
    Object.keys(groups).sort().forEach(group => {
      const heading = document.createElement('div');
      heading.className = 'field-label';
      heading.textContent = group;
      col.appendChild(heading);
      groups[group].sort((a, b) => a.path.localeCompare(b.path)).forEach(r => {
        const row = document.createElement('div');
        row.className = 'toggle hk-recording-row' + (r.tag === this.tag ? ' on' : '');
        row.title = r.path;
        row.appendChild(textSpan('hk-rec-name', r.path));
        if (r.missing) row.appendChild(textSpan('hk-rec-badge', 'orphaned'));
        row.onclick = () => this.select(r.tag);
        col.appendChild(row);
      });
    });
    if (!this.recordings.length) col.appendChild(textSpan('stat', 'no recordings, and nothing orphaned'));
  }

  async select(tag) {
    this.tag = tag;
    this.checked.clear();
    this.renderRecordings();
    await this.loadAssets();
  }

  async loadAssets() {
    this.groups = await api(`/api/housekeeping-assets?tag=${encodeURIComponent(this.tag)}`);
    this.renderAssets();
  }

  size(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    return bytes < 1048576 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1048576).toFixed(1)} MB`;
  }

  renderAssets() {
    const box = document.getElementById('hk-assets');
    box.innerHTML = '';
    this.rowPaths = {};
    this.checked.clear();
    let n = 0;
    Object.entries(HK_KINDS).forEach(([kind, title]) => {
      const rows = this.groups[kind] || [];
      if (!rows.length) return;
      box.appendChild(Object.assign(document.createElement('div'), {className: 'field-label', textContent: title}));
      rows.forEach(row => {
        const id = `hk-row-${n++}`;
        this.rowPaths[id] = row.paths;
        const label = document.createElement('label');
        label.className = 'toggle hk-asset-row';
        const check = document.createElement('input');
        check.type = 'checkbox';
        check.className = 'hk-check';
        check.id = id;
        check.onchange = () => {
          if (check.checked) this.checked.add(id); else this.checked.delete(id);
          this.count();
        };
        label.append(check, textSpan('hk-asset-label', row.label), textSpan('hk-asset-size', this.size(row.size_bytes)));
        box.appendChild(label);
      });
    });
    if (!box.children.length) box.appendChild(textSpan('stat', 'nothing traces back to this recording'));
    this.count();
  }

  count() {
    setText('hk-selected-count', this.checked.size ? `${this.checked.size} selected` : 'nothing selected');
  }

  paths() { return [...this.checked].flatMap(id => this.rowPaths[id] || []); }

  async act(kind) {
    const paths = this.paths();
    if (!paths.length) { this.say('nothing selected'); return; }
    // delete has no second copy, so it names the count and asks once more
    if (kind === 'delete' && !confirm(`delete ${paths.length} file(s) for good? this cannot be undone`)) return;
    try {
      const res = await api(`/api/housekeeping-${kind}`, {paths});
      this.say(kind === 'archive' ? `archived ${res.moved.length} file(s)` : `deleted ${res.moved.length} file(s)`);
    } catch (err) {
      this.say(`${kind} failed: ${err.message}`);
    }
    await this.loadAssets();
    await this.load();
  }

  wire() {
    document.getElementById('hk-select-all').onclick = () => {
      document.querySelectorAll('.hk-check').forEach(cb => { cb.checked = true; this.checked.add(cb.id); });
      this.count();
    };
    document.getElementById('hk-archive').onclick = () => this.act('archive');
    document.getElementById('hk-delete').onclick = () => this.act('delete');
  }
}
