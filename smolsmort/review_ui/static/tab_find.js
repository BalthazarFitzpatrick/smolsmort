// find: draw boxes on a recording's frames, choose how a saved tile is cut, save the set

smolsmortTabs.register({id: 'find', label: 'find', mount: mountFind});

const CROP_MAX = 200;

function mountFind(panel) {
  panel.innerHTML = `
    <div id="draw-panel">
      <div id="draw-bar" class="run-controls">
        <div class="toggle dropdown-head" id="draw-open" data-label="dataset">
          <span id="draw-where" class="placeholder">recordings</span>
        </div>
        <div class="toggle dropdown-head narrow" id="draw-percent" data-value="20"
             title="how much of the recording to draw on"><span>20%</span></div>
        <div class="toggle" id="draw-prev">prev</div>
        <div class="draw-scrub">
          <div class="draw-scrub-row">
            <span class="field-label" id="draw-first">-</span>
            <input type="range" class="range-slider" id="draw-slider" min="0" max="0" value="0"
                   step="1" title="scrub through the frames offered">
            <span class="field-label" id="draw-last">-</span>
          </div>
          <span class="field-value" id="draw-pos">-</span>
        </div>
        <div class="toggle" id="draw-next">next</div>
        <span id="draw-zoom-value" class="stat">100%</span>
        <div class="toggle" id="draw-zoom-fit">reset view</div>
        <span class="spacer"></span>
        <span class="stat" id="draw-count">0 boxes</span>
        <div class="toggle adds" id="draw-save">save</div>
      </div>
      <div id="crop-bar">
        <div class="crop-toggles">
          <div class="toggle" id="crop-mode" title="width and height in percent of the box or in px">percent</div>
          <div class="toggle" id="crop-aspect" title="fixed keeps the ratio when one slider moves">free</div>
        </div>
        <div class="crop-sliders">
          <div class="crop-row">
            <span class="field-label">width</span>
            <input type="range" class="range-slider" id="crop-w" min="0" max="${CROP_MAX}" value="100">
            <span class="field-value" id="crop-w-value"></span>
          </div>
          <div class="crop-row">
            <span class="field-label">height</span>
            <input type="range" class="range-slider" id="crop-h" min="0" max="${CROP_MAX}" value="100">
            <span class="field-value" id="crop-h-value"></span>
          </div>
        </div>
      </div>
      <div id="draw-wrap"><canvas id="draw-canvas"></canvas></div>
      <div id="find-status" class="stat"></div>
    </div>`;
  const find = new FindTab();
  return {enter: () => find.enter()};
}

class FindTab {
  constructor() {
    this.frames = [];
    this.index = 0;
    this.boundName = '';
    this.boundPercent = '';
    this.stamp = 0;
    this.boxes = {};        // frame name -> [{left, top, width, height}]
    this.image = new Image();
    this.drawing = null;
    this.crop = {pad_x: 0, pad_y: 0, crop_mode: 'percent', crop_w: 100, crop_h: 100, aspect: 'free'};
    this.ratio = 1;         // w / h captured when fixed aspect was turned on
    this.saveTimer = null;
    this.wireDraw();
    this.wireCrop();
  }

  async enter() {
    await this.loadCrop();
    await this.openFrames();
  }

  // ---- crop settings ----
  async loadCrop() {
    try {
      const data = await api('/api/crop-settings');
      if (!data.error) this.crop = {...this.crop, ...data};
    } catch (err) { /* keep the defaults, the sliders still work */ }
    if (this.crop.aspect === 'fixed') this.captureRatio();
    this.paintCrop();
  }

  captureRatio() {
    this.ratio = this.crop.crop_h > 0 && this.crop.crop_w > 0 ? this.crop.crop_w / this.crop.crop_h : 1;
  }

  paintCrop() {
    const unit = this.crop.crop_mode === 'percent' ? '%' : 'px';
    document.getElementById('crop-mode').textContent = this.crop.crop_mode;
    const fixed = this.crop.aspect === 'fixed';
    const aspect = document.getElementById('crop-aspect');
    aspect.textContent = this.crop.aspect;
    aspect.classList.toggle('on', fixed);
    [['w', 'crop_w'], ['h', 'crop_h']].forEach(([id, key]) => {
      document.getElementById(`crop-${id}`).value = String(this.crop[key]);
      setText(`crop-${id}-value`, `${Math.round(this.crop[key])}${unit}`);
    });
    this.redraw();
  }

  // posts after the slider settles, so a drag is one write and not sixty
  saveCrop() {
    clearTimeout(this.saveTimer);
    this.saveTimer = setTimeout(async () => {
      try { await api('/api/crop-settings', this.crop); } catch (err) { setText('find-status', err.message); }
    }, 120);
  }

  slide(changed, value) {
    const other = changed === 'crop_w' ? 'crop_h' : 'crop_w';
    let mine = Math.max(0, Math.min(CROP_MAX, value));
    let theirs = this.crop[other];
    if (this.crop.aspect === 'fixed') {
      // the partner follows at the captured ratio; the dragged one stops where the partner tops out
      const scale = changed === 'crop_w' ? 1 / this.ratio : this.ratio;
      theirs = mine * scale;
      if (theirs > CROP_MAX) { theirs = CROP_MAX; mine = theirs / scale; }
    }
    this.crop = {...this.crop, [changed]: Math.round(mine * 10) / 10, [other]: Math.round(theirs * 10) / 10};
    this.paintCrop();
    this.saveCrop();
  }

  wireCrop() {
    document.getElementById('crop-w').oninput = evt => this.slide('crop_w', Number(evt.target.value));
    document.getElementById('crop-h').oninput = evt => this.slide('crop_h', Number(evt.target.value));
    document.getElementById('crop-mode').onclick = () => {
      this.crop.crop_mode = this.crop.crop_mode === 'percent' ? 'absolute' : 'percent';
      this.paintCrop();
      this.saveCrop();
    };
    document.getElementById('crop-aspect').onclick = () => {
      this.crop.aspect = this.crop.aspect === 'fixed' ? 'free' : 'fixed';
      if (this.crop.aspect === 'fixed') this.captureRatio();
      this.paintCrop();
      this.saveCrop();
    };
  }

  // where a saved tile would sit around a box, so a slider change shows on the frame
  tileRect(box) {
    const {crop_mode: mode, crop_w: w, crop_h: h} = this.crop;
    const width = mode === 'percent' ? box.width * w / 100 : w;
    const height = mode === 'percent' ? box.height * h / 100 : h;
    return {
      left: box.left + box.width / 2 - width / 2, top: box.top + box.height / 2 - height / 2,
      width, height,
    };
  }

  // ---- frames ----
  percent() { return document.getElementById('draw-percent').dataset.value; }

  currentBoxes() { return this.boxes[this.frames[this.index]] || []; }

  redraw() {
    const canvas = document.getElementById('draw-canvas');
    const ctx = canvas.getContext('2d');
    if (!this.image.width) return;
    canvas.width = this.image.width;
    canvas.height = this.image.height;
    ctx.drawImage(this.image, 0, 0);
    const zoom = this.view ? this.view.zoom() : 1;
    ctx.lineWidth = Math.max(1, 2 / zoom);
    this.currentBoxes().forEach(box => {
      ctx.setLineDash([]);
      ctx.strokeStyle = '#ff29d6';
      ctx.strokeRect(box.left, box.top, box.width, box.height);
      const tile = this.tileRect(box);
      ctx.setLineDash([6 / zoom, 4 / zoom]);
      ctx.strokeStyle = '#c7ed5f';
      ctx.strokeRect(tile.left, tile.top, tile.width, tile.height);
    });
    ctx.setLineDash([]);
    if (this.drawing) {
      ctx.strokeStyle = '#4da6ff';
      ctx.strokeRect(this.drawing.left, this.drawing.top, this.drawing.width, this.drawing.height);
    }
    const total = Object.values(this.boxes).reduce((n, list) => n + list.length, 0);
    setText('draw-count', `${total} boxes`);
    setText('draw-pos', `${this.index + 1} / ${this.frames.length}  ${this.frames[this.index] || ''}`);
    this.syncSlider();
  }

  // the slider follows the index, it does not lead it
  syncSlider() {
    const slider = document.getElementById('draw-slider');
    const last = Math.max(0, this.frames.length - 1);
    slider.max = String(last);
    slider.disabled = this.frames.length < 2;
    if (slider.value !== String(this.index)) slider.value = String(this.index);
    const stem = name => (name || '').replace(/\.[^.]+$/, '') || '-';
    setText('draw-first', stem(this.frames[0]));
    setText('draw-last', stem(this.frames[last]));
  }

  loadImage() {
    if (!this.frames.length) return;
    this.image = new Image();
    this.image.onload = () => {
      if (this.view) this.view.reset(this.image.width, this.image.height);
      this.redraw();
    };
    this.image.src = '/draw-frame/' + encodeURIComponent(this.frames[this.index]) + '?v=' + this.stamp;
  }

  goTo(index) {
    if (index < 0 || index >= this.frames.length || index === this.index) return;
    this.index = index;
    this.loadImage();
  }

  async openFrames() {
    const data = await api('/api/draw-frames?percent=' + this.percent());
    const bound = data.dataset || data.recording || '';
    const percent = this.percent();
    if (data.ready && bound && bound === this.boundName && percent === this.boundPercent) return;
    const where = document.getElementById('draw-where');
    if (data.ready && bound && bound === this.boundName) {
      // a resample of the same recording keeps unsaved boxes and the frame in view
      const at = this.frames[this.index];
      this.boundPercent = percent;
      this.frames = data.frames;
      const found = at ? this.frames.indexOf(at) : -1;
      this.index = found >= 0 ? found : Math.min(this.index, Math.max(0, this.frames.length - 1));
      this.loadImage();
      return;
    }
    if (!data.ready) {
      setText('find-status', '');
      setText('draw-pos', '-');
      where.textContent = 'recordings';
      where.className = 'placeholder';
      this.frames = [];
      this.syncSlider();
      return;
    }
    this.boundName = bound;
    this.boundPercent = percent;
    this.stamp = Date.now();
    this.frames = data.frames;
    this.boxes = data.boxes || {};
    this.index = 0;
    where.textContent = bound;
    where.className = '';
    this.loadImage();
  }

  point(evt) {
    const canvas = document.getElementById('draw-canvas');
    const rect = canvas.getBoundingClientRect();
    const scale = rect.width / (canvas.width || 1);
    return [(evt.clientX - rect.left) / scale, (evt.clientY - rect.top) / scale];
  }

  wireDraw() {
    const canvas = document.getElementById('draw-canvas');
    this.view = makePanZoom(document.getElementById('draw-wrap'), canvas, {
      panModifier: 'shift', fit: 'contain',
      onChange: zoom => { setText('draw-zoom-value', `${Math.round(zoom * 100)}%`); this.redraw(); },
    });
    document.getElementById('draw-zoom-fit').onclick =
      () => this.view.reset(this.image.width, this.image.height);

    canvas.addEventListener('mousedown', evt => {
      if (evt.shiftKey) return;
      const [x, y] = this.point(evt);
      const name = this.frames[this.index];
      const boxes = this.currentBoxes();
      const hit = boxes.findIndex(b => x >= b.left && x <= b.left + b.width
        && y >= b.top && y <= b.top + b.height);
      if (hit >= 0) {
        boxes.splice(hit, 1);
        this.boxes[name] = boxes;
        this.redraw();
        return;
      }
      this.drawing = {left: x, top: y, width: 0, height: 0, x0: x, y0: y};
    });
    canvas.addEventListener('mousemove', evt => {
      if (!this.drawing) return;
      const [x, y] = this.point(evt);
      this.drawing.left = Math.min(this.drawing.x0, x);
      this.drawing.top = Math.min(this.drawing.y0, y);
      this.drawing.width = Math.abs(x - this.drawing.x0);
      this.drawing.height = Math.abs(y - this.drawing.y0);
      this.redraw();
    });
    window.addEventListener('mouseup', () => {
      if (!this.drawing) return;
      if (this.drawing.width > 6 && this.drawing.height > 3) {
        const name = this.frames[this.index];
        const box = {
          left: Math.round(this.drawing.left), top: Math.round(this.drawing.top),
          width: Math.round(this.drawing.width), height: Math.round(this.drawing.height),
        };
        this.boxes[name] = this.currentBoxes().concat([box]);
      }
      this.drawing = null;
      this.redraw();
    });

    document.getElementById('draw-percent').onclick = evt => {
      const el = evt.currentTarget;
      listMenu('how much to draw on',
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100].map(n => ({
          id: String(n), label: `${n}%`, on: el.dataset.value === String(n)})),
        item => { el.dataset.value = item.id; el.innerHTML = `<span>${item.id}%</span>`; this.openFrames(); }
      ).openAt(el);
    };
    document.getElementById('draw-prev').onclick = () => this.goTo(this.index - 1);
    document.getElementById('draw-next').onclick = () => this.goTo(this.index + 1);
    // input, not change, so a scrub shows the frames it passes over
    document.getElementById('draw-slider').oninput = evt => this.goTo(Number(evt.currentTarget.value));
    document.addEventListener('keydown', evt => {
      if (activeTab !== 'find') return;
      if (/^(INPUT|TEXTAREA)$/.test(document.activeElement?.tagName || '')) return;
      if (evt.metaKey || evt.ctrlKey || evt.altKey) return;
      const key = evt.key.toLowerCase();
      if (key !== 'b' && key !== 'n') return;
      evt.preventDefault();
      this.goTo(this.index + (key === 'n' ? 1 : -1));
    });
    document.getElementById('draw-open').onclick = () => this.openDatasetPicker();
    document.getElementById('draw-save').onclick = () => this.save();
  }

  async save() {
    const boxes = [];
    Object.entries(this.boxes).forEach(([path, list]) => list.forEach(b => boxes.push({...b, path})));
    if (!boxes.length) return;
    const res = await api('/api/find-run', {mode: 'drawn', boxes});
    if (res.error) { setText('find-status', res.error); return; }
    setText('find-status', `${res.count} boxes saved` + (res.tiles ? ` - ${res.tiles} tiles cut` : ''));
    // saving rebinds to the written set, so the picker head is told
    const now = await api('/api/draw-frames?percent=' + this.percent());
    const where = document.getElementById('draw-where');
    if (now.ready) { where.textContent = now.dataset || now.recording; where.className = ''; }
    this.redraw();
  }

  // ---- the dataset picker: tick one, press open; close lets go of the open one ----
  async openDatasetPicker() {
    const chosen = new Map();
    let menu = null;
    const unit = row => {
      const parts = [`${row.frame_count}${row.kind === 'dataset' ? ' proposals' : 'f'}`];
      if (row.boxes) parts.push(`${row.boxes} drawn`);
      if (row.classed) parts.push(`${row.classed} classed`);
      return parts.join(' · ');
    };
    const build = async () => {
      const data = await api('/api/bindable-recordings');
      const column = (label, kind) => ({
        label, empty: 'none',
        onPick: (item, on) => {
          if (item.opened) return;
          if (on) chosen.clear();
          if (on) chosen.set(item.id, item); else chosen.delete(item.id);
          menu.setButtonEnabled('open', chosen.size > 0);
        },
        items: data.rows.filter(r => r.kind === kind).sort((a, b) => a.name.localeCompare(b.name))
          .map(row => ({
            id: row.name, label: row.name, kind: row.kind, stats: unit(row),
            opened: row.name === data.current, state: row.name === data.current ? {opened: true} : {},
          })),
      });
      const columns = [column('recordings', 'session')];
      if (data.rows.some(r => r.kind === 'dataset')) columns.push(column('proposals', 'dataset'));
      return [
        {kind: 'columns', columns},
        {kind: 'buttons', buttons: [
          {id: 'open', label: 'open', tone: 'adds', enabled: false, onClick: async () => {
            for (const item of chosen.values()) {
              const res = await api('/api/bind-recording', {kind: item.kind, name: item.id});
              if (res.error) { setText('find-status', res.error); return; }
            }
            chosen.clear();
            await this.openFrames();
            menu.refresh(await build());
          }},
          {id: 'close-set', label: 'close', tone: 'removes', enabled: !!data.current,
            onClick: async () => {
              await api('/api/unbind-recording', {});
              this.boundName = '';
              await this.openFrames();
              setText('find-status', 'closed - boxes left on disk');
              menu.refresh(await build());
            }},
        ]},
      ];
    };
    menu = await openPicker('draw-open', build, {title: 'recordings', status: 'find-status'});
  }
}
