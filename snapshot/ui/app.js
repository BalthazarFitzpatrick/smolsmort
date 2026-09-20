// the review tool's page script.
//
// EXTRACTED FROM A PYTHON STRING LITERAL on 2026-09-02. It lived inside INDEX_HTML in
// tools/review_templates.py, which meant no syntax checking, no linting and no editor help -
// every change was a text substitution into a 3,472-line literal. Two failures came straight
// from that: a set of replacements that silently no-opped after ruff reflowed the target, and
// a removed line that left `else if` with no `if`, killing the whole script and every control
// on every tab at once.
//
// MARGIN and MARGIN_Y_JS are injected by the server in a small inline block before
// this file loads - they are the only dynamic values on the page, and keeping them there is
// what lets everything else be a static file.
// activeTab is declared by review_ui/shell.js, which owns tab state - redeclaring it here would
// be a duplicate top-level binding and kill the whole script
async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// A PICKER THAT FAILS SAYS SO. an async onclick whose body rejects is completely silent - no
// panel, no console line - which looks exactly like a dead button. every picker opens through
// here, so a failed build becomes a panel naming what broke instead of nothing at all
// A MENU OPENED *UNDER* A HEAD, RATHER THAN *ON* IT. ui_base reads "open a menu on the head that
// already owns the open menu" as toggle-shut (menu.js:318): it closes the open one and returns
// without building the new one. So a confirmation opened on its own picker's head was never built,
// its onDismiss never fired, and the promise it was meant to settle never resolved - close silently
// did nothing. Coordinates carry no trigger, so the toggle branch cannot fire, and the panel still
// lands exactly where anchoring put it.
function headPoint(anchorId) {
  const box = document.getElementById(anchorId).getBoundingClientRect();
  return {x: box.left, y: box.bottom + 6};
}

async function openPicker(anchorId, build, options = {}) {
  const head = document.getElementById(anchorId);
  try {
    const menu = new Menu({
      title: options.title || '',
      persistent: options.persistent !== false,
      sections: await build(),
    });
    menu.openAt(head);
    return menu;
  } catch (err) {
    const reason = (err && err.message) || String(err);
    const status = options.status && document.getElementById(options.status);
    if (status) status.textContent = reason;
    new Menu({
      title: 'could not open',
      sections: [{kind: 'list', empty: reason, items: []}],
    }).openAt(head);
    return null;
  }
}

// DECLARED ABOVE enterTab: shell.js restores the last tab while this script is still loading, so
// reopening on navigation read navTimer before its `let` ran - "Cannot access 'navTimer' before
// initialization", caught by the 2026-09-11 playwright audit
let navTimer = null;

// tab switching lives in review_ui/shell.js - this is only what each tab has to load on entry
function enterTab(name) {
  if (name === 'scripts') loadScripts();
  else if (name === 'cluster') loadClassDef().then(loadClusters);
  else if (name === 'train') { loadTrain(); loadTrainClasses(); loadCropFloor(); }
  else if (name === 'find') drawOpen();
  else if (name === 'vlm') loadVlm();
  else if (name === 'interface') ifaceLoad();
  else if (name === 'map') mapEnter();
  // ENTERING THE TAB HAS TO START THE FOLLOW, not merely redraw. The tab drew navDraw() alone,
  // and navSetFollow - the only thing that calls navLoad or starts the poll - was reachable ONLY
  // from the recordings dropdown. So opening the tab always showed an empty canvas reading
  // "nothing recorded" while /api/nav was serving a real walk, until you happened to pick a
  // session by hand. This predates the restore: the tab was built on 2026-09-03 and its ledger
  // entry says "BUILT, NOT YET SEEN IN A BROWSER", which is exactly how it survived.
  // Following is the default state (navSession === null means "whatever is being recorded now"),
  // so entering with nothing chosen picks it up; a session already chosen is left alone.
  else if (name === 'navigation') {
    if (!navTimer && navFollowing()) navSetFollow(true); else navDraw();
    navKeysLoad().catch(() => {});
  }
  else if (name === 'camera') camEnter();
  else if (name === 'control') controlEnter();
  else if (name === 'sessions') sessionsEnter();
  else if (name === 'housekeeping') hkLoad();
}

// the tune tab lived here: a paged keep/discard queue over one bound dataset, and the only
// place a crop could be re-aligned. its alignment work is review_ui/align.js now, mounted on
// the pool tiles in discard / promote, which is where a mis-cut crop is actually noticed

// ---- Interface tab: mark every ui element by hand, on real screenshots ----
// granular on purpose - one rect per BAR, not per frame. nothing here snaps or searches: what is
// drawn is what gets written, because every failure in this project so far came from a stage that
// inferred a label instead of being told it.
let ifaceState = null;
let ifaceSelected = null;
let ifaceShot = null;
let ifaceMode = 'rect';
let ifaceDrag = null;

// THE SINGLE REPAINT PATH. takes the authoritative state and redraws everything that reads from
// it, so no caller can repaint half of it.
//
// This existed, was deleted on 2026-08-31 while its SEVEN callers stayed, and was undefined for two
// days - every mark, add, remove and keybind threw a ReferenceError after its request succeeded, so
// the server updated and the page did not. node --check does not catch an unbound identifier, and
// the symptom looked like a caching bug rather than a crash. Restored, and widened: the name
// pickers and the owner line read ifaceState too, and repainting some views but not others is the
// same failure in a smaller size - it is how a profile switch left retina marks on screen under a
// header reading "nothing marked yet".
function ifaceRefresh(next) {
  if (next && !next.error) ifaceState = next;
  if (!ifaceState) return;
  ifaceFillPicker('iface-realm', ifaceState.known.realms, ifaceState.realm, 'server');
  ifaceFillPicker('iface-character', ifaceState.known.characters, ifaceState.character, 'character');
  ifaceFillPicker('iface-profile', ifaceState.known.profiles, ifaceState.profile_name,
                  'screen profile');
  ifaceRenderList();
  ifaceRenderStates();
  ifaceRenderKeybinds();
  ifaceRenderOwner();
  // THE SHOT STRIP READS ifaceState TOO, and was the one view this path forgot - so picking a
  // server/character/screen repainted the element trees while the screenshots stayed on the
  // previous selection's (usually empty) list until ifaceLoad ran again, which only happens on
  // navigating away and back. exactly the "repainting some views but not others" failure this
  // function's own docstring warns about, one view further down.
  //
  // the selected shot has to be re-pinned here as well: it belongs to the OUTGOING character, and
  // holding it would show one character's screenshot under another's marks.
  if (!ifaceState.screenshots.includes(ifaceShot)) {
    ifaceShot = ifaceState.screenshots[0] || null;
  }
  ifaceRenderShots();
  ifaceDrawOverlay();
}

async function ifaceLoad() {
  ifaceRefresh(await api('/api/interface'));
  if (!ifaceShot && ifaceState.screenshots.length) ifaceShot = ifaceState.screenshots[0];
  ifaceRenderShots();
  ifaceSyncMode();
}

// WHAT THE MARKS BELONG TO, said out loud. Balthazar Fitzpatrick: "I have data filled, I have marked things, but
// they belong to nothing if it is according to the interface." three blank fields above a list of
// finished marks is exactly that - the work is real and the page says nothing about where it goes.
function ifaceRenderOwner() {
  const el = document.getElementById('iface-saved');
  if (!ifaceState) return;
  const marked = ifaceState.elements.filter(e => e.marked).length;
  const examples = ifaceState.elements.reduce(
    (n, e) => n + Object.keys(e.examples || {}).length, 0);
  const realm = ifacePicked('iface-realm');
  const character = ifacePicked('iface-character');
  const profile = ifacePicked('iface-profile');

  if (!marked && !examples) { el.textContent = 'nothing marked yet'; el.className = 'field-label'; return; }
  if (!(realm && character && profile)) {
    // naming is what attaches the work to something; until then it lives only in interface.json
    const need = [!realm && 'server', !character && 'character', !profile && 'screen profile']
      .filter(Boolean).join(', ');
    el.textContent = `${marked} marks, ${examples} examples - NOT SAVED TO A PROFILE YET, needs: ${need}`;
    el.className = 'field-label iface-unattached';
    return;
  }
  // the path the server really writes: one folder per realm, character and screen
  el.textContent =
    `${marked} marks, ${examples} examples -> profiles/characters/`
    + `${realm}-${character}-${profile}`.toLowerCase() + '/character.toml';
  el.className = 'field-label';
}

// each of the three is a dropdown of what already exists, with "+ new" first. names typed once
// this way get reused rather than retyped, and a typo makes a second server rather than silently
// attaching a character to the wrong one. a name added here is remembered for the session even
// before anything is written to disk, so it survives a reload of the list.
// THE THREE NAME PICKERS, in the popup language the cluster tab's label picker already uses:
// click the field, a floating panel opens with an "add" row, a divider, then the existing list
// with the current one highlighted. Balthazar Fitzpatrick, after I had tried a <select> and then a bare text
// field: "it is regular web design, click > popup opens, several sections in there... the cluster
// right click also has multiselect and loads lists, so why is this so hard".
//
// a <select> was the wrong primitive twice over - its "+ new" row was itself the selected value,
// so choosing it fired no change event, and cancelling left it selected so the next attempt was
// dead too. a div plus a panel has no such states.
const IFACE_PICKERS = {
  'iface-realm': {field: 'realm', known: 'realms'},
  'iface-character': {field: 'character', known: 'characters'},
  'iface-profile': {field: 'profile_name', known: 'profiles'},
};
let ifacePickerOpen = null;
let ifacePickerMenu = null;

function ifacePicked(id) {
  const spec = IFACE_PICKERS[id];
  return (ifaceState && spec ? ifaceState[spec.field] : '') || '';
}

function ifaceFillPicker(id, known, current, label) {
  const el = document.getElementById(id);
  el.dataset.label = label;
  // COLLAPSED, THE FIELD IS ITS OWN LABEL: it shows what is chosen, or says what it wants
  el.innerHTML = current
    ? `<span>${current}</span>`
    : `<span class="placeholder">choose a ${label}</span>`;
  void known;
}

function ifaceClosePicker() {
  ifacePickerMenu?.close();
  ifacePickerMenu = null;
  ifacePickerOpen = null;
}

// THE ADD ROW STAYS FIRST, which is the whole reason a <select> was rejected here twice: its
// "+ new" row was itself the selected value, so choosing it fired no change event and cancelling
// left it selected so the next attempt was dead too. a menu with an add section has no such states
function ifaceOpenPicker(id) {
  const spec = IFACE_PICKERS[id];
  const el = document.getElementById(id);
  const current = ifacePicked(id);
  const known = (ifaceState && ifaceState.known[spec.known]) || [];
  ifacePickerOpen = id;
  ifacePickerMenu = new Menu({
    sections: [
      {
        kind: 'add',
        placeholder: `new ${el.dataset.label}`,
        button: 'add',
        onAdd: value => ifaceChoose(id, value),
      },
      {
        kind: 'list',
        items: known.map(name => ({id: name, label: name, on: name === current})),
        empty: `no ${el.dataset.label} yet - add one above`,
        onPick: item => ifaceChoose(id, item.id),
      },
    ],
    onDismiss: () => { ifacePickerOpen = null; ifacePickerMenu = null; },
  }).openAt(el);
  ifacePickerMenu.el.querySelector('input')?.focus();
}

async function ifaceChoose(id, name) {
  const spec = IFACE_PICKERS[id];
  const value = (name || '').trim();
  if (!value) return;
  ifaceClosePicker();
  await ifaceSaveNames({[spec.field]: value});
  // naming is creating: once all three are known the pair is written
  if (ifacePicked('iface-realm') && ifacePicked('iface-character')
      && ifacePicked('iface-profile')) {
    ifaceSave(true);
  }
}

Object.keys(IFACE_PICKERS).forEach(id => {
  document.getElementById(id).onclick = evt => {
    evt.stopPropagation();
    if (ifacePickerOpen === id) { ifaceClosePicker(); return; }
    ifaceOpenPicker(id);
  };
});
// the add button, the Enter/Escape keys and the outside-click dismiss that used to be written
// out here are all Menu's, and identical for every picker in the tool now

// which state is being marked, or null for "just the position"
let ifaceState_ = null;

// MANDATORY FIRST, THEN ADDITIONAL, and inside each of those target -> player -> rest. anything
// added with + keeps the order it was created in - Balthazar Fitzpatrick: "the customs get whatever order they
// are created in" - so customs are appended untouched rather than sorted into a bucket.
function ifaceOrdered() {
  const rows = [];
  const standard = ifaceState.elements.filter(e => e.standard);
  [true, false].forEach(mandatory => {
    const section = [];
    ['target', 'player', 'rest'].forEach(group => {
      standard.forEach(item => {
        if (!!item.required === mandatory && (item.group || 'rest') === group) section.push(item);
      });
    });
    // "standard" rather than "mandatory": these are the ones a reader cannot work without and
    // the only ones the - button refuses. additional and added are both removable.
    if (section.length) rows.push({heading: mandatory ? 'standard' : 'additional'}, ...section);
  });
  const customs = ifaceState.elements.filter(e => !e.standard);
  if (customs.length) rows.push({heading: 'added'}, ...customs);
  return rows;
}

function ifaceRenderList() {
  const el = document.getElementById('iface-list');
  const rows = ifaceOrdered().map(item => {
    if (item.heading) return item;
    const selected = item.name === ifaceSelected;
    const where = item.rect ? `[${item.rect.join(', ')}]`
      : item.point ? `(${item.point.join(', ')}) r${item.radius}` : '';
    return {
      id: item.name,
      label: item.name,
      badges: [item.keybind, where].filter(Boolean),
      // a required element that is still unmarked is the one thing this list must make obvious
      state: {marked: !!item.marked, on: selected, needed: item.required && !item.marked},
      title: (item.note ? item.note + ' - ' : '')
        + (item.screenshot ? `marked on ${item.screenshot}`
           : item.required ? 'REQUIRED, not marked yet' : 'not marked yet'),
      onPick: () => {
        // clicking the item IS what opens its states - they belong under the thing they describe,
        // not in a dropdown at the far end of the panel
        if (selected) { ifaceSelected = null; ifaceState_ = null; }
        else { ifaceSelected = item.name; ifaceMode = item.mode || 'rect'; ifaceState_ = null; }
        ifaceRenderList();
        ifaceSyncMode();
        ifaceStatus(ifaceSelected
          ? `${item.name}: drag on the screenshot, or pick a state first`
          : '');
      },
    };
  });
  renderTree(el, rows, {itemClass: 'iface-item'});

  // the selected element's states go directly under its row
  const selectedItem = ifaceOrdered().find(i => !i.heading && i.name === ifaceSelected);
  if (!selectedItem) return;
  const row = el.querySelector(`.tree-item[data-id="${CSS.escape(ifaceSelected)}"]`);
  if (!row) return;
  const after = document.createElement('div');
  after.className = 'iface-states';
  // "position only" first: marking without a state is the common case and must stay one click
  ifaceStateRow(after, selectedItem, null, 'position only');
  (selectedItem.states || []).forEach(state => ifaceStateRow(after, selectedItem, state, state));
  ifaceStateRow(after, selectedItem, '__add__', '+ add spec');
  row.after(after);
}

function ifaceStateRow(container, item, state, label) {
  const row = document.createElement('div');
  const active = state !== '__add__' && ifaceState_ === state;
  row.className = 'toggle iface-state-row' + (active ? ' on' : '');
  // "position only" is state === null, so an examples[state] lookup can never be true for it -
  // it was showing a grey dot on an element that plainly had a position. its evidence is the
  // element's own mark, not an entry in examples.
  const shown = state === null
    ? (item.marked ? {screenshot: item.screenshot} : null)
    : (state !== '__add__' && item.examples ? item.examples[state] : null);
  const needed = (item.missing_states || []).includes(state);
  // its own dot, same language as the item rows: green once this state has an example, amber
  // while it is required and missing, grey when it is optional and simply not marked
  if (shown) row.classList.add('marked');
  else if (needed) row.classList.add('needed');
  row.innerHTML = (state === '__add__' ? '' : '<span class="dot"></span>')
    + `<span>${label}</span>`
    + (shown ? `<span class="shown">${shown.screenshot || 'shown'}</span>` : '');
  row.onclick = async evt => {
    evt.stopPropagation();
    if (state !== '__add__') {
      ifaceState_ = state;
      // show the shot this state was marked on, so clicking through states walks the marks
      // instead of leaving you looking at whichever screenshot happened to be open
      const example = state && item.examples ? item.examples[state] : null;
      if (example && example.screenshot && example.screenshot !== ifaceShot) {
        ifaceShot = example.screenshot;
        ifaceRenderShots();
      }
      ifaceRenderList();
      ifaceDrawOverlay();
      ifaceStatus(!state
        ? `${item.name}: drag to set its position`
        : example
          ? `${item.name} - ${state}: marked on ${example.screenshot}. drag to replace it`
          : `${item.name} - ${state}: drag on a screenshot showing it`);
      return;
    }
    const name = (prompt('name for the new state (e.g. "channelling")') || '').trim();
    if (!name) return;
    const result = await api('/api/interface/add-state', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: item.name, state: name}),
    });
    if (result && result.error) { ifaceStatus(result.error); return; }
    ifaceState_ = name;
    ifaceRefresh(result);
  };
  container.appendChild(row);
}

function ifaceRenderStates() { /* states render inline in ifaceRenderList now */ }

function ifaceRenderShots() {
  const el = document.getElementById('iface-shots');
  el.innerHTML = '';
  // numbered rather than named: a dozen long filenames wrap the header away entirely. the name
  // is still the truth, so it stays on the tooltip and on every element that was marked here
  ifaceState.screenshots.forEach((name, index) => {
    const btn = document.createElement('div');
    btn.className = 'toggle' + (name === ifaceShot ? ' on' : '');
    btn.textContent = String(index + 1);
    btn.title = name;
    btn.onclick = () => { ifaceShot = name; ifaceRenderShots(); };
    el.appendChild(btn);
  });
  const img = document.getElementById('iface-img');
  img.src = ifaceShot ? `/interface-shot/${encodeURIComponent(ifaceShot)}` : '';
  img.onload = () => { ifaceResetView(); ifaceDrawOverlay(); };
  if (!ifaceShot) ifaceStatus('add a screenshot to start marking');
}

// ZOOM AND PAN, as a transform on the whole stage. a combo dot is a handful of pixels on a
// 3420-wide capture, so it has to be zoomable to aim at - and the first version panned by
// scrolling the container, which does nothing until the image already overflows. that is exactly
// backwards: at fit-to-width there is nothing to scroll and nothing pans.
//
// transforming the stage keeps the image and its overlay locked together, and leaves the canvas
// at the screenshot's NATURAL size - so a drawn rect is already in screenshot pixels and there is
// no conversion to get wrong at any zoom.
// the interface view now runs on the shared makePanZoom (review_ui/menu.js) - the cnn tab needed
// the same behaviour, and two copies of pointer-anchored zoom would drift apart
let ifaceView = null;

function ifaceApplyView() {
  ifaceView?.apply();
}

function ifaceResetView() {
  const img = document.getElementById('iface-img');
  ifaceView?.reset(img?.naturalWidth || 0, img?.naturalHeight || 0);
}

document.getElementById('iface-zoom-fit').onclick = ifaceResetView;

(function ifaceWireView() {
  const wrap = document.getElementById('iface-canvas-wrap');
  const stage = document.getElementById('iface-stage');
  if (!wrap || !stage) return;
  ifaceView = makePanZoom(wrap, stage, {
    onChange: z => {
      document.getElementById('iface-zoom-value').textContent = `${Math.round(z * 100)}%`;
    },
  });
})();

function ifaceSyncMode() {
  document.getElementById('iface-mode-rect').classList.toggle('on', ifaceMode === 'rect');
  document.getElementById('iface-mode-point').classList.toggle('on', ifaceMode === 'point');
}

function ifaceStatus(text) { document.getElementById('iface-status').textContent = text; }

// the image is displayed scaled to fit, so every drawn coordinate is converted back to the
// screenshot's OWN pixels before storing - a rect left in display space would be wrong by the
// zoom factor and nothing downstream would catch it
// the canvas sits at the screenshot's natural size inside a transformed stage, so a point in
// canvas coordinates IS a point in screenshot pixels. getBoundingClientRect reflects the applied
// transform, which is what turns a screen position back into one.
function ifaceCanvasPoint(evt) {
  const canvas = document.getElementById('iface-overlay');
  const box = canvas.getBoundingClientRect();
  const scale = box.width / (canvas.width || 1);
  return {x: (evt.clientX - box.left) / scale, y: (evt.clientY - box.top) / scale};
}

function ifaceDrawOverlay(live) {
  const img = document.getElementById('iface-img');
  const canvas = document.getElementById('iface-overlay');
  if (!img.naturalWidth) return;
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // the stage's scale applies to the stroke too, so undo it here or the dashes thicken as you
  // zoom out and vanish as you zoom in
  const k = 1;

  // thin and dotted: a solid 2px outline covered the very pixels being aimed at, so the bar
  // underneath could not be judged against its own edges
  const z = ifaceView ? ifaceView.zoom() : 1;
  ctx.lineWidth = Math.max(1, 1 / z);
  ctx.setLineDash([3 / z, 3 / z]);

  // EVERY MARK THAT LIVES ON THIS SCREENSHOT, not just the element's latest one. drawing only
  // item.rect meant the outline followed whatever was drawn LAST: mark "usable" on shot 1 and
  // "not usable" on shot 2, and shot 1 went blank because the element's own rect had moved to
  // shot 2. each state example carries its own rect and screenshot, so each is drawn where it
  // belongs and clicking through the states walks the marks.
  const shapes = [];
  (ifaceState ? ifaceState.elements : []).forEach(item => {
    const selected = item.name === ifaceSelected;
    (Object.entries(item.examples || {})).forEach(([state, example]) => {
      if (example.screenshot !== ifaceShot) return;
      shapes.push({item, example, selected, current: selected && ifaceState_ === state});
    });
    // the element's own position, when it was set without a state and is not already covered by
    // an example on this shot
    const covered = Object.values(item.examples || {}).some(e => e.screenshot === ifaceShot);
    if (item.screenshot === ifaceShot && !covered) {
      shapes.push({item, example: item, selected, current: selected && !ifaceState_});
    }
  });

  shapes.forEach(({item, example, selected, current}) => {
    // the state being marked right now is solid magenta; other marks on this element are dimmer
    // magenta; everything else is green. otherwise "which one am I editing" is unanswerable
    ctx.strokeStyle = current ? '#f0f' : selected ? 'rgba(255,0,255,0.45)' : 'rgba(120,220,120,0.8)';
    if (example.rect) {
      const [x, y, w, h] = example.rect;
      ctx.strokeRect(x * k, y * k, w * k, h * k);
    } else if (example.point) {
      ctx.beginPath();
      ctx.arc(example.point[0] * k, example.point[1] * k, (example.radius || 1) * k, 0, Math.PI * 2);
      ctx.stroke();
    }
    void item;
  });

  if (live) {
    ctx.strokeStyle = '#f0f';
    if (live.kind === 'rect') ctx.strokeRect(live.x, live.y, live.w, live.h);
    else {
      ctx.beginPath();
      ctx.arc(live.x, live.y, live.r, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
}

(function ifaceWireCanvas() {
  const canvas = document.getElementById('iface-overlay');
  if (!canvas) return;
  canvas.onmousedown = evt => {
    if (evt.shiftKey) return;  // shift+drag is a pan, handled on the wrapper
    if (!ifaceSelected) { ifaceStatus('pick an element on the left first'); return; }
    if (!ifaceShot) { ifaceStatus('add a screenshot first'); return; }
    ifaceDrag = ifaceCanvasPoint(evt);
  };
  canvas.onmousemove = evt => {
    if (!ifaceDrag) return;
    const {x, y} = ifaceCanvasPoint(evt);
    if (ifaceMode === 'rect') {
      ifaceDrawOverlay({kind: 'rect', x: Math.min(x, ifaceDrag.x), y: Math.min(y, ifaceDrag.y),
                        w: Math.abs(x - ifaceDrag.x), h: Math.abs(y - ifaceDrag.y)});
    } else {
      // point mode: the click is the CENTRE and the drag grows a radius, so a round combo dot can
      // be caught together with its frame instead of boxed
      ifaceDrawOverlay({kind: 'point', x: ifaceDrag.x, y: ifaceDrag.y,
                        r: Math.hypot(x - ifaceDrag.x, y - ifaceDrag.y)});
    }
  };
  canvas.onmouseup = async evt => {
    if (!ifaceDrag) return;
    const {x, y} = ifaceCanvasPoint(evt);
    let shape;
    if (ifaceMode === 'rect') {
      const w = Math.abs(x - ifaceDrag.x), h = Math.abs(y - ifaceDrag.y);
      if (w < 2 || h < 2) { ifaceDrag = null; ifaceDrawOverlay(); return; }
      shape = {rect: [Math.round(Math.min(x, ifaceDrag.x)), Math.round(Math.min(y, ifaceDrag.y)),
                      Math.round(w), Math.round(h)]};
    } else {
      const r = Math.hypot(x - ifaceDrag.x, y - ifaceDrag.y);
      if (r < 2) { ifaceDrag = null; ifaceDrawOverlay(); return; }
      shape = {point: [Math.round(ifaceDrag.x), Math.round(ifaceDrag.y)], radius: Math.round(r)};
    }
    ifaceDrag = null;
    const result = await api('/api/interface/mark', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: ifaceSelected, shape, screenshot: ifaceShot,
                            state: ifaceState_}),
    });
    if (result && result.error) { ifaceStatus(result.error); return; }
    // ONE PLACE THAT PAINTS THE STATE. the dot used to lag a mark behind, and every render path
    // reassigning ifaceState and then calling three render functions in its own order is how that
    // becomes possible at all. this refreshes everything from one object, every time.
    ifaceRefresh(result);
    ifaceStatus(ifaceState_ ? `marked ${ifaceSelected} - ${ifaceState_}` : `marked ${ifaceSelected}`);
    await ifaceSave(true);  // a mark is only useful once it reaches the file
    // and converge on what the SERVER holds, so a dropped or out-of-order response cannot leave
    // the list showing something the file does not say
    ifaceRefresh(await api('/api/interface'));
  };
})();

document.getElementById('iface-clear').onclick = async () => {
  // confirmed, because it throws away every position and example at once and there is no undo
  const total = ifaceState.elements.filter(e => e.marked).length;
  if (!confirm(`clear all marks?\n\n${total} marked element(s) lose their position and every `
               + `state example. the element list, keybinds and states are kept.`)) return;
  const result = await api('/api/interface/clear-marks', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}',
  });
  if (result && result.error) { ifaceStatus(result.error); return; }
  ifaceState_ = null;
  ifaceRefresh(result);
  ifaceStatus(`cleared ${total} marked element(s)`);
};

document.getElementById('iface-mode-rect').onclick = () => { ifaceMode = 'rect'; ifaceSyncMode(); };
document.getElementById('iface-mode-point').onclick = () => { ifaceMode = 'point'; ifaceSyncMode(); };

document.getElementById('iface-add').onclick = async () => {
  const name = prompt('name for the new element (e.g. "riposte", "pet health bar")');
  if (!name) return;
  // an ability gets the usable/not-usable states and can take a keybind; a readout gets neither.
  // asked rather than guessed from the name - "shadowmeld" is an ability and "pet health bar" is
  // not, and no naming rule tells them apart
  const ability = confirm(`is "${name}" an ability you press?\n\nOK = ability (gets states and a keybind)\nCancel = a readout you only look at`);
  const keybind = ability ? (prompt(`which key presses ${name}? (blank if none yet)`) || '') : '';
  const result = await api('/api/interface/add', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name, mode: ifaceMode, ability, keybind}),
  });
  if (result && result.error) { ifaceStatus(result.error); return; }
  ifaceRefresh(result);
};

document.getElementById('iface-add-probe').onclick = async () => {
  // a range probe is its own action rather than a third confirm() branch on iface-add: the
  // operator names it after the spell on the macro ("#showtooltip <spell>"), not after what it
  // shows, so the name IS the spell name the character profile's [range_probes] keys off
  const name = prompt('spell name on the #showtooltip macro (e.g. "Sinister Strike")');
  if (!name) return;
  const result = await api('/api/interface/add', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name, mode: ifaceMode, probe: true}),
  });
  if (result && result.error) { ifaceStatus(result.error); return; }
  ifaceRefresh(result);
};

// KEYBINDS DRAW FROM THE LEFT LIST, matched by name, and only the bindable ones - a health bar, a
// combo dot and a cast bar are readouts, and an entry no key can ever press is noise here.
function ifaceRenderKeybinds() {
  const el = document.getElementById('iface-keybinds');
  const rows = (ifaceState && ifaceState.keybinds) || [];
  // BOUND, NOT BINDABLE. this counted every row - bindable_elements returns what CAN take a key -
  // so nine abilities with no keys read "9 bound". both numbers, so the head is true in every state
  const bound = rows.filter(row => row.keybind).length;
  const text = !rows.length ? 'keybinds'
    : bound ? `${bound} of ${rows.length} bound` : `${rows.length} unbound`;
  el.innerHTML = `<span>${text}</span>`;
}

document.getElementById('iface-keybinds').onclick = evt => {
  const rows = (ifaceState && ifaceState.keybinds) || [];
  listMenu('which ability',
    rows.map(row => ({
      id: row.name, label: row.name,
      stats: row.keybind ? row.keybind : 'unbound',
    })),
    async item => { await ifaceSetKeybind(item.id); },
    {empty: 'no abilities yet - mark one on the list first'}
  ).openAt(evt.currentTarget);
};

async function ifaceSetKeybind(name) {
  const current = (ifaceState.keybinds.find(r => r.name === name) || {}).keybind || '';
  const keybind = prompt(`which key presses ${name}?`, current);
  if (keybind === null) return;
  const result = await api('/api/interface/keybind', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name, keybind}),
  });
  if (result && result.error) { ifaceStatus(result.error); return; }
  ifaceRefresh(result);
  ifaceStatus(`${name} -> ${keybind || 'unbound'}`);
};

document.getElementById('iface-remove').onclick = async () => {
  if (!ifaceSelected) { ifaceStatus('pick the element to remove first'); return; }
  const result = await api('/api/interface/remove', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name: ifaceSelected}),
  });
  if (result && result.error) { ifaceStatus(result.error); return; }
  ifaceStatus(`removed ${ifaceSelected}`);
  ifaceSelected = null;
  ifaceRefresh(result);
};

document.getElementById('iface-remove-shot').onclick = async () => {
  if (!ifaceShot) { ifaceStatus('no screenshot selected'); return; }
  const result = await api('/api/interface/remove-shot', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name: ifaceShot}),
  });
  if (result && result.error) { ifaceStatus(result.error); return; }
  const gone = ifaceShot;
  ifaceState = result;
  ifaceShot = ifaceState.screenshots[0] || null;
  ifaceRefresh(result);
  ifaceRenderShots();
  // the marks stay - a rect is still right after its picture is gone; what is lost is the proof
  ifaceStatus(result.orphaned && result.orphaned.length
    ? `removed ${gone} - these no longer show their evidence: ${result.orphaned.join(', ')}`
    : `removed ${gone}`);
};

document.getElementById('iface-upload').onclick =
  () => document.getElementById('iface-file').click();
document.getElementById('iface-file').onchange = async evt => {
  for (const file of evt.target.files) {
    const body = await file.arrayBuffer();
    const res = await fetch(`/api/interface/shot?name=${encodeURIComponent(file.name)}`,
                            {method: 'POST', body});
    const saved = await res.json();
    if (saved.name) ifaceShot = saved.name;
  }
  evt.target.value = '';
  await ifaceLoad();
};

// NO GENERATE BUTTON. naming a screen profile in the dropdown IS the act of creating one, so the
// button was a second way to do the same thing. the character file is written when the names are
// set and rewritten on every later mark. an existing screen profile is safe: the server leaves it
// exactly alone (its [readout] costs a calibration run), so an existing one is no reason to stop

async function ifaceSaveNames(changes) {
  // send the whole set, with whatever just changed applied on top - the server is the only place
  // these live, so a partial write would look like clearing the other two
  const result = await api('/api/interface/names', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      realm: ifacePicked('iface-realm'),
      character: ifacePicked('iface-character'),
      profile_name: ifacePicked('iface-profile'),
      ...(changes || {}),
    }),
  });
  // ONE PATH, so the control can never show one thing while the file says another - and neither
  // can the element list, which is what a partial repaint here used to leave stale
  if (result && !result.error) ifaceRefresh(result);
}

async function ifaceSave(quiet) {
  const profile = ifacePicked('iface-profile');
  if (!profile) return;
  const result = await api('/api/interface/generate', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      realm: ifacePicked('iface-realm'),
      character: ifacePicked('iface-character'),
      profile_name: profile,
      screenshot: ifaceShot,
      overwrite: true,
    }),
  });
  // a quiet save right after naming the profile has nothing marked and maybe no screenshot yet -
  // the server marks THAT as benign, and only that is swallowed on a quiet save. every other
  // failure - a screenshot the character folder does not have, a save that could not write -
  // must reach the status line even from the quiet save after a mark, or it fails silently again
  if (result.error) {
    if (!quiet || !result.benign) ifaceStatus(`save failed - ${result.error}`);
    return;
  }
  document.getElementById('iface-saved').textContent = result.message;
}

// ---- Clusters tab: group kept-but-unlabelled tiles by colour, flag the odd one out ----
// one flat list of every item, each carrying its own cluster's label/colour - server already
// orders clusters (manual-label buckets, then k-means by size), so flattening preserves that
// grouping visually without needing separate per-cluster boxes
let flatItems = [];

// ALWAYS THE UNSORTED POOL. this used to be a three-way picker (unsorted / library / none); the
// other two showed already-promoted composites, which is not what this tab is for - crops arrive
// here to be judged, and judged ones leave
const clusterSource = 'unsorted';

// bumped on every reload, so a tile re-cut under its unchanged url is fetched fresh rather than
// served from cache - see the img src below
let clusterCacheBust = Date.now();

async function loadClusters() {
  clusterCacheBust = Date.now();
  await refreshLibraryCount();
  const data = await api(`/api/clusters?source=${clusterSource}`);
  flatItems = data.clusters.flatMap(cluster =>
    cluster.items.map(item => ({...item, clusterLabel: cluster.label, meanRgb: cluster.mean_rgb}))
  );
  renderClusterFilter();
  renderClusterGrid();
}

// THE ACTIVE CLASS DEFINITION. everything that used to be a hardcoded axis - five generics, nine
// player classes, three brightness states, in this file AND in state.py - now comes from here.
let classDef = null;

async function loadClassDef(name) {
  const list = await api('/api/classdefs');
  const wanted = name || list.active;
  if (!wanted) { classDef = null; return; }
  const def = await api(`/api/classdefs?name=${encodeURIComponent(wanted)}`);
  classDef = def.error ? null : def;
  const head = document.getElementById('cluster-classdef');
  head.innerHTML = `<span>${classDef ? classDef.name : 'no classes yet'}</span>`;
}

const DIM_SEP = ' / ';

// a label written under a definition is its members joined; one written before definitions existed
// is the old "Name (state)" form. LEGACY IS PARSED, NEVER REWRITTEN - a judgement made under the
// old carve-up still means what it meant
function labelParts(label) {
  if (!label) return [];
  if (label.includes(DIM_SEP)) return label.split(DIM_SEP).map(p => p.trim());
  const state = /\((off_target|target|dim|selected)\)\s*$/.exec(label);
  const base = label.replace(/\s*\((off_target|target|dim|selected)\)\s*$/, '').trim();
  return state ? [base, state[1]] : [base];
}

// "none up to all acceptable" - zero toggles on in a group means that axis isn't filtered at
// all (everything passes), same as every option being on would; only a PARTIAL selection
// actually restricts anything
function matchesFilter(item) {
  const on = sel => [...document.querySelectorAll(sel)].filter(el => el.classList.contains('on')).map(el => el.dataset.name);
  const judged = on('#filter-judged .toggle');
  const parts = labelParts(item.clusterLabel.replace(/ \?$|\?$/g, '').trim());

  // ONE PASS PER DIMENSION, rather than three named axes. a member ticked anywhere restricts only
  // its own dimension, and a tile matches if every restricted dimension is satisfied
  for (const [index, dimension] of (classDef ? classDef.dimensions : []).entries()) {
    const picked = on(`#filter-dim-${index} .toggle`);
    if (picked.length && !picked.includes(parts[index])) return false;
  }
  // same "none selected = no filtering" rule as every other column, so by default judged tiles
  // still show - merely at the bottom. ticking it is how you review past judgements
  if (judged.length && !item.excluded) return false;
  return true;
}

function renderClusterFilter() {
  const dimsEl = document.getElementById('filter-dims');
  const built = dimsEl.children.length > 0;
  if (!built) {
    (classDef ? classDef.dimensions : []).forEach((dimension, index) => {
      if (index) dimsEl.insertAdjacentHTML('beforeend', '<div class="divider"></div>');
      const col = document.createElement('div');
      col.className = 'col';
      col.id = `filter-dim-${index}`;
      dimsEl.appendChild(col);
      dimension.members.forEach(m => buildToggle(col, m, true, renderClusterGrid));
    });
    if (!classDef) {
      dimsEl.innerHTML = '<span class="none">no class definition yet - make one under classes</span>';
    }
    buildToggle(document.getElementById('filter-judged'), 'not a class', true, renderClusterGrid);
  }
}

function renderClusterGrid() {
  const el = document.getElementById('cluster-groups');
  el.innerHTML = '';
  if (!flatItems.length) {
    document.getElementById('clusters-status').textContent = 'nothing kept yet';
    return;
  }
  // JUDGED TILES SINK. a "not a class" tile has been dealt with, so it drops below everything
  // still waiting on a decision. STABLE, deliberately: `sort` (#recluster) is server-side k-means
  // into k groups, a different axis entirely, and a stable partition keeps that grouping intact
  // WITHIN each of the two blocks instead of scrambling it.
  const visible = flatItems
    .filter(matchesFilter)
    .sort((a, b) => (a.excluded ? 1 : 0) - (b.excluded ? 1 : 0));
  // AFTER THE SORT, never before. this is the ON SCREEN order the net and the range logic read -
  // a range through tiles the filter is hiding would select things the human cannot see, and
  // assigning this before the sink would do the same thing one step later, with the net catching
  // tiles by the positions they held before they moved
  clusterVisible = visible;
  document.getElementById('clusters-count').textContent =
    `${visible.length} / ${flatItems.length}`;
  document.getElementById('clusters-status').textContent = '';

  visible.forEach(item => {
    const cell = document.createElement('div');
    cell.className = 'cluster-item' + (item.excluded ? ' excluded' : '');
    cell.dataset.name = item.name;
    const thumb = item.source === 'library' ? 'library-thumb' : 'unsorted-thumb';
    // CACHE-BUSTED, because a re-cut tile keeps its url. the no-store header on the image routes
    // is the real fix, but it only helps a browser that asks again - this also defeats a copy
    // already cached from before that header existed, and needs no server restart to take effect
    // WHERE THIS TILE STANDS, on the tile. the cluster's label is a guess every member shows, so
    // the grid could not say which tiles had actually been judged - you had to open the popup one
    // at a time. green has a class, red is "not a class", grey is nobody has said yet
    const verdict = item.excluded ? 'is-negative' : (item.assigned ? 'is-class' : 'is-unset');
    // LAZY, OR THE GRID NEVER FINISHES. Every visible tile got an <img> at once, and a browser
    // opens about six connections to one origin - so a 5,000 tile pool asked for 98MB of png
    // through a six-lane door and everything past the first screenful sat as an empty box forever.
    // Balthazar Fitzpatrick: "I only see the first page of images loading, everything after the
    // boxes are there but empty." The cells are cheap; it is the pixels that are not, and the
    // browser is already able to fetch those only as they come into view.
    cell.innerHTML = `<div class="tile-viewport"><img loading="lazy" decoding="async"`
      + ` src="/${thumb}/${item.name}?v=${clusterCacheBust}"></div>`
      + `<span class="x-mark">&times;</span><span class="tile-dot ${verdict}"></span>`;
    const stands = item.excluded ? 'not a class' : (item.assigned || 'unjudged');
    cell.title = `${item.name} — ${stands}\nin cluster: ${item.clusterLabel}`
      + `\nclick to pick · cmd/ctrl+click to add`
      + `\nshift+drag a net over several · right-click to tag every picked`;
    // LIBRARY TILES ARE NOT PICKABLE - a promoted template has left the unsorted pool and its
    // decision record no longer backs it. Marked in the DOM so the shared selection component can
    // see it without being handed this tab's item list
    if (item.source === 'library') cell.dataset.library = '1';
    el.appendChild(cell);
  });
  // THE GRID WAS JUST REBUILT, so every cell is a fresh element with no selected class. The
  // selection itself survives - it is keyed by name, not by element - and this paints it back on
  clusterSel?.repaint();
  applyTileMode();
}

// ---- train tab -----------------------------------------------------------------------------
// the numbers here are deliberately NOT precision/recall. the labels are known incomplete (see
// vision/plate_dataset), so a recall figure would be measuring the labelling, not the model. what
// is honest at this data size is separation: how much brighter the heatmap is on a confirmed plate
// than over the frame generally.
let trainEpochs = 200, trainBatch = 8, trainPoll = null, trainFrames = [];
// the training window, in INPUT pixels - the frame is downscaled first (by the factor the server
// reports, 2 by default), so 256 here is 512 capture px. seeded from the server's floor for the
// BOUND set, not from a fixed number, and re-seeded whenever the set's box size changes
let trainCrop = 256, trainCropFloor = 0, trainCropBox = null, trainCropDownscale = 2;

// A WINDOW BELOW THE FLOOR CLIPS A PLATE at the jitter extremes and teaches a half-plate as a
// whole one. say so before the run rather than refusing after the click
function showCropNote() {
  const note = document.getElementById('train-crop-note');
  if (!note) return;
  const capture = trainCrop * trainCropDownscale;
  if (trainCropFloor && trainCrop < trainCropFloor) {
    const box = trainCropBox ? `${trainCropBox[0]}x${trainCropBox[1]} box` : 'this box size';
    note.textContent = `too small - a ${box} needs ${trainCropFloor}+`;
    note.classList.add('warn');
  } else {
    note.textContent = `${capture}px of capture`;
    note.classList.remove('warn');
  }
}

async function loadCropFloor() {
  const data = await api('/api/window-floor');
  if (data.error) return;
  const newBox = !trainCropBox || trainCropBox[0] !== data.box[0] || trainCropBox[1] !== data.box[1];
  trainCropFloor = data.floor;
  trainCropBox = data.box;
  trainCropDownscale = data.downscale || trainCropDownscale;
  // A NEW SET IS A NEW BOX, and the old window belongs to the old box. this used to be a max(), so
  // a window seeded from the library's ~343 px fallback box (860) outlived binding a 95 px set
  trainCrop = newBox ? (data.default || trainCrop) : Math.max(trainCrop, data.default || trainCrop);
  document.getElementById('train-crop').textContent = trainCrop;
  showCropNote();
}

function facts(el, pairs) {
  el.innerHTML = '';
  pairs.forEach(([k, v]) => {
    const key = document.createElement('span'); key.className = 'k'; key.textContent = k;
    const val = document.createElement('span'); val.className = 'v'; val.textContent = v;
    el.appendChild(key); el.appendChild(val);
  });
}

async function loadTrain() {
  const info = await api('/api/train-info');
  const el = document.getElementById('train-summary');
  if (info.error) { facts(el, [['problem', info.error]]); return; }
  facts(el, [
    // FOUR LINES, each one a thing that changes what you do next. the pool's contents and the
    // mining era's "ignored regions" are neither - and the old block reported the bound candidates
    // queue, so it said "(none)" and "0 of 0" with a set plainly open above it
    ['training on', info.set
      ? `${info.frames} frames - ${info.plates} plates, ${info.negatives} negatives`
      : 'nothing yet - open a training set or load weights'],
    ['classes', info.thinnest
      ? `${info.classes}, thinnest is ${info.thinnest.label} with ${info.thinnest.count}`
      : (info.set ? 'none labelled' : '-')],
    ['device', info.device],
    ['model', info.model_exists ? 'trained, ready to sweep' : 'none yet - train or load weights'],
  ]);
  await loadTrainFrames();
  refreshTrainStatus();
}

async function loadTrainFrames() {
  const data = await api('/api/train-frames');
  trainFrames = data.frames;
  const list = document.getElementById('train-frames');
  list.innerHTML = '';
  trainFrames.forEach(row => {
    const btn = document.createElement('div');
    btn.className = 'toggle train-frame';
    btn.innerHTML = `<span class="name">${row.name}</span>` +
      `<span class="stats">${row.plates} plate${row.plates === 1 ? '' : 's'}</span>`;
    btn.onclick = () => {
      [...list.children].forEach(c => c.classList.remove('on'));
      btn.classList.add('on');
      trainFrameIndex = row.index;
      trainFitViewport();
      trainShowOverlay(row.index, trainChannel,
        `${row.name} - heatmap in red, confirmed plates boxed green`);
    };
    list.appendChild(btn);
  });
  if (!trainFrames.length) {
    list.textContent = 'no frame has a confirmed plate yet - promote some from select';
    return;
  }
  // SHOW THE FIRST ONE IMMEDIATELY. the channel picker acts on whatever frame is rendered, and
  // with none rendered it fired no request at all - clicking a channel silently did nothing,
  // which reads as a broken control rather than as "pick a frame first"
  if (trainFrameIndex < 0) list.querySelector('.train-frame')?.click();
}

function drawLoss(history) {
  const canvas = document.getElementById('train-loss');
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.clientWidth || 400;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (history.length < 2) return;
  // log scale: focal loss falls by orders of magnitude, and on a linear axis every epoch after the
  // first few is a flat line pinned to the bottom
  const ys = history.map(v => Math.log10(Math.max(v, 1e-3)));
  const lo = Math.min(...ys), hi = Math.max(...ys), span = (hi - lo) || 1;
  ctx.strokeStyle = '#e8ddc3'; ctx.lineWidth = 2; ctx.beginPath();
  ys.forEach((y, i) => {
    const px = (i / (ys.length - 1)) * (canvas.width - 4) + 2;
    const py = canvas.height - 4 - ((y - lo) / span) * (canvas.height - 8);
    i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
  });
  ctx.stroke();
}

async function refreshTrainStatus() {
  const job = await api('/api/train-status');
  const text = document.getElementById('train-progress-text');
  const fill = document.getElementById('train-bar-fill');
  fill.style.width = job.epochs ? `${(job.epoch / job.epochs) * 100}%` : '0';
  if (job.error) text.textContent = job.error;
  else if (job.running) text.textContent = `epoch ${job.epoch} / ${job.epochs}  loss ${(job.loss ?? 0).toFixed(2)}`;
  else if (job.finished) text.textContent = `done - ${job.epochs} epochs, final loss ${(job.loss ?? 0).toFixed(2)}` +
    (job.weights ? `, saved as ${job.weights}` : '');
  else if (job.aborted) text.textContent = `aborted at epoch ${job.aborted} of ${job.epochs} - nothing saved`;
  else text.textContent = 'idle';
  // only a running job can be stopped
  document.getElementById('train-abort').classList.toggle('disabled', !job.running);
  drawLoss(job.history || []);

  const acc = document.getElementById('train-accuracy');
  if (job.separation) {
    const s = job.separation;
    facts(acc, [
      ['on a plate', s.plate.toFixed(3)],
      ['background (99th pct)', s.background.toFixed(3)],
      ['ratio', s.ratio ? `${s.ratio.toFixed(2)}x brighter` : '-'],
      ['sampled', `${s.samples} plates`],
    ]);
  } else if (!job.running) {
    facts(acc, [['not measured yet', 'train a model to see this']]);
  }

  // scored on frames the model never trained on, so this outranks separation above
  const held = document.getElementById('train-holdout');
  const pct = x => `${Math.round(x * 100)}%`;
  if (job.holdout) {
    const h = job.holdout;
    facts(held, [
      ['recall', `${pct(h.recall)} of ${h.with_plate} frames with a plate`],
      ['false positives', `${pct(h.false_positive_rate)} of ${h.without_plate} empty frames`],
      ['spurious boxes', `${h.spurious}`],
      ['verdict', `${h.passes ? 'PASS' : 'FAIL'} - ${h.beats_teacher ? 'beats' : 'does not beat'} the teacher`],
    ]);
  } else if (!job.running) {
    facts(held, [['not measured', 'needs a set with a val split']]);
  }

  // the heatmap under the mask is from before this run, so it is covered rather than captioned
  const busy = document.getElementById('train-busy');
  if (busy) busy.hidden = !job.running;

  if (job.running && !trainPoll) trainPoll = setInterval(refreshTrainStatus, 700);
  if (!job.running && trainPoll) {
    clearInterval(trainPoll); trainPoll = null;
    loadTrainFrames();
  }
}

document.querySelectorAll('[data-train-step]').forEach(btn => {
  const [field, delta] = btn.dataset.trainStep.split(':');
  btn.onclick = () => {
    if (field === 'epochs') trainEpochs = Math.max(10, Math.min(2000, trainEpochs + parseInt(delta, 10)));
    else if (field === 'crop') trainCrop = Math.max(64, Math.min(1024, trainCrop + parseInt(delta, 10)));
    else trainBatch = Math.max(1, Math.min(64, trainBatch + parseInt(delta, 10)));
    document.getElementById('train-epochs').textContent = trainEpochs;
    document.getElementById('train-batch').textContent = trainBatch;
    document.getElementById('train-crop').textContent = trainCrop;
    showCropNote();
  };
});

// ---- the hyperparameter popup ----------------------------------------------------------------
// learning rate and seed are train()'s own arguments; nothing could reach them, so every run this
// tool ever started used 3e-4 and seed 0. Blank means "use the default" rather than zero.
document.getElementById('train-config').onclick = () =>
  document.getElementById('train-config-popup').classList.toggle('hidden');
document.getElementById('train-config-close').onclick = () =>
  document.getElementById('train-config-popup').classList.add('hidden');

// stops at the next epoch boundary - smolsmort only reports once per epoch - and saves nothing
document.getElementById('train-abort').onclick = async () => {
  const button = document.getElementById('train-abort');
  if (button.classList.contains('disabled')) return;
  const res = await api('/api/train-abort', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
  document.getElementById('train-progress-text').textContent =
    res.error || 'aborting after this epoch';
};

document.getElementById('train-start').onclick = async () => {
  const note = document.getElementById('train-progress-text');
  // THE WINDOW WAS NEVER SENT. the stepper moved trainCrop and the request carried only epochs and
  // batch, so the server fell back to its own default and the dial did nothing at all
  if (trainCropFloor && trainCrop < trainCropFloor) {
    note.textContent = `window ${trainCrop} is below the ${trainCropFloor} floor - it would clip a plate`;
    return;
  }
  const rate = document.getElementById('train-rate').value.trim();
  const seed = document.getElementById('train-seed').value.trim();
  const weightsName = document.getElementById('train-name').value.trim();
  const res = await api('/api/train-start', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      epochs: trainEpochs,
      batch: trainBatch,
      crop: trainCrop,
      learning_rate: rate === '' ? null : Number(rate),
      seed: seed === '' ? null : Number(seed),
      name: weightsName === '' ? null : weightsName,
    }),
  });
  if (res.error) { note.textContent = res.error; return; }
  refreshTrainStatus();
};


// ---- vlm tab -------------------------------------------------------------------------------
// the cnn proposes peaks, the vlm judges them, and anything it is unsure about is FLAGGED rather
// than guessed at. a wrong verdict that goes unflagged is worse than no verdict: it becomes a
// training label.
let vlmFrames = 3, vlmPeaks = 4, vlmPoll = null;

async function loadVlm() {
  const info = await api('/api/vlm-info');
  facts(document.getElementById('vlm-summary'), [
    ['model', info.model],
    ['backend', info.backend],
    ['weights on disk', info.weights_cached ? 'yes' : 'not yet - first run downloads them'],
    ['device', info.device],
    ['trained cnn', info.model_trained ? 'yes' : 'none - train one first'],
  ]);
  refreshVlm();
}

function renderVerdicts(verdicts) {
  const list = document.getElementById('vlm-list');
  if (list.children.length === verdicts.length) return;
  list.innerHTML = '';
  verdicts.forEach(v => {
    const btn = document.createElement('div');
    btn.className = 'toggle vlm-row' + (v.needs_human ? ' flagged' : '');
    btn.innerHTML = `<span class="name">${v.frame} @ ${v.x},${v.y}</span>` +
      `<span class="stats">${v.label}${v.needs_human ? ' - check me' : ''}</span>`;
    btn.onclick = () => {
      [...list.children].forEach(c => c.classList.remove('on'));
      btn.classList.add('on');
      document.getElementById('vlm-view-label').textContent =
        `${v.frame} - "${v.raw}" (${v.seconds}s)`;
      document.getElementById('vlm-view-img').src = `/vlm-crop/${v.index}?t=${Date.now()}`;
    };
    list.appendChild(btn);
  });
  if (!verdicts.length) list.textContent = 'nothing verified yet';
}

async function refreshVlm() {
  const job = await api('/api/vlm-status');
  const text = document.getElementById('vlm-progress-text');
  document.getElementById('vlm-bar-fill').style.width =
    job.total ? `${(job.done / job.total) * 100}%` : '0';
  if (job.error) text.textContent = job.error;
  else if (job.running) text.textContent = `${job.done} / ${job.total} checked`;
  else if (job.finished) text.textContent = `done - ${job.done} checked` +
    (job.per_check ? `, ${job.per_check}s each` : '');
  else text.textContent = 'idle';

  const counts = job.counts || {};
  facts(document.getElementById('vlm-counts'), [
    ['on a nameplate', String(counts['plate'] ?? 0)],
    ['near miss', String(counts['near-miss'] ?? 0)],
    ['nothing there', String(counts['nothing there'] ?? 0)],
    ['needs a human', String(job.needs_human ?? 0)],
    ['seconds per check', job.per_check ? String(job.per_check) : '-'],
  ]);
  renderVerdicts(job.verdicts || []);

  if (job.running && !vlmPoll) vlmPoll = setInterval(refreshVlm, 1200);
  if (!job.running && vlmPoll) { clearInterval(vlmPoll); vlmPoll = null; }
}

document.querySelectorAll('[data-vlm-step]').forEach(btn => {
  const [field, delta] = btn.dataset.vlmStep.split(':');
  btn.onclick = () => {
    if (field === 'frames') vlmFrames = Math.max(1, Math.min(50, vlmFrames + parseInt(delta, 10)));
    else vlmPeaks = Math.max(1, Math.min(20, vlmPeaks + parseInt(delta, 10)));
    document.getElementById('vlm-frames').textContent = vlmFrames;
    document.getElementById('vlm-peaks').textContent = vlmPeaks;
  };
});

document.getElementById('vlm-start').onclick = async () => {
  const res = await api('/api/vlm-start', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({frames: vlmFrames, peaks: vlmPeaks}),
  });
  if (res.error) { document.getElementById('vlm-progress-text').textContent = res.error; return; }
  refreshVlm();
};

document.getElementById('recluster').onclick = loadClusters;

// ---- right-click popup: narrow down a mismatched tile's identity by hand ----
// five primitives x three states = the fifteen classes any detector must carry. Balthazar Fitzpatrick named
// them 2026-09-01: default is the plain plate, off_target is the faded one, target is the plate
// you currently have selected (it reads a little brighter)
const BRIGHTNESS = ['default', 'off_target', 'target'];

// a label carries its state in the name: "Hostile NPC (off_target)" -> off_target. the older
// (dim)/(selected) spellings still read, so a label minted before the rename is not orphaned
function brightnessOf(label) {
  if (!label) return 'default';
  if (label.includes('(off_target)') || label.includes('(dim)')) return 'off_target';
  if (label.includes('(target)') || label.includes('(selected)')) return 'target';
  return 'default';
}

const GENERICS = ['Hostile NPC', 'Tagged NPC', 'Friendly NPC', 'Neutral NPC', 'Player'];
let sweepRecording = null;
let sweepTimer = null;

document.getElementById('sweep-open').onclick = async () => {
  const data = await api('/api/sweep-recordings');
  const items = data.recordings.map(rec => ({
    id: rec.name, label: rec.name, on: rec.name === sweepRecording,
    stats: `${rec.frames} frames`,
  }));
  listMenu('sweep which recording', items, item => {
    sweepRecording = item.id;
    document.getElementById('sweep-open').innerHTML = `<span>${item.id}</span>`;
  }, {empty: 'no recordings with frames under sessions/'})
    .openAt(document.getElementById('sweep-open'));
};

// plate_model.pt is overwritten by every run - this keeps a named copy, with the class map that
// makes it readable, so a good model survives the next experiment
// EITHER open a set and train, OR load weights already trained. loading brings the class map
// with it, because a checkpoint read under the wrong map renames every channel silently
document.getElementById('train-load').onclick = async () => {
  const data = await api('/api/saved-checkpoints');
  listMenu('weights',
    data.weights.map(w => ({
      id: w.name, label: w.name,
      stats: `${w.classes.length} classes, ${w.kb} kB`,
    })),
    async item => {
      const res = await api('/api/load-checkpoint', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name: item.id}),
      });
      const text = document.getElementById('train-progress-text');
      if (res.error) { text.textContent = res.error; return; }
      await loadTrain();
      await loadTrainClasses();
      // AFTER the refresh - loadTrain repaints this line, so setting it first loses the message
      text.textContent = `loaded ${res.name} - ${res.classes.length} classes`;
    },
    {empty: 'no saved weights yet - train a model and press save weights'}
  ).openAt(document.getElementById('train-load'));
};

// saving asks where and as what, in a ui_base menu: the name comes prefilled, folders open in
// place, and everything stays under the checkpoints folder so the weights picker can find it again
document.getElementById('train-save').onclick = async () => {
  const text = document.getElementById('train-progress-text');
  let where = await api('/api/checkpoint-folders');
  if (where.error) { text.textContent = where.error; return; }
  let name = where.name;
  let menu = null;
  const open = async under => {
    where = await api('/api/checkpoint-folders?under=' + encodeURIComponent(under));
    menu.refresh(build());
  };
  const build = () => [
    {kind: 'field', label: 'name', value: name, onInput: value => { name = value; }},
    {
      kind: 'list',
      label: `folder  checkpoints/${where.here}`,
      empty: 'no folders below this one',
      items: [
        ...(where.parent !== null ? [{id: where.parent, label: '..'}] : []),
        ...where.folders.map(f => ({id: where.here ? `${where.here}/${f}` : f, label: `${f}/`})),
      ],
      onPick: item => open(item.id),
    },
    {
      kind: 'add', placeholder: 'new folder here', button: 'new folder',
      onAdd: value => {
        where = {...where, parent: where.here, here: where.here ? `${where.here}/${value}` : value,
                 folders: []};
        menu.refresh(build());
      },
    },
    {kind: 'buttons', buttons: [{label: 'save', tone: 'adds', onClick: async m => {
      const res = await api('/api/save-checkpoint', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name, folder: where.here}),
      });
      text.textContent = res.error
        ? res.error
        : `saved ${res.name} (${res.classes} classes, ${Math.round(res.bytes / 1024)} kB)`;
      if (!res.error) m.close();
    }}]},
  ];
  menu = new Menu({title: 'save weights', persistent: true, sections: build()});
  menu.openAt(document.getElementById('train-save'));
};

// THE FLOOR IS A CHOICE, NOT A CONSTANT. it was hardcoded at 0.5, and a model whose peaks top out
// at 0.33 then wrote empty candidates files and called it success
// how many proposals sit at or above each 0.01 step, from the last sweep. lets the slider answer
// "and how many is that" without a round trip - see _counts_above
let sweepAbove = null;

const sweepMinScore = makeSlider(document.getElementById('sweep-sliders'), {
  id: 'sweep-score', label: 'min score - a peak below this is not a proposal',
  min: 0, max: 1, step: 0.01, value: 0.5,
  onChange: () => sweepShowCount(),
});

// THE COUNT IS THE POINT OF MOVING THE SLIDER. it used to appear only after sending, so the
// threshold was chosen blind and "nothing changed" was indistinguishable from "nothing happened"
function sweepShowCount() {
  const send = document.getElementById('sweep-send');
  if (!sweepAbove) return;
  const n = sweepAbove[Math.round(sweepMinScore.value() * 100)] ?? 0;
  const total = sweepAbove[0] ?? 0;
  document.getElementById('sweep-progress-text').textContent =
    `${n} of ${total} proposals at or above ${sweepMinScore.value().toFixed(2)}`
    + (n ? ' - send them to select' : ' - lower the threshold');
  // nothing to send is nothing to press, and a live count makes that visible rather than a surprise
  if (send) send.classList.toggle('disabled', !n);
}

// WHAT THE SWEEP ANSWERED, in 2.5% buckets, drawn over the slider. it comes from the sweep itself
// rather than from the training frames, so it describes the recording being judged
function showSweepDistribution(job) {
  if (!job || !job.histogram) return;
  sweepMinScore.setDistribution(job.histogram);
  const el = document.querySelector('#sweep-sliders .slider-label');
  if (el) {
    el.textContent = `min score - what the last sweep answered across ${job.total} frames, `
      + `best peak ${job.highest}`;
  }
}

// THE THRESHOLD IS CHOSEN AFTER LOOKING, so sending is its own step. the sweep keeps everything
// above a low floor; this cuts the part above the slider into the pool and goes to judge it. move
// the slider and press again to try another cut - no second sweep
document.getElementById('sweep-send').onclick = async evt => {
  if (evt.currentTarget.classList.contains('disabled')) return;
  const text = document.getElementById('sweep-progress-text');
  const at = sweepMinScore.value();
  text.textContent = `sending everything at or above ${at.toFixed(2)}...`;
  const res = await api('/api/sweep-send', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({min_score: at}),
  });
  if (res.error) { text.textContent = res.error; return; }
  text.textContent = `sent ${res.sent} of ${res.of} proposals - ${res.tiles} tiles`;
  activateTab('cluster');
};

document.getElementById('sweep-percent').onclick = evt => {
  const el = evt.currentTarget;
  listMenu('how much of it to sweep',
    [10, 25, 50, 100].map(n => ({
      id: String(n), label: `${n}%`, on: el.dataset.value === String(n)})),
    item => { el.dataset.value = item.id; el.innerHTML = `<span>${item.id}%</span>`; }
  ).openAt(el);
};

document.getElementById('sweep-start').onclick = async () => {
  const text = document.getElementById('sweep-progress-text');
  if (!sweepRecording) { text.textContent = 'pick a recording first'; return; }
  const res = await api('/api/sweep-start', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      recording: sweepRecording,
      percent: Number(document.getElementById('sweep-percent').dataset.value),
      min_score: sweepMinScore.value(),
    }),
  });
  if (res.error) { text.textContent = res.error; return; }
  // a new sweep invalidates the last one's proposals, so the send goes back to grey
  document.getElementById('sweep-send').classList.add('disabled');
  if (sweepTimer) clearInterval(sweepTimer);
  sweepTimer = setInterval(pollSweep, 500);
};

async function pollSweep() {
  const job = await api('/api/sweep-status');
  const text = document.getElementById('sweep-progress-text');
  const fill = document.getElementById('sweep-bar-fill');
  if (job.total) fill.style.width = `${Math.round(job.done / job.total * 100)}%`;
  showSweepDistribution(job);
  if (job.running) {
    // the note names why nothing is moving. a sweep queued behind a training run sits at 0 / N,
    // which on its own is indistinguishable from a hang
    text.textContent = job.note || `${job.done} / ${job.total} frames`;
    return;
  }
  if (sweepTimer) clearInterval(sweepTimer);
  sweepTimer = null;
  // THE PROPOSALS GO TO discard / promote, not back here - say so, because a count alone
  // leaves you wondering where the work landed
  if (job.error) text.textContent = job.error;
  else if (job.finished) {
    fill.style.width = '100%';
    // A SWEEP THAT FOUND SOMETHING HAS SOMEWHERE TO GO. saying "now in select" and
    // leaving no way to get there made the next step a thing to remember rather than press
    const send = document.getElementById('sweep-send');
    if (send) send.classList.toggle('disabled', !job.found);
    sweepAbove = job.above || null;
    if (job.found && sweepAbove) {
      sweepShowCount();
    } else {
      text.textContent = job.found
        ? `${job.found} proposals found - set a threshold, then send`
        // an empty sweep says how far under the floor it was, rather than looking like success
        : `nothing above ${Number(job.min_score ?? 0).toFixed(2)} - the model's best peak was `
          + `${job.highest ?? 0} · lower the min score, or train on more labels`;
    }
  }
}

let trainChannel = -1;
let trainFrameIndex = -1;

// the same pan/zoom the interface tab uses, on the shared implementation. a plain drag pans here
// because nothing else claims the drag on this tab - unlike the interface, where it marks rects
let trainView = null;

function trainFitViewport() {
  if (trainView) return;
  const wrap = document.querySelector('.tab-panel[data-panel="train"] .viewport');
  const stage = document.getElementById('train-stage');
  if (!wrap || !stage) return;
  trainView = makePanZoom(wrap, stage, {
    panModifier: null,
    onChange: z => {
      document.getElementById('train-zoom-value').textContent = `${Math.round(z * 100)}%`;
    },
  });
  // the css already fits the frame (object-fit: contain on a stage that fills the pane), so reset
  // is 1x and centred - passing the natural size shrank it a second time, to 51%
  document.getElementById('train-zoom-fit').onclick = () => trainView.reset();
}
let trainClasses = [];

// TWO AXES, NOT A FLAT LIST. a class name is a primitive plus a state, so picking one from a flat
// list of fifteen means reading fifteen strings. the same two-column shape discard/promote already
// uses picks it in two clicks, and "none selected" means every channel - which is the honest
// default because the channels are independent sigmoids and their max is "something fires here"
function trainSelection() {
  const on = sel => [...document.querySelectorAll(sel)]
    .filter(e => e.classList.contains('on')).map(e => e.dataset.name);
  return {primitives: on('#train-primitives .toggle'), states: on('#train-states .toggle')};
}

// a channel matches when its primitive is ticked (or none are) and its state is ticked (or none)
function trainChannelsChosen() {
  const {primitives, states} = trainSelection();
  return trainClasses
    .map((name, index) => ({name, index}))
    .filter(c => {
      const base = c.name.replace(/ \((off_target|target)\)$/, '');
      const state = brightnessOf(c.name);
      return (!primitives.length || primitives.includes(base))
          && (!states.length || states.includes(state));
    })
    .map(c => c.index);
}

// THE MODEL DECLINES WHILE IT IS BUSY, and that is deliberate: a training run or a sweep holds it
// for minutes and waiting behind one would freeze the page, so the overlay route answers 204. What
// was NOT deliberate is what the page did with that - it had already pointed the <img> at the new
// url, so a 204 replaced the picture with a broken-image glyph and no words. Clicking a confirmed
// frame mid-training looked exactly like a broken feature.
//
// LOADED OFF SCREEN FIRST, so the visible image is only replaced by one that actually arrived -
// which is what the server's own comment always assumed ("the tab keeps showing the picture it
// already has").
function trainShowOverlay(index, channel, label) {
  const img = document.getElementById('train-view-img');
  const caption = document.getElementById('train-view-label');
  if (!img) return;
  const url = `/train-overlay/${index}?channel=${channel}&t=${Date.now()}`;
  const probe = new Image();
  probe.onload = () => {
    img.src = url;
    if (caption && label) caption.textContent = label;
  };
  probe.onerror = () => {
    if (caption) {
      caption.textContent = img.getAttribute('src')
        ? `${label || ''} - the model is busy training or sweeping, so this is the previous frame`
        : `${label || ''} - the model is busy training or sweeping; the heatmap appears when it finishes`;
    }
  };
  probe.src = url;
}

function trainRepaintOverlay() {
  const chosen = trainChannelsChosen();
  // one channel picked -> that channel; several or none -> the loudest across them (-1)
  trainChannel = chosen.length === 1 ? chosen[0] : -1;
  if (trainFrameIndex < 0) {
    // nothing on screen to repaint - take the first frame rather than returning silently
    document.querySelector('#train-frames .train-frame')?.click();
    return;
  }
  const shown = trainFrames.find(f => f.index === trainFrameIndex);
  trainShowOverlay(trainFrameIndex, trainChannel,
    shown ? `${shown.name} - heatmap in red, confirmed plates boxed green` : '');
}

function paintTrainHeads(data) {
  // THE SERVER IS THE ONE SOURCE. a set and a checkpoint are mutually exclusive answers to "what
  // is loaded", so both heads are repainted together from whichever the server reports - a handler
  // that wrote only its own head left the other naming something no longer in play
  const set = document.getElementById('train-open');
  const weights = document.getElementById('train-load');
  if (set) {
    set.innerHTML = data.set
      ? `<span>${data.weights ? `${data.set} (from saved weights)` : data.set}</span>`
      : '<span>training sets</span>';
  }
  if (weights) {
    weights.innerHTML = data.weights ? `<span>${data.weights}</span>` : '<span>weights</span>';
  }
  trainingSet = data.set || null;
}

async function loadTrainClasses() {
  const el = document.getElementById('train-channels');
  if (!el) return;
  const data = await api('/api/train-classes');
  paintTrainHeads(data);
  trainClasses = data.classes || [];
  el.innerHTML = '';
  trainChannel = -1;
  if (!trainClasses.length) {
    el.innerHTML = '<span class="none">open a training set to pick a channel</span>';
    return;
  }
  el.innerHTML = '<span class="field-label">channel</span>'
    + '<div class="col" id="train-primitives"></div>'
    + '<div class="divider"></div>'
    + '<div class="col" id="train-states"></div>';
  // the primitives OFFERED are the ones this set actually has, so the picker can never point at a
  // channel the model does not carry
  const bases = [...new Set(trainClasses.map(n => n.replace(/ \((off_target|target)\)$/, '')))];
  bases.forEach(n => buildToggle(document.getElementById('train-primitives'), n, true,
                                 trainRepaintOverlay));
  BRIGHTNESS.forEach(n => buildToggle(document.getElementById('train-states'), n, true,
                                      trainRepaintOverlay));
  trainRepaintOverlay();
}

const CLASSES = ['Druid', 'Hunter', 'Mage', 'Paladin', 'Priest', 'Rogue', 'Shaman', 'Warlock', 'Warrior'];
// cmd/ctrl+click toggles membership here without acting - a plain click still does its normal
// single-item exclude toggle untouched, so a growing selection never accidentally excludes
// anything. only the NEXT right-click consumes it, tagging every selected tile at once
// instead of one at a time - the whole point of selecting several visually-obvious mismatches
// THE GESTURES LIVE IN ui_base NOW, not here. Click picks, cmd/ctrl+click adds, shift+drag draws
// a net with edge autoscroll, and right-click targets what you pointed at rather than a stale
// selection - all of it in select.js, because every one of those behaviours was learned from a
// failure here and the next tool should inherit the behaviour AND the reasons.
//
// This file keeps only what is specific to this grid: which tiles are pickable, what a selection
// count looks like in the bar, and what a right-click opens.
let clusterSel = null;

// the tiles currently on screen, in the order they are drawn - select-all needs to know which of
// them are pickable, since a library tile is only here to be looked at
let clusterVisible = [];

// A SELECTION YOU CANNOT SEE THE SIZE OF IS ONE YOU CANNOT SAFELY ACT ON. right-click tags every
// selected tile at once, and the previous version showed no count anywhere - so "remove 2 dim
// warlocks" could quietly hit thirty.
function clusterShowSelection() {
  const count = clusterSel ? clusterSel.size() : 0;
  const label = document.getElementById('cluster-selected');
  const clear = document.getElementById('cluster-clear-sel');
  if (label) {
    label.textContent = count ? `${count} selected` : '';
    label.hidden = !count;
  }
  if (clear) clear.hidden = !count;
  const centre = document.getElementById('cluster-smart-center');
  if (centre) centre.hidden = !count;
}

// SMART CENTER MOVES, IT DOES NOT JUDGE. every picked tile is already a plate - saying otherwise is
// what the class tags are for - so this only asks WHERE in the crop it sits and re-cuts there.
// the report is per-tile and shown, because a locator that guessed badly should be visible rather
// than silently baked into the training set
async function smartCenterPicked() {
  if (!clusterSel || !clusterSel.size()) return;
  const names = clusterSel.selected();
  const status = document.getElementById('clusters-status');
  const button = document.getElementById('cluster-smart-center');
  button.classList.add('busy');
  status.textContent = `centring ${names.length}...`;
  try {
    const res = await api('/api/smart-center', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({names}),
    });
    if (res.error) { status.textContent = res.error; return; }
    const moved = res.moved || [];
    const shifted = moved.map(m => Math.max(Math.abs(m.dx), Math.abs(m.dy)));
    const worst = shifted.length ? Math.max(...shifted) : 0;
    status.textContent =
      `${moved.length} re-cut` +
      (moved.length ? ` (worst ${worst}px)` : '') +
      `, ${(res.unchanged || []).length} already centred` +
      ((res.skipped || []).length ? `, ${res.skipped.length} skipped` : '');
    if (moved.length) await loadClusters();
  } catch (err) {
    status.textContent = `could not centre: ${err}`;
  } finally {
    button.classList.remove('busy');
  }
}

function wireClusterBand() {
  const grid = document.getElementById('cluster-groups');
  if (!grid || clusterSel) return;
  clusterSel = makeSelection(grid, {
    itemSelector: '.cluster-item',
    isPickable: el => el.dataset.library !== '1',
    onChange: clusterShowSelection,
    onContext: (names, x, y) => openLabelPopup(names, x, y),
  });
}

wireClusterBand();

let labelMenu = null;
let labelTargets = [];

function buildToggle(container, name, multi, onchange) {
  const btn = document.createElement('div');
  btn.className = 'toggle label-option';
  btn.textContent = name;
  btn.dataset.name = name;
  btn.onclick = () => {
    if (multi) {
      btn.classList.toggle('on');
    } else {
      [...container.children].forEach(c => c.classList.remove('on'));
      btn.classList.add('on');
    }
    (onchange || updateLabelSubmit)();
  };
  container.appendChild(btn);
}

function updateLabelSubmit() {
  // ANY ONE ANSWER IS ENOUGH, because an unanswered dimension is not a gap any more - it writes
  // "n/a" and keeps its position, so the label still splits back into the right dimensions.
  // Demanding every dimension is what made apply unreachable on a definition still being built:
  // an empty dimension has no toggle, so no click could ever satisfy it.
  //
  // ALL-UNSET IS STILL REFUSED, here and again in label_for. A label that answers nothing would
  // become a channel meaning only "a tile exists".
  const dims = classDef ? classDef.dimensions : [];
  const answered = dims.some((_, index) => document.querySelector(`#label-dim-${index} .toggle.on`));
  document.getElementById('label-submit').classList.toggle('disabled', !answered);
}

function openLabelPopup(names, x, y) {
  labelTargets = names;
  // "fix alignment" jumps to one exact candidate in Found - meaningless for a multi-selection
  document.getElementById('label-fix-alignment').style.display = names.length === 1 ? '' : 'none';
  // a single target shows its real current label, split across the aligned columns; several
  // selected at once have no ONE current label worth showing (they may all differ), so every
  // column just shows a dash placeholder instead
  const current = names.length === 1
    ? flatItems.find(i => i.name === names[0])?.clusterLabel
    : null;
  const parts = labelParts(current);
  const dims = classDef ? classDef.dimensions : [];

  // ONE COLUMN PER DIMENSION, in the definition's own order, mirrored in the "current" row above
  const currentEl = document.getElementById('current-dims');
  const pickEl = document.getElementById('label-dims');
  currentEl.innerHTML = '';
  pickEl.innerHTML = '';
  if (!dims.length) {
    pickEl.innerHTML = '<span class="none">no class definition yet - make one under classes</span>';
  }
  dims.forEach((dimension, index) => {
    if (index) {
      currentEl.insertAdjacentHTML('beforeend', '<div class="divider"></div>');
      pickEl.insertAdjacentHTML('beforeend', '<div class="divider"></div>');
    }
    const shown = document.createElement('div');
    shown.className = 'col';
    shown.id = `current-dim-${index}`;
    shown.textContent = current ? (parts[index] || '') : '-';
    currentEl.appendChild(shown);

    const col = document.createElement('div');
    col.className = 'col';
    col.id = `label-dim-${index}`;
    pickEl.appendChild(col);
    // ONE MEMBER PER DIMENSION - radio, not checkbox. picking several used to be how you handed
    // the server a shortlist to resolve by colour, and that resolution is gone
    dimension.members.forEach(m => buildToggle(col, m, false));
    // an empty column is indistinguishable from a broken one, and this is a definition being
    // built rather than a fault - so it says which dimension still wants members
    if (!(dimension.members || []).length) {
      const empty = document.createElement('span');
      empty.className = 'none';
      empty.textContent = `${dimension.name || 'this axis'}: no members yet - writes n/a`;
      col.appendChild(empty);
    }
  });
  updateLabelSubmit();

  // Menu owns showing, positioning, clamping and dismissal now. it is measured after opening for
  // the same reason as before: the real size depends on how many toggles were just built
  const popup = document.getElementById('label-popup');
  labelMenu = new Menu({adopt: popup, onDismiss: clearLabelSelection}).openAt({x, y});

  // REAL BUG FOUND AND FIXED HERE: #label-current and the picker below it are two independent
  // flex rows - each column's width is driven only by its OWN content, and "-" is far narrower
  // than "Friendly NPC"/"Warlock", so the two rows' columns never lined up. Match them by
  // explicit width instead of relying on both rows happening to size the same way.
  dims.forEach((_, index) => {
    const picker = document.getElementById(`label-dim-${index}`);
    document.getElementById(`current-dim-${index}`).style.width =
      picker.getBoundingClientRect().width + 'px';
  });
}

// REAL BUG FOUND AND FIXED HERE: dismissing the popup WITHOUT applying (click outside, or
// escape) never cleared the selection - a cmd/ctrl+click selection built up for one action
// silently survived a cancelled popup and got swept into the NEXT right-click's targets too,
// on tiles that were never re-selected for it. "remove 2 dim warlocks, way more disappeared"
// was exactly this: a stale selection from an earlier attempt riding along unseen.
// the clearing this does is the reason it exists - see the comment above it. Menu calls it on
// EVERY dismissal, including escape and an outside click, which is what the hand-rolled mousedown
// listener here used to do for this one popup and no other
function closeLabelPopup() {
  labelMenu?.close();
}

function clearLabelSelection() {
  labelTargets = [];
  clusterSel?.clear();   // the component owns the painting; onChange updates the count
}

document.getElementById('label-submit').onclick = async () => {
  if (document.getElementById('label-submit').classList.contains('disabled')) return;
  // WHAT YOU PICKED IS WHAT IS WRITTEN. this used to send a shortlist of plausible identities for
  // the server to choose between by nearest measured RGB - which cannot mean anything for a
  // dimension you defined yourself, and quietly dropped any member it did not recognise
  const picked = {};
  (classDef ? classDef.dimensions : []).forEach((dimension, index) => {
    const on = document.querySelector(`#label-dim-${index} .toggle.on`);
    if (on) picked[dimension.name] = on.dataset.name;
  });
  // ONE REQUEST FOR THE WHOLE SELECTION, and the repaint happens whatever it answers - see the
  // negative button below for what fanning out cost
  const names = [...labelTargets];
  try {
    await api('/api/manual-label', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({names, definition: classDef.name, picked}),
    });
  } finally {
    closeLabelPopup();
    await loadClusters();
  }
};

// "NONE OF THESE CLASSES", named for what it MEANS to the training set: a candidate marked here
// becomes a hard negative, the one kind of background worth sampling deliberately, because a
// random window almost never contains the model's own mistake.
//
// DECLARATIVE - it SETS every target excluded rather than flipping each. it used to send a bare
// toggle per name, so a selection where some were already judged came out inverted: the judged
// ones re-included and the rest excluded, the opposite of what the press said.
//
// ONE REQUEST, AND THE GRID REPAINTS EVEN IF IT FAILS. This fanned out a POST per tile inside a
// Promise.all with no catch: eighty tiles meant eighty round trips through a browser that opens
// six connections at a time, and a single rejection skipped closeLabelPopup and loadClusters
// entirely - so the button looked dead while the writes that landed stayed on the server, which
// is exactly what a refresh then showed. Balthazar Fitzpatrick: "the button seems unresponsive
// and nothing happens, but when refreshing the cards are marked as that".
document.getElementById('label-negative').onclick = async () => {
  const targets = [...labelTargets];
  let marked = 0;
  try {
    const answer = await api('/api/exclude', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({names: targets, excluded: true}),
    });
    marked = (answer && answer.done ? answer.done.length : targets.length);
  } finally {
    closeLabelPopup();
    await loadClusters();
    document.getElementById('clusters-status').textContent =
      `${marked} marked as not a class - they promote as hard negatives`;
  }
};

// ---- fix alignment, in place ----
// this used to jump to the tune tab, which could only ever correct the dataset that happened to
// be BOUND - a pool tile from any other recording was unreachable. the aligner is mounted right
// here on the tile's own source crop instead, and the correction goes back to the drawn box
let poolAligner = null, poolAlignName = null;

function closeAlignPopup() {
  if (poolAligner) { poolAligner.destroy(); poolAligner = null; }
  poolAlignName = null;
  document.getElementById('align-popup').classList.add('hidden');
}

document.getElementById('align-cancel').onclick = closeAlignPopup;

document.getElementById('label-fix-alignment').onclick = async () => {
  const name = labelTargets[0];
  closeLabelPopup();
  const info = await api(`/api/pool-box?name=${encodeURIComponent(name)}`);
  const popup = document.getElementById('align-popup');
  const status = document.getElementById('align-status');
  if (info.error) {
    document.getElementById('clusters-status').textContent = info.error;
    return;
  }
  poolAlignName = name;
  document.getElementById('align-name').textContent = name;
  status.textContent = '';
  popup.classList.remove('hidden');

  // zoomed so a 1px nudge is visible - this is last-pixel work, and at 1:1 it is not
  const scale = 6;
  const viewport = document.getElementById('align-viewport');
  const img = document.getElementById('align-image');
  viewport.style.width = img.style.width = (info.bounds.width * scale) + 'px';
  viewport.style.height = img.style.height = (info.bounds.height * scale) + 'px';
  img.src = `/tile/${encodeURIComponent(name)}?v=${Date.now()}`;

  if (poolAligner) poolAligner.destroy();
  // NO PREVIEW HERE. the pool tile is cut with the set's median padding, not this rect, so a
  // live picture would be a different crop from the one that gets written - misleading, not helpful
  poolAligner = makeAligner({
    viewport,
    rect: {...info.rect},
    bounds: info.bounds,
    target: {...info.rect},
    scale,
    onChange: rect => {
      const dx = Math.round(rect.left - info.rect.left), dy = Math.round(rect.top - info.rect.top);
      status.textContent = (dx || dy) ? `moved ${dx >= 0 ? '+' : ''}${dx}, ${dy >= 0 ? '+' : ''}${dy}` : '';
    },
  });
  // centred, and measured after it is visible - the crop's size decides the panel's, so a
  // position worked out before the image is sized lands in the wrong place
  const box = popup.getBoundingClientRect();
  popup.style.left = `${Math.max(8, (window.innerWidth - box.width) / 2)}px`;
  popup.style.top = `${Math.max(8, (window.innerHeight - box.height) / 2)}px`;
};

document.getElementById('align-save').onclick = async () => {
  if (!poolAligner || !poolAlignName) return;
  const {left, top} = poolAligner.rect;
  const name = poolAlignName;
  const res = await api('/api/realign-pool-tile', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name, left, top}),
  });
  if (res.error) { document.getElementById('align-status').textContent = res.error; return; }
  closeAlignPopup();
  await loadClusters();
  document.getElementById('clusters-status').textContent = `re-cut ${name}`;
};

// wasd nudging only while the aligner is open, and only when nothing is being typed into - the
// slug field is a text input sitting in the same tab
document.addEventListener('keydown', evt => {
  if (!poolAligner) return;
  // w/a/s/d are WoW's movement keys. while a panel floats over the game a stray keystroke reaching
  // the browser must not nudge a crop's alignment - see the tab-strip guard for the same failure
  if (poppedOutDocs.size) return;
  const active = document.activeElement;
  if (active && (active.tagName === 'INPUT' || active.tagName === 'SELECT')) return;
  if (evt.key === 'Escape') { closeAlignPopup(); return; }
  if (['w', 'a', 's', 'd'].includes(evt.key)) {
    evt.preventDefault();
    poolAligner.nudge(evt.key);
  }
});

// ---- load tab: pick which dataset the server's ReviewState is bound to, no restart needed ----

// ---- scripts tab: compose a command line, never run it from here ----
// deliberately no run button. several of these record input or drive the game, and a tool with
// that reach must be started deliberately in a terminal, not by a stray click in a browser
let scriptRows = [];

async function loadScripts() {
  const data = await api('/api/script-list');
  scriptRows = data.scripts;
  scriptCategory = null; scriptSubcategory = null;
  renderScriptColumns();
}

// picking in one column filters the columns to its right - the same cascade the label popup uses
let scriptCategory = null, scriptSubcategory = null;

function fillColumn(el, values, selected, onPick) {
  el.innerHTML = '';
  values.forEach(v => {
    const btn = document.createElement('div');
    btn.className = 'toggle' + (v === selected ? ' on' : '');
    btn.textContent = v;
    btn.onclick = () => onPick(v === selected ? null : v);
    el.appendChild(btn);
  });
}

function renderScriptColumns() {
  const uniq = (rows, key) => [...new Set(rows.map(r => r[key]))].sort();

  fillColumn(document.getElementById('script-categories'),
    uniq(scriptRows, 'category'), scriptCategory,
    v => { scriptCategory = v; scriptSubcategory = null; renderScriptColumns(); });

  // an unpicked column to the left means no filtering yet, so the right columns show everything
  const inCat = scriptCategory ? scriptRows.filter(r => r.category === scriptCategory) : scriptRows;
  fillColumn(document.getElementById('script-subcategories'),
    uniq(inCat, 'subcategory'), scriptSubcategory,
    v => { scriptSubcategory = v; renderScriptColumns(); });

  const inSub = scriptSubcategory ? inCat.filter(r => r.subcategory === scriptSubcategory) : inCat;
  const namesEl = document.getElementById('script-names');
  namesEl.innerHTML = '';
  // the commands he types himself first, most used first; the rest are run by tools or agents
  const byUse = (a, b) => (a.user_rank ?? Infinity) - (b.user_rank ?? Infinity)
    || a.name.localeCompare(b.name);
  [...inSub].sort(byUse).forEach(row => {
    const btn = document.createElement('div');
    btn.className = 'toggle script-row' + (row.user_rank == null ? ' internal' : '');
    btn.innerHTML = `<span class="name">${row.name}</span><span class="stats">${row.summary || ''}</span>`;
    btn.onclick = () => {
      [...namesEl.children].forEach(c => c.classList.remove('on'));
      btn.classList.add('on');
      showScript(row);
    };
    namesEl.appendChild(btn);
  });
}

// ARGUMENTS ARE PICKED, NOT JUST LISTED. the command line is what gets copied into a terminal, so
// the optional flags have to be selectable - otherwise every run means retyping them from a display
// that already knows them. positionals are always in (argparse requires them) and are not toggles.
let currentScript = null;
// name -> the value typed for it. a Set only recorded THAT an argument was picked, so the command
// could only ever show "<value>" and had to be finished by hand in the terminal - which is the
// retyping the tab exists to avoid
const chosenArgs = new Map();

// a value only needs quoting if it would otherwise split or be eaten by the shell
function shellQuote(value) {
  if (value === '') return "''";
  return /^[A-Za-z0-9_@%+=:,./-]+$/.test(value) ? value : `'${value.replace(/'/g, `'\\''`)}'`;
}

function renderScriptCommand() {
  if (!currentScript) return;
  const parts = [`uv run ${currentScript.name}`];
  currentScript.args.filter(a => !a.name.startsWith('-')).forEach(a => {
    const typed = chosenArgs.get(a.name);
    parts.push(typed ? shellQuote(typed) : `<${a.name}>`);
  });
  currentScript.args.filter(a => a.name.startsWith('-') && chosenArgs.has(a.name)).forEach(a => {
    const typed = chosenArgs.get(a.name);
    // a flag is the whole argument; anything else carries its value
    if (a.takes_value === false) parts.push(a.name);
    else parts.push(`${a.name} ${typed ? shellQuote(typed) : `<value>`}`);
  });
  const text = parts.join(' ');
  document.getElementById('script-command').textContent = text;
  const missing = currentScript.args.filter(
    a => a.required && a.takes_value !== false && !chosenArgs.get(a.name),
  ).length;
  document.getElementById('script-status').textContent =
    missing ? `${missing} required value${missing === 1 ? '' : 's'} still to fill in` : '';
}

// the help, with what argparse knows about the argument added - type, default, choices - so a
// week later it says what to actually put there rather than only what it means
function argExplain(a) {
  const bits = [];
  if (a.help) bits.push(a.help);
  const facts = [];
  if (a.takes_value === false) facts.push('a flag - no value');
  else if (a.type) facts.push(`takes a ${a.type}`);
  if (a.default !== null && a.default !== undefined) facts.push(`default ${a.default}`);
  if (a.choices?.length) facts.push(`one of: ${a.choices.join(', ')}`);
  if (a.required) facts.push('REQUIRED');
  if (facts.length) bits.push(facts.join(' \u00b7 '));
  return bits.join('\n');
}

function scriptArgField(a) {
  // A NAME THAT EXISTS SHOULD NOT HAVE TO BE REMEMBERED. a character or a screen profile is
  // whatever has been created, so argparse cannot list it as choices - but the filesystem can, and
  // a run that fails on a name differing by one hyphen is the failure this prevents. The add row
  // stays, because the list is what exists rather than what is allowed.
  if (a.suggests?.length) return scriptArgPicker(a);

  const field = document.createElement('input');
  field.className = 'text-field arg-value';
  field.placeholder = a.default !== null && a.default !== undefined
    ? `${a.default}` : (a.type ? a.type : 'value');
  field.value = chosenArgs.get(a.name) || '';
  field.oninput = () => { chosenArgs.set(a.name, field.value); renderScriptCommand(); };
  return field;
}

function scriptArgPicker(a) {
  const head = document.createElement('div');
  head.className = 'toggle dropdown-head arg-value';
  const paint = () => {
    const chosen = chosenArgs.get(a.name);
    head.innerHTML = `<span${chosen ? '' : ' class="placeholder"'}>${chosen || 'pick one'}</span>`;
  };
  const take = value => {
    chosenArgs.set(a.name, value);
    paint();
    renderScriptCommand();
  };
  paint();
  head.onclick = () => new Menu({
    title: a.name.replace(/^-+/, ''),
    sections: [
      {kind: 'list', items: a.suggests.map(v => ({
        id: v, label: v, on: v === chosenArgs.get(a.name),
      })), onPick: item => take(item.id)},
      // anything not on disk yet is still a legal value - the list offers what exists, it does
      // not restrict what can be typed
      {kind: 'add', placeholder: 'or type another', button: 'use', onAdd: value => take(value)},
    ],
  }).openAt(head);
  return head;
}

function showScript(row) {
  currentScript = row;
  chosenArgs.clear();
  const box = document.getElementById('script-args');
  box.innerHTML = '';
  renderScriptCommand();
  if (!row.args.length) { box.innerHTML = '<div class="stat">takes no arguments</div>'; return; }
  // REQUIRED FIRST, and marked. it used to sort by whether the name had a leading dash, which
  // put --profile (required=True) in with the optional flags - so the one argument a run cannot
  // omit looked exactly like the twelve it can. argparse knows the difference; now so does this.
  const ordered = [...row.args].sort((x, y) => (y.required === true) - (x.required === true));
  ordered.forEach(a => {
    const required = a.required === true;
    const cell = document.createElement('div');
    cell.className = 'arg-cell';

    const tag = document.createElement('div');
    tag.className = required ? 'toggle arg-req on' : 'toggle arg-opt';
    tag.textContent = a.name;
    const explain = argExplain(a);
    if (explain) {
      const tip = document.createElement('span');
      tip.className = 'arg-tip';
      tip.textContent = explain;
      tag.appendChild(tip);
    }
    cell.appendChild(tag);

    // a required argument always needs its value; an optional one only once it is picked
    if (required && a.takes_value !== false) cell.appendChild(scriptArgField(a));
    if (!required) {
      tag.onclick = () => {
        if (chosenArgs.has(a.name)) {
          chosenArgs.delete(a.name);
          cell.querySelector('.arg-value')?.remove();
        } else {
          chosenArgs.set(a.name, '');
          if (a.takes_value !== false) {
            const field = scriptArgField(a);
            cell.appendChild(field);
            field.focus();
          }
        }
        tag.classList.toggle('on', chosenArgs.has(a.name));
        renderScriptCommand();
      };
    }
    box.appendChild(cell);
  });
}

document.getElementById('script-copy').onclick = async () => {
  const text = document.getElementById('script-command').textContent;
  try {
    await navigator.clipboard.writeText(text);
    document.getElementById('script-status').textContent = 'copied: ' + text;
  } catch (err) {
    // clipboard needs a secure context, which plain http on a loopback address is not everywhere
    document.getElementById('script-status').textContent = 'select and copy: ' + text;
  }
};

// ---- map tab: one zone, every layer, pan and zoom ------------------------------------------
// REPLACED THE NAVIGATION TAB'S TWO CANVASES. a walk map and a heat map side by side were two
// fixed-fit views of the same ground that could not be compared, because neither could be moved
// and they used different projections - one flipped y, the other did not.
//
// ONE PROJECTION FOR EVERYTHING. every layer this tool has - the client's own art, the normals,
// the height, the no-go mask, the hand-drawn markers, the walks - is already in zone percentage
// 0-100, so the canvas is sized to the zone's own rectangle and a point is x/100 of the width.
// pan and zoom are a css transform on top, which is why nothing below has to know about them.
let mapName = null;          // the open zone's directory name
let mapManifest = null;      // its georeference and layer list
let mapView = null;          // makePanZoom handle
let mapMarkers = null;       // {points, lines}, or null until fetched
const mapWalks = new Map();  // session name -> its /api/nav maps, for the recordings layer
const mapImages = new Map(); // layer -> Image
const mapShown = new Set();  // which layers are drawn
const mapAlpha = {};         // layer -> 0..1
// THICKNESS ONLY MEANS SOMETHING FOR THE LAYERS WE STROKE, but COLOUR ALSO APPLIES TO MASKS.
// a mask is a silhouette: one baked colour and an all-or-nothing alpha, so its rgb carries no
// information and retinting loses nothing. the other rasters are real pictures - height, slope,
// normals, the client art - and tinting those WOULD destroy what they encode
const mapColour = {fences: '', routes: '', recordings: ''};
const mapWidth = {fences: 2, routes: 2, recordings: 1.5};

// off manifest.kind, never a hardcoded list - the exporter decides what a layer is, and buildings
// arriving as a mask after this was written is exactly what a list here would have got wrong
const mapIsMask = layer => (((mapManifest || {}).layers || {})[layer] || {}).kind === 'mask';
const mapCanTint = layer => MAP_DRAWN.includes(layer) || mapIsMask(layer);

// RETINT FROM THE ALPHA ALONE. source-in keeps what is already drawn as a stencil and takes the
// fill's colour, so one fill recolours the silhouette without touching its shape. the scratch
// canvas is reused - a fresh one per layer per frame is a lot of garbage to make while panning
let mapTintCanvas = null;
function mapTinted(img, colour, w, h) {
  if (!mapTintCanvas) mapTintCanvas = document.createElement('canvas');
  const c = mapTintCanvas;
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  const g = c.getContext('2d');
  g.clearRect(0, 0, w, h);
  g.drawImage(img, 0, 0, w, h);
  g.globalCompositeOperation = 'source-in';
  g.fillStyle = colour;
  g.fillRect(0, 0, w, h);
  g.globalCompositeOperation = 'source-over';
  return c;
}

// the order they stack in, bottom first. a layer absent from the manifest simply never appears
// the order they stack in, bottom first. corridors sit UNDER nogo because they are a routing hint
// and nogo is an obstacle - where the two overlap, the thing that stops you wins the pixel
const MAP_LAYER_ORDER = ['art', 'overlays', 'height', 'normal', 'slope', 'water', 'area',
                         'corridors', 'stringy-nav', 'buildings', 'nogo'];

// A LAYER THIS LIST HAS NEVER HEARD OF STILL DRAWS. the exporter grows layers on its own schedule -
// corridors arrived after this was written - and iterating only the known names would have left a
// new one toggleable in the menu and invisible on the map, which is worse than not offering it
function mapRasterOrder(manifest) {
  const known = Object.keys((manifest && manifest.layers) || {});
  return [...MAP_LAYER_ORDER.filter(l => known.includes(l)),
          ...known.filter(l => !MAP_LAYER_ORDER.includes(l))];
}
const MAP_DRAWN = ['fences', 'routes', 'recordings', 'npcs'];   // drawn by us, not fetched as an image
// what each mask is baked as, so its picker opens on the colour already on screen rather than
// jumping to an unrelated default the moment it is touched
const MAP_MASK_TINT = {nogo: '#dc462d', water: '#3c82c8', buildings: '#966e46', corridors: '#ebb950'};
const MAP_LABELS = {nogo: 'navmesh no go', fences: 'user fences', routes: 'user routes',
                    art: 'wow art', normal: 'normal map', area: 'named areas', buildings: 'buildings', corridors: 'corridors',
                    'stringy-nav': 'stringy nav',
                    overlays: 'place names'};
const mapLabel = id => MAP_LABELS[id] || id;

function mapStatus(text) { document.getElementById('map-status').textContent = text; }

// ---- the npc layer ---------------------------------------------------------------------------
// GROUPING COLOURS, FILTERING REMOVES. the layer draws whatever the filter keeps, in the colour the
// grouping gives it, as the style says - so the two dropdowns never mean the same thing twice
let mapNpcs = null;           // {map, npcs, points} from /api/map-npcs, or null before it is asked
const npcVis = {group: 'level', style: 'dot'};
const npcFilter = {lo: 1, hi: 70, reactions: new Set(['hostile', 'neutral', 'friendly']), names: null};
// the character's own side: questie stores which faction an npc is friendly TO, so the same row is
// a quest giver to one player and a kill to the other
const NPC_SIDE = 'H';
const NPC_LEVEL_BANDS = [[1, 10], [11, 20], [21, 30], [31, 40], [41, 50], [51, 70]];
const NPC_BAND_COLOUR = ['#6fa8c9', '#5f9e5a', '#c9b83e', '#d9883e', '#c8483c', '#8f5fbf'];
const NPC_REACTION_COLOUR = {hostile: '#c8483c', neutral: '#d9a441', friendly: '#5f9e5a'};
const NPC_NAME_COLOUR = ['#c8483c', '#4f8fd9', '#5f9e5a', '#d9a441', '#9b5fbf', '#3fa9a0', '#d96f9e', '#8a6a2a'];
// two spawns of one kind closer than this (in zone percent) stand in the same camp
const NPC_CAMP_SPREAD = 3.2;

function npcReaction(npc) {
  const friendly = npc.friendly_to || '';
  if (friendly.includes('A') && friendly.includes('H')) return 'neutral';
  return friendly.includes(NPC_SIDE) ? 'friendly' : 'hostile';
}

function npcGroup(npc) {
  if (npcVis.group === 'reaction') {
    const reaction = npcReaction(npc);
    return {key: reaction, label: reaction, colour: NPC_REACTION_COLOUR[reaction]};
  }
  if (npcVis.group === 'name') {
    return {key: `n${npc.npc_id}`, label: npc.name, colour: NPC_NAME_COLOUR[npc.npc_id % NPC_NAME_COLOUR.length]};
  }
  const level = Math.round((npc.min_level + npc.max_level) / 2);
  const band = Math.max(0, NPC_LEVEL_BANDS.findIndex(([lo, hi]) => level >= lo && level <= hi));
  return {key: `lv${band}`, label: `${NPC_LEVEL_BANDS[band][0]}-${NPC_LEVEL_BANDS[band][1]}`, colour: NPC_BAND_COLOUR[band]};
}

// a null name set means "every name": a filter nobody has touched should not have to list 267 ids
const npcNamed = npc => !npcFilter.names || npcFilter.names.has(npc.npc_id);

function npcKept() {
  if (!mapNpcs || !mapNpcs.npcs) return [];
  return mapNpcs.npcs.filter(npc =>
    npcNamed(npc)
    && npcFilter.reactions.has(npcReaction(npc))
    && npc.max_level >= npcFilter.lo && npc.min_level <= npcFilter.hi);
}

async function mapLoadNpcs() {
  if (!mapName) return;
  try {
    mapNpcs = await api('/api/map-npcs?map=' + encodeURIComponent(mapName));
    if (mapNpcs.error) mapStatus(mapNpcs.error);
  } catch (err) {
    mapNpcs = {npcs: [], error: String(err)};
  }
}

// spawns of ONE kind that stand together are one camp, which is what a grind route walks to
function npcCamps(list) {
  const camps = [];
  for (const npc of list) {
    const taken = new Set();
    (npc.points || []).forEach((point, i) => {
      if (taken.has(i)) return;
      const group = [point];
      taken.add(i);
      npc.points.forEach((other, j) => {
        if (taken.has(j)) return;
        if (Math.hypot(other[0] - point[0], other[1] - point[1]) < NPC_CAMP_SPREAD) {
          group.push(other);
          taken.add(j);
        }
      });
      const x = group.reduce((sum, p) => sum + p[0], 0) / group.length;
      const y = group.reduce((sum, p) => sum + p[1], 0) / group.length;
      camps.push({npc, x, y, count: group.length,
                  spread: Math.max(...group.map(p => Math.hypot(p[0] - x, p[1] - y)))});
    });
  }
  return camps;
}

// LOADED ONCE PER URL, and a redraw is queued for when it arrives - an image that decodes after
// the draw that asked for it would otherwise leave a blank layer until something else moved
function mapImage(layer, url) {
  if (mapImages.has(layer)) return mapImages.get(layer);
  const img = new Image();
  img.onload = () => mapDraw();
  img.src = url;
  mapImages.set(layer, img);
  return img;
}

async function mapOpen(name) {
  const manifest = await api('/api/map-manifest?map=' + encodeURIComponent(name));
  if (manifest.error) { mapStatus(manifest.error); return; }
  mapName = name;
  mapManifest = manifest;
  mapImages.clear();
  mapMarkers = null;
  // a line in progress belongs to the map it was drawn on, not the one just opened
  fenceDrawing = null;
  fenceArmed = false;
  // so does a pinned yard - its numbers describe the zone that was open
  mapTipHide();
  fenceSelected = null;
  // ONLY ONE LAYER STARTS SHOWN. eight rasters stacked over each other is not a map of anything -
  // the normals alone bury the art - so the stack starts at the one layer that reads as a place and
  // every other is one click away.
  //
  // ART IS OPTIONAL, and exactly one zone proves it: the client publishes none for Azshara. Keyed
  // on art alone this opened that zone to a blank canvas with every toggle off, which reads as a
  // broken tab rather than as a missing layer. Fall through the draw order to whatever IS there.
  mapShown.clear();
  const present = mapRasterOrder(manifest);
  const first = present.includes('art') ? 'art' : present[0];
  if (first) mapShown.add(first);
  [...Object.keys(manifest.layers || {}), ...MAP_DRAWN].forEach(layer => {
    if (mapAlpha[layer] === undefined) mapAlpha[layer] = layer === 'art' ? 1 : 0.6;
  });
  document.getElementById('map-pick').innerHTML = `<span>${manifest.zone_name || name}</span>`;
  // the zone is on the map dropdown and the layers on theirs, so the status only carries problems
  mapStatus('');
  mapFit();
  await mapLoadMarkers();
  // a zone's npcs belong to that zone, so the old ones go the moment another map opens
  mapNpcs = null;
  npcFilter.names = null;
  if (mapShown.has('npcs')) await mapLoadNpcs();
  fenceRenderList();
  mapDraw();
}

// THE MARKER LAYER MAY NOT BE SERVED YET. markers are authored in game and the endpoint arrives
// with that work, so a 404 here is "nothing drawn", not a fault
async function mapLoadMarkers() {
  if (!mapName) return;
  try {
    mapMarkers = await api('/api/map-markers?map=' + encodeURIComponent(mapName));
  } catch (err) {
    mapMarkers = {points: [], lines: []};
  }
}

// ---- fence drawer: "create zones" slide-in, drawing go/no-go lines on the map --------------
// card ad408d22. one line is drawn at a time (fenceDrawing); finishing or cancelling clears it.
// points are zone percent, the same space mapMarkers already uses, so no conversion is needed
// to render a saved line and an in-progress one with the same code.
let fenceKind = 'go';       // go | nogo - which kind the NEXT started line gets
let fenceDrawing = null;    // {line_id, line_type, points: [[x,y],...]} or null
let fenceArmed = false;     // while true, a map click appends a point to fenceDrawing
let fenceSelected = null;   // line_id highlighted/edited in the list

function fenceStatus(text) { document.getElementById('fence-active-status').textContent = text; }

function fenceSetKind(kind) {
  fenceKind = kind;
  if (fenceDrawing) fenceDrawing.line_type = kind;
  document.getElementById('fence-kind-go').classList.toggle('active', kind === 'go');
  document.getElementById('fence-kind-nogo').classList.toggle('active', kind === 'nogo');
}

function fenceNewLine() {
  if (!mapMarkers) mapMarkers = {points: [], lines: []};
  const used = (mapMarkers.lines || []).map(l => l.line_id);
  const nextId = used.length ? Math.max(...used) + 1 : 1;
  fenceDrawing = {line_id: nextId, line_type: fenceKind, points: []};
  fenceArmed = true;
  // drawing owns the cursor from here, pinned box included
  mapTipHide();
  document.getElementById('fence-mark').classList.add('active');
  fenceStatus(`line ${nextId} (${fenceKind}) - click the map to drop points`);
  mapDraw();
}

function fenceToggleArm() {
  if (!fenceDrawing) { fenceNewLine(); return; }
  fenceArmed = !fenceArmed;
  if (fenceArmed) mapTipHide();
  document.getElementById('fence-mark').classList.toggle('active', fenceArmed);
  fenceStatus(fenceArmed
    ? `line ${fenceDrawing.line_id} (${fenceDrawing.line_type}) - marking armed`
    : `line ${fenceDrawing.line_id} (${fenceDrawing.line_type}) - marking paused`);
}

function fenceUndo() {
  if (!fenceDrawing || !fenceDrawing.points.length) return;
  fenceDrawing.points.pop();
  mapDraw();
}

function fenceCancel() {
  fenceDrawing = null;
  fenceArmed = false;
  document.getElementById('fence-mark').classList.remove('active');
  fenceStatus('no line in progress');
  mapDraw();
}

async function fenceFinish() {
  if (!fenceDrawing || fenceDrawing.points.length < 2 || !mapName) return;
  const min = document.getElementById('fence-level-min').value;
  const max = document.getElementById('fence-level-max').value;
  mapMarkers = await api('/api/map-fence-line', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      map: mapName,
      line_id: fenceDrawing.line_id,
      line_type: fenceDrawing.line_type,
      points: fenceDrawing.points,
      level_min: min === '' ? null : Number(min),
      level_max: max === '' ? null : Number(max),
    }),
  });
  fenceDrawing = null;
  fenceArmed = false;
  document.getElementById('fence-mark').classList.remove('active');
  fenceStatus('no line in progress');
  fenceRenderList();
  mapDraw();
}

async function fenceDeleteLine(lineId) {
  if (!mapName) return;
  mapMarkers = await api('/api/map-fence-delete', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({map: mapName, line_id: lineId}),
  });
  if (fenceSelected === lineId) fenceSelected = null;
  fenceRenderList();
  mapDraw();
}

async function fenceSaveLevelRange(lineId, minEl, maxEl) {
  if (!mapName || minEl.value === '' || maxEl.value === '') return;
  mapMarkers = await api('/api/map-marker-level-range', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      map: mapName, line_id: lineId, level_min: Number(minEl.value), level_max: Number(maxEl.value),
    }),
  });
  fenceRenderList();
  mapDraw();
}

function fenceRenderList() {
  const host = document.getElementById('fence-list');
  const lines = (mapMarkers && mapMarkers.lines) || [];
  const yards = mapManifest && mapManifest.yards;
  // same axes fence_area.py scales by: percent-x spans east-west, percent-y spans north-south
  const rect = yards ? {yards_per_pct_x: yards.east_west / 100, yards_per_pct_y: yards.north_south / 100} : null;
  host.innerHTML = '';
  for (const line of lines) {
    const row = document.createElement('div');
    row.className = 'fence-item' + (line.line_id === fenceSelected ? ' selected' : '');
    const area = line.is_area && rect ? fenceAreaYd2(line.points, rect) : null;
    const head = document.createElement('div');
    head.className = 'fence-item-row';
    head.innerHTML = `<b>#${line.line_id} ${line.line_type}</b><span>${line.points.length} pts`
      + (area !== null ? ` · ${Math.round(area)} yd²` : '') + `</span>`;
    row.appendChild(head);

    const controls = document.createElement('div');
    controls.className = 'fence-item-row';
    const minEl = document.createElement('input');
    minEl.type = 'number'; minEl.style.width = '52px'; minEl.value = line.level_min ?? '';
    const maxEl = document.createElement('input');
    maxEl.type = 'number'; maxEl.style.width = '52px'; maxEl.value = line.level_max ?? '';
    const saveBtn = document.createElement('button');
    saveBtn.className = 'toggle'; saveBtn.textContent = 'save range';
    saveBtn.onclick = () => fenceSaveLevelRange(line.line_id, minEl, maxEl);
    const selectBtn = document.createElement('button');
    selectBtn.className = 'toggle'; selectBtn.textContent = 'select';
    selectBtn.onclick = () => { fenceSelected = line.line_id; fenceRenderList(); mapDraw(); };
    const delBtn = document.createElement('button');
    delBtn.className = 'toggle'; delBtn.textContent = 'delete';
    delBtn.onclick = () => fenceDeleteLine(line.line_id);
    controls.append(minEl, maxEl, saveBtn, selectBtn, delBtn);
    row.appendChild(controls);
    host.appendChild(row);
  }
}

// same shoelace-on-zone-percent math as parent/nav/fence_area.py, kept here only so the list
// can show area without a round trip - the server-saved value is the one that ever persists
function fenceAreaYd2(pointsPct, rect) {
  if (pointsPct.length < 3) return 0;
  const yards = pointsPct.map(([x, y]) => [x * rect.yards_per_pct_x, y * rect.yards_per_pct_y]);
  let total = 0;
  for (let i = 0; i < yards.length; i++) {
    const [x0, y0] = yards[i];
    const [x1, y1] = yards[(i + 1) % yards.length];
    total += x0 * y1 - x1 * y0;
  }
  return Math.abs(total) / 2;
}

function fenceCanvasXY(evt) {
  const canvas = document.getElementById('map-canvas');
  const r = canvas.getBoundingClientRect();
  return [(evt.clientX - r.left) / r.width * 100, (evt.clientY - r.top) / r.height * 100];
}

(function wireFence() {
  const openBtn = document.getElementById('fence-open');
  const panel = document.getElementById('fence-panel');
  openBtn.onclick = () => {
    panel.classList.toggle('open');
    if (panel.classList.contains('open')) fenceRenderList();
  };
  document.getElementById('fence-close').onclick = () => panel.classList.remove('open');
  document.getElementById('fence-keyhelp').onclick = () =>
    document.getElementById('fence-keyhelp-overlay').classList.toggle('hidden');
  document.getElementById('fence-kind-go').onclick = () => fenceSetKind('go');
  document.getElementById('fence-kind-nogo').onclick = () => fenceSetKind('nogo');
  document.getElementById('fence-new').onclick = fenceNewLine;
  document.getElementById('fence-mark').onclick = fenceToggleArm;
  document.getElementById('fence-undo').onclick = fenceUndo;
  document.getElementById('fence-finish').onclick = fenceFinish;
  document.getElementById('fence-cancel').onclick = fenceCancel;
  fenceSetKind('go');

  document.getElementById('map-canvas').addEventListener('click', evt => {
    if (!fenceArmed || !fenceDrawing) return;
    const [x, y] = fenceCanvasXY(evt);
    fenceDrawing.points.push([x, y]);
    fenceStatus(`line ${fenceDrawing.line_id} (${fenceDrawing.line_type}) - ${fenceDrawing.points.length} points`);
    mapDraw();
  });

  document.addEventListener('keydown', evt => {
    if (!panel.classList.contains('open')) return;
    if (evt.key === 'Backspace' && fenceDrawing) { evt.preventDefault(); fenceUndo(); }
    else if (evt.key === 'Enter' && fenceDrawing) { evt.preventDefault(); fenceFinish(); }
    else if (evt.key === 'Escape' && fenceDrawing) { evt.preventDefault(); fenceCancel(); }
  });
})();

// ---- the yard under the cursor -------------------------------------------------------------
// the page draws the layers but cannot read them: a mask is a tinted pixel, height is 16-bit
// scalar, an area is a colour standing in for an id. So the facts come from /api/map-point and
// this only renders them. A click pins the box so the numbers can be read and copied.
//
// WHILE THE FENCE DRAWER IS DRAWING, THERE IS NO TOOLTIP AT ALL. Drawing owns the cursor - every
// move is aiming a point and every click drops one - and a box following the aim is in the way.
const MAP_TIP_DELAY = 80;       // ms of stillness before asking; the cursor moves far more often
const MAP_MASK_WORDS = {nogo: 'no-go', water: 'water', buildings: 'buildings / indoors',
                        corridors: 'corridor'};
let mapTipPinned = false;
let mapTipBusy = false;         // one request in flight at a time - the rest are simply skipped
let mapTipTimer = null;
let mapTipAt = null;            // [clientX, clientY] of the last move, for placing the box

// drawing mode is a line in progress OR marking armed: both mean the next click belongs to the
// fence drawer, so neither may also be a pin
function mapTipSuppressed() { return !!fenceDrawing || fenceArmed; }

function mapTipHide() {
  mapTipPinned = false;
  const tip = document.getElementById('map-tip');
  tip.classList.add('hidden');
  tip.classList.remove('pinned');
  if (mapTipTimer) { clearTimeout(mapTipTimer); mapTipTimer = null; }
}

// beside the cursor, never under it, and flipped to the other side near a window edge so the box
// is never half off screen
function mapTipPlace(clientX, clientY) {
  const tip = document.getElementById('map-tip');
  const gap = 18;
  const box = tip.getBoundingClientRect();
  let left = clientX + gap;
  let top = clientY + gap;
  if (left + box.width > window.innerWidth - 8) left = clientX - gap - box.width;
  if (top + box.height > window.innerHeight - 8) top = clientY - gap - box.height;
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${Math.max(8, top)}px`;
}

function mapTipRows(facts) {
  const rows = [];
  const row = (key, value) => rows.push(`<div><span class="tip-key">${key}</span> ${value}</div>`);
  row('at', `${facts.x_pct.toFixed(3)} , ${facts.y_pct.toFixed(3)} %`);
  if (facts.yards) row('yards', `${facts.yards.east_west} E , ${facts.yards.north_south} S`);
  if (facts.height) row('ground', `${facts.height.yards} yd <span class="tip-key">(${facts.height.source})</span>`);
  if (facts.slope_degrees !== undefined) row('slope', `${facts.slope_degrees}&deg;`);
  if (facts.masks) {
    const on = Object.keys(MAP_MASK_WORDS).filter(k => facts.masks[k]).map(k => MAP_MASK_WORDS[k]);
    row('ground is', on.length ? on.join(', ') : 'clear');
  }
  if (facts.area) row('area', facts.area);
  (facts.inside || []).forEach(line => {
    const levels = (line.level_min !== null && line.level_min !== undefined)
      ? ` <span class="tip-key">lvl ${line.level_min}-${line.level_max}</span>` : '';
    row('inside', `${line.line_type} line ${line.line_id}${levels}`);
  });
  return rows.join('');
}

async function mapTipShow(clientX, clientY) {
  if (!mapName || mapTipBusy || mapTipSuppressed()) return;
  const [x, y] = fenceCanvasXY({clientX, clientY});
  if (x < 0 || x > 100 || y < 0 || y > 100) { mapTipHide(); return; }
  mapTipBusy = true;
  let facts;
  try {
    facts = await api(`/api/map-point?map=${encodeURIComponent(mapName)}&x=${x}&y=${y}`);
  } catch (err) {
    mapTipBusy = false;
    return;
  }
  mapTipBusy = false;
  // the drawer may have armed while this was in flight, and then the answer is no longer wanted
  if (facts.error || mapTipSuppressed()) return;
  const tip = document.getElementById('map-tip');
  tip.innerHTML = mapTipRows(facts) +
    `<div class="tip-note">${mapTipPinned ? 'esc or click to unpin' : 'click to pin'}</div>`;
  tip.classList.remove('hidden');
  mapTipPlace(clientX, clientY);
}

(function wireMapTip() {
  const wrap = document.getElementById('map-wrap');
  const canvas = document.getElementById('map-canvas');

  canvas.addEventListener('mousemove', evt => {
    if (mapTipSuppressed()) { mapTipHide(); return; }
    mapTipAt = [evt.clientX, evt.clientY];
    if (mapTipPinned) return;
    if (mapTipTimer) clearTimeout(mapTipTimer);
    mapTipTimer = setTimeout(() => mapTipShow(...mapTipAt), MAP_TIP_DELAY);
  });

  // a click while drawing belongs to the fence drawer, so the pin never competes for it
  canvas.addEventListener('click', evt => {
    if (mapTipSuppressed()) return;
    mapTipPinned = !mapTipPinned;
    document.getElementById('map-tip').classList.toggle('pinned', mapTipPinned);
    mapTipAt = [evt.clientX, evt.clientY];
    if (mapTipPinned) mapTipShow(evt.clientX, evt.clientY);
    else mapTipHide();
  });

  wrap.addEventListener('mouseleave', () => { if (!mapTipPinned) mapTipHide(); });
  document.addEventListener('keydown', evt => {
    if (evt.key === 'Escape' && mapTipPinned) mapTipHide();
  });
})();

// THE CANVAS IS THE ZONE. its aspect is the zone's real rectangle, read from the manifest, so a
// square metre of ground is square on screen - which a container-shaped canvas cannot promise
function mapFit() {
  const canvas = document.getElementById('map-canvas');
  const yards = (mapManifest && mapManifest.yards) || {};
  const aspect = (yards.east_west && yards.north_south) ? yards.east_west / yards.north_south : 1.5;
  const width = 1536;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round((width / aspect) * ratio);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${Math.round(width / aspect)}px`;
  if (!mapView) {
    mapView = makePanZoom(document.getElementById('map-wrap'), canvas,
      {panModifier: null, fit: 'contain'});
  }
  mapView.reset(width, width / aspect);
}

function mapDraw() {
  const canvas = document.getElementById('map-canvas');
  const ctx = canvas.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!mapManifest) return;
  const ratio = window.devicePixelRatio || 1;
  ctx.scale(ratio, ratio);
  const w = canvas.width / ratio, h = canvas.height / ratio;
  const at = (x, y) => [(x / 100) * w, (y / 100) * h];
  const zoom = mapView ? mapView.zoom() : 1;
  const pen = Math.max(0.5, 1 / zoom);

  // EACH RASTER COVERS THE WHOLE ZONE, whatever its own pixel size - the manifest's `size` is
  // resolution, not extent - so every one is stretched to the same rectangle
  for (const layer of mapRasterOrder(mapManifest)) {
    const meta = (mapManifest.layers || {})[layer];
    if (!meta || !mapShown.has(layer) || !mapAlpha[layer]) continue;
    const img = mapImage(layer, (mapManifest.urls || {})[layer]);
    if (!img.complete || !img.naturalWidth) continue;
    ctx.globalAlpha = mapAlpha[layer];
    const tint = mapColour[layer];
    ctx.drawImage(tint && mapIsMask(layer) ? mapTinted(img, tint, w, h) : img, 0, 0, w, h);
  }
  ctx.globalAlpha = 1;

  const styles = getComputedStyle(document.documentElement);
  const lichen = styles.getPropertyValue('--lichen').trim() || '#c7ed5f';
  const red = styles.getPropertyValue('--stone-red').trim() || '#996b62';
  const dim = styles.getPropertyValue('--text-dim').trim() || '#8b8b90';

  // A FENCE IS AN AREA, A ROUTE IS A LINE YOU WALK, and the server tells the two apart with
  // is_area rather than by counting points - a three-point route and a triangle look identical
  // from here. a waypoint is a line of one, so it falls out of the same loop as a dot.
  const drawLines = (which, colourFor) => {
    ctx.globalAlpha = mapAlpha[which];
    ctx.lineWidth = (mapWidth[which] ?? 2) * pen;
    for (const line of (mapMarkers && mapMarkers.lines) || []) {
      const wantArea = which === 'fences';
      if (!!line.is_area !== wantArea) continue;
      const points = line.points || [];
      ctx.strokeStyle = ctx.fillStyle = mapColour[which] || colourFor(line);
      if (points.length < 2) {
        const [px, py] = at(points[0][0], points[0][1]);
        ctx.beginPath();
        ctx.arc(px, py, 4 * pen, 0, Math.PI * 2);
        ctx.fill();
        continue;
      }
      ctx.beginPath();
      points.forEach(([x, y], i) => { const [px, py] = at(x, y); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
      if (wantArea) ctx.closePath();
      ctx.stroke();
    }
  };

  // UNDER THE DRAWN LINES: a fence or a route is what the operator is working on, and a rash of
  // spawn points must never sit on top of it
  if (mapShown.has('npcs') && mapAlpha.npcs) {
    mapDrawNpcs(ctx, at, pen, w, h);
  }

  if (mapShown.has('fences') && mapAlpha.fences) {
    drawLines('fences', line => (line.line_type === 'nogo' ? red : lichen));
  }
  if (mapShown.has('routes') && mapAlpha.routes) {
    drawLines('routes', () => lichen);
  }

  // THE LINE IN PROGRESS, drawn in a colour neither go nor nogo uses so it never reads as saved -
  // it auto-closes back to the first point the same way a finished line does, since that preview
  // is the whole point of drawing it this way (google-maps-distance style)
  if (fenceDrawing && fenceDrawing.points.length) {
    ctx.strokeStyle = ctx.fillStyle = '#f2c14e';
    ctx.lineWidth = 2 * pen;
    const pts = fenceDrawing.points;
    ctx.beginPath();
    pts.forEach(([x, y], i) => { const [px, py] = at(x, y); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
    if (pts.length >= 3) ctx.closePath();
    ctx.stroke();
    pts.forEach(([x, y]) => {
      const [px, py] = at(x, y);
      ctx.beginPath();
      ctx.arc(px, py, 3 * pen, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  if (mapShown.has('recordings') && mapAlpha.recordings) {
    ctx.globalAlpha = mapAlpha.recordings;
    ctx.strokeStyle = mapColour.recordings || dim;
    ctx.lineWidth = (mapWidth.recordings ?? 1.5) * pen;
    for (const maps of mapWalks.values()) {
      for (const walked of maps) {
        if (mapManifest.map_id && walked.map_id !== mapManifest.map_id) continue;
        const path = walked.path || [];
        if (path.length < 2) continue;
        ctx.beginPath();
        path.forEach(([x, y], i) => { const [px, py] = at(x, y); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
        ctx.stroke();
      }
    }
  }
  ctx.globalAlpha = 1;
}

// ONE STYLE PER DRAW, the grouping deciding every colour. dots are the truth, circles are the
// camps a route would walk to, and the density map answers "is there anything here at all" without
// drawing two thousand of anything
function mapDrawNpcs(ctx, at, pen, w, h) {
  const list = npcKept();
  if (!list.length) return;
  ctx.globalAlpha = mapAlpha.npcs;

  if (npcVis.style === 'dot') {
    const radius = (mapWidth.npcs ?? 2) * pen * 1.6;
    for (const npc of list) {
      ctx.fillStyle = npcGroup(npc).colour;
      for (const [x, y] of npc.points || []) {
        const [px, py] = at(x, y);
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fill();
      }
      // an elite is a different fight, not a different colour - the ring says so without spending
      // the grouping's own palette on it
      if (npc.rank > 0) {
        ctx.strokeStyle = '#f0e6d2';
        ctx.lineWidth = pen;
        for (const [x, y] of npc.points || []) {
          const [px, py] = at(x, y);
          ctx.beginPath();
          ctx.arc(px, py, radius + 2 * pen, 0, Math.PI * 2);
          ctx.stroke();
        }
      }
    }
  } else if (npcVis.style === 'circle') {
    for (const camp of npcCamps(list)) {
      const [px, py] = at(camp.x, camp.y);
      const colour = npcGroup(camp.npc).colour;
      const radius = Math.max(6 * pen, (camp.spread / 100) * w + 5 * pen + camp.count * pen);
      ctx.fillStyle = colour;
      ctx.globalAlpha = mapAlpha.npcs * 0.2;
      ctx.beginPath();
      ctx.arc(px, py, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = mapAlpha.npcs;
      ctx.strokeStyle = colour;
      ctx.lineWidth = (mapWidth.npcs ?? 1.5) * pen;
      ctx.beginPath();
      ctx.arc(px, py, radius, 0, Math.PI * 2);
      ctx.stroke();
      if (camp.count > 1) {
        ctx.fillStyle = '#f0e6d2';
        ctx.font = `${Math.round(11 * pen)}px "IBM Plex Mono", monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(camp.count), px, py);
      }
    }
  } else {
    // ONE FIELD PER GROUP so the grouping still decides the colour, all scaled against the busiest
    // cell in the zone - a shared scale is what makes two groups comparable by eye
    const cell = 18;
    const cols = Math.ceil(w / cell), rows = Math.ceil(h / cell);
    const fields = new Map();
    for (const npc of list) {
      const group = npcGroup(npc);
      if (!fields.has(group.key)) fields.set(group.key, {group, grid: new Float32Array(cols * rows)});
      const {grid} = fields.get(group.key);
      for (const [x, y] of npc.points || []) {
        const cx = Math.floor((x / 100) * cols), cy = Math.floor((y / 100) * rows);
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            const gx = cx + dx, gy = cy + dy;
            if (gx < 0 || gy < 0 || gx >= cols || gy >= rows) continue;
            grid[gy * cols + gx] += Math.exp(-(dx * dx + dy * dy) / 2.2);
          }
        }
      }
    }
    // SCALED AGAINST A BUSY CELL, NOT THE BUSIEST. one town square with forty spawns used to set
    // the scale for a whole zone and left the open ground barely tinted; the 90th percentile of the
    // cells that carry anything sets it instead, and everything above that is simply full strength
    const carrying = [];
    for (const {grid} of fields.values()) for (const value of grid) if (value > 0.02) carrying.push(value);
    carrying.sort((a, b) => a - b);
    const peak = carrying.length ? carrying[Math.floor(carrying.length * 0.9)] : 1;
    for (const {group, grid} of fields.values()) {
      ctx.fillStyle = group.colour;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const value = Math.min(1, grid[r * cols + c] / (peak || 1));
          if (value < 0.02) continue;
          // the square root lifts the thin ground off the floor: a cell with one spawn in it has to
          // be visible, since "there is something here at all" is what this style answers
          ctx.globalAlpha = mapAlpha.npcs * Math.min(0.95, 0.22 + Math.sqrt(value) * 0.78);
          ctx.fillRect(c * cell, r * cell, cell, cell);
        }
      }
    }
  }
  ctx.globalAlpha = 1;
}

// ---- the three heads -------------------------------------------------------------------------
// TWO COLUMNS AND A VERB. the continent narrows the list; the zone is chosen; `open` commits.
// picking a zone does not open it, because a mis-click on a 60-zone list would otherwise throw
// away the layers and opacities already set up
document.getElementById('map-pick').onclick = async () => {
  let menu = null;
  let continent = null;
  let picked = null;

  const build = async () => {
    const data = await api('/api/maps');
    const all = data.maps || [];
    // AN EMPTY GROUP IS MISSING DATA, not a zone that belongs nowhere. the exporter writes the
    // client's own hierarchy and leaves it blank rather than guessing, so the page says so too
    const groupOf = m => m.group_name || 'no hierarchy yet';
    const continents = [...new Set(all.map(groupOf))].sort();
    if (continent === null && continents.length) continent = continents[0];
    const zones = all.filter(m => groupOf(m) === continent);
    return [
      {kind: 'columns', columns: [
        {label: 'continent', multi: false, empty: 'nothing exported',
          items: continents.map(name => ({id: name, label: name, on: name === continent})),
          onPick: item => { continent = item.id; picked = null; build().then(s => menu.refresh(s)); }},
        {label: 'zone', multi: false, empty: 'none on this continent',
          items: zones.map(m => ({id: m.map_name, label: m.zone_name, on: m.map_name === picked,
                                  state: m.map_name === mapName ? {opened: true} : {}})),
          onPick: item => { picked = item.id; build().then(s => menu.refresh(s)); }},
      ]},
      {kind: 'buttons', buttons: [
        {id: 'open', label: 'open', tone: 'adds', enabled: !!picked, onClick: async m => {
          await mapOpen(picked);
          m.refresh(await build());
        }},
        {id: 'close-set', label: 'close', tone: 'removes', enabled: !!mapName, onClick: async m => {
          mapName = null; mapManifest = null; mapImages.clear(); mapShown.clear();
          document.getElementById('map-pick').innerHTML = '<span>map</span>';
          mapStatus('nothing opened');
          mapDraw();
          m.refresh(await build());
        }},
      ]},
    ];
  };

  menu = await openPicker('map-pick', build, {title: 'map', status: 'map-status'});
};

// WHICH WALKS, NOT WHETHER. the layers menu says whether recordings are drawn at all; this says
// which ones, and without it that layer could only ever be empty. Kept as its own head rather than
// a third column of the layers menu because it is a different question - one is about the map, the
// other about what has been walked on it - and the list grows with every session recorded.
document.getElementById('map-recordings').onclick = async () => {
  let menu = null;
  const chosen = new Set();

  const label = () => {
    const head = document.getElementById('map-recordings');
    head.innerHTML = `<span>recordings${mapWalks.size ? ` \u00b7 ${mapWalks.size}` : ''}</span>`;
  };

  const build = async () => {
    const data = await api('/api/nav-sessions');
    const rows = (data.sessions || []).map(item => ({
      id: item.name,
      label: item.name,
      on: chosen.has(item.name),
      state: mapWalks.has(item.name) ? {opened: true} : {},
    }));
    const open = rows.filter(r => mapWalks.has(r.id));
    const shut = rows.filter(r => !mapWalks.has(r.id));
    return [
      {kind: 'list', multi: true, empty: 'nothing recorded yet',
        onPick: (item, on) => {
          if (on) chosen.add(item.id); else chosen.delete(item.id);
          menu.setButtonEnabled('open', [...chosen].some(n => !mapWalks.has(n)));
          menu.setButtonEnabled('close-set', [...chosen].some(n => mapWalks.has(n)));
        },
        items: [
          ...(open.length ? [{heading: 'open'}, ...open] : []),
          ...(shut.length ? (open.length ? [{heading: 'not open'}, ...shut] : shut) : []),
        ]},
      {kind: 'buttons', buttons: [
        {id: 'open', label: 'open', tone: 'adds',
          enabled: [...chosen].some(n => !mapWalks.has(n)), onClick: async m => {
            for (const name of chosen) {
              if (mapWalks.has(name)) continue;
              const walk = await api('/api/nav?session=' + encodeURIComponent(name));
              if (walk.error) { mapStatus(walk.error); continue; }
              mapWalks.set(name, walk.maps || []);
            }
            chosen.clear();
            // opening a walk with the layer switched off would draw nothing and look broken
            mapShown.add('recordings');
            if (mapAlpha.recordings === undefined) mapAlpha.recordings = 1;
            label();
            mapDraw();
            m.refresh(await build());
          }},
        {id: 'close-set', label: 'close', tone: 'removes',
          enabled: [...chosen].some(n => mapWalks.has(n)), onClick: async m => {
            chosen.forEach(name => mapWalks.delete(name));
            chosen.clear();
            label();
            mapDraw();
            m.refresh(await build());
          }},
      ]},
    ];
  };

  menu = await openPicker('map-recordings', build, {title: 'recordings', status: 'map-status'});
};

// THE LAYER LIST COMES FROM THE MANIFEST, never from a list in here - the exporter grows layers
// (building, slope, water, named areas) and a hardcoded list would quietly omit them
document.getElementById('map-layers').onclick = async () => {
  let menu = null;
  const chosen = new Set();

  const build = async () => {
    const known = Object.keys((mapManifest && mapManifest.layers) || {});
    const rows = [...known, ...MAP_DRAWN].map(layer => ({
      id: layer,
      label: mapLabel(layer),
      on: chosen.has(layer),
      state: mapShown.has(layer) ? {opened: true} : {},
    }));
    const open = rows.filter(r => mapShown.has(r.id));
    const shut = rows.filter(r => !mapShown.has(r.id));
    return [
      {kind: 'list', multi: true, empty: 'open a map first',
        // TICKING A ROW HAS TO REACH THE VERBS. the buttons take their enabled state at build
        // time, so without this the menu opens with open and close greyed - correctly, nothing
        // is ticked yet - and then stays that way however much you tick, showing a not-allowed
        // cursor over the only two things you came here to press
        onPick: (item, on) => {
          if (on) chosen.add(item.id); else chosen.delete(item.id);
          menu.setButtonEnabled('open', [...chosen].some(l => !mapShown.has(l)));
          menu.setButtonEnabled('close-set', [...chosen].some(l => mapShown.has(l)));
        },
        items: [
          ...(open.length ? [{heading: 'shown'}, ...open] : []),
          ...(shut.length ? (open.length ? [{heading: 'not shown'}, ...shut] : shut) : []),
        ]},
      {kind: 'buttons', buttons: [
        {id: 'open', label: 'open', tone: 'adds',
          enabled: [...chosen].some(l => !mapShown.has(l)), onClick: async m => {
          chosen.forEach(layer => mapShown.add(layer));
          chosen.clear();
          if (mapShown.has('fences') || mapShown.has('routes')) await mapLoadMarkers();
          if (mapShown.has('npcs') && !mapNpcs) await mapLoadNpcs();
          mapDraw();
          m.refresh(await build());
        }},
        {id: 'close-set', label: 'close', tone: 'removes',
          enabled: [...chosen].some(l => mapShown.has(l)), onClick: async m => {
          chosen.forEach(layer => mapShown.delete(layer));
          chosen.clear();
          mapDraw();
          m.refresh(await build());
        }},
      ]},
    ];
  };

  menu = await openPicker('map-layers', build, {title: 'layers', status: 'map-status'});
};

// ONE SLIDER PER SHOWN LAYER. a slider for a layer that is not drawn is a control with no effect,
// so the column follows what the layers menu opened rather than everything that exists
// THREE COLUMNS, ONE ROW PER LAYER: how strong, how thick, what colour. Opacity applies to
// everything; thickness and colour only to the layers this page draws itself, because a raster
// carries its own pixels and a control that cannot move them is worse than an absent one.
document.getElementById('map-opacity').onclick = () => {
  const shown = [...mapRasterOrder(mapManifest), ...MAP_DRAWN].filter(l => mapShown.has(l));
  const wrap = document.createElement('div');
  wrap.className = 'menu-section visual-grid';

  if (!shown.length) {
    const none = document.createElement('span');
    none.className = 'none';
    none.textContent = 'no layers shown';
    wrap.appendChild(none);
  } else {
    ['opacity', 'thickness', 'colour'].forEach(text => {
      const head = document.createElement('div');
      head.className = 'field-label';
      head.textContent = text;
      wrap.appendChild(head);
    });
  }

  shown.forEach(layer => {
    // THE NPC LAYER TAKES ONLY OPACITY HERE. its marks and its colours come from the grouping and
    // the style in its own dropdown, so a line width and a tint would be two controls fighting one
    const ownControl = layer === 'npcs';
    const drawn = MAP_DRAWN.includes(layer) && !ownControl;
    const tintable = mapCanTint(layer) && !ownControl;

    const opacity = document.createElement('div');
    wrap.appendChild(opacity);
    makeSlider(opacity, {
      id: `opacity-${layer}`, label: mapLabel(layer),
      min: 0, max: 1, step: 0.05, value: mapAlpha[layer] ?? 1,
      format: v => `${Math.round(v * 100)}%`,
      onChange: v => { mapAlpha[layer] = v; mapDraw(); },
    });

    const thickness = document.createElement('div');
    wrap.appendChild(thickness);
    if (drawn) {
      makeSlider(thickness, {
        id: `width-${layer}`, label: 'line',
        min: 0.5, max: 6, step: 0.5, value: mapWidth[layer] ?? 2,
        format: v => `${v} px`,
        onChange: v => { mapWidth[layer] = v; mapDraw(); },
      });
    } else {
      thickness.className = 'visual-na';
      thickness.textContent = ownControl ? 'npc visualisation' : 'image';
    }

    const colour = document.createElement('div');
    wrap.appendChild(colour);
    if (!tintable) {
      colour.className = 'visual-na';
      // a scalar, an index or the client art: the pixels ARE the data and a tint would erase it.
      // the npc layer is neither - its colours ARE the grouping, so it says where they come from
      colour.textContent = ownControl ? 'grouped' : 'has colour';
      return;
    }
    // the OS colour wheel, reached through a styled button - an unstyled <input type=color> is a
    // grey slab that matches nothing else here, so it is hidden and the swatch triggers it
    const picker = document.createElement('input');
    picker.type = 'color';
    picker.className = 'visual-picker';
    picker.value = mapColour[layer] || MAP_MASK_TINT[layer] || '#c7ed5f';
    const button = document.createElement('button');
    button.className = 'toggle visual-swatch';
    button.style.background = picker.value;
    button.title = 'pick a colour';
    button.onclick = () => picker.click();
    picker.oninput = () => {
      mapColour[layer] = picker.value;
      button.style.background = picker.value;
      mapDraw();
    };
    colour.appendChild(button);
    colour.appendChild(picker);
  });

  // never refreshed: a rebuild would detach every slider makeSlider just returned
  new Menu({title: 'layer visualisation', persistent: true, sections: [{kind: 'node', node: wrap}]})
    .openAt(document.getElementById('map-opacity'));
};

// TWO COLUMNS, ONE PICK IN EACH. the grouping decides every colour, the style decides the marks,
// and neither is useful without the other - which is why they are one menu and not two
document.getElementById('map-npc-vis').onclick = () => {
  const wrap = document.createElement('div');
  wrap.className = 'menu-section npc-cols';

  const column = (title, key, options) => {
    const box = document.createElement('div');
    const head = document.createElement('div');
    head.className = 'field-label';
    head.textContent = title;
    box.appendChild(head);
    options.forEach(([id, label]) => {
      const row = document.createElement('label');
      row.className = 'npc-opt';
      const input = document.createElement('input');
      input.type = 'radio';
      input.name = `npc-${key}`;
      input.checked = npcVis[key] === id;
      input.onchange = () => { npcVis[key] = id; mapDraw(); };
      row.appendChild(input);
      row.appendChild(document.createTextNode(label));
      box.appendChild(row);
    });
    wrap.appendChild(box);
  };

  column('group by', 'group', [['level', 'level range'], ['reaction', 'reaction'], ['name', 'name']]);
  column('draw as', 'style', [['dot', 'dot'], ['circle', 'spawn circle'], ['density', 'density map']]);

  new Menu({title: 'npc visualisation', persistent: true, sections: [{kind: 'node', node: wrap}]})
    .openAt(document.getElementById('map-npc-vis'));
};

// WHAT IS ON THE MAP AT ALL, unlike the grouping, which only colours what is already there
document.getElementById('map-npc-filter').onclick = () => {
  const wrap = document.createElement('div');
  wrap.className = 'menu-section npc-cols';
  const left = document.createElement('div');
  const right = document.createElement('div');
  wrap.appendChild(left);
  wrap.appendChild(right);

  const heading = (box, text) => {
    const head = document.createElement('div');
    head.className = 'field-label';
    head.textContent = text;
    box.appendChild(head);
  };

  heading(left, 'level range');
  const range = document.createElement('div');
  range.className = 'npc-range';
  [['lo', 'from'], ['hi', 'to']].forEach(([key, label]) => {
    const input = document.createElement('input');
    input.type = 'number';
    input.min = 1;
    input.max = 70;
    input.value = npcFilter[key];
    input.title = label;
    input.oninput = () => { npcFilter[key] = Number(input.value); mapDraw(); };
    range.appendChild(input);
  });
  left.appendChild(range);

  heading(left, 'reaction');
  ['hostile', 'neutral', 'friendly'].forEach(reaction => {
    const row = document.createElement('label');
    row.className = 'npc-opt';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = npcFilter.reactions.has(reaction);
    input.onchange = () => {
      if (input.checked) npcFilter.reactions.add(reaction); else npcFilter.reactions.delete(reaction);
      mapDraw();
    };
    row.appendChild(input);
    row.appendChild(document.createTextNode(reaction));
    left.appendChild(row);
  });

  heading(right, `name${mapNpcs && mapNpcs.npcs ? ` (${mapNpcs.npcs.length})` : ''}`);
  const search = document.createElement('input');
  search.type = 'search';
  search.placeholder = 'search';
  right.appendChild(search);
  const list = document.createElement('div');
  list.className = 'npc-list';
  right.appendChild(list);

  const drawList = () => {
    const query = search.value.trim().toLowerCase();
    list.innerHTML = '';
    const rows = (mapNpcs && mapNpcs.npcs ? mapNpcs.npcs : []).filter(n => !query || n.name.toLowerCase().includes(query));
    if (!rows.length) {
      const none = document.createElement('span');
      none.className = 'none';
      none.textContent = mapNpcs ? 'no npc matches' : 'open the npc layer first';
      list.appendChild(none);
      return;
    }
    rows.slice(0, 400).forEach(npc => {
      const row = document.createElement('label');
      row.className = 'npc-opt';
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = npcNamed(npc);
      input.onchange = () => {
        // the set starts as null meaning "all", so the first tick has to write out today's answer
        if (!npcFilter.names) npcFilter.names = new Set(mapNpcs.npcs.map(n => n.npc_id));
        if (input.checked) npcFilter.names.add(npc.npc_id); else npcFilter.names.delete(npc.npc_id);
        mapDraw();
      };
      const level = document.createElement('span');
      level.className = 'npc-level';
      level.textContent = npc.min_level === npc.max_level ? `${npc.min_level}` : `${npc.min_level}-${npc.max_level}`;
      row.appendChild(input);
      row.appendChild(document.createTextNode(npc.name));
      row.appendChild(level);
      list.appendChild(row);
    });
  };
  search.oninput = drawList;
  drawList();

  const buttons = document.createElement('div');
  buttons.className = 'npc-buttons';
  [['all', () => { npcFilter.names = null; }], ['none', () => { npcFilter.names = new Set(); }],
   ['reset', () => {
     npcFilter.lo = 1;
     npcFilter.hi = 70;
     npcFilter.names = null;
     npcFilter.reactions = new Set(['hostile', 'neutral', 'friendly']);
   }]].forEach(([label, apply]) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'toggle';
    button.textContent = label;
    button.onclick = () => {
      apply();
      wrap.querySelectorAll('input[type="checkbox"]').forEach(box => { box.checked = true; });
      if (npcFilter.names && !npcFilter.names.size) {
        list.querySelectorAll('input[type="checkbox"]').forEach(box => { box.checked = false; });
      }
      range.children[0].value = npcFilter.lo;
      range.children[1].value = npcFilter.hi;
      drawList();
      mapDraw();
    };
    buttons.appendChild(button);
  });
  right.appendChild(buttons);

  new Menu({title: 'npc filter', persistent: true, sections: [{kind: 'node', node: wrap}]})
    .openAt(document.getElementById('map-npc-filter'));
};

function mapEnter() {
  if (!mapManifest) {
    api('/api/maps').then(data => {
      const first = (data.maps || [])[0];
      if (first) mapOpen(first.map_name).catch(err => mapStatus(String(err)));
      else mapStatus('nothing exported - run wt-export-map');
    }).catch(err => mapStatus(String(err)));
    return;
  }
  mapDraw();
}


// ---- settings: the ONE place a path is chosen. find derives its own from the bound dataset
// and these bases, so there is no second picker to disagree with them ----


// A PATH IS PICKED, NOT TYPED. Four of the five bases are directories under the base path, and a
// typed path is one that can be typed wrong - silently, since a base that does not exist simply
// offers nothing rather than failing. Only the root is a text field now; the rest browse from it.
const BASE_TYPES = ['sessions', 'labels', 'templates', 'pool'];
let baseChosen = {};

// WHICH BASES CAN BE POINTED SOMEWHERE ELSE. The rest of the dialog shows the layout's defaults and
// changes nothing - Balthazar Fitzpatrick: "those paths should be the defaults under settings, but
// one can also chose their own ones, but that can be deferred".
//
// THE TILE POOL IS THE EXCEPTION, because it is the one with a use today: `wt-synth-frames` writes
// its generated tiles to training/tiles-synth, and pointing the pool at them is how they are
// looked at - without copying anything and without disturbing the real pool they stand in for.
// With every row locked there was no way to reach them at all.
const EDITABLE_BASES = new Set(['pool']);

// WHICH BASES HOLD SEVERAL PATHS. Only the pool, and it is the only one where several directories
// mean something plain: the grid lists every tile across all of them, so the generated set is
// reviewed beside the real one without switching and without copying. The server agrees - see
// paths.MULTI_BASES, which says why sessions and templates are not on this list.
const MULTI_BASES = new Set(['pool']);

// every base is held as a LIST internally, however many it has - one shape to paint and to send
const asList = value => (Array.isArray(value) ? value : value ? [value] : []);

function paintBases() {
  const root = document.getElementById('base-root').value;
  // SHOWN RELATIVE TO THE ROOT, because the root is on screen directly above and repeating it on
  // every row buries the part that differs
  const short = full =>
    full.startsWith(root) ? (full.slice(root.length).replace(/^\//, '') || '.') : full;
  BASE_TYPES.forEach(type => {
    const head = document.getElementById(`base-${type}`);
    const paths = asList(baseChosen[type]);
    head.innerHTML = paths.length
      ? paths.map(full => `<span class="base-path">${short(full)}</span>`).join('')
      : '<span>-</span>';
    // the full paths in the tooltip, since each row shows only the part below the root
    head.title = paths.join('\n');
  });
}

// ONE LEVEL PER REQUEST, walked in place. sessions/ alone holds 21 recordings and the vision
// directory holds a pool of 10,000 tiles, so a whole tree would be enormous and mostly noise.
//
// THE PICKER SHOWS WHAT IS ALREADY CHOSEN, with one add row above it - the same shape the interface
// tab's name pickers use. Balthazar Fitzpatrick: "Click dropdown, see a list of things already
// there, but on top have an empty field with an add next to it. In this case the empty field on
// click just turns into a path explorer, dirs to drill, .. on top to go up one."
//
// WHAT IT REPLACED, and why it had to go: three buttons reading "use this folder", "add this one
// too" and "use only the first", above a tree and nothing else. The list they acted on was not on
// screen, so "add this one TOO" had no visible "one" to be too - "add this one too is a misleading
// button". Removing one specific folder was impossible; the only retreat was back to the first.
// And with the panel unbounded the buttons fell off the bottom of the screen entirely.
// KEEP A REFRESHED PANEL ON SCREEN. Menu.refresh holds the panel where it is on purpose, so it
// does not jump while you are reading it - but the folder browser replaces a two-row listing with a
// forty-row one, and a panel opened two thirds down the page then hangs off the bottom edge with
// its add row below the screen. ui_base caps the panel at 70vh and clamps it at OPEN time; this is
// the same clamp applied again after a rebuild. It only ever pulls the panel UP, and only when it
// would otherwise overflow, so a panel that already fits does not move.
function keepOnScreen(menu) {
  const el = menu && menu.el;
  if (!el) return;
  const top = parseFloat(el.style.top) || 0;
  const highest = window.innerHeight - el.getBoundingClientRect().height - 6;
  if (top > highest) el.style.top = `${Math.max(6, highest)}px`;
}

async function pickDirectory(type) {
  // null while the list of chosen folders is showing; a path once the field has been clicked into
  let browsing = null;
  let menu = null;

  const root = () => document.getElementById('base-root').value;
  const short = full =>
    full.startsWith(root()) ? (full.slice(root().length).replace(/^\//, '') || '.') : full;
  const chosen = () => asList(baseChosen[type]);

  const refresh = () =>
    build().then(sections => { menu.refresh(sections); wireField(); keepOnScreen(menu); });

  // a base that takes one path REPLACES on add; the pool appends. Either way the browser closes,
  // because the answer to "which folder" has just been given
  const take = path => {
    const keep = MULTI_BASES.has(type) ? chosen() : [];
    if (!keep.includes(path)) baseChosen[type] = [...keep, path];
    browsing = null;
    paintBases();
    refresh();
  };

  const drop = path => {
    baseChosen[type] = chosen().filter(p => p !== path);
    paintBases();
    refresh();
  };

  const build = async () => {
    const add = {
      kind: 'add',
      placeholder: 'click here to browse for a folder',
      button: 'add',
      onAdd: value => take(value),
    };
    if (browsing === null) {
      return [add, {
        kind: 'list',
        empty: 'no folder chosen yet - browse for one above',
        items: chosen().map(full => ({
          id: full,
          label: short(full),
          title: full,
          // the per-row close, which is what makes removing ONE folder possible at all
          action: {label: 'x', onPick: item => drop(item.id)},
        })),
      }];
    }
    const data = await api('/api/dir-tree?under=' + encodeURIComponent(browsing));
    browsing = data.here;
    const items = data.parent ? [{id: data.parent, label: '..'}] : [];
    data.entries.forEach(e => items.push({
      id: e.path,
      label: e.name + (e.has_children ? '/' : ''),
      on: chosen().includes(e.path),
    }));
    return [add, {
      kind: 'list',
      empty: 'nothing below this folder',
      // DRILLING IS NOT CHOOSING. a row opens the folder; `add` takes the one you are standing in,
      // which is the only way a folder that has children can be picked at all
      onPick: item => { browsing = item.id; refresh(); },
      items,
    }];
  };

  // THE FIELD IS THE BROWSER'S DOOR AND ITS ADDRESS BAR. Empty, clicking it opens the explorer at
  // the first chosen folder; open, it carries the path being walked, so `add` and a typed path are
  // the same button.
  const wireField = () => {
    const input = menu && menu.el.querySelector('.menu-add input');
    if (!input) return;
    input.value = browsing === null ? '' : browsing;
    input.onfocus = () => {
      if (browsing !== null) return;
      browsing = chosen()[0] || root();
      refresh();
    };
  };

  menu = await openPicker(`base-${type}`, build, {title: `${type} folder`, status: null});
  if (menu) wireField();
}

document.getElementById('open-settings').onclick = async () => {
  const data = await api('/api/dir-list');
  document.getElementById('base-root').value = data.bases.root || '';
  baseChosen = {};
  BASE_TYPES.forEach(type => { baseChosen[type] = asList(data.bases[type]); });
  paintBases();
  document.getElementById('settings-popup').classList.remove('hidden');
};
BASE_TYPES.filter(type => EDITABLE_BASES.has(type)).forEach(type => {
  const head = document.getElementById(`base-${type}`);
  head.classList.add('toggle', 'dropdown-head');
  head.onclick = () => pickDirectory(type);
});
if (EDITABLE_BASES.size) document.getElementById('settings-save').classList.remove('hidden');
document.getElementById('settings-close').onclick = () =>
  document.getElementById('settings-popup').classList.add('hidden');
document.getElementById('settings-save').onclick = async () => {
  // ONLY THE EDITABLE ONES ARE SENT. set_bases ignores what it does not know, but it has no way to
  // tell a locked row from an unlocked one - so the page does not offer it a value it must not take
  const body = {};
  BASE_TYPES.filter(type => EDITABLE_BASES.has(type)).forEach(type => {
    // a list where the server takes several, the single value where it does not
    const paths = asList(baseChosen[type]);
    body[type] = MULTI_BASES.has(type) ? paths : (paths[0] || '');
  });
  await api('/api/set-bases', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  document.getElementById('settings-popup').classList.add('hidden');
  // the pool may have moved, and the grid is reading the old one until it is told
  await loadClusters();
};

// k DEFAULTS to 24 - the identity count Balthazar Fitzpatrick actually sorts into - but is editable, because the
// the library's own count, as a bare hint. It used to double as the suggested cluster count, which
// is why it was called "inferred k" - the grid sorts by class now and chooses nothing.
async function refreshLibraryCount() {
  const data = await api('/api/dir-list');
  const note = document.getElementById('cluster-k-note');
  if (note) note.textContent = `library: ${data.inferred_k}`;
}

// ---- find tab: draw boxes on a recording's frames and save them as a set ----
const findMode = 'drawn';   // the only mode; kept as the name the save endpoint expects
// find IS drawing now. the mining modes needed a template library to match against, which is
// exactly what a cold start does not have - and once the cnn is the detector they are dead weight.
// Balthazar Fitzpatrick, 2026-09-01: "Find should also then just only have the draw boxes mode, which does not
// need to be explicitly selected by its own button."


// ---- open a dataset from find or cluster, without going back to the load tab ----
// binding is one fact, so both buttons drive the same popup and the same endpoint the load tab
// uses. datasets (already drawn or mined) come first: those are the ones with anything in them
// f d c n - frames, drawn, classes, negatives. one line per row, and it spans BOTH stages on
// purpose: d is what find produced, c and n are what discard/promote decided about it, so the row
// answers "how far along is this recording" rather than only "does it exist". a count that is zero
// is dropped rather than printed - "0n" is noise, and a set with nothing rejected should read as
// nothing rejected. the legend lives beside the menu title, so the letters are never unexplained
function statsFor(row, unit = 'f') {
  const parts = [`${row.frame_count}${unit === 'f' ? 'f' : ' ' + unit}`];
  if (row.boxes) parts.push(`${row.boxes}d`);
  if (row.classed) parts.push(`${row.classed}c`);
  if (row.negative) parts.push(`${row.negative}n`);
  // and nothing is appended when only the frame count survives: the ABSENCE of a d is already
  // "nothing drawn", by the same rule that drops a zero two lines above. spelled out it took 100px
  // of every row in the picker to say what the empty space said for free, and the name paid for it
  return parts.join(' · ');
}

// ONE DROPDOWN, TWO JOBS: what you may open on the left, what IS open on the right. Before this,
// opening and closing lived on separate heads and nothing on the page said which recording was
// bound - you read it off the head's own label, or you did not.
async function unbindRecording() {
  await api('/api/unbind-recording', {method: 'POST',
    headers: {'Content-Type': 'application/json'}, body: '{}'});
  await drawOpen();
  // closing touches nothing on disk - the boxes are saved with the save button and stay where
  // they are, so this only says what happened rather than warning about a move
  document.getElementById('find-status').textContent = 'closed - drawn boxes left on disk';
}

async function openDatasetPicker(anchorId) {
  let menu = null;

  // EACH COLUMN OPENS WITH WHAT IS OPEN. sorting the bound one to the top said where it was but
  // not what it was; a section says both, and it costs one heading row.
  const build = async () => {
    const data = await api('/api/bindable-recordings');
    const of = kind => data.rows.filter(r => r.kind === kind)
      .sort((a, b) => a.name.localeCompare(b.name));
    const asItem = (row, unit) => ({
      id: row.name, label: row.name, kind: row.kind, stats: statsFor(row, unit),
      opened: row.name === data.current,
    });

    const column = (label, rows, unit) => {
      const items = rows.map(r => asItem(r, unit));
      const open = items.filter(i => i.opened).map(i => ({...i, state: {opened: true}}));
      const shut = items.filter(i => !i.opened);
      return {
        label,
        empty: 'none',
        // TICK, THEN PRESS OPEN - the same two steps select takes, so the two menus are one habit
        onPick: (item, on) => {
          if (item.opened) return;                   // an open one is let go with the close button
          if (on) chosen.clear();                    // only one recording can be bound at a time
          if (on) chosen.set(item.id, item); else chosen.delete(item.id);
          menu.setButtonEnabled('open', chosen.size > 0);
        },
        // THE COLUMN ALREADY SAYS WHAT THESE ARE. repeating its name as a section heading gave
        // find a menu that said "recordings" three times - in the title, the column and the
        // section - and none of the three told you anything the others did not
        items: [
          ...(open.length ? [{heading: 'open'}, ...open] : []),
          ...(shut.length ? (open.length ? [{heading: 'not open'}, ...shut] : shut) : []),
        ],
      };
    };

    const columns = [column('recordings', of('session'), 'f')];
    if (of('dataset').length) columns.push(column('found by the cnn', of('dataset'), 'proposals'));

    return [
      {kind: 'columns', columns},
      {kind: 'buttons', buttons: [
        {id: 'open', label: 'open', tone: 'adds', enabled: false, onClick: async () => {
          for (const item of chosen.values()) await bind(item);
          chosen.clear();
          menu.refresh(await build());
        }},
        {id: 'close-set', label: 'close', tone: 'removes', enabled: !!data.current,
          onClick: async () => { await unbindRecording(); menu.refresh(await build()); }},
      ]},
    ];
  };

  const chosen = new Map();
  const bind = async item => {
    const res = await api('/api/bind-recording', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({kind: item.kind, name: item.id}),
    });
    if (res.error) { alert(res.error); return; }
    if (!document.querySelector('.tab-panel[data-panel="find"]').classList.contains('hidden')) {
      await drawOpen();
    }
    if (!document.querySelector('.tab-panel[data-panel="cluster"]').classList.contains('hidden')) {
      await loadClusters();
    }
  };

  menu = await openPicker(anchorId, build, {
    title: 'f frames \u00b7 d drawn \u00b7 c classes \u00b7 n negatives',
    status: 'find-status',
  });
}

document.getElementById('draw-open').onclick = () => openDatasetPicker('draw-open');

// THE CNN PICKS A TRAINING SET, NOT A RECORDING. Balthazar Fitzpatrick: "cnn should only be able to see datasets
// for candidates... that come out of the discard or promote". a candidates queue is unreviewed by
// definition, so offering one here would invite training on boxes nobody judged
document.getElementById('train-open').onclick = async () => {
  const data = await api('/api/training-sets');
  const items = data.sets.map(set => {
    const classes = Object.entries(set.classes).map(([k, v]) => `${k} ${v}`).join(' \u00b7 ');
    return {
      id: set.name,
      label: set.name,
      on: set.name === trainingSet,
      stats: `${set.rows} rows${classes ? ' - ' + classes : ''}`,
      rows: set.rows,
    };
  });
  listMenu('training sets', items, async item => {
    // THE SERVER HAS TO HEAR THIS. the name used to live only in the page, so the trainer went
    // on building examples from whatever candidates queue was bound - unreviewed boxes
    await api('/api/train-bind', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: item.id}),
    });
    // BOTH, and in this order. loadTrain repaints the summary block - without it the four facts
    // above still described the world at page load, so a bound set showed "nothing yet" while its
    // own name sat in the head right above
    await loadTrain();
    await loadTrainClasses();
    // THE WINDOW FLOOR COMES FROM THE SET'S BOXES, so it is stale until asked again
    await loadCropFloor();
  }, {empty: 'none promoted'})
    .openAt(document.getElementById('train-open'));
};
let trainingSet = null;

// PROMOTE: every kept, labelled tile into one durable training set. a candidates file is a
// review queue per recording; this is the accumulating answer the cnn actually reads
// ONE DROPDOWN FOR WHAT THIS TAB SHOWS. there used to be two - "all tiles in the pool", which
// bound a candidates file the grid does not read, and "sources". Balthazar Fitzpatrick: "there's two buttons with
// overlap... it should have neither name, it should be 'open dataset' and should list all the
// crops, be it from find, or be it from sweep."
// TWO DROPDOWNS, ONE JOB EACH. Balthazar Fitzpatrick: "Open dataset. When opened a dataset, it shows in the list,
// and I can toggle it on or off. I can also close a dataset, which is a dropdown that is populated
// with all open datasets". So the POOL is the open set: opening cuts a candidates file's tiles
// in, closing archives them back out, and the dataset filter row above the grid is the on/off.
// Both are split by origin, because a swept dataset is the cnn's guesses and a drawn one is his
// own hand - judging them is not the same act.
function bySource(rows, pick, extra = {}) {
  const sections = [];
  const find = rows.filter(r => r.source !== 'sweep');
  const sweep = rows.filter(r => r.source === 'sweep');
  sections.push({kind: 'list', label: 'drawn in find', items: find.map(pick), ...extra});
  if (sweep.length) {
    sections.push({kind: 'list', label: 'found by the cnn', items: sweep.map(pick), ...extra});
  }
  return sections;
}

// BOTH MENUS ARE MULTI-SELECT WITH AN APPLY, and the reason is loadClusters: it re-reads and
// re-renders every tile in the pool, four thousand of them, so a menu that fired it per row
// picked made opening three datasets three full repaints and three closes of the panel. Tick what
// you want, press once. Balthazar Fitzpatrick: "multi-select with an apply button at the bottom".
function pickAndApply({anchor, title, sections, button, tone, apply}) {
  const chosen = new Map();
  let menu = null;
  sections.forEach(section => {
    section.multi = true;
    section.onPick = (item, on) => {
      if (on) chosen.set(item.id, item.label); else chosen.delete(item.id);
      menu.setButtonEnabled(button, chosen.size > 0);
    };
  });
  sections.push({kind: 'buttons', buttons: [{
    // nothing ticked is nothing to apply, and a live button that does nothing is worse than a
    // grey one that says so
    id: button, label: button, tone, enabled: false,
    onClick: m => { m.close(); apply([...chosen.entries()]); },
  }]});
  menu = new Menu({title, sections});
  menu.openAt(document.getElementById(anchor));
}

// ONE HEAD FOR BOTH DIRECTIONS. open and close were two dropdowns that never showed each other's
// state, so "what is in the pool right now" was a question the ui could not answer - you opened
// the close menu to find out. Now the open column IS the answer, and clicking one closes it.
// PURE VIEW STATE - the npz holds the real pixels either way, so switching costs nothing and
// never needs a re-cut. Deleted by accident when the close-dataset menu it lived in was replaced,
// which left applyTileMode() called and undefined: a page error stops script evaluation dead, so
// every handler bound after it - crop size among them - silently never attached.
let uniformTiles = false;
try { uniformTiles = localStorage.getItem('wt-uniform-tiles') === '1'; } catch (err) { /* private */ }

function applyTileMode() {
  document.getElementById('cluster-groups')?.classList.toggle('uniform-tiles', uniformTiles);
}

async function recutPool(status) {
  status.textContent = 're-cutting...';
  const res = await api('/api/recut-pool', {method: 'POST',
    headers: {'Content-Type': 'application/json'}, body: '{}'});
  const total = (res.recut || []).reduce((n, r) => n + r.tiles, 0);
  await loadClusters();
  status.textContent = `re-cut ${total} tiles from ${(res.recut || []).length} sets`
    + ((res.failed || []).length ? `, ${res.failed.length} skipped` : '');
}

// ONE HEAD FOR BOTH DIRECTIONS, and it stays open while you work. Open and close were two
// dropdowns that never showed each other's state, so "what is in the pool right now" was a
// question the ui could not answer - you opened the close menu to find out.
async function openClusterSets() {
  const status = document.getElementById('clusters-status');
  const chosen = new Map();
  let menu = null;

  // CLOSING IS NOT ARCHIVING, and it used to be: every tile was moved out to the archive and
  // its label record dropped, so "close" silently cost the judging. Closing hides a source now -
  // the tiles stay, the judgements stay, and opening it again is instant.
  //
  // WHERE THERE ARE CHANGES IT ASKS FIRST. A tile carrying a class or marked "not a class" is work
  // someone did, so discarding it is offered as a deliberate second choice rather than being what
  // close happens to do.
  const closeOne = async tag => {
    const state = await api('/api/source-state?tag=' + encodeURIComponent(tag));
    const finish = async discard => {
      status.textContent = `closing ${tag}...`;
      const res = await api('/api/close-source', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({tag, discard}),
      });
      await loadClusters();
      status.textContent = `closed ${tag} - ${res.closed} tiles kept on disk`
        + (res.discarded ? `, ${res.discarded} judgements discarded` : '');
    };
    if (!state.judged) { await finish(false); return; }

    return new Promise(resolve => {
      let answered = false;
      const done = async (menu, discard) => {
        answered = true;
        menu.close();
        await finish(discard);
        resolve();
      };
      new Menu({
        title: `close ${tag}`,
        onDismiss: () => { if (!answered) resolve(); },
        sections: [
          {kind: 'node', node: (() => {
            const box = document.createElement('div');
            box.className = 'menu-section';
            box.innerHTML =
              `<div class="stat">${state.judged} of ${state.tiles} tiles have been judged</div>`
              + '<div class="stat">keeping them costs nothing - the tiles stay where they are</div>';
            return box;
          })()},
          {kind: 'buttons', buttons: [
            {id: 'keep', label: 'keep judgements', tone: 'adds', onClick: m => done(m, false)},
            {id: 'discard', label: 'discard them', tone: 'removes', onClick: m => done(m, true)},
          ]},
        ],
      }).openAt(headPoint('cluster-open-set'));
    });
  };

  // a source that was closed is still cut, so this is instant - it never goes near the cutter
  const reopenOne = async tag => {
    status.textContent = `opening ${tag}...`;
    const res = await api('/api/open-source', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({tag}),
    });
    await loadClusters();
    status.textContent = `opened ${tag} - ${res.tiles} tiles`;
  };

  const build = async () => {
    const [openable, sources] = await Promise.all([
      api('/api/openable-datasets'), api('/api/pool-sources'),
    ]);
    const openTags = new Set(sources.sources.map(src => src.tag));
    const isSweep = tag => tag.includes('_cnn-');
    // OPENED IS NOT TICKED. `on` used to mean both - "this is in the pool" and "you have selected
    // this to act on" - so after opening one the tick stayed lit and read as a live selection
    // a CLOSED source is listed with its tiles still counted, because they are still there - it
    // offers itself back rather than looking like a dataset that has never been opened
    const open = sources.sources.map(src => ({
      id: src.tag, label: src.tag,
      stats: `${src.tiles} tiles`,
      close: !src.closed,
      reopen: src.closed,
      state: src.closed ? {} : {opened: true},
      source: isSweep(src.tag) ? 'sweep' : 'find',
    }));
    const shut = openable.datasets
      .filter(d => !openTags.has(d.tag))
      .map(d => ({id: d.file, label: d.tag, stats: `${d.boxes} boxes`, source: d.source}));

    const column = (label, source) => {
      // TWO GROUPS, NOT THREE. A dataset is open or it is not, and "closed" was a third state a
      // reader had to place before finding out it meant "not open, but quick to reopen" -
      // Balthazar Fitzpatrick: "closed datasets go into the not open row in each column". Closing
      // earned its own heading because a just-closed dataset otherwise went on being listed under
      // "open"; the fix for that was never a heading of its own, it was not being under "open".
      //
      // WHAT THE HEADING CARRIED IS STILL CARRIED, by the row: a closed source keeps its cut tiles
      // and reads "N tiles", where one never opened reads "N boxes" and has to be cut. `reopen` on
      // the item is what the open button reads, so the distinction that matters is on the item.
      const live = open.filter(i => i.source === source && !i.reopen);
      const closed = open.filter(i => i.source === source && i.reopen);
      const rest = shut.filter(i => (i.source === 'sweep') === (source === 'sweep'));
      return {
        label,
        empty: 'none',
        onPick: (item, on) => {
          // an open one is let go by ticking it and pressing close, the same two steps as opening
          if (item.close) { if (on) chosen.set(item.id, item); else chosen.delete(item.id); }
          else if (on) chosen.set(item.id, item); else chosen.delete(item.id);
          menu.setButtonEnabled('open', [...chosen.values()].some(i => !i.close));
          menu.setButtonEnabled('close-set', [...chosen.values()].some(i => i.close));
        },
        // one group needs no heading - the column label already says what the list is, and an
        // empty group is dropped, so with nothing open there is no "open" heading over a blank
        items: (() => {
          // the closed ones lead the not-open group: they were in use a moment ago and they come
          // back instantly, so they are the likelier pick of the two kinds
          const groups = [['open', live], ['not open', [...closed, ...rest]]]
            .filter(([, items]) => items.length);
          if (groups.length === 1) return groups[0][1];
          return groups.flatMap(([heading, items]) => [{heading}, ...items]);
        })(),
      };
    };

    const columns = [column('drawn in find', 'find')];
    if (open.some(i => i.source === 'sweep') || shut.some(i => i.source === 'sweep')) {
      columns.push(column('found by the cnn', 'sweep'));
    }

    return [
      {kind: 'columns', columns},
      {kind: 'list', label: 'view', multi: true,
        items: [{id: 'uniform', label: 'uniform tile size', on: uniformTiles}],
        onPick: (item, on) => {
          uniformTiles = on;
          try { localStorage.setItem('wt-uniform-tiles', on ? '1' : '0'); } catch (err) { /* private */ }
          applyTileMode();
        }},
      {kind: 'buttons', buttons: [
        {id: 'recut', label: 're-cut every drawn set', onClick: () => recutPool(status)},
        {id: 'open', label: 'open', tone: 'adds', enabled: false, onClick: async () => {
          const picked = [...chosen.values()].filter(i => !i.close);
          status.textContent = `opening ${picked.length} dataset(s)...`;
          const done = [];
          for (const item of picked) {
            // A CLOSED SOURCE IS ALREADY CUT. sending it through open-dataset would cut every tile
            // again from the candidates and mint a fresh set of names, which is what closing used
            // to force because it archived the originals
            if (item.reopen) {
              await reopenOne(item.id);
              done.push(`${item.label} reopened`);
              continue;
            }
            const res = await api('/api/open-dataset', {
              method: 'POST', headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({name: item.id}),
            });
            done.push(res.error ? `${item.label}: ${res.error}` : `${res.tag} - ${res.tiles} tiles`);
          }
          chosen.clear();
          await loadClusters();
          status.textContent = `opened ${done.join(' \u00b7 ')}`;
          menu.refresh(await build());
        }},
        {id: 'close-set', label: 'close', tone: 'removes', enabled: false, onClick: async m => {
          // TAKE THE PICKS BEFORE CLOSING, and close before confirming. Only one menu is ever open,
          // so the confirmation closes this one on its way up - refreshing it afterwards was
          // refreshing a panel that no longer exists. Reopening shows the new state either way,
          // including when nothing was judged and no confirmation appeared at all.
          const picked = [...chosen.values()].filter(i => i.close);
          chosen.clear();
          m.close();
          for (const item of picked) await closeOne(item.id);
          await openClusterSets();
        }},
      ]},
    ];
  };

  menu = await openPicker('cluster-open-set', build, {
    title: 'boxes  \u00b7  tick to open, tick an open one to close',
    status: 'clusters-status',
  });
}

document.getElementById('cluster-open-set').onclick = openClusterSets;

// THE DEFINITION EDITOR. built on the interface tab's own tree rather than a new idiom: one
// heading per dimension, its members beneath, and an inline add row - which is exactly the shape
// `+ add spec` already uses there.
function classDefEditor() {
  const draft = classDef
    ? {name: classDef.name, dimensions: classDef.dimensions.map(d => ({...d, members: [...d.members]}))}
    : {name: '', dimensions: []};
  let menu = null;

  const save = async () => {
    const res = await api('/api/classdef-save', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(draft),
    });
    if (res.error) { document.getElementById('clusters-status').textContent = res.error; return; }
    menu.close();
    document.getElementById('filter-dims').innerHTML = '';   // rebuilt from the new definition
    await loadClassDef(res.name);
    await loadClusters();
  };

  const body = () => {
    const node = document.createElement('div');
    const rows = [];
    draft.dimensions.forEach((dimension, index) => {
      // A DIMENSION IS A ROW, NOT A HEADING. it was a heading, and a heading has no click handler -
      // so there was no way to remove a dimension at all once added, only to add up to three and
      // live with them. The badge says what the press does, because a row that deletes on click
      // without saying so is the same trap the members below had.
      rows.push({
        id: `dim:${index}`, label: dimension.name, dot: false,
        state: {removes: true}, badges: ['remove'],
        title: `remove the "${dimension.name}" dimension and its ${dimension.members.length} member(s)`,
        onPick: () => {
          draft.dimensions.splice(index, 1);
          menu.refresh(sections());
        },
      });
      dimension.members.forEach(member => rows.push({
        id: `${index}:${member}`, label: member, dot: false,
        state: {member: true},
        // clicking a member removes it - the same declarative press the rest of the tool uses.
        // SAID OUT LOUD now: it always did this, and nothing on screen mentioned it
        badges: ['remove'],
        title: `remove "${member}" from ${dimension.name}`,
        onPick: () => {
          dimension.members = dimension.members.filter(m => m !== member);
          menu.refresh(sections());
        },
      }));
      rows.push({
        id: `${index}:add`, label: '+ add member', dot: false,
        state: {adds: true},
        onPick: () => {
          const value = (prompt(`new member of ${dimension.name}`) || '').trim();
          if (value && !dimension.members.includes(value)) dimension.members.push(value);
          menu.refresh(sections());
        },
      });
    });
    if (draft.dimensions.length < 3) {
      rows.push({
        id: 'add-dim', label: '+ add dimension', dot: false, state: {adds: true},
        onPick: () => {
          const value = (prompt('new dimension, e.g. "player class"') || '').trim();
          if (value) draft.dimensions.push({name: value, members: []});
          menu.refresh(sections());
        },
      });
    }
    renderTree(node, rows, {itemClass: 'classdef-item'});
    return node;
  };

  const sections = () => [
    {kind: 'field', label: 'name', value: draft.name, placeholder: 'nameplates',
      onInput: value => { draft.name = value; }},
    {kind: 'node', node: body()},
    {kind: 'buttons', buttons: [{id: 'save', label: 'save', tone: 'adds', onClick: save}]},
  ];

  menu = new Menu({title: 'class definition', sections: sections()});
  menu.openAt(headPoint('cluster-classdef'));
}

document.getElementById('cluster-classdef').onclick = async () => {
  const build = async () => {
    const list = await api('/api/classdefs');
    const names = list.definitions || [];
    return [
      {kind: 'list', empty: 'none yet', multi: false,
        items: names.map(n => ({id: n, label: n, on: classDef && classDef.slug === n})),
        onPick: async item => {
          document.getElementById('filter-dims').innerHTML = '';
          await loadClassDef(item.id);
          await loadClusters();
        }},
      {kind: 'buttons', buttons: [
        {id: 'edit', label: classDef ? 'edit / new' : 'new', onClick: m => { m.close(); classDefEditor(); }},
      ]},
    ];
  };

  await openPicker('cluster-classdef', build, {
    title: 'class definition', persistent: false, status: 'clusters-status',
  });
};

// RESTORED. `runPromote` was called by the promote menu and DEFINED NOWHERE - deleted by 608877b,
// the same rewrite that took camLive, applyTileMode, uniformTiles and recutPool. The only call site
// sits inside an onClick, so the ReferenceError went to the console and the button simply did
// nothing. Its original body called askChoice, which that commit also deleted, so the
// overwrite/merge question is rebuilt on Menu rather than restored verbatim.
async function promoteChoice(res) {
  // A NAME THAT EXISTS IS A DECISION, not an overwrite - the server refuses `new` against an
  // existing set and reports what is in it so the question can be asked with real numbers
  return new Promise(resolve => {
    let answered = false;
    const done = (menu, choice) => { answered = true; menu.close(); resolve(choice); };
    new Menu({
      title: `${res.name} already exists`,
      onDismiss: () => { if (!answered) resolve(null); },
      sections: [
        {kind: 'node', node: (() => {
          const box = document.createElement('div');
          box.className = 'menu-section';
          box.innerHTML =
            `<div class="stat">it holds ${res.existing_rows} rows · this promote has `
            + `${res.would_add} to add</div>`
            + '<div class="stat">merging keeps every row from a source this pool does not hold</div>';
          return box;
        })()},
        {kind: 'buttons', buttons: [
          {id: 'merge', label: 'merge into', tone: 'adds', onClick: m => done(m, 'merge')},
          {id: 'overwrite', label: 'overwrite', tone: 'removes', onClick: m => done(m, 'overwrite')},
        ]},
      ],
    }).openAt(document.getElementById('cluster-promote'));
  });
}

async function runPromote(name, mode = 'new') {
  const status = document.getElementById('clusters-status');
  status.textContent = `promoting into ${name}...`;
  let res;
  try {
    res = await api('/api/promote-training', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name, mode}),
    });
  } catch (err) {
    status.textContent = `promote failed: ${err}`;
    return;
  }
  if (res.exists) {
    const choice = await promoteChoice(res);
    if (!choice) { status.textContent = `${res.name} left as it was`; return; }
    return runPromote(name, choice);
  }
  if (res.error) { status.textContent = res.error; return; }
  const classes = Object.entries(res.classes || {}).map(([k, v]) => `${k} ${v}`).join(' \u00b7 ');
  status.textContent = `${res.name}: ${res.rows} rows ${res.mode === 'merge' ? 'merged' : 'promoted'}`
    + ` from ${(res.sources || []).length} sources`
    + (res.unresolved ? `, ${res.unresolved} unresolved` : '')
    + (classes ? ` - ${classes}` : '');
  // THE CNN TAB HAS TO HEAR ABOUT IT. promoting into a set and then finding the cnn tab still
  // showing the old row count is the whole complaint - so the new set is listed there immediately
  await loadTrain?.();
}

// A SAVE DIALOGUE, like every other "which one, or a new one" in the tool: the sets that exist,
// with their sizes, plus a field for a name that does not exist yet. It was a bare text field
// seeded with one guess, which is how a typo quietly mints a second set nobody meant to make.
document.getElementById('cluster-promote').onclick = async () => {
  let sets = [];
  try {
    sets = (await api('/api/training-sets')).sets || [];
  } catch (err) {
    document.getElementById('clusters-status').textContent = `could not list training sets: ${err}`;
  }
  const go = (menu, name) => {
    if (!name) return;
    menu.close();
    runPromote(name);
  };
  new Menu({
    title: 'promote to a training set',
    sections: [
      {kind: 'list', empty: 'none yet - name one below', multi: false,
        items: sets.map(set => {
          const classes = Object.entries(set.classes || {}).map(([k, v]) => `${k} ${v}`).join(' \u00b7 ');
          return {
            id: set.name, label: set.name,
            // the set the cnn tab has bound is the commonest target, so it is marked rather than
            // merely present - promoting again into what is being trained on is the normal case
            state: set.name === trainingSet ? {opened: true} : {},
            stats: `${set.rows} rows${classes ? ' - ' + classes : ''}`,
          };
        }),
        onPick: (item, _on, menu) => go(menu, item.id)},
      {kind: 'add', placeholder: 'or a new set', button: 'promote',
        onAdd: (value, menu) => go(menu, (value || '').trim())},
    ],
  }).openAt(document.getElementById('cluster-promote'));
};
// the dismiss handler that used to live here hardcoded its triggers - "#draw-open, #cluster-open,
// #train-open" - so every new picker had to be added to that string or it closed on its own opening
// click. Menu owns dismissal now, for every panel, including the two that never had one.

// ---- draw boxes: the cold start. an empty library gives the matcher nothing to match, so the
// first tiles come from a human drawing on whole frames. positions are kept per frame; the
// SIZE is fitted across the whole set on save, since a hand-drawn box is a few px out either way
let drawFrames = [];
let drawStamp = 0;  // bumped per dataset, so frame urls stop colliding between them
let drawBound = '';  // which dataset the frames on screen belong to
let drawBoundPercent = '';  // and at which sampling, so changing it is not read as a no-op
let drawIndex = 0;
let drawBoxes = {};          // frame name -> [{left, top, width, height}]
// ONE NEGATIVE PER POSITIVE, drawn beside it and correctable by hand. Promote used to invent these
// at the last moment and nobody ever saw them, so nothing stopped one landing on the player frame
// or the target frame - which look like a nameplate, and which the detector then learned to fire
// on. Visible and draggable means a wrong one is a two-second fix instead of a training run.
let drawNegatives = {};      // frame name -> [{left, top, width, height}], index-paired with above
let draggingNegative = null;
let drawImg = new Image();
let drawing = null;
// PAN AND ZOOM, as a transform on the canvas itself - the same shared makePanZoom the interface
// tab uses (menu.js), so a drawn rect is always in the frame's own pixels with no scale to get
// wrong at any zoom. Balthazar Fitzpatrick: "fill width (or height) as far as it gets without the other one
// flowing over the screen border. Allow pan and zoom" - a plain fit-both-axes left a wide-short
// window with the frame shrunk far below what its width could hold, wasting the rest as dead
// space. makePanZoom's own reset() fits by width alone and lets the rest be reached by panning,
// which is the same trade the interface tab already made for the same reason.
let drawView = null;

function drawOverlaps(a, boxes, margin) {
  return boxes.some(b => a.left < b.left + b.width + margin && a.left + a.width + margin > b.left &&
                         a.top < b.top + b.height + margin && a.top + a.height + margin > b.top);
}

// somewhere on this frame that is NOT a nameplate. random rather than fixed, because a fixed
// corner would teach the model that one corner is background rather than that background is
// background - and margin-separated from every positive, since a negative overlapping a plate
// teaches the opposite of what it is for
function drawPlaceNegative(box, frameW, frameH, taken) {
  const margin = 8;
  for (let tries = 0; tries < 60; tries++) {
    const candidate = {
      left: Math.round(Math.random() * Math.max(1, frameW - box.width)),
      top: Math.round(Math.random() * Math.max(1, frameH - box.height)),
      width: box.width, height: box.height,
    };
    if (!drawOverlaps(candidate, taken, margin)) return candidate;
  }
  return null;  // a frame too crowded to stand anywhere is better skipped than overlapped
}

function drawRedraw() {
  const canvas = document.getElementById('draw-canvas');
  const ctx = canvas.getContext('2d');
  if (!drawImg.width) return;
  // THE CANVAS SITS AT THE FRAME'S NATURAL SIZE, always - pan and zoom are a CSS transform on the
  // canvas itself (drawView, wired below), not a scale baked into the pixels here. a drawn box is
  // therefore already in frame pixels with nothing to convert at any zoom level, the same reasoning
  // #iface-overlay already relies on.
  canvas.width = drawImg.width;
  canvas.height = drawImg.height;
  ctx.drawImage(drawImg, 0, 0, canvas.width, canvas.height);
  // strokes are undone against the CURRENT zoom so they read the same width whether zoomed in or
  // out, rather than thickening as you zoom in and vanishing as you zoom out
  const z = drawView ? drawView.zoom() : 1;
  const boxes = drawBoxes[drawFrames[drawIndex]] || [];
  ctx.lineWidth = Math.max(1, 2 / z);
  ctx.strokeStyle = '#ff29d6';
  boxes.forEach(b => ctx.strokeRect(b.left, b.top, b.width, b.height));
  // DASHED AND A DIFFERENT COLOUR, so a negative is never mistaken for a plate at a glance
  const negatives = drawNegatives[drawFrames[drawIndex]] || [];
  ctx.setLineDash([6 / z, 4 / z]);
  ctx.strokeStyle = '#4dd6c0';
  negatives.forEach(b => b && ctx.strokeRect(b.left, b.top, b.width, b.height));
  ctx.setLineDash([]);
  if (drawing) {
    ctx.setLineDash([4 / z, 3 / z]);
    ctx.strokeStyle = '#4da6ff';
    ctx.strokeRect(drawing.left, drawing.top, drawing.width, drawing.height);
    ctx.setLineDash([]);
  }
  // one background box is generated per drawn box at promote time, on the assumption that every
  // visible plate got marked - so anywhere else in the frame is background by construction
  const total = Object.values(drawBoxes).reduce((n, l) => n + l.length, 0);
  document.getElementById('draw-count').textContent =
    `${total} positives \u00b7 ${total} negatives`;
  document.getElementById('draw-pos').textContent =
    `${drawIndex + 1} / ${drawFrames.length}  ${drawFrames[drawIndex] || ''}`;
  drawSyncSlider();
}

// THE SLIDER FOLLOWS, IT DOES NOT LEAD. prev/next and the keys move drawIndex; this only mirrors
// it, so there is one source of truth for where we are and no feedback loop between the two
const drawFrameNumber = name => (name || '').replace(/\.[^.]+$/, '') || '-';
function drawSyncSlider() {
  const slider = document.getElementById('draw-slider');
  if (!slider) return;
  const last = Math.max(0, drawFrames.length - 1);
  slider.max = String(last);
  slider.disabled = drawFrames.length < 2;
  if (slider.value !== String(drawIndex)) slider.value = String(drawIndex);
  const first = document.getElementById('draw-first');
  const end = document.getElementById('draw-last');
  if (first) first.textContent = drawFrameNumber(drawFrames[0]);
  if (end) end.textContent = drawFrameNumber(drawFrames[last]);
}

function drawLoad() {
  if (!drawFrames.length) return;
  drawImg = new Image();
  // ONE NEGATIVE PER POSITIVE, INCLUDING FOR BOXES DRAWN BEFORE THEY EXISTED. 313 rectangles were
  // marked before this, and leaving them unpaired would mean the oldest and best-reviewed work is
  // the only work with no background examples. Generated on the frame being looked at, so they can
  // be dragged and are only written when the set is saved.
  drawImg.onload = () => {
    drawFillNegatives(drawFrames[drawIndex]);
    drawView?.reset(drawImg.width, drawImg.height);
    drawRedraw();
  };
  drawImg.src = '/draw-frame/' + encodeURIComponent(drawFrames[drawIndex]) + '?v=' + drawStamp;
}

function drawFillNegatives(name) {
  const boxes = drawBoxes[name] || [];
  const negatives = drawNegatives[name] || [];
  if (!boxes.length || negatives.length >= boxes.length || !drawImg.width) return;
  while (negatives.length < boxes.length) {
    const box = boxes[negatives.length];
    negatives.push(drawPlaceNegative(box, drawImg.width, drawImg.height,
                                     boxes.concat(negatives.filter(Boolean))));
  }
  drawNegatives[name] = negatives;
}

// how much of the recording to offer for drawing. a PERCENTAGE, so the same choice means the
// same coverage whether the run is 40 frames or 400
function drawPercent() {
  const el = document.getElementById('draw-percent');
  return el ? el.dataset.value : '20';
}

async function drawOpen() {
  const data = await api('/api/draw-frames?percent=' + drawPercent());
  // ADOPT ONLY WHAT IS ACTUALLY NEW. entering find used to leave whatever was loaded first sitting
  // there, so rebinding changed the rectangles and left the picture from the old dataset under
  // them. re-reading unconditionally would be worse - it would throw away boxes drawn and not yet
  // saved, which live in the page - so the bound name decides
  const bound = data.dataset || data.recording || '';
  const percent = drawPercent();
  // THE SAMPLING IS PART OF WHAT IS BOUND, not just the dataset. comparing the name alone made the
  // percentage menu inert: it re-fetched a longer frame list, arrived here, and dropped it because
  // the recording had not changed - so every run stayed capped at whatever it first loaded at.
  if (data.ready && bound && bound === drawBound && percent === drawBoundPercent) return;

  // A RESAMPLE IS NOT A REBIND. same recording, more frames of it - so boxes drawn and not yet
  // saved stay in the page, and the frame being looked at is kept if the new list still holds it
  if (data.ready && bound && bound === drawBound) {
    const at = drawFrames[drawIndex];
    drawBoundPercent = percent;
    drawFrames = data.frames;
    const found = at ? drawFrames.indexOf(at) : -1;
    drawIndex = found >= 0 ? found : Math.min(drawIndex, Math.max(0, drawFrames.length - 1));
    drawLoad();
    return;
  }
  const where = document.getElementById('draw-where');
  if (!data.ready) {
    // NO TEXT WHERE NO TEXT IS NEEDED. this said "no frames under /Users/.../sessions" - an
    // absolute path, in the first row, wrapping it to three lines. Nothing is bound yet and the
    // head already says so; the paths themselves live in settings, which is where you would go to
    // change them. The row keeps one height whatever the state is.
    document.getElementById('find-status').textContent = '';
    document.getElementById('draw-pos').textContent = '-';
    drawSyncSlider();
    if (where) { where.textContent = 'recordings'; where.className = 'placeholder'; }
    drawFrames = [];
    return;
  }
  drawBound = data.dataset || data.recording || '';
  drawBoundPercent = percent;
  drawStamp = Date.now();
  drawFrames = data.frames;
  // boxes drawn in an earlier sitting come back from the bound candidates rather than being lost
  drawBoxes = data.boxes || {};
  drawNegatives = data.negatives || {};
  drawIndex = 0;
  if (where) {
    where.textContent = data.dataset || data.recording;
    where.className = '';
  }
  drawLoad();
}

function drawXY(evt) {
  // the box.width/canvas.width ratio IS the live zoom, whatever it currently is - reading it off
  // the rendered box rather than a stored scale means this stays correct through any pan or zoom,
  // the same trick #ifaceCanvasPoint uses for the interface tab's own transformed stage
  const canvas = document.getElementById('draw-canvas');
  const r = canvas.getBoundingClientRect();
  const scale = r.width / (canvas.width || 1);
  return [(evt.clientX - r.left) / scale, (evt.clientY - r.top) / scale];
}

(function wireDraw() {
  const canvas = document.getElementById('draw-canvas');
  const wrap = document.getElementById('draw-wrap');
  drawView = makePanZoom(wrap, canvas, {
    panModifier: 'shift',
    // the whole frame visible and as large as it fits - a drawn box is judged against the frame
    // as a whole, unlike the interface tab where 1:1 pixels matter more than seeing all of it
    fit: 'contain',
    onChange: z => {
      document.getElementById('draw-zoom-value').textContent = `${Math.round(z * 100)}%`;
      drawRedraw();  // stroke widths are undone against zoom, so a redraw keeps them crisp
    },
  });
  document.getElementById('draw-zoom-fit').onclick = () => drawView.reset(drawImg.width, drawImg.height);

  const inside = (b, x, y) => b && x >= b.left && x <= b.left + b.width &&
                              y >= b.top && y <= b.top + b.height;

  canvas.addEventListener('mousedown', evt => {
    if (evt.shiftKey) return;  // shift+drag is a pan, handled on the wrapper by makePanZoom
    const [x, y] = drawXY(evt);
    const name = drawFrames[drawIndex];
    const boxes = drawBoxes[name] || [];
    const negatives = drawNegatives[name] || [];

    // NEGATIVES ARE TESTED FIRST, and they DRAG rather than delete. A negative is placed by the
    // tool and only ever needs correcting - most often onto the player or target frame, which look
    // enough like a nameplate that the detector learns them unless told otherwise
    const onNegative = negatives.findIndex(b => inside(b, x, y));
    if (onNegative >= 0) {
      draggingNegative = {name, index: onNegative, dx: x - negatives[onNegative].left,
                          dy: y - negatives[onNegative].top};
      return;
    }
    const hit = boxes.findIndex(b => inside(b, x, y));
    if (hit >= 0) {
      // a positive and its negative are one decision, so they leave together and the set stays
      // balanced without anyone counting
      boxes.splice(hit, 1);
      negatives.splice(hit, 1);
      drawBoxes[name] = boxes;
      drawNegatives[name] = negatives;
      drawRedraw();
      return;
    }
    drawing = {left: x, top: y, width: 0, height: 0, x0: x, y0: y};
  });
  canvas.addEventListener('mousemove', evt => {
    const [x, y] = drawXY(evt);
    if (draggingNegative) {
      const b = drawNegatives[draggingNegative.name][draggingNegative.index];
      b.left = Math.max(0, Math.min(drawImg.width - b.width, Math.round(x - draggingNegative.dx)));
      b.top = Math.max(0, Math.min(drawImg.height - b.height, Math.round(y - draggingNegative.dy)));
      drawRedraw();
      return;
    }
    if (!drawing) return;
    drawing.left = Math.min(drawing.x0, x);
    drawing.top = Math.min(drawing.y0, y);
    drawing.width = Math.abs(x - drawing.x0);
    drawing.height = Math.abs(y - drawing.y0);
    drawRedraw();
  });
  window.addEventListener('mouseup', () => {
    if (draggingNegative) { draggingNegative = null; drawRedraw(); return; }
    if (!drawing) return;
    if (drawing.width > 6 && drawing.height > 3) {
      const name = drawFrames[drawIndex];
      const box = {
        left: Math.round(drawing.left), top: Math.round(drawing.top),
        width: Math.round(drawing.width), height: Math.round(drawing.height),
      };
      const boxes = (drawBoxes[name] || []).concat([box]);
      const negatives = (drawNegatives[name] || []).slice();
      // placed clear of every positive AND every negative already down, so two do not stack
      negatives.push(drawPlaceNegative(box, drawImg.width, drawImg.height,
                                       boxes.concat(negatives.filter(Boolean))));
      drawBoxes[name] = boxes;
      drawNegatives[name] = negatives;
    }
    drawing = null;
    drawRedraw();
  });
  // a percentage of the recording, as a menu rather than a native select - the OS draws a
  // <select> and it can never match the rest of the tool
  document.getElementById('draw-percent').onclick = evt => {
    const el = evt.currentTarget;
    listMenu('how much to draw on',
      [10, 20, 30, 40, 50, 60, 70, 80, 90, 100].map(n => ({
        id: String(n), label: `${n}%`, on: el.dataset.value === String(n)})),
      item => { el.dataset.value = item.id; el.innerHTML = `<span>${item.id}%</span>`; drawOpen(); }
    ).openAt(el);
  };
  document.getElementById('draw-prev').onclick = () => {
    if (drawIndex > 0) { drawIndex--; drawLoad(); }
  };
  document.getElementById('draw-next').onclick = () => {
    if (drawIndex < drawFrames.length - 1) { drawIndex++; drawLoad(); }
  };
  // input, not change - a scrub should show the frames it passes over rather than only the one it
  // is let go on, which is the entire point of a scrubber on a long run
  document.getElementById('draw-slider').oninput = evt => {
    const at = Number(evt.currentTarget.value);
    if (!drawFrames.length || at === drawIndex) return;
    drawIndex = Math.max(0, Math.min(at, drawFrames.length - 1));
    drawLoad();
  };
  // B AND N STEP FRAMES, and they are NEW despite being asked for as "keep them as they are" -
  // nothing in this file ever bound them, so the request described muscle memory, not code.
  // scoped to the find tab and dropped while typing, like every other bare-key binding here
  document.addEventListener('keydown', evt => {
    if (document.querySelector('.nav-tab.active')?.dataset.tab !== 'find') return;
    if (/^(INPUT|TEXTAREA)$/.test(document.activeElement?.tagName || '')) return;
    if (evt.metaKey || evt.ctrlKey || evt.altKey) return;
    const key = evt.key.toLowerCase();
    if (key !== 'b' && key !== 'n') return;
    evt.preventDefault();
    const to = drawIndex + (key === 'n' ? 1 : -1);
    if (to < 0 || to >= drawFrames.length) return;
    drawIndex = to;
    drawLoad();
  });
  document.getElementById('draw-save').onclick = async () => {
    const boxes = [];
    Object.entries(drawBoxes).forEach(([path, list]) =>
      list.forEach(b => boxes.push({...b, path})));
    if (!boxes.length) return;
    // SENT, NOT IMPLIED. these are the negatives on screen, wherever they were dragged to - the
    // server no longer invents its own, so what was reviewed is exactly what gets trained on
    const negatives = [];
    Object.entries(drawNegatives).forEach(([path, list]) =>
      list.forEach(b => b && negatives.push({...b, path})));
    const res = await api('/api/find-run', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({mode: 'drawn', boxes, negatives}),
    });
    if (res.error) { document.getElementById('find-status').textContent = res.error; return; }
    document.getElementById('find-status').textContent =
      `${res.count} positives + ${res.count} implied negatives, tile fitted to ` +
      `${res.width}x${res.height} (drawn widths ${res.spread.width[0]}-${res.spread.width[1]})` +
      (res.tiles ? ` - ${res.tiles} tiles cut, ready to cluster` : '');
    // saving REBINDS to the newly written set, so the picker has to be told - it named the old
    // one until the page was refreshed by hand
    const now = await api('/api/draw-frames?percent=' + drawPercent());
    const where = document.getElementById('draw-where');
    if (where && now.ready) { where.textContent = now.dataset || now.recording; where.className = ''; }
    drawRedraw();
  };
})();

function applyFindMode() {
  drawOpen();
  setTimeout(drawRedraw, 0);
}


// ---------------------------------------------------------------- camera

// THE SITTING IS WATCHED WHILE IT IS RECORDED, so this polls like the navigation tab does and for
// the same reason: the server re-reads the jsonl wt-record is appending to. It does NOT tap live
// input - the review server would need Input Monitoring for that, and the browser does not start
// anything that records input or drives the game.
const CAM_POLL_MS = 1000;
let camSession = null;   // null means whichever recording is live
let camSize = -1;        // last size drawn, so an unchanged file costs no redraw
let camTimer = null;

function camSteps(step) {
  // the state IS the interface here - nothing is clickable, so the class carries all of it
  return `<div class="cam-step ${step.state}" title="${(step.detail || []).join(' \u00b7 ')}">`
    + `<span class="cam-dot"></span><span class="cam-name">${step.label}</span></div>`;
}

// THE TARGET DISTANCE SITS ON THE CYCLE, beside the setting, because it is read off the strip
// while standing there - a number in a guide on another screen is a number you estimate instead
function camDrawSteps(steps) {
  if (!steps) return;
  byId('cam-setup').innerHTML = steps.setup.map(camSteps).join('');
  byId('cam-closing').innerHTML = steps.closing.map(camSteps).join('');
  // ONE DIVIDED ROW, not a grid: .cam-grid fitted but had no rules between the columns, and a
  // .divider dropped into a repeat(5) grid becomes a sixth track. .label-columns has the rules,
  // and its .col now shrinks, so it does both
  byId('cam-cycles').innerHTML = steps.cycles.map((cycle, i) => `
    ${i ? '<div class="divider"></div>' : ''}
    <div class="col cam-cycle">
      <div class="cam-cycle-t">cycle ${cycle.index + 1}</div>
      <div class="cam-cycle-s">${cycle.setting}</div>
      <div class="cam-cycle-d">far ${cycle.far_yards} yd, then near ${cycle.near_yards} yd</div>
      <div class="cam-cycle-u">far is this camera's measured limit - past it the plate clamps</div>
      ${cycle.steps.map(camSteps).join('')}
    </div>`).join('');
}

function camDrawLog(events) {
  const el = byId('cam-log');
  el.innerHTML = (events || []).map(e =>
    `<div class="cam-ev"><span class="cam-ev-t">${e.t.toFixed(1)}s</span>${e.text}</div>`).join('');
  // TAILS. the newest line is the one you are living in, and an operator watching mid-sitting must
  // not have to scroll to it - older lines are allowed off the top instead
  el.scrollTop = el.scrollHeight;
}

// RESTORED. this was deleted by 608877b - the same rewrite that took applyTileMode and three
// others - and the loss hid for a day because both call sites sit inside camLoad(...).catch(),
// so the ReferenceError was swallowed on every poll. the cost was the whole live strip reading
// '-' and cam-status never advancing past "nothing recorded yet"
function camLive(live) {
  if (!live) return;
  const anchored = a => a ? '' : ' <span class="cam-drift">drifting</span>';
  // DISTANCE AND BEARING TO THE MOB, because that is what you position by. the anchor is the mob:
  // step 1 stands you in it, so this is measured rather than fitted
  byId('cam-dist').textContent =
    live.distance_yards === null || live.distance_yards === undefined
      ? '-' : `${live.distance_yards.toFixed(1)} yd`;
  byId('cam-bearing').textContent =
    live.bearing_degrees === null || live.bearing_degrees === undefined
      ? '-' : `${live.bearing_degrees > 0 ? '+' : ''}${live.bearing_degrees} deg`;
  byId('cam-pitch').innerHTML =
    `${live.pitch_degrees.toFixed(1)} deg${anchored(live.pitch_anchored)}`;
  byId('cam-zoom').innerHTML =
    `${live.zoom_ticks.toFixed(0)} notches${anchored(live.zoom_anchored)}`;
  // a left-drag swings the camera off the character's facing; say so while it is off
  const orbit = live.orbit_degrees ? `, camera ${live.orbit_degrees > 0 ? '+' : ''}${live.orbit_degrees.toFixed(0)}` : '';
  byId('cam-facing').textContent =
    live.facing_degrees === null ? '-' : `${live.facing_degrees.toFixed(0)} deg${orbit}`;
  byId('cam-pos').textContent =
    live.x === null ? '-' : `${live.x.toFixed(1)}, ${live.y.toFixed(1)}`;
  // at the anchor both axes are pinned to a known absolute, which is the one state worth calling out
  const steady = byId('cam-steady');
  const atAnchor = live.first_person && live.pitch_anchored && live.zoom_anchored;
  steady.textContent = atAnchor ? 'at the anchor' : 'set';
  steady.className = 'cam-v' + (atAnchor ? ' cam-ok' : '');
}

async function camLoad(force) {
  const q = camSession ? `?session=${encodeURIComponent(camSession)}` : '';
  const data = await api('/api/camera' + q);
  const status = byId('cam-status');
  // AN ERROR IS NOT A REASON TO DRAW NOTHING. "no session bound" still has a protocol to show, all
  // of it grey - the server sends the steps either way, so only the message is conditional
  if (data.steps) { camDrawSteps(data.steps); camDrawLog(data.events); camLive(data.live); }
  if (data.error) { status.textContent = data.error; return; }
  if (!force && data.size === camSize) return;   // nothing appended since the last look
  camSize = data.size;
  if (data.name) byId('cam-open').innerHTML = `<span>${data.name}</span>`;
  camLive(data.live);
  camDrawSteps(data.steps);
  camDrawLog(data.events);
  const cycles = data.steps ? data.steps.cycles : [];
  const done = cycles.filter(c => c.steps.every(s => s.state === 'done')).length;
  status.textContent = `${done}/${cycles.length} clean`;
}

// THE HEAD WAS INERT. it carried dropdown-head, camLoad wrote the recording's name into it, and
// nothing ever bound a click - so the only way to re-point the tab was to leave it and come back,
// and there was no way at all to look at an older sitting. Same shape as every other picker here.
byId('cam-open').onclick = async () => {
  let menu = null;
  const chosen = new Set();

  const build = async () => {
    const data = await api('/api/nav-sessions');
    const following = camSession === null;
    const rows = (data.sessions || []).map(item => ({
      id: item.name,
      label: item.name,
      on: chosen.has(item.name),
      state: item.name === camSession ? {opened: true} : {},
    }));
    return [
      // WATCHING FOR A LIVE RECORDING IS A CHOICE, so it is a row you can see and return to, not an
      // invisible default you fall into. the server already picks the newest file when no session
      // is named, so this only has to stop naming one
      {kind: 'list', multi: true, empty: 'nothing recorded yet',
        onPick: (item, on) => {
          chosen.clear();
          if (on) chosen.add(item.id);
          menu.setButtonEnabled('open', chosen.size > 0);
        },
        items: [
          {heading: following ? 'watching for the live recording' : 'watching one recording'},
          ...rows,
        ]},
      {kind: 'buttons', buttons: [
        {id: 'open', label: 'open', tone: 'adds', enabled: chosen.size > 0, onClick: async m => {
          camSession = [...chosen][0];
          chosen.clear();
          camSize = -1;
          byId('cam-open').innerHTML = `<span>${camSession}</span>`;
          await camLoad(true);
          m.refresh(await build());
        }},
        {id: 'close-set', label: 'close', tone: 'removes', enabled: !following, onClick: async m => {
          camSession = null;
          camSize = -1;
          byId('cam-open').innerHTML = '<span>live recording</span>';
          await camLoad(true);
          m.refresh(await build());
        }},
      ]},
    ];
  };

  menu = await openPicker('cam-open', build, {title: 'recordings', status: 'cam-status'});
};

// LIVE MODE follows wt-camera-live instead of a recording: pitch and zoom counted from the
// operator's own wheel and drag since the fully-in, fully-down anchor, and the views saved so far
let camMode = 'recording';
// the readout is watched while the wheel turns, so it polls faster than a recording
const CAM_LIVE_POLL_MS = 250;

function camApplyMode() {
  const live = camMode === 'live';
  byId('cam-mode').textContent = live ? 'live' : 'recording';
  byId('cam-mode').classList.toggle('active', live);
  byId('cam-open').classList.toggle('hidden', live);
  byId('cam-body').classList.toggle('hidden', live);
  byId('cam-views-body').classList.toggle('hidden', !live);
  // the readout knows pitch and zoom only; the boxes about the mob and the player sit it out
  for (const id of ['cam-dist', 'cam-bearing', 'cam-facing', 'cam-pos']) {
    byId(id).textContent = '-';
    byId(id).parentElement.classList.toggle('hidden', live);
  }
}

async function camLiveLoad() {
  const data = await api('/api/camera-live');
  const known = data.pitch_degrees !== null && data.pitch_degrees !== undefined;
  const exact = data.exact ? '' : ' <span class="cam-drift">not exact</span>';
  byId('cam-pitch').innerHTML = known ? `${data.pitch_degrees.toFixed(1)} deg${exact}` : '-';
  byId('cam-zoom').innerHTML = known ? `${data.zoom_ticks.toFixed(0)} notches${exact}` : '-';
  const steady = byId('cam-steady');
  steady.textContent = !data.live ? 'no feed' : known ? (data.exact ? 'counting' : 're-anchor') : 'press f9 at the stop';
  steady.className = 'cam-v' + (data.live && data.exact ? ' cam-ok' : '');
  byId('cam-live-note').textContent = data.note || '';
  const views = Object.entries(data.views || {}).sort(([a], [b]) => a.localeCompare(b));
  byId('cam-views').innerHTML = views.length
    ? views.map(([key, v]) =>
        `<div class="cam-ev"><span class="cam-ev-t">${key}</span>` +
        `pitch ${v.pitch_degrees.toFixed(1)} deg, zoom ${v.zoom_ticks.toFixed(0)} notches` +
        `${v.source ? ` <span class="cam-drift">${v.source}</span>` : ''}</div>`).join('')
    : '<div class="cam-ev cam-drift">none saved yet</div>';
}

byId('cam-mode').onclick = () => {
  camMode = camMode === 'live' ? 'recording' : 'live';
  camApplyMode();
  camEnter();
};

function camEnter() {
  if (camTimer) clearInterval(camTimer);
  camApplyMode();
  if (camMode === 'live') {
    camTimer = setInterval(() => camLiveLoad().catch(() => {}), CAM_LIVE_POLL_MS);
    camLiveLoad().catch(() => {});
    return;
  }
  camSize = -1;
  if (camSession === null) {
    byId('cam-open').innerHTML = '<span>live recording</span>';
  }
  camTimer = setInterval(() => camLoad(false).catch(() => {}), CAM_POLL_MS);
  camLoad(true).catch(() => {});
}

// find is the cold-start tab, and the one a refresh mid-drawing should not throw him out of
initShell({onEnter: enterTab, fallback: 'find'});
// point find at whatever is already bound, so it never opens demanding paths to be typed
applyFindMode();


// ---- selecting crops in discard / promote --------------------------------------------------
// PICKING FORTY TILES USED TO MEAN FORTY CMD+CLICKS, so nobody picked forty tiles - they picked
// three, tagged, and repeated. Range, select-all and a visible count are the whole of the fix.
(function wireClusterSelection() {
  const centre = document.getElementById('cluster-smart-center');
  if (centre) centre.onclick = smartCenterPicked;
  const all = document.getElementById('cluster-select-all');
  const clear = document.getElementById('cluster-clear-sel');
  if (all) {
    all.onclick = () => {
      // everything the filter is currently showing, not everything in the pool - selecting what
      // cannot be seen is how a tag lands somewhere nobody looked
      // ONE update for the whole batch rather than one per tile - setKeys fires onChange once
      clusterSel?.setKeys(
        clusterVisible.filter(item => item.source !== 'library').map(item => item.name), true);
    };
  }
  if (clear) clear.onclick = () => clearLabelSelection();

  document.addEventListener('keydown', evt => {
    // the shell marks the live tab, so no second source of truth about where we are
    if (document.querySelector('.nav-tab.active')?.dataset.tab !== 'cluster') return;
    const typing = /^(INPUT|TEXTAREA)$/.test(document.activeElement?.tagName || '');
    if (typing) return;
    if (evt.key === 'Escape' && clusterSel?.size()) { clearLabelSelection(); return; }
    if ((evt.metaKey || evt.ctrlKey) && evt.key.toLowerCase() === 'a') {
      evt.preventDefault();
      all?.onclick();
    }
  });
})();


// ---- guidance on demand: hover to peek, click to pin -----------------------------------------
// A CONTROL ROW MAY NOT CHANGE HEIGHT, which is what instructions-as-prose did - three lines at a
// narrow window, and the row grew with them. Balthazar Fitzpatrick's rule: "if there's only instructions, do the ?
// in a circle and tooltip it", and "hovering should show and collapse it when the mouse moves away,
// clicking it opening it permanently until clicked somewhere else OR on the questionmark again".
//
// ONE IMPLEMENTATION, used by every ? on the page - a second one would drift into a different
// dialect, which is the whole reason this row's prose became a button in the first place.
let helpPinned = null;

function helpTip(button, lines) {
  const tip = document.createElement('div');
  tip.className = 'help-tip';
  tip.hidden = true;
  tip.innerHTML = lines.map(line => `<div>${line}</div>`).join('');
  document.body.appendChild(tip);

  const place = () => {
    const box = button.getBoundingClientRect();
    tip.hidden = false;
    // clamped to the window, because a ? at the right-hand edge would otherwise open off-screen
    const width = tip.offsetWidth;
    tip.style.top = `${box.bottom + 6}px`;
    tip.style.left = `${Math.max(8, Math.min(box.left, window.innerWidth - width - 8))}px`;
  };
  const hide = () => { tip.hidden = true; tip.classList.remove('pinned'); button.classList.remove('on'); };

  button.addEventListener('mouseenter', () => { if (helpPinned !== tip) place(); });
  button.addEventListener('mouseleave', () => { if (helpPinned !== tip) tip.hidden = true; });
  button.addEventListener('click', evt => {
    evt.stopPropagation();
    if (helpPinned === tip) { helpPinned = null; hide(); return; }
    if (helpPinned) helpPinned._hide();
    helpPinned = tip;
    place();
    tip.classList.add('pinned');
    button.classList.add('on');
  });
  tip._hide = hide;
  return tip;
}

// anywhere else dismisses a pinned tip - including a click inside the tip itself, which is text to
// read rather than something to interact with
document.addEventListener('click', () => {
  if (helpPinned) { helpPinned._hide(); helpPinned = null; }
});

helpTip(document.getElementById('draw-help'), [
  'drag on the frame to add a box',
  'click a box to remove it',
  'everything not boxed counts as background, so each box you draw implies a negative',
]);


helpTip(byId('cam-help'), [
  'this watches the recording, it does not start it',
  'run <b>wt-record --frame-hz 10 --frame-width 2560</b> in a terminal first',
  'a step goes red when it was spoiled, with the reason - do that cycle again',
]);

// ---- housekeeping: find and clear what a deleted recording left behind ---------------------
// LEFT COLUMN PICKS A RECORDING (live or orphaned), RIGHT SHOWS WHAT IT OWNS. the server does
// the tag-peeling and the reverse lookup from box/tile/set filenames back to a recording; this
// only renders what comes back and turns checked rows into a flat list of absolute paths to send
// to archive or delete.
let hkRecordings = [];
let hkSelectedTag = null;
let hkAssetGroups = {};
let hkRowPaths = {};       // checkbox id -> the file paths it represents
let hkChecked = new Set(); // checkbox ids currently ticked

const HK_KIND_LABEL = {
  boxes: 'boxes', tiles: 'tiles', tiles_synth: 'tiles (synth)',
  sets: 'sets', checkpoints: 'checkpoints',
};

async function hkLoad() {
  hkStatus('');
  const data = await api('/api/housekeeping-recordings');
  hkRecordings = data.recordings;
  hkRenderRecordings();
  // a selection surviving a reload would show assets for a recording no longer in the list
  if (hkSelectedTag && !hkRecordings.some(r => r.tag === hkSelectedTag)) {
    hkSelectedTag = null;
    document.getElementById('hk-assets').innerHTML = '';
  }
}

function hkRenderRecordings() {
  const col = document.getElementById('hk-recordings');
  col.innerHTML = '';
  const groups = {};
  hkRecordings.forEach(r => { (groups[r.group] ||= []).push(r); });
  Object.keys(groups).sort().forEach(group => {
    const heading = document.createElement('div');
    heading.className = 'field-label';
    heading.textContent = group;
    col.appendChild(heading);
    groups[group].sort((a, b) => a.path.localeCompare(b.path)).forEach(r => {
      const row = document.createElement('div');
      row.className = 'toggle hk-recording-row' + (r.tag === hkSelectedTag ? ' on' : '');
      row.title = r.path;
      const badge = r.missing ? '<span class="hk-rec-badge">orphaned</span>' : '';
      row.innerHTML = `<span class="hk-rec-name">${r.path}</span>${badge}`;
      row.onclick = () => hkSelectRecording(r.tag);
      col.appendChild(row);
    });
  });
  if (!hkRecordings.length) {
    col.innerHTML = '<div class="stat">no recordings, and nothing orphaned</div>';
  }
}

function hkSelectRecording(tag) {
  hkSelectedTag = tag;
  hkChecked.clear();
  hkRenderRecordings();
  hkLoadAssets(tag);
}

async function hkLoadAssets(tag) {
  hkAssetGroups = await api(`/api/housekeeping-assets?tag=${encodeURIComponent(tag)}`);
  hkRenderAssets();
}

function hkFormatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function hkRenderAssets() {
  const box = document.getElementById('hk-assets');
  box.innerHTML = '';
  hkRowPaths = {};
  hkChecked.clear();
  let n = 0;
  Object.keys(HK_KIND_LABEL).forEach(kind => {
    const rows = hkAssetGroups[kind] || [];
    if (!rows.length) return;
    const heading = document.createElement('div');
    heading.className = 'field-label';
    heading.textContent = HK_KIND_LABEL[kind];
    box.appendChild(heading);
    rows.forEach(row => {
      const id = `hk-row-${n++}`;
      hkRowPaths[id] = row.paths;
      const label = document.createElement('label');
      label.className = 'toggle hk-asset-row';
      label.innerHTML = `<input type="checkbox" class="hk-check" id="${id}">` +
        `<span class="hk-asset-label">${row.label}</span>` +
        `<span class="hk-asset-size">${hkFormatSize(row.size_bytes)}</span>`;
      label.querySelector('input').onchange = evt => {
        if (evt.target.checked) hkChecked.add(id); else hkChecked.delete(id);
        hkUpdateSelectedCount();
      };
      box.appendChild(label);
    });
  });
  if (!box.children.length) {
    box.innerHTML = '<div class="stat">nothing under training/ traces back to this recording</div>';
  }
  hkUpdateSelectedCount();
}

function hkUpdateSelectedCount() {
  document.getElementById('hk-selected-count').textContent =
    hkChecked.size ? `${hkChecked.size} selected` : 'nothing selected';
}

function hkSelectedPaths() {
  const out = [];
  hkChecked.forEach(id => out.push(...(hkRowPaths[id] || [])));
  return out;
}

function hkStatus(msg) {
  document.getElementById('hk-status').textContent = msg;
}

(function wireHousekeeping() {
  document.getElementById('hk-select-all').onclick = () => {
    document.querySelectorAll('.hk-check').forEach(cb => {
      cb.checked = true;
      hkChecked.add(cb.id);
    });
    hkUpdateSelectedCount();
  };

  document.getElementById('hk-archive').onclick = async () => {
    const paths = hkSelectedPaths();
    if (!paths.length) { hkStatus('nothing selected'); return; }
    try {
      const res = await api('/api/housekeeping-archive', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({paths}),
      });
      hkStatus(`archived ${res.moved.length} file(s) to _archive/`);
    } catch (err) {
      hkStatus(`archive failed: ${err.message}`);
    }
    hkLoadAssets(hkSelectedTag);
    hkLoad();
  };

  // DELETE IS IRREVERSIBLE, and the click alone must never be enough - a second explicit step,
  // naming the count, is what separates "clearing clutter" from "lost data with no copy"
  document.getElementById('hk-delete').onclick = async () => {
    const paths = hkSelectedPaths();
    if (!paths.length) { hkStatus('nothing selected'); return; }
    const sure = confirm(
      `delete ${paths.length} file(s) for good?\n\nthis cannot be undone - training data has no second copy. archive instead unless you are sure.`
    );
    if (!sure) return;
    try {
      const res = await api('/api/housekeeping-delete', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({paths}),
      });
      hkStatus(`deleted ${res.moved.length} file(s)`);
    } catch (err) {
      hkStatus(`delete failed: ${err.message}`);
    }
    hkLoadAssets(hkSelectedTag);
    hkLoad();
  };
})();


// ================================================================================
// RESTORED 2026-09-09. The navigation tab was built on 2026-09-03 and then lost, not
// deleted: 52952ec ('the navigation tab is one zoomable map with layers and opacities')
// renamed data-tab="navigation" to data-tab="map" and rewrote 698 lines of this file,
// and the walk drawing and the pop-out went with it. The SERVER never stopped serving them
// - /api/nav, /api/nav-latest and /api/nav-heat are all still live routes, and
// parent/nav/walk.py is untouched - so this is the client half being reconnected
// beside the map tab rather than anything new. Balthazar Fitzpatrick: "do the navigation
// tab, the map tab already exists".
// ================================================================================
// ---- navigation tab: the walk, and the zones marked while walking --------------------------
// REPLACED THE OVERLAY TAB. the playback-over-game overlay went with it, on Balthazar Fitzpatrick's call - what
// he needs from a window floating over the game is which nav line he is on, not a replay.
//
// the server owns the model (nav/walk.py) and hands over plain points; this only draws. that is
// deliberate - an offset protocol streaming raw rows would mean rebuilding the model in javascript,
// which is one idea implemented twice, and two implementations of one idea drifting apart is the
// failure this codebase keeps paying for.
let navSession = null;     // null means "whatever is being recorded right now"
let navMapId = null;       // null means the first map the session saw
let navSize = -1;          // last file size drawn, so an unchanged file costs no redraw
let navMaps = [];

const NAV_POLL_MS = 1000;  // the recording is 1 Hz; polling faster would only re-read the same file

async function navLoad(force) {
  const q = navSession ? `?session=${encodeURIComponent(navSession)}` : '';
  const data = await api('/api/nav' + q);
  const status = byId('nav-status');
  if (data.error) { status.textContent = data.error; return; }
  if (!force && data.size === navSize) return;  // nothing appended since the last look
  navSize = data.size;
  navMaps = data.maps || [];
  if (data.name) byId('nav-open').innerHTML = `<span>${data.name}</span>`;
  navDraw();
}

function navCurrent() {
  if (!navMaps.length) return null;
  return navMaps.find(m => m.map_id === navMapId) || navMaps[0];
}

// wowhead zone images, loaded once per url and reused - a walked map draws every poll while
// following live, and re-fetching a 70KB jpg every second would be wasteful for a picture that
// never changes
const zoneImageCache = new Map();
function zoneImage(url) {
  if (!zoneImageCache.has(url)) {
    const img = new Image();
    img.src = url;
    zoneImageCache.set(url, img);
  }
  return zoneImageCache.get(url);
}

// COORDINATES ARE 0-100 PER AXIS, the same zone-percentage convention wowhead's own live map
// coordinate readout uses - verified against a real recording (x 35-59, y 26-39 on one session,
// squarely inside 0-100). So a zone image needs no per-map calibration: x/100 * width, y/100 *
// height. ORIENTATION UNCHECKED - see nav/zone_catalogue.py's header - which corner is (0,0) has
// not been confirmed against a real screen, so this is the working assumption, not a verified fact.
function drawZoneImage(ctx, canvas, url, onArrive) {
  const img = zoneImage(url);
  if (!img.complete || !img.naturalWidth) {
    img.onload = onArrive || null;  // draw again once it actually arrives
    return null;
  }
  // LETTERBOXED, NOT STRETCHED. drawImage into an arbitrary canvas aspect would visibly distort
  // wowhead's own art, and this is the only thing standing between "this tool understands the
  // world" and "this tool is guessing" on first look. the returned rect is what points project
  // against, so the walk still lands on the right pixels of the smaller drawn image.
  const scale = Math.min(canvas.width / img.naturalWidth, canvas.height / img.naturalHeight);
  const dw = img.naturalWidth * scale, dh = img.naturalHeight * scale;
  const dx = (canvas.width - dw) / 2, dy = (canvas.height - dh) / 2;
  ctx.globalAlpha = 0.8;  // the walk must stay legible over the art, not compete with it
  ctx.drawImage(img, dx, dy, dw, dh);
  ctx.globalAlpha = 1;
  return {x: dx, y: dy, width: dw, height: dh};
}

function navDraw() {
  const canvas = byId('nav-canvas');
  const wrap = byId('nav-canvas-wrap');
  const status = byId('nav-status');
  if (!canvas || !wrap) return;
  const map = navCurrent();
  canvas.width = wrap.clientWidth || 900;
  canvas.height = wrap.clientHeight || 420;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!map || !map.bounds) {
    status.textContent = 'nothing recorded';
    return;
  }

  // WHEN A ZONE IMAGE IS KNOWN, THE CANVAS IS THE WHOLE ZONE, not just where you have walked - a
  // path fitted to its own bounds would float over an arbitrary crop of the art with no relation
  // to the picture under it. only fall back to fitting the walk's own bounds when there is no
  // image to register against.
  if (map.zone_image) {
    const rect = drawZoneImage(ctx, canvas, map.zone_image, () => navDraw());
    if (rect) {
      navDrawOnZone(ctx, rect, map);
      navMarkSelected(ctx, (x, y) => [rect.x + (x / 100) * rect.w, rect.y + (y / 100) * rect.h]);
      navRenderTable();
      return;
    }
  }

  // FIT THE BOUNDS, HOLD THE ASPECT. these are game units with no map image behind them, so the
  // only wrong thing a projection can do here is stretch one axis and make a square walk look
  // rectangular - which would make a boundary impossible to recognise
  const [minX, minY, maxX, maxY] = map.bounds;
  const pad = 24;
  const spanX = Math.max(maxX - minX, 1e-6), spanY = Math.max(maxY - minY, 1e-6);
  const scale = Math.min((canvas.width - 2 * pad) / spanX, (canvas.height - 2 * pad) / spanY);
  const offX = (canvas.width - spanX * scale) / 2, offY = (canvas.height - spanY * scale) / 2;
  // y grows downward on a canvas and northward in the world, so it is flipped once, here
  const at = (x, y) => [offX + (x - minX) * scale, canvas.height - (offY + (y - minY) * scale)];

  navDrawWalk(ctx, map, at);
  navMarkSelected(ctx, at);
  navSetStatus(status, map);
  navRenderTable();
}

// THE SELECTED POINT, RINGED. the table is where a coordinate is read and typed; this is what
// catches one typed wrong, which is the whole reason there is a map beside the numbers at all
function navMarkSelected(ctx, at) {
  if (!navSelected) return;
  const line = navLines()[navSelected[0]];
  const point = line && line.points[navSelected[1]];
  if (!point) return;
  const [px, py] = at(point[0], point[1]);
  ctx.save();
  ctx.strokeStyle = '#ffd400';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(px, py, 9, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

// guide mode's planned route - a dashed sky-blue line, distinct from the walked path (dim) and the
// go/no-go lines (lichen/stone) - plus the waypoint it is heading for. prev_plan draws faded so a
// replan reads as a fade, not a pop.
const PLAN_COLOUR = '#5fb3ff';

function navDrawPlan(ctx, plan, at, alpha) {
  if (!plan || !plan.path || plan.path.length < 2) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = PLAN_COLOUR;
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  plan.path.forEach(([x, y], i) => {
    const [px, py] = at(x, y);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.stroke();
  ctx.setLineDash([]);
  if (plan.wp) {
    const [px, py] = at(plan.wp[0], plan.wp[1]);
    ctx.fillStyle = PLAN_COLOUR;
    ctx.beginPath();
    ctx.arc(px, py, 5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

// SHARED BY BOTH PROJECTIONS - fitted-to-bounds (no zone image known) and 0-100-over-the-zone
// (one is). Same path, same lines, same colours either way; only `at(x, y) -> [px, py]` differs.
function navDrawWalk(ctx, map, at) {
  const css = getComputedStyle(document.documentElement);
  const lichen = css.getPropertyValue('--lichen').trim() || '#c7ed5f';
  const stone = css.getPropertyValue('--stone-red').trim() || '#996b62';
  const dim = css.getPropertyValue('--muted').trim() || '#8a8a8a';

  navDrawPlan(ctx, map.prev_plan, at, 0.3);
  navDrawPlan(ctx, map.plan, at, 0.9);

  if (map.path.length > 1) {
    ctx.strokeStyle = dim;
    // 1.5, not 1 - against the plain axis-fit view a hairline is fine, but the same line all but
    // vanishes over a busy zone image. one width for both, since the walk is the same walk.
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    map.path.forEach(([x, y], i) => {
      const [px, py] = at(x, y);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }
  // where you are now, at the head of the walk
  if (map.path.length) {
    const [px, py] = at(...map.path[map.path.length - 1]);
    ctx.fillStyle = dim;
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  for (const line of map.lines) {
    if (!line.points.length) continue;
    const colour = line.kind === 'nogo' ? stone : lichen;
    ctx.strokeStyle = colour;
    ctx.fillStyle = colour;
    ctx.lineWidth = 2;
    ctx.beginPath();
    line.points.forEach(([x, y], i) => {
      const [px, py] = at(x, y);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    // THE CLOSING EDGE IS DRAWN, NEVER STORED - a line snaps back to its first point, so a
    // boundary reads as a boundary the moment it has three corners rather than once it is "done"
    if (line.closed) ctx.closePath();
    ctx.stroke();
    for (const [x, y] of line.points) {
      const [px, py] = at(x, y);
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function navSetStatus(status, map) {
  const zones = map.lines.filter(l => l.closed).length;
  const marks = map.lines.reduce((n, l) => n + l.points.length, 0);
  const where = map.zone_name ? `${map.zone_name} (map ${map.map_id})` : `map ${map.map_id}`;
  status.textContent =
    `${where} · ${map.path.length} positions · ${map.lines.length} line(s), ` +
    `${zones} closed · ${marks} marks`;
  const head = byId('nav-map');
  head.innerHTML = `<span>${map.zone_name || `map ${map.map_id}`}</span>`;
  head.dataset.value = String(map.map_id);
  byId('nav-legend').hidden = !map.plan;
}

function navDrawOnZone(ctx, rect, map) {
  // 0-100 PER AXIS, projected against the drawn (letterboxed) image rect - see drawZoneImage's
  // own note for the evidence this convention is safe to assume
  const at = (x, y) => [rect.x + (x / 100) * rect.width, rect.y + (y / 100) * rect.height];
  navDrawWalk(ctx, map, at);
  navSetStatus(byId('nav-status'), map);
}

function navFollowing() {
  return navSession === null;
}

function navSetFollow(on) {
  if (navTimer) { clearInterval(navTimer); navTimer = null; }
  if (!on) return;
  // following means the live recording, whichever one that turns out to be
  navSession = null;
  navSize = -1;
  byId('nav-open').innerHTML = '<span>recordings</span>';
  navTimer = setInterval(() => navLoad(false).catch(() => {}), NAV_POLL_MS);
  navLoad(true).catch(() => {});
}

// ONE DROPDOWN, TWO SECTIONS - not a picker plus a separate follow toggle. Following the live
// recording and opening an old one are the same choice ("which walk am I looking at"), and two
// controls for one choice can disagree: the toggle could read `on` while a recording was open.
byId('nav-open').onclick = async () => {
  let menu = null;

  // SINGLE COLUMN, SAME STRUCTURE. one column does not mean a different set of rules - the open
  // section, the persistent panel and the close verb are how every picker in the tool behaves, and
  // this one was the odd one out
  const build = async () => {
    const data = await api('/api/nav-sessions');
    const sessions = data.sessions || [];
    const following = navFollowing();
    const open = following
      ? [{id: '', label: 'live recording', opened: true, state: {opened: true}}]
      : (navSession ? [{id: navSession, label: navSession, opened: true, state: {opened: true}}] : []);
    const shut = [
      ...(following ? [] : [{id: '', label: 'live recording'}]),
      ...sessions.filter(x => x.name !== navSession).map(x => ({id: x.name, label: x.name})),
    ];

    return [
      {kind: 'list', empty: 'none', multi: false,
        onPick: item => {
          if (item.opened) return;  // an open one is let go with the close button
          if (item.id === '') { navSetFollow(true); } else {
            navSetFollow(false);
            navSession = item.id;
            navMapId = null;
            navSize = -1;
            byId('nav-open').innerHTML = `<span>${item.id}</span>`;
            navLoad(true).catch(() => {});
          }
          build().then(sections => menu.refresh(sections));
        },
        items: [
          ...(open.length ? [{heading: 'open'}, ...open] : []),
          ...(shut.length ? (open.length ? [{heading: 'not open'}, ...shut] : shut) : []),
        ]},
      {kind: 'buttons', buttons: [
        {id: 'close-set', label: 'close', tone: 'removes', enabled: !following,
          onClick: async () => {
            navSetFollow(true);
            menu.refresh(await build());
          }},
      ]},
    ];
  };

  menu = await openPicker('nav-open', build, {title: 'recordings', status: 'nav-status'});
};

byId('nav-map').onclick = evt => {
  const el = evt.currentTarget;
  const items = navMaps.map(m => ({id: String(m.map_id), label: `map ${m.map_id}`}));
  if (!items.length) return;
  listMenu('map', items, item => {
    navMapId = Number(item.id);
    navDraw();
  }).openAt(el);
};

// ---- looking up an element that may have been popped out ------------------------------------
// THE BUG THIS FIXES: popOutPanel MOVES the panel into a documentPictureInPicture window, so its
// elements leave the main document. Every `byId('cam-...')` then returns null
// and each update writes nowhere - the window floats, shows its last paint, and never changes
// again. Balthazar Fitzpatrick, first time it was actually driven: "The popout doesn't read the
// data back to me". The nav panel had the identical latent bug since the day it was written; it
// had simply never been opened.
//
// Falls back to the main document first, so this is a superset of getElementById and safe to call
// whether or not anything is floating.
const poppedOutDocs = new Set();

function byId(id) {
  const here = document.getElementById(id);
  if (here) return here;
  for (const doc of poppedOutDocs) {
    const there = doc.getElementById(id);
    if (there) return there;
  }
  return null;
}

// ---- while a panel floats over the game, the page must not answer the keyboard --------------
// THE BUG THIS FIXES, 09 Sep: Balthazar Fitzpatrick popped the camera panel out, played, and came back to
// "a lot of popups are open on the camera tab, and none of them belong there. Like dataset open,
// sets being judged".
//
// ui_base's tab strip implements the standard ARIA tablist keys: ArrowLeft/ArrowRight move between
// tabs and Enter/Space activate one, and activateTab runs that tab's loader. Those are WoW's turn
// keys and its jump key. So every stray keystroke that reached the browser while he thought he was
// playing walked the page along the tab strip, firing the dataset picker and the judging set on
// the way past.
//
// NOT FIXED IN ui_base, DELIBERATELY. That behaviour is correct for a tablist and wrong only in
// this one mode - a window floating over a game the user is typing into. The mode lives here, so
// the guard does too.
//
// Capture phase on the document runs BEFORE the element's own onkeydown, so stopping it there is
// what actually prevents ui_base's handler; preventDefault alone would not. Scoped as tightly as
// it can be: only the tab strip, only those keys, only while something is popped out - typing into
// an input in the main window still works.
const TABSTRIP_KEYS = new Set(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Enter', ' ']);

document.addEventListener('keydown', evt => {
  if (!poppedOutDocs.size) return;
  const on = document.activeElement;
  if (!on || !on.classList || !on.classList.contains('nav-tab')) return;
  if (!TABSTRIP_KEYS.has(evt.key)) return;
  evt.stopPropagation();
  evt.preventDefault();
}, true);

// ---- the pop-out: a panel itself, floating above the game ----
// THE REAL PANEL MOVES, IT IS NOT CLONED. a clone stops updating the moment it is made, which for
// a live view is exactly the wrong thing - the whole reason to float it over the game is to watch
// the numbers change as you play. it moves back when the window closes.
//
// ONE FUNCTION FOR EVERY TAB. this was written against the navigation panel by id and nothing
// else could use it. Balthazar Fitzpatrick asked for the camera tab the same way, on a single
// screen, so the panel name is now an argument.
//
// NOTHING HERE RECORDS THE SCREEN. requestWindow opens a browser window the user places himself;
// it reads no pixels and needs no os permission. That matters because a floating panel over a
// game looks like the sort of thing that would.
function popOutPanel(name, buttonId, statusId, redraw) {
  const btn = document.getElementById(buttonId);
  if (!btn) return;
  btn.onclick = async () => {
    const status = document.getElementById(statusId);
    if (!window.documentPictureInPicture) {
      if (status) status.textContent = 'this browser cannot float a window on top - chromium only';
      return;
    }
    const panel = document.querySelector('[data-panel="' + name + '"]');
    const home = panel.parentNode;
    // WHAT THE TAB SHOWS WHILE THE PANEL IS ELSEWHERE. the panel is moved, not copied, so the tab
    // body would otherwise be blank and read as broken - which on a single screen is the only
    // thing he can see.
    const stand_in = document.createElement('div');
    stand_in.className = 'stat';
    stand_in.textContent = 'popped out - close the floating window to bring it back';
    try {
      const pip = await window.documentPictureInPicture.requestWindow({width: 520, height: 420});
      for (const sheet of document.querySelectorAll('link[rel="stylesheet"], style')) {
        pip.document.head.append(sheet.cloneNode(true));
      }
      pip.document.body.style.margin = '0';
      poppedOutDocs.add(pip.document);
      // nothing on the page should hold focus while he goes back to the game
      if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
      home.append(stand_in);
      pip.document.body.append(panel);
      panel.classList.remove('hidden');
      btn.classList.add('on');
      if (redraw) redraw();
      pip.addEventListener('pagehide', () => {
        poppedOutDocs.delete(pip.document);
        stand_in.remove();
        home.append(panel);
        btn.classList.remove('on');
        if (redraw) redraw();
      });
    } catch (err) {
      stand_in.remove();
      if (status) status.textContent = 'pop-out refused: ' + err;
    }
  };
}

popOutPanel('navigation', 'nav-popout', 'nav-status', () => navDraw());
// the camera poll is started by camEnter and nothing stops it on leaving the tab, so the floating
// panel keeps updating while another tab is open in the main window - which is the whole point
popOutPanel('camera', 'cam-popout', 'cam-status', null);
popOutPanel('control', 'control-popout', 'control-status', () => controlLoad().catch(() => {}));

// ---- control tab: OverlayModel's state, ported from OverlayWindow.draw() -------------------
// SAME STRUCTURE AS THE TKINTER OVERLAY, deliberately: keyboard at real qwerty positions, mouse
// click/scroll lamps, a heading dial and a mouse vector, run state and a note - just drawn as one
// ui_base panel instead of a canvas widget, and read from /api/control instead of driven directly
// by a controller. see tools/review/control.py for where the numbers actually come from: a sink
// file wt-overlay-shadow writes, never this server capturing anything itself.
const CONTROL_POLL_MS = 100;  // ~10 Hz - the same rate OverlayModel.tick() is driven at
let controlTimer = null;

function controlToken(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

// ONE RENDERER, THREE PANELS. the control tab draws the live sink, and the sessions tab draws a
// recording's and the humaniser's; each panel's elements share a prefix (control-, srec-, ssim-)
const sinkBuilt = new Set();

function sinkBuildKeys(prefix, keys) {
  const size = 26, gap = 4, rowH = 32, unit = size + gap;
  const el = byId(prefix + '-keys');
  if (!el || !keys.length) return;
  const maxRow = Math.max(...keys.map(k => k.row));
  // #control-keys holds only absolutely-positioned children, so it has zero intrinsic size - an
  // explicit width/height here is what keeps the dials from drifting left over the keyboard
  const maxRight = Math.max(...keys.map(k => k.col * unit + k.width_units * size));
  el.style.height = ((maxRow + 1) * rowH) + 'px';
  el.style.width = (maxRight + gap) + 'px';
  el.innerHTML = keys.map(k => {
    const x = k.col * unit;
    const y = k.row * rowH;
    const w = k.width_units * size + (k.width_units - 1) * gap;
    return `<div class="control-key" data-key="${k.key}" `
      + `style="left:${x}px;top:${y}px;width:${w}px;height:${size}px">${k.key.toUpperCase()}</div>`;
  }).join('');
  sinkBuilt.add(prefix);
}

function sinkUpdateKeys(prefix, keys) {
  const el = byId(prefix + '-keys');
  if (!el) return;
  for (const k of keys) {
    const lamp = el.querySelector(`[data-key="${k.key}"]`);
    if (lamp) lamp.classList.toggle('lit', k.lit);
  }
}

const CONTROL_MOUSE_LABELS = {left: 'L', right: 'R', scroll_up: '▲', scroll_down: '▼'};

function sinkBuildMouseButtons(prefix) {
  const el = byId(prefix + '-mouse-buttons');
  if (!el) return;
  el.innerHTML = Object.entries(CONTROL_MOUSE_LABELS).map(([name, label]) =>
    `<div class="control-key" data-mouse="${name}">${label}</div>`).join('');
}

function sinkUpdateMouseButtons(prefix, buttons) {
  const el = byId(prefix + '-mouse-buttons');
  if (!el || !buttons) return;
  for (const [name, lit] of Object.entries(buttons)) {
    const lamp = el.querySelector(`[data-mouse="${name}"]`);
    if (lamp) lamp.classList.toggle('lit', lit);
  }
}

// PRIMITIVE PIXEL LINES, LIKE THE LOADING BAR - flat strokes, no gradient, no arrowhead. the only
// thing that varies is the mouse vector's line width, which IS the magnitude reading rather than
// a separate label; the dial's dot-on-the-rim convention is ported straight from
// OverlayWindow._draw_dial (a dot's position reads as a bearing at a glance, a centre line takes
// a moment longer to parse).
function sinkDrawCanvas(prefix, snap) {
  const canvas = byId(prefix + '-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const r = 60;
  const dialCx = 75, dialCy = 85;
  const mouseCx = 225, mouseCy = 85;
  const border = controlToken('--grey-border', '#4a4a52');
  const lit = controlToken('--lichen', '#c7ed5f');
  const warn = controlToken('--stone-red-lift', '#ad796f');
  const dim = controlToken('--text-dim', '#8b8b90');

  ctx.strokeStyle = border;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(dialCx, dialCy, r, 0, Math.PI * 2);
  ctx.stroke();

  const facing = snap.facing_degrees;
  if (facing === null || facing === undefined) {
    ctx.fillStyle = dim;
    ctx.font = '11px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('no fix', dialCx, dialCy + 4);
  } else {
    ctx.fillStyle = lit;
    ctx.beginPath();
    ctx.arc(dialCx, dialCy - r, 4, 0, Math.PI * 2);  // current facing is always straight up
    ctx.fill();
    const delta = snap.turn_delta_degrees;
    if (delta !== null && delta !== undefined) {
      const a = delta * Math.PI / 180;
      const dx = -Math.sin(a), dy = -Math.cos(a);
      ctx.fillStyle = warn;
      ctx.beginPath();
      ctx.arc(dialCx + dx * r, dialCy + dy * r, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  ctx.strokeStyle = border;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(mouseCx, mouseCy, r, 0, Math.PI * 2);
  ctx.stroke();

  const dx = snap.mouse_dx || 0, dy = snap.mouse_dy || 0;
  const magnitude = Math.hypot(dx, dy);
  if (magnitude > 0) {
    const maxCounts = snap.mouse_vector_max_counts || 200;
    const clamped = Math.min(1, magnitude / maxCounts);
    const scale = clamped * r / magnitude;
    ctx.strokeStyle = lit;
    ctx.lineWidth = 2 + clamped * 8;  // thickness IS the magnitude, nothing else draws it
    ctx.lineCap = 'butt';
    ctx.beginPath();
    ctx.moveTo(mouseCx, mouseCy);
    ctx.lineTo(mouseCx + dx * scale, mouseCy + dy * scale);
    ctx.stroke();
  }
}

function sinkDraw(prefix, snap) {
  if (!sinkBuilt.has(prefix)) {
    sinkBuildKeys(prefix, snap.keys || []);
    sinkBuildMouseButtons(prefix);
  }
  sinkUpdateKeys(prefix, snap.keys || []);
  sinkUpdateMouseButtons(prefix, snap.mouse_buttons);
  sinkDrawCanvas(prefix, snap);
}

async function controlLoad() {
  const snap = await api('/api/control');
  sinkDraw('control', snap);
  byId('control-run').textContent = snap.run_state || '-';
  byId('control-mode').textContent = snap.mode || '';
  byId('control-note').textContent = snap.note || '';
  const status = byId('control-status');
  if (status) status.textContent = snap.live ? 'live' : 'no live feed';
}

function controlEnter() {
  if (controlTimer) clearInterval(controlTimer);
  controlTimer = setInterval(() => controlLoad().catch(() => {}), CONTROL_POLL_MS);
  controlLoad().catch(() => {});
}

// ---- sessions tab: a recording played back beside the output sink ---------------------------
// THE PAGE OWNS THE CLOCK. elapsed advances by real time times the chosen speed, and every
// moment is asked of /api/playback-frame, which is stateless - so a scrub and a tick can never
// disagree about where playback is. frames change at the recording's own rate (2 Hz by default);
// state, keys and the sink are asked for every SESSION_POLL_MS
const SESSION_SPEEDS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2];
const SESSION_POLL_MS = 100;
const SESSION_MODE_COLOURS = {
  ROAM: ['--text-dim', '#8b8b90'], ACQUIRE: ['--cream', '#e9e1cf'], TRAVEL: ['--lichen', '#c7ed5f'],
  ENGAGE: ['--stone-red-lift', '#ad796f'], LOOT: [null, '#d9b35b'], REST: [null, '#6f8fbf'],
  FLEE: [null, '#c0504d'],
};
let sessionOpen = null;       // what /api/playback-open said about the recording
let sessionMoment = null;     // the last /api/playback-frame answer
let sessionElapsed = 0;
let sessionSpeed = 1;
let sessionPlaying = false;
let sessionLastTick = null;
let sessionFetchedAt = 0;
let sessionFrameName = null;
let sessionInFlight = false;

function sessionFmt(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

function sessionModeColour(mode) {
  const [token, fallback] = SESSION_MODE_COLOURS[mode] || [null, '#777'];
  return token ? controlToken(token, fallback) : fallback;
}

function sessionClock() {
  const total = sessionOpen ? sessionOpen.duration : 0;
  byId('session-clock').textContent = `${sessionFmt(sessionElapsed)} / ${sessionFmt(total)}`;
  byId('session-slider').value = sessionElapsed;
}

function sessionDrawStrip() {
  const canvas = byId('session-strip');
  if (!canvas || !sessionOpen) return;
  canvas.width = canvas.clientWidth || 600;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const total = sessionOpen.duration || 1;
  const seen = [];
  for (const run of sessionOpen.timeline || []) {
    const x0 = run.start / total * canvas.width;
    const x1 = Math.max(x0 + 1, run.end / total * canvas.width);
    ctx.fillStyle = sessionModeColour(run.mode);
    ctx.fillRect(x0, 0, x1 - x0, canvas.height);
    if (!seen.includes(run.mode)) seen.push(run.mode);
  }
  byId('session-legend').innerHTML = seen.map(mode =>
    `<span style="color:${sessionModeColour(mode)}">&#9632;</span> ${mode}`).join('&nbsp;&nbsp;');
}

function sessionFacts(moment) {
  const s = moment.state || {};
  const num = (v, digits = 1) => (v === null || v === undefined) ? '-' : Number(v).toFixed(digits);
  const heading = s.facing === null || s.facing === undefined ? '-'
    : (s.facing * 180 / Math.PI).toFixed(1) + ' deg';
  const facts = [
    ['x, y', `${num(s.x, 3)}, ${num(s.y, 3)}`], ['facing', heading], ['zone', s.map_id ?? '-'],
    ['health', num(s.health, 0)], ['resource', num(s.resource, 0)],
    ['target', s.target ? `${s.target_hostility || 'yes'} ${num(s.target_health, 0)}` : 'none'],
    ['combat', s.in_combat ? 'yes' : 'no'], ['cast', s.cast || '-'],
    ['mode machine', moment.machine_mode || '-'], ['labelled', moment.mode || 'unlabelled'],
    ['plates', moment.frame ? moment.frame.plates.length : '-'],
  ];
  byId('session-facts').innerHTML = facts.map(([k, v]) =>
    `<span class="k">${k}</span><span class="v">${v}</span>`).join('');
}

// the plate rows' boxes are in the saved frame's own pixels; the arrow starts at the character,
// projected through the fitted camera, and stops short of the plate like wt-bearing-overlay's
function sessionDrawOverlay() {
  const img = byId('session-image');
  const canvas = byId('session-overlay');
  if (!img || !canvas || !img.naturalWidth) return;
  canvas.width = img.clientWidth;
  canvas.height = img.clientHeight;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const frame = sessionMoment && sessionMoment.frame;
  if (!frame || !frame.plates.length) return;
  const k = canvas.width / (frame.width || img.naturalWidth);
  const lit = controlToken('--lichen', '#c7ed5f');
  const ink = controlToken('--cream', '#e9e1cf');
  ctx.lineWidth = 2;
  ctx.font = '11px monospace';
  for (const p of frame.plates) {
    const [l, t, w, h] = p.box.map(v => v * k);
    ctx.strokeStyle = lit;
    ctx.strokeRect(l, t, w, h);
    if (frame.start) {
      const sx = frame.start[0] * k, sy = frame.start[1] * k;
      const cx = l + w / 2, cy = t + h / 2;
      const len = Math.hypot(cx - sx, cy - sy) || 1;
      const ux = (cx - sx) / len, uy = (cy - sy) / len;
      const edge = Math.min(w / 2 / Math.max(Math.abs(ux), 1e-6), h / 2 / Math.max(Math.abs(uy), 1e-6));
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(cx - ux * (edge + 4), cy - uy * (edge + 4));
      ctx.stroke();
    }
    ctx.fillStyle = ink;
    const card = [`${p.d.toFixed(1)} yd`, `r ${p.r >= 0 ? '+' : ''}${p.r.toFixed(0)}`,
      `b ${p.b.toFixed(0)}`, `f ${p.f.toFixed(0)}`];
    card.forEach((line, i) => ctx.fillText(line, l, t + h + 12 + i * 12));
  }
}

// the humaniser's panel stays dark until a policy can say what it would press
function sessionDrawSimulated(sink) {
  sinkDraw('ssim', {
    ...sink,
    keys: (sink.keys || []).map(key => ({...key, lit: false})),
    mouse_buttons: Object.fromEntries(Object.keys(sink.mouse_buttons || {}).map(b => [b, false])),
    facing_degrees: null, turn_delta_degrees: null, mouse_dx: 0, mouse_dy: 0,
  });
}

async function sessionShow() {
  if (!sessionOpen || sessionInFlight) return;
  sessionInFlight = true;
  try {
    const moment = await api('/api/playback-frame?t=' + sessionElapsed.toFixed(3));
    if (moment.error) return;
    sessionMoment = moment;
    const frame = moment.frame;
    const img = byId('session-image');
    if (frame && frame.path !== sessionFrameName) {
      sessionFrameName = frame.path;
      img.onload = () => sessionDrawOverlay();
      img.src = '/session-frame/' + encodeURIComponent(frame.path);
    } else {
      sessionDrawOverlay();
    }
    sessionFacts(moment);
    sinkDraw('srec', moment.sink);
    sessionDrawSimulated(moment.sink);
    byId('srec-mode').textContent = moment.machine_mode || '';
    byId('srec-note').textContent = moment.sink.note || '';
  } finally {
    sessionInFlight = false;
  }
}

function sessionTick(now) {
  if (!sessionPlaying || !sessionOpen) return;
  if (sessionLastTick !== null) sessionElapsed += (now - sessionLastTick) / 1000 * sessionSpeed;
  sessionLastTick = now;
  if (sessionElapsed >= sessionOpen.duration) {
    sessionElapsed = sessionOpen.duration;
    sessionPause();
  }
  sessionClock();
  if (now - sessionFetchedAt >= SESSION_POLL_MS) {
    sessionFetchedAt = now;
    sessionShow().catch(() => {});
  }
  if (sessionPlaying) requestAnimationFrame(sessionTick);
}

function sessionPlay() {
  if (!sessionOpen) return;
  if (sessionElapsed >= sessionOpen.duration) sessionElapsed = 0;
  sessionPlaying = true;
  sessionLastTick = null;
  byId('session-play').textContent = 'pause';
  requestAnimationFrame(sessionTick);
}

function sessionPause() {
  sessionPlaying = false;
  byId('session-play').textContent = 'play';
}

async function sessionLoad(name) {
  sessionPause();
  byId('session-status').textContent = `opening ${name}...`;
  const data = await api('/api/playback-open', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({name}),
  });
  if (data.error) {
    byId('session-status').textContent = data.error;
    return;
  }
  sessionOpen = data;
  sessionElapsed = 0;
  sessionFrameName = null;
  byId('session-open').innerHTML = `<span>${data.name}</span>`;
  byId('session-slider').max = data.duration;
  byId('session-status').textContent =
    `${data.frames} frames, ${data.plate_frames} with plates, ${data.arrow_frames} with arrows, ` +
    `${data.labelled}/${data.windows} windows labelled`;
  sessionDrawStrip();
  sessionClock();
  await sessionShow();
}

function sessionsEnter() {
  if (!sessionOpen) byId('session-status').textContent = 'pick a recording';
  else sessionDrawStrip();
}

byId('session-open').onclick = async evt => {
  const head = evt.currentTarget;
  const data = await api('/api/playback-sessions');
  const items = (data.sessions || []).map(s => ({id: s.name, label: s.name}));
  listMenu('recordings', items, item => sessionLoad(item.id).catch(err => {
    byId('session-status').textContent = String(err);
  })).openAt(head);
};

byId('session-play').onclick = () => (sessionPlaying ? sessionPause() : sessionPlay());

byId('session-speed').onclick = evt => {
  const items = SESSION_SPEEDS.map(s => ({id: String(s), label: `${s}x`}));
  listMenu('speed', items, item => {
    sessionSpeed = Number(item.id);
    byId('session-speed').innerHTML = `<span>${item.label}</span>`;
  }).openAt(evt.currentTarget);
};

byId('session-slider').oninput = () => {
  sessionPause();
  sessionElapsed = Number(byId('session-slider').value);
  sessionClock();
  sessionShow().catch(() => {});
};

window.addEventListener('resize', () => { sessionDrawStrip(); sessionDrawOverlay(); });

// ---- nav lines: read live, corrected after ---------------------------------------------------
// A LINE IS DERIVED, WHICH IS WHY SAVING IS A SIDECAR. walked_maps() builds the lines out of the
// `nav` rows the F13-F15 hotkeys wrote; nothing stores them and the recording must never be
// rewritten, so an edit lands in nav_edits.json beside the session and is laid over the derived
// walk on read. Balthazar Fitzpatrick: "nothing writes the lines, the lines are just read from
// recordings".
let navSelected = null;   // [lineIndex, pointIndex] of the row being looked at

function navLines() {
  const map = navCurrent();
  return map && map.lines ? map.lines : [];
}

function navRenderTable() {
  const body = byId('nav-rows');
  if (!body) return;
  const lines = navLines();
  body.innerHTML = '';
  lines.forEach((line, li) => {
    (line.points.length ? line.points : [null]).forEach((point, pi) => {
      const tr = document.createElement('tr');
      const chosen = navSelected && navSelected[0] === li && navSelected[1] === pi;
      tr.className = chosen ? 'nav-row on' : 'nav-row';
      // the line id and its type are properties of the LINE, so they are only offered on its
      // first row - repeating them per point invites editing one copy of three
      tr.innerHTML =
        `<td>${pi === 0 ? line.id : ''}</td>` +
        `<td>${pi === 0 ? `<span class="nav-kind ${line.kind}">${line.kind}</span>` : ''}</td>` +
        `<td>${point ? pi : '<span class="nav-empty">no points</span>'}</td>` +
        `<td>${point ? point[0].toFixed(3) : ''}</td>` +
        `<td>${point ? point[1].toFixed(3) : ''}</td>` +
        `<td>${point ? '<span class="nav-drop">x</span>' : ''}</td>`;
      tr.onclick = (evt) => {
        if (evt.target.classList.contains('nav-drop')) {
          line.points.splice(pi, 1);
          line.closed = line.points.length >= 3;
          navSelected = null;
        } else if (evt.target.classList.contains('nav-kind')) {
          line.kind = line.kind === 'go' ? 'nogo' : 'go';
        } else if (point) {
          navSelected = [li, pi];
        }
        navRenderTable();
        navDraw();
      };
      body.appendChild(tr);
    });
  });
  const status = byId('nav-hint');
  if (status) {
    status.textContent = navSelected
      ? `line ${lines[navSelected[0]].id}, point ${navSelected[1]}`
      : 'click a row to see it on the zone';
  }
}

byId('nav-add').onclick = () => {
  // A NEW POINT NEEDS A LINE, and a session with no nav rows has none - so the first press makes
  // line 0 rather than refusing, which is what "type coordinates in by hand" has to mean
  const map = navCurrent();
  if (!map) return;
  map.lines = map.lines || [];
  if (!map.lines.length) map.lines.push({id: 0, kind: 'go', points: [], closed: false});
  const line = map.lines[navSelected ? navSelected[0] : map.lines.length - 1];
  const last = line.points[line.points.length - 1];
  line.points.push(last ? [last[0], last[1]] : [50, 50]);
  line.closed = line.points.length >= 3;
  navSelected = [map.lines.indexOf(line), line.points.length - 1];
  navRenderTable();
  navDraw();
};

byId('nav-save').onclick = async () => {
  const map = navCurrent();
  const status = byId('nav-status');
  if (!map) return;
  try {
    await api('/api/nav-edit', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        session: navSession || (byId('nav-open').textContent || '').trim(),
        map_id: map.map_id,
        lines: (map.lines || []).map(l => ({id: l.id, kind: l.kind, points: l.points})),
      }),
    });
    status.textContent = `saved ${(map.lines || []).length} line(s) beside the recording`;
  } catch (err) {
    status.textContent = `could not save: ${err}`;
  }
};

// ---- nav hotkeys: "change", then press the combo -----------------------------------------------
// The recorder reads profiles/nav_keys.json at startup and an explicit --nav-* flag still wins, so
// this sets a default rather than overriding a deliberate one. A binding is nav_bindings' string:
// modifiers in the order ctrl, alt, shift, cmd, then one trigger - "alt+scroll_up", "button6", "f15".
//
// TWO LISTENERS WAIT AT ONCE. The browser catches keys, mouse buttons 0-4 and the wheel; the server
// runs the recorder's own listener (/api/nav-keys/listen) for a button the browser never receives,
// which a remapped g502 dpi button may be. Whichever answers first is saved.
//
// BROWSER -> RECORDER NAMES, and they must agree or a binding never matches:
//   MouseEvent.button 0 / 1 / 2       left / middle / right   (pynput's names)
//   MouseEvent.button n >= 3          button{n+1}: 3 is back = button4, 4 is forward = button5
//   WheelEvent.deltaY < 0 / > 0       scroll_up / scroll_down (pynput dy > 0 is up)
//   keys                              timing.physical_key: lower case, unshifted, "space", "f13"
const NAV_KEY_ACTIONS = ['log_point', 'new_line', 'go_zone', 'nogo_zone'];
const NAV_MOD_ORDER = ['ctrl', 'alt', 'shift', 'cmd'];
const NAV_MOD_LABELS = {ctrl: 'Ctrl', alt: 'Alt', shift: 'Shift', cmd: 'Cmd'};
let navKeyCapturing = null;
let navKeyCaptureId = 0;
let navKeys = {};
let navDpiButton = 'mouse_button6';

function navKeyName(evt) {
  // the physical key rather than the character shift produces - shift+2 is "2", not "@"
  const digit = /^Digit(\d)$/.exec(evt.code || '');
  if (digit) return digit[1];
  if (evt.key === ' ') return 'space';
  if (evt.key === 'Escape') return 'esc';
  if (evt.key.length === 1) return evt.key.toLowerCase();
  return evt.key.toLowerCase().replace('arrow', '').replace('control', 'ctrl');
}

function navHeldMods(evt) {
  const held = {ctrl: evt.ctrlKey, alt: evt.altKey, shift: evt.shiftKey, cmd: evt.metaKey};
  return NAV_MOD_ORDER.filter(m => held[m]);
}

function navButtonName(button) {
  // explicit, like the recorder's triggers: 'left' alone is the arrow key
  return 'mouse_' + (({0: 'left', 1: 'middle', 2: 'right'})[button] || `button${button + 1}`);
}

function navTriggerLabel(trigger) {
  if (trigger === navDpiButton) return 'DPI button';
  if (trigger === 'scroll_up') return 'scroll up';
  if (trigger === 'scroll_down') return 'scroll down';
  if (['mouse_left', 'mouse_right', 'mouse_middle'].includes(trigger)) return `${trigger.slice(6)} click`;
  if (['left', 'right', 'up', 'down'].includes(trigger)) return `${trigger} arrow`;
  const extra = /^mouse_button(\d+)$/.exec(trigger);
  if (extra) return ({4: 'back button', 5: 'forward button'})[extra[1]] || `mouse button ${extra[1]}`;
  return /^f\d+$/.test(trigger) || trigger.length === 1 ? trigger.toUpperCase() : trigger;
}

function navComboLabel(combo) {
  if (!combo) return 'unbound';
  const plusKey = combo === '+' || combo.endsWith('++');
  const parts = (plusKey ? combo.slice(0, -1) : combo).split('+').filter(Boolean);
  const trigger = plusKey ? '+' : parts.pop();
  return [...parts.map(m => NAV_MOD_LABELS[m] || m), navTriggerLabel(trigger)].join(' + ');
}

function navKeysPaint() {
  NAV_KEY_ACTIONS.forEach(action => {
    const el = document.getElementById(`navkey-${action}`);
    const waiting = navKeyCapturing === action;
    if (el) {
      el.textContent = waiting ? 'press the combo now - esc cancels' : navComboLabel(navKeys[action]);
      el.classList.toggle('waiting', waiting);
    }
  });
  document.querySelectorAll('.nav-key-change').forEach(el => {
    el.classList.toggle('on', el.dataset.action === navKeyCapturing);
  });
  const note = byId('nav-keys-note');
  if (!note) return;
  const combos = NAV_KEY_ACTIONS.map(a => navKeys[a]).filter(Boolean);
  // a plain key at or below f12 can be bound in wow, so a mark would fire an ability too
  const risky = combos.filter(c => !c.includes('+') && !/^(f(1[3-9]|2[0-4])|button\d+|middle)$/.test(c));
  const notes = [];
  if (risky.length) notes.push(`${risky.map(navComboLabel).join(', ')} can also be bound in game - unbind in wow, or a mark fires an ability too`);
  if (combos.some(c => c.split('+').pop() === navDpiButton)) {
    notes.push(`DPI button = ${navDpiButton}, unverified: a G502 sends it only if G HUB maps it to a mouse button - press change and push it to see what arrives`);
  }
  note.textContent = notes.join('. ');
}

async function navKeysLoad() {
  const data = await api('/api/nav-keys');
  navKeys = data.keys || {};
  navDpiButton = data.dpi_button || navDpiButton;
  navKeysPaint();
}

function navKeysStop() {
  navKeyCapturing = null;
  navKeyCaptureId += 1;
  api('/api/nav-keys/listen', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({cancel: true}),
  }).catch(() => {});
  navKeysPaint();
}

async function navKeysSave(combo) {
  const action = navKeyCapturing;
  if (!action) return;
  navKeysStop();
  const status = byId('nav-status');
  try {
    const answer = await api('/api/nav-keys', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({[action]: combo}),
    });
    navKeys = answer.keys || navKeys;
    if (status) status.textContent = `${action.replace('_', ' ')} -> ${navComboLabel(combo)}`;
  } catch (err) {
    if (status) status.textContent = `could not save the binding: ${err.message}`;
  }
  navKeysPaint();
}

async function navKeysListen(id) {
  // the recorder's listener, re-armed until the browser or it catches something
  while (navKeyCaptureId === id) {
    let answer;
    try {
      answer = await api('/api/nav-keys/listen', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({seconds: 10}),
      });
    } catch (err) {
      return;
    }
    if (navKeyCaptureId !== id) return;
    if (answer.combo) { navKeysSave(answer.combo); return; }
    if (answer.reason === 'cancelled') { navKeysStop(); return; }
    if (answer.reason !== 'timeout') return;
  }
}

document.querySelectorAll('.nav-key-change').forEach(el => {
  el.onclick = () => {
    if (navKeyCapturing === el.dataset.action) { navKeysStop(); return; }
    navKeyCaptureId += 1;
    navKeyCapturing = el.dataset.action;
    navKeysPaint();
    navKeysListen(navKeyCaptureId);
  };
});

function navCapture(evt, trigger) {
  evt.preventDefault();
  evt.stopPropagation();
  navKeysSave([...navHeldMods(evt), trigger].join('+'));
}

document.addEventListener('keydown', (evt) => {
  if (!navKeyCapturing) return;
  // a modifier alone is a chord half, not a binding
  if (['shift', 'control', 'alt', 'meta'].includes(evt.key.toLowerCase())) return;
  if (evt.key === 'Escape' && !navHeldMods(evt).length) {
    evt.preventDefault();
    navKeysStop();
    return;
  }
  navCapture(evt, navKeyName(evt));
}, true);

document.addEventListener('mousedown', (evt) => {
  if (!navKeyCapturing) return;
  // a plain left or right click stays a click, so the page can still be used while choosing
  if ((evt.button === 0 || evt.button === 2) && !navHeldMods(evt).length) return;
  navCapture(evt, navButtonName(evt.button));
}, true);

// back / forward would otherwise navigate the page on release
['mouseup', 'auxclick', 'contextmenu'].forEach(type => document.addEventListener(type, (evt) => {
  if (navKeyCapturing && evt.button >= 3) evt.preventDefault();
}, true));

document.addEventListener('wheel', (evt) => {
  if (!navKeyCapturing || !evt.deltaY) return;
  navCapture(evt, evt.deltaY < 0 ? 'scroll_up' : 'scroll_down');
}, {capture: true, passive: false});
