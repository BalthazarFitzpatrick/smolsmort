// shared helpers for every tab: json api calls, a modal question, a status line, and the tab
// registry that builds the nav bar from data so a host can add tabs

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
  };
  const res = await fetch(path, options);
  if (!res.ok) {
    // the server answers a bad request as {error: ...}; show that text, not the raw body
    const text = await res.text();
    let message = text;
    try { message = JSON.parse(text).error || text; } catch (err) { /* not json */ }
    throw new Error(message);
  }
  return res.json();
}

const setText = (id, text) => { document.getElementById(id).textContent = text; };

// a point just under an element, so a menu opened from a menu is not read as a toggle-shut
function headPoint(anchorId) {
  const box = document.getElementById(anchorId).getBoundingClientRect();
  return {x: box.left, y: box.bottom + 6};
}

// a menu that fails to build says why instead of a button that looks dead
async function openPicker(anchorId, build, options = {}) {
  const head = document.getElementById(anchorId);
  try {
    const menu = new Menu({
      title: options.title || '', persistent: options.persistent !== false, sections: await build(),
    });
    menu.openAt(head);
    return menu;
  } catch (err) {
    const reason = (err && err.message) || String(err);
    if (options.status) setText(options.status, reason);
    new Menu({title: 'could not open', sections: [{kind: 'list', empty: reason, items: []}]})
      .openAt(head);
    return null;
  }
}

// a centred question with named answers; resolves to the pressed button's id, or 'cancel'
function askDialog({title, lines, buttons}) {
  return new Promise(resolve => {
    const backdrop = document.createElement('div');
    backdrop.className = 'modal-backdrop';
    const dialog = document.createElement('div');
    dialog.className = 'panel-floating modal-dialog';
    dialog.id = 'ask-dialog';
    dialog.innerHTML = `<div class="popup-title"></div><div class="modal-body"></div>`
      + '<div class="run-controls modal-buttons"></div>';
    dialog.querySelector('.popup-title').textContent = title;
    const body = dialog.querySelector('.modal-body');
    lines.forEach(line => {
      const row = document.createElement('div');
      row.textContent = line;
      body.appendChild(row);
    });
    const done = answer => { backdrop.remove(); resolve(answer); };
    buttons.forEach(button => {
      const el = document.createElement('div');
      el.className = `toggle ${button.tone || ''}`;
      el.id = `ask-${button.id}`;
      el.textContent = button.label;
      el.onclick = () => done(button.id);
      dialog.querySelector('.modal-buttons').appendChild(el);
    });
    backdrop.appendChild(dialog);
    document.body.appendChild(backdrop);
  });
}

// ---- the tab registry: the nav bar and panels are built from this list ----
// a host adds a tab with window.smolsmortTabs.register({id, label, mount(panelEl)}); mount runs
// once and may return {enter()} to be told each time the tab is shown
const smolsmortTabs = (() => {
  const tabs = [];
  let started = false;

  function build(tab) {
    const nav = document.getElementById('nav-bar');
    const head = document.createElement('div');
    head.className = 'nav-tab';
    head.dataset.tab = tab.id;
    head.setAttribute('role', 'tab');
    head.textContent = tab.label;
    nav.appendChild(head);
    const panel = document.createElement('div');
    panel.className = 'tab-panel hidden';
    panel.dataset.panel = tab.id;
    document.getElementById('tab-panels').appendChild(panel);
    tab.handle = tab.mount(panel) || {};
  }

  const enter = name => {
    const tab = tabs.find(t => t.id === name);
    if (tab && tab.handle.enter) tab.handle.enter();
  };

  function start() {
    started = true;
    tabs.forEach(build);
    initShell({onEnter: enter, fallback: tabs.length ? tabs[0].id : ''});
  }

  function register(tab) {
    if (!tab || !tab.id || !tab.label || typeof tab.mount !== 'function') {
      throw new Error('a tab needs id, label and mount(panelEl)');
    }
    if (tabs.some(t => t.id === tab.id)) throw new Error(`tab ${tab.id} is already registered`);
    tabs.push({...tab});
    if (started) {
      build(tabs[tabs.length - 1]);
      initShell({onEnter: enter, fallback: activeTab});
    }
  }

  return {register, start, list: () => tabs.map(t => t.id)};
})();

window.smolsmortTabs = smolsmortTabs;
