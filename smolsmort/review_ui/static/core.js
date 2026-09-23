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
// register({id, label, mount(panelEl), topic}); mount may return {enter()}. topic defaults to
// 'vision' so older hosts keep working; each topic remembers its tab and its flavour dropdown,
// and the topic switch stays hidden until a second topic has tabs
const smolsmortTabs = (() => {
  const tabs = [];
  const topicOrder = [];
  const topicLabels = {};
  const flavours = {};
  let started = false;
  let activeTopic = null;

  function storageGet(key) {
    try { return localStorage.getItem(key); } catch (err) { return null; }
  }
  function storageSet(key, value) {
    try { localStorage.setItem(key, value); } catch (err) { /* private window, no matter */ }
  }

  function defineTopic({id, label}) {
    if (!(id in topicLabels)) topicOrder.push(id);
    topicLabels[id] = label;
  }
  defineTopic({id: 'vision', label: 'vision'});
  defineTopic({id: 'regression', label: 'regression'});
  defineTopic({id: 'classification', label: 'classification'});

  const topicsWithTabs = () => topicOrder.filter(id => tabs.some(t => t.topic === id));

  function build(tab) {
    const head = document.createElement('div');
    head.className = 'nav-tab';
    head.dataset.tab = tab.id;
    head.setAttribute('role', 'tab');
    head.textContent = tab.label;
    tab.navEl = head;
    const panel = document.createElement('div');
    panel.className = 'tab-panel hidden';
    panel.dataset.panel = tab.id;
    document.getElementById('tab-panels').appendChild(panel);
    tab.handle = tab.mount(panel) || {};
  }

  const enter = name => {
    const tab = tabs.find(t => t.id === name);
    if (!tab) return;
    storageSet(`smolsmort:tab:${tab.topic}`, name);
    if (tab.handle.enter) tab.handle.enter();
  };

  // only the active topic's tab heads sit in #nav-bar, so the shell's own tab machinery (click,
  // arrow keys, its own remembered-tab key) only ever sees this topic's tabs
  function paintNav() {
    const nav = document.getElementById('nav-bar');
    const inTopic = tabs.filter(t => t.topic === activeTopic);
    nav.replaceChildren(...inTopic.map(t => t.navEl));
    const remembered = storageGet(`smolsmort:tab:${activeTopic}`);
    const ids = inTopic.map(t => t.id);
    const fallback = ids.includes(remembered) ? remembered : (ids[0] || '');
    initShell({onEnter: enter, fallback});
  }

  function paintTopicSwitch() {
    const bar = document.getElementById('topic-switch');
    if (!bar) return;
    const list = topicsWithTabs();
    bar.classList.toggle('hidden', list.length < 2);
    bar.replaceChildren(...list.map(id => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `toggle topic-tab${id === activeTopic ? ' on' : ''}`;
      btn.textContent = topicLabels[id];
      btn.dataset.topic = id;
      btn.setAttribute('role', 'tab');
      btn.setAttribute('aria-selected', id === activeTopic ? 'true' : 'false');
      btn.tabIndex = id === activeTopic ? 0 : -1;
      btn.onclick = () => switchTopic(id);
      // left/right move focus and switch, matching the tab row's own arrow-key behaviour
      btn.onkeydown = evt => {
        const buttons = [...bar.children];
        const i = buttons.indexOf(btn);
        let next = null;
        if (evt.key === 'ArrowRight') next = buttons[(i + 1) % buttons.length];
        else if (evt.key === 'ArrowLeft') next = buttons[(i - 1 + buttons.length) % buttons.length];
        if (next) { evt.preventDefault(); next.focus(); switchTopic(next.dataset.topic); }
      };
      return btn;
    }));
  }

  function paintFlavourRow() {
    const select = document.getElementById('topic-flavour');
    if (!select) return;
    const state = flavours[activeTopic];
    const has = !!(state && state.items.length);
    select.classList.toggle('hidden', !has);
    if (!has) return;
    select.replaceChildren(...state.items.map(item => {
      const opt = document.createElement('option');
      opt.value = item.id;
      opt.textContent = item.label;
      return opt;
    }));
    select.value = state.selected;
  }

  function switchTopic(id) {
    if (id === activeTopic || !topicsWithTabs().includes(id)) return;
    activeTopic = id;
    storageSet('smolsmort:topic', id);
    paintTopicSwitch();
    paintNav();
    paintFlavourRow();
  }

  function initialTopic() {
    const remembered = storageGet('smolsmort:topic');
    const avail = topicsWithTabs();
    if (remembered && avail.includes(remembered)) return remembered;
    return avail[0] || topicOrder[0];
  }

  function start() {
    started = true;
    tabs.forEach(build);
    activeTopic = initialTopic();
    paintTopicSwitch();
    paintNav();
    paintFlavourRow();
    const select = document.getElementById('topic-flavour');
    if (select) select.onchange = () => pickFlavour(activeTopic, select.value);
  }

  function register(tab) {
    if (!tab || !tab.id || !tab.label || typeof tab.mount !== 'function') {
      throw new Error('a tab needs id, label and mount(panelEl)');
    }
    if (tabs.some(t => t.id === tab.id)) throw new Error(`tab ${tab.id} is already registered`);
    const topic = tab.topic || 'vision';
    defineTopic({id: topic, label: topicLabels[topic] || topic});
    tabs.push({...tab, topic});
    if (started) {
      build(tabs[tabs.length - 1]);
      paintTopicSwitch();
      if (topic === activeTopic) paintNav();
    }
  }

  // items: [{id, label}]. selectedId wins when it names one of them; otherwise the last flavour
  // remembered for this topic, otherwise the first item
  function setFlavours(topicId, items, selectedId) {
    const remembered = storageGet(`smolsmort:flavour:${topicId}`);
    const valid = id => items.some(item => item.id === id);
    const chosen = valid(selectedId) ? selectedId : (valid(remembered) ? remembered
      : (items[0] ? items[0].id : null));
    const listeners = (flavours[topicId] && flavours[topicId].listeners) || [];
    flavours[topicId] = {items, selected: chosen, listeners};
    if (started && topicId === activeTopic) paintFlavourRow();
  }

  function flavour(topicId) {
    return (flavours[topicId] && flavours[topicId].selected) || null;
  }

  function onFlavour(topicId, fn) {
    if (!flavours[topicId]) flavours[topicId] = {items: [], selected: null, listeners: []};
    flavours[topicId].listeners.push(fn);
  }

  function pickFlavour(topicId, id) {
    const state = flavours[topicId];
    if (!state || state.selected === id) return;
    state.selected = id;
    storageSet(`smolsmort:flavour:${topicId}`, id);
    state.listeners.forEach(fn => fn(id));
  }

  return {
    register, start, list: () => tabs.map(t => t.id),
    defineTopic, setFlavours, flavour, onFlavour,
  };
})();

window.smolsmortTabs = smolsmortTabs;
