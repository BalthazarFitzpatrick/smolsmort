// the base-path dialog: where each kind of data lives. the tile base may hold several folders

const EDITABLE_BASES = new Set(['tiles', 'pool']);
const asList = value => (Array.isArray(value) ? value : value ? [value] : []);
const baseChosen = {};
let baseRoot = '';

function shortPath(full) {
  return full.startsWith(baseRoot) ? (full.slice(baseRoot.length).replace(/^\//, '') || '.') : full;
}

function paintBases() {
  document.querySelectorAll('#settings-popup [data-base]').forEach(head => {
    const paths = asList(baseChosen[head.dataset.base]);
    head.innerHTML = '';
    if (!paths.length) head.appendChild(textSpan('', '-'));
    paths.forEach(full => head.appendChild(textSpan('base-path', shortPath(full))));
    head.title = paths.join('\n');
  });
}

// one folder per request, walked in place; "add" takes the folder being stood in
async function pickDirectory(type, anchor) {
  let browsing = null;
  let menu = null;
  const take = path => {
    baseChosen[type] = [...new Set([...asList(baseChosen[type]), path])];
    browsing = null;
    paintBases();
    refresh();
  };
  const build = async () => {
    const add = {kind: 'add', placeholder: 'click to browse for a folder', button: 'add', onAdd: take};
    if (browsing === null) {
      return [add, {
        kind: 'list', empty: 'no folder chosen yet', items: asList(baseChosen[type]).map(full => ({
          id: full, label: shortPath(full), title: full,
          action: {label: 'x', onPick: item => {
            baseChosen[type] = asList(baseChosen[type]).filter(p => p !== item.id);
            paintBases();
            refresh();
          }},
        })),
      }];
    }
    const data = await api(`/api/dir-tree?root=${type === 'pool' ? 'tiles' : type}&under=${encodeURIComponent(browsing)}`);
    browsing = data.here;
    const items = data.parent ? [{id: data.parent, label: '..'}] : [];
    data.entries.forEach(e => items.push({id: e.path, label: e.name + (e.has_children ? '/' : '')}));
    return [add, {
      kind: 'list', empty: 'nothing below this folder', items,
      onPick: item => { browsing = item.id; refresh(); },
    }];
  };
  const wireField = () => {
    const input = menu && menu.el.querySelector('.menu-add input');
    if (!input) return;
    input.value = browsing === null ? '' : browsing;
    input.onfocus = () => {
      if (browsing !== null) return;
      browsing = asList(baseChosen[type])[0] || baseRoot;
      refresh();
    };
  };
  const refresh = () => build().then(sections => { menu.refresh(sections); wireField(); });
  menu = await openPicker(anchor.id, build, {title: `${type} folder`});
  if (menu) wireField();
}

function closeSettings() {
  document.getElementById('settings-popup')?.parentElement.remove();
}

async function openSettings() {
  const data = await api('/api/dir-list');
  const bases = data.bases || {};
  baseRoot = bases.root || '';
  Object.keys(bases).forEach(type => { baseChosen[type] = asList(bases[type]); });
  const types = Object.keys(bases).filter(type => type !== 'root');
  const backdrop = document.createElement('div');
  backdrop.className = 'modal-backdrop';
  backdrop.innerHTML = `<div class="panel-floating modal-dialog" id="settings-popup">
    <div class="popup-title">where each kind of data lives</div>
    <div class="h-divider"></div>
    <div class="settings-field"><span class="field-label">base path</span>
      <input id="base-root" class="text-field" type="text" spellcheck="false" readonly></div>
    ${types.map(type => `<div class="settings-field"><span class="field-label">${type}</span>
      <div class="grow" id="base-${type}" data-base="${type}"><span>-</span></div></div>`).join('')}
    <div class="run-controls modal-buttons">
      <div class="toggle adds" id="settings-save">apply</div>
      <div class="toggle" id="settings-close">close</div>
    </div></div>`;
  document.body.appendChild(backdrop);
  document.getElementById('base-root').value = baseRoot;
  types.filter(type => EDITABLE_BASES.has(type)).forEach(type => {
    const head = document.getElementById(`base-${type}`);
    head.classList.add('toggle', 'dropdown-head');
    head.onclick = () => pickDirectory(type, head);
  });
  paintBases();
  document.getElementById('settings-close').onclick = closeSettings;
  document.getElementById('settings-save').onclick = async () => {
    // only the editable bases are sent
    const body = {};
    types.filter(type => EDITABLE_BASES.has(type)).forEach(type => { body[type] = asList(baseChosen[type]); });
    await api('/api/set-bases', body);
    closeSettings();
  };
}

document.getElementById('open-settings').onclick = () => openSettings();
