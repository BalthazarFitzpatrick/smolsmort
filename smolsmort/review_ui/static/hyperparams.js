// the hyperparams dropdown: one ui_base Menu, built fresh from /api/hyperparams and refreshed in
// place as the viewer picks a preset, an optimiser, a size or types a number - the same "state
// lives in JS, menu.refresh(sections) redraws" pattern the review tool's other dropdowns use.

let state = {
  optimizer: 'adamw', learning_rate: 3e-4, momentum: 0.9, weight_decay: 1e-4,
  size: 'medium', seed: 0, custom_channels: 24, custom_scale: 1.0,
};
let menu = null;
let options = null; // last /api/hyperparams response

let backendName = 'heatmap';
function backend() { return backendName; }

function applyPreset(preset) {
  state = {
    ...state,
    optimizer: preset.optimizer,
    learning_rate: preset.learning_rate,
    momentum: preset.momentum,
    weight_decay: preset.weight_decay,
    size: preset.size,
  };
  renderMenu();
}

function fieldSection(label, key, placeholder) {
  return {
    kind: 'field',
    label,
    value: String(state[key]),
    placeholder,
    onInput: value => { state[key] = value; },
  };
}

function sizeLabel(name) {
  const count = options.sizes[name];
  return count == null ? name : `${name} (~${count.toLocaleString()} params)`;
}

function sections() {
  const presetItems = options.presets.map(p => ({
    label: p.name, stats: p.why, id: p.name, onPick: () => applyPreset(p),
  }));
  const optimizerItems = options.optimizers.map(name => ({
    label: name, id: name, on: state.optimizer === name,
  }));
  const sizeItems = options.size_names.map(name => ({
    label: sizeLabel(name), id: name, on: state.size === name,
  }));
  const custom = state.size === 'custom'
    ? [backend() === 'heatmap' ? fieldSection('channels', 'custom_channels', '24')
                                : fieldSection('width scale', 'custom_scale', '1.0')]
    : [];
  return [
    { kind: 'list', label: 'suggested starting points', items: presetItems },
    {
      kind: 'list', label: 'optimizer', items: optimizerItems,
      onPick: item => { state.optimizer = item.id; renderMenu(); },
    },
    fieldSection('learning rate', 'learning_rate', '3e-4'),
    fieldSection('momentum', 'momentum', '0.9'),
    fieldSection('weight decay', 'weight_decay', '1e-4'),
    {
      kind: 'list', label: 'model size', items: sizeItems,
      onPick: item => { state.size = item.id; renderMenu(); },
    },
    ...custom,
    fieldSection('seed', 'seed', '0'),
    // the menu adds its own close button
  ];
}

function renderMenu() {
  if (menu) menu.refresh(sections());
}

async function loadOptions() {
  const res = await fetch(`/api/hyperparams?backend=${backend()}`);
  options = await res.json();
}

// opens the menu on `anchor` for one backend; the values stay in `state` until the train tab
// reads them through hyperparamState()
async function openHyperparams(anchor, backendPicked) {
  if (backendPicked !== backendName) { backendName = backendPicked; options = null; }
  if (!options) await loadOptions();
  menu = new Menu({ title: 'training hyperparameters', persistent: true, sections: sections() });
  menu.openAt(anchor);
}

function hyperparamState() { return { ...state }; }

window.hyperparams = { open: openHyperparams, state: hyperparamState };
