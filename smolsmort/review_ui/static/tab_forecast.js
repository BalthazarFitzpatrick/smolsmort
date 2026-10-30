// forecast: the regression and classification topics share this one set of tabs, parameterised by
// topic/task. the flavour dropdown picks spec.mode; the topic itself picks spec.task

const FORECAST_ENTITY_ROLES = ['anchor', 'dimension', 'measure', 'target', 'ignore'];
const FORECAST_AGGREGATIONS = ['sum', 'mean', 'median', 'min', 'max'];
const FORECAST_UNITS = ['day', 'week', 'month', 'quarter', 'year'];
const FORECAST_PAGE_SIZE = 50;

function forecastSearchProgress(run) {
  const progress = run.progress || {};
  const budget = run.budget || {};
  const elapsed = Math.max(0, progress.elapsed ?? run.elapsed ?? 0);
  const generation = progress.generation ?? 0;
  const candidate = progress.candidate ?? 0;
  const population = progress.population || budget.population || 16;
  const flat = progress.flat_generations ?? 0;
  const timeCap = budget.time_cap || 1200;
  const plateau = budget.plateau || 5;
  const generations = budget.max_generations || 60;
  const done = run.state === 'done' && run.stopped !== 'cancelled';
  if (run.version === 2 || (!run.version && Number.isFinite(run.time_budget_s))) {
    const duration = run.time_budget_s || timeCap;
    const fill = done ? 1 : Math.min(1, elapsed / duration);
    const state = run.stopped === 'cancelled' ? 'cancelled' : run.state;
    return {fill, done, label: `${run.phase || state} - ${Math.floor(elapsed)}s of ${duration / 60} min - ${state}`};
  }
  const fill = done ? 1 : Math.min(1, Math.max(0, elapsed / timeCap,
    (generation + candidate / population) / generations, flat / plateau));
  let label = `generation ${generation} - candidate ${candidate} of ${population} - ${Math.floor(elapsed)}s of ${timeCap / 60} min - ${flat} of ${plateau} flat generations`;
  if (done) label += ' - done';
  else if (run.state === 'failed' || run.state === 'cancelled' || run.stopped === 'cancelled') {
    label += run.state === 'failed' ? ' - failed' : ' - cancelled';
  }
  return {fill, label, done};
}

function forecastLossSeries(curves) {
  const current = curves.at(-1);
  if (!current) return {curves: [], hidden: 0};
  const visible = [];
  for (const curve of curves) {
    if (curve.metric !== current.metric) continue;
    const previous = visible.at(-1);
    if (previous && curve.key && previous.key === curve.key
        && previous.lastGeneration + 1 === curve.generation) {
      previous.lastGeneration = curve.generation;
      previous.repeated = true;
    } else visible.push({...curve, lastGeneration: curve.generation, repeated: false});
  }
  return {curves: visible, hidden: curves.filter(curve => curve.metric !== current.metric).length};
}

function forecastModelFamily(entry) {
  return entry.family || entry.genome?.family || 'xgboost';
}

function forecastModelLabel(view) {
  return view.family === 'ensemble' ? `ensemble of ${(view.members || []).join(', ')}`
    : view.family || 'xgboost';
}

function forecastFillTable(table, columns, rows) {
  table.replaceChildren();
  const head = table.createTHead().insertRow();
  columns.forEach(label => head.append(label instanceof Node ? label : forecastNode('th', '', label)));
  const body = table.createTBody();
  rows.forEach(values => {
    const row = body.insertRow();
    values.forEach(value => {
      const cell = value instanceof Node ? value : forecastNode('td', '', String(value ?? '-'));
      if (value instanceof Node) { row.append(cell); return; }
      cell.title = cell.textContent;
      row.append(cell);
    });
  });
}

function todayIso() {
  return new Date().toISOString().slice(0, 10);
}

function forecastNode(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) { node.textContent = text; node.title = String(text); }
  return node;
}

function forecastSplitFamilyName(family) {
  const decode = value => {
    try { return decodeURIComponent(value); }
    catch (error) {
      if (error instanceof URIError) return value;
      throw error;
    }
  };
  const colon = family.indexOf(':');
  if (colon < 0) return [null, null, family];
  const owner = family.slice(0, colon);
  const op = family.slice(colon + 1);
  const columnOps = ['dim', 'freq', 'log', 'measure', 'lead', 'mix', 'spread'];
  if (columnOps.includes(owner)) return [op, null, owner];
  if (owner === 'date') return [null, null, `cal:${op}`];
  if (['cal', 'lag', 'roll', 'diff', 'slope', 'momentum', 'ewm'].includes(owner)) {
    return [null, null, family];
  }
  const at = owner.indexOf('@');
  if (at >= 0) return [decode(owner.slice(0, at)), decode(owner.slice(at + 1)), op];
  if (op.startsWith('cal:')) return [null, decode(owner), op];
  if (columnOps.includes(op)) return [decode(owner), null, op];
  return [null, null, family];
}

function forecastFamilyGroups(families) {
  if (!Array.isArray(families)) return [];
  const groups = new Map();
  const labels = {
    lag: 'lags', roll: 'rolling', cal: 'calendar', dim: 'dims',
    measure: 'measures', log: 'log measures', freq: 'frequencies', lead: 'measure lags',
  };
  families.forEach(family => {
    if (typeof family !== 'string') return;
    const [column, clock, op] = forecastSplitFamilyName(family);
    const colon = op.indexOf(':');
    const kind = op.startsWith('lag+') ? 'lags' : op.split(':', 1)[0];
    const label = labels[kind] || (colon < 0 && kind !== 'lags' ? 'other' : kind);
    const context = column && clock ? `${column}@${clock}` : column || clock;
    const part = op.startsWith('lag+') ? op.slice(4)
      : colon >= 0 ? op.slice(colon + 1) : op === 'season' ? 'seasonal lag' : op;
    const value = context ? (colon >= 0 || !labels[kind] ? `${context}:${part}` : context) : part;
    if (!groups.has(label)) groups.set(label, new Set());
    groups.get(label).add(value);
  });
  const order = ['lags', 'rolling', 'ewm', 'calendar', 'dims', 'measures',
    'log measures', 'frequencies', 'measure lags', 'other'];
  const calendar = ['day', 'weekday', 'week', 'month', 'quarter', 'year', 'trend'];
  return [...groups].sort(([a], [b]) => {
    const index = label => order.includes(label) ? order.indexOf(label) : order.length;
    return index(a) - index(b) || a.localeCompare(b);
  }).map(([kind, items]) => {
    const values = [...items].sort((a, b) => {
      if (['lags', 'rolling', 'ewm'].includes(kind)) {
        return Number(a) - Number(b) || a.localeCompare(b, undefined, {numeric: true});
      }
      const part = value => value.slice(value.lastIndexOf(':') + 1);
      if (kind === 'calendar' && calendar.includes(part(a)) && calendar.includes(part(b))) {
        return calendar.indexOf(part(a)) - calendar.indexOf(part(b)) || a.localeCompare(b);
      }
      return a.localeCompare(b);
    });
    return [kind, values.join(',')];
  });
}

function forecastNumber(value) {
  return Number.isFinite(value) ? String(Number(value.toPrecision(4))) : '-';
}

function forecastMetricNumber(metric, value) {
  if (!Number.isFinite(value)) return '-';
  if (metric.unit === 'percent') return `${(value * 100).toFixed(1)}%`;
  if (metric.unit === 'ratio') return value.toFixed(2);
  return forecastNumber(value);
}

function forecastMetricHelp(metric, scope = metric.scope) {
  return `${metric.explanation} ${scope}; ${metric.direction} is better. ${metric.computation}`;
}

function forecastMetricHeader(metric) {
  const node = forecastNode('th', '', metric.label);
  node.dataset.metricId = metric.id;
  node.title = forecastMetricHelp(metric);
  return node;
}

function forecastMetricValue(metric, row, tag = 'td', labelled = false) {
  const saved = row?.error_metrics?.[metric.id];
  const value = forecastMetricNumber(metric, saved?.value);
  const node = forecastNode(tag, 'forecast-metric', labelled ? `${metric.label}: ${value}` : value);
  node.dataset.metricId = metric.id;
  const source = saved?.scope && saved.scope !== metric.scope
    ? `this recipe uses its saved ${saved.scope} metric; no candidate test metric was saved` : null;
  const excluded = Number.isInteger(saved?.excluded_rows) ? `${saved.excluded_rows} rows excluded` : null;
  const reason = Number.isFinite(saved?.value) ? null : saved?.reason || 'metric not saved for this row';
  node.title = [saved?.note || source, reason, excluded, forecastMetricHelp(metric, saved?.scope)].filter(Boolean).join('. ');
  return node;
}

function forecastMetricLines(metrics, row, className = '') {
  const node = forecastNode('span', `forecast-metric-lines ${className}`);
  metrics.forEach(metric => node.append(forecastMetricValue(metric, row, 'span', true)));
  return node;
}

function forecastRatioText(verdict) {
  const ratio = verdict?.accuracy?.ratio;
  return Number.isFinite(ratio)
    ? `test ratio ${ratio.toFixed(2)}, ${ratio < 1 ? 'better' : 'worse'} than naive`
    : verdict?.summary || 'ratio unavailable';
}

function forecastRatioStyle(node, verdict) {
  const ratio = verdict?.accuracy?.ratio;
  node.classList.toggle('better', Number.isFinite(ratio) && ratio < 1);
  node.classList.toggle('worse', Number.isFinite(ratio) && ratio >= 1);
}

function forecastLegacyTileSearch(run, best) {
  const points = new Map();
  const finite = value => Number.isFinite(value) ? value : null;
  const getPoint = record => {
    const index = record.family_generation ?? record.generation;
    if (!Number.isInteger(index) || index < 0) return null;
    if (!points.has(index + 1)) points.set(index + 1,
      {generation: index + 1, validation_error: null, wape: null, mae: null});
    return points.get(index + 1);
  };
  // live legacy polls carry search records before result details can load
  for (const event of run.generations || []) {
    const point = getPoint(event);
    if (!point) continue;
    point.validation_error = finite(event.best);
    for (const metric of ['wape', 'mae']) point[metric] = finite(event.metrics?.[metric]);
  }
  for (const curve of run.curves || []) {
    if (!curve.metric || ['generation', 'validation_error', 'mape', 'test_ratio'].includes(curve.metric)
        || !Array.isArray(curve.validation) || !curve.validation.length) continue;
    const point = getPoint(curve);
    if (point && point[curve.metric] == null) point[curve.metric] = finite(curve.validation.at(-1));
  }
  if (points.size) return {
    search_curve: Array.from({length: Math.max(...points.keys())}, (_, index) =>
      points.get(index + 1) || {generation: index + 1}), search_single: false,
  };
  const values = {validation_error: finite(run.best_error ?? best.fitness),
    wape: finite(best.metrics?.wape), mae: finite(best.metrics?.mae)};
  const known = Object.values(values).some(Number.isFinite);
  return {search_curve: known ? [{generation: 1, ...values}] : [], search_single: known};
}

// live legacy polls carry the round curves themselves; family tiles get them from the server
function forecastLegacyRoundCurves(run) {
  const latest = new Map();
  for (const curve of run.curves || []) {
    const index = curve.family_generation ?? curve.generation;
    if (!Number.isInteger(index) || !Array.isArray(curve.validation) || !curve.validation.length) continue;
    latest.set(index, {generation: index + 1, key: curve.key, metric: curve.metric,
      validation: curve.validation, curves: curve.curves || {}});
  }
  return [...latest.keys()].sort((a, b) => a - b).slice(-30).map(index => latest.get(index));
}

function forecastTileEntries(run) {
  const families = run?.version === 1 ? [] : Object.entries(run?.families || {});
  const entries = families.length ? families.map(([name, status]) => ({
    name, ...status, ...(status.tile || {}),
  })) : Object.entries(run?.tiles || {}).map(([name, tile]) => ({name, state: run.state, ...tile}));
  if (!entries.length && run?.version === 1) {
    const best = run.leaderboard?.[0] || {};
    entries.push({
      name: run.family || 'xgboost', state: run.state, genome: best.genome,
      validation_error: run.best_error ?? best.fitness, test_ratio: run.verdict?.accuracy?.ratio,
      error_metrics: run.error_metrics,
      test_error: run.test_error, test_error_metric: run.verdict?.accuracy?.metric,
      test_naive_wape: run.verdict?.accuracy?.metric === 'wape' ? run.verdict.accuracy.baseline_error : null,
      elapsed: run.progress?.elapsed ?? run.elapsed, generation: run.progress?.generation ?? run.generation,
      folds: (best.metrics?.fold_scores || []).map((error, index) => ({label: `fold ${index + 1}`, error})),
      ...forecastLegacyTileSearch(run, best), round_curves: forecastLegacyRoundCurves(run),
    });
  }
  return entries.sort((a, b) => {
    const completed = Number(b.state === 'done') - Number(a.state === 'done');
    if (completed) return completed;
    const ratioA = Number.isFinite(a.test_ratio) ? a.test_ratio : Infinity;
    const ratioB = Number.isFinite(b.test_ratio) ? b.test_ratio : Infinity;
    if (ratioA !== ratioB) return ratioA - ratioB;
    const errorA = Number.isFinite(a.validation_error) ? a.validation_error : Infinity;
    const errorB = Number.isFinite(b.validation_error) ? b.validation_error : Infinity;
    return errorA - errorB || a.name.localeCompare(b.name);
  });
}

function forecastTileState(tile) {
  if (['starting', 'queued', 'pending', 'running', 'paused'].includes(tile.state)) return 'running';
  if (['pruned', 'failed', 'cancelled'].includes(tile.state)) return 'worse';
  if (Number.isFinite(tile.test_ratio)) return tile.test_ratio < 1 ? 'better' : 'worse';
  return 'unknown';
}

function forecastTileSummary(tile) {
  const minutes = Number.isFinite(tile.elapsed) ? `${Math.round(tile.elapsed / 60)} min` : null;
  const generation = Number.isFinite(tile.generation) ? `gen ${tile.generation}` : null;
  if (forecastTileState(tile) === 'running') {
    return [`${tile.state}${minutes ? ` ${minutes}` : ''}`, generation].filter(Boolean).join(', ') + '.';
  }
  if (tile.state === 'pruned') {
    return `pruned${minutes ? ` after ${minutes}` : ''}.`;
  }
  return [minutes, generation].filter(Boolean).join(', ') +
    (minutes || generation ? '.' : `${tile.state || 'pending'}.`);
}

function forecastTileMenus(tile) {
  const group = forecastNode('div', 'forecast-tile-menus');
  const importance = tile.importance?.features || [];
  const features = importance.length ? importance.map(item => [item.feature, Number.isFinite(item.importance)
    ? `${Math.round(item.importance * 100)}%` : ''])
    : (tile.genome?.families || []).map(name => [name, '']);
  const params = Object.entries(tile.genome?.params || {});
  for (const [key, rows, empty] of [
    ['feat', features, tile.genome ? 'no saved features' : 'features not saved yet'],
    ['param', params, tile.genome ? 'nothing to tune' : 'parameters not saved yet'],
  ]) {
    const details = forecastNode('details');
    details.dataset.menu = key;
    const summary = forecastNode('summary');
    summary.append(forecastNode('span', '', key), forecastNode('span', 'forecast-tile-dim', `${rows.length} ▾`));
    const menu = forecastNode('div', 'forecast-tile-menu');
    rows.forEach(([name, value]) => {
      const row = forecastNode('div', 'forecast-tile-menu-row');
      const text = value !== null && typeof value === 'object' ? JSON.stringify(value) : String(value ?? '-');
      row.append(forecastNode('span', '', name), forecastNode('span', 'forecast-tile-dim', text));
      menu.append(row);
    });
    if (!rows.length) menu.append(forecastNode('div', 'forecast-tile-menu-row forecast-tile-dim', empty));
    details.append(summary, menu);
    details.addEventListener('keydown', event => {
      if (event.key === 'Escape') { details.open = false; summary.focus(); }
    });
    group.append(details);
  }
  return group;
}

function forecastTileSearchScale(tiles, selected, registry) {
  const fallback = registry.find(metric => metric.id === 'validation_error')
    || {id: 'validation_error', label: 'validation error', unit: 'number'};
  const requested = selected || fallback;
  const tracked = ['validation_error', 'wape', 'mae'].includes(requested.id)
    || tiles.some(tile => (tile.search_curve || []).some(point => Number.isFinite(point[requested.id])));
  const unsupported = ['mape', 'test_ratio'].includes(requested.id) || !tracked;
  const metric = unsupported ? fallback : requested;
  const points = tiles.flatMap(tile => tile.search_curve || []);
  const values = points.map(point => point[metric.id]).filter(Number.isFinite);
  const low = values.length ? Math.min(...values) : 0;
  const high = values.length ? Math.max(...values) : 1;
  const padding = (high - low || Math.abs(high) || 1) * 0.1;
  return {metric, note: unsupported ? `${requested.label}: not tracked per generation` : '',
    low: low >= 0 ? Math.max(0, low - padding) : low - padding, high: high + padding,
    generation: Math.max(1, ...points.map(point => point.generation).filter(Number.isFinite))};
}

// tile charts share one drawing box: plot on top, axis and its labels below
const FORECAST_TILE_PLOT = {height: 58, top: 4, bottom: 40, axis: 42, labels: 55};

function forecastTileSvg(width, ariaLabel, className) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${FORECAST_TILE_PLOT.height}`);
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', ariaLabel);
  svg.classList.add(className);
  const add = (tag, attrs, parent = svg) => {
    const node = document.createElementNS(svg.namespaceURI, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    parent.append(node);
    return node;
  };
  const label = (text, attrs) => { add('text', attrs).textContent = text; };
  const axis = (last, name) => {
    const {axis: line, labels} = FORECAST_TILE_PLOT;
    add('line', {x1: 8, x2: width - 8, y1: line, y2: line, class: 'forecast-tile-axis'});
    label('1', {x: 8, y: labels});
    label(name, {x: width / 2, y: labels, 'text-anchor': 'middle'});
    if (last > 1) label(String(last), {x: width - 8, y: labels, 'text-anchor': 'end'});
  };
  return {svg, add, axis};
}

function forecastTileChartHead(className, title, value) {
  const head = forecastNode('div', className);
  head.append(forecastNode('span', '', title));
  if (value) head.append(forecastNode('span', 'forecast-tile-current-value', value));
  return head;
}

function forecastTileChart(tile, scale, width) {
  const {metric, low, high, generation} = scale;
  const wrap = forecastNode('div', 'forecast-tile-chart');
  wrap.dataset.chartMetric = metric.id;
  wrap.dataset.scaleMin = String(low);
  wrap.dataset.scaleMax = String(high);
  wrap.dataset.generationMax = String(generation);
  const metricLabel = metric.id === 'validation_error' ? metric.label : `${metric.label} · validation`;
  const points = (tile.search_curve || []).filter(point => Number.isFinite(point.generation));
  const known = points.filter(point => Number.isFinite(point[metric.id]));
  if (!known.length) {
    wrap.append(forecastTileChartHead('forecast-tile-chart-label', metricLabel),
      forecastNode('div', 'forecast-tile-placeholder', 'no search curve saved for this family'));
    if (scale.note) wrap.append(forecastNode('div', 'forecast-tile-chart-note', scale.note));
    return wrap;
  }
  const last = points.at(-1);
  const value = Number.isFinite(last[metric.id]) ? forecastMetricNumber(metric, last[metric.id]) : 'not saved';
  const end = (tile.state === 'pruned' ? 'pruned · ' : '')
    + (tile.search_single ? 'single value' : `gen ${last.generation}`);
  wrap.append(forecastTileChartHead('forecast-tile-chart-label', metricLabel, `${end}: ${value}`));
  const {svg, add, axis} = forecastTileSvg(width,
    `${tile.name}: ${metric.label} by generation; ${end}: ${value}; shared scale ${forecastMetricNumber(metric, low)} to ${forecastMetricNumber(metric, high)}`,
    'forecast-tile-search');
  const {top, bottom} = FORECAST_TILE_PLOT;
  const x = step => 8 + (step - 1) / Math.max(1, generation - 1) * (width - 16);
  const y = score => bottom - (score - low) / (high - low) * (bottom - top);
  axis(generation, 'generation');
  let segment = [];
  const draw = () => {
    if (segment.length > 1) add('polyline', {points: segment.join(' '), class: 'forecast-tile-search-line'});
    segment = [];
  };
  points.forEach(point => {
    if (!Number.isFinite(point[metric.id])) { draw(); return; }
    const cx = x(point.generation), cy = y(point[metric.id]);
    segment.push(`${cx.toFixed(1)},${cy.toFixed(1)}`);
    const marker = add('circle', {cx, cy, r: point === last ? 4 : 2,
      class: point === last ? 'forecast-tile-current' : 'forecast-tile-search-point',
      'data-generation': point.generation, 'data-value': point[metric.id]});
    add('title', {}, marker).textContent = `generation ${point.generation}: ${forecastMetricNumber(metric, point[metric.id])}`;
  });
  draw();
  if (tile.state === 'pruned' && Number.isFinite(last[metric.id])) {
    const cx = x(last.generation), cy = y(last[metric.id]);
    add('path', {d: `M${cx - 4},${cy - 4}l8,8m0,-8l-8,8`, class: 'forecast-tile-pruned'});
  }
  wrap.append(svg);
  if (scale.note) wrap.append(forecastNode('div', 'forecast-tile-chart-note', scale.note));
  return wrap;
}

// the picked metric when it is recorded per boosting round, else wape
function forecastTileRoundMetric(selectedIds, registry) {
  const id = selectedIds.find(name => ['wape', 'mae'].includes(name)) || 'wape';
  return registry.find(metric => metric.id === id) || {id, label: id, unit: id === 'wape' ? 'percent' : 'target'};
}

// one tile's per-round series: the picked metric where saved, else the fit's own loss metric
function forecastTileRoundSeries(tile, metric) {
  const curves = tile.round_curves || [];
  const newest = curves.at(-1);
  if (!newest) return null;
  if (Array.isArray(newest.curves?.[metric.id])) {
    return {metric, recorded: true, curves: curves.filter(curve => Array.isArray(curve.curves?.[metric.id]))
      .map(curve => ({generation: curve.generation, values: curve.curves[metric.id]}))};
  }
  const loss = {id: newest.metric, label: newest.metric || 'loss', unit: 'number'};
  return {metric: loss, recorded: false, curves: curves.filter(curve => curve.metric === newest.metric)
    .map(curve => ({generation: curve.generation, values: curve.validation}))};
}

// early rounds sit far above the rest; the scale starts a tenth of the way in so the tail reads
function forecastTileRoundRange(curves) {
  const values = [], tops = [];
  for (const curve of curves) {
    const finite = curve.values.filter(Number.isFinite);
    values.push(...finite);
    tops.push(...finite.slice(Math.floor(finite.length / 10)));
  }
  if (!values.length) return null;
  const low = Math.min(...values), high = Math.max(...tops);
  const padding = (high - low || Math.abs(high) || 1) * 0.1;
  return {low: Math.max(0, low - padding), high: high + padding};
}

// one shared scale across tiles, only over curves of the picked metric itself
function forecastTileRoundScale(tiles, metric) {
  return forecastTileRoundRange(tiles.map(tile => forecastTileRoundSeries(tile, metric))
    .filter(series => series?.recorded).flatMap(series => series.curves));
}

function forecastTileRoundChart(tile, metric, shared, width) {
  const wrap = forecastNode('div', 'forecast-tile-rounds');
  const series = forecastTileRoundSeries(tile, metric);
  const curves = (series?.curves || []).filter(curve => curve.values.some(Number.isFinite));
  if (!curves.length) {
    wrap.append(forecastTileChartHead('forecast-tile-chart-label', `${metric.label} per round`),
      forecastNode('div', 'forecast-tile-placeholder', 'no round curve saved for this family'));
    return wrap;
  }
  const shown = series.metric;
  wrap.dataset.chartMetric = shown.id;
  wrap.dataset.recorded = String(series.recorded);
  // a fallback loss metric differs per tile, so it gets its own scale
  const scale = (series.recorded && shared) || forecastTileRoundRange(curves);
  const format = value => shown.unit === 'number' ? String(Number(value.toPrecision(3))) : forecastMetricNumber(shown, value);
  const newest = curves.at(-1);
  const end = newest.values.filter(Number.isFinite).at(-1);
  const rounds = Math.max(...curves.map(curve => curve.values.length));
  const title = series.recorded ? `${shown.label} per round` : `${shown.label} · ${metric.label} not saved`;
  wrap.append(forecastTileChartHead('forecast-tile-chart-label', title,
    `gen ${newest.generation} · round ${newest.values.length}: ${format(end)}`));
  const {svg, add, axis} = forecastTileSvg(width,
    `${tile.name}: ${shown.label} per boosting round for the newest ${curves.length} generations; generation ${newest.generation} ends at ${format(end)}`,
    'forecast-tile-round-plot');
  const {top, bottom} = FORECAST_TILE_PLOT;
  const x = round => 8 + (round - 1) / Math.max(1, rounds - 1) * (width - 16);
  // values above the scale pin to the top edge rather than leave the box
  const y = value => Math.max(top, bottom - (value - scale.low) / (scale.high - scale.low) * (bottom - top));
  axis(rounds, 'round');
  curves.forEach(curve => {
    const age = newest.generation - curve.generation;
    const points = curve.values.map((value, index) => Number.isFinite(value)
      ? `${x(index + 1).toFixed(1)},${y(value).toFixed(1)}` : null).filter(Boolean);
    if (points.length < 2) return;
    const line = add('polyline', {points: points.join(' '),
      class: curve === newest ? 'forecast-tile-round-current' : 'forecast-tile-round-line',
      'data-generation': curve.generation,
      'stroke-opacity': curve === newest ? 1 : Math.max(0.15, 0.7 - age / 40).toFixed(2)});
    add('title', {}, line).textContent = `generation ${curve.generation}: ${format(curve.values.filter(Number.isFinite).at(-1))} after ${curve.values.length} rounds`;
  });
  wrap.append(svg);
  return wrap;
}

function forecastTileWapeBars(tile) {
  const folds = (tile.folds || []).filter(fold => Number.isFinite(fold.error) && fold.error >= 0);
  const saved = tile.test_error_metric === 'wape' ? tile.test_error : null;
  const error = Number.isFinite(saved) ? saved : tile.error_metrics?.wape?.value;
  return [...folds, {label: 'test', error, baseline: tile.test_naive_wape}];
}

function forecastTileWapeScale(tiles) {
  const values = tiles.flatMap(tile => forecastTileWapeBars(tile)
    .flatMap(bar => [bar.error, bar.baseline])).filter(value => Number.isFinite(value) && value >= 0);
  const largest = values.length ? Math.max(...values) : 0;
  // round up to a whole percentage point, with room for the naive test marker
  return Math.max(0.01, Math.ceil(largest * 100 - 1e-9) / 100);
}

function forecastTileBars(tile, scale) {
  const wrap = forecastNode('div', 'forecast-tile-bars');
  const bars = forecastTileWapeBars(tile);
  const height = `${Math.max(4, bars.length) * 22 + 6}px`;
  wrap.style.height = height;
  wrap.style.flexBasis = height;
  const percent = value => forecastMetricNumber({unit: 'percent'}, value);
  wrap.dataset.scaleMax = String(scale);
  wrap.setAttribute('aria-label', `wape scale: 0 to ${percent(scale)}; lower is better`);
  bars.forEach(bar => {
    const known = Number.isFinite(bar.error) && bar.error >= 0;
    const baseline = Number.isFinite(bar.baseline) && bar.baseline >= 0;
    const row = forecastNode('div', 'forecast-tile-bar');
    row.dataset.label = bar.label;
    row.title = known ? `${bar.label}: wape ${percent(bar.error)}; lower is better`
      : `${bar.label}: wape not saved`;
    if (baseline) row.title += `; naive wape ${percent(bar.baseline)}`;
    const track = forecastNode('div', 'forecast-tile-track');
    const fill = forecastNode('div', 'forecast-tile-fill');
    fill.style.width = `${known ? Math.min(100, bar.error / scale * 100) : 0}%`;
    if (!known) row.dataset.state = 'unknown';
    track.append(fill);
    if (baseline) {
      const mark = forecastNode('div', 'forecast-tile-mark');
      mark.style.left = `${Math.min(100, bar.baseline / scale * 100)}%`;
      mark.title = `naive test wape ${percent(bar.baseline)}`;
      track.append(mark);
    }
    row.append(forecastNode('span', 'forecast-tile-bar-label forecast-tile-dim', bar.label), track,
      forecastNode('span', 'forecast-tile-bar-value', known ? percent(bar.error) : '-'));
    row.setAttribute('aria-label', row.title);
    wrap.append(row);
  });
  return wrap;
}

// a square tile: its side is the height the one-chart tile needed, so both charts fit above
function forecastTileSize(tiles, metrics) {
  const folds = Math.max(4, ...tiles.map(tile => forecastTileWapeBars(tile).length));
  return 266 + metrics.length * 20 + (folds - 4) * 22;
}

function forecastCreateTile(tile, index, metrics, scale, searchScale, rounds) {
  const wrap = forecastNode('article', 'forecast-tile');
  wrap.dataset.family = tile.name;
  wrap.dataset.state = forecastTileState(tile);
  wrap.setAttribute('aria-label', `${tile.name}: ${tile.state}`);
  // face padding and border take 28px of the side
  const plotWidth = rounds.size - 28;
  const inner = forecastNode('div', 'forecast-tile-inner');
  for (const side of ['front', 'back']) {
    const face = forecastNode('div', `forecast-tile-face forecast-tile-${side}`);
    const header = forecastNode('div', 'forecast-tile-header');
    header.append(forecastNode('span', 'forecast-tile-name', `#${index + 1} ${tile.name}`));
    const score = forecastTileState(tile) === 'running' ? '…' : Number.isFinite(tile.test_ratio)
      ? `${tile.test_ratio <= 1 ? '+' : '−'}${Math.round(Math.abs(1 - tile.test_ratio) * 100)}%` : '-';
    const scoreNode = forecastNode('span', 'forecast-tile-score', score);
    scoreNode.title = Number.isFinite(tile.test_ratio) ? `test ratio ${tile.test_ratio.toFixed(2)} vs naive`
      : forecastTileState(tile) === 'running' ? 'test score pending' : 'test ratio not saved';
    header.append(scoreNode);
    const charts = side === 'front'
      ? [forecastTileChart(tile, searchScale, plotWidth),
        forecastTileRoundChart(tile, rounds.metric, rounds.scale, plotWidth)]
      : [forecastTileBars(tile, scale)];
    face.append(header, ...charts, forecastTileMenus(tile), forecastMetricLines(metrics, tile));
    const summary = side === 'front' ? forecastTileSummary(tile) : 'wape per validation window; mark = naive on test';
    const caption = forecastNode('div', 'forecast-tile-caption forecast-tile-dim', summary);
    if (side === 'front') caption.title = [summary, tile.pruned_reason || tile.reason].filter(Boolean).join(' ');
    face.append(caption);
    inner.append(face);
  }
  wrap.append(inner);
  return wrap;
}

const FORECAST_OBJECTIVES = {
  squared: 'squared error', absolute: 'absolute error', tweedie: 'tweedie loss',
  poisson: 'poisson loss', aft: 'survival loss', logistic: 'log loss',
  softprob: 'multiclass log loss',
};

const FORECAST_LOSS_HELP = 'l1 (absolute) is absolute-error loss: the model minimises mean absolute error; mae is the measured number. squared is squared-error loss. tweedie and poisson use their named distribution losses; logistic and softprob use log loss; hazard uses hazard log loss; aft uses survival loss';

function forecastDropdown(options, value, onPick, label) {
  const head = forecastNode('button', 'toggle dropdown-head');
  head.type = 'button';
  head.setAttribute('aria-label', label);
  head.append(forecastNode('span', null, value));
  head.onclick = () => listMenu(label, options.map(id => ({id, label: id, on: id === value})), item => {
    value = item.id;
    setHead(head, value);
    onPick(value);
  }).openAt(head);
  return head;
}

// choices open inside the parent menu: a second floating menu would dismiss its draft
function forecastInlineSelect(options, value, onPick, label) {
  const wrap = forecastNode('div', 'forecast-inline-select');
  const head = forecastNode('div', 'toggle dropdown-head menu-item');
  head.tabIndex = 0;
  head.setAttribute('role', 'button');
  head.setAttribute('aria-label', label);
  const text = forecastNode('span', null, value);
  const list = forecastNode('div', 'forecast-inline-list');
  head.append(text);
  const close = () => { list.replaceChildren(); head.classList.remove('open'); };
  head.onclick = () => {
    if (list.childElementCount) { close(); return; }
    head.classList.add('open');
    options.forEach(option => {
      const row = forecastNode('div', 'toggle menu-item' + (option === value ? ' on' : ''));
      row.append(forecastNode('span', 'name', option));
      row.onclick = () => {
        value = option;
        text.textContent = option;
        close();
        head.focus();
        onPick(option);
      };
      list.append(row);
    });
  };
  wrap.append(head, list);
  return wrap;
}

class ForecastSetupMenu extends Menu {
  constructor(key, options) {
    super(options);
    this.key = key;
  }

  _build() {
    const panel = super._build();
    panel.classList.add('forecast-setup-menu', `forecast-setup-${this.key}`);
    panel.querySelectorAll('.menu-item').forEach(row => { row.title = row.textContent; });
    return panel;
  }
}

class ForecastFieldMenu extends Menu {
  _build() {
    const panel = super._build();
    panel.classList.add('forecast-field-menu');
    panel.querySelectorAll('.menu-item').forEach(row => { row.title = row.textContent; });
    return panel;
  }
}

// warm starts retain their own v1 budget controls
function forecastStepperRow(prefix, row) {
  return `<div class="run-controls stepper-row" data-stepper="${row.key}">
    <span class="field-label">${row.label}</span>
    <span class="spacer"></span>
    <span class="field-value" id="${prefix}-${row.key}">${row.value}</span>
    <div class="toggle" data-step="-${row.step}">-</div>
    <div class="toggle" data-step="${row.step}">+</div>
  </div>`;
}

function forecastWireSteppers(panel, prefix, rows, values, onChange) {
  panel.querySelectorAll(`[data-stepper]`).forEach(row => {
    const key = row.dataset.stepper;
    const spec = rows.find(r => r.key === key);
    if (!spec) return;
    row.querySelectorAll('[data-step]').forEach(btn => {
      btn.onclick = () => {
        values[key] = Math.max(spec.min, Math.min(spec.max, values[key] + Number(btn.dataset.step)));
        setText(`${prefix}-${key}`, String(values[key]));
        onChange();
      };
    });
  });
}

class ForecastTopic {
  constructor(topic, task) {
    this.topic = topic;
    this.task = task;
    this.storageKey = `smolsmort:forecast:${topic}`;
    this.source = null;
    this.encoding = 'auto';
    this.columnsInfo = [];
    this.sourceRows = null;
    this.columnState = {}; // name -> {role, aggregation, known}
    this.seriesSetup = null; // persisted browser state keeps its existing key
    this.step = 'week';
    this.horizon = 8;
    this.asOf = todayIso();
    this.where = '';
    this.predictWhere = '';
    this.censorOn = false;
    this.censorUnit = 'week';
    this.censorAsOf = todayIso();
    this.lastPrep = null;
    this.preparedByMode = {};
    this.detectedEncoding = null;
    this.searchValues = {population: 16, plateau: 5, generations: 60, minutes: 20};
    this.currentRunId = null;
    this.selectedRunId = null;
    this.runView = 'tiles';
    this.tilesFlipped = false;
    this.tilesRunId = null;
    this.metricRegistry = [];
    this.selectedMetricIds = ['validation_error', 'test_ratio'];
    this.metricStorageKey = `${this.storageKey}:error-metrics`;
    try {
      const saved = JSON.parse(localStorage.getItem(this.metricStorageKey));
      if (Array.isArray(saved) && saved.every(id => typeof id === 'string')) this.selectedMetricIds = saved;
    } catch (error) { /* storage can be unavailable in private windows */ }
    this.runCache = new Map();
    this.runOrder = 'date, newest first';
    this.searchEntered = false;
    this.pollTimer = null;
    this.searchChart = null;
    this.lossChart = null;
    this.lossCurves = [];
    this.lossGeneration = -1;
    this.generationFamily = null;
    this.resultsRunId = null;
    this.resultsSpec = null;
    this.resultsModel = null;
    this.importanceFamily = null;
    this.resultsLoad = 0;
    this.horizonStorageKey = `${this.storageKey}:horizon-view`;
    this.horizonSelections = {};
    try {
      const saved = JSON.parse(localStorage.getItem(this.horizonStorageKey));
      if (saved && typeof saved === 'object' && !Array.isArray(saved)) this.horizonSelections = saved;
    } catch (error) { /* storage can be unavailable in private windows */ }
    this.viewRequest = 0;
    this.tableRequest = 0;
    this.chart = null;
    this.filters = {};
    this.breakdown = [];
    this.tableState = {page: 0, sort: null, desc: false, group: null};
    this.loadStored();
    if (!FORECAST_UNITS.includes(this.step)) this.step = 'week';
  }

  mode() {
    return smolsmortTabs.flavour(this.topic) || 'entity';
  }

  acceptMetricRegistry(view, context = this.metricContext) {
    if (this.metricContext && context !== this.metricContext) return;
    if (Array.isArray(view?.metric_registry)) this.metricRegistry = view.metric_registry;
    this.paintMetricPicker();
  }

  selectedMetrics(view) {
    const registry = Array.isArray(view?.metric_registry) ? view.metric_registry : this.metricRegistry;
    return registry.filter(metric => this.selectedMetricIds.includes(metric.id));
  }

  mountMetricPicker(slot) {
    this.metricContext = slot;
    if (!this.metricControls) {
      const controls = forecastNode('div', 'forecast-metric-controls');
      const picker = forecastNode('div', 'forecast-metric-picker');
      const button = forecastNode('button', 'toggle dropdown-head', 'error metrics');
      button.type = 'button';
      button.id = `${this.topic}-error-metrics`;
      button.setAttribute('aria-expanded', 'false');
      button.setAttribute('aria-controls', `${this.topic}-error-metrics-menu`);
      const menu = forecastNode('div', 'forecast-metric-menu hidden');
      menu.id = `${this.topic}-error-metrics-menu`;
      menu.setAttribute('role', 'group');
      menu.setAttribute('aria-label', 'error metrics');
      const close = (focus = false) => {
        menu.classList.add('hidden');
        button.setAttribute('aria-expanded', 'false');
        if (focus) button.focus();
      };
      button.onclick = () => {
        const open = menu.classList.toggle('hidden') === false;
        button.setAttribute('aria-expanded', String(open));
      };
      picker.addEventListener('keydown', event => {
        if (event.key === 'Escape') { event.preventDefault(); close(true); }
        if (event.key === 'ArrowDown' && event.target === button) {
          event.preventDefault();
          menu.classList.remove('hidden');
          button.setAttribute('aria-expanded', 'true');
          menu.querySelector('input')?.focus();
        }
      });
      document.addEventListener('pointerdown', event => {
        if (!picker.contains(event.target)) close();
      });
      picker.addEventListener('focusout', event => {
        if (event.relatedTarget && !picker.contains(event.relatedTarget)) close();
      });
      const help = forecastNode('details', 'forecast-metric-help');
      help.append(forecastNode('summary', '', 'what do these numbers mean'));
      help.append(forecastNode('div', 'forecast-metric-help-lines'));
      picker.append(button, menu);
      controls.append(picker, help);
      this.metricControls = controls;
    }
    document.getElementById(`${this.topic}-${slot}-metrics`).append(this.metricControls);
    this.paintMetricPicker();
  }

  paintMetricPicker() {
    if (!this.metricControls) return;
    const menu = this.metricControls.querySelector('.forecast-metric-menu');
    const signature = JSON.stringify(this.metricRegistry);
    if (menu.dataset.signature === signature) return;
    menu.dataset.signature = signature;
    menu.replaceChildren();
    const help = this.metricControls.querySelector('.forecast-metric-help-lines');
    help.replaceChildren();
    this.metricRegistry.forEach(metric => {
      const label = forecastNode('label', 'forecast-metric-choice');
      label.title = forecastMetricHelp(metric);
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.dataset.metricId = metric.id;
      input.checked = this.selectedMetricIds.includes(metric.id);
      input.onchange = () => {
        this.selectedMetricIds = input.checked ? [...this.selectedMetricIds, metric.id]
          : this.selectedMetricIds.filter(id => id !== metric.id);
        try { localStorage.setItem(this.metricStorageKey, JSON.stringify(this.selectedMetricIds)); }
        catch (error) { /* selection still works when storage is unavailable */ }
        this.paintRuns();
        this.paintRunDetail(this.runCache.get(this.selectedRunId));
        if (this.resultsView) {
          this.paintModelResults(this.resultsView);
          this.paintVerdict(this.resultsView.verdict, forecastModelLabel(this.resultsView));
        }
      };
      label.append(input, forecastNode('span', '', metric.label));
      menu.append(label);
      help.append(forecastNode('p', '', `${metric.label}: ${metric.explanation}`));
    });
    if (!this.metricRegistry.length) menu.append(forecastNode('span', 'field-label', 'metric definitions not saved'));
    help.append(forecastNode('p', '', FORECAST_LOSS_HELP));
    help.append(forecastNode('p', '', 'recipe wape and mae use saved validation metrics; test predictions are not saved per recipe. a dash means the metric is unavailable; hover for the reason'));
  }

  // ---- persistence: the last spec per topic, wrapped so a private window never breaks the tab ----
  loadStored() {
    try {
      const raw = localStorage.getItem(this.storageKey);
      if (!raw) return;
      const saved = JSON.parse(raw);
      Object.assign(this, saved.state || {});
      // utf-8 was the old default, and auto still picks it for any file that decodes as utf-8
      if (this.encoding === 'utf-8') this.encoding = 'auto';
    } catch (err) { /* nothing saved, or it did not parse - start clean */ }
  }

  saveStored() {
    try {
      const state = {
        source: this.source, encoding: this.encoding, columnState: this.columnState,
        seriesSetup: this.seriesSetup,
        step: this.step, horizon: this.horizon, asOf: this.asOf, where: this.where,
        predictWhere: this.predictWhere, censorOn: this.censorOn, censorUnit: this.censorUnit,
        censorAsOf: this.censorAsOf,
        columnsInfo: this.columnsInfo, sourceRows: this.sourceRows, detectedEncoding: this.detectedEncoding,
        preparedByMode: this.preparedByMode, modelPlan: this.modelPlan,
      };
      localStorage.setItem(this.storageKey, JSON.stringify({state}));
    } catch (err) { /* private window, no matter */ }
    this.paintDataset();
  }

  restorePrepared() {
    const saved = this.preparedByMode[this.mode()];
    this.spec = saved?.spec || null;
    this.lastPrep = saved?.prepared || null;
    this.paintPrepSummary(this.lastPrep || {});
    this.paintDataset();
  }

  rememberPrepared(spec, prepared) {
    this.spec = spec;
    this.lastPrep = prepared;
    this.detectedEncoding = prepared.summary?.encoding || this.detectedEncoding;
    this.preparedByMode[spec.mode] = {spec, prepared};
    this.saveStored();
    this.paintPrepSummary(prepared);
  }

  currentSpec() {
    const spec = this.buildSpec();
    if (!spec.source) throw new Error('no dataset selected');
    if (!spec.columns.some(col => col.role === 'target')) throw new Error('pick a variable to predict');
    if (spec.mode === 'timeseries' && !spec.columns.some(col => col.role === 'time')) {
      throw new Error('pick a date to extend in time');
    }
    if (spec.mode === 'entity' && !spec.columns.some(col => col.role === 'anchor')) {
      throw new Error('pick an anchor date');
    }
    return spec;
  }

  paintDataset() {
    if (this.topic !== forecastActiveTopic) return;
    const box = document.getElementById('forecast-dataset');
    box.classList.remove('hidden');
    box.replaceChildren(forecastNode('span', 'field-label', 'dataset'));
    if (!this.source) {
      box.append(forecastNode('span', 'forecast-dataset-file field-value', 'no dataset selected - pick one on the data tab'));
      return;
    }
    const saved = this.preparedByMode[this.mode()];
    const unchanged = saved && JSON.stringify(saved.spec) === JSON.stringify(this.buildSpec());
    const state = !saved ? 'not prepared' : unchanged ? 'prepared' : 'changed since prepare';
    const file = forecastNode('span', 'forecast-dataset-file field-value', this.source);
    file.title = this.source;
    box.append(file);
    const summary = unchanged ? saved.prepared.summary || {} : {};
    const facts = [];
    if (Number.isFinite(summary.source_rows)) facts.push(`${summary.source_rows} rows`);
    if (Number.isFinite(summary.source_columns)) facts.push(`${summary.source_columns} columns`);
    if (this.detectedEncoding) facts.push(this.detectedEncoding);
    if (facts.length) box.append(forecastNode('span', 'stat', facts.join(' · ')));
    box.append(forecastNode('span', 'field-value', state));
  }

  async onFlavourChange() {
    if (this.dataSay) this.dataSay('flavour changed - reloading columns for the new mode');
    const loading = this.source ? this.loadColumns() : Promise.resolve();
    this.paintDataMode();
    this.restorePrepared();
    this.models.clearCatalog();
    await loading;
    if (!this.models.panel.classList.contains('hidden')) this.models.enter();
  }

  // ================================================================== data tab

  mountData(panel) {
    const censorBlock = this.task === 'regression' ? `
      <div class="run-controls forecast-censor-row">
        <div class="toggle" id="${this.topic}-censor-toggle">censoring off</div>
        <span class="field-label">anchor</span>
        <span class="field-value" id="${this.topic}-censor-anchor">-</span>
        <span class="field-label">unit</span>
        <span id="${this.topic}-censor-unit"></span>
        <span class="field-label">as of</span>
        <input type="date" id="${this.topic}-censor-asof" class="text-field">
      </div>` : '';
    panel.innerHTML = `
      <div class="forecast-pane">
        <div class="run-controls">
          <div class="toggle dropdown-head grow" id="${this.topic}-data-source"><span>source</span></div>
          <span class="field-label">encoding</span>
          <input type="text" id="${this.topic}-data-encoding" class="text-field forecast-encoding">
          <span class="stat" id="${this.topic}-data-root"></span>
        </div>
        <div id="${this.topic}-data-columns"></div>
        <div class="h-divider"></div>
        <div id="${this.topic}-timeseries-row">
          <div class="forecast-setup-heads" id="${this.topic}-setup-heads"></div>
          <div class="stat forecast-setup-readout" id="${this.topic}-setup-readout" aria-live="polite"></div>
          <div class="run-controls">
            <span class="field-label">as of</span>
            <input type="date" id="${this.topic}-asof" class="text-field">
          </div>
        </div>
        <div class="field-label" id="${this.topic}-entity-heading">entity settings</div>
        <div id="${this.topic}-entity-row">
          ${censorBlock}
          <div class="run-controls">
            <span class="field-label">rows to predict</span>
            <input type="text" id="${this.topic}-predict-where"
                   placeholder="sql filter, optional" class="text-field grow">
          </div>
        </div>
        <div class="run-controls">
          <span class="field-label">filter</span>
          <input type="text" id="${this.topic}-where"
                 placeholder="sql filter over the source, optional" class="text-field grow">
        </div>
        <div class="run-controls">
          <div class="toggle adds" id="${this.topic}-prepare">prepare</div>
          <span class="stat" id="${this.topic}-data-status"></span>
        </div>
        <div id="${this.topic}-prep-summary" class="train-facts"></div>
        <details id="${this.topic}-sql-details">
          <summary>generated sql</summary>
          <pre id="${this.topic}-sql-text" class="forecast-sql"></pre>
        </details>
      </div>`;
    // typed text goes in through the property, never the markup: a sql filter carries quotes
    const typed = {
      'censor-asof': this.censorAsOf, 'data-encoding': this.encoding, asof: this.asOf,
      'predict-where': this.predictWhere, where: this.where,
    };
    for (const [suffix, value] of Object.entries(typed)) {
      const input = panel.querySelector(`#${this.topic}-${suffix}`);
      if (input) input.value = value || '';
    }
    this.dataSay = text => setText(`${this.topic}-data-status`, text);
    this.wireData(panel);
    this.renderSetupHeads();
    this.paintDataMode();
    this.restorePrepared();
    return {enter: () => this.enterData()};
  }

  async enterData() {
    if (this.source) setHead(document.getElementById(`${this.topic}-data-source`), baseName(this.source));
    if (this.columnsInfo.length) this.renderColumnGrid();
    this.restorePrepared();
    if (this.source && !this.columnsInfo.length) await this.loadColumns();
  }

  paintDataMode() {
    const timeseries = this.mode() === 'timeseries';
    document.getElementById(`${this.topic}-timeseries-row`).classList.toggle('hidden', !timeseries);
    document.getElementById(`${this.topic}-entity-row`).classList.toggle('hidden', timeseries);
    const heading = document.getElementById(`${this.topic}-entity-heading`);
    if (heading) heading.classList.toggle('hidden', timeseries);
    this.renderColumnGrid();
  }

  wireData(panel) {
    document.getElementById(`${this.topic}-data-source`).onclick = evt => this.openSourcePicker(evt.currentTarget);
    document.getElementById(`${this.topic}-data-encoding`).onchange = evt => {
      this.encoding = evt.currentTarget.value.trim() || 'auto';
      this.detectedEncoding = null;
      evt.currentTarget.value = this.encoding;
      this.saveStored();
      if (this.source) this.loadColumns();
    };
    const asOf = document.getElementById(`${this.topic}-asof`);
    if (asOf) asOf.onchange = () => { this.asOf = asOf.value; this.saveStored(); };
    const predictWhere = document.getElementById(`${this.topic}-predict-where`);
    if (predictWhere) predictWhere.onchange = () => { this.predictWhere = predictWhere.value; this.saveStored(); };
    const where = document.getElementById(`${this.topic}-where`);
    where.onchange = () => { this.where = where.value; this.saveStored(); };
    if (this.task === 'regression') {
      const toggle = document.getElementById(`${this.topic}-censor-toggle`);
      toggle.onclick = () => {
        this.censorOn = !this.censorOn;
        toggle.textContent = this.censorOn ? 'censoring on' : 'censoring off';
        toggle.classList.toggle('on', this.censorOn);
        this.saveStored();
      };
      toggle.classList.toggle('on', this.censorOn);
      toggle.textContent = this.censorOn ? 'censoring on' : 'censoring off';
      const unit = document.getElementById(`${this.topic}-censor-unit`);
      unit.replaceChildren(forecastDropdown(FORECAST_UNITS, this.censorUnit, value => {
        this.censorUnit = value;
        this.saveStored();
      }, 'censor unit'));
      const asof2 = document.getElementById(`${this.topic}-censor-asof`);
      asof2.onchange = () => { this.censorAsOf = asof2.value; this.saveStored(); };
      this.updateCensorAnchor();
    }
    document.getElementById(`${this.topic}-prepare`).onclick = () => this.prepare();
  }

  updateCensorAnchor() {
    const el = document.getElementById(`${this.topic}-censor-anchor`);
    if (!el) return;
    const anchor = Object.entries(this.columnState).find(([, c]) => c.role === 'anchor');
    el.textContent = anchor ? anchor[0] : '-';
  }

  async openSourcePicker(head) {
    try {
      const data = await api('/api/forecast-sources');
      setText(`${this.topic}-data-root`, `root: ${data.root}`);
      listMenu('source - ' + data.root, data.files.map(f => ({
        id: f.path, label: f.path, on: f.path === this.source,
      })), item => {
        if (this.source !== item.id) {
          this.columnsInfo = [];
          this.sourceRows = null;
          this.detectedEncoding = null;
        }
        this.source = item.id;
        setHead(head, baseName(item.id));
        this.saveStored();
        this.loadColumns();
      }, {empty: 'no source files under the forecast root'}).openAt(head);
    } catch (err) {
      this.dataSay(err.message);
    }
  }

  async loadColumns() {
    if (!this.source) return;
    const source = this.source;
    const encoding = this.encoding;
    const mode = this.mode();
    this.dataSay('reading columns...');
    try {
      const res = await api('/api/forecast-columns', {
        source, encoding, mode,
      });
      if (source !== this.source || encoding !== this.encoding || mode !== this.mode()) return;
      this.columnsInfo = res.columns;
      this.sourceRows = res.row_count;
      this.detectedEncoding = res.encoding;
      // entity roles survive flavour changes; timeseries picks have their own saved state
      const roles = FORECAST_ENTITY_ROLES;
      const next = {};
      this.columnsInfo.forEach(col => {
        const prior = this.columnState[col.name];
        const suggested = col.kind === 'date' ? 'anchor' : col.suggested_role;
        const role = prior && roles.includes(prior.role) ? prior.role : suggested;
        next[col.name] = {
          role,
          aggregation: (prior && prior.aggregation) || FORECAST_AGGREGATIONS[0],
          known: prior ? prior.known !== false : true,
        };
      });
      this.reconcileTimeseriesSetup();
      this.columnState = next;
      this.renderColumnGrid();
      this.updateCensorAnchor();
      const detected = this.encoding === 'auto' ? `, detected ${res.encoding}` : '';
      this.dataSay(`${this.columnsInfo.length} columns${detected}`);
      this.saveStored();
    } catch (err) {
      this.dataSay(err.message);
    }
  }

  reconcileTimeseriesSetup() {
    const fields = kind => this.columnsInfo.filter(col => col.kind === kind).map(col => col.name);
    const measures = fields('number');
    const dates = fields('date');
    const prior = this.seriesSetup || {
      targets: measures.filter(name => this.columnState[name]?.role === 'target'),
      aggregations: Object.fromEntries(measures.map(name => [name, this.columnState[name]?.aggregation])),
      time: dates.find(name => this.columnState[name]?.role === 'time') || dates[0],
      dimensions: [],
    };
    const targets = prior.targets.filter(name => measures.includes(name));
    const time = dates.includes(prior.time) ? prior.time : (dates[0] || null);
    const dimensions = this.scaffoldFields({targets, time}).map(col => col.name);
    this.seriesSetup = {
      targets,
      aggregations: Object.fromEntries(targets.map(name => [name,
        FORECAST_AGGREGATIONS.includes(prior.aggregations[name]) ? prior.aggregations[name] : 'sum'])),
      time,
      dimensions: prior.dimensions.filter(name => dimensions.includes(name)),
    };
    if (!FORECAST_UNITS.includes(this.step)) this.step = 'week';
    this.paintSetupHeads();
  }

  countSeries(dimensions) {
    return dimensions.reduce((product, name) => product *
      (this.columnsInfo.find(col => col.name === name)?.distinct || 0), 1);
  }

  scaffoldFields(setup) {
    return this.columnsInfo.filter(col => col.name !== setup.time &&
      !setup.targets.includes(col.name) && col.distinct !== this.sourceRows)
      .sort((a, b) => (a.kind !== 'text') - (b.kind !== 'text'));
  }

  seriesWarning(dimensions) {
    return this.countSeries(dimensions) > 500 ? ' · warning: more than 500 series' : '';
  }

  renderSetupHeads() {
    const box = document.getElementById(`${this.topic}-setup-heads`);
    box.replaceChildren();
    [['predict', 'predict'], ['aggregate', 'aggregate'], ['time', 'extend in time'],
      ['scaffold', 'scaffold along']].forEach(([key, label]) => {
      const slot = forecastNode('div', 'forecast-setup-slot');
      const head = forecastNode('button', 'toggle dropdown-head');
      head.type = 'button';
      head.id = `${this.topic}-setup-${key}`;
      head.append(forecastNode('span'));
      head.onclick = () => this.openSetupPicker(key, head);
      slot.append(forecastNode('div', 'field-label', label), head);
      box.append(slot);
    });
    this.paintSetupHeads();
  }

  paintSetupHeads() {
    const setup = this.seriesSetup || {targets: [], aggregations: {}, time: null, dimensions: []};
    const aggregate = setup.targets.map(name => `${setup.aggregations[name]}(${name})`).join(', ');
    const series = this.countSeries(setup.dimensions).toLocaleString('en');
    const captions = {
      predict: setup.targets.join(', ') || 'choose variables',
      aggregate: aggregate || 'predict something first',
      time: `${setup.time || 'choose date'} · ${this.step} · +${this.horizon}`,
      scaffold: setup.dimensions.length ? `${setup.dimensions.join(' x ')} · ${series}` : 'none (one total series)',
    };
    Object.entries(captions).forEach(([key, text]) => {
      const head = document.getElementById(`${this.topic}-setup-${key}`);
      if (head) { setHead(head, text); head.title = text; }
    });
    setText(`${this.topic}-setup-readout`, `${aggregate || 'nothing yet'} · ${setup.time || 'no date'}, ` +
      `${this.horizon} ${this.step}${this.horizon === 1 ? '' : 's'} ahead · ` +
      `${setup.dimensions.join(' x ') || 'total'} = ${series} series${this.seriesWarning(setup.dimensions)}`);
    document.getElementById(`${this.topic}-setup-readout`)?.classList.toggle('warn', this.countSeries(setup.dimensions) > 500);
  }

  openSetupPicker(key, head) {
    const saved = this.seriesSetup || {targets: [], aggregations: {}, time: null, dimensions: []};
    const draft = {
      ...saved, targets: [...saved.targets], aggregations: {...saved.aggregations},
      dimensions: [...saved.dimensions], step: this.step === 'auto' ? 'week' : this.step,
      horizon: this.horizon,
    };
    const fields = kind => this.columnsInfo.filter(col => col.kind === kind);
    const targets = fields('number').filter(col => !draft.dimensions.includes(col.name));
    const footer = {kind: 'buttons', buttons: [
      {id: 'save', label: 'save', onClick: menu => {
        this.seriesSetup = {
          targets: draft.targets,
          aggregations: Object.fromEntries(draft.targets.map(name => [name, draft.aggregations[name] || 'sum'])),
          time: draft.time, dimensions: draft.dimensions,
        };
        this.step = draft.step;
        this.horizon = draft.horizon;
        this.saveStored();
        this.paintSetupHeads();
        menu.close();
      }},
      {id: 'close', label: 'close', onClick: menu => menu.close()},
    ]};
    let title, sections;
    if (key === 'predict') {
      title = 'variables to predict';
      sections = [{kind: 'list', multi: true, empty: 'no numeric fields',
        items: targets.map(col => ({id: col.name, label: col.name, on: draft.targets.includes(col.name)})),
        onPick: (item, on) => {
          const picks = on ? [...draft.targets, item.id] : draft.targets.filter(name => name !== item.id);
          draft.targets = targets.map(col => col.name).filter(name => picks.includes(name));
          if (on && !draft.aggregations[item.id]) draft.aggregations[item.id] = 'sum';
        },
      }];
    } else if (key === 'aggregate') {
      title = 'aggregation per variable';
      const pane = forecastNode('div', 'forecast-setup-pane');
      if (!draft.targets.length) pane.append(forecastNode('span', 'none', 'pick variables to predict in the first dropdown'));
      draft.targets.forEach(name => {
        const row = forecastNode('div', 'forecast-agg-row');
        const field = forecastNode('div', 'toggle forecast-field');
        field.append(forecastNode('span', 'name', name));
        row.append(field, forecastInlineSelect(FORECAST_AGGREGATIONS, draft.aggregations[name], value => {
          draft.aggregations[name] = value;
        }, `aggregation for ${name}`));
        pane.append(row);
      });
      sections = [{kind: 'node', node: pane}];
    } else if (key === 'time') {
      title = 'time to extend';
      const right = forecastNode('div', 'forecast-setup-pane forecast-time-side');
      const aheadLabel = forecastNode('div', 'field-label', `${draft.step}s ahead`);
      const granularity = forecastInlineSelect(FORECAST_UNITS, draft.step, value => {
        draft.step = value;
        aheadLabel.textContent = `${value}s ahead`;
      }, 'date granularity');
      const ahead = forecastNode('input', 'text-field');
      ahead.type = 'number';
      ahead.min = '1';
      ahead.max = '366';
      ahead.value = draft.horizon;
      ahead.setAttribute('aria-label', 'steps into the future');
      ahead.oninput = () => { draft.horizon = Math.max(1, Math.min(366, Math.round(+ahead.value) || 1)); };
      ahead.onkeydown = evt => { if (evt.key !== 'Escape') evt.stopPropagation(); };
      right.append(forecastNode('div', 'field-label', 'date granularity'), granularity, aheadLabel, ahead);
      sections = [{kind: 'list', multi: true, label: 'date field', empty: 'no date fields',
        items: fields('date').map(col => ({id: col.name, label: col.name, on: draft.time === col.name})),
        onPick: (item, on, menu) => {
          draft.time = item.id;
          draft.dimensions = draft.dimensions.filter(name => name !== item.id);
          menu.el.querySelectorAll('.menu-list .menu-item').forEach(row => row.classList.toggle('on', row.dataset.id === item.id));
        },
      }, {kind: 'node', node: right}];
    } else {
      title = 'dimensions to scaffold along';
      const dimensions = this.scaffoldFields(draft);
      const note = forecastNode('div', 'stat forecast-series-product');
      const paint = () => {
        note.textContent = draft.dimensions.length ?
          `${draft.dimensions.map(name => this.columnsInfo.find(col => col.name === name).distinct).join(' x ')} = ` +
          `${this.countSeries(draft.dimensions).toLocaleString('en')} series${this.seriesWarning(draft.dimensions)}` : '1 series (the total)';
        note.classList.toggle('warn', this.countSeries(draft.dimensions) > 500);
      };
      paint();
      sections = [{kind: 'list', multi: true, empty: 'no dimensions',
        items: dimensions.map(col => ({id: col.name, label: col.name, stats: `[${col.distinct}]`, on: draft.dimensions.includes(col.name)})),
        onPick: (item, on) => {
          const picks = on ? [...draft.dimensions, item.id] : draft.dimensions.filter(name => name !== item.id);
          draft.dimensions = dimensions.map(col => col.name).filter(name => picks.includes(name));
          paint();
        },
      }, {kind: 'node', node: note}];
    }
    const menu = new ForecastSetupMenu(key, {
      title, columns: key === 'time', sections: [...sections, footer], onDismiss: () => head.focus(),
    });
    menu.openAt(head);
  }

  renderColumnGrid() {
    const box = document.getElementById(`${this.topic}-data-columns`);
    box.innerHTML = '';
    if (!this.columnsInfo.length || this.mode() === 'timeseries') return;
    const table = document.createElement('table');
    table.className = 'forecast-column-grid';
    const head = document.createElement('tr');
    ['name', 'kind', 'samples', 'role', 'known when predicting']
      .forEach(label => {
        const th = document.createElement('th');
        th.textContent = label;
        head.appendChild(th);
      });
    table.appendChild(head);
    this.columnsInfo.forEach(col => {
      const state = this.columnState[col.name] || {role: col.suggested_role, aggregation: 'sum', known: true};
      const row = document.createElement('tr');
      const nameCell = document.createElement('td');
      nameCell.textContent = col.name;
      nameCell.title = col.name;
      const kindCell = document.createElement('td');
      kindCell.textContent = col.kind;
      const samplesCell = document.createElement('td');
      samplesCell.textContent = col.sample.join(', ');
      samplesCell.title = samplesCell.textContent;
      const roleCell = document.createElement('td');
      const roleHead = forecastDropdown(FORECAST_ENTITY_ROLES, state.role, role => {
        this.columnState[col.name] = {...this.columnState[col.name], role};
        this.updateCensorAnchor();
        this.renderColumnGrid();
        this.saveStored();
      }, `role for ${col.name}`);
      roleCell.appendChild(roleHead);
      const lastCell = document.createElement('td');
      const known = state.known !== false;
      const enabled = ['dimension', 'measure'].includes(state.role);
      const check = forecastNode('button', 'toggle' + (known ? ' on' : '') +
        (enabled ? '' : ' disabled'), known ? 'known' : 'not known');
      check.type = 'button';
      check.setAttribute('role', 'checkbox');
      check.setAttribute('aria-checked', String(known));
      check.disabled = !enabled;
      check.onclick = () => {
        this.columnState[col.name] = {...this.columnState[col.name], known: !known};
        this.renderColumnGrid();
        this.saveStored();
      };
      lastCell.appendChild(check);
      row.append(nameCell, kindCell, samplesCell, roleCell, lastCell);
      table.appendChild(row);
    });
    box.appendChild(table);
  }

  buildSpec() {
    const mode = this.mode();
    const timeseries = mode === 'timeseries';
    const setup = this.seriesSetup;
    const columns = timeseries ? this.columnsInfo.map(col => ({
      name: col.name,
      role: setup?.targets.includes(col.name) ? 'target' : setup?.time === col.name ? 'time' :
        setup?.dimensions.includes(col.name) ? 'dimension' : 'ignore',
      aggregation: setup?.targets.includes(col.name) ? setup.aggregations[col.name] : null,
      known: true,
    })) : Object.entries(this.columnState).map(([name, state]) => ({
      name,
      role: state.role,
      aggregation: null,
      known: state.known !== false,
    }));
    const anchorEntry = columns.find(c => c.role === 'anchor');
    return {
      source: this.source,
      mode,
      task: this.task,
      columns,
      encoding: this.encoding || 'auto',
      step: timeseries ? (this.step === 'auto' ? null : this.step) : null,
      horizon: timeseries ? this.horizon : 8,
      where: this.where || null,
      predict_where: !timeseries ? (this.predictWhere || null) : null,
      censor: (!timeseries && this.task === 'regression' && this.censorOn && anchorEntry) ? {
        anchor: anchorEntry.name, unit: this.censorUnit, as_of: this.censorAsOf || null,
      } : null,
      as_of: timeseries ? (this.asOf || null) : null,
    };
  }

  async prepare() {
    this.dataSay('preparing...');
    try {
      const spec = this.currentSpec();
      const res = await api('/api/forecast-prep', {spec});
      this.rememberPrepared(spec, res);
      this.dataSay(res.reused ? 'prepared (reused cache)' : 'prepared');
    } catch (err) {
      this.dataSay(err.message);
    }
  }

  paintPrepSummary(res) {
    const box = document.getElementById(`${this.topic}-prep-summary`);
    box.innerHTML = '';
    const summary = res.summary || {};
    Object.entries(summary).forEach(([k, v]) => {
      const key = document.createElement('span');
      key.className = 'k';
      key.textContent = k;
      key.title = k;
      const val = document.createElement('span');
      val.className = 'v';
      val.textContent = String(v);
      val.title = String(v);
      box.append(key, val);
    });
    setText(`${this.topic}-sql-text`, res.sql || '');
  }

  // ================================================================== search tab

  mountSearch(panel) {
    const budgetRows = [
      {key: 'population', label: 'population', step: 2, min: 2, max: 400, value: this.searchValues.population},
      {key: 'plateau', label: 'plateau generations', step: 1, min: 1, max: 100, value: this.searchValues.plateau},
      {key: 'generations', label: 'max generations', step: 1, min: 1, max: 500, value: this.searchValues.generations},
      {key: 'minutes', label: 'time cap (minutes)', step: 1, min: 1, max: 240, value: this.searchValues.minutes},
    ];
    panel.innerHTML = `
      <div id="${this.topic}-search-metrics"></div>
      <div class="forecast-search-layout">
        <aside class="panel-inline forecast-run-rail">
          <div class="forecast-run-controls">
            <button class="toggle adds" id="${this.topic}-new-search">+ new run</button>
            <div class="forecast-run-sort" id="${this.topic}-run-sort"></div>
          </div>
          <section id="${this.topic}-running-section" class="hidden">
            <div class="field-label">running</div>
            <div id="${this.topic}-running-list"></div>
          </section>
          <section><div class="field-label">runs</div><div id="${this.topic}-runs-list"></div></section>
        </aside>
        <div class="forecast-pane forecast-run-detail">
        <div class="panel-inline forecast-pane">
          <div class="run-controls">
            <span class="field-value" id="${this.topic}-run-title">pick a run or start one on models</span>
            <span class="forecast-verdict hidden" id="${this.topic}-run-verdict"></span>
            <button class="toggle removes disabled" id="${this.topic}-search-cancel" disabled>cancel</button>
          </div>
          <div class="stat" id="${this.topic}-search-status">idle</div>
          <div id="${this.topic}-run-progress" class="hidden">
            <div class="progress-row"><div class="bar"><div class="bar-fill" id="${this.topic}-run-bar-fill"></div></div></div>
            <div class="stat" id="${this.topic}-run-progress-label"></div>
          </div>
          <div class="field-label" id="${this.topic}-run-stop"></div>
          <div class="field-value" id="${this.topic}-run-stats"></div>
          <div class="forecast-leaderboard-wrap hidden" id="${this.topic}-family-progress-wrap">
            <table class="forecast-column-grid" id="${this.topic}-family-progress"></table>
          </div>
          <div class="hidden" id="${this.topic}-warm-budget">
            <div class="field-label">warm start · budget</div>
            <div id="${this.topic}-budget">${budgetRows.map(r => forecastStepperRow(`${this.topic}-budget`, r)).join('')}</div>
          </div>
          <div class="run-controls hidden" id="${this.topic}-run-actions">
            <button class="toggle" id="${this.topic}-run-warm">warm start</button>
            <button class="toggle" id="${this.topic}-run-refit">refit</button>
            <button class="toggle" id="${this.topic}-run-open">open</button>
          </div>
        </div>
        <div class="run-controls hidden" id="${this.topic}-run-view" role="group" aria-label="candidate view">
          <button type="button" class="toggle on" id="${this.topic}-run-view-tiles" aria-pressed="true">tiles</button>
          <button type="button" class="toggle" id="${this.topic}-run-view-tables" aria-pressed="false">tables</button>
        </div>
        <section class="forecast-tiles hidden" id="${this.topic}-tiles" aria-label="model family candidates">
          <div class="forecast-tiles-bar">
            <div class="forecast-tiles-legend">
              <span id="${this.topic}-tiles-count"></span>
              <span><i data-state="better"></i>better</span>
              <span><i data-state="running"></i>running</span>
              <span><i data-state="worse"></i>worse</span>
            </div>
            <button type="button" class="toggle" id="${this.topic}-tiles-flip" aria-pressed="false">flip all: folds</button>
          </div>
          <div class="forecast-tiles-grid" id="${this.topic}-tiles-grid"></div>
        </section>
        <div class="forecast-pane" id="${this.topic}-run-tables">
        <div class="run-controls hidden" id="${this.topic}-generation-family-wrap">
          <span class="field-label">model family</span>
          <button class="toggle dropdown-head" id="${this.topic}-generation-family"><span>xgboost</span></button>
        </div>
        <div class="forecast-search-charts">
        <div id="${this.topic}-search-progress" class="hidden">
          <div class="field-label" id="${this.topic}-search-metric"></div>
          <div id="${this.topic}-search-chart" class="forecast-chart"></div>
          <div class="field-label">generation</div>
        </div>
        <div id="${this.topic}-search-loss" class="hidden">
          <div class="field-label" id="${this.topic}-loss-metric"></div>
          <div id="${this.topic}-loss-chart" class="forecast-chart forecast-loss-chart"></div>
          <div class="chart-legend">
            <div class="chart-legend-item"><span class="chart-legend-swatch" style="border-color:var(--kingfisher)"></span>current generation</div>
            <div class="chart-legend-item"><span class="chart-legend-swatch" style="border-color:var(--cream)"></span>previous generations</div>
          </div>
          <div class="stat hidden" id="${this.topic}-loss-hidden"></div>
          <div class="field-label">boosting round</div>
        </div>
        </div>
        <div id="${this.topic}-leaderboard" class="hidden">
          <div class="forecast-leaderboard-wrap">
            <table class="forecast-column-grid forecast-leaderboard" id="${this.topic}-leaderboard-table"></table>
          </div>
        </div>
        </div>
        </div>
      </div>`;
    forecastWireSteppers(panel, `${this.topic}-budget`, budgetRows, this.searchValues, () => {});
    for (const view of ['tiles', 'tables']) {
      document.getElementById(`${this.topic}-run-view-${view}`).onclick = () => {
        this.runView = view;
        this.paintRunDetail(this.runCache.get(this.selectedRunId));
        if (view === 'tables') window.dispatchEvent(new Event('resize'));
      };
    }
    document.getElementById(`${this.topic}-tiles-flip`).onclick = () => {
      this.tilesFlipped = !this.tilesFlipped;
      this.paintTileFlip();
    };
    document.getElementById(`${this.topic}-search-cancel`).onclick = () => this.cancelSearch();
    document.getElementById(`${this.topic}-new-search`).onclick = () => activateTab(`${this.topic}-models`);
    document.getElementById(`${this.topic}-run-sort`).append(forecastDropdown(
      ['date, newest first', 'best performance'], this.runOrder,
      value => { this.runOrder = value; this.paintRuns(); }, 'sort runs'
    ));
    document.getElementById(`${this.topic}-run-warm`).onclick = () => this.warmStartRun(this.selectedRunId);
    document.getElementById(`${this.topic}-run-refit`).onclick = () => this.refitRun(this.selectedRunId);
    document.getElementById(`${this.topic}-run-open`).onclick = () => this.openResults(this.selectedRunId);
    document.getElementById(`${this.topic}-generation-family`).onclick = evt => {
      const run = this.runCache.get(this.selectedRunId);
      listMenu('model family', this.generationFamilies(run).map(name => ({
        id: name, label: name, on: name === this.generationFamily,
      })), item => {
        this.generationFamily = item.id;
        this.resetSearchChart();
        this.paintRunDetail(run);
      }).openAt(evt.currentTarget);
    };
    return {enter: () => this.enterSearch()};
  }

  async enterSearch() {
    this.mountMetricPicker('search');
    await this.loadRuns();
    if (!this.searchEntered) {
      this.selectedRunId = this.selectedRunId || this.currentRunId || [...this.runCache.keys()].sort().at(-1) || null;
      this.searchEntered = true;
    }
    await this.selectRun(this.selectedRunId);
    this.scheduleSearchPoll();
  }

  searchSay(text) {
    setText(`${this.topic}-search-status`, text);
    document.getElementById(`${this.topic}-search-status`).classList.toggle('hidden', !text);
  }

  async cancelSearch() {
    if (!this.currentRunId) return;
    try {
      await api('/api/forecast-cancel', {id: this.currentRunId});
      this.searchSay('cancelling...');
    } catch (err) {
      this.searchSay(err.message);
    }
  }

  async pollSearch() {
    try {
      const live = [...this.runCache.values()].filter(run => this.isRunning(run));
      await Promise.all(live.map(run => this.readRun(run)));
    } catch (err) {
      this.searchSay(err.message);
    }
    this.scheduleSearchPoll();
  }

  scheduleSearchPoll() {
    clearTimeout(this.pollTimer);
    this.pollTimer = [...this.runCache.values()].some(run => this.isRunning(run))
      ? setTimeout(() => this.pollSearch(), 1000) : null;
  }

  isRunning(run) { return ['starting', 'queued', 'running'].includes(run.state); }

  attachRun(id, kind) {
    this.currentRunId = id;
    this.runCache.set(id, {id, kind, state: 'starting', curves: [], cursor: -1, boardCursor: -1});
    this.selectedRunId = id;
    this.resetSearchChart();
    this.paintRuns();
    this.paintRunDetail(this.runCache.get(id));
    this.readRun(this.runCache.get(id)).catch(err => this.searchSay(err.message));
    this.scheduleSearchPoll();
  }

  async readRun(run) {
    const params = new URLSearchParams({id: run.id,
      since_events: String(run.events_cursor ?? run.generations?.at(-1)?.generation ?? -1),
      since_generation: String(run.cursor), since_leaderboard: String(run.boardCursor)});
    const state = await api(`/api/forecast-run?${params}`);
    if (this.runCache.get(run.id) !== run) return;
    const curves = [...run.curves];
    let cursor = run.cursor;
    for (const curve of state.generation_curves || []) {
      if (!Number.isInteger(curve.generation) || curve.generation <= run.cursor) continue;
      if (!curve.metric || !Array.isArray(curve.validation) || !curve.validation.length
          || !curve.validation.every(Number.isFinite)) continue;
      if (!curves.some(entry => entry.generation === curve.generation)) curves.push(curve);
      cursor = Math.max(cursor, curve.generation);
    }
    const generations = [...(run.generations || [])];
    for (const entry of state.generations || []) {
      if (!generations.some(saved => (entry.sequence !== undefined
        ? saved.sequence === entry.sequence
        : saved.generation === entry.generation && forecastModelFamily(saved) === forecastModelFamily(entry)))) generations.push(entry);
    }
    const familyCounts = new Map();
    const recentCurves = curves.sort((a, b) => b.generation - a.generation).filter(curve => {
      const family = forecastModelFamily(curve);
      const count = (familyCounts.get(family) || 0) + 1;
      familyCounts.set(family, count);
      return count <= 60;
    }).reverse();
    Object.assign(run, state, {generations, curves: recentCurves, cursor});
    if (Number.isInteger(state.leaderboard_generation)) run.boardCursor = state.leaderboard_generation;
    if (Number.isInteger(state.events_cursor) && state.latest?.sequence > state.events_cursor) {
      return this.readRun(run);
    }
    if (this.currentRunId === run.id && !this.isRunning(run)) this.currentRunId = null;
    this.paintRuns();
    if (this.selectedRunId === run.id) this.paintRunDetail(run);
    this.readLegacyTiles(run);
  }

  async readLegacyTiles(run) {
    if (run.version !== 1 || run.state !== 'done' || this.selectedRunId !== run.id
        || run.tiles || run.tilesRequested) return;
    run.tilesRequested = true;
    try {
      const view = await api(`/api/forecast-view?id=${encodeURIComponent(run.id)}`);
      if (this.runCache.get(run.id) !== run) return;
      run.tiles = view.tiles;
      if (this.selectedRunId === run.id) this.paintRunDetail(run);
    } catch (error) {
      // legacy runs can outlive their prepared data; keep the available poll metadata
      run.tilesError = error.message;
    }
  }

  async selectRun(id) {
    if (this.selectedRunId !== id) this.generationFamily = null;
    this.selectedRunId = id;
    this.resetSearchChart();
    this.paintRuns();
    const run = this.runCache.get(id);
    this.paintRunDetail(run);
    if (run) await this.readRun(run);
  }

  stopReason(run) {
    if (run.state === 'failed') return `failed: ${run.reason || 'unknown error'}`;
    const reason = run.stopped || run.latest?.stopped;
    if (reason === 'plateau') return `no gain larger than the noise for ${run.budget?.plateau ?? 5} generations in a row`;
    return {time_cap: 'time cap', max_generations: 'generation cap', generation_cap: 'generation cap',
      cancelled: 'cancelled'}[reason] || (reason || (this.isRunning(run) ? 'search in progress' : '-'));
  }

  paintRunDetail(run) {
    this.acceptMetricRegistry(run, 'search');
    this.paintRunTiles(run);
    this.paintFamilyProgress(run);
    const progressWrap = document.getElementById(`${this.topic}-run-progress`);
    progressWrap.classList.toggle('hidden', !run);
    if (run) {
      const progress = forecastSearchProgress(run);
      const fill = document.getElementById(`${this.topic}-run-bar-fill`);
      fill.style.width = `${progress.fill * 100}%`;
      fill.classList.toggle('good', progress.done);
      setText(`${this.topic}-run-progress-label`, progress.label);
    }
    document.getElementById(`${this.topic}-search-status`).classList.toggle('hidden', run?.state === 'done');
    document.getElementById(`${this.topic}-warm-budget`).classList.toggle('hidden', !run);
    document.getElementById(`${this.topic}-run-actions`).classList.toggle('hidden', !run);
    const cancel = document.getElementById(`${this.topic}-search-cancel`);
    cancel.disabled = !this.currentRunId;
    cancel.classList.toggle('disabled', cancel.disabled);
    const verdict = document.getElementById(`${this.topic}-run-verdict`);
    const ratioMetric = this.selectedMetrics(run).find(metric => metric.id === 'test_ratio');
    verdict.classList.toggle('hidden', !run?.verdict || !ratioMetric);
    if (!run) {
      setText(`${this.topic}-run-title`, 'pick a run or start one on models');
      setText(`${this.topic}-run-stop`, '');
      setText(`${this.topic}-run-stats`, '');
      this.searchSay('idle');
      this.paintLeaderboard([]);
      return;
    }
    this.displayTask = run.spec?.task || run.task || this.task;
    setText(`${this.topic}-run-title`, `${run.id.split('-').at(-1)} · ${run.kind || 'search'} · ${run.state}`);
    setText(`${this.topic}-run-stop`, this.stopReason(run));
    if (run.verdict && ratioMetric) {
      verdict.replaceChildren(forecastMetricValue(ratioMetric, run, 'span', true));
      forecastRatioStyle(verdict, run.verdict);
    }
    const names = this.generationFamilies(run);
    if (!names.includes(this.generationFamily)) this.generationFamily = names[0] || 'xgboost';
    setHead(document.getElementById(`${this.topic}-generation-family`), this.generationFamily);
    document.getElementById(`${this.topic}-generation-family-wrap`).classList.toggle('hidden', !run.families || run.version === 1);
    const generations = (run.generations || []).filter(entry => forecastModelFamily(entry) === this.generationFamily);
    const latest = generations.at(-1);
    const stats = `elapsed ${forecastNumber(run.elapsed ?? latest?.elapsed)}s · generations ${generations.length}`;
    setText(`${this.topic}-run-stats`, stats);
    this.searchSay(this.isRunning(run)
      ? latest ? `generation ${latest.generation} - evaluated ${latest.evaluated} - elapsed ${latest.elapsed}s` : run.state
      : run.state === 'failed' ? this.stopReason(run) : '');
    const chartVersion = `${run.id}:${this.generationFamily}:${generations.length}:${run.cursor}`;
    if (this.paintedChartVersion !== chartVersion) {
      this.paintSearchChart(generations.map(entry => ({...entry, generation: entry.family_generation ?? entry.generation})));
      this.lossCurves = [];
      this.lossGeneration = -1;
      this.paintLossChart((run.curves || []).filter(entry => forecastModelFamily(entry) === this.generationFamily)
        .map(entry => ({...entry, generation: entry.family_generation ?? entry.generation})));
      this.paintedChartVersion = chartVersion;
    }
    const boardVersion = JSON.stringify([run.id, run.boardCursor, run.leaderboard?.length || 0, this.selectedMetrics(run)]);
    if (this.paintedBoardVersion !== boardVersion) {
      this.paintLeaderboard(run.leaderboard || []);
      this.paintedBoardVersion = boardVersion;
    }
  }

  generationFamilies(run) {
    return [...new Set([...Object.keys(run?.families || {}),
      ...(run?.generations || []).map(forecastModelFamily), ...(run?.curves || []).map(forecastModelFamily)])];
  }

  paintRunTiles(run) {
    const grid = document.getElementById(`${this.topic}-tiles-grid`);
    if (this.tilesRunId !== (run?.id || null)) {
      this.tilesRunId = run?.id || null;
      this.runView = 'tiles';
      this.tilesFlipped = false;
      grid.replaceChildren();
    }
    document.getElementById(`${this.topic}-run-view`).classList.toggle('hidden', !run);
    document.getElementById(`${this.topic}-tiles`).classList.toggle('hidden', !run || this.runView !== 'tiles');
    document.getElementById(`${this.topic}-run-tables`).classList.toggle('hidden', !!run && this.runView !== 'tables');
    for (const view of ['tiles', 'tables']) {
      const button = document.getElementById(`${this.topic}-run-view-${view}`);
      button.classList.toggle('on', this.runView === view);
      button.setAttribute('aria-pressed', String(this.runView === view));
    }
    const entries = forecastTileEntries(run);
    const scale = forecastTileWapeScale(entries);
    const metrics = this.selectedMetrics(run);
    const registry = run?.metric_registry || this.metricRegistry;
    const searchScale = forecastTileSearchScale(entries, null, registry);
    const roundMetric = forecastTileRoundMetric(this.selectedMetricIds, registry);
    const rounds = {metric: roundMetric, scale: forecastTileRoundScale(entries, roundMetric),
      size: forecastTileSize(entries, metrics)};
    grid.style.setProperty('--tile-size', `${rounds.size}px`);
    setText(`${this.topic}-tiles-count`, `candidates · ${entries.length}`);
    const existing = new Map([...grid.children].map(node => [node.dataset.family, node]));
    entries.forEach((tile, index) => {
      let node = existing.get(tile.name);
      const signature = JSON.stringify([tile, index, metrics, scale, searchScale, rounds]);
      if (!node || node.dataset.signature !== signature) {
        const openMenus = [...(node?.querySelectorAll('details[open]') || [])].map(details => ({
          side: details.closest('.forecast-tile-back') ? 'back' : 'front',
          key: details.dataset.menu, scroll: details.querySelector('.forecast-tile-menu').scrollTop,
        }));
        const active = node?.contains(document.activeElement) ? document.activeElement.closest('details') : null;
        const focus = active ? {
          side: active.closest('.forecast-tile-back') ? 'back' : 'front', key: active.dataset.menu,
        } : null;
        const fresh = forecastCreateTile(tile, index, metrics, scale, searchScale, rounds);
        fresh.dataset.signature = signature;
        if (node) node.replaceWith(fresh);
        else grid.append(fresh);
        node = fresh;
        openMenus.forEach(({side, key, scroll}) => {
          const details = node.querySelector(`.forecast-tile-${side} details[data-menu="${key}"]`);
          details.open = true;
          details.querySelector('.forecast-tile-menu').scrollTop = scroll;
        });
        if (focus) node.querySelector(`.forecast-tile-${focus.side} details[data-menu="${focus.key}"] summary`).focus({preventScroll: true});
      }
      if (grid.children[index] !== node) grid.insertBefore(node, grid.children[index] || null);
      existing.delete(tile.name);
    });
    existing.forEach(node => node.remove());
    if (run && !entries.length) grid.append(forecastNode('div', 'forecast-tile-dim', 'family details not saved yet'));
    this.paintTileFlip();
  }

  paintTileFlip() {
    const panel = document.getElementById(`${this.topic}-tiles`);
    panel.classList.toggle('flipped', this.tilesFlipped);
    const button = document.getElementById(`${this.topic}-tiles-flip`);
    button.classList.toggle('on', this.tilesFlipped);
    button.setAttribute('aria-pressed', String(this.tilesFlipped));
    button.textContent = this.tilesFlipped ? 'show charts' : 'flip all: folds';
    panel.querySelectorAll('.forecast-tile-face').forEach(face => {
      const hidden = face.classList.contains('forecast-tile-back') !== this.tilesFlipped;
      face.inert = hidden;
      face.setAttribute('aria-hidden', String(hidden));
    });
  }

  paintFamilyProgress(run) {
    const families = run?.version === 1 ? [] : Object.entries(run?.families || {});
    const metrics = this.selectedMetrics(run);
    document.getElementById(`${this.topic}-family-progress-wrap`).classList.toggle('hidden', !families.length || this.runView !== 'tables');
    forecastFillTable(document.getElementById(`${this.topic}-family-progress`),
      ['family', 'state', 'phase', 'candidates done', 'expected remaining', ...metrics.map(forecastMetricHeader), 'reason'],
      families.map(([name, status]) => [name, status.terminated ? `${status.state} (terminated)` : status.state,
        status.phase, status.candidates_done ?? 0, status.candidates_expected ?? 0,
        ...metrics.map(metric => forecastMetricValue(metric, status)), status.pruned_reason || status.reason || '-']));
    if (!run) document.getElementById(`${this.topic}-generation-family-wrap`).classList.add('hidden');
  }

  resetSearchChart() {
    this.paintedChartVersion = null;
    this.paintedBoardVersion = null;
    if (this.searchChart) this.searchChart.destroy();
    this.searchChart = null;
    document.getElementById(`${this.topic}-search-progress`).classList.add('hidden');
    if (this.lossChart) this.lossChart.destroy();
    this.lossChart = null;
    this.lossCurves = [];
    this.lossGeneration = -1;
    document.getElementById(`${this.topic}-search-loss`).classList.add('hidden');
  }

  paintSearchChart(generations) {
    if (!generations.length) return;
    const metric = (this.runCache.get(this.selectedRunId)?.metric_registry || this.metricRegistry)
      .find(entry => entry.id === 'validation_error');
    setText(`${this.topic}-search-metric`, 'validation error (lower is better)');
    document.getElementById(`${this.topic}-search-metric`).title = metric ? forecastMetricHelp(metric) : 'search score on held-out validation windows';
    document.getElementById(`${this.topic}-search-progress`).classList.remove('hidden');
    const el = document.getElementById(`${this.topic}-search-chart`);
    if (!this.searchChart) this.searchChart = timeChart(el, {
      height: 220, yFormat: value => metric?.unit === 'percent'
        ? `${Number((value * 100).toPrecision(3))}%` : forecastNumber(value),
    });
    this.searchChart.update({
      x: generations.map(entry => entry.generation),
      series: [{id: 'validation_error', label: 'validation error',
        values: generations.map(entry => Number.isFinite(entry.best) ? entry.best : null)}],
    });
  }

  paintLossChart(curves) {
    const wrap = document.getElementById(`${this.topic}-search-loss`);
    const previousGeneration = this.lossGeneration;
    let added = false;
    // malformed records must not advance the cursor past later usable curves
    for (const curve of curves) {
      if (!Number.isInteger(curve.generation) || curve.generation <= previousGeneration) continue;
      const usable = curve.metric && Array.isArray(curve.validation) && curve.validation.length
        && curve.validation.every(Number.isFinite);
      if (!usable || this.lossCurves.some(entry => entry.generation === curve.generation)) continue;
      this.lossCurves.push(curve);
      this.lossGeneration = Math.max(this.lossGeneration, curve.generation);
      added = true;
    }
    if (!added) return;
    this.lossCurves.sort((a, b) => a.generation - b.generation);
    this.lossCurves = this.lossCurves.slice(-60);
    const history = this.lossCurves[this.lossCurves.length - 1];
    const bucket = history.bucket ? ` - horizon bucket ${history.bucket.join('-')}` : '';
    setText(`${this.topic}-loss-metric`, `${history.metric} (lower is better)${bucket}`);
    wrap.classList.remove('hidden');
    if (!this.lossChart) this.lossChart = timeChart(
      document.getElementById(`${this.topic}-loss-chart`), {
        height: 220, yFormat: value => String(Number(value.toPrecision(4))),
      }
    );
    const style = getComputedStyle(document.documentElement);
    const kingfisher = style.getPropertyValue('--kingfisher').trim();
    const cream = style.getPropertyValue('--cream').trim();
    const filtered = forecastLossSeries(this.lossCurves);
    const hidden = document.getElementById(`${this.topic}-loss-hidden`);
    hidden.textContent = `${filtered.hidden} earlier generations used a different metric and are hidden`;
    hidden.classList.toggle('hidden', !filtered.hidden);
    const rounds = Math.max(...filtered.curves.map(curve => curve.validation.length));
    this.lossChart.update({
      x: Array.from({length: rounds}, (_, index) => index + 1),
      series: filtered.curves.map(curve => {
        const age = history.generation - curve.generation;
        return {
          id: `generation-${curve.generation}`, label: `generation ${curve.generation}`,
          values: Array.from({length: rounds}, (_, index) => curve.validation[index] ?? null),
          color: age === 0 && !curve.repeated ? kingfisher : cream,
          opacity: Math.max(0.2, 1 - age / 60),
        };
      }),
    });
  }

  paintLeaderboard(rows) {
    const wrap = document.getElementById(`${this.topic}-leaderboard`);
    const table = document.getElementById(`${this.topic}-leaderboard-table`);
    table.innerHTML = '';
    wrap.classList.toggle('hidden', !rows.length);
    if (!rows.length) return;
    const entries = rows.slice(0, 10);
    const metrics = this.selectedMetrics(this.runCache.get(this.selectedRunId));
    const integer = value => Number.isInteger(value) ? String(value) : '-';
    const lossHeader = forecastNode('th', '', 'loss (objective)');
    lossHeader.title = FORECAST_LOSS_HELP;
    const columns = ['rank', 'family', 'generation found', ...metrics.map(forecastMetricHeader),
      lossHeader, 'tree depth', 'learning rate (eta)', 'features'];
    const caption = forecastNode('caption', 'field-label',
      'best ten recipes, ranked by validation error; lower is better. recipe wape and mae use saved validation values; test metrics are unavailable');
    table.appendChild(caption);
    const thead = document.createElement('thead');
    const head = document.createElement('tr');
    columns.forEach(c => {
      if (c instanceof Node) { head.appendChild(c); return; }
      const th = document.createElement('th');
      th.textContent = c;
      head.appendChild(th);
    });
    thead.appendChild(head);
    table.appendChild(thead);
    const tbody = document.createElement('tbody');
    entries.forEach((entry, index) => {
      const row = document.createElement('tr');
      row.classList.toggle('on', index === 0);
      const genome = entry.genome || {};
      const params = genome.params || {};
      const count = Array.isArray(genome.families) ? new Set(genome.families).size : entry.families;
      const loss = forecastNode('td', '', FORECAST_OBJECTIVES[genome.objective] || genome.objective || '-');
      loss.title = FORECAST_LOSS_HELP;
      const values = [String(index + 1), forecastModelFamily(entry), integer(entry.generation),
        ...metrics.map(metric => forecastMetricValue(metric, entry)), loss,
        integer(params.max_depth), forecastNumber(params.eta)];
      values.forEach(value => {
        if (value instanceof Node) { row.appendChild(value); return; }
        const td = document.createElement('td');
        td.textContent = value;
        td.title = String(value);
        row.appendChild(td);
      });
      const features = forecastNode('td');
      const head = forecastNode('button', 'toggle dropdown-head', `${integer(count)} features`);
      head.onclick = () => this.openRecipeMenu(head, entry, index + 1);
      features.append(head);
      row.append(features);
      tbody.appendChild(row);
    });
    table.appendChild(tbody);
  }

  async loadRuns() {
    const data = await api('/api/forecast-runs');
    this.acceptMetricRegistry(data, 'search');
    for (const summary of data.runs || []) {
      const cached = this.runCache.get(summary.id);
      if (cached) Object.assign(cached, summary);
      else this.runCache.set(summary.id, {...summary, curves: [], cursor: -1, boardCursor: -1});
    }
    const live = [...this.runCache.values()].find(run => this.isRunning(run));
    this.currentRunId = live?.id || null;
    this.paintRuns();
  }

  paintRuns() {
    const box = document.getElementById(`${this.topic}-runs-list`);
    const running = document.getElementById(`${this.topic}-running-list`);
    box.replaceChildren();
    running.replaceChildren();
    const runs = [...this.runCache.values()].sort((a, b) => {
      if (this.runOrder === 'best performance') {
        const left = Number.isFinite(a.test_error) ? a.test_error : Infinity;
        const right = Number.isFinite(b.test_error) ? b.test_error : Infinity;
        if (left !== right) return left - right;
      }
      return b.id.localeCompare(a.id);
    });
    for (const run of runs) {
      const row = forecastNode('button', 'toggle forecast-run-row' + (run.id === this.selectedRunId ? ' on' : ''));
      row.dataset.runId = run.id;
      const status = this.isRunning(run) ? 'running' : run.state === 'failed' ? 'failed'
        : run.stopped === 'cancelled' ? 'cancelled' : 'done';
      row.append(forecastNode('span', 'name', run.id.split('-').at(-1)));
      row.append(forecastNode('span', 'field-label forecast-run-status', status));
      row.append(forecastMetricLines(this.selectedMetrics(run), run, 'forecast-run-metrics'));
      if (this.isRunning(run) || run.progress) {
        const progress = forecastSearchProgress(run);
        const bar = forecastNode('span', 'bar forecast-run-bar');
        const fill = forecastNode('span', 'bar-fill' + (progress.done ? ' good' : ''));
        fill.style.width = `${progress.fill * 100}%`;
        bar.append(fill);
        row.append(bar);
      }
      row.onclick = () => this.selectRun(run.id).catch(err => this.searchSay(err.message));
      (this.isRunning(run) ? running : box).append(row);
    }
    if (!box.childElementCount) box.append(forecastNode('span', 'stat', 'no finished runs yet'));
    document.getElementById(`${this.topic}-running-section`).classList.toggle('hidden', !running.childElementCount);
    document.getElementById(`${this.topic}-new-search`).classList.toggle('on', !this.selectedRunId);
    document.getElementById(`${this.topic}-search-cancel`).classList.toggle('disabled', !this.currentRunId);
  }

  openRecipeMenu(head, entry, rank) {
    const genome = entry.genome || {};
    const params = genome.params || {};
    const node = forecastNode('div', 'forecast-recipe-details');
    const add = (label, value) => {
      const row = forecastNode('div', 'forecast-recipe-field');
      row.append(forecastNode('span', 'field-label', label), forecastNode('span', 'field-value', value));
      node.append(row);
      return row;
    };
    node.append(forecastNode('div', 'field-label', 'feature families'));
    const groups = forecastFamilyGroups(genome.families);
    if (!groups.length) add('families', '-');
    groups.forEach(([kind, values]) => add(kind, values));
    const family = forecastModelFamily(entry);
    add('model family', family);
    node.append(forecastNode('div', 'h-divider'), forecastNode('div', 'field-label', family === 'xgboost' ? 'tree params' : 'model params'));
    if (family === 'xgboost') {
      [['colsample', 'colsample_bytree'], ['eta', 'eta'], ['lambda', 'reg_lambda'],
        ['depth', 'max_depth'], ['min child weight', 'min_child_weight'], ['subsample', 'subsample']]
        .forEach(([label, key]) => add(label, forecastNumber(params[key] ?? params[label])));
    } else {
      if (!Object.keys(params).length) add('params', 'none');
      Object.entries(params).forEach(([name, value]) => add(name, String(value)));
    }
    add('loss (objective)', FORECAST_OBJECTIVES[genome.objective] || '-').title = FORECAST_LOSS_HELP;
    add('note', typeof entry.note === 'string' && entry.note.trim() ? entry.note : '-');
    node.append(forecastNode('div', 'h-divider'), forecastNode('div', 'field-label', 'metrics'));
    const metrics = this.selectedMetrics(this.runCache.get(this.selectedRunId));
    node.append(forecastMetricLines(metrics, entry));
    new Menu({title: `recipe ${rank}`, persistent: true, sections: [{kind: 'node', node}]}).openAt(head);
  }

  async refitRun(runId) {
    try {
      if (this.source && !this.columnsInfo.length) await this.loadColumns();
      const spec = this.currentSpec();
      const res = await api('/api/forecast-refit', {spec, run_id: runId});
      if (res.prepared) this.rememberPrepared(spec, res.prepared);
      this.attachRun(res.run_id, 'refit');
    } catch (err) {
      this.searchSay(err.message);
    }
  }

  async warmStartRun(runId) {
    const budget = {
      population: this.searchValues.population,
      plateau: this.searchValues.plateau,
      max_generations: this.searchValues.generations,
      time_cap: this.searchValues.minutes * 60,
    };
    try {
      if (this.source && !this.columnsInfo.length) await this.loadColumns();
      const spec = this.currentSpec();
      const res = await api('/api/forecast-runs', {spec, budget, warm_from: runId});
      if (res.prepared) this.rememberPrepared(spec, res.prepared);
      this.attachRun(res.run_id, 'warm start');
    } catch (err) {
      this.searchSay(err.message);
    }
  }

  openResults(runId) {
    this.resultsRunId = runId;
    if (typeof activateTab === 'function') activateTab(`${this.topic}-results`);
    this.loadResults();
  }

  // ================================================================== results tab

  mountResults(panel) {
    panel.innerHTML = `
      <div class="forecast-pane forecast-results">
        <div id="${this.topic}-results-metrics"></div>
        <div id="${this.topic}-horizon-controls" class="forecast-metric-controls forecast-horizon-controls hidden"></div>
        <div id="${this.topic}-horizon-help" class="stat hidden"></div>
        <div class="run-controls">
          <div class="toggle dropdown-head" id="${this.topic}-results-run"><span>pick a run</span></div>
          <button class="toggle dropdown-head hidden" id="${this.topic}-results-model"><span>winner</span></button>
          <span class="field-label">break down by</span>
          <div class="toggle dropdown-head narrow" id="${this.topic}-breakdown-head"><span>none</span></div>
          <span class="spacer"></span>
          <a class="toggle adds hidden" id="${this.topic}-export" download>export csv</a>
        </div>
        <div id="${this.topic}-excluded-series" class="stat warn forecast-excluded-series hidden"></div>
        <div id="${this.topic}-verdict" class="forecast-verdict hidden"></div>
        <div class="forecast-results-models hidden" id="${this.topic}-results-models">
          <div class="forecast-pane hidden" id="${this.topic}-family-comparison-wrap">
            <div class="run-controls"><span class="field-label">family comparison · validation error ranks models</span></div>
            <div class="forecast-model-table-wrap"><table class="forecast-column-grid forecast-family-table" id="${this.topic}-family-comparison"></table></div>
          </div>
          <div class="forecast-pane">
            <div class="run-controls">
              <span class="field-label">feature importance</span>
              <button class="toggle dropdown-head" id="${this.topic}-importance-family"><span>xgboost</span></button>
            </div>
            <div class="stat" id="${this.topic}-importance-status"></div>
            <div class="forecast-model-table-wrap"><table class="forecast-column-grid forecast-family-table" id="${this.topic}-importance"></table></div>
          </div>
        </div>
        <div id="${this.topic}-filter-chips" class="label-columns"></div>
        <div id="${this.topic}-chart-status" class="stat hidden"></div>
        <div id="${this.topic}-chart" class="forecast-chart"></div>
        <div id="${this.topic}-chart-caption" class="stat hidden"></div>
        <div id="${this.topic}-view-metrics" class="forecast-metric-lines hidden"></div>
        <div id="${this.topic}-lead-errors-wrap" class="hidden">
          <div class="field-label">error by weeks ahead</div>
          <div class="forecast-model-table-wrap"><table id="${this.topic}-lead-errors" class="forecast-column-grid"></table></div>
        </div>
        <div class="run-controls hidden" id="${this.topic}-group-row">
          <span class="field-label">group by</span>
          <div class="toggle dropdown-head narrow" id="${this.topic}-group-head"><span>none</span></div>
        </div>
        <div class="forecast-table-wrap">
          <table class="forecast-column-grid" id="${this.topic}-results-table"></table>
        </div>
        <div class="run-controls">
          <div class="toggle" id="${this.topic}-page-prev">prev</div>
          <span class="stat" id="${this.topic}-page-info"></span>
          <div class="toggle" id="${this.topic}-page-next">next</div>
        </div>
      </div>`;
    document.getElementById(`${this.topic}-results-run`).onclick = evt => this.openRunPicker(evt.currentTarget);
    document.getElementById(`${this.topic}-breakdown-head`).onclick = evt => this.openBreakdownPicker(evt.currentTarget);
    document.getElementById(`${this.topic}-group-head`).onclick = evt => this.openGroupPicker(evt.currentTarget);
    document.getElementById(`${this.topic}-page-prev`).onclick = () => this.changePage(-1);
    document.getElementById(`${this.topic}-page-next`).onclick = () => this.changePage(1);
    document.getElementById(`${this.topic}-results-model`).onclick = evt => {
      listMenu('forecast', [
        {id: 'winner', label: this.resultsView.winner_family || 'xgboost', on: this.resultsModel === 'winner'},
        {id: 'ensemble', label: `ensemble of ${(this.resultsView.ensemble?.members || []).join(', ')}`, on: this.resultsModel === 'ensemble'},
      ], async item => {
        this.resultsModel = item.id;
        this.importanceFamily = null;
        this.tableRequest++;
        this.tableState.page = 0;
        await this.refreshView();
        await this.refreshTable();
      }).openAt(evt.currentTarget);
    };
    document.getElementById(`${this.topic}-importance-family`).onclick = evt => {
      const names = Object.keys(this.resultsView?.families || {});
      if (!names.length) names.push(this.resultsView?.family || 'xgboost');
      listMenu('feature importance', names.map(name => ({
        id: name, label: name, on: name === this.importanceFamily,
      })), item => { this.importanceFamily = item.id; this.refreshView(); }).openAt(evt.currentTarget);
    };
    return {enter: () => this.enterResults()};
  }

  async enterResults() {
    this.mountMetricPicker('results');
    if (this.resultsRunId) await this.loadResults();
  }

  async openRunPicker(head) {
    const data = await api('/api/forecast-runs');
    listMenu('runs', data.runs.map(r => ({
      id: r.id, label: `${r.id} - ${r.state}`, on: r.id === this.resultsRunId,
    })), item => {
      this.resultsRunId = item.id;
      setHead(head, item.id);
      this.loadResults();
    }, {empty: 'no runs yet'}).openAt(head);
  }

  async loadResults() {
    if (!this.resultsRunId) return;
    const load = ++this.resultsLoad;
    const runId = this.resultsRunId;
    this.viewRequest++;
    this.tableRequest++;
    try {
      const state = await api(`/api/forecast-run?id=${encodeURIComponent(runId)}`);
      if (load !== this.resultsLoad || runId !== this.resultsRunId) return;
      this.resultsSpec = state.spec;
      this.acceptMetricRegistry(state, 'results');
      this.resultsModel = null;
      this.importanceFamily = null;
      this.paintExcludedSeries(state.excluded_series || []);
      setHead(document.getElementById(`${this.topic}-results-run`), this.resultsRunId);
      const exportLink = document.getElementById(`${this.topic}-export`);
      exportLink.href = `/api/forecast-export/${encodeURIComponent(this.resultsRunId)}.csv`;
      exportLink.classList.remove('hidden');
      this.filters = {};
      this.breakdown = [];
      this.tableState = {page: 0, sort: null, desc: false, group: null};
      await this.loadFilterChoices();
      if (load !== this.resultsLoad) return;
      await this.refreshView();
      if (load !== this.resultsLoad) return;
      await this.refreshTable();
    } catch (err) {
      if (load !== this.resultsLoad) return;
      this.paintVerdict({summary: err.message, checks: []});
    }
  }

  dims() {
    if (!this.resultsSpec) return [];
    return (this.resultsSpec.columns || []).filter(c => c.role === 'dimension').map(c => c.name);
  }

  paintExcludedSeries(series) {
    const box = document.getElementById(`${this.topic}-excluded-series`);
    const combinations = series.map(item => {
      const dims = Object.entries(item.dims).map(([name, value]) => `${name}=${value}`).join(', ');
      return `[${dims}] (last seen ${item.last_seen || 'unknown'})`;
    });
    box.textContent = combinations.length
      ? `excluded from the forecast, no data before the forecast start: ${combinations.join(', ')}` : '';
    box.classList.toggle('hidden', !combinations.length);
  }

  // one page of the table, unfiltered, just to learn what values each dimension carries
  async loadFilterChoices() {
    const load = this.resultsLoad;
    this.filterChoices = {};
    const dims = this.dims();
    if (!dims.length) { this.renderFilterChips(); return; }
    const data = await api(
      `/api/forecast-table?id=${encodeURIComponent(this.resultsRunId)}&page=0&size=2000`
    );
    if (load !== this.resultsLoad) return;
    dims.forEach(dim => {
      const idx = data.columns.indexOf(dim);
      if (idx < 0) return;
      this.filterChoices[dim] = [...new Set(data.rows.map(r => r[idx]))].filter(v => v != null);
    });
    this.renderFilterChips();
  }

  renderFilterChips() {
    const box = document.getElementById(`${this.topic}-filter-chips`);
    box.innerHTML = '';
    const dims = this.dims();
    dims.forEach((dim, index) => {
      if (index) box.appendChild(Object.assign(document.createElement('div'), {className: 'divider'}));
      const col = document.createElement('div');
      col.className = 'col';
      col.appendChild(Object.assign(document.createElement('div'), {className: 'field-label', textContent: dim}));
      (this.filterChoices[dim] || []).forEach(value => {
        const chip = document.createElement('div');
        chip.className = 'toggle';
        chip.textContent = String(value);
        chip.onclick = () => {
          chip.classList.toggle('on');
          const picked = [...col.querySelectorAll('.toggle.on')].map(el => el.textContent);
          if (picked.length) this.filters[dim] = picked; else delete this.filters[dim];
          this.refreshView();
          this.tableState.page = 0;
          this.refreshTable();
        };
        col.appendChild(chip);
      });
      box.appendChild(col);
    });
  }

  openBreakdownPicker(head) {
    const dims = this.dims();
    new ForecastFieldMenu({
      title: 'break down by (up to 3)', persistent: true,
      sections: [{
        kind: 'list', multi: true, empty: 'no dimensions to break down by',
        items: dims.map(d => ({id: d, label: d, on: this.breakdown.includes(d)})),
        onPick: (item, on) => {
          if (on && this.breakdown.length >= 3) return; // capped, same as the server itself
          this.breakdown = on
            ? [...this.breakdown, item.id]
            : this.breakdown.filter(d => d !== item.id);
          setHead(head, this.breakdown.length ? this.breakdown.join(', ') : 'none');
          this.refreshView();
          this.refreshTable();
        },
      }],
    }).openAt(head);
  }

  openGroupPicker(head) {
    const dims = this.dims();
    new ForecastFieldMenu({title: 'group by', sections: [{kind: 'list',
      items: dims.map(d => ({id: d, label: d, on: d === this.tableState.group})),
      empty: 'no dimensions', onPick: item => {
        this.tableState.group = this.tableState.group === item.id ? null : item.id;
        setHead(head, this.tableState.group || 'none');
        this.tableState.page = 0;
        this.refreshTable();
      },
    }]}).openAt(head);
  }

  addHorizonParams(params) {
    const selected = this.horizonSelections[this.resultsRunId];
    if (selected && ['lead', 'blend'].includes(selected.mode) && Number.isInteger(selected.n)
        && selected.n >= 1 && selected.n <= this.resultsSpec?.horizon) {
      params.set(selected.mode, String(selected.n));
    }
  }

  paintHorizonControls(view) {
    const controls = document.getElementById(`${this.topic}-horizon-controls`);
    const help = document.getElementById(`${this.topic}-horizon-help`);
    controls.classList.toggle('hidden', !view.view);
    help.classList.toggle('hidden', !view.view);
    const errors = view.lead_errors || [];
    document.getElementById(`${this.topic}-lead-errors-wrap`).classList.toggle('hidden', !errors.length);
    const percent = value => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '-';
    forecastFillTable(document.getElementById(`${this.topic}-lead-errors`),
      ['weeks ahead', 'model wape', 'naive wape', 'ratio', 'rows'], errors.map(row => [
        row.weeks_ahead, percent(row.model_wape), percent(row.naive_wape),
        Number.isFinite(row.ratio) ? row.ratio.toFixed(2) : '-', row.rows ?? '-',
      ]));
    if (!view.view) return;
    this.horizonSelections[this.resultsRunId] = {...view.view};
    const descriptions = {
      lead: 'show only the forecast made n weeks before each week. nothing is averaged.',
      blend: 'average every forecast of a week. equal weights for now. a blend is not a forecast the model made; use it to smooth, not to judge.',
    };
    help.textContent = descriptions[view.view.mode];
    if (controls.dataset.horizon !== String(view.horizon)) {
      controls.dataset.horizon = String(view.horizon);
      controls.replaceChildren();
      for (const mode of ['lead', 'blend']) {
        const picker = forecastNode('div', 'forecast-metric-picker');
        const button = forecastNode('button', 'toggle dropdown-head');
        button.type = 'button';
        button.id = `${this.topic}-${mode}`;
        button.title = descriptions[mode];
        button.setAttribute('aria-expanded', 'false');
        button.setAttribute('aria-controls', `${this.topic}-${mode}-menu`);
        const menu = forecastNode('div', 'forecast-metric-menu hidden');
        menu.id = `${this.topic}-${mode}-menu`;
        menu.setAttribute('role', 'group');
        menu.setAttribute('aria-label', mode);
        const close = (focus = false) => {
          menu.classList.add('hidden');
          button.setAttribute('aria-expanded', 'false');
          if (focus) button.focus();
        };
        button.onclick = () => {
          const open = menu.classList.toggle('hidden') === false;
          button.setAttribute('aria-expanded', String(open));
        };
        picker.addEventListener('keydown', event => {
          if (event.key === 'Escape') { event.preventDefault(); close(true); }
          if (event.key === 'ArrowDown' && event.target === button) {
            event.preventDefault();
            menu.classList.remove('hidden');
            button.setAttribute('aria-expanded', 'true');
            menu.querySelector('input')?.focus();
          }
        });
        document.addEventListener('pointerdown', event => {
          if (!picker.contains(event.target)) close();
        });
        picker.addEventListener('focusout', event => {
          if (event.relatedTarget && !picker.contains(event.relatedTarget)) close();
        });
        const options = Array.from({length: view.horizon}, (_, index) => index + 1);
        if (mode === 'blend') options.unshift('off');
        options.forEach(n => {
          const label = forecastNode('label', 'forecast-metric-choice');
          const input = document.createElement('input');
          input.type = 'checkbox';
          input.dataset[mode] = String(n);
          input.onchange = async () => {
            this.horizonSelections[this.resultsRunId] = n === 'off'
              ? {mode: 'lead', n: view.horizon} : {mode, n};
            try { localStorage.setItem(this.horizonStorageKey, JSON.stringify(this.horizonSelections)); }
            catch (error) { /* selection still works when storage is unavailable */ }
            close(true);
            this.tableRequest++;
            this.tableState.page = 0;
            await this.refreshView();
            await this.refreshTable();
          };
          const text = n === 'off' ? 'off' : mode === 'blend' ? `around ${n}`
            : `${n} ${n === 1 ? 'week' : 'weeks'} ahead`;
          label.append(input, forecastNode('span', '', text));
          menu.append(label);
        });
        picker.append(button, menu);
        controls.append(picker);
      }
    }
    for (const mode of ['lead', 'blend']) {
      const active = view.view.mode === mode;
      const n = view.view.n;
      document.getElementById(`${this.topic}-${mode}`).textContent = `${mode}: ${!active ? 'off'
        : mode === 'blend' ? `around ${n}` : `${n} ${n === 1 ? 'week' : 'weeks'} ahead`}`;
      controls.querySelectorAll(`input[data-${mode}]`).forEach(input => {
        input.checked = active ? input.dataset[mode] === String(n) : input.dataset[mode] === 'off';
      });
    }
  }

  paintVerdict(verdict, label = null) {
    const box = document.getElementById(`${this.topic}-verdict`);
    box.classList.remove('hidden');
    forecastRatioStyle(box, verdict);
    box.innerHTML = '';
    const summary = document.createElement('div');
    summary.className = 'forecast-verdict-summary';
    const showRatio = this.selectedMetricIds.includes('test_ratio');
    const pooled = this.resultsView?.view && verdict === this.resultsView.verdict ? 'pooled verdict · ' : '';
    summary.textContent = pooled + (label ? `${label}: ` : '') + (showRatio ? forecastRatioText(verdict)
      : Number.isFinite(verdict?.accuracy?.ratio) ? 'test checks' : verdict?.summary || 'test checks');
    box.appendChild(summary);
    (verdict.checks || []).forEach(check => {
      const line = document.createElement('div');
      line.className = 'forecast-verdict-check';
      line.textContent = `${check.check}: ${check.detail}`;
      box.appendChild(line);
    });
  }

  paintModelResults(view) {
    this.resultsView = view;
    this.paintHorizonControls(view);
    this.acceptMetricRegistry(view, 'results');
    this.resultsModel = view.model || this.resultsModel || 'winner';
    const family = view.family || 'xgboost';
    this.importanceFamily = view.importance_family || this.importanceFamily || (family === 'ensemble' ? view.winner_family : family) || 'xgboost';
    const modelHead = document.getElementById(`${this.topic}-results-model`);
    modelHead.classList.toggle('hidden', !view.ensemble);
    setHead(modelHead, forecastModelLabel(view));
    const exportSuffix = view.ensemble ? `/${this.resultsModel}` : '';
    document.getElementById(`${this.topic}-export`).href = `/api/forecast-export/${encodeURIComponent(this.resultsRunId)}${exportSuffix}.csv`;
    const families = Object.entries(view.families || {});
    document.getElementById(`${this.topic}-results-models`).classList.remove('hidden');
    document.getElementById(`${this.topic}-family-comparison-wrap`).classList.toggle('hidden', !families.length);
    const metrics = this.selectedMetrics(view);
    const viewMetrics = document.getElementById(`${this.topic}-view-metrics`);
    viewMetrics.classList.toggle('hidden', !view.view);
    viewMetrics.replaceChildren(...(view.view ? [forecastMetricLines(metrics, view)] : []));
    const rows = families.map(([name, entry]) => [name, entry.state,
      ...metrics.map(metric => forecastMetricValue(metric, entry)),
      entry.candidates ?? 0, entry.pruned_reason || entry.reason || '-']);
    if (families.length && view.ensemble) rows.push([
      `ensemble of ${(view.ensemble.members || []).join(', ')}`, 'done',
      ...metrics.map(metric => forecastMetricValue(metric, view.ensemble)), '-', '-',
    ]);
    forecastFillTable(document.getElementById(`${this.topic}-family-comparison`),
      ['family', 'state', ...metrics.map(forecastMetricHeader), 'candidates', 'reason'], rows);
    setHead(document.getElementById(`${this.topic}-importance-family`), this.importanceFamily);
    const importance = view.importance;
    const method = importance?.method === 'absolute_standardized_coefficient' ? 'absolute standardised coefficient' : importance?.method;
    const noImportance = this.importanceFamily === 'ensemble' ? 'select a member for feature importance'
      : ['snaive', 'ets', 'theta', 'arima'].includes(this.importanceFamily)
        ? 'no feature importance for this family' : 'no saved feature importance for this run';
    setText(`${this.topic}-importance-status`, importance ? `${method} · normalised top features` : noImportance);
    const table = document.getElementById(`${this.topic}-importance`);
    table.classList.toggle('hidden', !importance?.features?.length);
    forecastFillTable(table, ['feature', 'importance'], (importance?.features || []).map(entry => [
      entry.feature, `${(entry.importance * 100).toFixed(2)}%`,
    ]));
  }

  async refreshView() {
    if (!this.resultsRunId) return;
    const request = ++this.viewRequest;
    const params = new URLSearchParams({id: this.resultsRunId});
    if (this.resultsModel) params.set('model', this.resultsModel);
    if (this.importanceFamily) params.set('importance_family', this.importanceFamily);
    if (Object.keys(this.filters).length) params.set('filters', JSON.stringify(this.filters));
    if (this.breakdown.length) params.set('breakdown', this.breakdown.join(','));
    this.addHorizonParams(params);
    try {
      const view = await api(`/api/forecast-view?${params.toString()}`);
      if (request !== this.viewRequest) return;
      this.paintModelResults(view);
      this.paintVerdict(view.verdict, forecastModelLabel(view));
      this.paintChart(view);
      document.getElementById(`${this.topic}-group-row`).classList.toggle('hidden', view.mode !== 'entity');
    } catch (err) {
      if (request !== this.viewRequest) return;
      this.paintVerdict({summary: err.message, checks: []});
    }
  }

  paintChart(view) {
    const el = document.getElementById(`${this.topic}-chart`);
    const status = document.getElementById(`${this.topic}-chart-status`);
    const hazard = view.objective === 'hazard';
    const aggregate = view.aggregate;
    status.textContent = hazard ? view.note || (aggregate
      ? `weekly ${aggregate.rule.replaceAll('_', ' ')}`
      : 'weekly aggregation is off; entity dates and duration quantiles are below')
      : view.view?.mode === 'blend' ? 'no band for a blend' : '';
    status.classList.toggle('hidden', !status.textContent);
    el.classList.toggle('hidden', hazard && !aggregate);
    if (hazard && !aggregate) return;
    if (!this.chart) this.chart = timeChart(el, {height: 220});
    this.chart.update({x: view.x, series: view.series, bands: view.bands, markers: view.markers});
    const caption = document.getElementById(`${this.topic}-chart-caption`);
    caption.textContent = !view.view ? '' : view.view.mode === 'lead'
      ? `each week shows the forecast made ${view.view.n} weeks before that week.`
      : `each week shows the average of all its forecasts, equal weights, centred on ${view.view.n} for later weighting.`;
    caption.classList.toggle('hidden', !caption.textContent);
  }

  async refreshTable() {
    if (!this.resultsRunId) return;
    const request = ++this.tableRequest;
    const params = new URLSearchParams({
      id: this.resultsRunId, page: String(this.tableState.page), size: String(FORECAST_PAGE_SIZE),
    });
    if (this.resultsModel) params.set('model', this.resultsModel);
    if (Object.keys(this.filters).length) params.set('filters', JSON.stringify(this.filters));
    if (this.breakdown.length) params.set('breakdown', this.breakdown.join(','));
    if (this.tableState.sort) {
      params.set('sort', this.tableState.sort);
      if (this.tableState.desc) params.set('desc', '1');
    }
    if (this.tableState.group) params.set('group', this.tableState.group);
    this.addHorizonParams(params);
    try {
      const data = await api(`/api/forecast-table?${params.toString()}`);
      if (request !== this.tableRequest) return;
      this.paintTable(data);
    } catch (err) {
      if (request !== this.tableRequest) return;
      this.paintVerdict({summary: err.message, checks: []});
    }
  }

  paintTable(data) {
    const table = document.getElementById(`${this.topic}-results-table`);
    table.innerHTML = '';
    const head = document.createElement('tr');
    const labels = data.objective === 'hazard'
      ? {median_weeks: 'median weeks', p10_weeks: 'p10 weeks', p90_weeks: 'p90 weeks', predicted_date: 'predicted date'} : {};
    data.columns.forEach(col => {
      const th = document.createElement('th');
      th.textContent = labels[col] || col;
      th.classList.add('sortable');
      th.onclick = () => {
        this.tableState.desc = this.tableState.sort === col ? !this.tableState.desc : false;
        this.tableState.sort = col;
        this.refreshTable();
      };
      head.appendChild(th);
    });
    table.appendChild(head);
    data.rows.forEach(row => {
      const tr = document.createElement('tr');
      row.forEach(value => {
        const td = document.createElement('td');
        td.textContent = value === null || value === undefined ? '' : String(value);
        td.title = td.textContent;
        tr.appendChild(td);
      });
      table.appendChild(tr);
    });
    const pages = Math.max(1, Math.ceil(data.total / FORECAST_PAGE_SIZE));
    setText(`${this.topic}-page-info`, `page ${data.page + 1} / ${pages} - ${data.total} rows`);
  }

  changePage(delta) {
    this.tableState.page = Math.max(0, this.tableState.page + delta);
    this.refreshTable();
  }
}

function textSpan(className, text) {
  const span = document.createElement('span');
  span.className = className;
  span.textContent = text;
  return span;
}

let forecastActiveTopic = null;
const forecastTopics = new Map();
window.addEventListener('smolsmort-topic', event => {
  forecastActiveTopic = event.detail;
  document.getElementById('forecast-dataset').classList.toggle('hidden', !forecastTopics.has(event.detail));
  forecastTopics.get(event.detail)?.paintDataset();
});

['regression', 'classification'].forEach(topic => {
  const ctl = new ForecastTopic(topic, topic);
  forecastTopics.set(topic, ctl);
  smolsmortTabs.register({id: `${topic}-data`, label: 'data', topic, mount: p => ctl.mountData(p)});
  registerForecastModels(ctl);
  smolsmortTabs.register({id: `${topic}-search`, label: 'training', topic, mount: p => ctl.mountSearch(p)});
  smolsmortTabs.register({id: `${topic}-results`, label: 'results', topic, mount: p => ctl.mountResults(p)});
  const flavours = topic === 'regression'
    ? [{id: 'entity', label: 'Entity'}, {id: 'timeseries', label: 'Time series'}]
    : [{id: 'entity', label: 'Entity'}];
  smolsmortTabs.setFlavours(topic, flavours);
  smolsmortTabs.onFlavour(topic, () => ctl.onFlavourChange());
});
