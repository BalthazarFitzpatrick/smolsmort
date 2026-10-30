"""shared page states and measurements for the forecast chromium size audit"""


def settle_menu(page):
    """wait for the shared menu's deferred listener setup before sending keys"""
    page.evaluate("() => new Promise(resolve => setTimeout(resolve, 0))")


FAMILY_LAYOUT = r"""() => {
  const visible = node => node.getBoundingClientRect().height && node.getBoundingClientRect().width;
  const target = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--row-height'));
  const tables = [...document.querySelectorAll('[id$="-family-progress"], [id$="-family-comparison"], [id$="-importance"]')].filter(visible);
  const menus = [...document.querySelectorAll('[id$="-generation-family"], [id$="-results-model"], [id$="-importance-family"]')].filter(visible);
  return {
    target,
    rows: tables.flatMap(table => [...table.querySelectorAll('tr')].filter(visible).map(row => ({
      table: table.id, text: row.innerText, height: row.getBoundingClientRect().height
    }))),
    headers: tables.flatMap(table => [...table.querySelectorAll('th')].filter(visible).map(header => {
      const range = document.createRange();
      range.selectNodeContents(header);
      const lines = new Set([...range.getClientRects()].filter(rect => rect.width && rect.height).map(rect => Math.round(rect.top)));
      return {table: table.id, text: header.innerText, lines: lines.size};
    })),
    menus: menus.map(menu => ({id: menu.id, toggle: menu.classList.contains('toggle'), height: menu.getBoundingClientRect().height})),
    selects: [...tables, ...menus].flatMap(node => [...node.querySelectorAll('select'), ...(node.matches('select') ? [node] : [])]).length
  };
}"""


def assert_family_layout(page):
    """family tables and pickers share the host's row height and stay on one line"""
    measured = page.evaluate(FAMILY_LAYOUT)
    assert measured["selects"] == 0, measured
    for row in measured["rows"]:
        assert abs(row["height"] - measured["target"]) < 0.01, row
    for header in measured["headers"]:
        assert header["lines"] <= 1, header
    for menu in measured["menus"]:
        assert menu["toggle"], menu
        assert abs(menu["height"] - measured["target"]) < 0.01, menu
    return measured


CONTROL_SIZES = r"""() => {
  const selector = 'button, .toggle, .dropdown-head, input, select, .nav-tab, [role="combobox"], .menu-item, .menu-inline-action';
  const path = node => {
    if (node.id) return '#' + CSS.escape(node.id);
    const parts = [];
    while (node && node.nodeType === 1 && !node.id) {
      if (!node.parentElement) { parts.unshift(node.tagName.toLowerCase()); break; }
      const siblings = [...node.parentElement.children];
      parts.unshift(node.tagName.toLowerCase() + ':nth-child(' + (siblings.indexOf(node) + 1) + ')');
      node = node.parentElement;
    }
    return (node?.id ? '#' + CSS.escape(node.id) + ' > ' : '') + parts.join(' > ');
  };
  const target = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--row-height'));
  const borderWidth = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--row-border-width'));
  const fieldWidth = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--forecast-field-width'));
  const controlKind = node => {
    if (node.matches('.forecast-setup-slot .dropdown-head, .forecast-agg-row .forecast-field, [id$="-data-source"], [id$="-group-head"]')) return 'data-field';
    if (node.matches('.forecast-setup-menu .menu-list .menu-item, .forecast-field-menu .menu-list .menu-item')) return 'data-field';
    if (node.matches('.forecast-leaderboard .dropdown-head')) return 'recipe-head';
    if (node.matches('.forecast-column-grid .dropdown-head')) return 'role-head';
    if (node.matches('.stepper-row .toggle')) return 'stepper-button';
    if (node.matches('.forecast-pane input[type="date"]')) return 'date-input';
    if (node.matches('.forecast-pane input.grow')) return 'sql-input';
    if (node.matches('.forecast-run-row')) return 'run-row';
    return null;
  };
  const visible = node => {
    const rect = node.getBoundingClientRect();
    const style = getComputedStyle(node);
    const closed = node.closest('details:not([open])');
    if (closed && node !== closed && !closed.querySelector(':scope > summary')?.contains(node)) return false;
    return rect.width && rect.height && style.visibility !== 'hidden';
  };
  const bounds = node => {
    const rect = node.getBoundingClientRect();
    const style = getComputedStyle(node);
    return {selector: path(node), height: rect.height, left: rect.left,
      contentLeft: rect.left + parseFloat(style.paddingLeft) + parseFloat(style.borderLeftWidth),
      top: rect.top, bottom: rect.bottom};
  };
  const containers = [...document.querySelectorAll('#nav-bar, .nav-bar, .strip, #forecast-dataset, #top-bar-actions, .forecast-run-rail .toggle, .forecast-run-rail .dropdown-head')]
    .filter(node => visible(node) && !node.matches('.forecast-run-row')).map(bounds);
  const runMetrics = [...document.querySelectorAll('.forecast-run-row')].filter(visible).map(row => ({
    ...bounds(row), right: row.getBoundingClientRect().right,
    lines: [...row.querySelectorAll('.forecast-metric')].map(line => ({
      ...bounds(line), right: line.getBoundingClientRect().right,
      clientWidth: line.clientWidth, scrollWidth: line.scrollWidth
    }))
  }));
  const headerRows = [...document.querySelectorAll('#topic-switch, #forecast-dataset, #nav-bar')]
    .filter(visible).map(bounds);
  const headerControls = [...document.querySelectorAll('#topic-switch, #topic-flavour, #nav-bar, #open-settings')]
    .filter(visible).map(bounds);
  const gap = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--gap'));
  const texts = [];
  const roots = document.querySelectorAll('#topic-bar, #top-bar, #forecast-dataset, .forecast-pane, .forecast-run-rail, .forecast-results, .menu-panel, #settings-popup');
  const parents = new Set();
  for (const root of roots) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const text = walker.currentNode;
      const node = text.parentElement;
      if (!text.textContent.trim() || parents.has(node) || !visible(node) || node.closest('svg, canvas, pre, script, style')) continue;
      parents.add(node);
      const rects = [...node.childNodes].filter(child => child.nodeType === Node.TEXT_NODE && child.textContent.trim()).flatMap(child => {
        const range = document.createRange();
        range.selectNodeContents(child);
        return [...range.getClientRects()].filter(rect => rect.width && rect.height);
      });
      const tops = new Set(rects.map(rect => Math.round(rect.top * 100) / 100));
      const style = getComputedStyle(node);
      const lineHeight = parseFloat(style.lineHeight) || parseFloat(style.fontSize) * 1.2;
      const plainLabel = node.matches('.field-label, .field-value, .stat, .popup-title, summary')
        && !node.matches(selector) && !node.children.length;
      texts.push({selector: path(node), text: text.textContent.trim(), lines: tops.size,
        clientHeight: node.clientHeight, lineHeight,
        lineBoxHeight: plainLabel ? node.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom) : null,
        textHeight: rects.length ? Math.max(...rects.map(rect => rect.bottom)) - Math.min(...rects.map(rect => rect.top)) : 0});
    }
  }
  return {target, fieldWidth, gap, containers, headerRows, headerControls, texts, runMetrics, controls: [...document.querySelectorAll(selector)].flatMap(node => {
    const rect = node.getBoundingClientRect();
    const style = getComputedStyle(node);
    if (!rect.width || !rect.height || style.visibility === 'hidden' || node.matches('.forecast-run-row')) return [];
    return [{selector: path(node), text: (node.innerText || node.getAttribute('aria-label') || node.value || node.type || '').replace(/\s+/g, ' ').trim(),
      height: rect.height, targetHeight: node.matches('.nav-tab') ? target - 2 * borderWidth : target,
      width: rect.width, font: style.fontFamily, size: style.fontSize,
      padding: style.padding, box: style.boxSizing, classes: node.className, kind: controlKind(node)}];
  })};
}"""


def visit_control_states(page, *, source, targets, run_id, capture, set_results_mode=None):
    """visit host flavours, forecast panes and their menus without starting a search"""

    def tab(name):
        page.click(f'.nav-tab[data-tab="{name}"]')
        page.wait_for_timeout(200)

    def topic(name):
        page.click(f'#topic-switch .topic-tab:has-text("{name}")')
        page.wait_for_timeout(100)

    topic("vision")
    tab("train")
    page.wait_for_function("document.querySelector('#topic-flavour option') !== null")
    flavours = page.locator("#topic-flavour option").evaluate_all(
        "items => items.map(item => ({id: item.value, label: item.textContent}))"
    )
    for flavour in flavours:
        page.evaluate(
            "([items, id]) => smolsmortTabs.setFlavours('vision', items, id)",
            [flavours, flavour["id"]],
        )
        capture(f"host-vision-{flavour['id']}")

    for name in ("regression", "classification"):
        if set_results_mode:
            set_results_mode("timeseries")
        topic(name)
        tab(f"{name}-data")
        page.click(f"#{name}-data-source")
        page.locator(f'.menu-panel .menu-item[data-id="{source}"]').click()
        page.wait_for_function(
            "id => /^\\d+ columns/.test(document.getElementById(id).innerText)",
            arg=f"{name}-data-status",
        )
        options = page.locator("#topic-flavour option").evaluate_all(
            "items => items.map(item => item.value)"
        )
        for flavour in options:
            tab(f"{name}-data")
            page.select_option("#topic-flavour", flavour)
            page.wait_for_timeout(150)
            capture(f"data-{name}-{flavour}")
            tab(f"{name}-models")
            page.locator(f"#{name}-models-families [data-family]").first.wait_for()
            capture(f"models-{name}-{flavour}")
        tab(f"{name}-search")
        page.locator(f'#{name}-runs-list [data-run-id="{run_id}"]').click()
        page.click(f"#{name}-run-view-tables")
        page.locator(f"#{name}-leaderboard-table tbody tr").first.wait_for()
        assert_family_layout(page)
        capture(f"search-{name}-finished")
        if page.locator(f"#{name}-generation-family").is_visible():
            page.click(f"#{name}-generation-family")
            page.locator(".menu-panel").wait_for()
            capture(f"search-{name}-family-menu")
            page.keyboard.press("Escape")
        page.locator(f"#{name}-leaderboard-table tbody tr .dropdown-head").first.click()
        page.locator(".forecast-recipe-details").wait_for()
        capture(f"recipe-{name}")
        page.keyboard.press("Escape")
        page.click(f"#{name}-new-search")
        capture(f"models-{name}-new")
        tab(f"{name}-search")
        page.locator(f'#{name}-runs-list [data-run-id="{run_id}"]').click()
        page.click(f"#{name}-run-open")
        page.locator(f"#{name}-results-table tr").nth(1).wait_for()
        assert_family_layout(page)
        capture(f"results-{name}")
        for picker in ("results-model", "importance-family"):
            if page.locator(f"#{name}-{picker}").is_visible():
                page.click(f"#{name}-{picker}")
                page.locator(".menu-panel").wait_for()
                capture(f"results-{name}-{picker}-menu")
                page.keyboard.press("Escape")
        page.click(f"#{name}-results-run")
        page.locator(".menu-panel").wait_for()
        capture(f"results-{name}-run-menu")
        page.keyboard.press("Escape")
        for picker in ("breakdown", "group"):
            if picker == "group" and set_results_mode:
                set_results_mode("entity")
                page.click(f"#{name}-results-run")
                page.locator(f'.menu-panel .menu-item[data-id="{run_id}"]').click()
                page.locator(f"#{name}-group-head").wait_for()
            if not page.locator(f"#{name}-{picker}-head").is_visible():
                continue
            page.click(f"#{name}-{picker}-head")
            page.locator(".menu-panel").wait_for()
            capture(f"results-{name}-{picker}-menu")
            page.keyboard.press("Escape")

    topic("regression")
    page.select_option("#topic-flavour", "timeseries")
    tab("regression-data")
    page.click("#regression-setup-predict")
    for target in targets:
        row = page.locator(f'.forecast-setup-menu .menu-item[data-id="{target}"]')
        if "on" not in (row.get_attribute("class") or "").split():
            row.click()
    capture("setup-predict")
    page.locator('.forecast-setup-menu .menu-buttons [data-id="save"]').click()
    for key in ("aggregate", "time", "scaffold"):
        page.click(f"#regression-setup-{key}")
        capture(f"setup-{key}")
        if key in ("aggregate", "time"):
            page.locator(".forecast-setup-menu .dropdown-head").first.click()
            capture(f"setup-{key}-choices")
        page.locator('.forecast-setup-menu .menu-buttons [data-id="close"]').click()

    page.click("#open-settings")
    page.locator("#settings-popup").wait_for()
    capture("settings")
    page.click("#settings-close")
