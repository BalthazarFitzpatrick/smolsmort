"""shared page states and measurements for the forecast chromium size audit"""

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
  const fieldWidth = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--forecast-field-width'));
  const controlKind = node => {
    if (node.matches('.forecast-setup-slot .dropdown-head, .forecast-agg-row .forecast-field, [id$="-data-source"], [id$="-breakdown-head"], [id$="-group-head"]')) return 'data-field';
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
    return rect.width && rect.height && style.visibility !== 'hidden';
  };
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
      const plainLabel = node.matches('.field-label, .field-value, .stat, .popup-title, summary') && !node.matches(selector) && !node.children.length;
      texts.push({selector: path(node), text: text.textContent.trim(), lines: tops.size,
        clientHeight: node.clientHeight, lineHeight,
        lineBoxHeight: plainLabel ? node.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom) : null,
        textHeight: rects.length ? Math.max(...rects.map(rect => rect.bottom)) - Math.min(...rects.map(rect => rect.top)) : 0});
    }
  }
  return {target, fieldWidth, texts, controls: [...document.querySelectorAll(selector)].flatMap(node => {
    const rect = node.getBoundingClientRect();
    const style = getComputedStyle(node);
    if (!rect.width || !rect.height || style.visibility === 'hidden') return [];
    return [{selector: path(node), text: (node.innerText || node.getAttribute('aria-label') || node.value || node.type || '').replace(/\s+/g, ' ').trim(),
      height: rect.height, width: rect.width, font: style.fontFamily, size: style.fontSize,
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
            set_results_mode("series")
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
            page.select_option("#topic-flavour", flavour)
            page.wait_for_timeout(150)
            capture(f"data-{name}-{flavour}")
        tab(f"{name}-search")
        page.locator(f'#{name}-runs-list [data-run-id="{run_id}"]').click()
        page.locator(f"#{name}-leaderboard-table tbody tr").first.wait_for()
        capture(f"search-{name}-finished")
        page.locator(f"#{name}-leaderboard-table tbody tr .dropdown-head").first.click()
        page.locator(".forecast-recipe-details").wait_for()
        capture(f"recipe-{name}")
        page.keyboard.press("Escape")
        page.click(f"#{name}-new-search")
        capture(f"search-{name}-new")
        page.locator(f'#{name}-runs-list [data-run-id="{run_id}"]').click()
        page.click(f"#{name}-run-open")
        page.locator(f"#{name}-results-table tr").nth(1).wait_for()
        capture(f"results-{name}")
        page.click(f"#{name}-results-run")
        page.locator(".menu-panel").wait_for()
        capture(f"results-{name}-run-menu")
        page.keyboard.press("Escape")
        for picker in ("breakdown", "group"):
            if picker == "group" and set_results_mode:
                set_results_mode("row")
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
    page.select_option("#topic-flavour", "series")
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
