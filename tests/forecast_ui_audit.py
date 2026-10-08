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
  return {target, controls: [...document.querySelectorAll(selector)].flatMap(node => {
    const rect = node.getBoundingClientRect();
    const style = getComputedStyle(node);
    if (!rect.width || !rect.height || style.visibility === 'hidden') return [];
    return [{selector: path(node), text: (node.innerText || node.getAttribute('aria-label') || node.value || node.type || '').replace(/\s+/g, ' ').trim(),
      height: rect.height, width: rect.width, font: style.fontFamily, size: style.fontSize,
      padding: style.padding, box: style.boxSizing, classes: node.className}];
  })};
}"""


def visit_control_states(page, *, source, targets, run_id, capture):
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
