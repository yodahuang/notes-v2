function normalizePseudocodeSource(source: string) {
  return source.replace(/\\\[((?:.|\n)*?)\\\]/g, (_match, inner) => {
    const compact = inner.replace(/\s+/g, " ").trim()
    return `$${compact}$`
  })
}

document.addEventListener("nav", () => {
  const nodes = document.querySelectorAll("pre.pseudocode") as NodeListOf<HTMLElement>
  if (nodes.length === 0) return

  const pseudocode = (window as { pseudocode?: { renderElement: Function } }).pseudocode
  if (!pseudocode) return

  for (const node of nodes) {
    if (node.dataset.pseudocodeRendered === "true") continue
    try {
      const raw = node.textContent ?? ""
      node.textContent = normalizePseudocodeSource(raw)
      pseudocode.renderElement(node)
      node.dataset.pseudocodeRendered = "true"
    } catch (err) {
      console.error("[pseudocode] render failed", err)
    }
  }
})
