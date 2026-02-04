document.addEventListener("nav", () => {
  const nodes = document.querySelectorAll("pre.pseudocode") as NodeListOf<HTMLElement>
  if (nodes.length === 0) return

  const pseudocode = (window as { pseudocode?: { renderElement: Function } }).pseudocode
  if (!pseudocode) return

  for (const node of nodes) {
    if (node.dataset.pseudocodeRendered === "true") continue
    try {
      pseudocode.renderElement(node)
      node.dataset.pseudocodeRendered = "true"
    } catch (err) {
      console.error("[pseudocode] render failed", err)
    }
  }
})
