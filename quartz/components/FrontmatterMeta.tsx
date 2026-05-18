import { QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"
import style from "./styles/frontmatterMeta.scss"

interface FrontmatterMetaOptions {
  fields: string[]
}

function isUrl(value: string): boolean {
  try {
    new URL(value)
    return value.startsWith("http://") || value.startsWith("https://")
  } catch {
    return false
  }
}

export default ((opts: FrontmatterMetaOptions) => {
  function FrontmatterMeta({ fileData, displayClass }: QuartzComponentProps) {
    const fm = fileData.frontmatter
    if (!fm) return null

    const entries = opts.fields
      .map((key) => ({ key, value: fm[key] }))
      .filter(({ value }) => value !== undefined && value !== null && value !== "")

    if (entries.length === 0) return null

    return (
      <div class={classNames(displayClass, "frontmatter-meta")}>
        {entries.map(({ key, value }) => {
          const str = String(value)
          return (
            <span class="fm-item">
              <span class="fm-key">{key}</span>
              {isUrl(str) ? (
                <a class="fm-value fm-link" href={str} target="_blank" rel="noopener noreferrer">
                  {str}
                </a>
              ) : (
                <span class="fm-value">{str}</span>
              )}
            </span>
          )
        })}
      </div>
    )
  }

  FrontmatterMeta.css = style
  return FrontmatterMeta
}) satisfies QuartzComponentConstructor<FrontmatterMetaOptions>
