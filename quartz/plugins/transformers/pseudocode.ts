import { QuartzTransformerPlugin } from "../types"
import { Root } from "mdast"
import { Root as HtmlRoot, Element } from "hast"
import { Code } from "mdast"
import { visit } from "unist-util-visit"
import { toString } from "hast-util-to-string"
import pseudocodeScript from "../../components/scripts/pseudocode.inline"
import pseudocodeStyle from "../../components/styles/pseudocode.inline.scss"

const PSEUDOCODE_LANGS = new Set(["pseudo", "pseudocode"])

export const Pseudocode: QuartzTransformerPlugin = () => {
  return {
    name: "Pseudocode",
    markdownPlugins() {
      return [
        () => {
          return (tree: Root) => {
            visit(tree, "code", (node: Code) => {
              const lang = node.lang?.toLowerCase()
              if (!lang || !PSEUDOCODE_LANGS.has(lang)) return

              // Prevent rehype-pretty-code from transforming this block
              node.lang = "math"
              const existingProps =
                (node.data as { hProperties?: Record<string, unknown> })?.hProperties ?? {}
              const existingClass = Array.isArray((existingProps as { className?: unknown }).className)
                ? ((existingProps as { className?: unknown }).className as string[])
                : []
              node.data = {
                ...(node.data ?? {}),
                hProperties: {
                  ...existingProps,
                  className: [...new Set([...existingClass, "pseudocode"])],
                  "data-pseudocode": "true",
                },
              }
            })
          }
        },
      ]
    },
    htmlPlugins() {
      return [
        () => {
          return (tree: HtmlRoot) => {
            visit(tree, "element", (node: Element, index, parent) => {
              if (!parent || typeof index !== "number") return
              if (node.tagName !== "pre") return
              const code = node.children?.[0] as Element | undefined
              if (!code || code.type !== "element" || code.tagName !== "code") return
              const className = (code.properties?.className ?? []) as string[]
              const isPseudocodeClass = Array.isArray(className) && className.includes("pseudocode")
              const isPseudocodeData = code.properties?.["data-pseudocode"] === "true"
              if (!isPseudocodeClass && !isPseudocodeData) return

              const text = toString(code)
              parent.children[index] = {
                type: "element",
                tagName: "pre",
                properties: { className: ["pseudocode"], "data-pseudocode": "true" },
                children: [{ type: "text", value: text }],
              }
            })
          }
        },
      ]
    },
    externalResources() {
      return {
        js: [
          {
            src: "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js",
            loadTime: "afterDOMReady",
            contentType: "external",
          },
          {
            src: "https://cdn.jsdelivr.net/npm/pseudocode@2.4.1/build/pseudocode.min.js",
            loadTime: "afterDOMReady",
            contentType: "external",
          },
          {
            script: pseudocodeScript,
            loadTime: "afterDOMReady",
            contentType: "inline",
          },
        ],
        css: [
          { content: "https://cdn.jsdelivr.net/npm/pseudocode@2.4.1/build/pseudocode.min.css" },
          { content: pseudocodeStyle, inline: true },
        ],
      }
    },
  }
}
