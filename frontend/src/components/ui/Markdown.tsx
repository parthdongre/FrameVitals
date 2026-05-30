import { type ReactNode } from "react";
import { cn } from "@/lib/utils";

/**
 * Editorial markdown renderer for LLM output.
 *
 * Designed for the kind of text the agent actually produces — headings,
 * bullets, ordered lists, **bold**, *italic*, `inline code`, fenced code
 * blocks, blockquotes, simple links, and horizontal rules. We deliberately
 * keep it dependency-free so the bundle stays small and the security
 * surface stays tiny (no innerHTML anywhere).
 *
 * Used by:
 *   - AskAnythingTab — renders chat answers
 *   - AiReportTab    — renders the AI narrative report
 */

type Block =
  | { kind: "h1"; text: string }
  | { kind: "h2"; text: string }
  | { kind: "h3"; text: string }
  | { kind: "h4"; text: string }
  | { kind: "ul"; items: string[] }
  | { kind: "ol"; items: string[] }
  | { kind: "quote"; text: string }
  | { kind: "code"; lang: string; text: string }
  | { kind: "hr" }
  | { kind: "p"; text: string };

interface MarkdownProps {
  /**
   * Source markdown. Pass an empty string for "no content yet".
   */
  children: string | null | undefined;
  /**
   * Optional density: "default" for the AI report (looser leading) or
   * "compact" for chat bubbles (tighter leading + smaller spacing).
   */
  density?: "default" | "compact";
  className?: string;
}

export function Markdown({ children, density = "default", className }: MarkdownProps) {
  const text = (children ?? "").toString();
  if (!text.trim()) return null;
  const blocks = parseBlocks(text);
  return (
    <div
      className={cn(
        density === "compact" ? "space-y-2" : "space-y-3",
        className,
      )}
    >
      {blocks.map((block, idx) => (
        <RenderBlock key={idx} block={block} density={density} />
      ))}
    </div>
  );
}

/* --------------------------------------------------------------------------
 * Parser
 * -------------------------------------------------------------------------- */

function parseBlocks(raw: string): Block[] {
  // Strip leftover <think>...</think> blocks just in case.
  let src = raw.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  src = src.replace(/\r\n/g, "\n");

  const blocks: Block[] = [];
  const lines = src.split("\n");
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Blank line — skip
    if (!line.trim()) {
      i += 1;
      continue;
    }

    // Horizontal rule
    if (/^\s*([-*_])\s*\1\s*\1[\s-*_]*$/.test(line)) {
      blocks.push({ kind: "hr" });
      i += 1;
      continue;
    }

    // Fenced code block
    const fenceOpen = /^\s*```([a-zA-Z0-9_-]*)\s*$/.exec(line);
    if (fenceOpen) {
      const lang = fenceOpen[1] || "";
      const codeLines: string[] = [];
      i += 1;
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) {
        codeLines.push(lines[i]);
        i += 1;
      }
      // Skip the closing fence
      if (i < lines.length) i += 1;
      blocks.push({ kind: "code", lang, text: codeLines.join("\n") });
      continue;
    }

    // ATX-style heading
    const heading = /^(#{1,4})\s+(.*?)\s*#*\s*$/.exec(line);
    if (heading) {
      const level = heading[1].length;
      const text = heading[2].trim();
      const kind = (level === 1 ? "h1" : level === 2 ? "h2" : level === 3 ? "h3" : "h4") as Block["kind"];
      blocks.push({ kind, text } as Block);
      i += 1;
      continue;
    }

    // Blockquote
    if (/^\s*>\s?/.test(line)) {
      const buf: string[] = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
        buf.push(lines[i].replace(/^\s*>\s?/, ""));
        i += 1;
      }
      blocks.push({ kind: "quote", text: buf.join(" ").trim() });
      continue;
    }

    // Unordered list
    if (/^\s*[-*·•]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*·•]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*·•]\s+/, "").trim());
        i += 1;
      }
      blocks.push({ kind: "ul", items });
      continue;
    }

    // Ordered list
    if (/^\s*\d+[.)]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+[.)]\s+/, "").trim());
        i += 1;
      }
      blocks.push({ kind: "ol", items });
      continue;
    }

    // Paragraph — gather consecutive non-blank lines that aren't another block
    const buf: string[] = [];
    while (i < lines.length) {
      const cur = lines[i];
      if (
        !cur.trim() ||
        /^(#{1,4})\s+/.test(cur) ||
        /^\s*[-*·•]\s+/.test(cur) ||
        /^\s*\d+[.)]\s+/.test(cur) ||
        /^\s*>\s?/.test(cur) ||
        /^\s*```/.test(cur) ||
        /^\s*([-*_])\s*\1\s*\1[\s-*_]*$/.test(cur)
      ) {
        break;
      }
      buf.push(cur);
      i += 1;
    }
    if (buf.length) {
      blocks.push({ kind: "p", text: buf.join(" ").trim() });
    }
  }

  return blocks;
}

/* --------------------------------------------------------------------------
 * Block renderers
 * -------------------------------------------------------------------------- */

function RenderBlock({
  block,
  density,
}: {
  block: Block;
  density: "default" | "compact";
}) {
  const isCompact = density === "compact";

  switch (block.kind) {
    case "h1":
      return (
        <h1
          className={cn(
            "font-display font-semibold leading-tight text-ink-1 tracking-tight",
            isCompact ? "text-[18px]" : "text-[22px]",
          )}
        >
          {renderInline(block.text)}
        </h1>
      );
    case "h2":
      return (
        <h2
          className={cn(
            "font-display font-semibold leading-tight text-ink-1 tracking-tight",
            isCompact ? "text-[16px]" : "text-[18px]",
          )}
        >
          {renderInline(block.text)}
        </h2>
      );
    case "h3":
      return (
        <h3
          className={cn(
            "font-display font-semibold leading-tight text-ink-1",
            isCompact ? "text-[14px]" : "text-[15px]",
          )}
        >
          {renderInline(block.text)}
        </h3>
      );
    case "h4":
      return (
        <h4
          className={cn(
            "font-display font-semibold leading-tight text-ink-1 uppercase tracking-[0.16em]",
            isCompact ? "text-[12px]" : "text-[12px]",
          )}
        >
          {renderInline(block.text)}
        </h4>
      );
    case "ul":
      return (
        <ul
          className={cn(
            "list-disc pl-5 text-ink-2",
            isCompact ? "space-y-0.5 text-[13px] leading-6" : "space-y-1 text-[14px] leading-7",
          )}
        >
          {block.items.map((it, i) => (
            <li key={i}>{renderInline(it)}</li>
          ))}
        </ul>
      );
    case "ol":
      return (
        <ol
          className={cn(
            "list-decimal pl-5 text-ink-2",
            isCompact ? "space-y-0.5 text-[13px] leading-6" : "space-y-1 text-[14px] leading-7",
          )}
        >
          {block.items.map((it, i) => (
            <li key={i}>{renderInline(it)}</li>
          ))}
        </ol>
      );
    case "quote":
      return (
        <blockquote
          className={cn(
            "border-l-2 border-accent-line pl-3 italic text-ink-2",
            isCompact ? "text-[13px] leading-6" : "text-[14px] leading-7",
          )}
        >
          {renderInline(block.text)}
        </blockquote>
      );
    case "code":
      return (
        <pre
          className={cn(
            "overflow-x-auto rounded-sm border border-line bg-bg-2 p-3 font-mono text-ink-1",
            isCompact ? "text-[11px] leading-5" : "text-[12px] leading-6",
          )}
          aria-label={block.lang ? `${block.lang} code block` : undefined}
        >
          <code>{block.text}</code>
        </pre>
      );
    case "hr":
      return <hr className="border-line" aria-hidden="true" />;
    case "p":
    default:
      return (
        <p
          className={cn(
            "text-ink-2",
            isCompact ? "text-[13px] leading-6" : "text-[14px] leading-7",
          )}
        >
          {renderInline(block.text)}
        </p>
      );
  }
}

/* --------------------------------------------------------------------------
 * Inline parser — supports **bold**, *italic*, `code`, [text](url),
 * autolinks, and BR-on-double-space. Order matters: longest tokens first.
 * -------------------------------------------------------------------------- */

interface InlineToken {
  type: "text" | "bold" | "italic" | "code" | "link" | "br";
  value: string;
  href?: string;
}

function tokenizeInline(text: string): InlineToken[] {
  const tokens: InlineToken[] = [];
  let i = 0;
  let buffer = "";

  const flush = () => {
    if (buffer) {
      tokens.push({ type: "text", value: buffer });
      buffer = "";
    }
  };

  while (i < text.length) {
    const ch = text[i];

    // Inline code with backticks
    if (ch === "`") {
      const end = text.indexOf("`", i + 1);
      if (end !== -1) {
        flush();
        tokens.push({ type: "code", value: text.slice(i + 1, end) });
        i = end + 1;
        continue;
      }
    }

    // Bold: ** ... **
    if (ch === "*" && text[i + 1] === "*") {
      const end = text.indexOf("**", i + 2);
      if (end !== -1) {
        flush();
        tokens.push({ type: "bold", value: text.slice(i + 2, end) });
        i = end + 2;
        continue;
      }
    }

    // Italic: * ... *  (single asterisk, no double)
    if (ch === "*" && text[i + 1] !== "*") {
      const end = text.indexOf("*", i + 1);
      if (end !== -1 && text[end + 1] !== "*" && end > i + 1) {
        flush();
        tokens.push({ type: "italic", value: text.slice(i + 1, end) });
        i = end + 1;
        continue;
      }
    }

    // Markdown link: [text](url)
    if (ch === "[") {
      const closeBracket = text.indexOf("]", i + 1);
      if (closeBracket !== -1 && text[closeBracket + 1] === "(") {
        const closeParen = text.indexOf(")", closeBracket + 2);
        if (closeParen !== -1) {
          const linkText = text.slice(i + 1, closeBracket);
          const href = text.slice(closeBracket + 2, closeParen);
          flush();
          tokens.push({ type: "link", value: linkText, href });
          i = closeParen + 1;
          continue;
        }
      }
    }

    // Autolink: bare http(s) URL
    if ((ch === "h" && text.slice(i, i + 7) === "http://") || (ch === "h" && text.slice(i, i + 8) === "https://")) {
      const match = /^https?:\/\/[^\s)]+/.exec(text.slice(i));
      if (match) {
        flush();
        tokens.push({ type: "link", value: match[0], href: match[0] });
        i += match[0].length;
        continue;
      }
    }

    // Hard break: two trailing spaces + newline
    if (ch === " " && text[i + 1] === " " && (text[i + 2] === "\n" || i + 2 === text.length)) {
      flush();
      tokens.push({ type: "br", value: "" });
      i += text[i + 2] === "\n" ? 3 : 2;
      continue;
    }

    buffer += ch;
    i += 1;
  }
  flush();
  return tokens;
}

function renderInline(text: string): ReactNode {
  const tokens = tokenizeInline(text);
  return tokens.map((t, i) => {
    switch (t.type) {
      case "bold":
        return (
          <strong key={i} className="font-semibold text-ink-1">
            {t.value}
          </strong>
        );
      case "italic":
        return (
          <em key={i} className="italic">
            {t.value}
          </em>
        );
      case "code":
        return (
          <code
            key={i}
            className="rounded-sm border border-line bg-bg-2 px-1.5 py-0.5 font-mono text-[12px] text-ink-1"
          >
            {t.value}
          </code>
        );
      case "link":
        return (
          <a
            key={i}
            href={t.href}
            target="_blank"
            rel="noreferrer noopener"
            className="text-accent underline-offset-2 hover:underline"
          >
            {t.value}
          </a>
        );
      case "br":
        return <br key={i} />;
      case "text":
      default:
        return <span key={i}>{t.value}</span>;
    }
  });
}
