import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export interface DataTableColumn<Row> {
  key: string;
  header: ReactNode;
  /**
   * How to render a cell. Defaults to `String(row[key])`.
   */
  cell?: (row: Row, rowIndex: number) => ReactNode;
  /**
   * Render the cell as mono / tabular-nums. Useful for numeric columns.
   */
  mono?: boolean;
  /**
   * Right-align (numeric columns).
   */
  align?: "left" | "right" | "center";
  /**
   * Optional explicit width hint applied as `style.width`.
   */
  width?: string;
  /**
   * Wrap longer cell values onto multiple lines. Default false (truncated).
   */
  wrap?: boolean;
}

interface DataTableProps<Row> {
  columns: DataTableColumn<Row>[];
  rows: Row[];
  /**
   * Optional row key extractor. Falls back to the index.
   */
  rowKey?: (row: Row, index: number) => string;
  /**
   * Caption rendered above the table for screen readers.
   */
  caption?: string;
  /**
   * Sticky header. Default true (the column headers stay pinned during
   * vertical scroll).
   */
  stickyHeader?: boolean;
  /**
   * Max-height for scrollable bodies (e.g. "60vh"). Default `none` (table
   * grows with its content).
   */
  maxHeight?: string;
  className?: string;
  /**
   * Compact variant for dense rows. Default false.
   */
  compact?: boolean;
  /**
   * Empty-state node when `rows` is empty. Defaults to a small inline message.
   */
  emptyState?: ReactNode;
}

/**
 * Editorial data table. Hairline borders, mono headers, no zebra fill, sticky
 * header by default. The component is generic over the row shape so consumers
 * keep their type safety.
 *
 * Use `KeyValueGrid` for label/value pairs and `DataTable` for tabular data.
 */
export function DataTable<Row extends Record<string, unknown>>({
  columns,
  rows,
  rowKey,
  caption,
  stickyHeader = true,
  maxHeight,
  className,
  compact,
  emptyState,
}: DataTableProps<Row>) {
  const padding = compact ? "px-3 py-2" : "px-4 py-3";

  if (!rows || rows.length === 0) {
    return (
      <div className="rounded-md border border-line bg-bg-1 px-4 py-6 text-center text-[13px] text-ink-3">
        {emptyState ?? "No rows to display."}
      </div>
    );
  }

  return (
    <div
      className={cn(
        "overflow-auto rounded-md border border-line bg-bg-1",
        className,
      )}
      style={maxHeight ? { maxHeight } : undefined}
    >
      <table className="w-full border-collapse text-[13px] text-ink-2">
        {caption ? <caption className="sr-only">{caption}</caption> : null}
        <thead className={cn(stickyHeader && "sticky top-0 z-10 bg-bg-1")}>
          <tr className="border-b border-line">
            {columns.map((col) => (
              <th
                key={col.key}
                scope="col"
                style={col.width ? { width: col.width } : undefined}
                className={cn(
                  "font-mono text-[10px] uppercase tracking-[0.28em] text-ink-3",
                  padding,
                  col.align === "right" && "text-right",
                  col.align === "center" && "text-center",
                  col.align !== "right" && col.align !== "center" && "text-left",
                )}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={rowKey ? rowKey(row, i) : i}
              className="border-b border-line last:border-b-0 hover:bg-bg-2"
            >
              {columns.map((col) => {
                const value = col.cell ? col.cell(row, i) : (row[col.key] as ReactNode);
                return (
                  <td
                    key={col.key}
                    style={col.width ? { width: col.width } : undefined}
                    className={cn(
                      padding,
                      col.mono && "font-mono tabular-nums text-ink-1",
                      col.align === "right" && "text-right",
                      col.align === "center" && "text-center",
                      !col.wrap && "whitespace-nowrap",
                      col.wrap && "whitespace-pre-line",
                    )}
                  >
                    {value as ReactNode}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
