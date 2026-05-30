import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "framer-motion";

import { cn } from "@/lib/utils";
import { Skeleton } from "./skeleton";

interface StaticChartImageProps {
  src: string;
  alt: string;
  /**
   * Aspect ratio hint used for the placeholder skeleton. Default 16/10
   * (matches how seaborn/matplotlib export figures by default).
   */
  aspectRatio?: number;
  className?: string;
  /** Set false to disable click-to-fullscreen. Default: true. */
  zoomable?: boolean;
}

/**
 * Lazy-loaded backend chart PNG with a skeleton placeholder while decoding.
 * Falls back to an inline error message so a broken chart never blocks the
 * rest of the gallery.
 *
 * Clicking the chart opens a fullscreen lightbox portal (mounted on
 * document.body so it escapes any overflow-hidden ancestor). Close by
 * clicking the backdrop, the image itself, the × button, or pressing Escape.
 */
export function StaticChartImage({
  src,
  alt,
  aspectRatio = 16 / 10,
  className,
  zoomable = true,
}: StaticChartImageProps) {
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  const [open, setOpen] = useState(false);
  const ratioPadding = `${(1 / aspectRatio) * 100}%`;
  const canZoom = zoomable && state === "ready";

  return (
    <>
      {/* ── thumbnail card ─────────────────────────────────────── */}
      <div
        className={cn(
          "group relative w-full overflow-hidden rounded-md bg-bg-2",
          canZoom &&
            "cursor-zoom-in transition-shadow duration-200 hover:shadow-[0_0_0_1px_rgba(255,255,255,0.16)]",
          className,
        )}
        role={canZoom ? "button" : undefined}
        tabIndex={canZoom ? 0 : undefined}
        aria-label={canZoom ? `${alt} — click to expand` : undefined}
        onClick={() => canZoom && setOpen(true)}
        onKeyDown={(e) => {
          if (canZoom && (e.key === "Enter" || e.key === " ")) {
            e.preventDefault();
            setOpen(true);
          }
        }}
      >
        {/* aspect-ratio spacer */}
        <div style={{ paddingTop: ratioPadding }} aria-hidden />

        {state === "loading" && (
          <Skeleton className="absolute inset-0 rounded-md" />
        )}
        {state === "error" && (
          <div className="absolute inset-0 flex items-center justify-center px-4 text-center font-mono text-[11px] text-ink-3">
            Could not load chart.
          </div>
        )}

        {/* eslint-disable-next-line jsx-a11y/img-redundant-alt */}
        <img
          src={src}
          alt={alt}
          loading="lazy"
          decoding="async"
          onLoad={() => setState("ready")}
          onError={() => setState("error")}
          className={cn(
            "absolute inset-0 h-full w-full object-contain transition-opacity duration-300",
            state === "ready" ? "opacity-100" : "opacity-0",
          )}
        />

        {/* hover badge */}
        {canZoom && (
          <span
            aria-hidden
            className="pointer-events-none absolute right-2 top-2 flex items-center gap-1 rounded border border-white/20 bg-black/65 px-2 py-0.5 font-mono text-[9px] uppercase tracking-[0.2em] text-white/75 opacity-0 backdrop-blur-sm transition-opacity duration-200 group-hover:opacity-100"
          >
            <ExpandIcon />
            expand
          </span>
        )}
      </div>

      {/* ── lightbox portal — mounted on document.body ─────────── */}
      <AnimatePresence>
        {open && <Lightbox src={src} alt={alt} onClose={() => setOpen(false)} />}
      </AnimatePresence>
    </>
  );
}

/* ── Lightbox ─────────────────────────────────────────────────────────────── */

function Lightbox({
  src,
  alt,
  onClose,
}: {
  src: string;
  alt: string;
  onClose: () => void;
}) {
  // Lock body scroll + listen for Escape
  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener("keydown", handler);
    };
  }, [onClose]);

  const node = (
    // Backdrop
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.16 }}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9999,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "rgba(0,0,0,0.92)",
        padding: "24px",
      }}
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={alt}
    >
      {/* Image — stops click propagation so clicking the image doesn't close */}
      <motion.img
        src={src}
        alt={alt}
        initial={{ scale: 0.94, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.94, opacity: 0 }}
        transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
        onClick={(e) => e.stopPropagation()}
        style={{
          maxWidth: "min(96vw, 1400px)",
          maxHeight: "92vh",
          width: "auto",
          height: "auto",
          objectFit: "contain",
          borderRadius: "8px",
          cursor: "zoom-out",
          boxShadow: "0 30px 100px rgba(0,0,0,0.7)",
          background: "#141414",
        }}
      />

      {/* Close button — top-right, always on top */}
      <button
        type="button"
        onClick={onClose}
        aria-label="Close fullscreen"
        style={{
          position: "fixed",
          top: "16px",
          right: "16px",
          zIndex: 10000,
          display: "inline-flex",
          alignItems: "center",
          gap: "6px",
          padding: "6px 14px",
          background: "rgba(20,20,20,0.9)",
          border: "1px solid rgba(255,255,255,0.18)",
          borderRadius: "999px",
          color: "rgba(245,239,230,0.85)",
          fontFamily: "monospace",
          fontSize: "11px",
          letterSpacing: "0.2em",
          textTransform: "uppercase",
          cursor: "pointer",
          backdropFilter: "blur(8px)",
        }}
      >
        <CloseIcon />
        ESC
      </button>
    </motion.div>
  );

  return createPortal(node, document.body);
}

/* ── Tiny inline SVG icons ────────────────────────────────────────────────── */

function ExpandIcon() {
  return (
    <svg width="9" height="9" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5">
      <polyline points="1,5 1,1 5,1" />
      <polyline points="9,1 13,1 13,5" />
      <polyline points="13,9 13,13 9,13" />
      <polyline points="5,13 1,13 1,9" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg width="10" height="10" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.8">
      <line x1="2" y1="2" x2="12" y2="12" strokeLinecap="round" />
      <line x1="12" y1="2" x2="2" y2="12" strokeLinecap="round" />
    </svg>
  );
}
