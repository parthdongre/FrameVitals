import type { Config } from "tailwindcss";

/**
 * Tailwind 4 token map.
 *
 * Two namespaces are intentionally present:
 *
 * 1. Editorial tokens (`bg-bg-0`, `text-ink-1`, `border-line`, `text-accent`)
 *    — the v3 rebuild palette. Cream + near-black + single teal accent.
 *    All values resolve to CSS variables defined in `src/styles/globals.css`,
 *    so swapping themes is a one-line change.
 *
 * 2. Legacy palette (`space-*`, `cockpit-*`, `supernova-*`, `pulsar-*`,
 *    custom shadows, gradients, keyframes) — retained for now because
 *    the v3 dashboard panels still consume them. Each phase of the rebuild
 *    retires more of these in favor of the editorial namespace.
 */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // === v3 editorial tokens ===
        bg: {
          0: "var(--bg-0)",
          1: "var(--bg-1)",
          2: "var(--bg-2)",
          3: "var(--bg-3)",
        },
        line: {
          DEFAULT: "var(--line)",
          strong: "var(--line-strong)",
          accent: "var(--accent-line)",
        },
        ink: {
          1: "var(--ink-1)",
          2: "var(--ink-2)",
          3: "var(--ink-3)",
          4: "var(--ink-4)",
        },
        accent: {
          DEFAULT: "var(--accent)",
          soft: "var(--accent-soft)",
          line: "var(--accent-line)",
          strong: "var(--accent-strong)",
          glow: "var(--accent-glow)",
        },
        status: {
          ok: "var(--ok)",
          warn: "var(--warn)",
          bad: "var(--bad)",
        },

        // === legacy palette (retained until each component is restyled) ===
        space: {
          950: "#030712",
          900: "#0f172a",
          800: "#111827",
          700: "#1e293b",
          600: "#334155",
        },
        cockpit: {
          line: "rgba(255,255,255,0.05)",
          panel: "rgba(255,255,255,0.03)",
          glow: "rgba(6, 182, 212, 0.18)",
        },
        supernova: {
          100: "#ecfeff",
          200: "#cffafe",
          300: "#a5f3fc",
          400: "#22d3ee",
          500: "#06b6d4",
          600: "#0891b2",
        },
        pulsar: {
          300: "#c4b5fd",
          400: "#a78bfa",
          500: "#8b5cf6",
          600: "#7c3aed",
        },
      },
      fontFamily: {
        // v3 editorial
        display: ["Inter", "system-ui", "sans-serif"],
        // shared mono / legacy sans alias
        sans: ["Inter", "Geist", "Satoshi", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      borderRadius: {
        sm: "var(--r-sm)",
        md: "var(--r-md)",
        lg: "var(--r-lg)",
      },
      transitionTimingFunction: {
        "out-soft": "cubic-bezier(0.22, 1, 0.36, 1)",
      },
      boxShadow: {
        panel: "0 1px 0 rgba(255,255,255,0.05), 0 24px 60px rgba(0,0,0,0.45)",
        halo: "0 0 0 1px rgba(6, 182, 212, 0.14), 0 0 28px rgba(6, 182, 212, 0.14)",
        warp: "0 0 0 1px rgba(6, 182, 212, 0.25), 0 0 42px rgba(6, 182, 212, 0.18)",
      },
      backgroundImage: {
        "radial-cockpit":
          "radial-gradient(circle at top left, rgba(6, 182, 212, 0.16), transparent 30%), radial-gradient(circle at top right, rgba(139, 92, 246, 0.12), transparent 24%)",
        "void-grid":
          "linear-gradient(rgba(148, 163, 184, 0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(148, 163, 184, 0.04) 1px, transparent 1px)",
      },
      keyframes: {
        warp: {
          "0%": { opacity: "0.2", transform: "translateX(-30%) scaleX(0.92)" },
          "50%": { opacity: "1", transform: "translateX(0%) scaleX(1)", filter: "drop-shadow(0 0 10px #06b6d4)" },
          "100%": { opacity: "0.2", transform: "translateX(30%) scaleX(1.04)" },
        },
        scan: {
          "0%": { backgroundPosition: "-220% 0" },
          "100%": { backgroundPosition: "220% 0" },
        },
        orbit: {
          from: { transform: "rotate(0deg)" },
          to: { transform: "rotate(360deg)" },
        },
        trace: {
          "0%": { strokeDashoffset: "220" },
          "100%": { strokeDashoffset: "0" },
        },
      },
      animation: {
        warp: "warp 1.5s ease-in-out infinite",
        scan: "scan 1.55s linear infinite",
        orbit: "orbit 32s linear infinite",
        trace: "trace 700ms ease-out forwards",
      },
    },
  },
  plugins: [],
} satisfies Config;
