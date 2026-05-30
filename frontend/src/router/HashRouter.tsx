import { useEffect, useState, type AnchorHTMLAttributes, type MouseEvent } from "react";
import type { Route } from "@/components/site/SiteShell";

/**
 * Hash router — stays explicitly off `react-router`.
 *
 * URL shape:
 *   #/                            → { route: "home" }
 *   #/analyze                     → { route: "analyze" }
 *   #/report                      → { route: "report", tab: undefined }
 *   #/report?tab=overview         → { route: "report", tab: "overview" }
 *   #/charts | #/modules | #/about
 *
 * The `route` union is shared with `SiteShell` so the existing pages keep
 * working unchanged.
 */

const ROUTES: Route[] = ["home", "analyze", "report", "charts", "modules", "about"];

export interface HashRoute {
  route: Route;
  tab?: string;
}

export function parseHash(hash: string = window.location.hash): HashRoute {
  // Strip the leading "#/" or "#" if present.
  const raw = hash.replace(/^#\/?/, "").trim();
  if (!raw) return { route: "home" };

  // Split path from query.
  const [pathRaw, queryRaw = ""] = raw.split("?");
  const path = pathRaw.replace(/^\/+|\/+$/g, "");

  let route: Route = "home";
  if (path === "" || path === "home") route = "home";
  else if ((ROUTES as string[]).includes(path)) route = path as Route;
  else route = "home";

  const params = new URLSearchParams(queryRaw);
  const tab = params.get("tab") ?? undefined;

  return { route, tab };
}

export function buildHash(route: Route, tab?: string): string {
  const path = route === "home" ? "/" : `/${route}`;
  if (!tab) return `#${path}`;
  return `#${path}?tab=${encodeURIComponent(tab)}`;
}

export function useHashRoute(): HashRoute & {
  navigate: (route: Route, tab?: string) => void;
} {
  const [state, setState] = useState<HashRoute>(() => parseHash());

  useEffect(() => {
    const onHashChange = () => setState(parseHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const navigate = (route: Route, tab?: string) => {
    const next = buildHash(route, tab);
    if (window.location.hash !== next) {
      window.location.hash = next;
    } else {
      // Same hash — force a re-render anyway in case caller wants to scroll.
      setState({ route, tab });
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return { ...state, navigate };
}

interface LinkProps extends Omit<AnchorHTMLAttributes<HTMLAnchorElement>, "href"> {
  to: Route;
  tab?: string;
}

/**
 * Hash-aware anchor. Renders a real <a> with the correct href so the link is
 * accessible (right-click "open in new tab", screen readers, etc.) but
 * intercepts plain left-clicks to call `navigate` for instant transitions.
 */
export function Link({ to, tab, onClick, ...rest }: LinkProps) {
  const handleClick = (e: MouseEvent<HTMLAnchorElement>) => {
    if (onClick) onClick(e);
    if (e.defaultPrevented) return;
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    if (e.button !== 0) return;
    e.preventDefault();
    window.location.hash = buildHash(to, tab);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  return <a href={buildHash(to, tab)} onClick={handleClick} {...rest} />;
}
