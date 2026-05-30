import { useEffect, useRef, useState } from "react";

interface InViewOptions {
  /**
   * Margin around the root for triggering the observer. Same syntax as the
   * underlying IntersectionObserver `rootMargin`.
   */
  rootMargin?: string;
  /**
   * Visibility ratio (0–1) at which the element is considered visible.
   */
  threshold?: number;
  /**
   * If true, only fires once and disconnects.
   */
  once?: boolean;
}

/**
 * Lightweight `useInView` wrapper around IntersectionObserver. We expose a
 * mutable ref the caller attaches to the target element, plus a boolean.
 *
 * framer-motion ships its own `useInView`; this one exists so non-framer
 * consumers (e.g. the count-up tween, lazy-render guards) can use the same
 * primitive without pulling in framer's runtime.
 */
export function useInView<T extends Element = HTMLElement>(
  options: InViewOptions = {},
): [React.MutableRefObject<T | null>, boolean] {
  const { rootMargin = "0px", threshold = 0.2, once = true } = options;
  const ref = useRef<T | null>(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") {
      // Conservative fallback: assume visible so animations don't get stuck
      // in their initial state on environments without the API.
      setInView(true);
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setInView(true);
            if (once) observer.disconnect();
          } else if (!once) {
            setInView(false);
          }
        }
      },
      { rootMargin, threshold },
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, [rootMargin, threshold, once]);

  return [ref, inView];
}
