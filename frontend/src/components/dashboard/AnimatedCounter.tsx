import { useEffect, useState } from "react";

interface AnimatedCounterProps {
  value: number;
  duration?: number;
  precision?: number;
  className?: string;
}

export function AnimatedCounter({ value, duration = 700, precision = 0, className }: AnimatedCounterProps) {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    let animationFrame = 0;
    const startValue = 0;
    const startTime = performance.now();

    const step = (now: number) => {
      const progress = Math.min((now - startTime) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplayValue(startValue + (value - startValue) * eased);

      if (progress < 1) {
        animationFrame = window.requestAnimationFrame(step);
      }
    };

    setDisplayValue(startValue);
    animationFrame = window.requestAnimationFrame(step);

    return () => {
      window.cancelAnimationFrame(animationFrame);
    };
  }, [duration, value]);

  const formattedValue =
    precision > 0
      ? displayValue.toLocaleString(undefined, {
          minimumFractionDigits: precision,
          maximumFractionDigits: precision,
        })
      : Math.round(displayValue).toLocaleString();

  return <span className={className}>{formattedValue}</span>;
}
