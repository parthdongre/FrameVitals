import { Component, type ReactNode } from "react";
import { EmptyState } from "./EmptyState";

interface Props {
  /**
   * Custom fallback. Receives the current error and a reset callback. If
   * omitted, a default `EmptyState` titled "This panel hit an error" renders.
   */
  fallback?: (error: Error, reset: () => void) => ReactNode;
  /**
   * Friendly label shown by the default fallback (e.g. "ML Lab"). The label
   * is also used in dev-mode console diagnostics.
   */
  label?: string;
  /**
   * Optional callback invoked with the captured error. Useful for routing to
   * a telemetry hook later without coupling this component to one now.
   */
  onError?: (error: Error, info: { componentStack?: string }) => void;
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * Per-panel error boundary. The previous rebuild attempt white-screened on
 * analyze because a single render error in one tab took down the whole page.
 * Wrapping every report sub-tab in this class component caps the blast radius
 * to that one tab — the rest of the report stays usable, and the user can
 * try again with the inline reset button.
 *
 * In dev mode the error is logged to the console with the panel label so
 * you know exactly which tab fell over. In prod we stay silent.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: { componentStack?: string }) {
    if (import.meta.env?.DEV) {
      // eslint-disable-next-line no-console
      console.error(
        `[ErrorBoundary${this.props.label ? `:${this.props.label}` : ""}]`,
        error,
        info,
      );
    }
    if (this.props.onError) {
      try {
        this.props.onError(error, info);
      } catch {
        /* swallow telemetry errors */
      }
    }
  }

  reset = () => this.setState({ error: null });

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    if (this.props.fallback) {
      return this.props.fallback(error, this.reset);
    }

    return (
      <div className="space-y-4">
        <EmptyState
          title="This panel hit an error"
          hint={
            <>
              The rest of the report is still available. {" "}
              <button
                type="button"
                onClick={this.reset}
                className="text-accent underline-offset-4 hover:underline"
              >
                Try again
              </button>
              .
            </>
          }
        />
        {import.meta.env?.DEV ? (
          <details className="rounded-md border border-line bg-bg-1 p-4 text-[12px] text-ink-3">
            <summary className="cursor-pointer text-ink-2">Error details (dev only)</summary>
            <pre className="mt-2 overflow-x-auto whitespace-pre-wrap font-mono text-[11px] leading-5">
              {error.stack ?? error.message}
            </pre>
          </details>
        ) : null}
      </div>
    );
  }
}
