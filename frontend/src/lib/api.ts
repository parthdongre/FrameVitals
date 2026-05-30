/**
 * Single-source HTTP helpers. Everything that calls the Flask backend goes
 * through these so error semantics (status, JSON body parsing, abort signals)
 * stay consistent across hooks and pages.
 */

export interface ApiErrorBody {
  error?: string;
  [key: string]: unknown;
}

export class ApiError extends Error {
  status: number;
  body: ApiErrorBody;

  constructor(status: number, body: ApiErrorBody, message?: string) {
    let errMsg = message ?? body.error;
    if (typeof errMsg === "object") {
      errMsg = JSON.stringify(errMsg);
    }
    super(typeof errMsg === "string" ? errMsg : `Request failed (${status})`);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

interface RequestOptions {
  signal?: AbortSignal;
  /**
   * Optional headers to merge with the defaults. `Content-Type` is set
   * automatically for JSON bodies.
   */
  headers?: Record<string, string>;
  /**
   * Hard timeout in milliseconds. Defaults to no timeout (the browser's own
   * default, typically 5 minutes). Use a large value for analyze, smaller
   * for ask / health.
   */
  timeoutMs?: number;
}

async function parseBody<T>(response: Response): Promise<T> {
  const contentType = response.headers.get("content-type") ?? "";
  // Prefer the streaming-friendly response.json() when we know it's JSON;
  // fall back to text() (then JSON.parse) for anything else so we can still
  // surface a backend error page that came back as text/html.
  if (contentType.includes("application/json")) {
    // Read the body as text first so we can both parse JSON *and* preserve
    // the raw payload for error diagnostics. Calling response.json() and
    // then trying to read text() afterwards (even via clone) is fragile —
    // an empty body makes json() throw and our recovery path was returning
    // an empty {} that callers were treating as success.
    const raw = await response.text();
    if (!raw) {
      throw new ApiError(
        response.status || 0,
        { error: "Empty response body" },
        "Empty response body",
      );
    }
    try {
      return JSON.parse(raw) as T;
    } catch (err) {
      throw new ApiError(
        response.status || 0,
        { error: raw },
        `Failed to parse JSON: ${(err as Error).message}`,
      );
    }
  }

  const raw = await response.text();
  if (!raw) return {} as T;
  return { error: raw } as unknown as T;
}

async function request<T>(url: string, init: RequestInit, opts: RequestOptions = {}): Promise<T> {
  // Compose the caller's signal with our optional timeout-driven abort.
  let timeoutHandle: ReturnType<typeof setTimeout> | null = null;
  let signal: AbortSignal | undefined = opts.signal;
  if (opts.timeoutMs && opts.timeoutMs > 0) {
    const ctrl = new AbortController();
    timeoutHandle = setTimeout(() => ctrl.abort(), opts.timeoutMs);
    if (opts.signal) {
      // If the caller already provided a signal, hook into it too.
      const onAbort = () => ctrl.abort();
      if (opts.signal.aborted) ctrl.abort();
      else opts.signal.addEventListener("abort", onAbort, { once: true });
    }
    signal = ctrl.signal;
  }

  let response: Response;
  try {
    response = await fetch(url, { ...init, signal });
  } catch (err) {
    if (timeoutHandle) clearTimeout(timeoutHandle);
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError(
        0,
        { error: `Request to ${url} aborted (likely a timeout)` },
        `Request to ${url} aborted`,
      );
    }
    throw err;
  }

  if (timeoutHandle) clearTimeout(timeoutHandle);

  const body = await parseBody<T & ApiErrorBody>(response);
  if (!response.ok) {
    throw new ApiError(response.status, body, body.error);
  }
  return body;
}

export function getJSON<T>(url: string, opts: RequestOptions = {}): Promise<T> {
  return request<T>(
    url,
    {
      method: "GET",
      headers: { Accept: "application/json", ...(opts.headers ?? {}) },
    },
    opts,
  );
}

export function postJSON<T>(url: string, body: unknown, opts: RequestOptions = {}): Promise<T> {
  return request<T>(
    url,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        ...(opts.headers ?? {}),
      },
      body: JSON.stringify(body ?? {}),
    },
    opts,
  );
}

export function postFormData<T>(url: string, fd: FormData, opts: RequestOptions = {}): Promise<T> {
  return request<T>(
    url,
    {
      method: "POST",
      // NOTE: do NOT set Content-Type; the browser fills in the multipart
      // boundary automatically when the body is a FormData instance.
      headers: { Accept: "application/json", ...(opts.headers ?? {}) },
      body: fd,
    },
    opts,
  );
}
