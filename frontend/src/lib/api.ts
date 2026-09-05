/**
 * Single-source HTTP helpers. Everything that calls the Flask backend goes
 * through these so status handling, JSON parsing, abort signals, and timeouts
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
  /** Optional headers to merge with the request defaults. */
  headers?: Record<string, string>;
  /** Hard timeout in milliseconds. Omit to use the browser/network default. */
  timeoutMs?: number;
}

async function parseBody<T>(response: Response): Promise<T> {
  const contentType = response.headers.get("content-type") ?? "";
  const raw = await response.text();

  if (!contentType.includes("application/json")) {
    const message = raw || "Expected a JSON response from the FrameVitals API.";
    throw new ApiError(
      response.status || 0,
      { error: message },
      `Unexpected response content type: ${contentType || "unknown"}`,
    );
  }

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

async function request<T>(url: string, init: RequestInit, opts: RequestOptions = {}): Promise<T> {
  let timeoutHandle: ReturnType<typeof setTimeout> | null = null;
  let removeAbortListener: (() => void) | null = null;
  let signal: AbortSignal | undefined = opts.signal;

  if (opts.timeoutMs && opts.timeoutMs > 0) {
    const controller = new AbortController();
    timeoutHandle = setTimeout(() => controller.abort(), opts.timeoutMs);

    if (opts.signal) {
      const callerSignal = opts.signal;
      const onAbort = () => controller.abort();
      if (callerSignal.aborted) {
        controller.abort();
      } else {
        callerSignal.addEventListener("abort", onAbort, { once: true });
        removeAbortListener = () => callerSignal.removeEventListener("abort", onAbort);
      }
    }
    signal = controller.signal;
  }

  const cleanup = () => {
    if (timeoutHandle) {
      clearTimeout(timeoutHandle);
      timeoutHandle = null;
    }
    if (removeAbortListener) {
      removeAbortListener();
      removeAbortListener = null;
    }
  };

  let response: Response;
  try {
    response = await fetch(url, { ...init, signal });
  } catch (err) {
    cleanup();
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError(
        0,
        { error: `Request to ${url} aborted (likely a timeout)` },
        `Request to ${url} aborted`,
      );
    }
    throw err;
  }
  cleanup();

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
      // Do not set Content-Type: the browser supplies the multipart boundary.
      headers: { Accept: "application/json", ...(opts.headers ?? {}) },
      body: fd,
    },
    opts,
  );
}
