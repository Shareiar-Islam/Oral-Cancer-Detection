import axios, { AxiosError, type AxiosInstance } from 'axios';
import type { ApiErrorBody, ApiErrorCode, PredictionResponse } from '../types';

const BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

/** Uploads are large and inference is CPU-bound; allow generous headroom. */
const PREDICT_TIMEOUT_MS = 60_000;

/**
 * A failure the UI can render: always carries a stable `code` plus a message
 * safe to show a user, whether it came from the API, the network, or us.
 */
export class ApiError extends Error {
  readonly code: ApiErrorCode;
  readonly status: number | undefined;

  constructor(code: ApiErrorCode, message: string, status?: number) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
  }

  /** True when retrying the same request could plausibly succeed. */
  get isRetryable(): boolean {
    return (
      this.code === 'NETWORK_ERROR' ||
      this.code === 'TIMEOUT' ||
      this.code === 'INTERNAL_ERROR' ||
      this.code === 'INFERENCE_FAILED'
    );
  }
}

const client: AxiosInstance = axios.create({ baseURL: BASE_URL });

/**
 * Detects the two misconfigurations that only appear once deployed, where the
 * symptom ("backend offline") points nowhere near the cause.
 *
 * Returns null when the configuration is coherent.
 */
export function detectConfigProblem(): string | null {
  if (typeof window === 'undefined') return null;

  const pageOrigin = window.location.origin;
  const pageIsLocal = /^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:|$)/.test(pageOrigin);
  const apiIsLocal = /^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:|$)/.test(BASE_URL);

  // Deployed page still pointing at a developer machine: VITE_API_BASE_URL was
  // never set in the hosting project, so the default leaked into the build.
  if (!pageIsLocal && apiIsLocal) {
    return `This deployment is configured to call ${BASE_URL}, which only exists on a local machine. Set VITE_API_BASE_URL to your backend's public URL in your hosting project's environment variables and redeploy.`;
  }

  // A browser silently blocks an http:// request from an https:// page.
  if (pageOrigin.startsWith('https://') && BASE_URL.startsWith('http://')) {
    return `This page is served over HTTPS but the API URL (${BASE_URL}) is plain HTTP, which browsers block as mixed content. Use an https:// URL for VITE_API_BASE_URL.`;
  }

  return null;
}

function isApiErrorBody(value: unknown): value is ApiErrorBody {
  if (typeof value !== 'object' || value === null) return false;
  const candidate = (value as { error?: unknown }).error;
  if (typeof candidate !== 'object' || candidate === null) return false;
  const { code, message } = candidate as { code?: unknown; message?: unknown };
  return typeof code === 'string' && typeof message === 'string';
}

/** Human-facing wording, keyed by backend code. Falls back to the API's text. */
function friendlyMessage(code: string, fallback: string): string {
  switch (code) {
    case 'MODEL_NOT_LOADED':
    case 'MODEL_LOAD_FAILED':
      return 'The classification model is not available. The server started but could not load it — check the backend logs.';
    case 'FILE_TOO_LARGE':
      return fallback;
    case 'UNSUPPORTED_MEDIA_TYPE':
      return `${fallback} Try JPEG, PNG, WebP, BMP, or TIFF.`;
    case 'INVALID_IMAGE':
      return 'That file could not be read as an image. It may be corrupt, incomplete, or not an image at all.';
    case 'EMPTY_UPLOAD':
      return 'That file is empty. Please choose another image.';
    case 'INFERENCE_FAILED':
      return 'The model could not process this image. Please try a different one.';
    default:
      return fallback;
  }
}

/** Normalise anything thrown by axios into an ApiError. */
function toApiError(error: unknown): ApiError {
  if (error instanceof ApiError) return error;

  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<unknown>;

    if (axiosError.code === 'ECONNABORTED') {
      return new ApiError(
        'TIMEOUT',
        'The request took too long. The server may be overloaded — please try again.',
      );
    }

    // No response at all: server down, wrong URL, or CORS blocked it.
    if (!axiosError.response) {
      return new ApiError(
        'NETWORK_ERROR',
        `Could not reach the server at ${BASE_URL}. Check that the backend is running and that this origin is listed in ALLOWED_ORIGINS.`,
      );
    }

    const { status, data } = axiosError.response;
    if (isApiErrorBody(data)) {
      return new ApiError(
        data.error.code as ApiErrorCode,
        friendlyMessage(data.error.code, data.error.message),
        status,
      );
    }
    return new ApiError(
      'INTERNAL_ERROR',
      `The server returned an unexpected ${status} response.`,
      status,
    );
  }

  return new ApiError(
    'INTERNAL_ERROR',
    error instanceof Error ? error.message : 'An unexpected error occurred.',
  );
}

export async function predict(
  file: File,
  signal?: AbortSignal,
): Promise<PredictionResponse> {
  const form = new FormData();
  // Field name must be `file` -- it matches the backend's UploadFile parameter.
  form.append('file', file);

  try {
    const response = await client.post<PredictionResponse>('/api/predict', form, {
      timeout: PREDICT_TIMEOUT_MS,
      ...(signal ? { signal } : {}),
    });
    return response.data;
  } catch (error) {
    throw toApiError(error);
  }
}

export { BASE_URL };
