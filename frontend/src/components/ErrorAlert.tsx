import type { ApiError } from '../lib/api';

interface ErrorAlertProps {
  error: ApiError;
  onRetry?: (() => void) | undefined;
  onDismiss?: (() => void) | undefined;
}

/** Short, plain-language heading per failure class. */
function headingFor(code: string): string {
  switch (code) {
    case 'NETWORK_ERROR':
      return 'Cannot reach the server';
    case 'TIMEOUT':
      return 'The request timed out';
    case 'MODEL_NOT_LOADED':
    case 'MODEL_LOAD_FAILED':
      return 'Model unavailable';
    case 'FILE_TOO_LARGE':
      return 'File too large';
    case 'UNSUPPORTED_MEDIA_TYPE':
    case 'CLIENT_VALIDATION':
      return 'Unsupported file type';
    case 'INVALID_IMAGE':
      return 'Could not read that image';
    case 'EMPTY_UPLOAD':
      return 'That file is empty';
    default:
      return 'Something went wrong';
  }
}

export function ErrorAlert({ error, onRetry, onDismiss }: ErrorAlertProps): JSX.Element {
  const showRetry = Boolean(onRetry) && error.isRetryable;

  return (
    <div role="alert" className="animate-rise rounded-xl bg-alert-tint p-4 ring-1 ring-alert-strong">
      <div className="flex gap-3">
        <span
          aria-hidden="true"
          className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-alert text-canvas"
        >
          <svg viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 1 0 0-16 8 8 0 0 0 0 16Zm0-13a.75.75 0 0 1 .75.75v4.5a.75.75 0 0 1-1.5 0v-4.5A.75.75 0 0 1 10 5Zm0 9a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z"
              clipRule="evenodd"
            />
          </svg>
        </span>

        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-semibold text-alert">{headingFor(error.code)}</h3>
          <p className="mt-1 text-sm leading-relaxed text-muted">{error.message}</p>

          {(showRetry || onDismiss) && (
            <div className="mt-3 flex gap-2">
              {showRetry && onRetry && (
                <button
                  type="button"
                  onClick={onRetry}
                  className="rounded-lg bg-alert px-3 py-1.5 text-sm font-semibold text-canvas transition-colors hover:bg-alert-strong hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-alert focus-visible:ring-offset-2 focus-visible:ring-offset-canvas"
                >
                  Try again
                </button>
              )}
              {onDismiss && (
                <button
                  type="button"
                  onClick={onDismiss}
                  className="rounded-lg px-3 py-1.5 text-sm font-medium text-muted transition-colors hover:bg-alert-deep hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-faint"
                >
                  Dismiss
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
