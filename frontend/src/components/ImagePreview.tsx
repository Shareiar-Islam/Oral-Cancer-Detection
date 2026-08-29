import { useObjectUrl } from '../hooks/useObjectUrl';
import { formatBytes } from '../lib/fileValidation';

interface ImagePreviewProps {
  file: File;
  onRemove: () => void;
  disabled: boolean;
}

export function ImagePreview({ file, onRemove, disabled }: ImagePreviewProps): JSX.Element {
  const url = useObjectUrl(file);

  return (
    <figure className="animate-rise overflow-hidden rounded-2xl bg-surface ring-1 ring-line">
      <div className="relative flex items-center justify-center bg-canvas">
        {url ? (
          <img
            src={url}
            alt={`Selected photograph: ${file.name}`}
            className="max-h-[22rem] w-full object-contain"
          />
        ) : (
          <div className="h-[22rem] w-full animate-pulse bg-raised" />
        )}

        <button
          type="button"
          onClick={onRemove}
          disabled={disabled}
          aria-label={`Remove ${file.name}`}
          className="absolute right-3 top-3 flex items-center gap-1.5 rounded-full bg-canvas/80 px-3 py-1.5 text-xs font-medium text-ink ring-1 ring-line backdrop-blur-sm transition-colors hover:bg-canvas focus:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:opacity-40"
        >
          <svg viewBox="0 0 20 20" fill="currentColor" className="h-3.5 w-3.5" aria-hidden="true">
            <path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
          </svg>
          Remove
        </button>
      </div>

      <figcaption className="px-4 py-3.5">
        <div className="flex items-center justify-between gap-4">
          <p className="min-w-0 truncate text-sm font-medium text-ink" title={file.name}>
            {file.name}
          </p>
          <p className="shrink-0 font-mono text-xs tabular-nums text-faint">
            {formatBytes(file.size)}
          </p>
        </div>

        {/* The model never sees this image at full resolution. Saying so up
            front explains why fine detail cannot influence the result. */}
        <p className="mt-2.5 flex items-start gap-2 border-t border-line-soft pt-2.5 text-xs leading-relaxed text-faint">
          <svg viewBox="0 0 20 20" fill="currentColor" className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true">
            <path
              fillRule="evenodd"
              d="M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0Zm-7-4a1 1 0 1 1-2 0 1 1 0 0 1 2 0ZM9 9a.75.75 0 0 0 0 1.5h.25v2.25H9a.75.75 0 0 0 0 1.5h2a.75.75 0 0 0 0-1.5h-.25V9.75A.75.75 0 0 0 10 9H9Z"
              clipRule="evenodd"
            />
          </svg>
          The model analyses a 224 × 224 version of this image. Detail finer than that
          is not available to it.
        </p>
      </figcaption>
    </figure>
  );
}
