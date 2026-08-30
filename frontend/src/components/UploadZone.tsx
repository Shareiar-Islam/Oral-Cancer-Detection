import { useCallback, useRef, useState } from 'react';
import type { ChangeEvent, DragEvent, KeyboardEvent } from 'react';
import { ACCEPTED_EXTENSIONS, MAX_UPLOAD_MB } from '../lib/fileValidation';

interface UploadZoneProps {
  onFileSelected: (file: File) => void;
  disabled: boolean;
}

export function UploadZone({ onFileSelected, disabled }: UploadZoneProps): JSX.Element {
  const inputRef = useRef<HTMLInputElement>(null);
  const cameraRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  // Nested dragenter/dragleave fire per child element; counting them keeps the
  // highlight from flickering as the pointer crosses the icon.
  const dragDepth = useRef(0);

  const openPicker = useCallback((): void => {
    if (!disabled) inputRef.current?.click();
  }, [disabled]);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>): void => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        openPicker();
      }
    },
    [openPicker],
  );

  const handleDragEnter = useCallback(
    (event: DragEvent<HTMLDivElement>): void => {
      event.preventDefault();
      if (disabled) return;
      dragDepth.current += 1;
      setIsDragging(true);
    },
    [disabled],
  );

  const handleDragLeave = useCallback((event: DragEvent<HTMLDivElement>): void => {
    event.preventDefault();
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (dragDepth.current === 0) setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (event: DragEvent<HTMLDivElement>): void => {
      event.preventDefault();
      dragDepth.current = 0;
      setIsDragging(false);
      if (disabled) return;
      const file = event.dataTransfer.files.item(0);
      if (file) onFileSelected(file);
    },
    [disabled, onFileSelected],
  );

  const handleInputChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>): void => {
      const file = event.target.files?.item(0);
      if (file) onFileSelected(file);
      // Reset so re-picking the same file fires change again.
      event.target.value = '';
    },
    [onFileSelected],
  );

  return (
    <div>
      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-disabled={disabled}
        aria-label="Upload an intraoral photograph. Activate to browse, or drop a file here."
        onClick={openPicker}
        onKeyDown={handleKeyDown}
        onDragEnter={handleDragEnter}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={[
          'group relative flex flex-col items-center justify-center overflow-hidden rounded-2xl px-6 py-14 text-center',
          'ring-1 transition-all duration-200',
          'focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-canvas',
          disabled
            ? 'cursor-not-allowed bg-surface/50 ring-line-soft'
            : isDragging
              ? 'scale-[1.01] cursor-copy bg-raised ring-2 ring-accent glow-accent'
              : 'cursor-pointer bg-surface ring-line hover:bg-raised hover:ring-faint',
        ].join(' ')}
      >
        {/* Dashed outline drawn inset so the ring above stays crisp. */}
        <span
          aria-hidden="true"
          className={`pointer-events-none absolute inset-3 rounded-xl border border-dashed transition-colors ${
            isDragging ? 'border-accent/40' : 'border-line'
          }`}
        />

        <span
          aria-hidden="true"
          className={`relative mb-5 flex h-14 w-14 items-center justify-center rounded-full transition-all duration-200 ${
            disabled
              ? 'bg-raised text-faint'
              : isDragging
                ? 'scale-110 text-on-accent [background-image:linear-gradient(135deg,var(--color-accent),var(--color-accent2))]'
                : 'bg-raised text-muted group-hover:text-on-accent group-hover:[background-image:linear-gradient(135deg,var(--color-accent),var(--color-accent2))]'
          }`}
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} className="h-6 w-6">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 16.5V4.5m0 0L7.5 9M12 4.5 16.5 9M3.75 16.5v1.875A2.625 2.625 0 0 0 6.375 21h11.25a2.625 2.625 0 0 0 2.625-2.625V16.5"
            />
          </svg>
        </span>

        <p className="font-display relative text-lg font-bold tracking-tight text-ink">
          {isDragging ? 'Release to upload' : 'Drop a photograph here'}
        </p>
        <p className="relative mt-1.5 text-sm text-muted">
          or{' '}
          <span className="font-medium text-ink underline decoration-faint underline-offset-4">
            browse your files
          </span>
        </p>

        <p className="relative mt-5 flex flex-wrap items-center justify-center gap-x-2 gap-y-1 text-xs text-faint">
          {['JPEG', 'PNG', 'WebP', 'BMP', 'TIFF'].map((format) => (
            <span key={format} className="rounded-full bg-raised px-2 py-0.5 font-medium text-muted">
              {format}
            </span>
          ))}
          <span className="ml-1">up to {MAX_UPLOAD_MB} MB</span>
        </p>

        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_EXTENSIONS.join(',')}
          onChange={handleInputChange}
          disabled={disabled}
          className="sr-only"
          tabIndex={-1}
        />
      </div>

      {/* Separate input: `capture` opens the rear camera directly on mobile.
          Hidden on desktop, where it would only duplicate the file picker. */}
      <div className="mt-3 sm:hidden">
        <button
          type="button"
          onClick={() => cameraRef.current?.click()}
          disabled={disabled}
          className="flex w-full items-center justify-center gap-2 rounded-xl bg-surface px-4 py-3 text-sm font-semibold text-ink ring-1 ring-line transition-colors hover:bg-raised focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-canvas disabled:opacity-50"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} className="h-5 w-5" aria-hidden="true">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M6.827 6.175A2.31 2.31 0 0 1 5.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 0 0 2.25 2.25h15A2.25 2.25 0 0 0 21.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 0 0-1.134-.175 2.31 2.31 0 0 1-1.64-1.055l-.822-1.316a2.192 2.192 0 0 0-1.736-1.039 48.774 48.774 0 0 0-5.232 0 2.192 2.192 0 0 0-1.736 1.039l-.821 1.316Z"
            />
            <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 12.75a4.5 4.5 0 1 1-9 0 4.5 4.5 0 0 1 9 0Z" />
          </svg>
          Take a photo
        </button>
        <input
          ref={cameraRef}
          type="file"
          accept="image/*"
          capture="environment"
          onChange={handleInputChange}
          disabled={disabled}
          className="sr-only"
          tabIndex={-1}
        />
      </div>
    </div>
  );
}
