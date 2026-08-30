import { useCallback, useState } from 'react';
import { AnalyzeButton } from './components/AnalyzeButton';
// import { DisclaimerBanner } from './components/DisclaimerBanner';
import { ErrorAlert } from './components/ErrorAlert';
import { ImagePreview } from './components/ImagePreview';
import { ResultCard } from './components/ResultCard';
import { UploadZone } from './components/UploadZone';
import { usePrediction } from './hooks/usePrediction';
import { ApiError, detectConfigProblem } from './lib/api';
import { validateFile } from './lib/fileValidation';

/**
 * There is deliberately no backend health polling here.
 *
 * A readiness probe on page load adds a request on every visit, races the
 * user's first action, and reports "offline" for transient conditions that
 * would have resolved by the time they pressed Analyze. Connection problems
 * surface where they matter instead — as an error on the request the user
 * actually made, with a retry attached.
 */
export default function App(): JSX.Element {
  const [file, setFile] = useState<File | null>(null);
  const [validationError, setValidationError] = useState<ApiError | null>(null);

  const { status, result, error, isLoading, analyze, reset, retry } = usePrediction();

  // Build-time configuration check, not a liveness probe: silent unless the
  // deployed bundle points somewhere unreachable by construction.
  const configProblem = detectConfigProblem();

  const handleFileSelected = useCallback(
    (selected: File): void => {
      const problem = validateFile(selected);
      if (problem) {
        setValidationError(problem);
        setFile(null);
        reset();
        return;
      }
      setValidationError(null);
      // Clear any previous verdict so a stale result never sits beside a new image.
      reset();
      setFile(selected);
    },
    [reset],
  );

  const handleRemove = useCallback((): void => {
    setFile(null);
    setValidationError(null);
    reset();
  }, [reset]);

  const handleAnalyze = useCallback((): void => {
    if (file) void analyze(file);
  }, [analyze, file]);

  const displayedError = validationError ?? error;

  return (
    <div className="min-h-screen bg-canvas">
      {/* <DisclaimerBanner /> */}

      <header className="border-b border-line bg-surface/70 backdrop-blur-xl">
        <div className="mx-auto flex max-w-5xl items-center gap-3 px-4 py-5 sm:px-6">
          <span
            aria-hidden="true"
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl text-on-accent" style={{ backgroundImage: 'linear-gradient(135deg, var(--color-accent), var(--color-accent2))' }}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.7} className="h-5 w-5">
              <circle cx="11" cy="11" r="6.5" />
              <path strokeLinecap="round" d="m16 16 4.5 4.5" />
            </svg>
          </span>
          <div className="min-w-0">
            <h1 className="font-display text-xl font-bold tracking-tight text-ink">
              Oral Cancer Classifier
            </h1>
            <p className="mt-0.5 truncate text-sm text-muted">
              Binary screening support from intraoral photographs · EfficientNet-B0
            </p>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-4 py-8 sm:px-6 sm:py-10">
        {configProblem && (
          <div role="alert" className="mb-6 rounded-xl bg-warn-tint p-4 ring-1 ring-warn-line">
            <h2 className="text-sm font-semibold text-warn">Configuration problem</h2>
            <p className="mt-1 text-sm leading-relaxed text-muted">{configProblem}</p>
          </div>
        )}

        <div className="grid gap-6 lg:grid-cols-2 lg:gap-8">
          {/* Left column: input */}
          <div className="space-y-4">
            <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-faint">
              Photograph
            </h2>

            {file ? (
              <ImagePreview file={file} onRemove={handleRemove} disabled={isLoading} />
            ) : (
              <UploadZone onFileSelected={handleFileSelected} disabled={false} />
            )}

            <AnalyzeButton onClick={handleAnalyze} disabled={!file} isLoading={isLoading} />

            {displayedError && (
              <ErrorAlert
                error={displayedError}
                onRetry={validationError ? undefined : () => void retry()}
                onDismiss={() => {
                  setValidationError(null);
                  if (status === 'error') reset();
                }}
              />
            )}
          </div>

          {/* Right column: output */}
          <div className="space-y-4">
            <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-faint">
              Analysis
            </h2>

            {result ? (
              <ResultCard result={result} />
            ) : (
              <div className="flex min-h-56 flex-col items-center justify-center rounded-2xl border border-dashed border-line bg-surface/40 px-6 py-14 text-center">
                {isLoading ? (
                  <>
                    <svg
                      aria-hidden="true"
                      viewBox="0 0 24 24"
                      className="mb-3 h-6 w-6 animate-spin text-muted"
                    >
                      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" fill="none" opacity="0.25" />
                      <path fill="currentColor" d="M12 2a10 10 0 0 1 10 10h-3.5A6.5 6.5 0 0 0 12 5.5V2Z" />
                    </svg>
                    <p className="text-sm font-medium text-muted">Analyzing the photograph…</p>
                  </>
                ) : (
                  <>
                    <span
                      aria-hidden="true"
                      className="mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-raised text-faint"
                    >
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} className="h-5 w-5">
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M3 13.5 8.25 8.25l4.5 4.5L21 4.5M21 4.5h-5.25M21 4.5v5.25"
                        />
                      </svg>
                    </span>
                    <p className="text-sm text-faint">
                      Results appear here once you analyze an image.
                    </p>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="mx-auto max-w-5xl px-4 pb-10 sm:px-6">
        <p className="border-t border-line-soft pt-6 text-xs leading-relaxed text-faint">
          Research prototype — not a diagnostic device. Not validated for clinical use;
          output must not replace evaluation by a qualified clinician. Uploaded images are
          processed in memory and are never written to disk or retained by the server.
        </p>
      </footer>
    </div>
  );
}
