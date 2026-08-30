import { useState } from 'react';
import type { PredictionResponse } from '../types';
import { ProbabilityBar } from './ProbabilityBar';

interface ResultCardProps {
  result: PredictionResponse;
}

/** 42 ms stays milliseconds; 4198 ms reads better as 4.2 s. */
function formatDuration(ms: number): { value: string; unit: string } {
  if (ms < 1000) return { value: ms.toFixed(0), unit: 'ms' };
  return { value: (ms / 1000).toFixed(1), unit: 's' };
}

/** A probability is easier to weigh as a percentage than as 0.039882. */
function formatPercent(p: number): string {
  if (p > 0 && p < 0.01) return '<1%';
  if (p < 1 && p > 0.99) return '>99%';
  return `${(p * 100).toFixed(p * 100 < 10 ? 1 : 0)}%`;
}

function DetailRow({ label, value }: { label: string; value: string }): JSX.Element {
  return (
    <div className="flex items-baseline justify-between gap-4 py-2">
      <dt className="text-faint">{label}</dt>
      <dd className="text-right font-mono tabular-nums text-muted">{value}</dd>
    </div>
  );
}

function StatTile({ label, value, unit }: { label: string; value: string; unit?: string }): JSX.Element {
  return (
    <div className="rounded-2xl bg-canvas/60 px-4 py-3.5 ring-1 ring-inset ring-line">
      <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-faint">{label}</p>
      <p className="mt-1.5 truncate font-mono text-lg font-semibold tabular-nums text-ink">
        {value}
        {unit && <span className="ml-1 text-xs font-normal text-faint">{unit}</span>}
      </p>
    </div>
  );
}

export function ResultCard({ result }: ResultCardProps): JSX.Element {
  const [showDetails, setShowDetails] = useState(false);

  // Derived from probability and threshold rather than string-matching the
  // label, so a renamed class in the checkpoint cannot break the styling.
  const isPositive = result.probability >= result.threshold;

  return (
    <section
      aria-labelledby="result-heading"
      className={`animate-rise overflow-hidden rounded-2xl bg-surface ring-1 ${
        isPositive ? 'ring-alert-strong glow-alert' : 'ring-line glow-accent'
      }`}
    >
      {/* On a dark ground the verdict is set apart by a tinted band and a
          coloured rule, not by brightness alone. */}
      <div
        className={`relative px-6 py-7 ${isPositive ? 'bg-alert-deep' : 'bg-raised'}`}
      >
        {/* Gradient hairline instead of a flat border: the one place the
            accent pair reappears outside the primary action. */}
        <span
          aria-hidden="true"
          className="absolute inset-x-0 bottom-0 h-px"
          style={{
            backgroundImage: isPositive
              ? 'linear-gradient(90deg, transparent, #ff6b5e, transparent)'
              : 'linear-gradient(90deg, transparent, var(--color-accent), var(--color-accent2), transparent)',
          }}
        />
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-3">
          <div className="min-w-0">
            <h2
              id="result-heading"
              className={`text-[11px] font-semibold uppercase tracking-[0.12em] ${
                isPositive ? 'text-alert' : 'text-faint'
              }`}
            >
              Classification
            </h2>
            <p
              aria-live="polite"
              className={`font-display mt-2 whitespace-nowrap text-[2rem] font-extrabold leading-none tracking-tight sm:text-[2.5rem] ${
                isPositive ? 'text-alert' : 'text-ink'
              }`}
            >
              {result.prediction}
            </p>
          </div>

          <span
            className={`shrink-0 rounded-full px-3 py-1 text-xs font-semibold tracking-wide ring-1 ${
              isPositive
                ? 'bg-alert-tint text-alert ring-alert-strong'
                : 'bg-canvas text-accent ring-accent/30'
            }`}
          >
            {formatPercent(result.confidence)} confident
          </span>
        </div>

        <p className="mt-3 text-xs leading-relaxed text-faint">
          Estimated {formatPercent(result.probability)} likelihood of cancer ·
          Screening support, not a diagnosis
        </p>
      </div>

      <div className="space-y-5 px-6 py-6">
        <ProbabilityBar
          probability={result.probability}
          threshold={result.threshold}
          isPositive={isPositive}
        />

        <div className="grid grid-cols-3 gap-3">
          <StatTile label="Confidence" value={formatPercent(result.confidence)} />
          <StatTile
            label="Image"
            value={`${result.image.original_size[0]}×${result.image.original_size[1]}`}
          />
          <StatTile
            label="Analysed in"
            value={formatDuration(result.inference_time_ms).value}
            unit={formatDuration(result.inference_time_ms).unit}
          />
        </div>

        <div className="border-t border-line-soft pt-4">
          <button
            type="button"
            onClick={() => setShowDetails((open) => !open)}
            aria-expanded={showDetails}
            aria-controls="technical-details"
            className="flex w-full items-center justify-between rounded-lg px-1 py-1 text-sm font-medium text-muted transition-colors hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            Technical details
            <svg
              aria-hidden="true"
              viewBox="0 0 20 20"
              fill="currentColor"
              className={`h-4 w-4 transition-transform duration-200 ${showDetails ? 'rotate-180' : ''}`}
            >
              <path
                fillRule="evenodd"
                d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z"
                clipRule="evenodd"
              />
            </svg>
          </button>

          {showDetails && (
            <div id="technical-details" className="animate-rise mt-3 space-y-4">
              <dl className="divide-y divide-line-soft text-xs">
                <DetailRow label="Raw model output (logit)" value={result.raw_output.toFixed(4)} />
                <DetailRow label="P(Cancer) after sigmoid" value={result.probability.toFixed(6)} />
                <DetailRow label="Decision threshold" value={result.threshold.toFixed(2)} />
                <DetailRow
                  label="Original image"
                  value={`${result.image.original_size[0]} × ${result.image.original_size[1]} px`}
                />
                <DetailRow
                  label="Model input"
                  value={`${result.image.processed_size[0]} × ${result.image.processed_size[1]} px`}
                />
                <DetailRow label="Source format" value={`${result.image.format} (${result.image.mode})`} />
                <DetailRow
                  label="EXIF orientation"
                  value={result.image.exif_corrected ? 'corrected' : 'none needed'}
                />
              </dl>

              <div className="rounded-xl bg-canvas p-4 ring-1 ring-inset ring-line-soft">
                <p className="mb-2.5 text-[11px] font-semibold uppercase tracking-[0.08em] text-faint">
                  Preprocessing applied
                </p>
                <ol className="space-y-1.5 font-mono text-[11px] leading-relaxed text-muted">
                  {[
                    'Decode + EXIF orientation',
                    'Convert to RGB',
                    `Resize to ${result.image.processed_size[0]} × ${result.image.processed_size[1]}`,
                    'ToTensor — scale to [0, 1]',
                    'Normalize — ImageNet mean/std',
                    'Batch to (1, 3, 224, 224)',
                  ].map((step, index) => (
                    <li key={step}>
                      <span className="mr-2 text-faint">
                        {String(index + 1).padStart(2, '0')}
                      </span>
                      {step}
                    </li>
                  ))}
                </ol>
              </div>

              <p className="text-xs leading-relaxed text-faint">
                A logit above 0 corresponds to P(Cancer) above 0.5. The probability is
                the sigmoid of the raw output; confidence is the probability of
                whichever class was predicted.
              </p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
