interface ProbabilityBarProps {
  /** P(Cancer), 0–1. */
  probability: number;
  threshold: number;
  isPositive: boolean;
}

/**
 * P(Cancer) on a fixed 0–1 track with the decision threshold marked, so the
 * reading is always "how far past the line", not a bare number. The scale never
 * rescales to the value — that would make every result look equally extreme.
 */
export function ProbabilityBar({
  probability,
  threshold,
  isPositive,
}: ProbabilityBarProps): JSX.Element {
  const percent = Math.max(0, Math.min(100, probability * 100));
  const thresholdPercent = Math.max(0, Math.min(100, threshold * 100));

  // Roughly the width the "threshold 0.00" caption needs, as a share of track.
  const LABEL_CLEARANCE_PERCENT = 24;
  const nearStart = thresholdPercent < LABEL_CLEARANCE_PERCENT;
  const nearEnd = thresholdPercent > 100 - LABEL_CLEARANCE_PERCENT;

  return (
    <div>
      <div className="mb-2.5 flex items-baseline justify-between">
        <span className="text-[11px] font-semibold uppercase tracking-[0.08em] text-faint">
          P(Cancer)
        </span>
        <span className="font-mono text-sm font-semibold tabular-nums text-ink">
          {probability.toFixed(2)}
        </span>
      </div>

      <div
        role="img"
        aria-label={`Probability of cancer: ${probability.toFixed(2)}. Decision threshold: ${threshold.toFixed(2)}.`}
        className="relative h-2.5 w-full overflow-hidden rounded-full bg-canvas ring-1 ring-inset ring-line"
      >
        <div
          className={`h-full rounded-full transition-[width] duration-700 ease-out ${
            isPositive ? 'bg-alert' : 'bg-accent'
          }`}
          style={{ width: `${percent}%` }}
        />
        {/* Threshold marker sits above the fill so it stays visible either side. */}
        <div
          className="absolute inset-y-0 w-px bg-muted"
          style={{ left: `${thresholdPercent}%` }}
          aria-hidden="true"
        />
      </div>

      {/* The caption is centred on its marker, so near either end it would
          collide with the 0.00 / 1.00 endpoints. Anchor it to the edge instead
          and drop the endpoint it would have overlapped. */}
      <div className="relative mt-2 h-4">
        {!nearStart && <span className="absolute left-0 text-[11px] text-faint">0.00</span>}
        <span
          className={`absolute whitespace-nowrap text-[11px] font-medium text-muted ${
            nearStart ? '' : nearEnd ? '-translate-x-full' : '-translate-x-1/2'
          }`}
          style={{ left: nearStart ? 0 : `${thresholdPercent}%` }}
        >
          threshold {threshold.toFixed(2)}
        </span>
        {!nearEnd && <span className="absolute right-0 text-[11px] text-faint">1.00</span>}
      </div>
    </div>
  );
}
