/**
 * Persistent and non-dismissible by design. The tool produces a clinical-
 * sounding verdict, so the limits of that verdict must be visible at the same
 * time as the result — never behind a dismissed banner.
 */
export function DisclaimerBanner(): JSX.Element {
  return (
    <div role="note" className="border-b border-warn-line bg-warn-tint px-4 py-3 sm:px-6">
      <div className="mx-auto flex max-w-5xl items-start gap-2.5">
        <svg
          aria-hidden="true"
          viewBox="0 0 20 20"
          className="mt-0.5 h-4 w-4 shrink-0 text-warn"
          fill="currentColor"
        >
          <path
            fillRule="evenodd"
            d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.19-1.458-1.515-2.625L8.485 2.495ZM10 5a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-1.5 0v-3.5A.75.75 0 0 1 10 5Zm0 9a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z"
            clipRule="evenodd"
          />
        </svg>
        <p className="text-xs leading-relaxed text-warn/90 sm:text-sm">
          <span className="font-semibold text-warn">
            Research prototype — not a diagnostic device.
          </span>{' '}
          This model has not been validated for clinical use and is not approved by any
          regulatory body. Its output must not be used for clinical decisions, and must
          not replace examination, biopsy, or diagnosis by a qualified clinician. Any oral
          lesion of concern warrants professional assessment regardless of what this tool
          reports.
        </p>
      </div>
    </div>
  );
}
