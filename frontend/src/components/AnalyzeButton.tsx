interface AnalyzeButtonProps {
  onClick: () => void;
  disabled: boolean;
  isLoading: boolean;
}

export function AnalyzeButton({ onClick, disabled, isLoading }: AnalyzeButtonProps): JSX.Element {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled || isLoading}
      aria-busy={isLoading}
      className="flex w-full items-center justify-center gap-2.5 rounded-xl bg-accent px-4 py-3.5 text-sm font-semibold tracking-tight text-on-accent transition-all hover:bg-accent-hover focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-canvas active:scale-[0.99] disabled:cursor-not-allowed disabled:bg-raised disabled:text-faint"
    >
      {isLoading ? (
        <>
          <svg aria-hidden="true" viewBox="0 0 24 24" className="h-4 w-4 animate-spin">
            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3.5" fill="none" opacity="0.25" />
            <path fill="currentColor" d="M12 2a10 10 0 0 1 10 10h-3.5A6.5 6.5 0 0 0 12 5.5V2Z" />
          </svg>
          Analyzing…
        </>
      ) : (
        <>
          Analyze image
          <svg aria-hidden="true" viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4 opacity-60">
            <path
              fillRule="evenodd"
              d="M3 10a.75.75 0 0 1 .75-.75h10.638L10.23 5.29a.75.75 0 1 1 1.04-1.08l5.5 5.25a.75.75 0 0 1 0 1.08l-5.5 5.25a.75.75 0 1 1-1.04-1.08l4.158-3.96H3.75A.75.75 0 0 1 3 10Z"
              clipRule="evenodd"
            />
          </svg>
        </>
      )}
    </button>
  );
}
