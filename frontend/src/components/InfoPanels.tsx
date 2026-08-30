/**
 * Supporting context below the main tool.
 *
 * The figures here mirror the metadata recorded inside the EXP-4A checkpoint
 * (visible at GET /api/model-info). They are written as static content rather
 * than fetched, so the page still makes no network request until the user
 * asks for a prediction. If you swap the checkpoint, update MODEL_FACTS.
 */

const MODEL_FACTS: ReadonlyArray<{ label: string; value: string }> = [
  { label: 'Architecture', value: 'EfficientNet-B0' },
  { label: 'Experiment', value: 'EXP-4A' },
  { label: 'Input', value: '224 × 224 RGB' },
  { label: 'Output', value: 'Single logit → sigmoid' },
  { label: 'Threshold', value: '0.50' },
  { label: 'Training set', value: 'Dataset02' },
  { label: 'External set', value: 'Dataset01' },
  { label: 'Best val. F1', value: '0.942' },
];

const STEPS: ReadonlyArray<{ n: string; title: string; body: string }> = [
  {
    n: '01',
    title: 'Upload',
    body: 'A photograph is read in memory. Orientation is corrected from EXIF so the model sees it the way you framed it.',
  },
  {
    n: '02',
    title: 'Normalise',
    body: 'Converted to RGB, resized to 224 × 224, and scaled with the same ImageNet statistics used during training.',
  },
  {
    n: '03',
    title: 'Classify',
    body: 'EfficientNet-B0 emits one logit. A sigmoid turns it into P(Cancer), compared against the 0.50 threshold.',
  },
];

const CAPTURE_TIPS: readonly string[] = [
  'Light the area evenly — avoid direct flash, which blows out detail and casts glare.',
  'Fill the frame with the lesion and the tissue immediately around it.',
  'Hold steady until focus locks; motion blur removes the texture the model relies on.',
  'Retract lips or cheek so nothing obscures the area of interest.',
];

const LIMITATIONS: readonly string[] = [
  'Trained on a specific dataset — accuracy on photographs unlike it is unknown.',
  'Sees a 224 × 224 image, so lesions smaller than a few pixels at that scale are invisible to it.',
  'Returns two classes only. It does not stage, subtype, or measure anything.',
  'A low probability is not clearance. Any lesion of concern needs a clinician.',
];

function Panel({
  title,
  children,
  className = '',
}: {
  title: string;
  children: React.ReactNode;
  className?: string;
}): JSX.Element {
  return (
    <section className={`rounded-2xl bg-surface p-5 ring-1 ring-line ${className}`}>
      <h2 className="font-display mb-4 text-sm font-bold tracking-tight text-ink">{title}</h2>
      {children}
    </section>
  );
}

export function InfoPanels(): JSX.Element {
  return (
    <div className="mt-10 space-y-5">
      {/* How it works — the pipeline, in the user's terms */}
      <Panel title="How it works">
        <ol className="grid gap-5 sm:grid-cols-3">
          {STEPS.map((step) => (
            <li key={step.n} className="relative">
              <span
                aria-hidden="true"
                className="font-display mb-2 block text-2xl font-extrabold leading-none"
                style={{
                  backgroundImage:
                    'linear-gradient(135deg, var(--color-accent), var(--color-accent2))',
                  WebkitBackgroundClip: 'text',
                  backgroundClip: 'text',
                  color: 'transparent',
                }}
              >
                {step.n}
              </span>
              <h3 className="mb-1 text-sm font-semibold text-ink">{step.title}</h3>
              <p className="text-xs leading-relaxed text-muted">{step.body}</p>
            </li>
          ))}
        </ol>
      </Panel>

      <div className="grid gap-5 lg:grid-cols-2">
        {/* Model card — the same values /api/model-info reports */}
        <Panel title="Model">
          <dl className="grid grid-cols-2 gap-x-5 gap-y-3">
            {MODEL_FACTS.map((fact) => (
              <div key={fact.label} className="min-w-0">
                <dt className="text-[11px] uppercase tracking-[0.08em] text-faint">
                  {fact.label}
                </dt>
                <dd className="mt-0.5 truncate font-mono text-xs text-ink" title={fact.value}>
                  {fact.value}
                </dd>
              </div>
            ))}
          </dl>
          <p className="mt-4 border-t border-line-soft pt-3 text-xs leading-relaxed text-faint">
            F1 is measured on a held-out validation split, not on new clinical cases.
            Performance elsewhere will differ.
          </p>
        </Panel>

        {/* Capture guidance — the highest-leverage thing a user controls */}
        <Panel title="Taking a usable photograph">
          <ul className="space-y-2.5">
            {CAPTURE_TIPS.map((tip) => (
              <li key={tip} className="flex gap-2.5 text-xs leading-relaxed text-muted">
                <span
                  aria-hidden="true"
                  className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-accent"
                />
                {tip}
              </li>
            ))}
          </ul>
        </Panel>
      </div>

      {/* Limitations — stated plainly, not buried */}
      <Panel title="What this cannot tell you" className="ring-warn-line/60">
        <ul className="grid gap-2.5 sm:grid-cols-2">
          {LIMITATIONS.map((item) => (
            <li key={item} className="flex gap-2.5 text-xs leading-relaxed text-muted">
              <span aria-hidden="true" className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-warn" />
              {item}
            </li>
          ))}
        </ul>
      </Panel>
    </div>
  );
}
