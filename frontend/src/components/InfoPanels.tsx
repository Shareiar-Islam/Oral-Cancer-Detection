/**
 * Supporting content below the tool.
 *
 * Deliberately about the user's mouth, not the model. Someone arriving here is
 * worried about something they can see; what helps them is knowing which signs
 * warrant a clinician, what raises risk, and how to check. Architecture and
 * validation figures belong in the README and at GET /api/model-info.
 *
 * Static content — the page still issues no network request until the user
 * asks for a prediction.
 */

interface Sign {
  readonly title: string;
  readonly body: string;
}

/* The two-week rule is the single most useful thing a layperson can carry away:
   ordinary mouth injuries heal well within it. */
const WARNING_SIGNS: readonly Sign[] = [
  {
    title: 'A sore that will not heal',
    body: 'An ulcer or sore still present after two weeks. Ordinary mouth injuries heal well inside that window.',
  },
  {
    title: 'White or red patches',
    body: 'Persistent white (leukoplakia) or red (erythroplakia) patches on the tongue, gums, or lining of the cheek.',
  },
  {
    title: 'A lump or thickening',
    body: 'Any new lump, rough patch, or thickened area in the mouth, on the lip, or in the neck.',
  },
  {
    title: 'Pain, numbness, or bleeding',
    body: 'Persistent pain, numbness or tingling, or unexplained bleeding anywhere in the mouth.',
  },
  {
    title: 'Difficulty swallowing or speaking',
    body: 'Trouble chewing, swallowing, or moving the tongue or jaw — or a hoarse voice that lingers.',
  },
  {
    title: 'A change in how teeth fit',
    body: 'Loose teeth with no clear cause, or dentures that stop fitting the way they used to.',
  },
];

/* Betel quid / areca nut is listed first deliberately: it is a leading driver
   of oral cancer across South and Southeast Asia and is often not recognised
   as a risk at all. */
const RISK_FACTORS: readonly string[] = [
  'Betel quid, areca nut, paan, gutka or zarda — carcinogenic on their own, even without tobacco',
  'Smokeless tobacco: chewing tobacco, snuff, or khaini held against the gum',
  'Smoking cigarettes, bidis, cigars or pipes',
  'Heavy alcohol use, which multiplies risk sharply when combined with tobacco',
  'HPV infection, associated with cancers at the back of the mouth and throat',
  'Prolonged sun exposure to the lips without protection',
  'Long-term poor oral hygiene, or chronic irritation from a broken tooth or ill-fitting denture',
];

const SELF_CHECK: readonly string[] = [
  'In good light with a mirror, look at your lips and the front of your gums, then pull each cheek out to see the lining and rear gums.',
  'Tilt your head back and look at the roof of your mouth; pull your tongue forward and inspect the top, underside, and both edges.',
  'Feel along both sides of your neck and under your jaw for lumps or tender spots.',
  'Do this monthly. Anything unusual that is still there after two weeks warrants an appointment.',
];

const REDUCE_RISK: readonly string[] = [
  'Stop using betel quid, areca nut, and all forms of tobacco — the single largest change you can make.',
  'Limit alcohol.',
  'Use lip balm with SPF when outdoors for long periods.',
  'Eat plenty of fruit and vegetables.',
  'See a dentist regularly; ask for an oral cancer screening as part of the visit.',
];

function Panel({
  title,
  children,
  tone = 'neutral',
}: {
  title: string;
  children: React.ReactNode;
  tone?: 'neutral' | 'warn';
}): JSX.Element {
  return (
    <section
      className={`rounded-2xl bg-surface p-5 ring-1 ${
        tone === 'warn' ? 'ring-warn-line' : 'ring-line'
      }`}
    >
      <h2
        className={`font-display mb-4 text-sm font-bold tracking-tight ${
          tone === 'warn' ? 'text-warn' : 'text-ink'
        }`}
      >
        {title}
      </h2>
      {children}
    </section>
  );
}

function Bullets({
  items,
  dot = 'bg-accent',
}: {
  items: readonly string[];
  dot?: string;
}): JSX.Element {
  return (
    <ul className="space-y-2.5">
      {items.map((item) => (
        <li key={item} className="flex gap-2.5 text-xs leading-relaxed text-muted">
          <span aria-hidden="true" className={`mt-1.5 h-1 w-1 shrink-0 rounded-full ${dot}`} />
          {item}
        </li>
      ))}
    </ul>
  );
}

export function InfoPanels(): JSX.Element {
  return (
    <div className="mt-10 space-y-5">
      <Panel title="Signs worth getting checked" tone="warn">
        <ul className="grid gap-x-6 gap-y-4 sm:grid-cols-2 lg:grid-cols-3">
          {WARNING_SIGNS.map((sign) => (
            <li key={sign.title}>
              <h3 className="mb-1 text-xs font-semibold text-ink">{sign.title}</h3>
              <p className="text-xs leading-relaxed text-muted">{sign.body}</p>
            </li>
          ))}
        </ul>
        <p className="mt-4 border-t border-line-soft pt-3 text-xs leading-relaxed text-warn">
          Most mouth sores are harmless and settle within two weeks. Anything that
          persists beyond that should be looked at by a dentist or doctor — early
          detection is what changes outcomes.
        </p>
      </Panel>

      <div className="grid gap-5 lg:grid-cols-2">
        <Panel title="What raises risk">
          <Bullets items={RISK_FACTORS} dot="bg-warn" />
        </Panel>

        <Panel title="Checking your own mouth">
          <ol className="space-y-2.5">
            {SELF_CHECK.map((step, index) => (
              <li key={step} className="flex gap-3 text-xs leading-relaxed text-muted">
                <span
                  aria-hidden="true"
                  className="font-display shrink-0 text-xs font-bold text-accent"
                >
                  {String(index + 1).padStart(2, '0')}
                </span>
                {step}
              </li>
            ))}
          </ol>
        </Panel>
      </div>

      <Panel title="Lowering your risk">
        <div className="sm:columns-2 sm:gap-6">
          <Bullets items={REDUCE_RISK} />
        </div>
      </Panel>

      <p className="px-1 text-xs leading-relaxed text-faint">
        General health information, not medical advice, and not specific to you or
        to any image you upload. If something in your mouth is worrying you, see a
        dentist or doctor — do not wait on a screening tool.
      </p>
    </div>
  );
}
