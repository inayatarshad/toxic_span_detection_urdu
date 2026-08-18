import React from "react";
import { Section, Reveal, Pill, Source } from "./ui";

export default function FutureWork({ items }) {
  return (
    <Section
      id="future"
      index="14"
      eyebrow="Future research"
      title="Where this goes next"
      lead="The directions the authors set out, in the priority order they gave them. Each is tied to the specific result that motivates it. Nothing here is speculation added after the fact."
    >
      <ol className="space-y-4">
        {items.map((f, i) => (
          <Reveal key={f.title} delay={i * 70}>
            <li className="grid gap-5 rounded-lg border border-sand-deep/70 bg-white/70 p-6 sm:grid-cols-[auto_1fr] sm:p-7">
              <div className="sm:w-28">
                <Pill tone={f.priority === "Deferred" ? "outline" : "merlot"}>{f.priority}</Pill>
              </div>
              <div>
                <h3 className="text-lg leading-snug">{f.title}</h3>
                <p className="mt-2.5 max-w-prose text-sm leading-relaxed text-forest-mid">
                  {f.body}
                </p>
                <div className="mt-4 rounded-md border-l-2 border-sand-deep bg-sand/25 px-4 py-3">
                  <p className="text-[0.68rem] font-semibold uppercase tracking-wider text-forest-soft">
                    What motivates it
                  </p>
                  <p className="mt-1 text-sm leading-relaxed text-forest-mid">{f.evidence}</p>
                </div>
                <p className="mt-2.5 font-sans text-xs text-forest-soft">
                  <span className="font-medium">Source:</span> {f.source}
                </p>
              </div>
            </li>
          </Reveal>
        ))}
      </ol>

      <Source className="mt-8">
        Priorities 1–4 and the deferred items are stated as such in MUTEX-M §7. The earlier paper's
        longer-horizon directions, namely larger Urdu-specific pretrained models, few-shot adaptation,
        active learning and continual learning, are described there as possibilities rather than
        planned work, and are not restated here as commitments.
      </Source>
    </Section>
  );
}
