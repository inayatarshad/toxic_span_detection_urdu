import React, { useState } from "react";
import {
  Languages,
  Binary,
  GitBranch,
  Shuffle,
  Scissors,
  Database,
  MessageCircleQuestion,
  Layers,
} from "lucide-react";
import { Section, Reveal, Source } from "./ui";
import { URDU_CHALLENGES } from "../data/research";

const ICONS = {
  Languages,
  Binary,
  GitBranch,
  Shuffle,
  Scissors,
  Database,
  MessageCircleQuestion,
  Layers,
};

export default function Challenges() {
  const [open, setOpen] = useState("script");

  return (
    <Section
      id="challenges"
      index="05"
      tone="forest"
      eyebrow="Urdu NLP challenges"
      title="Why this is not a solved problem in translation"
      lead="These are the specific obstacles the research documents. Each one is paired with the measurement from the papers that quantifies it, rather than left as a general claim about difficulty."
    >
      <div className="grid gap-3 sm:grid-cols-2">
        {URDU_CHALLENGES.map((c, i) => {
          const Icon = ICONS[c.icon];
          const isOpen = open === c.key;
          return (
            <Reveal key={c.key} delay={i * 45}>
              <button
                type="button"
                onClick={() => setOpen(isOpen ? null : c.key)}
                aria-expanded={isOpen}
                className={`h-full w-full rounded-lg border p-5 text-left transition-colors ${
                  isOpen
                    ? "border-sand/50 bg-white/10"
                    : "border-sand/20 bg-white/[0.04] hover:border-sand/40 hover:bg-white/[0.07]"
                }`}
              >
                <div className="flex items-start gap-3">
                  <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-sand/15">
                    <Icon size={16} className="text-sand" aria-hidden="true" />
                  </span>
                  <div className="min-w-0 flex-1">
                    <h3 className="!text-cream text-base font-medium">{c.title}</h3>
                    <p className="mt-2 text-sm leading-relaxed text-sand/75">{c.body}</p>

                    <div
                      className={`grid transition-all duration-300 ${
                        isOpen ? "mt-4 grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"
                      }`}
                    >
                      <div className="overflow-hidden">
                        <div className="rounded-md border-l-2 border-merlot-bright/70 bg-black/15 px-4 py-3">
                          <p className="text-[0.7rem] font-semibold uppercase tracking-wider text-sand-deep">
                            Measured effect
                          </p>
                          <p className="mt-1.5 text-sm leading-relaxed text-sand/90">
                            {c.evidence}
                          </p>
                          <p className="mt-2 font-mono text-[0.65rem] text-sand/45">{c.source}</p>
                        </div>
                      </div>
                    </div>

                    {!isOpen && (
                      <span className="mt-3 inline-block text-xs text-sand/45">
                        Show measured effect →
                      </span>
                    )}
                  </div>
                </div>
              </button>
            </Reveal>
          );
        })}
      </div>

      <div className="mt-10 max-w-3xl rounded-lg border border-sand/20 bg-white/[0.04] p-6">
        <p className="text-sm leading-relaxed text-sand/80">
          Taken together, these are the reasons the authors give for building a dedicated Urdu
          resource rather than transferring an English one. The ablation results support that
          reasoning directly: removing the Urdu-specific preprocessing pipeline entirely costs 6.2
          token-level F1 points, more than the architectural contribution of the CRF layer several
          times over.
        </p>
        <Source className="!text-sand/45">MUTEX Table 11.</Source>
      </div>
    </Section>
  );
}
