import React from "react";
import { FileText, ExternalLink, Users, Clock, Info } from "lucide-react";
import { Section, Reveal, Pill, Callout } from "./ui";
import { PAPERS, INSTITUTION } from "../data/research";

export default function Publications() {
  return (
    <Section
      id="publications"
      index="10"
      eyebrow="Publications"
      title="The research behind the dataset"
      lead="Two manuscripts. Neither has been published in a peer-reviewed venue at the time of writing; both are listed with their actual status."
    >
      <div className="mb-10 flex flex-wrap items-center gap-3">
        <Pill tone="outline">Published · none yet</Pill>
        <Pill tone="merlot">Under review · 2</Pill>
      </div>

      <div className="space-y-6">
        {PAPERS.map((p, i) => (
          <Reveal key={p.id} delay={i * 80}>
            <article className="rounded-lg border border-sand-deep/70 bg-white/70 p-6 sm:p-8">
              <div className="flex flex-wrap items-center gap-2.5">
                <Pill tone="merlot">{p.status}</Pill>
                {p.arxivId && <Pill tone="sand">{p.arxivId}</Pill>}
                <Pill tone="outline">{p.year}</Pill>
              </div>

              <h3 className="mt-4 max-w-3xl text-xl leading-snug sm:text-2xl">{p.title}</h3>

              <div className="mt-4 flex flex-wrap items-start gap-x-8 gap-y-3 text-sm">
                <div className="flex items-start gap-2">
                  <Users size={15} className="mt-0.5 shrink-0 text-forest-soft" aria-hidden="true" />
                  <div>
                    <p className="text-forest">{p.authors.join(", ")}</p>
                    {p.authorNote && (
                      <p className="mt-0.5 text-xs text-forest-soft">{p.authorNote}</p>
                    )}
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <Clock size={15} className="mt-0.5 shrink-0 text-forest-soft" aria-hidden="true" />
                  <div>
                    <p className="text-forest">{p.venue}</p>
                    {p.posted && (
                      <p className="mt-0.5 text-xs text-forest-soft">Posted {p.posted}</p>
                    )}
                  </div>
                </div>
              </div>

              <p className="mt-5 border-l-2 border-merlot-bright/50 bg-merlot-wash/40 py-2 pl-4 text-sm font-medium text-forest-deep">
                {p.headline}
              </p>

              <div className="mt-5">
                <p className="eyebrow">Stated contributions</p>
                <ul className="mt-3 space-y-2">
                  {p.contributions.map((c) => (
                    <li key={c} className="flex gap-3 text-sm leading-relaxed text-forest-mid">
                      <span
                        aria-hidden="true"
                        className="mt-[0.55rem] h-1 w-1 shrink-0 rounded-full bg-merlot-bright"
                      />
                      {c}
                    </li>
                  ))}
                </ul>
              </div>

              <div className="mt-6 flex flex-wrap items-center gap-3 border-t border-sand-deep/50 pt-5">
                {p.url ? (
                  <a
                    href={p.url}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-2 rounded-md bg-forest px-3.5 py-2 text-sm font-medium text-cream transition-colors hover:bg-forest-deep"
                  >
                    <FileText size={14} aria-hidden="true" />
                    Read the preprint
                    <ExternalLink size={12} aria-hidden="true" className="opacity-70" />
                  </a>
                ) : (
                  <span className="rounded-md border border-dashed border-sand-deep px-3.5 py-2 text-sm text-forest-soft">
                    No public preprint yet
                  </span>
                )}
                <a
                  href="#citation"
                  className="inline-flex items-center gap-2 rounded-md border border-sand-deep bg-white px-3.5 py-2 text-sm text-forest hover:border-forest-soft"
                >
                  Citation
                </a>
                <span className="text-xs text-forest-soft">
                  Corresponding author: {p.corresponding}
                </span>
              </div>
            </article>
          </Reveal>
        ))}
      </div>

      <div className="mt-10">
        <Callout icon={Info} title="Affiliation">
          {INSTITUTION}.
        </Callout>
      </div>
    </Section>
  );
}
