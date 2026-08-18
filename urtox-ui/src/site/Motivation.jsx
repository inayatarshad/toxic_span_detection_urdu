import React, { useEffect, useState } from "react";
import { ArrowRight } from "lucide-react";
import { Section, Prose, Source, Reveal, useInView } from "./ui";

/**
 * Record 175 from URTOX_v2.csv, reproduced exactly as annotated. It carries a
 * single-token span with an intensifier sitting just outside it, which is the
 * boundary pattern the error analysis in both papers keeps returning to.
 */
const EXAMPLE = {
  id: 175,
  tokens: ["تمہاری", "بات", "بالکل", "فالتو", "ہے"],
  toxic: [3],
  label: "toxic",
  sub: "offensive",
  gloss: "What you are saying is completely worthless.",
};

function SpanDemo() {
  const [ref, inView] = useInView({ threshold: 0.4 });
  const [revealed, setRevealed] = useState(false);

  useEffect(() => {
    if (!inView) return undefined;
    const timer = setTimeout(() => setRevealed(true), 700);
    return () => clearTimeout(timer);
  }, [inView]);

  return (
    <div ref={ref} className="rounded-lg border border-sand-deep/70 bg-white/75 p-6">
      <p className="eyebrow">The same record, two task formulations</p>

      <div className="mt-5 rounded-md border border-sand-deep/50 bg-ivory p-4">
        <p className="text-xs font-medium text-forest-soft">Document-level classification returns</p>
        <p className="mt-2 font-mono text-sm text-forest-deep">
          label = <span className="text-merlot-mid">"toxic"</span>
        </p>
        <p className="mt-2 text-xs leading-relaxed text-forest-soft">
          Enough to filter on. Silent about cause.
        </p>
      </div>

      <div className="my-3 flex justify-center">
        <ArrowRight
          size={18}
          aria-hidden="true"
          className={`rotate-90 text-sand-deep transition-all duration-700 ${
            revealed ? "translate-y-0 opacity-100" : "-translate-y-1 opacity-0"
          }`}
        />
      </div>

      <div
        className={`rounded-md border border-merlot-bright/30 bg-merlot-wash/50 p-4 transition-all duration-700 ease-[cubic-bezier(0.16,1,0.3,1)] ${
          revealed ? "translate-y-0 opacity-100" : "translate-y-3 opacity-0"
        }`}
      >
        <p className="text-xs font-medium text-merlot-mid">Span-level detection returns</p>

        <p className="urdu mt-3 text-xl text-forest-deep" lang="ur" dir="rtl">
          {EXAMPLE.tokens.map((tok, i) => {
            const isToxic = EXAMPLE.toxic.includes(i);
            return (
              <span key={i}>
                <span
                  className={`transition-all duration-500 ${
                    isToxic && revealed
                      ? "rounded bg-merlot-bright/25 px-1 pb-0.5 underline decoration-merlot-bright decoration-2 underline-offset-[7px]"
                      : ""
                  }`}
                  style={{ transitionDelay: isToxic ? "300ms" : "0ms" }}
                >
                  {tok}
                </span>{" "}
              </span>
            );
          })}
        </p>
        <p className="mt-2 text-xs italic text-forest-soft">Gloss: {EXAMPLE.gloss}</p>

        <div className="mt-4 space-y-1 border-t border-merlot-bright/20 pt-3 font-mono text-[0.7rem] leading-relaxed">
          {EXAMPLE.tokens.map((tok, i) => {
            const isToxic = EXAMPLE.toxic.includes(i);
            return (
              <div
                key={i}
                className="flex items-center gap-2 transition-all duration-500"
                style={{
                  opacity: revealed ? 1 : 0,
                  transform: revealed ? "translateX(0)" : "translateX(6px)",
                  transitionDelay: `${350 + i * 70}ms`,
                }}
              >
                <span className="w-5 text-right text-forest-soft">{i}</span>
                <span className="urdu-inline w-24 text-right text-sm" lang="ur">
                  {tok}
                </span>
                <span className={isToxic ? "font-semibold text-merlot-mid" : "text-forest-soft"}>
                  {isToxic ? "B-Toxic" : "O"}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <p className="mt-4 text-xs leading-relaxed text-forest-soft">
        Record <span className="font-mono">id = {EXAMPLE.id}</span>, sub_label{" "}
        <span className="font-mono">{EXAMPLE.sub}</span>, shown exactly as annotated. Note that the
        intensifier <span className="urdu-inline" lang="ur">بالکل</span> (completely) sits outside
        the span. That boundary judgement is the single largest source of model error in both
        papers.
      </p>
    </div>
  );
}

export default function Motivation() {
  return (
    <Section
      id="motivation"
      index="01"
      tone="cream"
      eyebrow="Why this dataset"
      title="Locating toxicity, not just detecting it"
      lead="Most toxicity systems assign one label to a whole post. That is enough to filter, but not enough to explain a decision, to mask selectively, or to warn a reader about a specific phrase."
    >
      <div className="grid gap-12 lg:grid-cols-[1.05fr_0.95fr] lg:gap-16">
        <Reveal from="left">
          <Prose>
            <p>
              Toxicity classification answers a binary question about an entire passage. Toxic-span
              detection reformulates the task as token-level sequence labelling: each token receives
              a tag, and maximal contiguous runs of toxic tags define the spans responsible for the
              toxicity. The distinction matters for the applications the research targets, namely
              selective masking, explainable moderation decisions, and accessibility tools that flag
              specific offensive terms rather than hiding whole messages.
            </p>
            <p>
              For English this task has an established benchmark in SemEval-2021 Task 5, where
              transformer systems reach character-level F1 in the 65 to 70% range. For Urdu, the
              authors report that no prior span-level work existed, and that earlier Urdu toxicity
              resources operate at the sentence level only, identifying that a passage is toxic
              without identifying which words make it so.
            </p>
            <p>
              Urdu is spoken by a very large population yet is thinly represented in the corpora
              multilingual models are pretrained on, at under 1% of XLM-RoBERTa's pretraining data.
              A model arriving at the task therefore has comparatively weak Urdu semantics before
              fine-tuning even begins, and there was no token-level Urdu resource to fine-tune it
              on. Building a dedicated annotated dataset was a precondition for the work rather than
              a by-product of it.
            </p>
          </Prose>
          <Source>
            MUTEX §1, §2 and §5.2; MUTEX-M §1 and §2.2. The pretraining-share and benchmark figures
            are as cited by the papers.
          </Source>
        </Reveal>

        <Reveal from="right" delay={80}>
          <SpanDemo />
        </Reveal>
      </div>

    </Section>
  );
}
