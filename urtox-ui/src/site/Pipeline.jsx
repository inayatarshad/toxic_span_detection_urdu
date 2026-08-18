import React, { useState } from "react";
import {
  Download,
  Filter,
  PenLine,
  ShieldCheck,
  Boxes,
  Cpu,
  BarChart4,
  AudioLines,
} from "lucide-react";
import { Section, Source, useInView, Nastaliq } from "./ui";

const STEPS = [
  {
    icon: Download,
    title: "Data collection",
    summary: "Three domains, deliberately mixed",
    detail:
      "Urdu text gathered from social media (X, Instagram, Reddit), Urdu newspapers (Daily Jang, UrduPoint, BOL News Urdu, Independent Urdu) and YouTube comments, captions and descriptions.",
    figures: ["5,254 social media", "4,300 newspapers", "4,788 YouTube"],
    source: "MUTEX §3.2",
  },
  {
    icon: Filter,
    title: "Preprocessing",
    summary: "Normalise before anything else touches the text",
    detail:
      "Unicode NFC normalisation, diacritic handling, rule-based Roman-to-Nastaliq transliteration, URL and punctuation removal, whitespace normalisation and word segmentation, then spaCy tokenisation customised for Urdu.",
    figures: ["Removing all steps costs 6.2 F1"],
    source: "MUTEX §3.5, Table 11",
  },
  {
    icon: PenLine,
    title: "Annotation",
    summary: "Manual, token-level, BIO",
    detail:
      "Each token receives B-TOXIC, I-TOXIC or O. Spans are maximal contiguous runs of B/I tokens. Each record additionally carries a binary label and a toxicity category.",
    figures: ["9,885 spans", "1.77 tokens mean span"],
    source: "MUTEX §3.3; recomputed from the released CSV",
  },
  {
    icon: ShieldCheck,
    title: "Quality assessment",
    summary: "Agreement measured, disagreement adjudicated",
    detail:
      "Inter-annotator reliability assessed with Cohen's κ and Krippendorff's α, followed by an adjudication phase resolving cases where annotators disagreed.",
    figures: ["κ = 0.82", "α = 0.81", "≈15% adjudicated"],
    source: "MUTEX §3.3",
  },
  {
    icon: Boxes,
    title: "Dataset construction",
    summary: "Deduplicated, stratified, split",
    detail:
      "Fuzzy deduplication with a Levenshtein threshold below 0.8 and stratified sampling per domain, then stratified train/validation/test splits preserving the toxic ratio.",
    figures: ["14,337 released records", "54% / 46% balance"],
    source: "MUTEX §3.2, §3.6; released CSV",
  },
  {
    icon: Cpu,
    title: "Model development",
    summary: "Sequence labelling with structured decoding",
    detail:
      "XLM-RoBERTa fine-tuned for token classification with a CRF layer over the BIO label set, trained across all domains with inverse-frequency class weighting.",
    figures: ["60.0% F1 (MUTEX)", "67.0% F1 (MUTEX-M)"],
    source: "MUTEX §3.5; MUTEX-M §4.2",
  },
  {
    icon: BarChart4,
    title: "Evaluation",
    summary: "Token-level F1, ablations, error analysis",
    detail:
      "Token-level F1 as the primary metric, with cross-domain transfer experiments, component ablations, statistical significance testing over five seeds, and manual error categorisation.",
    figures: ["5 seeds", "5-fold CV on ablations"],
    source: "MUTEX §4; MUTEX-M §5",
  },
  {
    icon: AudioLines,
    title: "Multimodal extension",
    summary: "Synthesise, classify, fuse, then test on real speech",
    detail:
      "Speech synthesised from the annotations, an MMS-300M utterance classifier trained on the cached embeddings, weighted late fusion of the two probabilities, and evaluation on a held-out real-speech set.",
    figures: ["83.2% fused (TTS)", "77.1% fused (real speech)"],
    source: "MUTEX-M §3.2, §4.4, §5.7",
  },
];

export default function Pipeline() {
  const [open, setOpen] = useState(2);
  const [spineRef, spineIn] = useInView({ threshold: 0.1 });

  return (
    <Section
      id="pipeline"
      index="09"
      tone="cream"
      eyebrow="Research pipeline"
      title="From collection to multimodal evaluation"
      lead="The sequence of steps the two papers actually followed, with the figure attached to each stage."
    >
            {/* طریقہ, "method" */}
      <Nastaliq className={"-right-6 top-12 text-forest/[0.055] sm:right-0"} size="clamp(7rem, 15vw, 15rem)">
        طریقہ
      </Nastaliq>

      <ol ref={spineRef} className="relative space-y-2">
        {/* spine, drawing itself downward as the section is reached */}
        <div
          aria-hidden="true"
          className="absolute left-[19px] top-4 bottom-4 w-px bg-sand-deep"
        />
        <div
          aria-hidden="true"
          className="absolute left-[19px] top-4 w-px origin-top bg-merlot-mid transition-transform duration-[1400ms] ease-out"
          style={{ bottom: "1rem", transform: `scaleY(${spineIn ? 1 : 0})` }}
        />

        {STEPS.map((s, i) => {
          const Icon = s.icon;
          const isOpen = open === i;
          return (
            <li
              key={s.title}
              className="relative transition-all duration-700 ease-[cubic-bezier(0.16,1,0.3,1)]"
              style={{
                opacity: spineIn ? 1 : 0,
                transform: spineIn ? "translateX(0)" : "translateX(-10px)",
                transitionDelay: `${i * 110}ms`,
              }}
            >
              <button
                type="button"
                onClick={() => setOpen(isOpen ? -1 : i)}
                aria-expanded={isOpen}
                className="flex w-full items-start gap-4 rounded-lg px-2 py-3 text-left transition-colors hover:bg-white/60"
              >
                <span
                  className={`relative z-10 flex h-10 w-10 shrink-0 items-center justify-center rounded-full border transition-colors ${
                    isOpen
                      ? "border-merlot-bright/40 bg-merlot text-cream"
                      : "border-sand-deep bg-ivory text-forest-mid"
                  }`}
                >
                  <Icon size={17} aria-hidden="true" />
                </span>

                <div className="min-w-0 flex-1 pt-1">
                  <div className="flex flex-wrap items-baseline gap-x-3">
                    <h3 className="text-base">{s.title}</h3>
                    <span className="text-xs text-forest-soft">{s.summary}</span>
                  </div>

                  <div
                    className={`grid transition-all duration-300 ${
                      isOpen ? "mt-3 grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"
                    }`}
                  >
                    <div className="overflow-hidden">
                      <p className="max-w-prose text-sm leading-relaxed text-forest-mid">
                        {s.detail}
                      </p>
                      <ul className="mt-3 flex flex-wrap gap-2">
                        {s.figures.map((f) => (
                          <li
                            key={f}
                            className="rounded border border-sand-deep/60 bg-white px-2.5 py-1 font-mono text-[0.68rem] text-forest-deep"
                          >
                            {f}
                          </li>
                        ))}
                      </ul>
                      <p className="mt-2.5 font-sans text-xs text-forest-soft">
                        <span className="font-medium">Source:</span> {s.source}
                      </p>
                    </div>
                  </div>
                </div>

                <span
                  aria-hidden="true"
                  className={`mt-2 shrink-0 text-sand-deep transition-transform ${
                    isOpen ? "rotate-180" : ""
                  }`}
                >
                  ⌄
                </span>
              </button>
            </li>
          );
        })}
      </ol>

      <Source className="mt-8">
        Stage descriptions follow the methodology sections of both papers; dataset counts marked as
        recomputed come from URTOX_v2.csv.
      </Source>
    </Section>
  );
}
