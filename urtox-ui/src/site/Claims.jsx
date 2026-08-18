import React from "react";
import { Sparkles, Languages, Radio } from "lucide-react";
import { Reveal, CountUp, useInView } from "./ui";
import stats from "../data/stats.json";

/**
 * The novelty claims, stated as the papers state them and attributed on the
 * card so a reader can check each one against a section rather than take it
 * on trust.
 */
const CLAIMS = [
  {
    icon: Languages,
    lead: "First",
    rest: "manually annotated token-level toxic-span dataset for Urdu",
    body: "Earlier Urdu toxicity resources label whole sentences. URTOX marks the specific tokens that carry the toxicity.",
    source: "MUTEX §1, contribution 1",
  },
  {
    icon: Sparkles,
    lead: "First",
    rest: "supervised baseline for Urdu toxic span detection",
    body: "MUTEX establishes the benchmark that later Urdu span work can be measured against, and MUTEX-M improves on it.",
    source: "MUTEX §1, contribution 3",
  },
  {
    icon: Radio,
    lead: "First",
    rest: "multimodal Urdu toxicity framework with span localisation",
    body: "Text and audio evidence combined, then validated on real conversational speech across four Pakistani regional accents.",
    source: "MUTEX-M §1, contribution 5",
  },
];

const FIGURES = [
  { value: stats.records, label: "annotated records", suffix: "" },
  { value: stats.totalSpans, label: "toxic spans marked", suffix: "" },
  { value: 67, label: "token-level F1, text", suffix: "%" },
  { value: 77.1, label: "weighted F1, real speech", suffix: "%", decimals: 1 },
];

export default function Claims() {
  const [ref] = useInView();

  return (
    <section
      ref={ref}
      aria-label="Research contributions"
      className="relative overflow-hidden border-y border-sand-deep/50 bg-forest"
    >
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -right-8 -top-16 select-none font-urdu text-[16rem] leading-none text-white/[0.025]"
        style={{ direction: "rtl" }}
      >
        نیا
      </div>

      <div className="relative mx-auto max-w-6xl px-5 py-16 sm:px-8 sm:py-20">
        <Reveal from="none">
          <p className="eyebrow text-sand-deep">What is new here</p>
        </Reveal>

        <div className="mt-8 grid gap-4 md:grid-cols-3">
          {CLAIMS.map((c, i) => {
            const Icon = c.icon;
            return (
              <Reveal key={c.rest} delay={i * 110}>
                <article className="group h-full rounded-lg border border-sand/20 bg-white/[0.045] p-6 transition-all duration-300 hover:-translate-y-1 hover:border-sand/45 hover:bg-white/[0.08]">
                  <span className="flex h-9 w-9 items-center justify-center rounded-md bg-merlot-bright/25 text-sand transition-transform duration-300 group-hover:scale-110">
                    <Icon size={17} aria-hidden="true" />
                  </span>
                  <h3 className="mt-4 text-lg leading-snug !text-cream">
                    <span className="font-serif text-2xl text-merlot-wash">{c.lead}</span>{" "}
                    {c.rest}
                  </h3>
                  <p className="mt-3 text-sm leading-relaxed text-sand/70">{c.body}</p>
                  <p className="mt-4 border-t border-sand/15 pt-3 font-mono text-[0.65rem] text-sand/40">
                    {c.source}
                  </p>
                </article>
              </Reveal>
            );
          })}
        </div>

        <Reveal delay={200}>
          <dl className="mt-10 grid grid-cols-2 gap-6 border-t border-sand/15 pt-8 sm:grid-cols-4">
            {FIGURES.map((f) => (
              <div key={f.label}>
                <dd className="font-serif text-3xl leading-none text-cream sm:text-4xl">
                  <CountUp value={f.value} decimals={f.decimals || 0} suffix={f.suffix} />
                </dd>
                <dt className="mt-2 text-xs leading-relaxed text-sand/60">{f.label}</dt>
              </div>
            ))}
          </dl>
        </Reveal>

      </div>
    </section>
  );
}
