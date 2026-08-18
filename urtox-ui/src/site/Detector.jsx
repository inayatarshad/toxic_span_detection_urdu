import React from "react";
import { Zap, ShieldAlert, Server, ArrowRight } from "lucide-react";
import MutexWithXAI from "../MutexWithXAI.jsx";
import { Reveal, Pill } from "./ui";
import { P2_TEXT_MODELS, P2_REAL_SPEECH, LINKS } from "../data/research";

const text = P2_TEXT_MODELS.find((m) => m.best);
const fused = P2_REAL_SPEECH.find((m) => m.best);

export default function Detector() {
  return (
    <>
      <section className="relative overflow-hidden border-b border-sand-deep/60 bg-cream">
        <div className="mx-auto max-w-6xl px-5 py-14 sm:px-8 sm:py-16">
          <Reveal from="none">
            <div className="flex flex-wrap items-center gap-2">
              <Pill tone="merlot">
                <Zap size={11} className="mr-1" aria-hidden="true" />
                Live model
              </Pill>
              <Pill tone="outline">MUTEX-M</Pill>
            </div>
          </Reveal>

          <Reveal delay={60}>
            <h1 className="mt-4 max-w-3xl text-[2.2rem] font-medium leading-[1.08] tracking-tight sm:text-5xl">
              Detect toxic spans
              <br />
              in Urdu text and speech
            </h1>
          </Reveal>

          <Reveal delay={120}>
            <p className="mt-5 max-w-2xl text-base leading-relaxed text-forest-mid sm:text-lg">
              Paste Urdu text or record a voice message. The model returns a verdict and highlights
              the specific tokens carrying the toxicity, with attribution showing why each one was
              flagged.
            </p>
          </Reveal>

          <Reveal delay={180}>
            <dl className="mt-8 flex flex-wrap gap-x-10 gap-y-4">
              {[
                ["Token-level F1, text", `${text.f1.toFixed(1)}%`],
                ["Weighted F1, real speech", `${(fused.f1 * 100).toFixed(1)}%`],
                ["Span labels", "B-Toxic · I-Toxic · O"],
              ].map(([k, v]) => (
                <div key={k}>
                  <dd className="font-serif text-2xl leading-none text-forest-deep">{v}</dd>
                  <dt className="mt-1.5 text-xs text-forest-soft">{k}</dt>
                </div>
              ))}
            </dl>
          </Reveal>

          <Reveal delay={240}>
            <div className="mt-8 grid gap-3 sm:grid-cols-2">
              <div className="flex items-start gap-3 rounded-lg border border-merlot-bright/30 bg-merlot-wash/50 px-4 py-3">
                <ShieldAlert size={16} className="mt-0.5 shrink-0 text-merlot-mid" aria-hidden="true" />
                <p className="text-sm leading-relaxed text-forest-mid">
                  Predictions are a research prototype, not a moderation decision. The authors
                  describe the system as a first baseline rather than production ready.
                </p>
              </div>
              <div className="flex items-start gap-3 rounded-lg border border-sand-deep bg-sand/35 px-4 py-3">
                <Server size={16} className="mt-0.5 shrink-0 text-forest-soft" aria-hidden="true" />
                <p className="text-sm leading-relaxed text-forest-mid">
                  Inference runs against the deployed model API. If it is asleep the first request
                  can take a moment, and the status banner below reports the connection.
                </p>
              </div>
            </div>
          </Reveal>
        </div>
      </section>

      <MutexWithXAI />

      <section className="border-t border-sand-deep/60 bg-forest">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-6 px-5 py-12 sm:px-8">
          <div>
            <h2 className="!text-cream text-xl">Want the data behind it?</h2>
            <p className="mt-2 max-w-xl text-sm leading-relaxed text-sand/75">
              The dataset this model was trained on is public, with the annotation methodology,
              statistics and full results documented on the research hub.
            </p>
          </div>
          <div className="flex flex-wrap gap-2.5">
            <a
              href="#top"
              className="inline-flex items-center gap-2 rounded-md bg-sand px-4 py-2.5 text-sm font-medium text-forest-deep transition-colors hover:bg-cream"
            >
              Research hub
              <ArrowRight size={14} aria-hidden="true" />
            </a>
            <a
              href={LINKS.dataset}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-md border border-sand/30 px-4 py-2.5 text-sm font-medium text-sand transition-colors hover:bg-white/10"
            >
              Get the dataset
            </a>
          </div>
        </div>
      </section>
    </>
  );
}
