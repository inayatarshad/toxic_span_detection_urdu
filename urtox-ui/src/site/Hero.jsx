import React from "react";
import { ArrowUpRight, Database, FileText, Quote, Compass, Zap } from "lucide-react";
import stats from "../data/stats.json";
import { LINKS, AGREEMENT, INSTITUTION } from "../data/research";
import { GithubMark, Kinetic, Nastaliq, n } from "./ui";

const toxic = stats.labels.find((l) => l.key === "toxic").count;

const FACTS = [
  { value: n(stats.records), label: "Annotated records", note: "Rows in the released CSV" },
  { value: "Urdu", label: "Language", note: "Nastaliq script, multi-domain" },
  { value: "Token-level BIO", label: "Annotation type", note: `${n(stats.totalSpans)} toxic spans` },
  { value: "Toxic span detection", label: "Research domain", note: "Low-resource NLP" },
  { value: "MIT", label: "Licence", note: "Public release on Hugging Face" },
];

const CTAS = [
  { href: "#/detector", label: "Try the live detector", icon: Zap, primary: true },
  { href: "#explore", label: "Explore dataset", icon: Compass },
  { href: LINKS.dataset, label: "Download dataset", icon: Database, external: true },
  { href: LINKS.arxiv, label: "Read the research", icon: FileText, external: true },
  { href: LINKS.github, label: "View on GitHub", icon: GithubMark, external: true },
  { href: "#citation", label: "Cite dataset", icon: Quote },
];

export default function Hero() {
  return (
    <div id="top" className="relative overflow-hidden bg-ivory">
      {/* Nastaliq sets the language of the work before a word of English is read */}
      <Nastaliq
        className="-right-6 top-8 text-sand/50 sm:-right-2 lg:right-4"
        size="clamp(9rem, 20vw, 20rem)"
      >
        اردو
      </Nastaliq>

      <div className="relative mx-auto max-w-6xl px-5 pb-20 pt-14 sm:px-8 sm:pb-28 sm:pt-20">
        <p className="eyebrow">Open research resource · PIEAS, Islamabad</p>

        <h1 className="mt-5 max-w-3xl text-[2.6rem] font-medium leading-[1.06] tracking-tight sm:text-6xl">
          <Kinetic as="span" className="block" stagger={70}>
            Urdu Toxic
          </Kinetic>
          <Kinetic as="span" className="block" delay={180} stagger={70}>
            Span Dataset
          </Kinetic>
        </h1>

        <p className="mt-7 max-w-2xl text-lg leading-relaxed text-forest-mid sm:text-xl">
          A manually annotated research resource for toxic-span detection in Urdu, identifying not
          only <em>whether</em> a passage is toxic, but <em>which tokens</em> carry the toxicity.
        </p>

        <p className="mt-4 max-w-2xl text-sm leading-relaxed text-forest-soft">
          Built to support fine-grained toxicity research in a comparatively under-resourced
          language. Every record is annotated at the token level using a BIO scheme, with
          inter-annotator agreement of κ&nbsp;=&nbsp;{AGREEMENT.kappa} and α&nbsp;=&nbsp;
          {AGREEMENT.alpha} reported by the authors.
        </p>

        <div className="mt-9 flex flex-wrap gap-2.5">
          {CTAS.map(({ href, label, icon: Icon, primary, external }) => (
            <a
              key={label}
              href={href}
              {...(external ? { target: "_blank", rel: "noreferrer" } : {})}
              className={`inline-flex items-center gap-2 rounded-md px-4 py-2.5 text-sm font-medium transition-colors ${
                primary
                  ? "bg-merlot text-cream hover:bg-merlot-mid"
                  : "border border-sand-deep bg-white/60 text-forest hover:border-forest-soft hover:bg-white"
              }`}
            >
              <Icon size={15} aria-hidden="true" />
              {label}
              {external && <ArrowUpRight size={13} aria-hidden="true" className="opacity-60" />}
            </a>
          ))}
        </div>

        <dl className="mt-14 grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-sand-deep/70 bg-sand-deep/50 sm:grid-cols-3 lg:grid-cols-5">
          {FACTS.map((f) => (
            <div key={f.label} className="bg-ivory px-4 py-5">
              <dd className="font-serif text-lg leading-tight text-forest-deep sm:text-[1.4rem]">
                {f.value}
              </dd>
              <dt className="mt-1.5 text-xs font-medium uppercase tracking-wide text-forest-soft">
                {f.label}
              </dt>
              <p className="mt-1 text-xs text-forest-soft/80">{f.note}</p>
            </div>
          ))}
        </dl>

        <p className="mt-6 max-w-3xl text-xs leading-relaxed text-forest-soft">
          {INSTITUTION}. Dataset statistics on this page are computed directly from the released{" "}
          <code className="font-mono text-[0.7rem] text-merlot-mid">URTOX_v2.csv</code>, where {n(toxic)}{" "}
          of {n(stats.records)} records are labelled toxic.
        </p>
      </div>
    </div>
  );
}
