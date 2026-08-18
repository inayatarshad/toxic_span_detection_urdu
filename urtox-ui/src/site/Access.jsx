import React from "react";
import { Database, FileText, AudioLines, Scale, ExternalLink, Info, Mail } from "lucide-react";
import { Section, Reveal, Callout, Pill, GithubMark, n } from "./ui";
import { LINKS, HUMAN_AUDIO, URTOX_MM } from "../data/research";
import stats from "../data/stats.json";

const RESOURCES = [
  {
    icon: Database,
    name: "URTOX",
    kind: "Text dataset",
    availability: "Public",
    tone: "merlot",
    detail: `${n(stats.records)} records, ${stats.columns.length} columns, single CSV. Distributed as one train split.`,
    licence: "MIT",
    href: LINKS.dataset,
    cta: "Open on Hugging Face",
  },
  {
    icon: GithubMark,
    name: "Code repository",
    kind: "Notebooks, API and this site",
    availability: "Public",
    tone: "sand",
    detail:
      "Training and fusion notebooks, the FastAPI inference service, and the source for this research hub.",
    licence: "See repository",
    href: LINKS.github,
    cta: "Open on GitHub",
  },
  {
    icon: FileText,
    name: "MUTEX preprint",
    kind: "Paper",
    availability: "Public",
    tone: "sand",
    detail: "The full methodology, annotation protocol, benchmark and ablation studies.",
    licence: "See arXiv listing",
    href: LINKS.arxiv,
    cta: "Read on arXiv",
  },
  {
    icon: AudioLines,
    name: "URTOX-MM",
    kind: "Synthesised speech dataset",
    availability: "Public",
    tone: "sand",
    detail: `${n(URTOX_MM.clips)} TTS-synthesised clips paired to the URTOX annotations. ${URTOX_MM.duration}.`,
    licence: "See dataset page",
    href: LINKS.audioDataset,
    cta: "Open on Hugging Face",
  },
  {
    icon: AudioLines,
    name: "URTOX-HumanAudio",
    kind: "Real-speech evaluation set",
    availability: "On request",
    tone: "outline",
    detail: `${n(HUMAN_AUDIO.clips)} real conversational clips across four regional accents. ${HUMAN_AUDIO.availability}.`,
    licence: "See request",
    href: `mailto:${LINKS.contact}?subject=${encodeURIComponent(
      "URTOX-HumanAudio access request"
    )}`,
    cta: "Request access by email",
    internal: true,
  },
];

export default function Access() {
  return (
    <Section
      id="access"
      index="11"
      tone="cream"
      eyebrow="Dataset access"
      title="Getting the data"
      lead="Every resource associated with this work, with its actual availability. Where licensing is not stated by the authors, that is said plainly rather than assumed."
    >
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {RESOURCES.map((r, i) => {
          const Icon = r.icon;
          return (
            <Reveal key={r.name} delay={i * 60}>
              <div className="flex h-full flex-col rounded-lg border border-sand-deep/70 bg-white/70 p-6">
                <div className="flex items-start justify-between gap-3">
                  <span className="flex h-9 w-9 items-center justify-center rounded-md bg-sand/50 text-forest-mid">
                    <Icon size={17} aria-hidden="true" />
                  </span>
                  <Pill tone={r.tone}>{r.availability}</Pill>
                </div>
                <h3 className="mt-4 text-base">{r.name}</h3>
                <p className="text-xs text-forest-soft">{r.kind}</p>
                <p className="mt-3 flex-1 text-sm leading-relaxed text-forest-mid">{r.detail}</p>
                <p className="mt-4 flex items-center gap-1.5 border-t border-sand-deep/50 pt-3 text-xs text-forest-soft">
                  <Scale size={12} aria-hidden="true" />
                  Licence: <span className="text-forest">{r.licence}</span>
                </p>
                {r.href && (
                  <a
                    href={r.href}
                    {...(r.internal ? {} : { target: "_blank", rel: "noreferrer" })}
                    className="mt-3 inline-flex items-center gap-1.5 text-sm font-medium text-merlot-mid hover:underline"
                  >
                    {r.cta}
                    {r.internal ? (
                      <Mail size={12} aria-hidden="true" />
                    ) : (
                      <ExternalLink size={12} aria-hidden="true" />
                    )}
                  </a>
                )}
              </div>
            </Reveal>
          );
        })}
      </div>

      <Reveal className="mt-10 grid gap-8 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="rounded-lg border border-sand-deep/70 bg-forest p-6 text-sand sm:p-8">
          <h3 className="!text-cream text-lg">Load it in three lines</h3>
          <p className="mt-2 text-sm text-sand/75">
            The dataset is a single CSV with one train split.
          </p>
          <pre className="mt-5 overflow-x-auto rounded-md bg-black/25 p-4 font-mono text-[0.8rem] leading-relaxed text-sand">
{`from datasets import load_dataset

ds = load_dataset("inayatarshad/URTOX", split="train")
print(ds)          # ${n(stats.records)} rows, ${stats.columns.length} columns`}
          </pre>
          <p className="mt-4 text-xs leading-relaxed text-sand/60">
            The <code className="font-mono">tokens</code>,{" "}
            <code className="font-mono">BIO_tags</code> and{" "}
            <code className="font-mono">toxic_list</code> columns are stored as stringified Python
            lists, so parse them with <code className="font-mono">ast.literal_eval</code> before use.
            A small number of rows have malformed list strings. See the data quality notes above.
          </p>
        </div>

        <div className="space-y-4">
          <Callout icon={Scale} title="Licence">
            The Hugging Face dataset page lists URTOX under the MIT License, which permits reuse
            and redistribution with attribution.
          </Callout>
          <Callout icon={Info} title="Audio files are hosted separately">
            The audio dataset CSV in the repository references paths under a Google Drive mount used
            during training. The clips themselves live in separate Hugging Face dataset
            repositories, linked above.
          </Callout>
          <Callout icon={Mail} title="Requesting the real-speech set">
            URTOX-HumanAudio is shared on request rather than published, because it contains real
            recorded voices. Email{" "}
            <a href={`mailto:${LINKS.contact}`} className="link-underline text-merlot-mid">
              {LINKS.contact}
            </a>{" "}
            with a short note on your intended research use.
          </Callout>
        </div>
      </Reveal>
    </Section>
  );
}
