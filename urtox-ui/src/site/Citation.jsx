import React, { useState } from "react";
import { Copy, Check, Info } from "lucide-react";
import { Section, Callout, Source, Reveal } from "./ui";
import { LINKS } from "../data/research";

const BIBTEX = {
  dataset: `@misc{arshad2026urtox,
  title        = {URTOX: A Manually Annotated Urdu Toxic Span Dataset},
  author       = {Arshad, Inayat and Saleem, Fajar and Hussain, Ijaz},
  year         = {2026},
  howpublished = {Hugging Face Datasets},
  url          = {${LINKS.dataset}},
  note         = {MIT License}
}`,
  mutex: `@misc{arshad2026mutex,
  title         = {MUTEX: Leveraging Multilingual Transformers and Conditional
                   Random Fields for Enhanced Urdu Toxic Span Detection},
  author        = {Arshad, Inayat and Saleem, Fajar and Hussain, Ijaz},
  year          = {2026},
  eprint        = {2603.05057},
  archivePrefix = {arXiv},
  url           = {${LINKS.arxiv}}
}`,
  mutexm: `@unpublished{saleem2026advancing,
  title  = {Advancing Urdu Toxicity Detection: Improved Span Models and a
            Multimodal Audio Extension},
  author = {Saleem, Fajar and Arshad, Inayat and Hussain, Ijaz},
  year   = {2026},
  note   = {Manuscript under review}
}`,
};

const PLAIN = {
  dataset:
    "Arshad, I., Saleem, F., & Hussain, I. (2026). URTOX: A manually annotated Urdu toxic span dataset [Data set]. Hugging Face. " +
    LINKS.dataset,
  mutex:
    "Arshad, I., Saleem, F., & Hussain, I. (2026). MUTEX: Leveraging multilingual transformers and conditional random fields for enhanced Urdu toxic span detection. arXiv:2603.05057.",
  mutexm:
    "Saleem, F., Arshad, I., & Hussain, I. (2026). Advancing Urdu toxicity detection: Improved span models and a multimodal audio extension. Manuscript under review.",
};

const TABS = [
  { key: "dataset", label: "Dataset" },
  { key: "mutex", label: "MUTEX paper" },
  { key: "mutexm", label: "MUTEX-M paper" },
];

function CopyBlock({ text, label }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      // Clipboard access can be denied; the text stays selectable either way.
      setCopied(false);
    }
  };

  return (
    <div className="relative">
      <button
        type="button"
        onClick={copy}
        className="absolute right-3 top-3 inline-flex items-center gap-1.5 rounded-md border border-sand/25 bg-white/10 px-2.5 py-1.5 text-xs font-medium text-sand transition-colors hover:bg-white/20"
        aria-label={`Copy ${label}`}
      >
        {copied ? <Check size={13} aria-hidden="true" /> : <Copy size={13} aria-hidden="true" />}
        {copied ? "Copied" : "Copy"}
      </button>
      <pre className="overflow-x-auto rounded-lg bg-forest-deep p-5 pr-24 font-mono text-[0.78rem] leading-relaxed text-sand">
        {text}
      </pre>
    </div>
  );
}

export default function Citation() {
  const [tab, setTab] = useState("dataset");

  return (
    <Section
      id="citation"
      index="12"
      eyebrow="Citation"
      title="How to cite this work"
      lead="If you use URTOX, please cite the dataset and the paper that introduced it."
    >
      <div
        role="tablist"
        aria-label="Citation targets"
        className="flex flex-wrap gap-1 border-b border-sand-deep/60 pb-3"
      >
        {TABS.map((t) => (
          <button
            key={t.key}
            role="tab"
            aria-selected={tab === t.key}
            onClick={() => setTab(t.key)}
            className={`rounded-md px-3.5 py-2 text-sm font-medium transition-colors ${
              tab === t.key ? "bg-forest text-cream" : "text-forest-mid hover:bg-sand/40"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="mt-8 grid gap-8 lg:grid-cols-[1.15fr_0.85fr]">
        <Reveal from="left">
          <p className="eyebrow">BibTeX</p>
          <div className="mt-3">
            <CopyBlock text={BIBTEX[tab]} label="BibTeX entry" />
          </div>
        </Reveal>

        <Reveal from="right" delay={80}>
          <p className="eyebrow">Plain citation</p>
          <div className="mt-3 rounded-lg border border-sand-deep/70 bg-white/70 p-5">
            <p className="text-sm leading-relaxed text-forest-mid">{PLAIN[tab]}</p>
            <button
              type="button"
              onClick={() => navigator.clipboard?.writeText(PLAIN[tab])}
              className="mt-4 inline-flex items-center gap-1.5 rounded-md border border-sand-deep bg-white px-3 py-1.5 text-xs font-medium text-forest hover:border-forest-soft"
            >
              <Copy size={12} aria-hidden="true" />
              Copy plain citation
            </button>
          </div>

          <div className="mt-5 space-y-4">
            <Callout icon={Info} title="No DOI exists yet">
              Neither the dataset nor either manuscript has been assigned a DOI. The entries above
              therefore carry none, because a placeholder would be worse than its absence.
            </Callout>
            <Callout icon={Info} title="The MUTEX-M entry is provisional">
              The second paper is under review with no venue assigned, so it is given as an{" "}
              <code className="font-mono text-xs">@unpublished</code> entry. Replace it with the
              published record once one exists.
            </Callout>
          </div>
        </Reveal>
      </div>

      <Source className="mt-8">
        Author names, ordering and titles are taken verbatim from the manuscripts. The arXiv
        identifier is the one listed for the MUTEX preprint.
      </Source>
    </Section>
  );
}
