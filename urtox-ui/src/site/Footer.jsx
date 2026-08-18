import React from "react";
import { Database, FileText, ArrowUp } from "lucide-react";
import { LINKS, INSTITUTION } from "../data/research";
import Logo from "./Logo";
import stats from "../data/stats.json";
import { GithubMark, n } from "./ui";

const COLUMNS = [
  {
    title: "Dataset",
    links: [
      { label: "Overview", href: "#dataset" },
      { label: "Annotation methodology", href: "#annotation" },
      { label: "Explore records", href: "#explore" },
      { label: "Access and licence", href: "#access" },
    ],
  },
  {
    title: "Research",
    links: [
      { label: "Motivation", href: "#motivation" },
      { label: "Urdu NLP challenges", href: "#challenges" },
      { label: "System architecture", href: "#method" },
      { label: "Results", href: "#results" },
      { label: "Multimodal extension", href: "#multimodal" },
    ],
  },
  {
    title: "Use it",
    links: [
      { label: "Research pipeline", href: "#pipeline" },
      { label: "Publications", href: "#publications" },
      { label: "Citation", href: "#citation" },
      { label: "Reproducibility", href: "#reproducibility" },
      { label: "Future work", href: "#future" },
    ],
  },
];

const EXTERNAL = [
  { label: "Hugging Face", href: LINKS.dataset, icon: Database },
  { label: "GitHub", href: LINKS.github, icon: GithubMark },
  { label: "arXiv preprint", href: LINKS.arxiv, icon: FileText },
];

export default function Footer() {
  return (
    <footer className="border-t border-sand-deep/60 bg-cream">
      <div className="mx-auto max-w-6xl px-5 py-16 sm:px-8">
        <div className="grid gap-10 lg:grid-cols-[1.3fr_2fr]">
          <div>
            <div className="flex items-center gap-3">
              <Logo size={36} />
              <p className="font-serif text-2xl leading-none text-forest-deep">URTOX</p>
            </div>
            <p className="mt-3 max-w-sm text-sm leading-relaxed text-forest-mid">
              A manually annotated Urdu toxic-span dataset developed to support research in
              low-resource NLP. {n(stats.records)} records, annotated at the token level.
            </p>
            <ul className="mt-6 flex flex-wrap gap-2">
              {EXTERNAL.map(({ label, href, icon: Icon }) => (
                <li key={label}>
                  <a
                    href={href}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1.5 rounded-md border border-sand-deep bg-white/70 px-3 py-1.5 text-xs font-medium text-forest transition-colors hover:border-forest-soft"
                  >
                    <Icon size={13} aria-hidden="true" />
                    {label}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          <nav aria-label="Footer" className="grid grid-cols-2 gap-8 sm:grid-cols-3">
            {COLUMNS.map((col) => (
              <div key={col.title}>
                <p className="eyebrow">{col.title}</p>
                <ul className="mt-3 space-y-2">
                  {col.links.map((l) => (
                    <li key={l.href}>
                      <a
                        href={l.href}
                        className="text-sm text-forest-mid transition-colors hover:text-merlot-mid"
                      >
                        {l.label}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </nav>
        </div>

        <div className="mt-12 flex flex-wrap items-end justify-between gap-6 border-t border-sand-deep/60 pt-8">
          <div className="max-w-2xl">
            <p className="text-xs leading-relaxed text-forest-soft">{INSTITUTION}.
            </p>
            <p className="mt-2 text-xs leading-relaxed text-forest-soft">
              Dataset statistics on this site are computed from the released{" "}
              <code className="font-mono">URTOX_v2.csv</code>; model results are attributed to the
              paper and table they come from. Both manuscripts were under review at the time this
              page was built, and neither has a DOI.
            </p>
          </div>
          <a
            href="#top"
            className="inline-flex items-center gap-1.5 text-xs font-medium text-forest-mid hover:text-merlot-mid"
          >
            <ArrowUp size={13} aria-hidden="true" />
            Back to top
          </a>
        </div>
      </div>
    </footer>
  );
}
