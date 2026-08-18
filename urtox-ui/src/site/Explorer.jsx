import React, { useEffect, useMemo, useState } from "react";
import {
  Search,
  Eye,
  EyeOff,
  Loader2,
  ArrowUpDown,
  X,
  ExternalLink,
  AlertTriangle,
} from "lucide-react";
import { Section, Pill, Source, n } from "./ui";
import stats from "../data/stats.json";
import { LINKS } from "../data/research";

const PAGE_SIZE = 12;

const SUB_LABELS = ["normal", "offensive", "hate", "insult", "slur", "threat"];

const SORTS = [
  { key: "id", label: "Record id" },
  { key: "spans", label: "Span count" },
  { key: "length", label: "Token count" },
];

/** Splits a record's tokens into runs, marking which runs are annotated toxic. */
function segments(tokens, spans) {
  const out = [];
  let cursor = 0;
  for (const [start, end] of spans) {
    if (start > cursor) out.push({ toxic: false, text: tokens.slice(cursor, start).join(" ") });
    out.push({ toxic: true, text: tokens.slice(start, end).join(" ") });
    cursor = end;
  }
  if (cursor < tokens.length) out.push({ toxic: false, text: tokens.slice(cursor).join(" ") });
  return out;
}

function Record({ rec, reveal }) {
  const tokens = rec.toks.split(" ");
  const parts = segments(tokens, rec.spans);
  const toxic = rec.label === "toxic";
  const hidden = toxic && !reveal;
  // the released file contains records whose sentence label and token tags
  // disagree; flagging them keeps the explorer from looking like it mis-renders
  const mismatch =
    (!toxic && rec.spans.length > 0) || (toxic && rec.spans.length === 0);

  return (
    <li className="rounded-lg border border-sand-deep/60 bg-white/70 p-4 transition-colors hover:border-forest-soft/50">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-[0.68rem] text-forest-soft">id {rec.id}</span>
        <Pill tone={toxic ? "merlot" : "sand"}>{rec.label}</Pill>
        <Pill tone="outline">{rec.sub}</Pill>
        {mismatch && (
          <span
            title="The sentence label and the token tags disagree in the released file"
            className="inline-flex items-center gap-1 rounded-full border border-amber-700/30 bg-amber-100/60 px-2 py-0.5 text-[0.65rem] font-medium text-amber-900"
          >
            <AlertTriangle size={10} aria-hidden="true" />
            label / span mismatch
          </span>
        )}
        <span className="ml-auto font-mono text-[0.65rem] text-forest-soft">
          {tokens.length} tokens · {rec.spans.length} span{rec.spans.length === 1 ? "" : "s"}
        </span>
      </div>

      <p
        className={`urdu mt-3 text-lg text-forest-deep transition-[filter] ${
          hidden ? "select-none blur-[5px]" : ""
        }`}
        lang="ur"
        dir="rtl"
        aria-hidden={hidden}
      >
        {parts.map((p, i) =>
          p.toxic ? (
            <span
              key={i}
              className="rounded bg-merlot-bright/22 px-1 underline decoration-merlot-bright decoration-2 underline-offset-[7px]"
            >
              {p.text}{" "}
            </span>
          ) : (
            <span key={i}>{p.text} </span>
          )
        )}
      </p>

      {hidden && (
        <p className="mt-2 text-xs text-forest-soft">
          Hidden, because this record is labelled toxic. Use “Show toxic text” to reveal.
        </p>
      )}

      {rec.spans.length > 0 && !hidden && (
        <ul className="mt-3 flex flex-wrap gap-1.5 border-t border-sand-deep/40 pt-3">
          {rec.spans.map(([s, e], i) => (
            <li
              key={i}
              className="rounded border border-merlot-bright/30 bg-merlot-wash/60 px-2 py-0.5"
            >
              <span className="urdu-inline text-sm text-merlot" lang="ur">
                {tokens.slice(s, e).join(" ")}
              </span>
              <span className="ml-1.5 font-mono text-[0.6rem] text-forest-soft">
                [{s},{e})
              </span>
            </li>
          ))}
        </ul>
      )}
    </li>
  );
}

export default function Explorer() {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState(null);
  const [query, setQuery] = useState("");
  const [label, setLabel] = useState("all");
  const [sub, setSub] = useState("all");
  const [spanFilter, setSpanFilter] = useState("all");
  const [sort, setSort] = useState("id");
  const [asc, setAsc] = useState(true);
  const [page, setPage] = useState(0);
  const [reveal, setReveal] = useState(false);

  // The explorer sample lives in public/ so it is fetched only when this
  // section is actually used, keeping it out of the main JS bundle.
  useEffect(() => {
    let cancelled = false;
    fetch(`${process.env.PUBLIC_URL || ""}/data/sample.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`sample.json returned ${r.status}`);
        return r.json();
      })
      .then((data) => {
        if (!cancelled) setRows(data);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const filtered = useMemo(() => {
    if (!rows) return [];
    const q = query.trim();
    let out = rows;
    if (label !== "all") out = out.filter((r) => r.label === label);
    if (sub !== "all") out = out.filter((r) => r.sub === sub);
    if (spanFilter === "with") out = out.filter((r) => r.spans.length > 0);
    if (spanFilter === "without") out = out.filter((r) => r.spans.length === 0);
    if (spanFilter === "multi") out = out.filter((r) => r.spans.length > 1);
    if (spanFilter === "mismatch")
      out = out.filter(
        (r) =>
          (r.label !== "toxic" && r.spans.length > 0) ||
          (r.label === "toxic" && r.spans.length === 0)
      );
    if (q) out = out.filter((r) => r.text.includes(q) || String(r.id) === q);

    const key = {
      id: (r) => r.id,
      spans: (r) => r.spans.length,
      length: (r) => r.toks.split(" ").length,
    }[sort];

    return [...out].sort((a, b) => (asc ? key(a) - key(b) : key(b) - key(a)));
  }, [rows, query, label, sub, spanFilter, sort, asc]);

  // Any change to the result set should return the reader to the first page.
  useEffect(() => {
    setPage(0);
  }, [query, label, sub, spanFilter, sort, asc]);

  const pages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const view = filtered.slice(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE);
  const active = label !== "all" || sub !== "all" || spanFilter !== "all" || query.trim();

  const selectClass =
    "rounded-md border border-sand-deep bg-white px-3 py-2 text-sm text-forest focus:border-forest-soft";

  return (
    <Section
      id="explore"
      index="04"
      eyebrow="Explore the dataset"
      title="Inspect the records"
      lead="Search, filter and sort a stratified sample of the corpus, with toxic spans rendered in place. Toxic text is blurred until you choose to reveal it."
    >
      <div className="rounded-lg border border-merlot-bright/30 bg-merlot-wash/50 px-5 py-4">
        <div className="flex items-start gap-3">
          <AlertTriangle size={17} className="mt-0.5 shrink-0 text-merlot-mid" aria-hidden="true" />
          <p className="text-sm leading-relaxed text-forest-mid">
  <span className="font-semibold text-forest-deep">Content warning.</span> These are
            unedited records collected from public social media, news comment sections and YouTube.
            They contain insults, profanity and hate speech in Urdu. The material is presented for
            research inspection, exactly as annotated. Some records carry a highlighted span even
            though the sentence is labelled non-toxic, and the reverse. Those are flagged, and the
            reason is explained below the results.
          </p>
        </div>
      </div>

      {/* controls */}
      <div className="mt-8 rounded-lg border border-sand-deep/60 bg-white/60 p-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center">
          <div className="relative flex-1">
            <Search
              size={15}
              className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-forest-soft"
              aria-hidden="true"
            />
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search Urdu text or a record id…"
              aria-label="Search records"
              className="w-full rounded-md border border-sand-deep bg-white py-2 pl-9 pr-3 text-sm text-forest placeholder:text-forest-soft/70 focus:border-forest-soft"
            />
          </div>

          <div className="flex flex-wrap gap-2">
            <select
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              aria-label="Filter by label"
              className={selectClass}
            >
              <option value="all">All labels</option>
              <option value="toxic">Toxic</option>
              <option value="non_toxic">Non-toxic</option>
            </select>

            <select
              value={sub}
              onChange={(e) => setSub(e.target.value)}
              aria-label="Filter by category"
              className={selectClass}
            >
              <option value="all">All categories</option>
              {SUB_LABELS.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>

            <select
              value={spanFilter}
              onChange={(e) => setSpanFilter(e.target.value)}
              aria-label="Filter by span presence"
              className={selectClass}
            >
              <option value="all">Any spans</option>
              <option value="with">Has a span</option>
              <option value="multi">Multiple spans</option>
              <option value="without">No span</option>
              <option value="mismatch">Label / span mismatch</option>
            </select>

            <button
              type="button"
              onClick={() => {
                const i = SORTS.findIndex((s) => s.key === sort);
                if (asc) setAsc(false);
                else {
                  setAsc(true);
                  setSort(SORTS[(i + 1) % SORTS.length].key);
                }
              }}
              className="inline-flex items-center gap-1.5 rounded-md border border-sand-deep bg-white px-3 py-2 text-sm text-forest hover:border-forest-soft"
              aria-label={`Sort by ${SORTS.find((s) => s.key === sort).label}, ${
                asc ? "ascending" : "descending"
              }`}
            >
              <ArrowUpDown size={14} aria-hidden="true" />
              {SORTS.find((s) => s.key === sort).label}
              <span className="text-forest-soft">{asc ? "↑" : "↓"}</span>
            </button>

            <button
              type="button"
              onClick={() => setReveal((v) => !v)}
              aria-pressed={reveal}
              className={`inline-flex items-center gap-1.5 rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                reveal
                  ? "bg-merlot text-cream hover:bg-merlot-mid"
                  : "border border-sand-deep bg-white text-forest hover:border-forest-soft"
              }`}
            >
              {reveal ? <EyeOff size={14} aria-hidden="true" /> : <Eye size={14} aria-hidden="true" />}
              {reveal ? "Hide toxic text" : "Show toxic text"}
            </button>
          </div>
        </div>

        {active && (
          <div className="mt-3 flex items-center gap-3 border-t border-sand-deep/50 pt-3">
            <p className="text-xs text-forest-soft">
              {n(filtered.length)} matching record{filtered.length === 1 ? "" : "s"}
            </p>
            <button
              type="button"
              onClick={() => {
                setQuery("");
                setLabel("all");
                setSub("all");
                setSpanFilter("all");
              }}
              className="inline-flex items-center gap-1 text-xs text-merlot-mid hover:underline"
            >
              <X size={12} aria-hidden="true" />
              Clear filters
            </button>
          </div>
        )}
      </div>

      {/* results */}
      <div className="mt-6" aria-live="polite">
        {error && (
          <div className="rounded-lg border border-merlot-bright/40 bg-merlot-wash/50 p-6 text-sm text-forest-mid">
            Could not load the sample file ({error}). The complete dataset remains available{" "}
            <a href={LINKS.dataset} target="_blank" rel="noreferrer" className="link-underline">
              on Hugging Face
            </a>
            .
          </div>
        )}

        {!rows && !error && (
          <div className="flex items-center justify-center gap-2.5 rounded-lg border border-sand-deep/60 bg-white/50 py-16 text-sm text-forest-soft">
            <Loader2 size={16} className="animate-spin" aria-hidden="true" />
            Loading records…
          </div>
        )}

        {rows && filtered.length === 0 && (
          <div className="rounded-lg border border-sand-deep/60 bg-white/50 py-16 text-center text-sm text-forest-soft">
            No records match these filters.
          </div>
        )}

        {view.length > 0 && (
          <>
            <ul className="grid gap-3 md:grid-cols-2">
              {view.map((rec) => (
                <Record key={rec.id} rec={rec} reveal={reveal} />
              ))}
            </ul>

            <nav
              className="mt-6 flex items-center justify-between gap-4"
              aria-label="Record pagination"
            >
              <button
                type="button"
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
                className="rounded-md border border-sand-deep bg-white px-3.5 py-2 text-sm text-forest disabled:opacity-40"
              >
                Previous
              </button>
              <p className="font-mono text-xs text-forest-soft">
                Page {page + 1} of {n(pages)} · showing {page * PAGE_SIZE + 1}–
                {Math.min((page + 1) * PAGE_SIZE, filtered.length)} of {n(filtered.length)}
              </p>
              <button
                type="button"
                onClick={() => setPage((p) => Math.min(pages - 1, p + 1))}
                disabled={page >= pages - 1}
                className="rounded-md border border-sand-deep bg-white px-3.5 py-2 text-sm text-forest disabled:opacity-40"
              >
                Next
              </button>
            </nav>
          </>
        )}
      </div>

      <div className="mt-8 rounded-lg border border-amber-700/25 bg-amber-50/70 p-5">
        <div className="flex items-start gap-3">
          <AlertTriangle size={17} className="mt-0.5 shrink-0 text-amber-800" aria-hidden="true" />
          <div>
            <p className="text-sm font-semibold text-forest-deep">
              Why some non-toxic records show a highlighted span
            </p>
            <p className="mt-2 text-sm leading-relaxed text-forest-mid">
              This is a property of the released annotation, not a rendering error. Recomputing the
              file shows {n(stats.quality.nonToxicWithSpan)} records labelled{" "}
              <code className="font-mono text-xs">non_toxic</code> that still carry at least one
              toxic token tag, and {n(stats.quality.toxicWithoutSpan)} records labelled{" "}
              <code className="font-mono text-xs">toxic</code> that carry none. The two annotation
              layers, the sentence label and the token tags, were not fully reconciled.
            </p>
            <p className="mt-2.5 text-sm leading-relaxed text-forest-mid">
              The disagreements fall into two kinds. Sometimes the span is wrong: a place name or a
              personal name gets tagged, as with{" "}
              <span className="urdu-inline" lang="ur">جہلم</span> (Jhelum, a city) in record 14083.
              Sometimes the sentence label is the questionable one: record 14008 is labelled
              non-toxic yet contains{" "}
              <span className="urdu-inline" lang="ur">بے وقوف</span> (fool), which the annotator
              did tag. Filter by <span className="font-medium">Label / span mismatch</span> above to
              inspect them, and treat those records with care when training.
            </p>
          </div>
        </div>
      </div>

      <div className="mt-8 rounded-lg border border-sand-deep/60 bg-sand/25 p-5">
        <p className="text-sm leading-relaxed text-forest-mid">
          <span className="font-semibold text-forest-deep">What you are browsing.</span> This
          explorer loads a {n(stats.sample.size)}-record sample, drawn with a fixed seed of{" "}
          {stats.sample.seed} and stratified by <code className="font-mono text-xs">sub_label</code>{" "}
          so every category is represented in proportion. Records whose{" "}
          <code className="font-mono text-xs">tokens</code> and{" "}
          <code className="font-mono text-xs">BIO_tags</code> lengths disagree are excluded, because
          their spans cannot be rendered accurately. Serving a sample rather than all{" "}
          {n(stats.records)} records keeps the page fast; search and filtering therefore operate over
          the sample, not the full corpus.
        </p>
        <a
          href={LINKS.dataset}
          target="_blank"
          rel="noreferrer"
          className="mt-3 inline-flex items-center gap-1.5 text-sm font-medium text-merlot-mid hover:underline"
        >
          Browse or download all {n(stats.records)} records on Hugging Face
          <ExternalLink size={13} aria-hidden="true" />
        </a>
        <Source>Sample generated by scripts/build_data.py from URTOX_v2.csv.</Source>
      </div>

      {/* toxic lexicon */}
      <div className="mt-14">
        <h3 className="text-lg">Most frequent annotated toxic phrases</h3>
        <p className="mt-2 max-w-prose text-sm leading-relaxed text-forest-mid">
          Derived from the <code className="font-mono text-xs">toxic_list</code> column across all{" "}
          {n(stats.records)} records, giving {n(stats.distinctToxicPhrases)} distinct phrases in total.
          Frequency counts describe what the corpus contains; they are not a lexicon endorsement.
        </p>
        <ul className="mt-5 flex flex-wrap gap-2">
          {stats.topToxicPhrases.map((p) => (
            <li
              key={p.phrase}
              className="flex items-center gap-2 rounded-md border border-sand-deep/60 bg-white/70 px-3 py-1.5"
            >
              <span className="urdu-inline text-base text-forest-deep" lang="ur">
                {p.phrase}
              </span>
              <span className="font-mono text-[0.65rem] text-forest-soft">{p.count}</span>
            </li>
          ))}
        </ul>
        <Source>Computed from URTOX_v2.csv.</Source>
      </div>
    </Section>
  );
}
