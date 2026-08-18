import React from "react";
import { AlertTriangle, FileSpreadsheet } from "lucide-react";
import stats from "../data/stats.json";
import { SOURCE_DOMAINS, SPLITS } from "../data/research";
import {
  Section,
  Reveal,
  Source,
  Callout,
  Donut,
  BarChart,
  Histogram,
  Table,
  Th,
  Td,
  Tr,
  Pill,
  n,
} from "./ui";

const C = {
  forest: "#26342B",
  forestMid: "#3A4E41",
  forestSoft: "#5C7264",
  merlot: "#3F0B0D",
  merlotMid: "#6E1B1F",
  merlotBright: "#8F3438",
  sand: "#D6C7AE",
};

const COLUMN_DOCS = {
  id: ["integer", "Unique record identifier"],
  text: ["string", "Raw Urdu text as collected"],
  label: ["categorical", "Sentence-level label: toxic / non_toxic"],
  sub_label: ["categorical", "Toxicity category: normal, offensive, hate, insult, slur, threat"],
  toxic_spans: ["string", "Free-text record of the annotated toxic surface form"],
  tokens: ["list[string]", "Word-level tokenisation of text"],
  toxic_list: ["list[string]", "Toxic words or phrases identified in the record"],
  BIO_tags: ["list[string]", "Per-token tag: B-Toxic, I-Toxic or O, aligned to tokens"],
};

const labelCount = (key) => stats.labels.find((l) => l.key === key).count;
const bioCount = (key) => stats.bio.find((b) => b.key === key).count;

export default function DatasetOverview() {
  const toxicTokens = bioCount("B-Toxic") + bioCount("I-Toxic");

  return (
    <Section
      id="dataset"
      index="02"
      eyebrow="Dataset overview"
      title="What is in URTOX"
      lead="Every figure in this section is computed from the released URTOX_v2.csv rather than transcribed from the papers, so it reflects the file a researcher actually downloads."
    >
      {/* ------------------------------------------------------------ schema */}
      <Reveal>
        <div className="grid gap-8 lg:grid-cols-[1.1fr_0.9fr]">
          <div>
            <h3 className="text-lg">Schema</h3>
            <p className="mt-2 text-sm text-forest-mid">
              {stats.columns.length} columns, {n(stats.records)} rows, single CSV file.
            </p>
            <Table className="mt-4" caption="URTOX column schema">
              <>
                <Th>Column</Th>
                <Th>Type</Th>
                <Th>Description</Th>
              </>
              {stats.columns.map((col) => (
                <Tr key={col}>
                  <Td strong>
                    <code className="font-mono text-xs">{col}</code>
                  </Td>
                  <Td>
                    <span className="font-mono text-xs text-forest-soft">
                      {COLUMN_DOCS[col]?.[0] ?? "n/a"}
                    </span>
                  </Td>
                  <Td className="text-xs">{COLUMN_DOCS[col]?.[1] ?? "n/a"}</Td>
                </Tr>
              ))}
            </Table>
          </div>

          <div className="space-y-6">
            <div className="card p-6">
              <h3 className="text-base">Label balance</h3>
              <div className="mt-5">
                <Donut
                  centerValue={n(stats.records)}
                  centerLabel="RECORDS"
                  data={[
                    { label: "Toxic", value: labelCount("toxic"), color: C.merlotMid },
                    { label: "Non-toxic", value: labelCount("non_toxic"), color: C.forestSoft },
                  ]}
                />
              </div>
              <Source>
                Computed from URTOX_v2.csv. Matches the 54% / 46% balance reported in MUTEX §3.2.
              </Source>
            </div>

            <div className="card p-6">
              <h3 className="text-base">Token-level tag distribution</h3>
              <div className="mt-4">
                <BarChart
                  data={[
                    { label: "O", value: bioCount("O"), tone: "forest" },
                    { label: "B-Toxic", value: bioCount("B-Toxic"), tone: "merlot" },
                    { label: "I-Toxic", value: bioCount("I-Toxic"), tone: "merlot" },
                  ]}
                  tone="mixed"
                />
              </div>
              <p className="mt-4 text-xs leading-relaxed text-forest-soft">
                {n(toxicTokens)} of {n(stats.totalTokens)} tokens carry a toxic tag, or{" "}
                {((toxicTokens / stats.totalTokens) * 100).toFixed(1)}%. Toxicity is sparse, which is
                why token-level F1 rather than accuracy is the meaningful metric.
              </p>
            </div>
          </div>
        </div>
      </Reveal>

      {/* -------------------------------------------------- category + spans */}
      <Reveal className="mt-16">
        <div className="grid gap-8 md:grid-cols-2">
          <div className="card p-6">
            <h3 className="text-base">Toxicity categories</h3>
            <p className="mt-1.5 text-xs text-forest-soft">
              The <code className="font-mono">sub_label</code> column, all {n(stats.records)} records.
            </p>
            <div className="mt-5">
              <BarChart
                data={stats.subLabels.map((s) => ({
                  label: s.key,
                  value: s.count,
                  tone: s.key === "normal" ? "forest" : "merlot",
                }))}
                tone="mixed"
              />
            </div>
            <Source>
              Computed from URTOX_v2.csv. MUTEX Table 5 reports a different category breakdown
              (hate 2,145 / insults 3,892 / offensive 5,124 / profanity 1,652) against a
              14,342-sample count; the released file is shown here.
            </Source>
          </div>

          <div className="card p-6">
            <h3 className="text-base">Span length</h3>
            <p className="mt-1.5 text-xs text-forest-soft">
              Tokens per annotated span, across all {n(stats.totalSpans)} spans.
            </p>
            <div className="mt-6">
              <Histogram
                tone={C.merlotMid}
                data={stats.spanLengthHist.map((s) => ({
                  label: s.plus ? "9+" : String(s.length),
                  value: s.count,
                }))}
              />
            </div>
            <p className="mt-4 text-xs leading-relaxed text-forest-soft">
              Mean span length is {stats.meanSpanTokens} tokens, and{" "}
              {((stats.spanLengthHist[0].count / stats.totalSpans) * 100).toFixed(0)}% of spans are a
              single token. Long spans are rare, which is consistent with the reported rise in
              false-negative rate on spans of five or more tokens.
            </p>
          </div>
        </div>
      </Reveal>

      {/* ---------------------------------------------------- text structure */}
      <Reveal className="mt-16">
        <div className="grid gap-8 md:grid-cols-2">
          <div className="card p-6">
            <h3 className="text-base">Record length</h3>
            <p className="mt-1.5 text-xs text-forest-soft">Tokens per record.</p>
            <div className="mt-6">
              <Histogram
                tone={C.forestMid}
                data={stats.tokenBuckets.map((b) => ({ label: b.key, value: b.count }))}
              />
            </div>
            <dl className="mt-5 grid grid-cols-4 gap-3 border-t border-sand-deep/50 pt-4 text-center">
              {[
                ["Mean", stats.tokenLength.mean],
                ["Median", stats.tokenLength.median],
                ["Min", stats.tokenLength.min],
                ["Max", stats.tokenLength.max],
              ].map(([k, v]) => (
                <div key={k}>
                  <dd className="font-mono text-sm text-forest-deep">{v}</dd>
                  <dt className="mt-0.5 text-[0.65rem] uppercase tracking-wide text-forest-soft">
                    {k}
                  </dt>
                </div>
              ))}
            </dl>
            <Source>
              Computed from URTOX_v2.csv. MUTEX Table 3 reports a mean post length of 102 tokens; the
              released file gives {stats.tokenLength.mean}.
            </Source>
          </div>

          <div className="card p-6">
            <h3 className="text-base">Text characteristics</h3>
            <p className="mt-1.5 text-xs text-forest-soft">
              Surface features detected across the released records.
            </p>
            <div className="mt-5">
              <BarChart
                max={stats.records}
                data={[
                  {
                    label: "Contains digits",
                    value: stats.scriptFeatures.digits,
                    display: `${n(stats.scriptFeatures.digits)} (${(
                      (stats.scriptFeatures.digits / stats.records) *
                      100
                    ).toFixed(1)}%)`,
                  },
                  {
                    label: "Contains a Latin-script run",
                    value: stats.scriptFeatures.latinRun,
                    display: `${n(stats.scriptFeatures.latinRun)} (${(
                      (stats.scriptFeatures.latinRun / stats.records) *
                      100
                    ).toFixed(1)}%)`,
                  },
                  {
                    label: "Contains emoji",
                    value: stats.scriptFeatures.emoji,
                    display: `${n(stats.scriptFeatures.emoji)} (${(
                      (stats.scriptFeatures.emoji / stats.records) *
                      100
                    ).toFixed(1)}%)`,
                  },
                ]}
              />
            </div>
            <div className="mt-5 rounded-md border border-sand-deep/50 bg-sand/25 p-4">
              <p className="text-xs leading-relaxed text-forest-mid">
                The Latin-script figure is a crude proxy, counting any run of three or more Latin characters, and it
                and is a lower bound on code-switching, since Romanised Urdu written entirely in
                Latin script and short English insertions are counted differently. The papers report
                that roughly 18% of online Urdu content is Roman script and that code-switching
                affects 35–40% of posts; those are characterisations of Urdu online content in the
                literature, not measurements of this CSV.
              </p>
            </div>
          </div>
        </div>
      </Reveal>

      {/* -------------------------------------------------- sources + splits */}
      <Reveal className="mt-16">
        <div className="grid gap-8 lg:grid-cols-2">
          <div>
            <h3 className="text-lg">Source domains</h3>
            <p className="mt-2 max-w-prose text-sm leading-relaxed text-forest-mid">
              Data was collected across three domains to encourage generalisation. The released CSV
              carries no source column, so this breakdown is as reported by the papers and cannot be
              recomputed from the file.
            </p>
            <Table className="mt-5" caption="Source domain breakdown">
              <>
                <Th>Domain</Th>
                <Th align="right">Samples</Th>
                <Th align="right">Toxic</Th>
              </>
              {SOURCE_DOMAINS.map((d) => (
                <Tr key={d.name}>
                  <Td strong>
                    {d.name}
                    <span className="mt-0.5 block text-xs font-normal text-forest-soft">
                      {d.detail}
                    </span>
                  </Td>
                  <Td align="right" className="font-mono text-xs">
                    {n(d.samples)}
                  </Td>
                  <Td align="right" className="font-mono text-xs">
                    {d.toxic}%
                  </Td>
                </Tr>
              ))}
              <Tr highlight>
                <Td strong>Total (as reported)</Td>
                <Td align="right" className="font-mono text-xs font-medium">
                  {n(SOURCE_DOMAINS.reduce((s, d) => s + d.samples, 0))}
                </Td>
                <Td align="right" className="font-mono text-xs font-medium">
                  54%
                </Td>
              </Tr>
            </Table>
            <Source>MUTEX §3.2 and Table 2; MUTEX-M Table 1.</Source>
          </div>

          <div>
            <h3 className="text-lg">Evaluation splits</h3>
            <p className="mt-2 max-w-prose text-sm leading-relaxed text-forest-mid">
              The public release is distributed as a single file. Both papers construct their own
              stratified splits, preserving the toxic / non-toxic ratio, and they differ, so a
              reproduction should follow whichever paper it is comparing against.
            </p>
            <div className="mt-5 space-y-4">
              {[
                { key: "p1", name: "MUTEX, 80 / 10 / 10", note: "MUTEX §3.6" },
                { key: "p2", name: "MUTEX-M, 80 / 20, seed 42", note: "MUTEX-M §5" },
              ].map(({ key, name, note }) => (
                <div key={key} className="card p-5">
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-sm font-semibold text-forest-deep">{name}</p>
                    <Pill tone="outline">{note}</Pill>
                  </div>
                  <div className="mt-4 flex overflow-hidden rounded-md">
                    {SPLITS[key].map((s, i) => {
                      const total = SPLITS[key].reduce((acc, x) => acc + x.samples, 0);
                      const colors = [C.forest, C.forestSoft, C.merlotMid];
                      return (
                        <div
                          key={s.name}
                          className="px-2 py-2.5 text-center text-[0.65rem] font-medium text-cream"
                          style={{
                            width: `${(s.samples / total) * 100}%`,
                            backgroundColor: colors[i],
                          }}
                          title={`${s.name}: ${n(s.samples)}`}
                        >
                          {s.share}
                        </div>
                      );
                    })}
                  </div>
                  <dl className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-xs">
                    {SPLITS[key].map((s) => (
                      <div key={s.name} className="flex gap-1.5">
                        <dt className="text-forest-soft">{s.name}</dt>
                        <dd className="font-mono text-forest-deep">{n(s.samples)}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Reveal>

      {/* ------------------------------------------------------ data quality */}
      <Reveal className="mt-16">
        <h3 className="text-lg">Data quality notes</h3>
        <p className="mt-2 max-w-prose text-sm leading-relaxed text-forest-mid">
          Recomputing the annotation columns surfaces some inconsistencies in the released file.
          They are listed here so anyone building on URTOX knows what to expect before writing a
          loader.
        </p>

        <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[
            {
              value: n(stats.quality.toxicWithoutSpan),
              label: "Toxic records with no B-Toxic tag",
              note: `of ${n(labelCount("toxic"))} toxic records, where the sentence label is present but the span annotation is absent`,
            },
            {
              value: n(stats.quality.nonToxicWithSpan),
              label: "Non-toxic records carrying a toxic tag",
              note: `of ${n(labelCount("non_toxic"))} non-toxic records, where the sentence and token labels disagree`,
            },
            {
              value: String(stats.quality.tokenTagLengthMismatch),
              label: "tokens / BIO_tags length mismatches",
              note: "spans cannot be aligned for these records",
            },
            {
              value: n(stats.quality.toxicListPopulated),
              label: "Records with a usable toxic_list",
              note: `${n(stats.distinctToxicPhrases)} distinct toxic phrases across the corpus`,
            },
          ].map((q) => (
            <div key={q.label} className="rounded-lg border border-merlot-bright/25 bg-merlot-wash/40 p-5">
              <div className="font-serif text-2xl text-merlot">{q.value}</div>
              <p className="mt-2 text-sm font-medium text-forest-deep">{q.label}</p>
              <p className="mt-1 text-xs leading-relaxed text-forest-soft">{q.note}</p>
            </div>
          ))}
        </div>

        <div className="mt-6 grid gap-4 md:grid-cols-2">
          <Callout icon={AlertTriangle} tone="merlot" title="The toxic_spans column is not offsets">
            Despite its name, <code className="font-mono text-xs">toxic_spans</code> holds a
            free-text brace-wrapped surface form (populated in 7,437 records, empty in 6,884), not
            character offsets. Span positions should be derived from{" "}
            <code className="font-mono text-xs">BIO_tags</code> aligned against{" "}
            <code className="font-mono text-xs">tokens</code>, which is what both the papers and this
            site do.
          </Callout>
          <Callout icon={FileSpreadsheet} title="Row count">
            The released file contains {n(stats.records)} rows, confirmed by the Hugging Face dataset
            viewer. MUTEX reports 14,342 samples and MUTEX-M reports 14,338; identifiers run from 1
            to 14,342 with gaps. This site uses the file.
          </Callout>
        </div>
      </Reveal>
    </Section>
  );
}
