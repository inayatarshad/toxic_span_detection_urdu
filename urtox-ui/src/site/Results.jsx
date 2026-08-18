import React, { useState } from "react";
import { Section, Reveal, Source, Table, Th, Td, Tr, MetricBars, BarChart, Callout, Pill } from "./ui";
import { Info } from "lucide-react";
import {
  P1_MODELS,
  P1_DOMAIN,
  P1_TRANSFER,
  P1_PREPROCESS_ABLATION,
  P1_LEARNING_CURVE,
  P1_ERRORS,
  P2_TEXT_MODELS,
  P2_ABLATION,
  P2_LLM_BASELINES,
  P2_ERRORS,
  P2_FN_BY_SPAN_LENGTH,
} from "../data/research";

const TABS = [
  { key: "models", label: "Model comparison" },
  { key: "ablation", label: "Ablations" },
  { key: "domain", label: "Cross-domain" },
  { key: "errors", label: "Error analysis" },
  { key: "llm", label: "LLM baselines" },
];

/** Colour-graded cell for the cross-domain transfer matrix. */
function transferCell(value, isDiagonal) {
  const t = Math.max(0, Math.min(1, (value - 50) / 14));
  return {
    backgroundColor: `rgba(110, 27, 31, ${0.06 + t * 0.24})`,
    fontWeight: isDiagonal ? 600 : 400,
  };
}

export default function Results() {
  const [tab, setTab] = useState("models");

  return (
    <Section
      id="results"
      index="07"
      tone="cream"
      eyebrow="Research results"
      title="What the dataset enabled"
      lead="Results reported by the two papers on their own held-out test sets. The two papers use different splits, so their numbers are comparable within a paper but should be read carefully across them."
    >
      <div
        role="tablist"
        aria-label="Result views"
        className="flex flex-wrap gap-1 border-b border-sand-deep/60 pb-3"
      >
        {TABS.map((t) => (
          <button
            key={t.key}
            role="tab"
            aria-selected={tab === t.key}
            onClick={() => setTab(t.key)}
            className={`rounded-md px-3.5 py-2 text-sm font-medium transition-colors ${
              tab === t.key
                ? "bg-forest text-cream"
                : "text-forest-mid hover:bg-sand/40"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="mt-8">
        {/* ---------------------------------------------------- models */}
        {tab === "models" && (
          <Reveal className="grid gap-10 lg:grid-cols-2">
            <div>
              <h3 className="text-lg">MUTEX, token-level F1</h3>
              <p className="mt-2 text-sm text-forest-mid">
                Test set of 1,434 samples, averaged over five random seeds. Standard deviations in
                the paper are given in parentheses on each figure.
              </p>
              <Table className="mt-5" caption="MUTEX model comparison">
                <>
                  <Th>Model</Th>
                  <Th align="center">CRF</Th>
                  <Th align="right">P</Th>
                  <Th align="right">R</Th>
                  <Th align="right">F1</Th>
                </>
                {P1_MODELS.map((m) => (
                  <Tr key={m.model} highlight={m.best}>
                    <Td strong>{m.model}</Td>
                    <Td align="center" className="text-forest-soft">
                      {m.crf ? "✓" : "n/a"}
                    </Td>
                    <Td align="right" className="font-mono text-xs">{m.precision.toFixed(1)}</Td>
                    <Td align="right" className="font-mono text-xs">{m.recall.toFixed(1)}</Td>
                    <Td align="right" className="font-mono text-xs font-semibold">
                      {m.f1.toFixed(1)}
                    </Td>
                  </Tr>
                ))}
              </Table>
              <Source>MUTEX Table 4.</Source>
            </div>

            <div>
              <h3 className="text-lg">MUTEX-M, token-level F1</h3>
              <p className="mt-2 text-sm text-forest-mid">
                Test set of 2,868 samples. The gain over the baseline comes from training-procedure
                corrections, with no architectural change.
              </p>
              <Table className="mt-5" caption="MUTEX-M model comparison">
                <>
                  <Th>Model</Th>
                  <Th align="right">P</Th>
                  <Th align="right">R</Th>
                  <Th align="right">F1</Th>
                </>
                {P2_TEXT_MODELS.map((m) => (
                  <Tr key={m.model} highlight={m.best}>
                    <Td strong>{m.model}</Td>
                    <Td align="right" className="font-mono text-xs">{m.precision.toFixed(1)}</Td>
                    <Td align="right" className="font-mono text-xs">{m.recall.toFixed(1)}</Td>
                    <Td align="right" className="font-mono text-xs font-semibold">
                      {m.f1.toFixed(1)}
                    </Td>
                  </Tr>
                ))}
              </Table>
              <Source>MUTEX-M Table 9.</Source>

              <div className="mt-6">
                <Callout icon={Info} title="Recall moved more than precision">
                  Recall rises from 58.0 to 71.5 while precision moves from 60.0 to 64.2. The authors
                  attribute this to subword label propagation recovering span coverage that the
                  ignore-index artifact had been discarding.
                </Callout>
              </div>
            </div>
          </Reveal>
        )}

        {/* -------------------------------------------------- ablations */}
        {tab === "ablation" && (
          <Reveal className="grid gap-12 lg:grid-cols-2">
            <div>
              <h3 className="text-lg">Training-procedure ablation</h3>
              <p className="mt-2 text-sm text-forest-mid">
                Each row adds one change to the configuration above it.
              </p>
              <ol className="mt-5 space-y-2">
                {P2_ABLATION.map((a) => (
                  <li
                    key={a.config}
                    className="flex items-center gap-4 rounded-md border border-sand-deep/60 bg-white/70 px-4 py-3"
                  >
                    <div className="min-w-0 flex-1">
                      <p className="text-sm text-forest-deep">{a.config}</p>
                      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-sand/70">
                        <div
                          className="h-full rounded-full bg-merlot-mid transition-[width] duration-700"
                          style={{ width: `${((a.f1 - 55) / 13) * 100}%` }}
                        />
                      </div>
                    </div>
                    <div className="shrink-0 text-right">
                      <span className="font-mono text-sm font-semibold text-forest-deep">
                        {a.f1.toFixed(1)}
                      </span>
                      {a.gain != null && (
                        <span className="ml-2 font-mono text-xs text-merlot-mid">+{a.gain}</span>
                      )}
                    </div>
                  </li>
                ))}
              </ol>
              <Source>MUTEX-M Table 12.</Source>

              <h3 className="mt-10 text-lg">Learning curve</h3>
              <Table className="mt-4" caption="Effect of training data size">
                <>
                  <Th>Training data</Th>
                  <Th align="right">Samples</Th>
                  <Th align="right">F1</Th>
                  <Th align="right">SD</Th>
                </>
                {P1_LEARNING_CURVE.map((r) => (
                  <Tr key={r.share}>
                    <Td strong>{r.share}</Td>
                    <Td align="right" className="font-mono text-xs">
                      {r.samples.toLocaleString()}
                    </Td>
                    <Td align="right" className="font-mono text-xs">{r.f1.toFixed(1)}</Td>
                    <Td align="right" className="font-mono text-xs text-forest-soft">
                      ±{r.sd}
                    </Td>
                  </Tr>
                ))}
              </Table>
              <Source>
                MUTEX Table 10. The curve flattens after roughly 11,500 samples, which the authors
                read as the dataset approaching sufficiency for this task rather than as evidence
                that more data would not help.
              </Source>
            </div>

            <div>
              <h3 className="text-lg">Preprocessing ablation</h3>
              <p className="mt-2 text-sm text-forest-mid">
                Five-fold cross-validation, removing one preprocessing step at a time from the full
                pipeline.
              </p>
              <Table className="mt-5" caption="Preprocessing ablation">
                <>
                  <Th>Configuration</Th>
                  <Th align="right">F1</Th>
                  <Th align="right">Δ</Th>
                  <Th align="right">p</Th>
                </>
                {P1_PREPROCESS_ABLATION.map((r) => (
                  <Tr key={r.config} highlight={r.delta === null}>
                    <Td strong className="text-xs sm:text-sm">{r.config}</Td>
                    <Td align="right" className="font-mono text-xs">
                      {r.f1.toFixed(1)}
                      <span className="ml-1 text-forest-soft">±{r.sd}</span>
                    </Td>
                    <Td align="right" className="font-mono text-xs">
                      {r.delta === null ? (
                        "n/a"
                      ) : (
                        <span className="text-merlot-mid">{r.delta.toFixed(1)}</span>
                      )}
                    </Td>
                    <Td align="right" className="font-mono text-xs text-forest-soft">
                      {r.p}
                    </Td>
                  </Tr>
                ))}
              </Table>
              <Source>MUTEX Table 11.</Source>

              <div className="mt-6">
                <Callout icon={Info} title="Preprocessing outweighs architecture">
                  Removing the whole preprocessing pipeline costs 6.2 F1 points; the CRF layer
                  contributes 1.3. Roman-to-Nastaliq conversion alone accounts for 3.7 of those
                  points, the single largest lever measured in either paper for the text model
                  before the subword fix.
                </Callout>
              </div>
            </div>
          </Reveal>
        )}

        {/* ------------------------------------------------- cross-domain */}
        {tab === "domain" && (
          <Reveal className="grid gap-12 lg:grid-cols-2">
            <div>
              <h3 className="text-lg">Performance by domain</h3>
              <p className="mt-2 text-sm text-forest-mid">
                XLM-RoBERTa + CRF trained on all domains, evaluated on each separately.
              </p>
              <div className="mt-6">
                <MetricBars
                  domainMin={50}
                  domainMax={70}
                  data={P1_DOMAIN.map((d) => ({
                    label: d.domain,
                    value: d.f1,
                    best: d.best,
                    note: `P ${d.precision.toFixed(1)} · R ${d.recall.toFixed(1)}`,
                  }))}
                />
              </div>
              <Source>MUTEX Table 6.</Source>

              <div className="mt-8">
                <Callout icon={Info} title="Formality drives the spread">
                  News text scores highest at 62.3 and social media lowest at 57.6. The authors
                  attribute the 4.7-point gap to standardised vocabulary and spelling in news
                  against slang, abbreviation and creative spelling in social posts.
                </Callout>
              </div>
            </div>

            <div>
              <h3 className="text-lg">Cross-domain transfer</h3>
              <p className="mt-2 text-sm text-forest-mid">
                Rows are the training domain, columns the test domain. Darker means higher F1.
              </p>
              <div className="mt-5 rounded-lg border border-sand-deep/60 bg-white/60">
                <table className="w-full table-auto border-collapse text-[0.8rem] sm:text-sm">
                  <caption className="sr-only">Cross-domain transfer matrix</caption>
                  <thead>
                    <tr className="border-b border-sand-deep/70 bg-sand/30">
                      <Th>Train ↓ / Test →</Th>
                      {P1_TRANSFER.columns.map((c) => (
                        <Th key={c} align="center">
                          {c}
                        </Th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {P1_TRANSFER.rows.map((r) => (
                      <tr key={r.train} className="border-b border-sand-deep/40 last:border-0">
                        <Td strong className="text-xs sm:text-sm">
                          {r.train}
                        </Td>
                        {r.values.map((v, i) => (
                          <td
                            key={i}
                            className="px-4 py-3 text-center font-mono text-xs text-forest-deep"
                            style={transferCell(v, r.diagonal === i)}
                          >
                            {v.toFixed(1)}
                            {r.diagonal === i && (
                              <span className="ml-1 text-[0.6rem] text-forest-soft">in-domain</span>
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <Source>
                MUTEX Table 8. The multi-domain row is a model trained on all three domains, not a
                single-domain transfer result.
              </Source>

              <p className="mt-5 max-w-prose text-sm leading-relaxed text-forest-mid">
                Single-domain models are stronger on their own domain. Social media reaches 61.3
                when trained only on social media, against 57.6 for the multi-domain model. The
                authors argue the multi-domain model is nonetheless preferable for deployment
                because it holds a consistent level across all three sources with one model instead
                of three.
              </p>
            </div>
          </Reveal>
        )}

        {/* ------------------------------------------------------ errors */}
        {tab === "errors" && (
          <Reveal className="grid gap-12 lg:grid-cols-2">
            <div>
              <h3 className="text-lg">MUTEX-M error types</h3>
              <p className="mt-2 text-sm text-forest-mid">
                816 misclassified tokens on the held-out test set, with 95% Wilson confidence
                intervals.
              </p>
              <Table className="mt-5" caption="Error type distribution">
                <>
                  <Th>Error type</Th>
                  <Th align="right">Count</Th>
                  <Th align="right">Share</Th>
                  <Th align="right">95% CI</Th>
                </>
                {P2_ERRORS.map((e) => (
                  <Tr key={e.type}>
                    <Td strong>{e.type}</Td>
                    <Td align="right" className="font-mono text-xs">{e.count}</Td>
                    <Td align="right" className="font-mono text-xs">{e.share}%</Td>
                    <Td align="right" className="font-mono text-xs text-forest-soft">
                      {e.ci}
                    </Td>
                  </Tr>
                ))}
              </Table>
              <Source>MUTEX-M Table 17.</Source>

              <h3 className="mt-10 text-lg">False negatives by span length</h3>
              <div className="mt-4">
                <BarChart
                  tone="merlot"
                  max={50}
                  data={P2_FN_BY_SPAN_LENGTH.map((r) => ({
                    label: `${r.length} token${r.length === "1" ? "" : "s"}`,
                    value: r.fnRate,
                    display: `${r.fnRate}%`,
                    note: `${r.spans.toLocaleString()} gold spans`,
                  }))}
                />
              </div>
              <Source>
                MUTEX-M Table 19. Longer spans are missed far more often, which is what motivates
                the span-based detection head listed as the first future priority.
              </Source>
            </div>

            <div>
              <h3 className="text-lg">MUTEX error types</h3>
              <p className="mt-2 text-sm text-forest-mid">
                From a manual review of 500 randomly sampled predictions in the earlier paper.
              </p>
              <div className="mt-5">
                <BarChart
                  tone="forest"
                  max={40}
                  data={P1_ERRORS.map((e) => ({
                    label: e.type,
                    value: e.share,
                    display: `${e.share}%`,
                  }))}
                />
              </div>
              <Source>MUTEX §4.4.1.</Source>

              <div className="mt-8 space-y-4">
                <Callout icon={Info} title="Boundary errors dominate both papers">
                  Boundaries are the largest failure category in MUTEX (34%) and in MUTEX-M (38.2%).
                  A recurring pattern: annotators excluded intensifiers from the span while the model
                  treats intensifier-plus-toxic-word as one unit.
                </Callout>
                <Callout icon={Info} title="Negation is only partially modelled">
                  SHAP attribution shows the toxic word <em>ahmaq</em> scoring +0.71 in an
                  unnegated sentence and +0.43 when explicitly negated, so reduced but still
                  positive.
                </Callout>
              </div>
            </div>
          </Reveal>
        )}

        {/* --------------------------------------------------------- LLMs */}
        {tab === "llm" && (
          <Reveal>
            <div className="grid gap-10 lg:grid-cols-[1.1fr_0.9fr]">
              <div>
                <h3 className="text-lg">Prompted LLMs against supervised fine-tuning</h3>
                <p className="mt-2 max-w-prose text-sm leading-relaxed text-forest-mid">
                  All models are evaluated on the same 2,868-sample test set. Outputs that could not
                  be parsed into a BIO sequence were treated as all-O predictions rather than
                  excluded, so every system is scored over the identical set.
                </p>
                <Table className="mt-5" caption="LLM baseline comparison">
                  <>
                    <Th>Model</Th>
                    <Th>Setting</Th>
                    <Th align="right">Parse fail</Th>
                    <Th align="right">F1</Th>
                  </>
                  {P2_LLM_BASELINES.map((m, i) => (
                    <Tr key={`${m.model}-${i}`} highlight={m.best}>
                      <Td strong className="text-xs sm:text-sm">{m.model}</Td>
                      <Td className="text-xs">{m.setting}</Td>
                      <Td align="right" className="font-mono text-xs text-forest-soft">
                        {m.parseFail === null ? "n/a" : `${m.parseFail}%`}
                      </Td>
                      <Td align="right" className="font-mono text-xs font-semibold">
                        {m.f1.toFixed(1)}
                      </Td>
                    </Tr>
                  ))}
                </Table>
                <Source>MUTEX-M Tables 13 and 14.</Source>
              </div>

              <div className="space-y-5">
                <div className="rounded-lg border border-sand-deep/60 bg-white/70 p-6">
                  <p className="eyebrow">Margin over the strongest prompted baseline</p>
                  <p className="mt-3 font-serif text-4xl text-merlot">+4.2</p>
                  <p className="mt-1 text-sm text-forest-mid">
                    F1 points, against GPT-4o with five-shot prompting.
                  </p>
                </div>
                <div className="rounded-lg border border-sand-deep/60 bg-white/70 p-6">
                  <p className="eyebrow">Margin over the Urdu-specialised model</p>
                  <p className="mt-3 font-serif text-4xl text-merlot">+5.1</p>
                  <p className="mt-1 text-sm text-forest-mid">
                    F1 points, against Qalb, a Llama-3.1 derivative with continued Urdu
                    pretraining.
                  </p>
                </div>
                <Callout icon={Info} title="Where prompting fails">
                  Parse failures are low across all models (1.8–8.7%). Over 91% of failures in every
                  model are valid-format but incorrectly labelled output, so the gap reflects label
                  quality rather than an inability to follow the format.
                </Callout>
              </div>
            </div>
          </Reveal>
        )}
      </div>

      <div className="mt-14 rounded-lg border border-sand-deep/60 bg-sand/25 p-6">
        <div className="flex flex-wrap items-center gap-2">
          <Pill tone="outline">Reading these numbers</Pill>
        </div>
        <p className="mt-3 max-w-3xl text-sm leading-relaxed text-forest-mid">
          Both papers evaluate with token-level F1 over the BIO label set. The SemEval-2021 English
          benchmark uses character-level F1 with exact offset matching, which the authors state is
          strictly harder and not directly comparable. MUTEX reports 57% when recomputed at
          character level against a SemEval baseline of 65.89%. Cross-language comparisons on this
          page are therefore presented as context, not as like-for-like rankings.
        </p>
        <Source>MUTEX §5.1 and §6.1.</Source>
      </div>
    </Section>
  );
}
