import React, { useState } from "react";
import { ShieldAlert, Users, Scale, CheckCheck, Type } from "lucide-react";
import {
  Section,
  Reveal,
  Source,
  Callout,
  Table,
  Th,
  Td,
  Tr,
  Pill,
  Timeline,
  CountUp,
  n,
} from "./ui";
import { AGREEMENT, PREPROCESSING } from "../data/research";
import stats from "../data/stats.json";

/**
 * Real records from URTOX_v2.csv, reproduced exactly as annotated. Glosses are
 * approximate and provided for readers who do not read Urdu. They are not part
 * of the dataset.
 */
const EXAMPLES = [
  {
    id: 48,
    sub: "insult",
    label: "toxic",
    caption: "Single-token span",
    tokens: ["آپ", "ایک", "مکمل", "بیوقوف", "ہیں", "۔"],
    spans: [[3, 4]],
    gloss: "You are a complete fool.",
    note: "One token is marked B-Toxic. The intensifier مکمل (complete) sits outside the span, a boundary decision the models frequently disagree with.",
  },
  {
    id: 140,
    sub: "offensive",
    label: "toxic",
    caption: "Multi-token span",
    tokens: ["یہ", "بے", "ہودہ", "امریکہ", "کے", "لئے", "شرمندگی", "ہے", "۔"],
    spans: [[1, 3]],
    gloss: "This indecent one is a disgrace for America.",
    note: "بے ہودہ is tokenised as two words, so the span opens with B-Toxic and continues with I-Toxic. This is exactly the label dependency the CRF layer exists to enforce.",
  },
  {
    id: 3494,
    sub: "offensive",
    label: "toxic",
    caption: "Two disjoint spans",
    tokens: [
      "جس", "کا", "جتنا", "زہن", "گندا", "ہوگا", "اُتنا", "ہی", "اِسکے", "متعلق", "بکواس", "کرے", "گا",
    ],
    spans: [
      [3, 5],
      [10, 12],
    ],
    gloss: "The dirtier a person's mind, the more nonsense they will speak about it.",
    note: "Two separate two-token spans in one record. Note also that mind is spelled زہن here rather than the standard ذہن, an authentic instance of the orthographic variation the papers describe.",
  },
  {
    id: 5299,
    sub: "normal",
    label: "non_toxic",
    caption: "No span",
    tokens: ["دل", "خون", "کے", "آنسو", "رو", "رہا", "ہے"],
    spans: [],
    gloss: "The heart is weeping tears of blood.",
    note: "Every token is tagged O. Non-toxic records are not empty rows. They carry a full O-tagged sequence, which is what the model learns the negative class from.",
  },
];

const STAGES = ["Raw text", "Tokenised", "BIO tagged", "Spans extracted"];

function tagFor(index, spans) {
  for (const [start, end] of spans) {
    if (index === start) return "B-Toxic";
    if (index > start && index < end) return "I-Toxic";
  }
  return "O";
}

const TAG_STYLE = {
  "B-Toxic": "border-merlot-bright/50 bg-merlot-wash text-merlot",
  "I-Toxic": "border-merlot-bright/30 bg-merlot-wash/60 text-merlot-mid",
  O: "border-sand-deep/60 bg-white text-forest-soft",
};

function Walkthrough() {
  const [active, setActive] = useState(0);
  const [stage, setStage] = useState(3);
  const ex = EXAMPLES[active];

  return (
    <div className="overflow-hidden rounded-lg border border-sand-deep/70 bg-white/75">
      <div className="flex flex-wrap gap-1 border-b border-sand-deep/60 p-2">
        {EXAMPLES.map((e, i) => (
          <button
            key={e.id}
            type="button"
            onClick={() => setActive(i)}
            aria-pressed={active === i}
            className={`rounded px-3 py-1.5 text-xs font-medium transition-all ${
              active === i
                ? "bg-forest text-cream"
                : "text-forest-mid hover:-translate-y-px hover:bg-sand/40"
            }`}
          >
            {e.caption}
          </button>
        ))}
      </div>

      <div className="p-5 sm:p-6">
        <div className="flex flex-wrap items-center gap-1.5">
          {STAGES.map((s, i) => (
            <React.Fragment key={s}>
              <button
                type="button"
                onClick={() => setStage(i)}
                aria-pressed={stage === i}
                className={`rounded-full px-3 py-1 text-[0.7rem] font-medium transition-all ${
                  stage >= i ? "bg-merlot-wash text-merlot-mid" : "bg-sand/40 text-forest-soft hover:bg-sand/70"
                } ${stage === i ? "ring-1 ring-merlot-bright/40" : ""}`}
              >
                {i + 1}. {s}
              </button>
              {i < STAGES.length - 1 && (
                <span aria-hidden="true" className="text-sand-deep">
                  ›
                </span>
              )}
            </React.Fragment>
          ))}
        </div>

        <div className="mt-5 flex flex-wrap items-center gap-2">
          <Pill tone="outline">id {ex.id}</Pill>
          <Pill tone={ex.label === "toxic" ? "merlot" : "sand"}>{ex.label}</Pill>
          <Pill tone="sand">{ex.sub}</Pill>
        </div>

        <div className="mt-5 rounded-md border border-sand-deep/50 bg-ivory p-5">
          <p className="urdu text-2xl text-forest-deep" lang="ur" dir="rtl">
            {ex.tokens.map((tok, i) => {
              const tag = tagFor(i, ex.spans);
              const marked = stage >= 2 && tag !== "O";
              return (
                <span key={i}>
                  <span
                    className={`transition-all duration-500 ${
                      marked
                        ? "rounded bg-merlot-bright/22 px-1 underline decoration-merlot-bright decoration-2 underline-offset-[7px]"
                        : stage === 1
                        ? "rounded bg-sand/50 px-1"
                        : ""
                    }`}
                  >
                    {tok}
                  </span>{" "}
                </span>
              );
            })}
          </p>
          <p className="mt-3 border-t border-sand-deep/40 pt-3 text-xs italic text-forest-soft">
            Approximate gloss: {ex.gloss}
          </p>
        </div>

        {stage >= 1 && (
          <div className="mt-4 overflow-x-auto">
            <div className="flex min-w-max gap-1.5" dir="rtl">
              {ex.tokens.map((tok, i) => {
                const tag = tagFor(i, ex.spans);
                return (
                  <div
                    key={i}
                    className="flex flex-col items-center gap-1"
                    style={{
                      animation: `fade-up 420ms cubic-bezier(0.16,1,0.3,1) both`,
                      animationDelay: `${i * 45}ms`,
                    }}
                  >
                    <div className="rounded border border-sand-deep/60 bg-white px-2.5 py-1.5">
                      <span className="urdu-inline text-base text-forest-deep" lang="ur">
                        {tok}
                      </span>
                    </div>
                    <span className="font-mono text-[0.6rem] text-forest-soft" dir="ltr">
                      {i}
                    </span>
                    {stage >= 2 && (
                      <span
                        dir="ltr"
                        className={`rounded border px-1.5 py-0.5 font-mono text-[0.6rem] ${TAG_STYLE[tag]}`}
                      >
                        {tag}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {stage >= 3 && (
          <div className="mt-5 animate-fade-up rounded-md border border-merlot-bright/30 bg-merlot-wash/40 p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-merlot-mid">
              Extracted spans
            </p>
            {ex.spans.length === 0 ? (
              <p className="mt-2 font-mono text-sm text-forest-mid">[ ] no toxic span</p>
            ) : (
              <ul className="mt-2.5 space-y-1.5">
                {ex.spans.map(([s, e], i) => (
                  <li key={i} className="flex flex-wrap items-center gap-3 text-sm">
                    <span className="font-mono text-xs text-forest-soft">
                      tokens [{s}, {e})
                    </span>
                    <span className="urdu-inline text-lg text-merlot" lang="ur">
                      {ex.tokens.slice(s, e).join(" ")}
                    </span>
                    <span className="font-mono text-[0.65rem] text-forest-soft">
                      {e - s} token{e - s > 1 ? "s" : ""}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        <p className="mt-4 text-xs leading-relaxed text-forest-soft">{ex.note}</p>
      </div>
    </div>
  );
}

export default function Annotation() {
  return (
    <Section
      id="annotation"
      index="03"
      tone="cream"
      eyebrow="Annotation methodology"
      title="How the spans were marked"
      lead="Annotation is the substance of this resource. The protocol, the label set, the agreement measures and the resulting inconsistencies are all set out below."
    >
      <Reveal>
        <Callout icon={ShieldAlert} tone="merlot" title="Content note">
          The examples in this section are real records from the dataset and contain insulting
          language in Urdu. They were selected to illustrate the annotation scheme, and the more
          severe material in the corpus is deliberately not reproduced here.
        </Callout>
      </Reveal>

      <div className="mt-12 grid gap-10 lg:grid-cols-[0.85fr_1.15fr] lg:gap-14">
        <Reveal from="left">
          <h3 className="text-lg">The protocol</h3>
          <dl className="mt-5 space-y-5">
            {[
              [
                "Annotation unit",
                "The word-level token. Sentences are tokenised first, and every token receives exactly one tag.",
              ],
              [
                "Label set",
                "B-TOXIC opens a toxic phrase, I-TOXIC continues one, and O falls outside any toxic span.",
              ],
              [
                "Span representation",
                "A span is a maximal contiguous run of B or I tagged tokens. Character offsets are derived from token positions when needed, not stored.",
              ],
              [
                "Sentence-level labels",
                "Each record also carries a binary toxic or non_toxic label and a category in sub_label, used for utterance-level classification.",
              ],
              ["Adjudication", AGREEMENT.adjudicated + "."],
            ].map(([term, def], i) => (
              <Reveal key={term} delay={i * 60}>
                <div className="border-l-2 border-sand-deep pl-4 transition-colors hover:border-merlot-mid">
                  <dt className="text-sm font-semibold text-forest-deep">{term}</dt>
                  <dd className="mt-1 text-sm leading-relaxed text-forest-mid">{def}</dd>
                </div>
              </Reveal>
            ))}
          </dl>
          <Source>MUTEX §3.3.</Source>

          <h3 className="mt-10 text-lg">Agreement</h3>
          <div className="mt-4 grid grid-cols-2 gap-4">
            {[
              [Users, "Cohen's κ", AGREEMENT.kappa],
              [Scale, "Krippendorff's α", AGREEMENT.alpha],
            ].map(([Icon, label, value]) => (
              <div
                key={label}
                className="rounded-lg border border-sand-deep/60 bg-white/70 p-5 transition-all duration-300 hover:-translate-y-0.5 hover:border-forest-soft/60"
              >
                <div className="flex items-center gap-2 text-forest-soft">
                  <Icon size={14} aria-hidden="true" />
                  <span className="text-xs font-medium uppercase tracking-wide">{label}</span>
                </div>
                <p className="mt-2 font-serif text-3xl text-forest-deep">
                  <CountUp value={value} decimals={2} />
                </p>
              </div>
            ))}
          </div>
          <p className="mt-3 text-xs leading-relaxed text-forest-soft">
            Both figures are reported by the authors in MUTEX §3.3. The papers do not state the
            number of annotators or their per-pair agreement, so those details cannot be presented
            here.
          </p>
        </Reveal>

        <Reveal from="right" delay={80}>
          <h3 className="text-lg">Walk through a real record</h3>
          <p className="mt-2 text-sm leading-relaxed text-forest-mid">
            Step through the stages of annotation on four records taken unmodified from the released
            CSV.
          </p>
          <div className="mt-5">
            <Walkthrough />
          </div>
        </Reveal>
      </div>

      <div className="mt-16 grid gap-8 lg:grid-cols-2">
        <Reveal from="left">
          <h3 className="text-lg">How many spans per record</h3>
          <Table className="mt-4" caption="Spans per record">
            <>
              <Th>Spans in record</Th>
              <Th align="right">Records</Th>
              <Th align="right">Share</Th>
            </>
            {stats.spansPerRecord.map((s) => (
              <Tr key={s.spans}>
                <Td strong>{s.plus ? "6 or more" : s.spans}</Td>
                <Td align="right" className="font-mono text-xs tabular-nums">
                  {n(s.count)}
                </Td>
                <Td align="right" className="font-mono text-xs tabular-nums">
                  {((s.count / stats.records) * 100).toFixed(1)}%
                </Td>
              </Tr>
            ))}
          </Table>
          <Source>
            Computed from URTOX_v2.csv by re-deriving spans from BIO_tags. The 0-span row includes
            all non-toxic records as well as the {n(stats.quality.toxicWithoutSpan)} toxic-labelled
            records that carry no span annotation.
          </Source>
        </Reveal>

        <Reveal from="right" delay={80}>
          <h3 className="text-lg">Preprocessing before annotation</h3>
          <p className="mt-2 text-sm leading-relaxed text-forest-mid">
            Applied in sequence, so that annotations and later model predictions stay aligned on the
            same token boundaries.
          </p>
          <div className="mt-6">
            <Timeline steps={PREPROCESSING} />
          </div>
          <Source>MUTEX §3.5.</Source>
        </Reveal>
      </div>

      <Reveal className="mt-14 grid gap-4 md:grid-cols-2">
        <Callout icon={CheckCheck} title="Deduplication and sampling">
          Duplicate removal used fuzzy string matching with a Levenshtein distance threshold below
          0.8, with stratified sampling of 20% per domain to keep the source balance (MUTEX §3.2).
          Removing deduplication from the pipeline changes token-level F1 by 0.2 points, the
          smallest effect of any preprocessing step measured.
        </Callout>
        <Callout icon={Type} title="Why word-level rather than character-level">
          The authors evaluate at token level because Urdu toxic expressions are generally word
          bounded, the BIO scheme operates naturally on token boundaries, and Urdu's morphological
          richness makes character granularity finer than the phenomenon being annotated.
        </Callout>
      </Reveal>
    </Section>
  );
}
