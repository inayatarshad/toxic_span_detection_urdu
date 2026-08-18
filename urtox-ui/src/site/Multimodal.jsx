import React from "react";
import { ArrowDown, Mic, FileText, Layers, Info, AlertTriangle } from "lucide-react";
import {
  Section,
  Reveal,
  Source,
  Callout,
  Table,
  Th,
  Td,
  Tr,
  MetricBars,
  Donut,
  Pill,
  n,
} from "./ui";
import {
  URTOX_MM,
  HUMAN_AUDIO,
  P2_AUDIO,
  P2_FUSION,
  P2_FUSION_BASELINES,
  P2_REAL_SPEECH,
  P2_ACCENTS,
  LINKS,
} from "../data/research";

const STAGES = [
  {
    icon: FileText,
    tag: "MUTEX · 2026",
    title: "Urdu toxic span detection",
    body: "URTOX is annotated at the token level and an XLM-RoBERTa + CRF sequence labeller establishes a supervised baseline of 60.0% token-level F1. The scope is text only.",
  },
  {
    icon: Layers,
    tag: "MUTEX-M · text module",
    title: "Improved span model",
    body: "A subword label-assignment artifact is identified and corrected, raising token-level F1 to 67.0% with no architectural change. Four of those seven points come from the correction alone.",
  },
  {
    icon: Mic,
    tag: "MUTEX-M · multimodal",
    title: "Text + audio fusion",
    body: "A speech dataset is synthesised from the existing annotations, an utterance-level acoustic classifier is trained on it, and the two probabilities are combined by a weighted average.",
  },
];

export default function Multimodal() {
  return (
    <Section
      id="multimodal"
      index="08"
      eyebrow="From text to multimodal"
      title="How the research progressed"
      lead="The second paper does not replace the first. It corrects the text model, then adds an acoustic modality on top of the same annotated corpus."
    >
      {/* progression */}
      <ol className="grid gap-4 md:grid-cols-3">
        {STAGES.map((s, i) => {
          const Icon = s.icon;
          return (
            <Reveal key={s.title} delay={i * 90}>
              <li className="relative h-full rounded-lg border border-sand-deep/60 bg-white/70 p-6">
                <div className="flex items-center gap-3">
                  <span className="flex h-9 w-9 items-center justify-center rounded-md bg-forest text-cream">
                    <Icon size={17} aria-hidden="true" />
                  </span>
                  <span className="font-mono text-[0.68rem] uppercase tracking-wide text-forest-soft">
                    {s.tag}
                  </span>
                </div>
                <h3 className="mt-4 text-base">{s.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-forest-mid">{s.body}</p>
                {i < STAGES.length - 1 && (
                  <ArrowDown
                    size={18}
                    aria-hidden="true"
                    className="absolute -bottom-3 left-1/2 -translate-x-1/2 rotate-0 text-sand-deep md:-right-3 md:bottom-1/2 md:left-auto md:translate-x-0 md:translate-y-1/2 md:-rotate-90"
                  />
                )}
              </li>
            </Reveal>
          );
        })}
      </ol>

      {/* honesty note on TTS */}
      <Reveal className="mt-12">
        <Callout icon={AlertTriangle} tone="merlot" title="The audio is synthesised, not recorded">
          URTOX-MM was produced by running every URTOX text record through Microsoft Edge TTS with a
          single neural voice. It is not a corpus of recorded toxic speech. The authors chose this
          because no real Urdu toxic speech corpus exists, collecting one raises consent concerns,
          and synthesis preserves the token-level annotations exactly. The consequence is an
          acoustic domain gap, which they measure rather than set aside. See the real-speech
          results below.
        </Callout>
      </Reveal>

      {/* the two audio datasets */}
      <Reveal className="mt-12 grid gap-8 lg:grid-cols-2">
        <div className="card p-6">
          <div className="flex items-center justify-between gap-3">
            <h3 className="text-lg">URTOX-MM</h3>
            <Pill tone="sand">Synthesised · public</Pill>
          </div>
          <p className="mt-2 text-sm text-forest-mid">
            A paired text–audio dataset generated from the URTOX annotations.
          </p>
          <dl className="mt-5 space-y-2.5 text-sm">
            {[
              ["Total clips", n(URTOX_MM.clips)],
              ["Excluded at preprocessing", String(URTOX_MM.excluded)],
              ["Voice", URTOX_MM.voice],
              ["Format", URTOX_MM.format],
              ["File size", URTOX_MM.fileSize],
              ["Duration", URTOX_MM.duration],
              ["Balance", URTOX_MM.balance],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between gap-4 border-b border-sand-deep/40 pb-2 last:border-0">
                <dt className="shrink-0 text-forest-soft">{k}</dt>
                <dd className="text-right font-mono text-xs text-forest-deep">{v}</dd>
              </div>
            ))}
          </dl>
          <a
            href={LINKS.audioDataset}
            target="_blank"
            rel="noreferrer"
            className="mt-4 inline-block text-sm font-medium text-merlot-mid hover:underline"
          >
            View URTOX-MM on Hugging Face →
          </a>
          <Source>MUTEX-M §3.2 and Table 2.</Source>
        </div>

        <div className="card p-6">
          <div className="flex items-center justify-between gap-3">
            <h3 className="text-lg">URTOX-HumanAudio</h3>
            <Pill tone="merlot">Real speech · on request</Pill>
          </div>
          <p className="mt-2 text-sm text-forest-mid">
            A real conversational evaluation set, used for testing only, never for training.
          </p>

          <div className="mt-5">
            <Donut
              size={150}
              thickness={22}
              centerValue={n(HUMAN_AUDIO.clips)}
              centerLabel="CLIPS"
              data={HUMAN_AUDIO.accents.map((a, i) => ({
                label: a.name,
                value: a.share,
                color: ["#26342B", "#6E1B1F", "#8F3438", "#D6C7AE"][i],
              }))}
            />
          </div>

          <dl className="mt-5 space-y-2.5 text-sm">
            {[
              ["Toxic / non-toxic", `${n(HUMAN_AUDIO.toxic)} / ${n(HUMAN_AUDIO.nonToxic)}`],
              ["Average duration", HUMAN_AUDIO.duration],
              ["Agreement (Cohen's κ)", String(HUMAN_AUDIO.kappa)],
              ["Annotators", "Three native Urdu speakers, majority vote"],
              ["Availability", HUMAN_AUDIO.availability],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between gap-4 border-b border-sand-deep/40 pb-2 last:border-0">
                <dt className="shrink-0 text-forest-soft">{k}</dt>
                <dd className="text-right text-xs text-forest-deep">{v}</dd>
              </div>
            ))}
          </dl>
          <Source>MUTEX-M §3.5 and Table 4.</Source>
        </div>
      </Reveal>

      {/* audio classifier + fusion */}
      <Reveal className="mt-14 grid gap-10 lg:grid-cols-2">
        <div>
          <h3 className="text-lg">Audio classifier</h3>
          <p className="mt-2 text-sm leading-relaxed text-forest-mid">
            Utterance-level results on the 2,868-sample test set. The comparison is between an
            English-pretrained encoder and one whose pretraining includes Urdu.
          </p>
          {P2_AUDIO.map((m) => (
            <div key={m.model} className="mt-5">
              <div className="flex items-center gap-2">
                <h4 className="text-sm font-semibold text-forest-deep">{m.model}</h4>
                <Pill tone={m.best ? "merlot" : "outline"}>{m.pretraining}</Pill>
              </div>
              <Table className="mt-2" caption={`${m.model} classification report`}>
                <>
                  <Th>Class</Th>
                  <Th align="right">P</Th>
                  <Th align="right">R</Th>
                  <Th align="right">F1</Th>
                  <Th align="right">Support</Th>
                </>
                {m.rows.map((r) => (
                  <Tr key={r.cls} highlight={r.total && m.best}>
                    <Td strong={r.total}>{r.cls}</Td>
                    <Td align="right" className="font-mono text-xs">{r.precision.toFixed(2)}</Td>
                    <Td align="right" className="font-mono text-xs">{r.recall.toFixed(2)}</Td>
                    <Td align="right" className="font-mono text-xs">{r.f1.toFixed(2)}</Td>
                    <Td align="right" className="font-mono text-xs text-forest-soft">
                      {n(r.support)}
                    </Td>
                  </Tr>
                ))}
              </Table>
            </div>
          ))}
          <Source>MUTEX-M Table 10.</Source>
        </div>

        <div>
          <h3 className="text-lg">Fusion weight</h3>
          <p className="mt-2 text-sm leading-relaxed text-forest-mid">
            α weights the text module and (1−α) the audio module. The weight was chosen on the
            validation set, so the test figure is not tuned on test data.
          </p>
          <div className="mt-6">
            <MetricBars
              domainMin={60}
              domainMax={88}
              data={[
                { label: "Text only", value: P2_FUSION_BASELINES.textOnly },
                { label: "Audio only", value: P2_FUSION_BASELINES.audioOnly },
                ...P2_FUSION.map((f) => ({
                  label: `Fusion α = ${f.alpha}`,
                  value: f.f1,
                  best: f.best,
                })),
              ]}
            />
          </div>
          <Source>MUTEX-M Table 11. All values are weighted F1 on the synthesised test set.</Source>

          <div className="mt-8">
            <Callout icon={Info} title="No fusion setting falls below audio-only">
              Every weight in the grid scores at or above the 79.0% audio-only baseline, which the
              authors read as evidence that the two modalities carry complementary rather than
              redundant signal.
            </Callout>
          </div>
        </div>
      </Reveal>

      {/* real speech */}
      <Reveal className="mt-14">
        <h3 className="text-lg">Holding up on real speech</h3>
        <p className="mt-2 max-w-prose text-sm leading-relaxed text-forest-mid">
          The most informative result in the second paper. Every system is evaluated on 2,000 real
          conversational clips using the configuration fixed on synthesised data, with no adaptation
          to the real-speech distribution.
        </p>

        <div className="mt-6 grid gap-8 lg:grid-cols-[1.15fr_0.85fr]">
          <Table caption="Performance on real conversational speech">
            <>
              <Th>System</Th>
              <Th align="right">P</Th>
              <Th align="right">R</Th>
              <Th align="right">Weighted F1</Th>
              <Th align="right">Δ vs TTS</Th>
            </>
            {P2_REAL_SPEECH.map((r) => (
              <Tr key={r.setup} highlight={r.best}>
                <Td strong className="text-xs sm:text-sm">{r.setup}</Td>
                <Td align="right" className="font-mono text-xs">{r.precision.toFixed(2)}</Td>
                <Td align="right" className="font-mono text-xs">{r.recall.toFixed(2)}</Td>
                <Td align="right" className="font-mono text-xs font-semibold">
                  {(r.f1 * 100).toFixed(1)}%
                </Td>
                <Td align="right" className="font-mono text-xs text-merlot-mid">
                  {r.drop} pp
                </Td>
              </Tr>
            ))}
          </Table>

          <div>
            <h4 className="text-sm font-semibold text-forest-deep">By regional accent</h4>
            <div className="mt-4">
              <MetricBars
                domainMin={65}
                domainMax={85}
                data={P2_ACCENTS.map((a, i) => ({
                  label: a.accent,
                  value: a.f1 * 100,
                  best: i === 0,
                  note: `${n(a.clips)} clips`,
                }))}
              />
            </div>
            <Source>MUTEX-M Tables 15 and 16.</Source>
          </div>
        </div>

        <div className="mt-8 max-w-3xl">
          <Callout icon={Info} title="Why the encoder choice matters">
            The English-pretrained wav2vec 2.0 loses 9.0 points moving from synthesised to real
            speech; MMS-300M, whose pretraining includes Urdu, loses 6.5. The authors present this
            as the clearest evidence that the encoder choice reflects robustness rather than
            benchmark optimisation. They also characterise the whole system as a first baseline
            rather than production-ready, and note that the real-speech set skews toward intelligible
            audio, making 6.5 points a lower bound.
          </Callout>
        </div>
      </Reveal>
    </Section>
  );
}
