import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  FileText,
  AudioLines,
  Brain,
  Waves,
  Network,
  Sigma,
  GitMerge,
  ShieldCheck,
  Captions,
  Play,
  Pause,
  RotateCcw,
} from "lucide-react";
import { Section, Source, Pill, useInView } from "./ui";

/**
 * MUTEX-M architecture in a fixed SVG coordinate space so it scales cleanly
 * from phone to desktop. Signal packets travel the edges continuously; the
 * stage walkthrough lights each module in turn and advances a worked example
 * underneath. Node copy comes from MUTEX-M §4.
 */

const W = 1060;
const H = 620;

const NODES = {
  "text-in": {
    lane: "text",
    x: 24, y: 108, w: 150, h: 92,
    icon: FileText,
    title: "Text input",
    sub: "X = {x₁ … x_T}",
    detail:
      "A tokenised Urdu sequence. When the system is driven from audio, this arrives from Whisper instead of from a paired dataset record.",
    facts: [["Max sequence length", "128 tokens"]],
  },
  xlmr: {
    lane: "text",
    x: 212, y: 108, w: 180, h: 92,
    icon: Brain,
    title: "XLM-RoBERTa",
    sub: "token classification",
    detail:
      "Encodes the sequence and projects 768-dimensional representations to logits over B-Toxic, I-Toxic and O. MUTEX-M propagates each word's label to all of its subword pieces. The original implementation assigned continuation subwords the ignore index, which silently discarded most of the gradient signal.",
    facts: [
      ["Checkpoint", "xlm-roberta-base"],
      ["Subword propagation", "+4.0 F1"],
    ],
  },
  crf: {
    lane: "text",
    x: 430, y: 108, w: 168, h: 92,
    icon: Network,
    title: "CRF decoder",
    sub: "Viterbi, valid BIO",
    detail:
      "Models transition scores between adjacent labels so decoding can only produce a well-formed BIO sequence. In the MUTEX baseline this removed every invalid transition, which occurred in 8.3% of predictions without it.",
    facts: [
      ["Invalid BIO", "8.3% to 0.0%"],
      ["Contribution", "+1.3 F1"],
    ],
  },
  "p-text": {
    lane: "text",
    x: 636, y: 116, w: 120, h: 76,
    icon: Sigma,
    title: "p_text",
    sub: "max B-Toxic prob.",
    detail:
      "Token predictions are aggregated into one utterance-level score by taking the maximum B-Toxic softmax probability across the sequence. This is the value that enters fusion.",
    facts: [["Range", "[0, 1]"]],
  },
  "audio-in": {
    lane: "audio",
    x: 24, y: 396, w: 150, h: 92,
    icon: AudioLines,
    title: "Audio input",
    sub: "16 kHz mono",
    detail:
      "Each clip is resampled to 16 kHz mono and truncated to ten seconds. Fewer than 3% of clips exceed that length.",
    facts: [["Truncation", "10 s"]],
  },
  mms: {
    lane: "audio",
    x: 212, y: 396, w: 180, h: 92,
    icon: Waves,
    title: "MMS-300M",
    sub: "frozen · urd adapter",
    detail:
      "A wav2vec 2.0 derivative pretrained on over 1,400 languages including Urdu. The encoder stays frozen because the dataset is small relative to its pretraining corpus, and fine-tuning on single-voice synthesised speech would bias it toward clean TTS audio.",
    facts: [
      ["Checkpoint", "facebook/mms-300m"],
      ["Language adapter", "urd"],
    ],
  },
  mlp: {
    lane: "audio",
    x: 430, y: 396, w: 168, h: 92,
    icon: Sigma,
    title: "Mean-pool + MLP",
    sub: "768 → 256 → 64 → 2",
    detail:
      "Frame-level states are mean-pooled into a fixed utterance embedding, then classified by a small MLP. Mean pooling is preferred over attention pooling because it adds no learnable parameters to overfit.",
    facts: [
      ["Dropout", "0.3 / 0.2"],
      ["Phase 2 training", "under 5 min"],
    ],
  },
  "p-audio": {
    lane: "audio",
    x: 636, y: 404, w: 120, h: 76,
    icon: Sigma,
    title: "p_audio",
    sub: "softmax toxic",
    detail:
      "The utterance-level acoustic toxicity probability, produced independently of the text module. That independence is what lets fusion tolerate a degraded audio signal without corrupting the text prediction.",
    facts: [["Weighted F1 (TTS)", "79.0%"]],
  },
  whisper: {
    lane: "bridge",
    x: 212, y: 252, w: 180, h: 82,
    icon: Captions,
    title: "Whisper ASR",
    sub: "lang = ur",
    detail:
      "Used only when there is no paired transcript, for example a raw WhatsApp voice message. Transcription errors propagate into the text module and cannot be recovered downstream, which the paper identifies as the primary real-world failure mode.",
    facts: [
      ["WER, read speech", "33.68%"],
      ["WER, conversational", "38 to 45%"],
    ],
  },
  fusion: {
    lane: "fuse",
    x: 806, y: 200, w: 216, h: 108,
    icon: GitMerge,
    title: "Weighted late fusion",
    sub: "α·p_text + (1−α)·p_audio",
    detail:
      "Equal weighting at α = 0.5 was chosen by grid search on the validation set, so the reported test figure carries no optimistic bias. The flat peak at 0.5 says the two modalities carry signal of comparable reliability.",
    facts: [
      ["α", "0.5"],
      ["Weighted F1 (TTS)", "83.2%"],
      ["Weighted F1 (real)", "77.1%"],
    ],
  },
  out: {
    lane: "fuse",
    x: 806, y: 388, w: 216, h: 108,
    icon: ShieldCheck,
    title: "Verdict + span set",
    sub: "ŷ, S",
    detail:
      "If the fused score clears 0.5 the record is toxic and the text module's spans are returned, otherwise the span set is empty. Span localisation is performed exclusively by the text module. The audio module acts as an utterance-level gate.",
    facts: [["Span-level ceiling", "the text model's 67% F1"]],
  },
};

const EDGES = [
  { from: "text-in", to: "xlmr" },
  { from: "xlmr", to: "crf" },
  { from: "crf", to: "p-text" },
  { from: "audio-in", to: "mms" },
  { from: "mms", to: "mlp" },
  { from: "mlp", to: "p-audio" },
  { from: "p-text", to: "fusion", curve: true },
  { from: "p-audio", to: "fusion", curve: true },
  { from: "fusion", to: "out", drop: true },
  { from: "audio-in", to: "whisper", bridge: true },
  { from: "whisper", to: "xlmr", bridge: true },
];

/**
 * Worked example carried through the stages, taken from the WhatsApp voice
 * message qualitative test in MUTEX-M Table 6, example 4.
 */
const STAGES = [
  {
    lit: ["text-in", "audio-in"],
    label: "Input",
    caption: "A voice message arrives. Text and audio enter their own modules.",
    readout: [["p_text", null], ["p_audio", null], ["fused", null]],
  },
  {
    lit: ["whisper"],
    label: "Transcribe",
    caption: "With no paired transcript, Whisper produces one before the text module can run.",
    readout: [["p_text", null], ["p_audio", null], ["fused", null]],
  },
  {
    lit: ["xlmr", "mms"],
    label: "Encode",
    caption: "Both encoders run independently. Neither sees the other's representation.",
    readout: [["p_text", null], ["p_audio", null], ["fused", null]],
  },
  {
    lit: ["crf", "mlp"],
    label: "Decode",
    caption: "The CRF forces a valid BIO sequence. The MLP scores the pooled acoustic embedding.",
    readout: [["p_text", null], ["p_audio", null], ["fused", null]],
  },
  {
    lit: ["p-text", "p-audio"],
    label: "Score",
    caption: "Two independent probabilities, one per modality.",
    readout: [["p_text", 0.976], ["p_audio", 0.809], ["fused", null]],
  },
  {
    lit: ["fusion"],
    label: "Fuse",
    caption: "Equal weighting averages the two scores.",
    readout: [["p_text", 0.976], ["p_audio", 0.809], ["fused", 0.893]],
  },
  {
    lit: ["out"],
    label: "Decide",
    caption: "Above 0.5, so the verdict is toxic and the text module's spans are returned.",
    readout: [["p_text", 0.976], ["p_audio", 0.809], ["fused", 0.893]],
    spans: ["bewaqoof", "moun band karwa donga"],
  },
];

const LANE = {
  text: { fill: "#FFFFFF", stroke: "#D6C7AE", icon: "#3A4E41" },
  audio: { fill: "#FFFFFF", stroke: "#D6C7AE", icon: "#6E1B1F" },
  bridge: { fill: "#F4EEE2", stroke: "#D6C7AE", icon: "#5C7264" },
  fuse: { fill: "#26342B", stroke: "#26342B", icon: "#E5D9C6" },
};

function pathFor(e) {
  const a = NODES[e.from];
  const b = NODES[e.to];

  if (e.bridge) {
    const x1 = a.x + a.w / 2;
    const y1 = a.y;
    const x2 = b.x + b.w / 2;
    const y2 = b.y + b.h;
    return `M ${x1} ${y1} C ${x1} ${y1 - 34}, ${x2} ${y2 + 34}, ${x2} ${y2}`;
  }
  if (e.drop) {
    const cx = a.x + a.w / 2;
    return `M ${cx} ${a.y + a.h} L ${cx} ${b.y}`;
  }

  const x1 = a.x + a.w;
  const y1 = a.y + a.h / 2;
  const x2 = b.x;
  const y2 = b.y + b.h / 2;

  if (e.curve) {
    const mid = x1 + (x2 - x1) / 2;
    return `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`;
  }
  return `M ${x1} ${y1} L ${x2} ${y2}`;
}

export default function Architecture() {
  const [selected, setSelected] = useState("fusion");
  const [stage, setStage] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [containerRef, inView] = useInView({ threshold: 0.25 });
  const started = useRef(false);

  // Start the walkthrough by itself the first time the diagram is reached.
  useEffect(() => {
    if (inView && !started.current) {
      started.current = true;
      setPlaying(true);
    }
  }, [inView]);

  useEffect(() => {
    if (!playing) return undefined;
    const timer = setInterval(() => {
      setStage((s) => {
        if (s >= STAGES.length - 1) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, 1250);
    return () => clearInterval(timer);
  }, [playing]);

  const current = STAGES[stage];
  const lit = useMemo(() => new Set(current.lit), [current]);
  const node = NODES[selected];
  const Icon = node.icon;

  // an edge is active when the stage has just lit its destination
  const edgeActive = (e) => lit.has(e.to) || lit.has(e.from);

  const restart = () => {
    setStage(0);
    setPlaying(true);
  };

  return (
    <Section
      id="method"
      index="06"
      eyebrow="System architecture"
      title="How MUTEX-M is put together"
      lead="Two modules that never see each other's representations, producing two probabilities that a weighted average combines. Watch a real voice message move through it, or select any component to see what it does."
    >
      {/* controls */}
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={() => (stage >= STAGES.length - 1 ? restart() : setPlaying((p) => !p))}
          className={`inline-flex items-center gap-2 rounded-md px-3.5 py-2 text-sm font-medium transition-all hover:-translate-y-px ${
            playing
              ? "bg-merlot text-cream hover:bg-merlot-mid"
              : "border border-sand-deep bg-white text-forest hover:border-forest-soft"
          }`}
        >
          {stage >= STAGES.length - 1 ? (
            <RotateCcw size={14} aria-hidden="true" />
          ) : playing ? (
            <Pause size={14} aria-hidden="true" />
          ) : (
            <Play size={14} aria-hidden="true" />
          )}
          {stage >= STAGES.length - 1 ? "Replay" : playing ? "Pause" : "Play walkthrough"}
        </button>

        {/* stage chips double as a scrubber */}
        <div className="flex flex-wrap gap-1">
          {STAGES.map((s, i) => (
            <button
              key={s.label}
              type="button"
              onClick={() => {
                setPlaying(false);
                setStage(i);
              }}
              aria-pressed={stage === i}
              className={`rounded-full px-2.5 py-1 text-[0.7rem] font-medium transition-all ${
                i === stage
                  ? "bg-forest text-cream"
                  : i < stage
                  ? "bg-merlot-wash text-merlot-mid"
                  : "bg-sand/40 text-forest-soft hover:bg-sand/70"
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      {/* diagram */}
      <div
        ref={containerRef}
        className="mt-6 overflow-x-auto rounded-lg border border-sand-deep/70 bg-gradient-to-br from-cream via-ivory to-cream p-3 sm:p-5"
      >
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="h-auto w-full min-w-[680px]"
          role="img"
          aria-label="MUTEX-M architecture: a text module and an audio module feeding weighted late fusion"
        >
          <defs>
            <marker
              id="arrowhead" viewBox="0 0 10 10" refX="9" refY="5"
              markerWidth="5" markerHeight="5" orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#A8B5AC" />
            </marker>
            <marker
              id="arrowhead-live" viewBox="0 0 10 10" refX="9" refY="5"
              markerWidth="5" markerHeight="5" orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#6E1B1F" />
            </marker>
            <filter id="lift" x="-40%" y="-40%" width="180%" height="180%">
              <feDropShadow dx="0" dy="3" stdDeviation="4" floodColor="#26342B" floodOpacity="0.12" />
            </filter>
            <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
              <feGaussianBlur stdDeviation="6" result="b" />
              <feMerge>
                <feMergeNode in="b" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* lane backdrops */}
          <rect x="12" y="92" width="756" height="124" rx="12" fill="#26342B" opacity="0.035" />
          <rect x="12" y="380" width="756" height="124" rx="12" fill="#6E1B1F" opacity="0.035" />
          <rect x="794" y="184" width="240" height="328" rx="12" fill="#26342B" opacity="0.03" />

          <text x="24" y="82" fill="#5C7264" style={{ fontSize: 10.5, letterSpacing: "0.18em" }}>
            TEXT MODULE
          </text>
          <text x="24" y="370" fill="#5C7264" style={{ fontSize: 10.5, letterSpacing: "0.18em" }}>
            AUDIO MODULE
          </text>
          <text x="806" y="174" fill="#5C7264" style={{ fontSize: 10.5, letterSpacing: "0.18em" }}>
            FUSION
          </text>

          {/* edges */}
          {EDGES.map((e) => {
            const d = pathFor(e);
            const live = edgeActive(e);
            return (
              <g key={`${e.from}-${e.to}`}>
                <path
                  d={d}
                  fill="none"
                  stroke={e.bridge ? "#8FA396" : live ? "#6E1B1F" : "#BCC7BE"}
                  strokeWidth={live ? 2.2 : 1.5}
                  strokeDasharray={e.bridge ? "5 4" : undefined}
                  markerEnd={live ? "url(#arrowhead-live)" : "url(#arrowhead)"}
                  opacity={e.bridge ? 0.55 : 1}
                  style={{ transition: "stroke 400ms ease, stroke-width 400ms ease" }}
                />
                {/* a packet of signal riding the edge */}
                {!e.bridge && (
                  <circle r={live ? 4 : 2.6} fill={live ? "#8F3438" : "#9BAA9E"} opacity={live ? 1 : 0.5}>
                    <animateMotion
                      dur={live ? "1.5s" : "3.4s"}
                      repeatCount="indefinite"
                      path={d}
                      keyPoints="0;1"
                      keyTimes="0;1"
                      calcMode="linear"
                    />
                  </circle>
                )}
              </g>
            );
          })}

          {/* nodes */}
          {Object.entries(NODES).map(([id, nd]) => {
            const c = LANE[nd.lane];
            const isSel = selected === id;
            const isLit = lit.has(id);
            const NIcon = nd.icon;
            const dark = nd.lane === "fuse";

            return (
              <g
                key={id}
                role="button"
                tabIndex={0}
                aria-label={`${nd.title}, ${nd.sub}`}
                aria-pressed={isSel}
                onClick={() => setSelected(id)}
                onKeyDown={(ev) => {
                  if (ev.key === "Enter" || ev.key === " ") {
                    ev.preventDefault();
                    setSelected(id);
                  }
                }}
                style={{ cursor: "pointer" }}
                className="group"
              >
                {isLit && (
                  <rect
                    x={nd.x - 8} y={nd.y - 8}
                    width={nd.w + 16} height={nd.h + 16}
                    rx="16"
                    fill="#8F3438"
                    opacity="0.16"
                    filter="url(#glow)"
                  />
                )}
                {isSel && !isLit && (
                  <rect
                    x={nd.x - 5} y={nd.y - 5}
                    width={nd.w + 10} height={nd.h + 10}
                    rx="14"
                    fill="none"
                    stroke="#6E1B1F"
                    strokeWidth="1.6"
                    opacity="0.5"
                  />
                )}

                <rect
                  x={nd.x} y={nd.y} width={nd.w} height={nd.h}
                  rx="10"
                  fill={c.fill}
                  stroke={isLit ? "#8F3438" : isSel ? "#6E1B1F" : c.stroke}
                  strokeWidth={isLit || isSel ? 1.8 : 1.1}
                  filter="url(#lift)"
                  style={{ transition: "stroke 350ms ease, stroke-width 350ms ease" }}
                  className="group-hover:brightness-[0.99]"
                />

                <g
                  transform={`translate(${nd.x + 15}, ${nd.y + 15})`}
                  style={{ color: isLit ? "#8F3438" : c.icon, transition: "color 350ms ease" }}
                >
                  <NIcon size={18} aria-hidden="true" />
                </g>

                <text
                  x={nd.x + 15} y={nd.y + 60}
                  fill={dark ? "#F4EEE2" : "#18221C"}
                  style={{ fontSize: 14.5, fontFamily: "Fraunces, Georgia, serif", fontWeight: 500 }}
                >
                  {nd.title}
                </text>
                <text
                  x={nd.x + 15} y={nd.y + 77}
                  fill={dark ? "#E5D9C6" : "#5C7264"}
                  style={{ fontSize: 10, fontFamily: "JetBrains Mono, monospace" }}
                >
                  {nd.sub}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* live readout for the worked example */}
      <div className="mt-4 grid gap-4 lg:grid-cols-[1.4fr_1fr]">
        <div className="rounded-lg border border-sand-deep/70 bg-white/70 p-5">
          <div className="flex items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-merlot text-[0.68rem] font-medium text-cream">
              {stage + 1}
            </span>
            <p className="text-sm font-semibold text-forest-deep">{current.label}</p>
          </div>
          <p className="mt-2.5 text-sm leading-relaxed text-forest-mid">{current.caption}</p>

          <div className="mt-4 rounded-md border border-sand-deep/50 bg-ivory p-4">
            <p className="text-[0.68rem] uppercase tracking-wide text-forest-soft">
              Worked example, real WhatsApp voice message
            </p>
            <p className="mt-2 font-mono text-[0.8rem] leading-relaxed text-forest-deep">
              {["bewaqoof", "band kar yeh bakwaas warna", "moun band karwa donga"].map((chunk, i) => {
                const isSpan = current.spans && (i === 0 || i === 2);
                return (
                  <span
                    key={i}
                    className={
                      isSpan
                        ? "rounded bg-merlot-bright/22 px-1 underline decoration-merlot-bright decoration-2 underline-offset-4 transition-colors"
                        : "transition-colors"
                    }
                  >
                    {chunk}{" "}
                  </span>
                );
              })}
            </p>
            <p className="mt-2 text-xs italic text-forest-soft">
              Roman transliteration. Gloss: stop this nonsense or I will shut your mouth.
            </p>
          </div>
        </div>

        <div className="rounded-lg border border-sand-deep/70 bg-forest p-5">
          <p className="text-[0.68rem] uppercase tracking-wide text-sand-deep">Live scores</p>
          <dl className="mt-4 space-y-3">
            {current.readout.map(([k, v]) => (
              <div key={k}>
                <div className="flex items-baseline justify-between gap-3">
                  <dt className="font-mono text-xs text-sand/70">{k}</dt>
                  <dd className="font-mono text-sm tabular-nums text-cream">
                    {v === null ? <span className="text-sand/30">pending</span> : v.toFixed(3)}
                  </dd>
                </div>
                <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-white/10">
                  <div
                    className="h-full rounded-full bg-merlot-bright transition-[width] duration-700 ease-[cubic-bezier(0.16,1,0.3,1)]"
                    style={{ width: v === null ? "0%" : `${v * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </dl>
          <div className="mt-4 border-t border-sand/15 pt-3">
            <p className="text-xs text-sand/60">Verdict</p>
            <p className="mt-1 font-serif text-xl text-cream">
              {current.spans ? "Toxic, 89.3%" : <span className="text-sand/30">awaiting fusion</span>}
            </p>
          </div>
          <p className="mt-3 text-[0.65rem] leading-relaxed text-sand/40">
            MUTEX-M Table 6, example 4. No retraining or domain adaptation was applied.
          </p>
        </div>
      </div>

      {/* detail panel */}
      <div className="mt-6 grid gap-6 rounded-lg border border-sand-deep/70 bg-white/70 p-6 md:grid-cols-[1fr_auto]">
        <div>
          <div className="flex items-center gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-md bg-sand/50 text-forest-mid">
              <Icon size={17} aria-hidden="true" />
            </span>
            <div>
              <h3 className="text-lg leading-tight">{node.title}</h3>
              <p className="font-mono text-xs text-forest-soft">{node.sub}</p>
            </div>
          </div>
          <p className="mt-4 max-w-prose text-sm leading-relaxed text-forest-mid">{node.detail}</p>

          <div className="mt-5 flex flex-wrap items-center gap-1.5">
            {Object.entries(NODES).map(([id, nd]) => (
              <button
                key={id}
                type="button"
                onClick={() => setSelected(id)}
                className={`rounded-full border px-2.5 py-1 text-[0.7rem] transition-all ${
                  selected === id
                    ? "border-merlot-bright/40 bg-merlot-wash text-merlot-mid"
                    : "border-sand-deep bg-white/60 text-forest-mid hover:-translate-y-px hover:border-forest-soft"
                }`}
              >
                {nd.title}
              </button>
            ))}
          </div>
        </div>

        <dl className="flex shrink-0 flex-col gap-3 md:w-60">
          {node.facts.map(([k, v]) => (
            <div key={k} className="rounded-md border border-sand-deep/50 bg-ivory px-4 py-3">
              <dt className="text-[0.66rem] uppercase tracking-wide text-forest-soft">{k}</dt>
              <dd className="mt-0.5 font-mono text-sm text-forest-deep">{v}</dd>
            </div>
          ))}
        </dl>
      </div>

      <div className="mt-6 flex flex-wrap gap-2">
        <Pill tone="forest">Single NVIDIA RTX A6000, 48 GB</Pill>
        <Pill tone="forest">PyTorch 2.1 · Transformers 4.38</Pill>
        <Pill tone="forest">Seed 42 across PyTorch, NumPy and the Trainer</Pill>
      </div>

      <Source>
        MUTEX-M §4 and Figure 2. Component figures from §4.2 to §4.4, §5 and Table 8. Worked example
        from Table 6.
      </Source>
    </Section>
  );
}
