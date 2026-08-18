import React from "react";
import { CheckCircle2, Cloud, Mail, Terminal, Rocket } from "lucide-react";
import { Section, Reveal, Callout, Source, GithubMark } from "./ui";
import { LINKS, HYPERPARAMS, TRAINING_COST } from "../data/research";

const ARTEFACTS = [
  { name: "URTOX_v2.csv", what: "The annotated text dataset", status: "available",
    note: "In the repository and on Hugging Face" },
  { name: "urdu_toxic_audio_dataset.csv", what: "Text records paired with audio paths and transcripts",
    status: "available", note: "In the repository" },
  { name: "requirements-training.txt", what: "Pinned training environment", status: "available",
    note: "Versions match the configuration reported in MUTEX-M section 5" },
  { name: "urdu_toxic_span_detection.ipynb", what: "Text model training", status: "available",
    note: "Runs end to end in Google Colab" },
  { name: "URTOX_XLM+CRF_with_improv(2).ipynb", what: "Improved text model with subword label propagation",
    status: "available", note: "Runs end to end in Google Colab" },
  { name: "train_audio_wav2vec.ipynb", what: "Audio classifier training", status: "available",
    note: "Runs end to end in Google Colab" },
  { name: "final_fused_results.ipynb", what: "Late fusion and final evaluation", status: "available",
    note: "Includes the real-world voice message inference pipeline" },
  { name: "hf-space-api/", what: "FastAPI inference service", status: "available",
    note: "Dockerfile and pinned requirements.txt included" },
  { name: "urtox-ui/", what: "This research hub", status: "available",
    note: "React application, deployable as a static build" },
  { name: "Trained model weights", what: "Text and audio checkpoints", status: "hosted",
    note: "Published on Hugging Face and fetched by the API at startup" },
  { name: "URTOX-MM", what: "Synthesised speech dataset, 14,338 clips", status: "hosted",
    note: "Published on Hugging Face" },
  { name: "URTOX-HumanAudio", what: "Real-speech evaluation set, 2,000 clips", status: "request",
    note: "Shared on request because it contains real recorded voices" },
];

const STATUS = {
  available: { icon: CheckCircle2, label: "In the repository" },
  hosted: { icon: Cloud, label: "Hosted" },
  request: { icon: Mail, label: "On request" },
};

export default function Reproducibility() {
  const counts = ARTEFACTS.reduce((acc, a) => {
    acc[a.status] = (acc[a.status] || 0) + 1;
    return acc;
  }, {});

  return (
    <Section
      id="reproducibility"
      index="13"
      tone="forest"
      eyebrow="Reproducibility"
      title="Run it yourself"
      lead="Every component of this work is published: the dataset, the pinned training environment, the training and fusion notebooks, the trained weights and the inference service. Here is where each piece lives and how to start."
    >
      <Reveal>
        <div className="grid gap-3 sm:grid-cols-3">
          {[
            ["available", "In the repository"],
            ["hosted", "Hosted on Hugging Face"],
            ["request", "Shared on request"],
          ].map(([key, label]) => (
            <div key={key} className="rounded-lg border border-sand/20 bg-white/[0.05] p-5">
              <p className="font-serif text-3xl !text-cream">{counts[key] || 0}</p>
              <p className="mt-1 text-sm text-sand/70">{label}</p>
            </div>
          ))}
        </div>
      </Reveal>

      <Reveal className="mt-8">
        <div className="rounded-lg border border-sand/20 bg-white/[0.04]">
          <table className="w-full table-auto border-collapse text-[0.8rem] sm:text-sm">
            <caption className="sr-only">Repository artefact inventory</caption>
            <thead>
              <tr className="border-b border-sand/20">
                <th className="px-4 py-3 text-left text-[0.7rem] font-semibold uppercase tracking-wider text-sand-deep">
                  Artefact
                </th>
                <th className="px-4 py-3 text-left text-[0.7rem] font-semibold uppercase tracking-wider text-sand-deep">
                  What it is
                </th>
                <th className="px-4 py-3 text-left text-[0.7rem] font-semibold uppercase tracking-wider text-sand-deep">
                  Status
                </th>
              </tr>
            </thead>
            <tbody>
              {ARTEFACTS.map((a) => {
                const s = STATUS[a.status];
                const Icon = s.icon;
                return (
                  <tr key={a.name} className="border-b border-sand/10 last:border-0">
                    <td className="px-4 py-3">
                      <code className="font-mono text-xs text-cream">{a.name}</code>
                    </td>
                    <td className="px-4 py-3 text-sand/75">
                      {a.what}
                      <span className="mt-0.5 block text-xs text-sand/45">{a.note}</span>
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={`inline-flex items-center gap-1.5 whitespace-nowrap text-xs font-medium ${
                          a.status === "request" ? "text-merlot-wash" : "text-sand"
                        }`}
                      >
                        <Icon size={13} aria-hidden="true" />
                        {s.label}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Reveal>

      <Reveal className="mt-10 grid gap-8 lg:grid-cols-2">
        <div>
          <h3 className="!text-cream text-lg">Reported training configuration</h3>
          <p className="mt-2 text-sm text-sand/70">
            The hyperparameters given in MUTEX-M Table 8. These are what a reproduction should
            target.
          </p>
          <div className="mt-5 rounded-lg border border-sand/20 bg-white/[0.04]">
            <table className="w-full table-auto border-collapse text-[0.8rem] sm:text-sm">
              <caption className="sr-only">Reported hyperparameters</caption>
              <thead>
                <tr className="border-b border-sand/20">
                  <th className="px-4 py-2.5 text-left text-[0.7rem] font-semibold uppercase tracking-wider text-sand-deep">
                    Hyperparameter
                  </th>
                  <th className="px-4 py-2.5 text-left text-[0.7rem] font-semibold uppercase tracking-wider text-sand-deep">
                    Text
                  </th>
                  <th className="px-4 py-2.5 text-left text-[0.7rem] font-semibold uppercase tracking-wider text-sand-deep">
                    Audio
                  </th>
                </tr>
              </thead>
              <tbody>
                {HYPERPARAMS.map((h) => (
                  <tr key={h.name} className="border-b border-sand/10 last:border-0">
                    <td className="px-4 py-2.5 text-sand/70">{h.name}</td>
                    <td className="px-4 py-2.5 font-mono text-xs text-cream">{h.text}</td>
                    <td className="px-4 py-2.5 font-mono text-xs text-cream">{h.audio}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div>
          <h3 className="!text-cream text-lg">Compute</h3>
          <div className="mt-4 space-y-2.5 text-sm">
            {[
              ["Hardware", TRAINING_COST.hardware],
              ["Stack", TRAINING_COST.stack],
              ["Determinism", TRAINING_COST.seed],
            ].map(([k, v]) => (
              <div key={k} className="rounded-md border border-sand/20 bg-white/[0.04] px-4 py-3">
                <p className="text-xs uppercase tracking-wide text-sand-deep">{k}</p>
                <p className="mt-1 text-sand/85">{v}</p>
              </div>
            ))}
          </div>

          <div className="mt-5 overflow-hidden rounded-lg border border-sand/20">
            <table className="w-full border-collapse text-sm">
              <caption className="sr-only">Reported training time</caption>
              <tbody>
                {TRAINING_COST.components.map((c) => (
                  <tr key={c.component} className="border-b border-sand/10">
                    <td className="px-4 py-2.5 text-sand/85">{c.component}</td>
                    <td className="px-4 py-2.5 text-xs text-sand/50">{c.phase}</td>
                    <td className="px-4 py-2.5 text-right font-mono text-xs text-cream">{c.time}</td>
                  </tr>
                ))}
                <tr className="bg-white/[0.06]">
                  <td className="px-4 py-2.5 font-medium text-cream" colSpan={2}>
                    Total
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-xs text-cream">
                    {TRAINING_COST.total}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <Source className="!text-sand/45">MUTEX-M Tables 7 and 8.</Source>
        </div>
      </Reveal>

      <Reveal className="mt-10 grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className="!text-cream text-lg">Running the inference service</h3>
          <p className="mt-2 text-sm text-sand/70">
            The FastAPI service in <code className="font-mono text-xs">hf-space-api/</code> has a
            pinned <code className="font-mono text-xs">requirements.txt</code> and a Dockerfile, and
            downloads its model artefacts at startup.
          </p>
          <pre className="mt-4 overflow-x-auto rounded-lg bg-black/25 p-4 font-mono text-[0.78rem] leading-relaxed text-sand">
{`# training environment
pip install -r requirements-training.txt

# inference service
cd hf-space-api
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860`}
          </pre>

          <h3 className="!text-cream mt-8 text-lg">Regenerating the figures on this page</h3>
          <p className="mt-2 text-sm text-sand/70">
            Every dataset statistic shown here is produced by one script, so the numbers can be
            checked against the CSV directly.
          </p>
          <pre className="mt-4 overflow-x-auto rounded-lg bg-black/25 p-4 font-mono text-[0.78rem] leading-relaxed text-sand">
{`cd urtox-ui
python scripts/build_data.py`}
          </pre>
        </div>

        <div className="space-y-4">
          <Callout icon={Rocket} tone="merlot" title="Start here">
            Clone the repository, create the environment from{" "}
            <code className="font-mono text-xs">requirements-training.txt</code>, then open the
            notebooks in order: text model, audio classifier, then fusion. The notebooks are written
            for Google Colab, so a GPU runtime and a mounted Drive folder for the audio clips is the
            quickest path. Reported figures come from a single RTX A6000 run with all seeds fixed at
            42.
          </Callout>

          <div className="rounded-lg border border-sand/20 bg-white/[0.04] p-5">
            <p className="flex items-center gap-2 text-sm font-medium text-cream">
              <Terminal size={15} aria-hidden="true" />
              Start from the repository
            </p>
            <p className="mt-2 text-sm text-sand/70">
              The notebooks, the dataset, the API and this site all live in one place.
            </p>
            <a
              href={LINKS.github}
              target="_blank"
              rel="noreferrer"
              className="mt-4 inline-flex items-center gap-2 rounded-md bg-sand px-3.5 py-2 text-sm font-medium text-forest-deep transition-colors hover:bg-cream"
            >
              <GithubMark size={14} />
              Open the repository
            </a>
          </div>
        </div>
      </Reveal>
    </Section>
  );
}
