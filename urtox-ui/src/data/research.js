/**
 * Every figure the site attributes to a paper lives here, with the table or
 * section it was taken from recorded alongside it.
 *
 * P1 = "MUTEX: Leveraging Multilingual Transformers and Conditional Random
 *       Fields for Enhanced Urdu Toxic Span Detection" (arXiv:2603.05057)
 * P2 = "Advancing Urdu Toxicity Detection: Improved Span Models and a
 *       Multimodal Audio Extension" (manuscript, under review)
 *
 * Dataset statistics computed from the released CSV live in stats.json and are
 * deliberately kept separate. See the data-provenance note in the UI.
 */

export const LINKS = {
  github: "https://github.com/inayatarshad/toxic_span_detection_urdu",
  dataset: "https://huggingface.co/datasets/inayatarshad/URTOX",
  datasetFile:
    "https://huggingface.co/datasets/inayatarshad/URTOX/resolve/main/URTOX_v2.csv",
  arxiv: "https://arxiv.org/abs/2603.05057",
  audioDataset: "https://huggingface.co/datasets/finalyear226/URTOX-MM",
  audioFiles: "https://huggingface.co/datasets/finalyear226/audiofiles",
  modelRepo: "https://huggingface.co/finalyear226/urdu-toxic-span-detector",
  contact: "enayyat156@gmail.com",
};

export const INSTITUTION =
  "Department of Computer and Information Sciences, Pakistan Institute of Engineering and Applied Sciences (PIEAS), Islamabad, Pakistan";

export const PAPERS = [
  {
    id: "mutex",
    title:
      "MUTEX: Leveraging Multilingual Transformers and Conditional Random Fields for Enhanced Urdu Toxic Span Detection",
    authors: ["Inayat Arshad", "Fajar Saleem", "Ijaz Hussain"],
    corresponding: "Ijaz Hussain, ijazhussain@pieas.edu.pk",
    status: "Preprint · under review",
    venue: "Preprint on arXiv; submitted to a peer-reviewed journal",
    year: "2026",
    posted: "5 March 2026",
    arxivId: "arXiv:2603.05057",
    url: LINKS.arxiv,
    headline: "60.0% token-level F1, the first supervised baseline for Urdu toxic span detection.",
    contributions: [
      "URTOX, a manually annotated token-level Urdu toxic-span dataset drawn from social media, Urdu newspapers and YouTube.",
      "MUTEX, an XLM-RoBERTa + CRF sequence-labelling framework for span-level toxicity in Urdu.",
      "A supervised benchmark comparing BiLSTM, mBERT, XLM-RoBERTa and XLM-RoBERTa + CRF.",
      "Gradient-based token attribution (integrated gradients) for token-level explanations.",
      "Ablations isolating the contribution of preprocessing, the CRF layer and multi-domain training.",
    ],
  },
  {
    id: "mutex-m",
    title:
      "Advancing Urdu Toxicity Detection: Improved Span Models and a Multimodal Audio Extension",
    authors: ["Fajar Saleem†", "Inayat Arshad†", "Ijaz Hussain"],
    authorNote: "† These authors contributed equally to this work.",
    corresponding: "Ijaz Hussain, ijazhussain@pieas.edu.pk",
    status: "Manuscript · under review",
    venue: "Not yet assigned",
    year: "2026",
    headline:
      "67.0% token-level F1 for text; 83.2% weighted F1 fused on synthesised speech, 77.1% on real speech.",
    contributions: [
      "Identifies and corrects a subword label-assignment artifact in the MUTEX baseline that discarded most of the available gradient signal.",
      "URTOX-MM, a paired text–audio dataset of 14,338 TTS-synthesised Urdu clips.",
      "An MMS-300M utterance-level toxicity classifier with the Urdu language adapter activated.",
      "URTOX-HumanAudio, a 2,000-clip real-speech evaluation set stratified across four Pakistani regional accents.",
      "Weighted late fusion of the text and audio modules, with span localisation performed by the text module.",
    ],
  },
];

/* ---------------------------------------------------------------- dataset */

// P1 §3.2 and Table 2 / P2 Table 1. The released CSV carries no source column,
// so this breakdown is reported by the papers rather than recomputed.
export const SOURCE_DOMAINS = [
  {
    name: "Social media",
    samples: 5254,
    toxic: 57,
    detail: "X (formerly Twitter), Instagram, Reddit",
  },
  {
    name: "Urdu newspapers",
    samples: 4300,
    toxic: 52,
    detail: "Daily Jang, UrduPoint, BOL News Urdu, Independent Urdu",
  },
  {
    name: "YouTube",
    samples: 4788,
    toxic: 56,
    detail: "Video comments, captions and descriptions",
  },
];

// P1 §3.3
export const AGREEMENT = {
  kappa: 0.82,
  alpha: 0.81,
  adjudicated: "≈15% of samples went through adjudication to resolve disagreement",
};

// P1 §3.6 and P2 §5
export const SPLITS = {
  p1: [
    { name: "Train", samples: 11474, share: "80%" },
    { name: "Validation", samples: 1434, share: "10%" },
    { name: "Test", samples: 1434, share: "10%" },
  ],
  p2: [
    { name: "Train", samples: 10036, share: "70%" },
    { name: "Validation", samples: 1434, share: "10%" },
    { name: "Test", samples: 2868, share: "20%" },
  ],
};

// P1 §3.5. Preprocessing sequence applied before tokenisation
export const PREPROCESSING = [
  { step: "Unicode normalisation (NFC)", note: "Uniform representation of diacritics and ligatures" },
  { step: "Diacritic handling", note: "Aeraab marks removed, essential characters preserved" },
  { step: "Roman-to-Nastaliq transliteration", note: "Rule-based conversion of Romanised Urdu" },
  { step: "Noise removal", note: "URLs, emails and excessive punctuation" },
  { step: "Whitespace normalisation", note: "Collapses irregular spacing" },
  { step: "Word segmentation", note: "Separates concatenated words common in Urdu social text" },
  { step: "Tokenisation", note: "spaCy, customised for Urdu-specific behaviour" },
];

/* ------------------------------------------------------------- P1 results */

// P1 Table 4. Token-level, test set of 1,434 samples, averaged over 5 seeds
export const P1_MODELS = [
  { model: "BiLSTM", crf: false, precision: 57.0, recall: 55.0, f1: 56.0, p: "p < 0.001" },
  { model: "mBERT", crf: false, precision: 57.0, recall: 55.0, f1: 56.0, p: "p < 0.001" },
  { model: "XLM-RoBERTa", crf: false, precision: 60.0, recall: 58.0, f1: 59.0, p: "p = 0.023" },
  { model: "XLM-RoBERTa + CRF", crf: true, precision: 61.0, recall: 59.0, f1: 60.0, p: "n/a", best: true },
];

// P1 Table 6
export const P1_DOMAIN = [
  { domain: "Social media", precision: 58.5, recall: 56.8, f1: 57.6 },
  { domain: "News", precision: 63.2, recall: 61.4, f1: 62.3 },
  { domain: "YouTube", precision: 59.8, recall: 58.1, f1: 58.9 },
  { domain: "Multi-domain (overall)", precision: 61.0, recall: 59.0, f1: 60.0, best: true },
];

// P1 Table 8. Rows are training domain, columns are test domain
export const P1_TRANSFER = {
  columns: ["Social media", "News", "YouTube"],
  rows: [
    { train: "Social media", values: [61.3, 53.8, 55.7], diagonal: 0 },
    { train: "News", values: [52.4, 59.3, 52.9], diagonal: 1 },
    { train: "YouTube", values: [54.8, 53.4, 60.7], diagonal: 2 },
    { train: "Multi-domain", values: [57.6, 62.3, 58.9], diagonal: -1 },
  ],
};

// P1 Table 11. 5-fold cross-validation
export const P1_PREPROCESS_ABLATION = [
  { config: "Full preprocessing (baseline)", f1: 60.0, sd: 0.8, delta: null, p: "n/a" },
  { config: "w/o Unicode normalisation", f1: 58.2, sd: 1.1, delta: -1.8, p: "0.007" },
  { config: "w/o diacritic handling", f1: 59.0, sd: 0.9, delta: -1.0, p: "0.042" },
  { config: "w/o Roman Urdu conversion", f1: 56.3, sd: 1.3, delta: -3.7, p: "0.001" },
  { config: "w/o URL / emoji removal", f1: 59.5, sd: 0.7, delta: -0.5, p: "0.223" },
  { config: "w/o deduplication", f1: 59.8, sd: 0.8, delta: -0.2, p: "0.634" },
  { config: "No preprocessing", f1: 53.8, sd: 1.9, delta: -6.2, p: "0.001" },
];

// P1 Table 10
export const P1_LEARNING_CURVE = [
  { share: "20%", samples: 2868, f1: 53.5, sd: 2.8 },
  { share: "40%", samples: 5737, f1: 54.2, sd: 2.1 },
  { share: "60%", samples: 8605, f1: 57.5, sd: 1.6 },
  { share: "80%", samples: 11474, f1: 59.3, sd: 1.2 },
  { share: "100%", samples: 14342, f1: 60.0, sd: 0.9 },
];

// P1 §4.4.1. Error analysis over 500 sampled predictions
export const P1_ERRORS = [
  { type: "Boundary errors", share: 34 },
  { type: "Context-dependent toxicity", share: 28 },
  { type: "Code-switched spans", share: 18 },
  { type: "Implicit toxicity", share: 12 },
  { type: "Multi-span posts", share: 8 },
];

/* ------------------------------------------------------------- P2 results */

// P2 Table 9. Token-level on the 2,868-sample test set
export const P2_TEXT_MODELS = [
  { model: "BiLSTM-CRF", precision: 57.0, recall: 55.0, f1: 56.0 },
  { model: "mBERT", precision: 59.0, recall: 57.5, f1: 59.2 },
  { model: "XLM-RoBERTa (MUTEX baseline)", precision: 60.0, recall: 58.0, f1: 60.0 },
  { model: "XLM-RoBERTa (MUTEX-M)", precision: 64.2, recall: 71.5, f1: 67.0, best: true },
];

// P2 Table 12. Each row adds one change to the row above
export const P2_ABLATION = [
  { config: "MUTEX baseline (original)", f1: 60.0, gain: null },
  { config: "+ Subword label propagation", f1: 64.0, gain: 4.0 },
  { config: "+ Extended training (8 epochs) + early stopping", f1: 65.5, gain: 1.5 },
  { config: "+ Batch size 32, LR 2×10⁻⁵", f1: 66.5, gain: 1.0 },
  { config: "+ Linear warmup (ratio 0.1)", f1: 67.0, gain: 0.5 },
];

// P2 Table 10. Utterance level, 2,868-sample test set
export const P2_AUDIO = [
  {
    model: "wav2vec 2.0",
    pretraining: "English",
    rows: [
      { cls: "Non-toxic", precision: 0.86, recall: 0.46, f1: 0.6, support: 1319 },
      { cls: "Toxic", precision: 0.68, recall: 0.94, f1: 0.79, support: 1549 },
      { cls: "Weighted avg", precision: 0.76, recall: 0.72, f1: 0.7, support: 2868, total: true },
    ],
  },
  {
    model: "MMS-300M",
    pretraining: "1,400+ languages incl. Urdu",
    best: true,
    rows: [
      { cls: "Non-toxic", precision: 0.89, recall: 0.61, f1: 0.72, support: 1319 },
      { cls: "Toxic", precision: 0.76, recall: 0.95, f1: 0.84, support: 1549 },
      { cls: "Weighted avg", precision: 0.83, recall: 0.8, f1: 0.79, support: 2868, total: true },
    ],
  },
];

// P2 Table 11. α weights the text module, (1−α) the audio module
export const P2_FUSION = [
  { alpha: 0.4, f1: 80.1 },
  { alpha: 0.5, f1: 83.2, best: true },
  { alpha: 0.6, f1: 82.8 },
  { alpha: 0.7, f1: 81.4 },
];

export const P2_FUSION_BASELINES = { textOnly: 67.0, audioOnly: 79.0 };

// P2 Table 15. 2,000 real Urdu clips, no adaptation to the real-speech distribution
export const P2_REAL_SPEECH = [
  { setup: "Text only (Whisper + XLM-R)", precision: 0.72, recall: 0.69, f1: 0.705, drop: -8.5 },
  { setup: "Audio only, wav2vec 2.0", precision: 0.59, recall: 0.63, f1: 0.61, drop: -9.0 },
  { setup: "Audio only, MMS-300M", precision: 0.71, recall: 0.74, f1: 0.725, drop: -6.5 },
  { setup: "MUTEX-M fusion (α = 0.5)", precision: 0.78, recall: 0.76, f1: 0.771, drop: -6.1, best: true },
];

// P2 Table 16
export const P2_ACCENTS = [
  { accent: "Standard (Islamabad) Urdu", clips: 760, f1: 0.812 },
  { accent: "Punjabi-accented Urdu", clips: 620, f1: 0.751 },
  { accent: "Sindhi-accented Urdu", clips: 360, f1: 0.734 },
  { accent: "Pashto-accented Urdu", clips: 260, f1: 0.718 },
];

// P2 Table 14. LLM prompting baselines on the same 2,868-sample test set
export const P2_LLM_BASELINES = [
  { model: "Llama-3.1 8B Instruct", setting: "Zero-shot", parseFail: 8.7, f1: 51.2 },
  { model: "Llama-3.1 8B Instruct", setting: "5-shot", parseFail: 5.1, f1: 57.3 },
  { model: "GPT-4o", setting: "Zero-shot", parseFail: 2.4, f1: 58.9 },
  { model: "GPT-4o", setting: "5-shot", parseFail: 1.8, f1: 62.8 },
  { model: "Qalb (Urdu-specialised)", setting: "Zero / 5-shot", parseFail: 3.2, f1: 61.9 },
  { model: "XLM-RoBERTa + CRF (MUTEX-M)", setting: "Supervised", parseFail: null, f1: 67.0, best: true },
];

// P2 Table 17. 816 misclassified tokens on the held-out test set
export const P2_ERRORS = [
  { type: "Span boundary errors", count: 312, share: 38.2, ci: "34.8–41.8" },
  { type: "Missed toxic spans", count: 201, share: 24.6, ci: "21.6–27.9" },
  { type: "False positives", count: 198, share: 24.3, ci: "21.3–27.5" },
  { type: "Code-switched token errors", count: 105, share: 12.9, ci: "10.7–15.4" },
];

// P2 Table 19
export const P2_FN_BY_SPAN_LENGTH = [
  { length: "1", spans: 1842, fnRate: 18.4 },
  { length: "2", spans: 763, fnRate: 24.1 },
  { length: "3–4", spans: 298, fnRate: 31.6 },
  { length: "5+", spans: 87, fnRate: 44.8 },
];

// P2 Table 2
export const URTOX_MM = {
  clips: 14338,
  excluded: 5,
  format: "MP3 (32 kbps), resampled to 16 kHz",
  voice: "ur-PK-AsadNeural (Microsoft Edge TTS)",
  fileSize: "16 KB – 60 KB (mean ≈ 35 KB)",
  duration: "≈ 8–9 s per clip, ≈ 35 h total",
  balance: "54% toxic / 46% non-toxic",
};

// P2 Table 4
export const HUMAN_AUDIO = {
  clips: 2000,
  toxic: 1214,
  nonToxic: 786,
  duration: "9.8 s average",
  kappa: 0.81,
  sources: [
    { name: "YouTube comments and replies", share: 61 },
    { name: "Consented recordings", share: 29 },
    { name: "Anonymised public voice chats", share: 10 },
  ],
  accents: [
    { name: "Standard (Islamabad) Urdu", share: 38 },
    { name: "Punjabi-accented Urdu", share: 31 },
    { name: "Sindhi-accented Urdu", share: 18 },
    { name: "Pashto-accented Urdu", share: 13 },
  ],
  availability: "Available on request, for privacy reasons",
};

// P2 Table 8
export const HYPERPARAMS = [
  { name: "Base model", text: "XLM-RoBERTa (xlm-roberta-base)", audio: "MMS-300M (facebook/mms-300m)" },
  { name: "Language adapter", text: "n/a", audio: "urd" },
  { name: "Optimizer", text: "AdamW", audio: "Adam" },
  { name: "Learning rate", text: "2×10⁻⁵", audio: "1×10⁻³" },
  { name: "Batch size", text: "32", audio: "32" },
  { name: "Epochs", text: "8", audio: "20" },
  { name: "Warmup ratio", text: "0.1", audio: "n/a" },
  { name: "Dropout", text: "0.1", audio: "0.3 / 0.2" },
  { name: "Max sequence length", text: "128 tokens", audio: "10 seconds" },
  { name: "Early stopping", text: "Validation F1", audio: "Validation F1" },
  { name: "Loss function", text: "Weighted CRF NLL", audio: "Binary cross-entropy" },
  { name: "Class weighting", text: "Inverse frequency", audio: "None" },
];

// P2 Table 7
export const TRAINING_COST = {
  hardware: "Single NVIDIA RTX A6000 (48 GB VRAM)",
  stack: "PyTorch 2.1 · HuggingFace Transformers 4.38 · torchaudio 2.1 · edge-tts 6.1.9",
  seed: "All random seeds fixed at 42",
  components: [
    { component: "XLM-R + CRF", phase: "Fine-tune (8 epochs)", time: "≈ 18 h" },
    { component: "MMS-300M", phase: "Embedding extraction (phase 1)", time: "≈ 3 h" },
    { component: "MLP classifier", phase: "Training (phase 2, 20 epochs)", time: "< 5 min" },
    { component: "Edge TTS", phase: "Audio synthesis (14,338 clips)", time: "≈ 4 h" },
  ],
  total: "≈ 26 h",
};

/* -------------------------------------------------------- linguistic notes */

// P1 §5.2, §6 and P2 §2.2. Reported for URTOX / Urdu online content generally
export const URDU_CHALLENGES = [
  {
    key: "script",
    title: "Dual-script reality",
    icon: "Languages",
    body: "Urdu is written in the cursive Nastaliq script, but a substantial share of online Urdu appears in Latin characters instead. The same toxic expression can therefore surface in two entirely different orthographies.",
    evidence: "Removing Roman-to-Nastaliq conversion costs 3.7 F1 points, the largest single preprocessing effect measured.",
    source: "P1 Table 11, §6",
  },
  {
    key: "unicode",
    title: "Unicode normalisation",
    icon: "Binary",
    body: "The same visual character can be encoded in several ways across platforms, and zero-width non-joiner usage is inconsistent. Without normalisation, tokenisation fragments in ways that break span alignment.",
    evidence: "Removing Unicode normalisation costs 1.8 F1 points.",
    source: "P1 Table 11",
  },
  {
    key: "morphology",
    title: "Morphological richness",
    icon: "GitBranch",
    body: "Affixation, compounding and derivation mean one toxic concept appears in many surface forms. A model must recognise a derived form as toxic even when only the root was seen in training.",
    evidence: "Cited as a principal contributor to the gap against English systems.",
    source: "P1 §5.2, §6",
  },
  {
    key: "codeswitch",
    title: "Code-switching",
    icon: "Shuffle",
    body: "Urdu–English mixing is common in online posts, and toxic spans frequently straddle the language boundary, which challenges a BIO tagger that must decide boundaries within a single sequence.",
    evidence: "Code-switched token errors account for 12.9% of misclassified tokens in MUTEX-M.",
    source: "P2 Table 17; P1 §5.2",
  },
  {
    key: "subword",
    title: "Subword segmentation",
    icon: "Scissors",
    body: "SentencePiece splits morphologically rich Urdu words into three to five subword pieces. When a toxic word is fragmented, no individual piece carries the toxic signal on its own.",
    evidence: "bewaqoof segments into be/wa/qo/of, producing an O label in 68% of such cases.",
    source: "P2 §5.8",
  },
  {
    key: "lowresource",
    title: "Thin pretraining coverage",
    icon: "Database",
    body: "Urdu makes up under 1% of XLM-RoBERTa's pretraining data. Multilingual models arrive with far weaker Urdu semantics than they have for high-resource languages, before any fine-tuning begins.",
    evidence: "Reported as an ≈8% contribution to the error rate relative to English systems.",
    source: "P1 §5.2",
  },
  {
    key: "implicit",
    title: "Sarcasm and implicit toxicity",
    icon: "MessageCircleQuestion",
    body: "Toxicity carried by pragmatics rather than vocabulary, such as sarcasm, culturally specific euphemism and negation, has no explicit lexical anchor for a token classifier to attach to.",
    evidence: "SHAP scores show only partial negation handling: ahmaq scores +0.71 unnegated and +0.43 when negated.",
    source: "P2 §5.9; P1 §6.1",
  },
  {
    key: "domain",
    title: "Domain heterogeneity",
    icon: "Layers",
    body: "Formal newspaper Urdu, informal social media Urdu and mixed-register YouTube commentary differ enough that a model trained on one transfers poorly to another.",
    evidence: "Cross-domain testing drops 7–12 F1 points, and news → social media is the weakest transfer in the matrix.",
    source: "P1 Table 8",
  },
];

/* ------------------------------------------------------------ future work */

// P2 §7 (priorities) and P1 §6.2 (longer-horizon directions)
export const FUTURE_WORK = [
  {
    priority: "Priority 1",
    title: "Span-based detection heads",
    body: "Boundary errors are the largest failure mode. A SpanBERT-style objective would score candidate spans holistically rather than tagging tokens independently, enforcing span-level consistency.",
    evidence: "38.2% of misclassifications are boundary errors; false-negative rate rises from 18.4% on single-token spans to 44.8% on spans of five or more tokens.",
    source: "P2 §7",
  },
  {
    priority: "Priority 2",
    title: "Real-speech collection and parameter-efficient audio fine-tuning",
    body: "Collecting 2,000–3,000 consented real-speech clips and applying LoRA or adapter fine-tuning is identified as the primary path to closing the residual acoustic domain gap.",
    evidence: "Fusion degrades 6.1 points from synthesised to real speech.",
    source: "P2 §7",
  },
  {
    priority: "Priority 3",
    title: "ASR-robust text-module training",
    body: "Fine-tuning the text model on ASR-transcribed Urdu, or applying ASR noise augmentation, would decouple text-model accuracy from transcription quality at lower cost than improving Urdu ASR outright.",
    evidence: "Performance falls from 81.2% on standard Islamabad Urdu to 71.8% on Pashto-accented Urdu, tracking rising Whisper word error rate.",
    source: "P2 §7",
  },
  {
    priority: "Priority 4",
    title: "Audio-informed span boundary refinement",
    body: "The audio module currently contributes only utterance-level gating, capping span-level F1 at the text model's 67%. Forced alignment could map frame-level acoustic representations onto transcript tokens.",
    evidence: "Span localisation is performed exclusively by the text module.",
    source: "P2 §7",
  },
  {
    priority: "Deferred",
    title: "Implicit toxicity and cross-lingual extension",
    body: "Sarcasm, negation and pragmatic inference need discourse-level context beyond the current single-utterance scope. Extension to Hindi, Punjabi and Bengali is noted as technically straightforward but explicitly not motivated by any finding in the present work.",
    evidence: "Recorded by the authors as deferred rather than planned.",
    source: "P2 §7",
  },
];
