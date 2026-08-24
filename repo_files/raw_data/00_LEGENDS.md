# Supplemental raw data — legends for PeerJ upload

Upload each CSV as a **Supplemental File**, set **file type = Dataset**, and paste the
matching legend into the Edit box. Each legend already contains the term **raw data**
as required by PeerJ.

---

**raw_data_table4.csv** — *Raw data: token-level precision, recall and F1 for all four
model configurations (BiLSTM-CRF, mBERT, XLM-RoBERTa, XLM-RoBERTa+CRF) on the held-out
URTOX test set of 1,434 posts, averaged over five random seeds (42, 123, 456, 789, 1011),
with paired t-test statistics against the proposed MUTEX model. Underlies Table 4 and
Figure 5.*

**raw_data_table5.csv** — *Raw data: token-level F1 and mean toxic-span length for each
toxicity category (hate speech, personal insults, offensive language, profanity) produced
by XLM-RoBERTa+CRF. Underlies Table 5.*

**raw_data_table6_7_8_domain.csv** — *Raw data: domain-specific performance of the
multi-domain model, single-domain versus multi-domain training comparison, and the full
3x3 cross-domain transfer matrix, all averaged over five runs. Underlies Tables 6, 7 and 8.*

**raw_data_table9_domain_bias.csv** — *Raw data: token-level F1 broken down by script
variation (Nastaliq vs Roman Urdu), code-switching and formality level, with deltas from
the 0.600 baseline. Underlies Table 9.*

**raw_data_table10_learning_curve.csv** — *Raw data: learning-curve measurements of
token-level F1, precision and recall at 20%, 40%, 60%, 80% and 100% of the training data,
with paired t-test p-values against the previous data size. Underlies Table 10.*

**raw_data_table11_preprocessing_ablation.csv** — *Raw data: 5-fold cross-validation
ablation of each preprocessing component, reporting mean F1, standard deviation, delta,
p-value and 95% confidence interval. Underlies Table 11.*

**raw_data_table12_supervision.csv** — *Raw data: F1 comparison of fully supervised
span-level detection against weakly supervised alternatives (attention-based rationale
extraction, attention analysis). Underlies Table 12.*

**raw_data_table13_benchmark_comparison.csv** — *Raw data: comparison of the proposed
Urdu systems against the top SemEval-2021 Task 5 English systems, with metric and dataset
size stated for each. Underlies Table 13.*

**raw_data_statistical_tests.csv** — *Raw data: complete set of statistical significance
tests performed in this study (paired two-sided t-tests over five seeds or five
cross-validation folds), reporting n, degrees of freedom, t statistic, p-value, effect
size in F1 percentage points, and the significance decision at alpha = 0.05.*

**raw_data_error_analysis.csv** — *Raw data: manual error analysis of 500 randomly
sampled test-set predictions, categorised into boundary errors, context-dependent
toxicity, code-switched spans, implicit toxicity and multi-span posts.*

**raw_data_test_predictions.csv** — *Raw data: token-level gold and predicted BIO labels
for every token of the 1,434 held-out test posts, for each of the five random seeds,
together with the per-post F1. This is the primary raw output from which all aggregate
scores in the manuscript are computed.*
  → **Generate this file by running `export_raw_results.py` (see below).** It is the one
    file that must come from your notebooks rather than from the reported tables.

---

## IMPORTANT — before you upload

The table CSVs above were transcribed from the values reported in the manuscript so that
the file structure and legends are ready. **Re-export them from your notebooks** so that
the per-seed values (not just the means) are the ones actually written by your code, and
fill the blank `statistic_t` cells for the ablation rows. Reviewers may recompute the
t-tests from these files, so the numbers must be your real logged outputs.
