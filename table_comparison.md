

### Comparison Table: Confidence \& Uncertainty Metrics for LLM Outputs

| Metric | Granularity | Best-suited Tasks / Domains | Key Strengths | Key Limitations | Typical Thresholds (rule-of-thumb) | Primary References |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **Token probability (max-softmax)** | Per-token | Code generation, information extraction, step-wise QA | -  *Fine-grained* error localisation<br>-  Zero extra compute at inference | -  Over-confident in OOD contexts | Flag any token < 0.80 for human review | [^6] |
| **Sequence likelihood / NLL** | Whole sequence | Summarisation, chat ranking, retrieval-augmented generation | -  Aligns with training loss<br>-  Good for ranking multiple drafts | -  Length-sensitive; needs normalisation | Reject if NLL > 2.5 (length-norm) | [^1] |
| **Entropy** | Per-token or averaged | Safety filters, OOD detection, medical advice | -  Captures *distributional spread* not seen in max-probability | -  Requires vocabulary-size scaling | Escalate if H > 3.0 (≈high uncertainty) | [^7] |
| **Margin (top-2 gap)** | Per-token | Multi-choice, legal citations, risk triage | -  Direct ambiguity signal; trivial to compute | -  Unstable for large vocabularies | Escalate if margin < 0.10 | [^1] |
| **Perplexity / inverse perplexity** | Whole sequence | Model benchmarking, creative writing QA | -  Standardised; cross-model comparable | -  Weak link to factual correctness | Re-rank drafts with PP > 15 (news) | [^1] |
| **Consistency / sampling agreement** | Output set | Safety-critical chatbots, audit trails | -  No labels needed; measures robustness | -  Extra inference cost (≥5 samples) | Require ≥80% agreement among 5 samples | [^1] |
| **Semantic similarity (BERTScore / ROUGE)** | Whole sequence vs. reference | Translation, headline generation | -  Captures meaning beyond n-grams | -  Needs reference text | Accept if BERTScore > 0.85 | [^1] |
| **Faithfulness / contextual support** | Claim-level | Retrieval-augmented systems, regulated docs | -  Detects hallucinations; audit-ready | -  Relies on NLI or retrieval quality | Require ≥95% supported claims | [^4] |
| **Responsible-AI scores (bias / toxicity)** | Whole or span | Public-facing content, social platforms | -  Direct ethical safeguard | -  Classifier drift over time | Block if toxicity > 0.20 prob. | [^6] |
| **Reliability diagram / ECE** | Population | Release gating, regression tests | -  Quantifies global calibration | -  Needs labelled eval set | ECE < 0.05 before deployment | [^4] |
| **Deep ensemble variance** | Prediction set | Medical diagnostics, autonomous agents | -  Strong epistemic uncertainty signal | -  Training \& storage × N models | Escalate if variance > 2× baseline | [^7] |
| **Conformal prediction set size** | Instance | Insurance, legal compliance | -  Formal error guarantees | -  Wider sets on hard inputs | Empty set error rate ≤5% by design | [^4] |

**Legend:**

- “Escalate” = route to human review -  All thresholds are illustrative and must be tuned on in-domain validation data.

Citations
= “LLM Evaluation Metrics: The Ultimate Guide,” Confident-AI blog, 2025.[^1]
= Latitude Ghost, “5 Methods for Calibrating LLM Confidence Scores,” 2025.[^7]
= Infrrd AI blog, “Confidence Scores in LLMs,” 2025.[^6]
= arXiv:2406.03441 “Calibrating LLM Confidence with Semantic Steering,” 2024.[^4]

<div style="text-align: center">⁂</div>

[^1]: https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation

[^2]: https://arxiv.org/html/2406.03441v1

[^3]: https://www.deepchecks.com/llm-evaluation-metrics/

[^4]: https://arxiv.org/html/2503.02863v1

[^5]: https://hdsr.mitpress.mit.edu/pub/jaqt0vpb

[^6]: https://www.infrrd.ai/blog/confidence-scores-in-llms

[^7]: https://latitude-blog.ghost.io/blog/5-methods-for-calibrating-llm-confidence-scores/

