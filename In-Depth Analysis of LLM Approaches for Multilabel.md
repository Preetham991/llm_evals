<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

### In-Depth Analysis of LLM Approaches for Multilabel Email Classification with Low Confidence Scores

As a senior data scientist leveraging LLMs for multilabel email classification (e.g., tagging emails with labels like "spam," "urgent," "work," or "personal"), low log probabilities (logprobs) often arise from the autoregressive nature of LLMs, which prioritize sequential token generation over joint probability estimation. This leads to suppressed confidences for multiple labels, as probability mass is distributed across a vast vocabulary rather than balanced across labels. Building on our prior discussion, this expanded analysis delves deeper into LLM-specific mechanics, probabilistic challenges, and empirical benchmarks as of August 20, 2025. I'll integrate mathematical insights, highlight state-of-the-art techniques like multicalibration and adaptive temperature scaling, and provide ready-to-use Python code snippets (using libraries like transformers and torch). These are tailored for your workflow, assuming access to models like GPT-4o or Llama via APIs or local inference.[^1][^2][^3]

#### Deeper Dive into LLM Mechanics Causing Low Logprobs

LLMs generate outputs token-by-token, computing logprobs via softmax over logits: \$ p(w_t | w_{<t}) = \frac{\exp(z_{w_t})}{\sum_i \exp(z_i)} \$, where \$ z \$ are logits and the denominator spans the vocabulary (often 50k+ tokens)[^2][^1]. In multilabel tasks, this creates:

- **Probability Dilution**: For an email prompt like "Classify: [email text]. Labels: spam (yes/no), urgent (yes/no)", the model allocates mass to all possible tokens, diluting probs for "yes" across labels. If two labels are positive, their joint prob is underestimated, yielding low logprobs (e.g., -2.0 for P=0.135). Empirical studies on subjective datasets show average logprobs drop 20-30% in multilabel vs. single-label.[^2][^3][^4]
- **Conditioning Bias**: Autoregression conditions later tokens on earlier ones, suppressing correlated labels (e.g., "urgent" prob drops if "work" is generated first). Larger models exacerbate "spikiness" (low entropy distributions), but without calibration, this manifests as underconfidence in multilabel.[^5][^6][^1][^2]
- **Calibration Gaps**: LLMs minimize next-token cross-entropy, not multilabel metrics like Brier score (\$ \frac{1}{N} \sum (p_i - y_i)^2 \$) or ECE. On QA-like tasks (analogous to email nuance), uncalibrated ECE exceeds 15%, with logprobs misaligned to empirical accuracies.[^7][^1][^5]

Benchmarks (e.g., GLUE multilabel variants) reveal that zero-shot LLMs like GPT-4o achieve ~75% accuracy but logprobs ~ -1.5 for positives, improving to -0.7 post-calibration.[^1][^2]

#### Advanced Remedies with Code Implementations

Focus on post-hoc calibration (no finetuning needed) and prompt engineering to boost logprobs while preserving accuracy. These methods can increase average probs by 15-30% and reduce ECE to <5%.[^8][^5][^1]

1. **Temperature Scaling: Softening Distributions**
    - **Detailed Mechanism**: Scale logits by T >1 to increase entropy: \$ p' = \softmax(z / T) \$. Optimize T on validation NLL to align probs with accuracies. For multilabel emails, apply per-label or globally; adaptive variants predict T per token via a small model.[^9][^5][^8]
    - **Why Effective for LLMs**: Counters spikiness without altering rankings; on text classification, reduces calibration error by 40-50%.[^5][^8]
    - **Python Code** (Using Hugging Face transformers for Llama or similar; adapt for OpenAI API):

```python
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

# Load model (e.g., Llama-7B)
model_name = "meta-llama/Llama-2-7b-hf"  # Replace with your model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def get_logprobs(prompt, labels):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]  # Last token logits for simplicity
    probs = torch.softmax(logits, dim=-1)
    label_ids = tokenizer.convert_tokens_to_ids(labels)  # e.g., ['yes', 'no']
    return torch.log(probs[:, label_ids]).detach().numpy()  # Logprobs for labels

def temperature_scale(val_prompts, val_targets, init_T=1.0):
    # val_prompts: list of email prompts, val_targets: binary multilabel (N, num_labels)
    val_logits = np.array([get_logprobs(p, ['yes', 'no']) for p in val_prompts])  # Shape (N, num_labels, 2)
    val_logits = val_logits[:, :, 0] - val_logits[:, :, 1]  # Logit diff for 'yes' confidence

    def nll(T):
        scaled = val_logits / T
        probs = 1 / (1 + np.exp(-scaled))  # Sigmoid for multilabel
        return log_loss(val_targets.ravel(), probs.ravel())

    res = minimize_scalar(nll, bounds=(0.1, 10), method='bounded')
    return res.x

# Example usage
val_prompts = ["Classify email: [text]. Is it spam? yes/no"] * 100  # Your validation set
val_targets = np.random.randint(0, 2, (100, 5))  # Dummy multilabel targets (5 labels)
optimal_T = temperature_scale(val_prompts, val_targets)
print(f"Optimal T: {optimal_T}")

# Apply to test: scaled_logprobs = logprobs / optimal_T  # Then softmax
```

        - **Tuning Note**: For emails, use 100-500 validation examples; T~1.5-3 often optimal. This preserves monotonicity and runs in milliseconds.[^8][^5]
2. **Multicalibration: Group-Wise Alignment**
    - **Detailed Mechanism**: Calibrate probs across intersecting groups (e.g., prompt clusters by embeddings or self-annotations like "formal tone?"). Use variants like Isotonic Grouped Least Squares (IGLS) to minimize group-conditional ECE. Math: For groups G, solve $\min \sum_{g \in G} (p_g - acc_g)^2$, ensuring fairness across subsets.[^1]
    - **Why for Multilabel Emails**: Handles subjectivity (e.g., varying confidence by email length); outperforms standard calibration by 20-30% on QA datasets.[^10][^1]
    - **Python Code** (Adapted from arXiv implementations; requires scikit-learn):

```python
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# Embed prompts for grouping
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def multicalibrate(prompts, raw_probs, targets, num_groups=10):
    # prompts: list, raw_probs: (N, num_labels), targets: (N, num_labels) binary
    embeddings = embedder.encode(prompts)
    kmeans = KMeans(n_clusters=num_groups).fit(embeddings)
    groups = kmeans.labels_

    calibrated_probs = np.zeros_like(raw_probs)
    for g in range(num_groups):
        mask = groups == g
        iso = IsotonicRegression(out_of_bounds='clip').fit(raw_probs[mask].ravel(), targets[mask].ravel())
        calibrated_probs[mask] = iso.predict(raw_probs[mask].ravel()).reshape(-1, raw_probs.shape[^16])
    return calibrated_probs

# Example: Assume raw_probs from LLM (N samples, 5 labels)
N, num_labels = 100, 5
raw_probs = np.random.rand(N, num_labels)
targets = np.random.randint(0, 2, (N, num_labels))
prompts = ["Email text here"] * N
cal_probs = multicalibrate(prompts, raw_probs, targets)
```

        - **Extension**: For self-annotation groups, query the LLM with "Is this email formal? yes/no" to form groups. This reduces overfitting on small sets.[^1]
3. **Prompt Engineering: Unary and Pairwise Decompositions**
    - **Detailed Mechanism**: Break multilabel into unary yes/no per label or pairwise comparisons to avoid joint suppression. For unary: Aggregate probs via averaging; for pairwise, use Bradley-Terry: \$ P(A > B) = \sigmoid(z_A - z_B) \$.[^11][^4][^2]
    - **Why Effective**: Yields higher logprobs by isolating decisions; competitive on GLUE multilabel with few-shot.[^12][^11][^2]
    - **Python Code** (For OpenAI API or similar; unary example):

```python
import openai

openai.api_key = "your-api-key"

def unary_multilabel_classify(email_text, labels, model="gpt-4o"):
    probs = []
    for label in labels:  # Unary per label
        prompt = f"Classify email: {email_text}. Is it {label}? Output: yes/no and confidence (0-1)."
        response = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}])
        # Parse response (assume format: "yes, 0.8")
        answer, conf = response.choices.message.content.split(", ")
        prob = float(conf) if answer == "yes" else 1 - float(conf)
        probs.append(prob)
    return np.array(probs)  # Per-label probabilities

# Usage
labels = ["spam", "urgent", "work", "personal", "phishing"]
email = "Your email text here"
conf_scores = unary_multilabel_classify(email, labels)
print(f"Confidences: {conf_scores}")
```

        - **Pairwise Variant**: Prompt "Which fits better: spam or urgent?" and aggregate via tournament ranking to derive probs.[^4][^11]

#### Evaluation and Best Practices

Compute ECE pre/post: Bin probs into M=10 groups, measure |acc - conf| weighted by bin size[^1][^5]. For your emails, use a 200-sample validation set; combine temperature + multicalibration for 40% ECE drop[^1][^8]. If logprobs remain low, finetune with LoRA on calibrated losses[^10]. These techniques are extensible—test on your blob data for iterative refinement.
<span style="display:none">[^13][^14][^15]</span>

<div style="text-align: center">⁂</div>

[^1]: https://arxiv.org/html/2404.04689v1

[^2]: https://arxiv.org/html/2505.17510

[^3]: https://arxiv.org/html/2505.17510v1

[^4]: https://arxiv.org/pdf/2505.17510.pdf

[^5]: https://latitude-blog.ghost.io/blog/5-methods-for-calibrating-llm-confidence-scores/

[^6]: https://www.reddit.com/r/LocalLLaMA/comments/1gh4ht7/are_confidence_scores_from_llms_meaningful/

[^7]: https://community.openai.com/t/evaluating-the-confidence-levels-of-outputs-generated-by-large-language-models-gpt-4o/1127104

[^8]: https://github.com/gpleiss/temperature_scaling

[^9]: https://arxiv.org/abs/2409.19817

[^10]: https://www.amazon.science/publications/label-with-confidence-effective-confidence-calibration-and-ensembles-in-llm-powered-classification

[^11]: https://openreview.net/forum?id=t8ZZ2Y356Ix

[^12]: https://arxiv.org/abs/2401.12178

[^13]: https://arxiv.org/html/2503.02863v1

[^14]: https://openreview.net/forum?id=uuXPWRtwvK

[^15]: https://software-lab.org/publications/icse2025_calibration.pdf

[^16]: https://arxiv.org/html/2312.09304v1

