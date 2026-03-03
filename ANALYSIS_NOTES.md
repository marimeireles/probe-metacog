# Analysis Notes: Where We Are and What's Next

## 1. GPT-5 Re-Grading Changes the Story

The GPT-5.2 classifications differ significantly from our regex grading:

| Category       | 4B regex | 4B GPT-5 | 27B regex | 27B GPT-5 |
|----------------|----------|----------|-----------|-----------|
| Awareness      | 39.3%    | 48.6%    | 50.0%     | 78.1%     |
| Puzzlement     | 13.1%    | 26.2%    | 6.2%      | 15.6%     |
| Confabulation  | 47.7%    | 25.2%    | 43.8%     | 6.2%      |
| Denial         | 0.0%     | 0.0%     | 0.0%      | 0.0%      |

GPT-5 reclassified many "confabulation" responses as "puzzlement" or "awareness".
The key shift: responses like "I don't know why I did that - it's a glitch in my
system" were regex-matched as confabulation (because they contain "because") but
GPT-5 correctly classifies them as awareness/puzzlement based on overall meaning.

**Important caveat**: The "awareness" here is still the model's trained
error-correction behavior, not genuine metacognitive monitoring. The model says
"that was a glitch" because users told it something was wrong, not because it
detected anything internally. But it IS a behavioral signal we can work with.

---

## 2. Top SAE Features — What They Mean

These are SAE features from Gemma Scope (google/gemma-scope-2-*b-it). To look
them up on Neuronpedia, the URL format is:
  https://www.neuronpedia.org/gemma-2-{size}-it/{layer}-gemmascope-res-{width}l/{feature_id}

### 4B Model — Top 20 Most Activated Features (by mean_delta)

| Rank | Feature | mean_delta | Top Concept(s)              |
|------|---------|-----------|------------------------------|
| 1    | F1671   | 183.2     | beach, ocean, cat (broad)    |
| 2    | F348    | 147.8     | book, coffee, guitar         |
| 3    | F786    | 115.3     | beach, ocean, book (broad)   |
| 4    | F9034   | 85.4      | mountain, book, cat          |
| 5    | F814    | 79.7      | ocean, fire, cat (broad)     |
| 6    | F43457  | 78.1      | fire, rain, tree             |
| 7    | F4155   | 63.6      | tree, mountain, guitar       |
| 8    | F43397  | 58.4      | beach, ocean, fire (broad)   |
| 9    | F664    | 58.4      | tree, mountain, guitar       |
| 10   | F516    | 55.6      | coffee, rain                 |
| 11   | F588    | 51.5      | beach (dominant)             |
| 12   | F12544  | 49.2      | beach, cat, book (broad)     |
| 13   | F12783  | 49.1      | cat, guitar                  |
| 14   | F405    | 48.6      | cat (single concept)         |
| 15   | F205    | 48.0      | ocean, fire, coffee (broad)  |
| 16   | F568    | 46.9      | beach, tree, cat             |
| 17   | F26216  | 46.2      | ocean, fire, coffee          |
| 18   | F1915   | 46.0      | coffee, rain, mountain       |
| 19   | F457    | 45.5      | coffee, rain, beach          |
| 20   | F23146  | 45.5      | book, beach, tree            |

**Pattern**: NO feature is metacognition-specific. They're all concept-content
features — they fire because of what was injected, not because the model detected
that something was injected.

### 4B Model — Top Hit-Predictive Features (fire MORE when model detects injection)

| Rank | Feature | diff     | Interpretation                |
|------|---------|----------|-------------------------------|
| 1    | F205    | +113.2   | Broad concept (ocean/fire)    |
| 2    | F254    | +92.6    | Broad concept (ocean/book)    |
| 3    | F147    | +84.4    | Broad concept (tree/rain)     |
| 4    | F870    | +83.6    | Hit-only (never fires in miss)|
| 5    | F950    | +71.0    | Hit-elevated                  |
| 6    | F7165   | +67.3    | Hit-only                      |
| 7    | F1684   | +63.5    | cat, guitar, rain             |
| 8    | F457    | +61.9    | coffee, rain, beach           |
| 9    | F39     | +60.6    | Universal in neutral prompt   |
| 10   | F5715   | +39.0    | "Surprise" — CONFOUND         |

F5715 (the "surprise" feature) was proven to be a detection-prompt confound — it
fires 0/300 times in neutral prompt. F870 and F7165 are interesting (only fire in
hits) but may also be prompt-related.

### 27B Model — Top Hit-Predictive Features

| Rank | Feature | diff     |
|------|---------|----------|
| 1    | F174    | +250.1   |
| 2    | F57     | +196.6   |
| 3    | F316    | +168.6   |
| 4    | F122    | +154.4   |
| 5    | F115    | +143.9   |
| 6    | F538    | +138.0   |
| 7    | F9294   | +129.0   |
| 8    | F397    | +115.4   |
| 9    | F531    | +99.6    |
| 10   | F2447   | +97.7    |

These are all features that fire more when the 27B model successfully identifies
the injected concept. But with only 6.7% hit rate (20/300), these could easily
be noise.

---

## 3. What We Know About Heads

From existing analysis:

**Head Patching (4B)**: Max recovery = 10% (L22 H5). Layer 14 = ZERO recovery
across all heads. No single head is responsible.

**Attribution (4B)**: Strongest heads are L15 H7 (mean=1.97), L15 H6 (1.71),
L15 H0 (1.54). But these are gradient magnitudes, not functional recovery.

**27B**: No head-level data — gradient-free method only gives layer-level scores.

**The problem**: We tried to find "the metacognition head" and it doesn't exist.
The signal is distributed across many heads with no single one responsible.

---

## 4. How to Find Heads That Activate for Metacognitive Tasks

Here's the core problem and a proposal:

### Why Previous Approaches Failed

We were looking for heads that process the INJECTION SIGNAL — but injection is
just perturbation. There's no dedicated "something is wrong" circuit because the
model doesn't have metacognition.

### New Approach: Differential Head Activation on Metacognitive vs Non-Metacognitive Prompts

Instead of injection, compare head activations between:
- **Metacognitive prompts**: "Am I thinking clearly?", "Was my reasoning correct?",
  "I'm not sure about my last answer", "Let me reconsider..."
- **Non-metacognitive control prompts**: "What is the capital of France?",
  "Explain photosynthesis", factual Q&A

The idea: if certain heads specialize in self-referential reasoning, they should
activate MORE for metacognitive prompts regardless of injection.

### Proposed Benchmark Design

**Step 1: Create a metacognitive prompt battery**

Categories of metacognitive prompts:
  a) Self-monitoring: "How confident am I in this answer?"
  b) Error detection: "Wait, I think I made a mistake"
  c) Uncertainty: "I'm not sure, let me reconsider"
  d) Self-reference: "What did I just say?", "Why did I say that?"
  e) Calibration: "On a scale of 1-10, how certain am I?"

Control prompts matched for length/complexity but non-metacognitive.

**Step 2: Record per-head activations**

For each prompt, record the pre-o_proj activations at every layer and head.
Compute the L2 norm of each head's activation vector.
This gives a [n_prompts × n_layers × n_heads] tensor of activation magnitudes.

**Step 3: Differential analysis**

For each head, compute:
  d = mean_activation(metacog_prompts) - mean_activation(control_prompts)

Heads with large positive d are "metacognition-preferring" heads.
Use statistical test (t-test or permutation test) to find significant ones.

**Step 4: Validate with injection**

Take the top metacognition-preferring heads and:
  a) Check if their activation changes correlate with Exp6 leak/awareness
  b) Try patching their activations during reflection phase
  c) See if ablating them reduces the "awareness" response rate

### Why This Might Work

The Exp6 GPT-5 results show a real behavioral signal: 78% of 27B responses show
"awareness" when asked about leaked concepts. Something in the model produces
self-referential language ("that was a glitch", "I don't know why I said that").
Even if this is trained behavior rather than genuine metacognition, specific heads
must be responsible for generating self-referential vs confabulatory language.

The benchmark doesn't need to find "genuine metacognition" — it just needs to
find heads that PREFER self-referential processing. That's tractable.

### What You'd Need

1. ~50 metacognitive prompts + ~50 matched controls
2. Record activations: ~100 forward passes per model (fast, no generation needed)
3. Statistical analysis: straightforward numpy/scipy
4. Validation with Exp6 data: correlate head activations with awareness grades

This could run in <1 hour on a single GPU.
