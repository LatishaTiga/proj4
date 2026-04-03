## QUESTION 1: *EditLens shifts from binary classification (human vs. AI-generated) to continuous scoring (0-1 scale representing degree of AI intervention). Why is continuous scoring more useful than binary classification for detecting AI text?*

Continuous scoring is more useful because AI text in practice is not purely human or AI-generated, but lies on a spectrum of edits. Binary classification fails due to label noise, mixed authorship, and loss of information, whereas continuous scoring preserves the degree of transformation.

### **1\. General ML Intuition: Why Binary Classification Fails**

**Assumption in Binary Classification**  
Binary classification assumes two separable distributions:![][image1]

This works only when:

* Data is cleanly separable  
* Labels are well-defined

However, modern AI-generated text does not satisfy these assumptions.

**Reality: Data Lies on a Continuum**  
In practice, text spans a continuous spectrum of transformations:

* purely human-written  
* lightly edited  
* paraphrased  
* heavily rewritten  
* fully AI-generated

Thus, instead of a discrete binary label space, the underlying data is more accurately modeled as a continuous range representing degrees of AI involvement.

Binary classification imposes:

* hard decision thresholds  
* loss of intermediate information  
* instability for samples near the decision boundary

**Key ML Issue: Label Noise**  
Binary labeling introduces inherent noise:

* Light Grammarly edit → labeled "AI"  
* Heavy rewrite → also labeled "AI"

These correspond to significantly different underlying distributions. As a result, the model learns a blurred decision boundary rather than meaningful structure.

**Regression (continuous scoring) fixes this**  
Regression allows the model to learn:

![][image2]

Benefits:

* Preserves relative ordering between samples  
* Captures gradual transitions in editing intensity  
* Enables downstream calibration

This is exactly what EditLens exploits.

### **2\. Paper Insight: AI Editing is a Latent Continuous Variable**

The paper explicitly reframes the task:  
“AI edits exist on a continuous spectrum…”

**Core Conceptual Shift**  
Instead of “Was this written by AI?”, they ask “What is the extent of AI-driven editing applied to this text?”

**Why Binary Fails (paper’s argument)**

(a) Most real usage is editing, not generation  
“about two-thirds of all writing messages ask ChatGPT to modify user text…”  
Thus, binary detectors are addressing a misaligned objective.

(b) Mixed authorship is fundamentally ambiguous

The paper distinguishes:

1. Heterogeneous (easy)  
   Human paragraph \+ AI paragraph → token-level attribution is feasible  
2. Homogeneous (hard, real-world)  
   “authorship is entangled by the editing process”

Example:  
AI paraphrases an entire passage, resulting in every sentence reflecting both human intent and AI transformation.

 In such cases, binary labeling is no longer well-defined

**Key insight:**  
Binary classification is ill-posed for homogeneous mixed text because:

* There is no clear ground-truth separation between “AI” and “human”  
* Only the magnitude of transformation is observable

**Their formalization**  
They define:

![][image3]

Where:

* ( x ): original human text  
* ( y ): edited text

And train:

![][image4]

At inference:

* Only ( y ) is observed

The model estimates:

![][image5]

This implies the model learns statistical regularities associated with editing intensity, rather than explicit authorship labels.

**Why this is powerful**

1. Captures nuance  
   “Was the text lightly edited... or completely rewritten?”  
   Binary → cannot represent this distinction  
   Continuous → directly encodes it  
2. Better alignment with human perception  
   Humans naturally assess relative degrees of editing  
   Model outputs correlate with these judgments  
3. Enables policy flexibility  
   Eg. “light AI editing is acceptable but fully AI-generated text is not”  
   Continuous scores allow:  
   * post-hoc threshold selection  
   * context-dependent policies  
4. Reduces false positives  
   Binary models often misclassify minor edits as AI-generated  
   Continuous scoring assigns low values to such cases, preserving distinction

### **3\. How the Code Implements Continuous Scoring**

(A) Training: discretized regression via buckets  
Instead of pure regression, they use:

result\["label"\] \= score\_to\_bucket(score, ...)

From preprocess.py:

def score\_to\_bucket(score, n\_buckets, lo\_threshold, hi\_threshold):

This maps:

* a continuous score in \[0,1\] → discrete bins

This approach provides:

* improved training stability  
* advantages of classification-based optimization

(B) But decoding recovers continuity  
In inference:

probs \= softmax(output.predictions, axis=1)  
bucket\_labels \= np.arange(n\_buckets)  
score\_preds \= (probs @ bucket\_labels) / (n\_buckets \- 1\)

Mathematically:

![][image6]

This corresponds to the expected value of the score under the predicted distribution.

**Why this is powerful**  
Instead of selecting a single class via argmax, the model computes a probability-weighted average, thereby reconstructing a smooth continuous estimate.

(C) This directly matches paper equation  
From the paper:

![][image7]

The implementation follows the same principle using discretized midpoints.

(D) Why not pure regression?  
They evaluate:

* direct regression  
* n-way classification with weighted decoding

“We experiment with directly training a regression head… and n-way classification… then using weighted-average decoding”

Classification offers:

* more stable gradients  
* better empirical optimization (cross-entropy often outperforming MSE)

(E) Final result: hybrid model

* Training: classification  
* Output: continuous score

This combines optimization stability with expressive outputs.

**Geometric intuition**  
In representation space:

* human-written text forms one region  
* AI-generated text forms another

Edits induce gradual transitions between these regions, forming trajectories rather than discrete jumps

Binary classification:

* approximates a separating boundary

EditLens:

* models position along the transformation trajectory

### **4\. Summary (key reasons continuous scoring is better)**

1. Real-world data is inherently continuous rather than discrete  
2. Binary labels introduce significant noise by collapsing distinct cases  
3. Homogeneous mixed text invalidates discrete authorship labels  
4. Continuous scoring preserves ordering and magnitude of edits  
5. Enables flexible, context-dependent decision thresholds  
6. Aligns with human perception of editing intensity  
7. Code reflects this via:  
   * bucketed training  
   * expected-value decoding producing smooth scores

**Final intuition**  
Binary detection asks “Is this AI?” EditLens asks “How much AI is present in this text?” — which more accurately reflects the underlying problem.

## QUESTION 2:

*EditLens computes continuous scores via weighted sum of bucket probabilities: (probs @ bucket\_labels) / (n\_buckets \- 1). Why normalize by dividing by (n\_buckets \- 1\) rather than just using the weighted sum directly?*

### **1\. The exact code we’re analyzing**

From `inference.py`:

probs \= softmax(output.predictions, axis=1)  
bucket\_labels \= np.arange(n\_buckets)  
score\_preds \= (probs @ bucket\_labels) / (n\_buckets \- 1\)

### **2\. First: what happens without normalization?**

Focus on:

![][image8]

This computes the expected bucket index. The issue is that its scale depends on *N*: the range is *\[0,N−1\]*, so the value is not comparable across models with different numbers of buckets.

### **3\. Why divide by (n\_buckets−1)?**

The normalization:

![][image9]

maps the expected bucket index to the interval *\[0,1\]*, since:

* minimum: *0/(N−1)=0*  
* maximum: *(N−1)/(N−1)=1*

### **4\. Direct connection to the paper**

The paper defines the target:

*Δ(x,y)∈\[0,1\]*

as a normalized similarity score representing edit magnitude. Since training uses bucketed versions of a *\[0,1\]* variable, inference must recover a value in the same space.

### **5\. Why this normalization is mathematically correct**

During training, the preprocessing maps:

*S∈ \[0,1\]→ j∈  {0,…,N−1} ,   j ≈ s⋅(N−1)*

Inverting:  
*S ≈ j /(N−1)*

With a distribution over buckets:

	![][image10]

This exactly matches the code:

(probs@bucket\_labels)/(nbuckets−1)

### **6\. Intuition: undoing discretization**

Training maps a continuous score to a discrete bucket index. Inference reverses this by converting a distribution over buckets back into a continuous score. Without division, the output remains in bucket index space; with division, it is mapped back to the original continuous scale.

### **7\. Why this matters practically**

**(A) Consistent interpretation**  
 Normalization ensures outputs lie in *\[0,1\]*, giving a fixed semantic meaning (e.g., low → human-like, high → heavily AI-edited), independent of the number of buckets.

**(B) Model-agnostic outputs**  
 The paper’s goal is a continuous score representing degree of AI editing. This requires outputs to be invariant to architectural choices like *N*.

**(C) Meaningful thresholding**  
 With normalized scores, thresholds are stable across models. Without normalization, thresholds would scale with NNN, making them inconsistent and difficult to interpret.

### **8\. Deeper ML perspective: expectation over latent variable**

The model predicts a distribution *p(j∣x)*, while the true latent variable is *s∈\[0,1\]*. The computation:

![][image11]

corresponds to estimating:

![][image12]

Thus, the model outputs expected edit intensity rather than the most likely bucket.

### **9\. Why not just use midpoints?**

The paper defines:

              				*ŝ=∑ p(j∣x)⋅mj*

where *mj* are bucket midpoints. The implementation approximates this by using indices *j*

and normalizing:

*j/(N−1)*

which is equivalent to evenly spaced midpoints in \[0,1\].

### **10\. What would go wrong without normalization?**

1. Loss of semantic meaning: outputs depend on arbitrary choice of N  
2. Inconsistent evaluation: metrics like MAE or correlation are not comparable  
3. Misalignment with training target: training uses \[0,1\], inference would produce \[0,N−1\]

### **11\. Final intuition (most important)**

The model predicts a distribution over edit levels. The weighted sum computes the average bucket index, and dividing by (N−1) converts it into the average edit intensity on a normalized \[0,1\] scale.

**Summary**  
Dividing by (n\_buckets−1) converts the expected bucket index into the original \[0,1\] edit-intensity scale, ensuring interpretability, consistency across models, and alignment with the training objective.

## QUESTION 3:

*EditLens supports three evaluation modes: human vs. AI, human vs. rest, AI vs. rest. Why evaluate multiple classification scenarios rather than just human vs. AI?*

### **1\. The fundamental problem: AI text is not just “AI vs human”**

The paper explicitly introduces three categories:

* human\_written  
* ai\_edited  
* ai\_generated

This already rejects the classic binary framing (human vs AI). Real-world text spans:

* Human: fully original writing  
* AI-edited: Grammarly / ChatGPT rewrite  
* AI-generated: full LLM output

**Key insight:** the important variable is the *degree of AI intervention*, not just its presence.  
 More precisely, this degree of AI involvement is a **latent variable**—it cannot be directly observed as a clean label, but must be inferred from the text.

### 

### **2\. What happens if you only evaluate “human vs AI”?**

Define:

* AI \= ai\_generated \+ ai\_edited

**Problem 1: Collapsing distinct phenomena**

* ai\_edited → subtle transformations  
* ai\_generated → fully synthetic  
   Binary evaluation treats them identically.

**Problem 2: Hiding model weaknesses**  
 A model may:

* perform well on ai\_generated  
* fail on ai\_edited

Yet still achieve high binary accuracy → misleading.

### **3\. EditLens evaluation modes \= probing different capabilities**

From `binary_eval.py`:

MODES \= ("human\_vs\_ai", "human\_vs\_rest", "ai\_vs\_rest")

Each mode isolates a distinct detection challenge, but more fundamentally represents a **different thresholding (projection) of the same continuous score** corresponding to the latent degree of AI editing.

### 

### **4\. Mode 1: human\_vs\_ai**

if mode \== "human\_vs\_ai":  
  val\_df \= val\_df\[val\_df\["label"\] \!= \-1\]

Removes ai\_edited → evaluates:

* human vs ai\_generated

**Tests:** ability to detect fully AI-generated text  
 **Property:** easiest case (strongest signal, most separable)  
 **Limitation:** ignores partial edits (core challenge of the paper)

### **5\. Mode 2: human\_vs\_rest**

elif mode \== "human\_vs\_rest":  
  return (raw\_labels \!= 0).astype(int)

Evaluates:

* human vs (ai\_edited \+ ai\_generated)

**Tests:** detection of *any* AI involvement  
 **Importance:** aligns with real-world need to detect intervention  
 **Difficulty:** harder due to overlap between human and lightly edited text

**Insight:** failure here implies inability to detect subtle edits

### **6\. Mode 3: ai\_vs\_rest**

elif mode \== "ai\_vs\_rest":  
  return (raw\_labels \== 1).astype(int)

Evaluates:

* ai\_generated vs (human \+ ai\_edited)

**Tests:** separation of fully AI-generated text from mixed or human text  
 **Insight:** probes whether the model captures *degree of AI involvement*

### 

### **7\. Why all three are necessary (deep intuition)**

Consider a continuous spectrum of edit intensity from human → edited → generated.  
 Each evaluation mode corresponds to a different decision boundary on this latent variable:

* human\_vs\_ai → distinguishes extreme ends  
* human\_vs\_rest → detects any deviation from purely human  
* ai\_vs\_rest → detects strongly AI-generated text

**Key idea:** a single binary task cannot adequately evaluate a continuous latent variable.

### **8\. How this connects to EditLens scoring**

From inference:

score\_preds \= (probs @ bucket\_labels) / (n\_buckets \- 1\)

This produces a continuous score in \[0,1\], estimating the latent degree of AI editing.

Binary evaluation applies thresholds:

threshold, val\_f1 \= find\_optimal\_threshold(val\_scaled, val\_binary)  
preds \= (test\_scaled \>= threshold).astype(int)

Different modes correspond to different thresholds:

* human\_vs\_rest → lower threshold  
* ai\_vs\_rest → higher threshold

**Insight:** the same underlying score yields different binary decisions depending on how it is thresholded.

### 

### 

### **9\. Ternary evaluation: fully embracing the paper**

From `ternary_eval.py`:

preds \= np.full(len(scores), 2\)  \# default: ai\_edited  
preds\[scores \< h\_thresh\] \= 0    \# human  
preds\[scores \> ai\_thresh\] \= 1   \# ai\_generated

This introduces two thresholds:

* low → human  
* mid → ai\_edited  
* high → ai\_generated

This maps the continuous latent variable into three semantic regions.

### **10\. Why multiple evaluation modes are necessary (ML perspective)**

1. **Different decision boundaries**  
    Binary classification corresponds to thresholding a continuous latent variable; different tasks require different thresholds.  
2. **Class overlap and separability**  
   * ai\_generated is highly separable  
   * ai\_edited overlaps with human  
      A single metric cannot capture both regimes.  
3. **Calibration quality**  
    A good model should:  
   * distinguish human vs edited  
   * distinguish edited vs generated  
      These are distinct subproblems.

### **11\. What would go wrong with only human\_vs\_ai?**

* **Overestimated performance**  
   Strong performance on ai\_generated masks failures on ai\_edited  
* **Missed core contribution**  
   The paper’s focus on AI editing would not be evaluated  
* **Loss of granularity**  
   Binary output cannot distinguish levels of editing  
* **Fundamental issue (paper insight)**  
   In homogeneous edited text, authorship is entangled; “AI vs human” is not just difficult—it is **undefined**, since no clean ground-truth label exists.

### 

### 

### **12\. Final intuition**

EditLens measures a continuous latent spectrum from human → edited → generated. Each evaluation mode corresponds to a different question:

* human\_vs\_ai → distinguish extremes  
* human\_vs\_rest → detect any AI influence  
* ai\_vs\_rest → detect fully AI-generated text

**Core insight:** multiple evaluation modes are required because each is a different projection of the same latent variable, probing different decision boundaries that a single human-vs-AI evaluation cannot capture.

### **Summary**

EditLens evaluates multiple classification scenarios because AI text lies on a continuum of latent editing intensity, and each mode corresponds to a different threshold-based projection of this variable (detecting any AI, distinguishing fully AI-generated text, or separating extremes), which a single human-vs-AI evaluation cannot capture.

## QUESTION 4: 

*EditLens finds the optimal threshold on the validation set by maximizing F1, then evaluates on the test set. Why is this two-stage approach necessary, and what pathology would occur if you optimized thresholds on the test set?*

### **1\. What the code is doing (mechanically)**

**Stage 1: Find threshold on validation set**  
From binary\_eval.py:

threshold, val\_f1 \= find\_optimal\_threshold(val\_scaled, val\_binary)

Inside threshold.py:

thresholds \= np.linspace(0, 1, num\_thresholds)

for threshold in thresholds:  
  pred\_labels \= (preds \>= threshold).astype(int)  
  ...  
  f1 \= ...  
  if f1 \> best\_f1:  
      best\_threshold \= threshold

This performs:

* search over candidate thresholds  
* selection of the threshold that maximizes F1

**Stage 2: Apply to test set**

preds \= (test\_scaled \>= threshold).astype(int)  
metrics \= evaluate(test\_binary, preds)

Important:

* no tuning occurs on the test set

### **2\. Why do we need two stages?**

Because **threshold selection is itself a learned parameter**.

Even with fixed model weights, the decision rule:

* score ≥ threshold → AI

is adjustable.

So effectively:

* model weights → learned on training set  
* threshold → learned on validation set  
* final metric → evaluated on test set

This follows the standard pipeline:  
 **train → validate → test**

### **3\. Why threshold tuning is necessary (EditLens-specific)**

EditLens outputs:

* score ∈ \[0,1\] representing degree of AI intervention

This score is a **continuous latent variable capturing edit intensity**, and the threshold acts as a **decision boundary in this edit-intensity space**.

Different tasks require different thresholds:

* human\_vs\_rest → lower threshold  
* ai\_vs\_rest → higher threshold

There is no universal threshold, so it must be estimated from data:

threshold, val\_f1 \= find\_optimal\_threshold(val\_scaled, val\_binary)

### **4\. Why NOT use the test set?**

If threshold is optimized on the test set:

* threshold \= argmax F1(test\_scores, test\_labels)

This leads to **data leakage and overfitting to the test set**.

Selecting a threshold from many candidates on the test set introduces selection bias, leading to an optimistically biased estimate of performance.

### **5\. Intuition: what goes wrong?**

The test set represents unseen data.  
 Tuning on it means using its labels to adjust the model.

From the search loop:

for threshold in thresholds:  
  ...  
  if f1 \> best\_f1:  
      best\_threshold \= threshold

This selects a threshold that:

* fits the specific test distribution  
* exploits noise or sampling artifacts  
* may not generalize

### **6\. Concrete pathology: overfitting via threshold**

Example:

| Threshold | True general F1 | Test F1 |
| ----- | ----- | ----- |
| 0.45 | 0.80 | 0.79 |
| 0.51 | 0.79 | 0.83 ← selected |

The chosen threshold overfits to test-specific fluctuations.

Result:

* reported performance is inflated  
* real-world performance is worse

### 

### **7\. Why this is especially dangerous in EditLens**

Threshold defines the decision boundary on a continuous latent score:

pred\_labels \= (preds \>= threshold).astype(int)

Because the score reflects edit intensity:

* small threshold changes shift the boundary in edit space  
* many samples (especially ai\_edited) lie near this boundary

So:

* small shifts → many label flips  
* large changes in F1

This makes overfitting via threshold selection particularly easy.

### **8\. What the validation set is doing**

The validation set acts as a proxy for unseen data.

Selecting:

* threshold \= argmax F1(validation)

means:

* choosing a threshold expected to generalize

The test set is then used only for unbiased evaluation.

### **9\. Connection to ternary evaluation**

From `ternary_eval.py`:

h\_thresh, h\_f1 \= find\_optimal\_threshold(...)  
ai\_thresh, ai\_f1 \= find\_optimal\_threshold(...)

Two thresholds are learned on validation, then applied:

preds \= predict\_ternary(test\_scaled, h\_thresh, ai\_thresh)

Same principle:

* thresholds learned on validation  
* test used only for evaluation

### **10\. Deep ML perspective**

Threshold tuning is equivalent to:

* learning a 1D decision boundary over a continuous latent variable

If done on the test set:

* the model is effectively adapted to that specific sample of the latent distribution

This violates the core assumption:

* the test set must remain untouched

### **11\. Subtle but important detail in code**

Normalization uses validation statistics:

val\_min, val\_max \= val\_scores.min(), val\_scores.max()  
val\_scaled \= (val\_scores \- val\_min) / (val\_max \- val\_min)  
test\_scaled \= (test\_scores \- val\_min) / (val\_max \- val\_min)

Test distribution is not used.

Reason:

* using test statistics introduces another form of leakage

### 

### **12\. Final intuition (most important)**

Pipeline:

* model → score → threshold → prediction

The threshold is part of the model.

Therefore:

* it must be learned without using test data

### **13\. Final one-line answer**

The two-stage approach is necessary because threshold selection defines a decision boundary over a continuous edit-intensity score and must be calibrated on validation data to generalize; optimizing it on the test set would cause data leakage, overfitting to test-specific noise, and inflated performance that does not reflect real-world behavior.

## QUESTION 5:

*EditLens applies score orientation correction to ensure higher scores indicate AI-generated content. Why is this normalization necessary, and what would happen without it?*

### **1\. Where scores come from in EditLens**

From `inference.py`:

output \= trainer.predict(ds\_tokenized)  
probs \= softmax(output.predictions, axis=1)

The model outputs logits, which are converted to probabilities over buckets.

Then:

bucket\_labels \= np.arange(n\_buckets)  
score\_preds \= (probs @ bucket\_labels) / (n\_buckets \- 1\)

This produces a continuous score.

### **2\. EditLens itself is internally consistent**

Within EditLens:

* Buckets are ordered from 0 (human) to N−1 (AI)  
* The score is computed as:  
  * score \= E\[j\] / (N−1)

Thus:

* higher score indicates more AI involvement  
* the mapping is monotonic and consistent

### **3\. Source of inversion**

Inversion does not originate from EditLens, but from external detectors.

From evaluation:

score\_cols \= \[c for c in val\_df.columns if c.endswith("\_score")\]

The system evaluates multiple detectors, including:

* EditLens  
* Perplexity-based models  
* Binoculars  
* N-gram / stylometric methods

### **4\. Detectors are not standardized**

Different detectors assign opposite semantics to scores:

**AI-likelihood scores**

* higher → more AI  
* aligned with EditLens

**Human-likelihood scores**

* higher → more human  
* inverted relative to EditLens

Example: Perplexity

* low perplexity → AI-like  
* high perplexity → human-like  
* thus higher score corresponds to more human content

### **5\. Preprocessing does not resolve orientation**

The function:

* score\_to\_bucket(score) maps s ∈ \[0,1\] → bucket j

This applies only to EditLens training and does not affect:

* external detectors  
* their score definitions

| Component | Controls direction? |
| ----- | ----- |
| preprocess.py | No |
| EditLens model | Yes |
| external detectors | Unknown |
| evaluation code | Yes |

### **6\. Orientation is handled during evaluation**

From `binary_eval.py`:

val\_scores, flipped \= orient\_scores(val\_scores, val\_binary)

Pipeline:

* model → score column → evaluation → orientation correction

Orientation is therefore applied post hoc, not during preprocessing.

### **7\. Interpretation: coordinate alignment**

All detectors attempt to measure the same underlying quantity—degree of AI involvement—but encode it differently.

All detectors can be viewed as producing monotonic but differently oriented projections of the same latent variable—the degree of AI editing—making orientation necessary to align them.

This assumes detectors are monotonic functions of edit intensity, even if differently scaled or oriented.

Thus, orientation ensures a shared semantic interpretation: higher score corresponds to more AI.

## 

## 

## 

### **8\. Concrete example**

| text | true label | score |
| ----- | ----- | ----- |
| human | 0 | 0.9 |
| AI | 1 | 0.2 |

Here, higher scores indicate human-likeness.

Evaluation computes:

* human\_mean \= 0.9  
* ai\_mean \= 0.2

Since human\_mean \> ai\_mean, scores are flipped:

scores \= \-scores

After flipping:

* higher scores correspond to AI content

### **9\. Why correction is applied at evaluation time**

Standardizing upstream would require:

* explicit knowledge of each detector’s semantics  
* custom handling per model

Instead, evaluation applies:

* automatic, data-driven orientation using label statistics

### **10\. Calibration insight**

The condition:

if human\_mean \> ai\_mean:

uses labels to infer score direction.

This functions as a form of post-hoc calibration, aligning uncalibrated scores before thresholding.

### **11\. Role of multiple detectors**

The system evaluates multiple models side-by-side:

* Perplexity-based: inverted orientation  
* Binoculars: variable semantics  
* N-gram / stylometric: often human-aligned  
* Classifier baselines: typically AI-aligned

Each produces a scalar score with different semantics.

### **12\. Motivation from the paper**

The paper evaluates multiple detectors to demonstrate that:

* traditional methods assume binary human vs AI separation  
* real-world text lies on a continuum (human → edited → AI)  
* in homogeneous edited text, authorship is entangled, making “AI vs human” not just difficult but fundamentally ill-defined

Failure modes:

| Detector | Strength | Weakness |
| :---- | ----- | ----- |
| Perplexity | strong on pure AI | fails on edits |
| Classifiers | good separation | collapse on edits |
| N-gram | stylistic cues | shallow |

### **13\. Unified evaluation pipeline**

For each detector:

* orientation correction  
* scaling  
* threshold tuning  
* evaluation

This ensures comparability across models.

### **14\. Consequences without orientation**

Evaluation assumes:

preds \= (score \>= threshold)

Without correcting orientation:

* predictions are inverted  
* thresholds are misapplied  
* metrics become invalid

This misalignment would affect evaluation modes differently:

* human\_vs\_rest thresholds would shift incorrectly toward human-like regions  
* ai\_vs\_rest thresholds would shift incorrectly toward AI-like regions

### **15\. Representation alignment**

Detectors differ in:

* scale  
* direction

Evaluation aligns them by:

* correcting direction (flip)  
* normalizing scale  
* applying consistent thresholds

This alignment enables recovering a meaningful position on the continuous edit spectrum, not just fixing direction.

Without this alignment, compari

sons are not meaningful.

### **16\. Final mental model**

| Model | Scale | Direction |
| ----- | ----- | ----- |
| EditLens | \[0,1\] | AI ↑ |
| Perplexity | unbounded | Human ↑ |
| Binoculars | arbitrary | unknown |
| N-gram | arbitrary | human ↑ |

The evaluation pipeline enforces a shared representation before comparison.

### **Final Answer**

Score orientation correction is necessary because different detectors produce scores with inconsistent semantics—some increase with AI likelihood while others increase with human-likeness. Since evaluation assumes a unified interpretation in which higher scores indicate more AI involvement, orientation aligns all models to a common semantic axis. This alignment recovers a consistent position on the underlying edit-intensity spectrum and enables meaningful thresholding. Without this step, thresholds would be misapplied, predictions inverted, evaluation modes would behave incorrectly, and comparisons across models would be invalid and misleading.

##  

## QUESTION 6:

*EditLens includes min-max scaling of scores using validation data ranges. Why scale using validation statistics rather than leaving raw model outputs?*

### **1\. What the code is doing**

From `binary_eval.py`:

\# Min-max scale using val's range, apply same to test  
val\_min, val\_max \= val\_scores.min(), val\_scores.max()

val\_scaled \= (val\_scores \- val\_min) / (val\_max \- val\_min)

Then for test:

test\_scaled \= (test\_scores \- val\_min) / (val\_max \- val\_min)

Key detail:

* Scaling parameters (min, max) are computed only on validation data and reused for test data

### **2\. Raw scores are not comparable across models**

From:

score\_cols \= \[c for c in val\_df.columns if c.endswith("\_score")\]

The system evaluates multiple detectors, such as:

* EditLens  
* Binoculars  
* Perplexity-based detectors  
* N-gram models

Raw scores exist in fundamentally different spaces:

| Model | Score Range | Meaning |
| ----- | ----- | ----- |
| EditLens | 0 → 1 | probability-like |
| Perplexity | 5 → 200 | lower \= more AI |
| Binoculars | \-10 → 10 | arbitrary logit-like |

Without scaling:

* A threshold like 0.5 is meaningful for EditLens  
* The same threshold is meaningless for perplexity or other detectors

### **3\. Role of min-max scaling**

Min-max scaling:

s′=(s−min⁡) / (max⁡−min⁡)

This transforms all scores into a shared \[0,1\] space:

* 0 → most human-like (relative to validation set)  
* 1 → most AI-like (relative to validation set)

### **4\. Why use validation statistics**

#### **4.1 Prevent test leakage**

If test statistics were used:

test\_scaled \= (test\_scores \- test\_min) / (test\_max \- test\_min)

This would:

* use information from the test distribution  
* violate the principle that test data must remain unseen

Result:

* scaling adapts to test distribution  
* scores become artificially well-conditioned  
* threshold performance may improve due to leakage

This constitutes a subtle but real form of data leakage.

#### **4.2 Ensure consistency between validation and test**

Pipeline:

* threshold is learned on val\_scaled  
* applied to test\_scaled

This requires both to be in the same coordinate system.

If test were scaled independently:

* validation and test would have different min/max ranges  
* a threshold like 0.6 would correspond to different regions

Result:

* thresholds lose semantic meaning  
* calibration becomes inconsistent

#### **4.3 Threshold is defined in validation space**

From code:

threshold, val\_f1 \= find\_optimal\_threshold(val\_scaled, val\_binary)

The threshold is learned relative to validation scaling.

Therefore:

* test data must be mapped into the same space before applying it

### **5\. Deeper interpretation: normalization and calibration**

This step serves two roles:

#### **(A) Feature normalization**

* ensures comparable numerical ranges  
* enables detector-agnostic evaluation

#### **(B) Score calibration**

* assigns consistent meaning to scores

After scaling:

* 0.2 → relatively human-like (based on validation)  
* 0.8 → relatively AI-like

Without scaling:

* scores are arbitrary magnitudes with no shared interpretation


### **6\. What breaks without scaling**

#### **6.1 Threshold search becomes invalid**

From:

thresholds \= np.linspace(0, 1, num\_thresholds)

This assumes scores lie in \[0,1\].

Without scaling:

* ranges like \[5, 200\] or \[-10, 10\] make threshold search meaningless

#### **6.2 Cross-model comparison fails**

* A threshold of 0.6 has different meanings across models  
* evaluation becomes inconsistent

#### **6.3 Ternary evaluation breaks**

From `ternary_eval.py`:

preds\[scaled\_scores \< h\_thresh\] \= 0  
preds\[scaled\_scores \> ai\_thresh\] \= 1

This assumes:

* low \= human  
* high \= AI  
* middle \= edited

Without scaling:

* these regions are not comparable across detectors

### **7\. Why min-max instead of mean/std**

Min-max is chosen because:

* it preserves ranking structure  
* thresholding depends on ordering, not absolute values  
* F1 optimization relies on rank consistency  
* it provides interpretable bounds (0 \= min, 1 \= max)  
* it aligns with bounded threshold search

### 

### **8\. Conceptual connection to the research paper**

The design reflects a core evaluation principle:

* detectors produce heterogeneous score distributions  
* these must be normalized before comparison  
* threshold-based evaluation requires a shared score space

EditLens enforces this by:

* treating detectors as black boxes  
* applying normalization during evaluation

### **9\. Pipeline view**

Full pipeline:

raw model scores  
 → orientation correction  
 → min-max scaling (using validation statistics)  
 → threshold search (on validation)  
 → apply same scaling and threshold to test

### **10\. Final intuition**

Each detector produces scores in its own scale:

* probabilities  
* perplexities  
* logits

Min-max scaling using validation data converts these into a common reference scale:

* a normalized ranking from 0 to 1  
* anchored to a fixed dataset

### **Final Answer**

Min-max scaling using validation statistics is necessary because different detectors produce scores with incompatible ranges and semantics. Scaling maps all outputs into a shared coordinate system, allowing thresholds learned on validation data to be meaningfully applied to test data. Using validation statistics prevents data leakage, ensures consistency between validation and test distributions, and preserves the interpretability of thresholds. Without this step, threshold selection would be invalid, predictions inconsistent, and comparisons across models unreliable.

## QUESTION 7:

*EditLens trains RoBERTa-Large and Llama-3.2-3B separately rather than finding a single best model. Why support multiple model sizes, and what trade-offs exist?*

### **1\. What EditLens is doing (core idea)**

EditLens is not just a classifier; it is a calibrated scoring system over a spectrum of AI involvement. Instead of binary detection, outputs are mapped to a continuous notion of “AI-ness.”

This is implemented via bucketed classification followed by a continuous score:

score\_preds \= (probs @ bucket\_labels) / (n\_buckets \- 1)

The model learns an ordered representation of AI involvement rather than a hard label.

### **2\. Why not just pick one “best” model?**

In standard ML, one would train multiple models and select the best. EditLens instead supports multiple architectures:

* RoBERTa-Large (encoder)  
* Llama-3.2-3B (decoder LLM)

The goal is not a single optimal detector, but a robust, model-agnostic evaluation framework.

### **3\. Core reason: different inductive biases**

Different architectures capture different signals:

| Model type | Strength |
| ----- | ----- |
| Encoder (RoBERTa) | local patterns, classification boundaries |
| Decoder (Llama) | generative structure, long-range coherence |

**RoBERTa-Large**

* Strong at stylistic cues and local token patterns  
* Produces sharp decision boundaries  
* Effective for clear human vs AI separation

**Llama-3.2-3B**

* Strong at modeling generative processes and global coherence  
* Captures subtle AI artifacts and editing traces  
* Better for nuanced, partially edited text

AI detection involves multiple signals (fluency, entropy, consistency, artifacts), and no single model captures all of them fully.

### **4\. Evidence in the code: model-specific adaptation**

model \= AutoModelForSequenceClassification.from\_pretrained(  
  cfg.model.name,  
  num\_labels=cfg.data.n\_buckets,  
)

A shared pipeline is used, with model-specific adjustments.

For decoder LLMs:

if hasattr(model, "score") and isinstance(model.score, torch.nn.Linear):  
  model.score \= NormedLinear(hidden\_size, cfg.data.n\_buckets)

class NormedLinear(torch.nn.Module):  
  """Linear layer preceded by LayerNorm to keep logits well-scaled."""

This reflects architectural differences: LLMs require stabilization for classification-style outputs.

### **5\. Training constraints and practical trade-offs**

if cfg.get("quantization", {}).get("enabled", False):  
model \= prepare\_model\_for\_kbit\_training(model)

**Llama requires:**

* QLoRA  
* 4-bit quantization  
* LoRA adapters

**RoBERTa:**

* Fully fine-tuned directly  
* No special memory optimizations needed

### **6\. Trade-offs between model sizes**

This difference is partly architectural:

* **RoBERTa (encoder, bidirectional)** → sees full context → better at classification boundaries and local consistency  
* **Llama (decoder, autoregressive)** → predicts next token → better at modeling generative structure and fluency

**RoBERTa-Large**

Pros:

* Fast training and inference  
* Low memory usage  
* Stable optimization  
* Strong local consistency and clear decision boundaries

Cons:

* Limited expressivity  
* May miss subtle or long-range patterns

**Llama-3.2-3B**

Pros:

* Strong semantic and generative understanding  
* Captures nuanced edits and artifacts  
* Better modeling of global structure

Cons:

* Slow inference  
* High compute requirements  
* Requires quantization and tuning tricks  
* Less stable during training

### **7\. Why EditLens supports both**

EditLens is designed to compare and evaluate detectors across models rather than collapse them into one.

score\_cols \= \[c for c in val\_df.columns if c.endswith("\_score")\]  
bucket\_col \= f"editlens\_{base\_model\_tag}\_bucket"  
score\_col \= f"editlens\_{base\_model\_tag}\_score"

Outputs are tracked per model (e.g., RoBERTa vs Llama), enabling direct comparison.

**System-level trade-off:**

A subtle trade-off of supporting multiple models is that they may produce differently calibrated score distributions, meaning the same score or threshold may not correspond to identical semantic meaning across models, especially under distribution shift.

### **8\. Deeper ML intuition: bias–variance and deployment**

| Model | Bias | Variance |
| ----- | ----- | ----- |
| RoBERTa | higher | lower |
| Llama | lower | higher |

This yields a complementary trade-off:

* RoBERTa → stability and efficiency  
* Llama → flexibility and expressivity

Deployment:

* real-time / low-resource → RoBERTa  
* high-accuracy / analysis → Llama

### **9\. Conceptual role in EditLens**

The framework emphasizes:

* continuous, calibrated scoring  
* model-agnostic evaluation  
* comparability across detectors

Supporting multiple models:

* avoids overfitting to one architecture  
* enables cross-model comparison  
* captures diverse detection signals

### **10\. How model behavior differs in scoring**

Both models output probabilities over buckets, averaged into a final score:

probs \= softmax(output.predictions, axis=1)  
score\_preds \= (probs @ bucket\_labels) / (n\_buckets \- 1\)

* RoBERTa concentrates probability in one bucket → discrete, step-like scores  
* Llama spreads probability across buckets → smoother, continuous scores

A peaked distribution yields coarse outputs; a spread distribution yields fine-grained values.

### **11\. Why this matters**

Because EditLens models a continuous notion of AI involvement:

* RoBERTa → stronger classification, coarser ranking  
* Llama → better ranking and calibration across intermediate cases

This reflects a trade-off between decisiveness and resolution.

### **12\. Final takeaway**

EditLens supports multiple model sizes because:

1. Different models capture different aspects of AI-generated text  
   * RoBERTa → local, stylistic signals  
   * Llama → global, generative behavior  
2. Practical constraints differ  
   * RoBERTa → efficient and stable  
   * Llama → powerful but resource-intensive  
3. The system is designed for comparison and calibration across models, not single-model optimization

**Trade-offs:**

* efficiency vs expressivity  
* stability vs flexibility  
* deployment feasibility vs analytical depth

Multiple models provide complementary views of the same underlying variable—degree of AI involvement—while introducing calibration and consistency challenges across score distributions.

## QUESTION 8:

*EditLens performs three distinct classification tasks (human vs. AI, human vs. rest, AI vs. rest) with per-task optimization (finding thresholds). Why not train one model for all three scenarios simultaneously?*

### **1\. Training Objective**

From `train.py`:

result\["label"\] \= score\_to\_bucket(  
 score, cfg.data.n\_buckets, cfg.data.lo\_threshold, cfg.data.hi\_threshold  
)

The model is trained as a multiclass classifier over buckets, not directly for:

* human vs AI  
* human vs rest  
* AI vs rest

Instead, it learns a fine-grained spectrum of “edit intensity”:

| Bucket | Meaning |
| ----- | ----- |
| 0 | human |
| 3 | lightly edited |
| 6 | heavily edited |
| 9 | fully AI |

The bucket labels are not arbitrary; they form an ordered structure (increasing AI involvement). The model therefore learns a monotonic, structured representation of edit intensity rather than independent class labels.

So the model learns:  
 “Where does this text lie on a continuum from human → AI?”

### **2\. Model Outputs**

From `inference.py`:  
probs \= softmax(output.predictions, axis=1)  
score\_preds \= (probs @ bucket\_labels) / (n\_buckets \- 1\)

The output is:

* a distribution over buckets  
* converted into a continuous score ∈ \[0,1\]

This produces a single latent variable (degree of AI involvement), not a task-specific decision.

### **3\. Evaluation Setup and Modes**

From `binary_eval.py`:  
threshold, val\_f1 \= find\_optimal\_threshold(val\_scaled, val\_binary)  
preds \= (test\_scaled \>= threshold).astype(int)

And:

MODES \= ("human\_vs\_ai", "human\_vs\_rest", "ai\_vs\_rest")

Evaluation defines three tasks:

* human vs AI → ignore edited text  
* human vs rest → edited counts as AI-like  
* AI vs rest → edited counts as human-like

Tasks are not learned, they are defined post hoc via thresholds on this shared latent variable.

### **4\. Key Conflict: Incompatible Learning Objectives**

The same sample can receive different labels depending on the task.

Example: AI-edited text

| Task | Label |
| ----- | ----- |
| human vs AI | ignored |
| human vs rest | AI (1) |
| AI vs rest | NOT AI (0) |

This creates contradictory supervision if trained jointly:

* positive in one task  
* negative in another

The three tasks are incompatible as learning objectives but compatible as decision rules over a shared latent variable.

Joint training would force inconsistent label mappings for the same underlying variable, leading to:

* unstable gradients  
* conflicting decision boundaries  
* degraded performance

### **5\. What EditLens Does Instead**

EditLens learns a task-agnostic representation and defers decisions to evaluation:

* Learn structure: scores → buckets → ordered representation  
* Convert to score:

   score\_preds \= (probs @ bucket\_labels) / (n\_buckets \- 1\)

* Apply task-specific thresholds:

   threshold \= find\_optimal\_threshold(val\_scaled, val\_binary)

This yields one shared axis with multiple decision rules.

### **6\. Intuition: One Representation, Many Decisions**

The model learns a general scale, while tasks apply different thresholds:

* high threshold → strict AI detection  
* low threshold → sensitive detection  
* ranges → multi-class decisions

A single underlying variable supports multiple decision boundaries.

### **7\. Advantages of This Design**

* **Avoids conflicting supervision**  
   Learns a consistent continuum: human → edited → AI  
* **Preserves structure**  
   Avoids collapsing edited \+ AI into one class  
    
* **Flexible deployment**

  threshold \= 0.8  \# strict  
  threshold \= 0.5  \# sensitive

* **Per-task calibration**  
   Each task independently optimizes its threshold  
* **Decoupling of learning and evaluation**  
   Representation learning (continuous score) is separated from evaluation (task-specific thresholds), enabling reuse across tasks and datasets

### **8\. Drawbacks of a Multi-Task Approach**

A shared encoder with multiple heads introduces:

* label inconsistency → gradient conflict  
* inefficient learning → multiple boundaries instead of one structure  
* poorer generalization → task-specific heads don’t transfer well

### **9\. Core Design Insight**

EditLens separates:

* **Representation learning**

   num\_labels \= cfg.data.n\_buckets

   → learn ordered structure of AI involvement

* **Decision making**

   threshold \= find\_optimal\_threshold(...)

   → adapt to task

Learning the latent variable first avoids contradictions while preserving flexibility.

### 

### **10\. Final Takeaway**

EditLens does not train separate models for each task because:

* Tasks have conflicting labels and cannot be jointly optimized  
* The problem is fundamentally ordered, not binary  
* A single latent variable is more expressive and reusable  
* Thresholding enables flexible, per-task optimization

In essence:  
 The model learns the underlying variable (degree of AI involvement), and the three tasks are different thresholdings of that variable, not separate learning problems.

## QUESTION 9:

*How does computing continuous scores via bucket probability weighting avoid the need for explicit regression outputs, and why is this better than directly predicting a continuous score?*

### **1\. Core Idea (High-Level)**

The goal is to predict a continuous score representing the amount of AI editing. Instead of directly regressing a scalar, the method converts the problem into classification over discrete buckets and then recovers a continuous score via probability-weighted averaging. This turns regression into probabilistic classification followed by an expectation. The method is not just convenient—it is aligned with the Bayes-optimal solution.

### **2\. What the Code Does**

During training, continuous scores are discretized into buckets:

def score\_to\_bucket(score, n\_buckets, lo\_threshold, hi\_threshold):  
   return 1 \+ int(normalized \* (n\_buckets \- 2))

So a score in \[0,1\] becomes one of N ordered classes. The model is then trained as a classifier:

model \= AutoModelForSequenceClassification(  
   cfg.model.name,  
   num\_labels=cfg.data.n\_buckets,  
)

At inference, logits are converted to probabilities and then to a continuous score:

probs \= softmax(output.predictions, axis=1)  
score\_preds \= (probs @ bucket\_labels) / (n\_buckets \- 1\)

This computes:

ŝ \= (∑ p(j|x) · j) / (N − 1\)

which is the expected bucket index. The final score is the expectation over bucket midpoints, making this a discretized approximation of the Bayes-optimal predictor E\[edit magnitude | x\].

The method introduces a discretization trade-off: more buckets improve precision and better approximate the continuous score, while fewer buckets make learning more stable and robust but reduce resolution**.**

### **3\. Why This Avoids Explicit Regression**

Instead of learning a direct mapping f(x) → scalar, the model learns f(x) → p(bucket | x). The final score is computed as:

ŝ \= E\[bucket | x\]

So the model predicts a distribution over plausible values and averages them, rather than committing to a single scalar.

### **4\. Intuition: Why This Works**

Direct regression forces the model to output a precise value (e.g., 0.63) with no structure. In contrast, classification provides an ordered space (human → edited → AI) and allows uncertainty to be expressed via probability mass across buckets. The expectation then naturally interpolates between levels, which is critical because the task is inherently ambiguous.

### **5\. Why This Is Better Than Direct Regression**

#### **5.1. Optimization & Robustness**

* **Better optimization:** Cross-entropy provides stronger, more stable gradients than MSE.  
* **Robust to noisy supervision:** Bucketing reduces sensitivity to small label noise and emphasizes coarse structure.

#### **5.2. Representation & Expressiveness**

* **Captures ordinal structure:** Buckets encode the ordered nature of editing intensity, unlike scalar regression.  
* **Represents uncertainty explicitly:** Distributions capture ambiguity (e.g., partially edited text).  
* **Implicit mixture modeling:** The expectation acts as a weighted mixture over editing levels.

#### **5.3. Output Quality & Practical Use**

* **Produces smooth, calibrated scores:** Softmax \+ expectation yields continuous outputs. As noted in the research, *“Binary classifiers predict scores clustered near 0 or 1, while EditLens produces a nuanced distribution tracking increasing levels of AI polish.”*  
* **More reliable thresholding:** Scores vary smoothly and preserve ordering, making threshold-based decisions more stable and interpretable.  
* **Better calibration:** Scores derived from a probability distribution better reflect underlying likelihoods and are more comparable across examples.

#### **5.4. Practical Considerations**

* **Easier scaling with transformers:** Classification heads are more stable than regression heads.  
* **Empirical support:** The paper shows classification \+ weighted averaging is more stable and expressive than direct regression.

### **6\. Deeper ML View: Learning a Distribution**

The optimal predictor is the conditional expectation E\[Δ | x\]. Instead of learning this expectation directly, the model learns the full distribution p(bucket | x) and computes the expectation analytically, whereas regression tries to learn the expectation in one step, which is harder and less expressive. Bucket probability weighting replaces direct regression by learning the conditional distribution and computing its expectation, which is both easier to optimize and better aligned with the uncertainty in the data.

### **7\. Key Conceptual Insight**

Regression predicts a single value directly, while this method predicts a distribution over plausible values and averages them. This decomposes a harder problem into a more learnable one.

### **8\. Why This Matters for This Task**

The task is inherently fuzzy and ambiguous. The same text can correspond to multiple plausible editing levels, and even human annotations show only moderate agreement. A distribution-based approach better reflects this uncertainty than forcing a precise scalar prediction.

### **9\. Concrete Example**

If a text could plausibly be 50% lightly edited (0.4) and 50% heavily edited (0.8), the correct prediction is 0.6. A regression model must output 0.6 directly. The bucket approach assigns probability mass and computes:

0.5 · 0.4 \+ 0.5 · 0.8 \= 0.6

This explicitly models uncertainty before resolving it via expectation.

### **10\. Final Summary**

Mechanism:

* Discretize score into buckets  
* Train a classifier  
* Predict probabilities  
* Compute expectation

Why it avoids regression:

* Does not directly predict a scalar  
* Uses a probability distribution and computes the score afterward

Why it is better:

* More stable optimization  
* Robust to noisy labels  
* Preserves ordinal structure  
* Captures uncertainty  
* Produces smooth, calibrated scores  
* Supports reliable thresholding and calibration  
* Approximates the Bayes-optimal predictor

In essence, the model learns a distribution over editing levels and computes its expectation, achieving the same goal as regression in a way that is more stable, expressive, and aligned with the structure and uncertainty of the problem.

## 

## QUESTION 10:

*EditLens uses bucket-based classification (predicting which bucket, then converting to continuous score) rather than direct regression. What happens at bucket boundaries (e.g., between human and AI buckets), and could this create adversarial opportunities?*

### **1\. What EditLens is doing**

EditLens takes the approach of discretizing similarity scores into N buckets and, during inference, reconstructing a continuous score using a weighted average over those buckets.

Mathematically:

![][image13]

Using bucket midpoints 𝑚𝑗​, the expected score becomes a true discretized approximation of the continuous signal. In the implementation, these midpoints are approximated using normalized bucket indices 𝑗 /  (𝑛−1), which is equivalent when buckets are uniformly spaced.

From `inference.py`:

probs \= softmax(output.predictions, axis=1)  
bucket\_labels \= np.arange(n\_buckets)  
score\_preds \= (probs @ bucket\_labels) / (n\_buckets \- 1\)

So the model predicts a distribution over buckets, and the final score is computed as a weighted average.

### 

### **2\. Where bucket boundaries come from**

From the research paper — *EditLens: Quantifying the Extent of AI Editing in Text*:

![][image14]

From the bucket-mapping implementation in `preprocess.py`:

if score \<= lo\_threshold:  
 return 0  
elif score \>= hi\_threshold:  
 return n\_buckets \- 1  
else:  
 normalized \= (score \- lo\_threshold) / (hi\_threshold \- lo\_threshold)  
 return 1 \+ int(normalized \* (n\_buckets \- 2))

This creates hard partitions of the continuous score space.

### **3\. What happens at bucket boundaries**

A key fact is that labels are discontinuous, while the underlying phenomenon is continuous.

Example:  
Suppose:

* lo\_threshold \= 0.2  
* hi\_threshold \= 0.8  
* n\_buckets \= 5

Then:

* score \= 0.39 → bucket 2  
* score \= 0.41 → bucket 3

Even though 0.39 and 0.41 are very similar, they receive different labels. This creates artificial discontinuities: a tiny change in score leads to a different class, and during training these are treated as categorically different.

### **4\. How the model compensates (important insight)**

Even though labels are discrete, the model outputs probabilities rather than hard assignments.

Near boundaries:

* P(bucket 2\) ≈ 0.5  
* P(bucket 3\) ≈ 0.5

Then:

**In theory:**

Score ≈  0.5⋅𝑚2 \+ 0.5⋅𝑚3

**In the implementation, this is approximated using normalized indices:**

Score ≈ (2⋅0.5+3⋅0.5) / (𝑛−1)

which is equivalent when bucket midpoints are uniformly spaced over \[0,1\].

This smooths the discontinuity introduced by bucketing. The key insight is that bucket boundaries are hard in supervision but soft in prediction.

### 

### **5\. Why boundaries still matter (subtle but important)**

Despite softmax smoothing, the model is trained using cross-entropy with targets like “this is bucket 2, not bucket 3.” This forces it to learn decision boundaries.

As a result, near bucket edges:

* gradients push representations toward one side  
* learned representations can become locally unstable

So the discontinuity is not fully removed.

### **6\. Adversarial opportunities** 

**A. Boundary flipping attacks**  
If an adversary knows the boundary between categories (e.g., “human” and “AI-ish”), they can make small edits to push an input across that boundary.

This works because buckets impose classification margins, creating relatively sharp decision surfaces. It is analogous to image classifiers where tiny perturbations can flip “cat” to “dog.”

**B. Exploiting threshold regions**  
From the paper: “Some edits are too small to be detectable… we choose a minimum threshold.” This introduces an explicit blind zone. An attacker can stay just below lo\_threshold while still applying meaningful AI edits.

**C. Mid-bucket ambiguity exploitation**  
Middle buckets correspond to uncertain or mixed cases. An adversary can aim to produce outputs that fall into these regions to avoid strong classification.

From `ternary_eval.py`:

preds\[scaled\_scores \< h\_thresh\] \= 0   \# human  
preds\[scaled\_scores \> ai\_thresh\] \= 1  \# ai\_generated

The middle region effectively becomes a “safe zone” (AI-edited).

**D. Gradient-based adversarial attacks**  
Because the model is differentiable and outputs softmax probabilities, an attacker can optimize inputs to minimize the predicted score while preserving semantics.

### **7\. A robustness advantage of bucketing**

Compared to regression:

* Regression predicts an exact value and is sensitive to small perturbations across the entire input space.  
* Buckets compress predictions into discrete regions, making the model tolerant to noise within a bucket.

So:

* Smoothness: regression high, buckets medium  
* Boundary issues: regression none, buckets present  
* Noise robustness: regression lower, buckets higher  
* Adversarial surface: regression global, buckets concentrated at boundaries

### **8\. The key tradeoff**

Bucketing converts continuous vulnerability into localized vulnerability.

Instead of:  
→ small changes anywhere causing small output changes

You get:  
→ small changes near boundaries causing large effects  
→ small changes away from boundaries having little to no effect

This also introduces a discretization tradeoff: increasing the number of buckets improves resolution but creates more boundaries and harder learning, while fewer buckets improve robustness but reduce precision.

### **9\. Summary**

At bucket boundaries, continuous edit magnitude is discretized into hard categories, so very similar inputs can receive different labels. The model therefore learns sharp decision regions, even though its outputs are probabilistic.

This does create adversarial opportunities in structured ways:

* boundary flipping  
* threshold exploitation  
* ambiguity targeting  
* gradient-based manipulation

At the same time, probability-weighted decoding smooths predictions, and bucketing improves robustness away from boundaries.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAREAAAAzCAYAAACwhzwHAAALx0lEQVR4Xu2c9asVTRjH379AFBUVFQsbAwUTuzAQW7EwUFAUO1ERu0Wxf1AxULF+sLu7UewWAwO7Y1++w/ucd85zZuvM2evV+3xguWefmZ2d2u/OPDN7/3EEQRAs+IcbBEEQwiAiIgiCFSIigiBYISIiCIIVIiKCIFghIiIIghUiIoIgWCEiIgiCFSIigiBYISIiCIIVIiKCIFgRuYhkyZKFm1JGmzZtuEnwYP78+c7+/fu5OcOA8l+8eJGbU8bs2bO56bfQrl07p0KFCtwcGZGJyKlTp5ysWbM6P3784EEpo2zZstxkZObMmU7v3r2d9u3bp2nlpkcKFizo3L17l5v/etAXGzZsyM0ppUuXLtwUmkyZMnFTHM+fP3cWL17srFu3ztm0aZOzefNm9RfnixYtisXbt2+fU7duXe3K6IhERD5//qwqA4WLkjJlynCTkQ0bNjiVK1dWefJrpIxARquDmjVrRt4Xga2IjB07VrXNxIkTeVCMN2/eOBs3bnSWLFkS6884cA67To0aNdQLNGoiEZEJEyaoUcLPnz95UEopXbo0N7ly9epVEZH/wBQzPU5rxowZ46xcuZKbrUGbR90Xga2I4KWIvJYvX54HGdFFxAREJXfu3NycciIRERTqw4cP3JxySpUqxU2uID9eFZ6RuHDhgqqHXbt28aDfCvLUvHlzbrZi9erVTp48ebg5EmxEZPTo0b6iwAkSH2GYAkVJykXk5MmTnoVKJSVLluQmVz59+uRb4RmJ9FgXUYgI0ty7dy83R4KNiCCfnTp1cvLnz69+X7p0iUdJIIiINGrUKPC0P1kCiQh3jv769ctV3QoVKuRUrFiRmxN4//49Nznv3r3jJk9KlCjBTa58/fo1ocJxv2/fvmmxEnn16hU3KUz5B7xecF8O6g+HF8jX9u3bnQcPHvgOxXkev3//HndugtdFEPzqSidIXMTRyxZERNAXP378GGdDXb59+zbORgQtY9i+Z8JGRHLkyKHKdfDgQZXnIFN1asPMmTPzoBg7d+5UcVasWMGDUoaviCADgwcPVn93796t5ms9e/Z0Vq1apWzTp09PiO+VYXQcxJk3b56TM2dO9fvatWtO0aJFlRMIlXfs2DF+mZHixYtzkyvofFTpa9ascTp37qzm4OXKlVM2/cGDneI2adIkZq9atWrMjoPQbVR+rAagjJQ+ePjwoRpaw7tOb5zJkyfH0iEoDQhIx44d1XmVKlXi4uzZsyd2vwYNGqjyUdoFChRQdgyR3ejbt6+Kc+vWLR6UwKRJkxLKWK1atbg4ePhhx/KiHs/UwbFqwNODEOCvl4hgBQJ9sV+/firulStXlH8H/bFp06bKprNly5YEG2fq1KlO/fr1Y2Vs27atyh+cm1jN8bteJ1kRGTFiRJzjl+rED6861kGcwoULc3PK8BQRrLK0bNlS/UZG0PH1UcbAgQMTCotzdHA3MGQ7dOiQ+k0Nh2XHFy9eqAbFw9enTx92lZkwIoI3HlU67kdA/WHDHgIC/hMsncGui8jLly+dKVOmJDTykydP1P4DPf2jR4+qMEofy29YXn706JGy07SP19/169eVTd8DM2TIEGXThQ4jHNwDdghMq1atlOiA8+fPG9PWGT9+vAo/cOAAD0oA8dD2KMPChQuV1x82fZRFIoID7YgHuE6dOur89u3bsXjUDmhnPPhwpEL8SLi9RAR5ABgFIi5eQlRmABuWNolx48Z51gHQwyn/tLcJo2q/63WSFZFixYrFja4oH35QvCAiEiS9ZPEUETxANOxGJrJnzx4XjgaEHWJD4PzmzZtarP/BQ1ipUqXY+fr161V8NBagwp49ezYWxwtUflB0EeHAli9fPm5Wdl1ECCwZuqXjZcfIxGTnoFPA+amDeKbNdZQGX8rr37+/MW2CRpJYGvQC0zPE5eBafRRKIsLJli1bXJtj1c4UD0ICe7NmzXhQDGrv48ePq7iPHz+OC+f1iX1BpnsREDmMDgm6/saNG+qlxtPzI1kR4fcYNGiQso0cOTLOzqH8pWsRwYYxApnAG0aHRADLpwTO3ebDEBEdeqN5TX+8SKWIYE7KgT2VIsJxs+tcvnzZ2bp1q4rHpzSA0vjy5UucfdiwYZ5pHzlyRIVjepAMuFYfNbiJCEZIuh2/8ZBw4P9BmJeI4MEGiGO6F69P1JcpHsCoTu/fgF8flmRE5MSJE063bt3ibDS1w8HzqENx0rWIEBgZIBNcHEwPE86548sN28LBjxIU3SfCgY2Psshu2uVoKjfwSh9vZI5bfEwbYIcPBtON+/fvq3NsmONQGtz5OmrUKGPaBIb9CIfY+NG6devYffRDf+DdRATTV92O33AecujB8RIRQH3RTfSrV68eOyffSxDgmKVyJUsyIlKkSBE1/eUH5cWrj1OcP0JEWrRoYcyEKXM4h8MrCKbrw+BVwRy8qd3uB5vbQw6nGwdORbd03Oxu6fP4vXr1UrZ79+7F2WHTpwW6nacBaN+BG0uXLlXhui/IxIwZM1Q8vkIHW5CRiElETMN0LMMizE9EqC9OmzYtzo6NVbDre1/wqYMpTyZo5NajRw8eFJhkRMQtf7SY4RYOKNwrDggSxwZfEUHnyZUrlzETsOGtzG1eqytz5sxRfzHXNhWOT5m8CCMi5ODk9wOwuT3kpu8PsLTslo6b3S19PT6Eg9sI2CAi8EPh4dDtpvh+I5FZs2apcGzG8gIjMbQ/B9cmOxIxCcXcuXNdw3SoL8IvooPVNn5/vzrAVIGmC1gwQNzly5fHwrGrN8hSKxFWRDB6hHPYBPl9cPBRJkHhXmUEQeLY4Csi+DIRGdC/xsXQk891CdhIKHRwjf7wdejQQf2Gc4vAdnlTmm6EERGvHau8fLodfhsdfbmT42V3S1+Pv2PHjgQbIMHF6g4EWg+n+Hyqibc9T0eHlo51p7gJLJ/ydMgXloyI0OrW06dPYzZ9quknIhSP/COApkJ8uwHtkTBB5aJw+q37lnC+YMGC2LkfYUUEvpBly5ZxcwzK04ABA3iQgsLdykggHNsAosJXRGgvB+ZueFvcuXPHM+P16tWLrbZwMKfHR0ZwFOLhxNsZ6WDZE50ejeC3CUsnqIjUrl1b+Two31ghwDIoRlHYOk92vI30kQcaD/auXbuqzUjYg4FhJj2AqBssUSJ9POB6+rVq1VIPvJ4+5vFIHyKLdXuyY+ly+PDh6p7YLwAbRhtYysS9IUDdu3dX9rx586q37unTp2MrHThQPuQD0xO0lZ62yZFJ4X7oDzgdtPclmekMwNudp0n7h7yWeOEM1a+BEGGPBX7z1SmC35sYOnSoCoNA40UG3xPO4f/BaA8jntevX/PLPAkqIrRHiA6+QIB9TLpfBAfqDPtjsFyOvsXrELtSYefQKhyEPyp8RYQyCfAJud8WYv6m5GDpUvc4Y6iGIXVQZ6xOUBGxAfk6c+aMmnNj6zxAHcAxCYdnGNELA1YreF3reyJsQRuF+bcIEFFsCkwVeONjTwzqMCjkn2ncuLE6h+D61Qnir127lpsVaDu0q77/BlNKfa9JGIKKSFqCNnYbyaQKTxExbRUPAuIH2QlpS1qIyN8IHirsy0hGuH8nGMmhbx0+fJgHuUIju7QgPYpIWpTdU0TIR4EpShi2bdumhvphh4NhERFJDrSpvsnqTyGZvghwXdR9EaQ3EYH/DtsEosZVRDCkw1wNDYAvAcMO8XBdkD0INoiIhId8HH8SmEbS3B59Ed9w6Y5ZP7BzNeq+CNKbiGAXdlq8LFxFBF50bNOGMxVOK5yH4dmzZ2o1hr4ViQIRkfDgQYzKjxMV586dU85oOI2pL6J/hQF9kb69iYr0JCL4FgnfOaUFriKSKrCqEBVuDjPBDP5NZCo+ef9TwTddYRy5YYEDPj2Ajyv5ZrwoiVxEBEH4uxEREQTBChERQRCsEBERBMEKERFBEKwQEREEwQoREUEQrBAREQTBChERQRCsEBERBMEKERFBEKwQEREEwQoREUEQrBAREQTBChERQRCsEBERBMEKERFBEKz4FxHIuGkWmxkAAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,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>

[image3]: <data:image/png;base64,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>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIsAAAAjCAYAAACkVvl8AAAFhklEQVR4Xu2Z3UsVTxjH+wtE6CJFQUVQ8MLEICQQFRTJQKJAxEQUQ/TCNBTDl0QUUiyQlECqCxEJutB8iTBQSCUMxUiRwjTypUikUEnxLZr4Duzp7OPOnpk9x58/Yz6wnDPfZ3b22TnfmZ2dc4ppNJKcooJGI0KbRSONNotGGm0WjTTaLBpptFk00hypWXZ3d1lERASVbfH396fSicdJP/iKzMxMtry8TGVHeGWW1tZW5ufnx0ZGRmiIg5gqv379cnTef8XVq1eV81Ot72t8dX3HZomPj+ezQEpKCgsODqZhlpqaSiVpfv/+7bMb9CXT09M8LxxRUVE0bMn/5T6CgoKopIxjs6ATCgsLqezC204KCwtjNTU1VD5WcE+GkXFgFrQD/ZOenk7lYwH5rq2tUVkJR2YxOmxsbIyGOHfv3vV67YG2vTUc5fHjx1RS4ty5c/yzoKCA52aURaDO9vY2lY+FtLQ0dvbsWSor4cgsvb29tj8kYmVlZVRWxu4aTvj48SN7+vQplaVYXFw0zSTG7IKBI8JT/t+/fzeVsRBWZXNzk0qWDA4OeszHE8pmWV1dZZGRkfzC+G41tSG2s7NDZVeyNOno6OhDGoBWX19PZa9ISEhgP378oLJHaH4zMzMuw4gQxWJjY9nS0hL/jhEfGBjIDg4OWF5envAcyvj4OLt9+zb/jnN+/vzpiqHc2NjoKrvruI5TlM0CcFG7BZPVDbe0tPDRCWgc5aSkJJNm6Dk5OVT2Gjwi3TvXE1VVVezhw4dU9miW8PBwKnFKS0td30tKSlxt4NPqZcEK45y+vj7+3Zj1FhYWeHlra8u9Ogf6hw8fqCyNY7OUl5dTmYNp0aoD379/zz9HR0cPGQ31X7x4YdIMPS4ujsom2tvb2f3795UPtJ2dnU2bs8TqfsCbN2947PXr1zTE+yE5OZnKh4BxRe3bMTs7yz/PnDljOj83N1fYHnSYyynKZsFzFRfFNGzF/v6+MFkQEhLCXr586SpjxIrqQ79w4QKVTeC5/+3bN+UDbff399PmDnHnzh0+K4pAO1b5ox/wyPMEzsWC2Sk43/2tUZQPgP78+XMqS6Nslo6ODmEyBnZxGgsNDT2kGUDPysqisteovKmJcjMYGhridawGDwaGHevr6/zcz58/05AU2AzF+XThXVFR4VbrL6I8ZVE2S0BAgMcORPzt27dU5rifa+zW3rp1y63GXxB79eoVlb1CxSjYoseor6ysZNXV1ay2tpbV1dUdOpCnVZ9YaTCGoWMvyb3OpUuX2N7enqtsGFEEtihonJbdsYvJoGwWXDAxMZHKJk6fPs2fnVZg57O5uZk9e/bM1cmi1z9vb44yMDDAJiYmqGwJ3hqM/GSP4uJiUxtW+ePtEVsPDx484Atd1Hny5Am7cuUKe/TokaluZ2cnj9v9t4P4jRs3+KAy8rCiq6tLGJPFkVmGh4epbKKpqck2se7ubjY5OckyMjKE9WAmlVlABtEm4lGBe1tZWaEyz+PTp0+uMn5o0W4w6vX09FDZBF4O0EZbWxu7ePEiDXPOnz/Pbt68SWUlpM2CKRiIflyKVT3MKpcvX3aVUefevXtuNf4Co2Av4SSDH0f2PyQRRUVFVOJ8/fqV95/7DrFVnxvYxWSRMgseO7jY/Pw8e/fuHQ1bgo0vuvJGG3gmG3sBc3NzprgBjIk1wr8ANhxpP6ggGkzYnzGWAxsbG7w/Rdf58uWL7aNMFimzgPz8fHbt2jUq24K1i/uO4dTUFL9B0YIWYNqOiYmh8omG9oOvwMIbi/Dr168LH2PAF7MKkDaLUzACVBAtjE86qv3gKxoaGmz/v1LhyM2i+XfQZtFIo82ikUabRSONNotGGm0WjTTaLBpptFk00mizaKTRZtFI8wdbq9upQq33dgAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,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>

[image6]: <data:image/png;base64,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>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAAA7CAYAAABi1IYNAAAI1UlEQVR4Xu2c96vVShDH318ggoqCgg0rKKIoFuyiiL2gYq8o2BUFFTvYsKOCYsMfLNg7VhSxd6zYsGDBgti7+/juY0IyNzlnk1P25N35QDjJ7GRv2je7Ozu5/yhBEKzxDzcIgpA9RICCYBERoCBYRAQoCBYRAQqCRUSAgmAREaAgWEQEKAgWEQEKgkVEgIJgERGgIFhEBCgIFhEBCoJFRICCYBERoCBYRAQoCBYRAQqCRUSALn79+qVatmypChQooEqUKOEpK1WqlLZjWb58uafMlCdPnqjevXtzs5CPEQH6AIFBaJyaNWtykxFnz55VnTt3VrVq1VLdu3fnxUI+RgTI2LFjh/6FAHv27OnY9+/fr8qUKeNsR6Fdu3YiQMGDCJCBVgps3brV0wrWr19frV692tmOgghQ4MRSgD9+/HDGY1GXAwcO8Go1NWrUcNbht3nzZr1erlw5xx6VTAuwdu3a3BSZ6tWrc1PKFCxYkJvyPZEFePToUW7KKkWKFNECmTRpEi/y5fPnz2rGjBmOAAsVKsRdNNu2bXPWBwwYoH3BunXrHDtx8OBBtWfPnsDl9+/fHn8IsGvXrh5buujWrVta7wmddzrB9chEvXEmkgB37tzpPMh//vzhxVmDjuHly5e8KCFjxozxfRDc4iPgt3btWm6OBATYpUsXbk6ZXbt2GbV+VatW9T1vP0z9woKxdNmyZbk53xJagFu2bHG6Y58+fdI3CuF7G5w+fdoRYVjGjx+vpkyZ4rFVq1bNsw1Kly4dqX4/IEBEQ9MNjg/d8mTAzzSSm65z9gN1v3nzhpvzJaEF+PTpU26ySvPmzfUNrVKlCi9KCj1kX79+VYsWLdLbaN3dPHv2TFWqVMljC8v379/ViRMnVMWKFXVdJ0+eTNtLC2PZoO50KmRSgNOnT1dFixbl5nxJaAHmItQKbtiwgRf9r7l//74+7wkTJvCilMmkAL98+ZLR+uNEJAGuX79ed82QHZJqaD4doAUjEaK1iTto1dHNp95Gjx49fIU2bNgwbX/x4oXHzkH2DfwWLFjAiwIJI5Ddu3dr//bt2zs2RHthQ5kfKHvw4AE3RwJTRHgW379/r7eXLl2qSpYsqerUqaNu3bqlbbiWQ4cO1X8X87m50gUOLUBcZPcEdfHixZMGQS5fvhx6CQteBCTCOIOH58KFCzq6S+dDgS6sN2vWzPFFWD/Z+RYrVswRKHxfv37NPPxJVi/Rq1cv3aUGs2bN8jzgc+fODawHdkSlU2XatGnq7t27asSIEbpOpBCS6CijadmyZfp6kug6duwYeFzZJrQAceBoccCVK1eMTmT79u2BC8ZciOIhbL93714dJcMSBYyxcDytW7fmRbGBrmebNm3yXNvBgwd7bFjnPm5WrFihDh06pNeRhwpf/JqQqF7i7du3nugrJS9QlDPR8cGejrxYqr9evXp63d0YXL9+Xdt44IvG+7lAJAFiQSv4+PFjXmwdOr6LFy/yolhw9epV/YtzwFynG0yOux+cRA84wAuSGDVqVEJfjokvBOimYcOGej+TsTj8kk2dUAuGVjaImzdv6l/48emNcePG+Z4HAmF+dgJlx48f5+aMEFqArVq1cm48lokTJ3IXq6Cvj+P69u0bL4oVOId58+blsbkfHL6diDC+IIwvgX1MI8bwrVu3Ljd7OHPmjPZDtzwZ8Fu1alUem995wNaiRQtutkIoAbq7L3fu3Ak8QQ4mn8MuUcFcHr0V48qlS5f0df3586fHDlujRo082ybXH8Cvf//+3ByIab1usI9J6wfgm660vMOHD+v6/v7967H7nfO9e/e0/dWrVx67LYwFSEEBzIsRI0eOVOXLl3d52QVRrqjf6rkpXLhw5AABglJ4UWFs26RJE15shF+QYOzYsXlslI6XDARe4Bcm6mhS75w5cxw/5MzyfRArwLjVD/guXLiQmyPRuHHjPH8bARfYHj586LEjYkq+yHw6cuSIU4YuLK4R7n+2MBZg06ZNVadOnZxtROZwIhT6tc2+fft0polNIP6ZM2c625hsfvfuncvDDFxXLBRQIAHR+JCYOnWqtt+4ccNj54wePTrPA5oME386TkARWXf+a6I6UPbhwwdujgTqqlChgscWNP6DjRLN3eX04qbnOlsYCxDgUx266JgHzJW5FITZMQ8UFnwom05wXdAdIjCfN2TIEJeHGahn9uzZegoB6+hlYPKag3Eu3taDBg3iRR7cQjHFxB+tC9WNlwSycmgb94MnoxPPnz83qt8U1HXu3DmPDa1i5cqVPTZw/vz5hNdj5cqVzidp2SCUAHMR9PuDLmYiPn786Pn0CGB8RUGcKGA/3GCCvoIPAx4k1GOaUMC/W/QD5cOHD+fmhCSrMxUw/dC3b19uzgnQsGB6LFvEXoB4UPjgOxkIbmA/mrAF165dc8Lq7ocPb/ZHjx7lWTAFg7EeFprcxn7u6Q98eoQxYRgoyhwG+KNVIeizK7Bx48ZQgibCHkMYMll3qmT72GItQHRzkqVhcWg8FHShb9++HViWDOyH0DmBYArmxkxBAgId27Fjx3hxIMiccf8TKaoD40/8IhMkLFGvQTLwBYrpN5zZJmpvKhViK0CkxCHlCf/FDJkvbdu21TZaEJBBa4IMCYoWupegz3Iw6YtJawL5l/jYNdFCoW4EXdzfFGKiGalSpiAqiGDA4sWLdVpXGJDnOXnyZL2O1hzd6VQ+/s3Eg4gIut8nX7ahVD9c+wYNGrDSzBJLAWJwP3DgQD2OwHgCokFmjt+CMvj06dNH+/fr108LBt1IP/DgmeZLciA+d6Q4Ew9xIubPn89NkVmyZAk3pUyujvtwn+jfnGSbWAowk6R6Ezp06KATg9H13LRpEy8WcpBTp06pNWvWcHNWEAG6QPKu/OMgIZuIANV/rR4mufEb9xxSIV6IANV/GfXpSGEThLCIAAXBIiJAQbCICFAQLCICFASLiAAFwSIiQEGwiAhQECwiAhQEi4gABcEiIkBBsIgIUBAsIgIUBIuIAAXBIv8CsZiTWRvP10UAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,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>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJEAAABDCAYAAACcLucHAAAFP0lEQVR4Xu2d2ys+TxjAv3+BCFGSU0pJSE7JMTfklguKcigJKSIXcsGNS4cLuXAopxyLJHEhV3JWyqmIEjlEyTHM9/eMZn+78+77el/jy9h9PjXtzDzzlt79vLvPzs6uPwRBBPnDdyCIo6BEiDAoESIMSoQIgxIhwqBEiDAoESIMSoQIgxIJcnd3pxR7eXt7Uz4D9d8OSiSIk5MTqa6uJg0NDXzIKpeXl3R8ZmYm2d3d5cO/DtNLdHt7S4KDg6kMUVFRJDw8nNYjIyNpv6enJ227u7uTsLAwsrGxofk8xI6OjjR9epyfn5OZmRlNX3d3N0pkJCIiIuj29fWVjI2NcVFC2tvbSVpaGlldXdX02ysRjIOiBiUyGDExMXT79PREpqenueg72dnZn5ZodnaW70KJjEZ8fDzdPjw8WOzwpKQkuoVc5vj4WBOzVyI9UCKDkZCQQLf39/dkbm5O6b+5uVGOUnp8JBHkXFD0QIkMhp5EOTk5VJLo6Gj1UA22JIJYT08PKSwsJM7OznwYJTIaaokqKyvpVZqXlxcZHh7+lEQdHR2kpqaG1uEUyCfVAEpkMPgj0crKCvH19SX+/v6fkggEZPT396NEZoCXiLG/v+9wTvT4+Eiv8hgw54QSmQAmEVydqSUCkpOTNW01ehKpGR8fp2OCgoL4EEpkNOLi4ugWjiL8zLKavLw8TfsjiWAWHMZ0dXXxIZTIaLC85/n5mUxNTXHRd3Z2dixOSx9JBHH+MwyUyGAEBAQo9ZaWFlXk/U59RkYGKSoqshACJfoHEpWWlvJdUnN1dUV8fHyIm5sbncthO12vQJzNXjNsSTQyMkLjISEhfIiCEukAv1j40uCwbxZsSVReXk7jJSUlfIiCEnHA1D58YWwL8yxmgJdIXWdHsLOzM6VPDUrEAV+WGj8/P03bqKgl6uvro+2hoSE6TQD1w8ND7QdUoEQIRS3R4OAgbefm5tL86eTkhButBSWygnqm1gyANDCRaC151gOkg/FwWwUlUjE6Okp/fTCpVlVVZfNWAWIsvkSil5cXTU5UV1dnkSPpwRJPRwsiF18iEcsFDg4OaHt9fd3qElPEeHyJROxKBEptbS29dYCYhy+RCOBne09PT/khiEH5EolgHTIDVgKCRImJif8PsEJ+fj7JyspyuCByISTR9fU1cXV1tUh2od3Z2anpQ4yLkEQsF+rt7VX6Njc3LaRCjI2QRAA8Mert7a3kQvCUKGIuhCX6LTQ2NmoSf1hxyDMxMaEZw8pPMT8/b/GwpIyYRiKGPXJAbGFhge/+FmBNdllZmfI38i+QkBFTSgS3ZWBbX1/Phym2BPvXtLW10aUjIDFKJCnwupi9vT26g+AGKDzdwfOTEjFQIkmZnJwkAwMDtM5OF+xRIQbcYZdhiS9KJCmxsbFKvbW1VTc3Ki4u/nAd0HeAEkkKLwyTqKmpSdMnAyiRpISGhmra6enpFkcj2SSCFRGyYxqJ1PkQA9ZBwapE2FlbW1t0PXRFRYVmjC1gxr65udmhYi9MorW1NT4kHaaRyNpKy4uLC7qzPDw8SEFBAX1Bp73AI1KwWsGRYi9Mot/w1IxpJLJ1mmKnNFtjvhsm0fLyMh+SDpToP1JTU6WVaGlpiQ9Jhykk2t7epq/OswasxIQdZu2U9xMwiRYXF/mQdBhaIpCCrXeCAjPU1v4NgouLC70B+9OkpKTQ19yoT7GBgYG0X9bHsQwtEfI9oESIMCgRIgxKhAiDEiHCoESIMCgRIgxKhAiDEiHCoESIMCgRIgxKhAiDEiHCoESIMH8BR8smIoiLksIAAAAASUVORK5CYII=>

[image10]: <data:image/png;base64,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>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKkAAAAqCAYAAADI8XtZAAAFYElEQVR4Xu2ayyt/TxjHv3+BCEWR24INiVyyoUiRLTsLLBQhlyKRa4kSwndDuSSXUGIva4pcskBuKSULlAVhfj1TczrnMfM5h8/nfL7j53nV9JnzPDPnzDnnfebyzOcPIwjN+YMNBKEbJFJCe0ikhPaQSAntIZES2kMiJbSHREpoD4mU0B4SKaE9JFJCe36VSA8PD1lAQABP19fX2C1FlHfC1tYWu7u7w2ZOWVkZNv1IxsfH2ebmJja7is9EOjAwgE2uAtfr7e1l3d3dPHV1dVkS2MB/eXlp1AGRzs/Pm87iDKciDQoKwibO6OgobxPm7e2Nv/SxsTH+K2NmZoatrKywxcVFNjU1hd3/hNjYWHZ+fo7NruETkba0tLCIiAgWGBiIXa6Smppq5JOSkoz80tISF9bT0xMbGRkx7E5ECufBosTHKhITE7GJv8yQkBBs5jw+PrLl5WWWk5PDr9HT02Pxv7+/s/z8fO4rLCxka2trFr+bTE5OerxvTz5f47VI6+vrWV5eHs+fnJywv3//ohLu0dDQYOThQ8FERkay4eFh49iJSOHhp6WlfbLZ0dnZye8fA3WPjo6w2YKYUsiu8/HxIRW/W0AbYESIiYmRtkdQUVGBTa7htUjxCzg+PrYcu0lzc7ORb2trM/JDQ0P89/7+/ss9qQxPL0sgKwPzU5kdA2USEhL47+7ursU3PT3NFhYWLDZ/UFBQYNt2GAH8gdciFby8vGCT67S2thr5jo4O/gs9j+rh2okURC1DdT4BTC9g2MZAvcbGRmy2sLGxYYhQ1pviY3/hRKR2fl/htUhXV1f5XBQm9U1NTez5+RkXcQ2ZSGUvWqASKazKo6Oj2eDgoLSuzGZGNReHehcXF9hsISsry8jDtAnqgHAFdtd2C6ciPT09xWaf45VIYXVqvpH29nZWWVlpKmHl7OzMEJGTZBfqMItU1Nnb21M+XJVIk5OTjTzUxUOu6nwCGKpl2NUDzGVeX1+N+xDU1dUZeX/iVKQQeXAbr0QKYRFoqAhHgEBguPUXsp4UFlOqh6sSqQCGbFldmU0AH6ZsHu5p2mHGHJUAYNEC9SAyAdze3lr8/sKpSPv6+rDZ53gl0oeHB+PLh5Seno6LuIpMpIDq4dqJFNovqyuzCVQ+J4um9fX1T+3Z3t7m9VJSUr4UizS/B09J1etjnIq0qqoKm32O1yIVqHohMzBfLSkpcZwODg7wKSyYw07m1b2ZiYkJdnV1xfN2IoX2l5eXY7PyvmDBBEmFqp5A5RfhH4g9/yucilS1CeFLvi3S4ODgTzeBj93GvHKWxUlFTy/wJFLR88G8GaO6L9WCSQD1xLAtQ3Ve2LIFn8rvD5yKdGdnB5t9zrdFCg2cm5szjvf3921vyteUlpYa+eLiYiMPOzUwDGVnZ/MtPIEnkcKmhKr9Krvd0An1ZJsbMF/Nzc3lftUcHnyZmZnY7DcyMjK+/ZH5mm+LFIQAOzrii4cvz59AyCg0NNRYaMgS9HTQRoEnkYo6MmR22GGSLZjMwDPBQoZQXXh4OIuKiuJtgy1TWXx2dnbWNrrhBmFhYSwuLo63G1J8fDy3yZA9Fzf4tkh/InYirampwWaO7GXIbBiYCzsp9xOBcBns7/uDXytS+HcUCKi2tpb3ap7EJPM5jQ/CXBmG9v8bsmfiFr9WpLBTAg+6urqaFRUVoZJW8AvxNE+TAcO60/+v/gQg3Nff34/NrvHrRArhHfhX0c3NDXZLgbJYpBAe+yr4HD8V+NjMW7n+4FeJlPiZkEgJ7SGREtpDIiW0h0RKaA+JlNAeEimhPSRSQntIpIT2kEgJ7SGREtpDIiW0h0RKaM9/jYNOI+sMQOcAAAAASUVORK5CYII=>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAE8AAAAsCAYAAAAkVOVyAAACFklEQVR4Xu2WQatBQRSA3y+QBWWnKCkpUSgRS+UX2FnY+AEWLGUtG/kn/oCtjZWykmxEKbFRzutM3Wlm3vBwurf3OF9N554zM7fO171z7xcwL/NlFpjHYXkEWB4BlkeA5RFgeQRYHgGWR4DlEWB5BFgeAZZHgOURYHkE/pS80WgE7XYber2eOWUF1+Hw+XzmlCe4Iu94PEIikRBNZbNZCAaDkEwmIZ1OQygUEnWspVIpbR/KW61WWu0R3kqeQyaTEdHW3Hg8hmq1qtVYnkI+nxfxVnP1el3LWZ5CsVgU0WyuXC6LuNvttDrLUyiVSiKqzR0OB/lEmvwm73q9miXBR8hbr9fiOpfLqcskt+TN53OxD+f7/b746Ki8vbztdguDwQAikcjT8lQ5w+Hwhywz9wrP5Dksl8unXtvT6ST242i1WuK1N/kYeUilUtFyB5s8JBaLSYE4JpOJNm/e3ytclVcoFES81xz+PDvY5G02G3k9nU6lQBUz9wpX5Tln263mFosFdDodmdvkmXvxx7vRaGg1c41XuCovGo2KaH4d8Ryr1WrQbDZhNpvJuinvfD5Dt9uV+X6/t4qy1bzAFXnYZDgchkAgoJ1V5vD7/do+Ux4Sj8flenzqLpeLNo+8lbxXscl7BJYHLI8EyyOA8vCMU39f7oHrcLC8fwjLI8DyCLA8AiyPAMsjwPIIsDwCLI8AyyPA8gh8A5X46wIC06DnAAAAAElFTkSuQmCC>

[image13]: <data:image/png;base64,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>

[image14]: <data:image/png;base64,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>
