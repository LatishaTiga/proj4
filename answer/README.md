# QUESTION 1

**EditLens shifts from binary classification (human vs. AI-generated) to continuous scoring (0–1 scale representing degree of AI intervention). Why is continuous scoring more useful than binary classification for detecting AI text?**

---

## Short Answer

Continuous scoring is more useful because AI-generated text lies on a **spectrum of edits**, not a strict binary. Binary classification introduces **label noise, ambiguity, and loss of information**, whereas continuous scoring preserves the **degree of transformation**.

---

## 1. General ML Intuition: Why Binary Classification Fails

### Assumption in Binary Classification

Binary classification assumes **two separable distributions**:

\[
P(x \mid \text{human}) \quad \text{vs} \quad P(x \mid \text{AI})
\]

This only works when:
- Data is cleanly separable  
- Labels are well-defined  

❌ Modern AI-edited text violates both assumptions.

---

### Reality: Data Lies on a Continuum

Text exists on a **continuous spectrum**:

- Human-written  
- Lightly edited  
- Paraphrased  
- Heavily rewritten  
- Fully AI-generated  

So instead of:

\[
y \in \{0,1\}
\]

we should model:

\[
y \in [0,1]
\]

---

### Problems with Binary Classification

- Hard decision boundary:
  \[
  \hat{y} = \mathbb{1}(f(x) > \tau)
  \]

- Loss of intermediate information  
- Instability near threshold  

---

### Key ML Issue: Label Noise

Binary labels collapse distinct cases:

- Light edit → labeled AI  
- Heavy rewrite → labeled AI  

But these correspond to very different underlying values:

\[
\Delta_1 \ll \Delta_2
\]

This causes the model to learn a **blurred boundary** instead of meaningful structure.

---

### Continuous Scoring Fix

Model learns:

\[
f(x) \approx \mathbb{E}[\Delta \mid x]
\]

**Benefits:**
- Preserves ordering  
- Captures gradual transitions  
- Enables calibration  

---

## 2. Paper Insight: AI Editing is a Latent Continuous Variable

### Core Shift

Instead of:

> “Was this written by AI?”

The paper asks:

\[
\text{Estimate } \Delta(x, y)
\]

where:
- \( x \): original human text  
- \( y \): edited text  

---

### Training Objective

\[
\min_\theta \; \mathbb{E}_{(x,y)} \left[ (f_\theta(y) - \Delta(x,y))^2 \right]
\]

---

### Inference

Only \( y \) is observed:

\[
\hat{\Delta}(y) = f_\theta(y)
\]

---

### Why Binary Fails

#### (a) Editing ≠ Generation
Most usage is editing, not full generation.

#### (b) Mixed Authorship

- **Heterogeneous:** separable  
- **Homogeneous:** entangled  

👉 No clear ground-truth label exists.

---

### Key Insight

Binary classification is **ill-posed** because:

- No clear human vs AI boundary  
- Only **magnitude of transformation** is observable  

---

## 3. How the Code Implements Continuous Scoring

### (A) Training: Bucketed Classification

Continuous score:

\[
s \in [0,1]
\]

Mapped to bucket:

\[
j = \lfloor s \cdot (N-1) \rfloor
\]

---

### (B) Inference: Recover Continuous Score

```python
probs = softmax(output.predictions)
score = (probs @ bucket_labels) / (N - 1)
