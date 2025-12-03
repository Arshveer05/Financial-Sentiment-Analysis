
# **Fin-RoBERTa: A Financial Sentiment Analysis Approach Using Keyword Masking**

### *Authors: Arnav Sawhney, Arshveer Chhabra, Manya Mittal, Parth Sanghvi*



---

##  **Project Overview**

Financial sentiment analysis is fundamentally different from general-domain NLP sentiment tasks. The language is more technical, context-heavy, and nuanced. Subtle modifiers, negations, and financial jargon often determine sentiment. Traditional transformer models—despite being powerful—treat all tokens uniformly, causing key financial terms to be lost in dense textual noise.

To address this gap, **Fin-RoBERTa** introduces a **keyword-aware masked pooling mechanism** combined with **domain-adaptive pretraining**. This produces a specialized, interpretable, and high-performance model tailored for financial news sentiment analysis.

---

#  **Table of Contents**

* [1. Introduction](#1-introduction)
* [2. Problem Motivation](#2-problem-motivation)
* [3. Review of Existing Solutions](#3-review-of-existing-solutions)
* [4. Dataset & Preparation](#4-dataset--preparation)
* [5. Methodology](#5-methodology)
* [6. Experiments & Baselines](#6-experiments--baselines)
* [7. Results](#7-results)
* [8. Lessons Learned](#8-lessons-learned)
* [9. Future Work](#9-future-work)
* [10. References](#10-references)

---

# **1. Introduction**

Sentiment analysis in finance is critical for:

* Market movement prediction
* Trading automation
* Risk management
* Event-driven analytics
* News-based investment strategies

However, financial text is **dense, factual, and lexically complex**, with over 60% of sentences often labeled *neutral*. This makes generic models perform poorly due to missed subtleties.

Fin-RoBERTa addresses this through:

* **Financial domain-adaptive pretraining (DAP)**
* **Keyword masking (token-level gating)**
* **Keyword-focused mean pooling**
* **Smart attribute discovery (Chi-Square + manual lexicon)**

This combination allows the model to attend to the *right* words instead of processing all tokens equally.

---

# **2. Problem Motivation**

### Why Financial Sentiment is Hard


1. **Domain-Specific Vocabulary**
   Terms such as *liquidity, liability, guidance, EPS* can have sentiment polarity only in financial contexts.

2. **High Neutrality of News**
   Most financial articles are factual, making neutral classification ambiguous and challenging.

3. **Polysemy & Modifier-Dependence**
   Words like *loss*, *gain*, *beat*, *slump* rely heavily on nearby modifiers such as *slightly, unexpectedly, marginally*.

4. **Token Importance is Uneven**
   Only 1–3 keywords may determine the entire sentiment of an article.

5. **Latency & Practical Constraints**
   Financial systems require real-time inference and cannot afford full-scale model retraining.

### 🎯 Core Problem Statement

> **How can we bias transformer models to prioritize financially meaningful tokens without increasing model size or relying on costly domain-pretraining alone?**

Fin-RoBERTa is designed precisely to solve this.

---

# **3. Review of Existing Solutions**

## **3.1 FinBERT (Araci, 2019)**

* Great domain adaptation
* Poor interpretability
* Treats all tokens equally
* Computationally heavy
* Struggles with neutral statements

**Gap:** Lacks explicit financial keyword emphasis.

---

## **3.2 DistilRoBERTa (Khaliq et al., 2025)**

* Lightweight & fast
* Suitable for real-time systems
* No domain-pretraining
* Misses subtle context cues

**Gap:** High efficiency ≠ financial domain competence.

---

# **4. Dataset & Preparation**

We use the **Polygon.io Financial News Dataset** (Kaggle).
It contains high-quality real-world financial articles with structured metadata.

### **Data Fields Used**

* Title
* Description
* Keywords
* Sentiment (pos/neu/neg)

### **Preprocessing Steps**

* Remove duplicates
* Remove empty & non-English rows
* Normalize labels
* Stratified 80/10/10 split

### **Dataset Characteristics**

* **Positive:** 3626 (65.3%)
* **Neutral:** 1270 (22.9%)
* **Negative:** 649 (11.7%)
* Avg description length: **41.3 words**
* Avg keywords: **4.59**

### **Why this dataset?**

* Realistic financial language
* Natural class imbalance
* High-quality labeling
* Ideal for supervised financial sentiment modeling

---

# **5. Methodology**

##  Overview Diagram

<img width="573" height="420" alt="image" src="https://github.com/user-attachments/assets/7e1a3ab6-890e-4b0d-a091-bb94061cdeec" />




---

## **5.1 Domain-Adaptive Pretraining**

We extend RoBERTa-base using MLM on a large financial news corpus (US-based articles).
This improves:

* Financial jargon understanding
* Contextual reasoning
* Detection of subtle sentiment shifts

---

## **5.2 Smart Attribute Discovery**

### Hybrid Keyword Vocabulary:

1. **Chi-Square selection** → top 200 high-sentiment-correlation words
2. **Manual lexicon** → essential financial terms

This ensures coverage of rare but crucial keywords (e.g., *guidance cut, regulatory scrutiny*).

---

## **5.3 Keyword Masking Mechanism**

During tokenization:

| Token Type | Mask Value |
| ---------- | ---------- |
| Keyword    | **1**      |
| Other      | **0**      |

The mask is applied element-wise to the last hidden layer outputs.

---

## **5.4 Keyword-Focused Pooling**

Instead of CLS pooling:

```
pooled = Σ(maskᵢ * hᵢ) / Σ(maskᵢ)
```

This ensures:

* Higher weight for sentiment-bearing tokens
* Reduced noise from filler text
* Better performance in subtle contexts
* Increased interpretability

---

## **5.5 Classification Head**

* Masked pooled embedding
* Dropout
* Linear layer
* Softmax for 3-class output

---

## **5.6 Full Architecture Diagram**

<img width="573" height="420" alt="image" src="https://github.com/user-attachments/assets/8f6862a4-3e25-4624-8f25-e2247fb8b265" />


---

# **6. Experiments & Baselines**

## **6.1 Baseline Model: Standard RoBERTa**

* CLS pooling
* No pretraining
* No keyword masking

---

## **6.2 Transformer Fusion Model (Exploratory)**

### Architecture:

* BERT-base + RoBERTa-base + DistilBERT
* Mean pooled individually
* Concatenated (2304-dim)
* Fed into a BiLSTM (256 units)


<img width="649" height="351" alt="image" src="https://github.com/user-attachments/assets/997cc8fc-b415-4275-9cad-64da66ba04e9" />




---

## **6.3 Ablation: RoBERTa + Masking Only**

* Showed improvement
* Lacked strong financial semantic grounding
* Confirmed pretraining + masking is essential

---

# **7. Results**

### **Overall Model Comparison**

| Model             | Accuracy | F1 Score   | Loss     |
| ----------------- | -------- | ---------- | -------- |
| RoBERTa Baseline  | 79%      | 69.13      | 0.46     |
| RoBERTa + Masking | 84–85%   | 78–80      | 0.44     |
| **Fin-RoBERTa**   | **88%**  | **83.30%** | **0.41** |

### 🔍 Key Observations

* Keyword masking gives significant boosts
* Domain-adaptive pretraining improves subtle reasoning
* Neutral class is hardest due to ambiguity
* Final model surpasses both FinBERT and DistilRoBERTa on financial news

---

# **8. Lessons Learned**

1. **Keyword emphasis is critical**
   Financial sentiment often lives in a handful of key terms.

2. **Pretraining + Masking = the perfect combination**
   Contextual grounding + keyword amplification is essential.

3. **Neutral class is inherently difficult**
   Requires advanced modeling to capture factual but non-polar statements.

4. **Statistical + manual vocabulary works best**

5. **Complex ≠ better**
   Multi-transformer fusion adds computation but little benefit.

6. **CLS pooling is suboptimal for finance**
   Keyword-focused pooling dramatically boosts performance.

7. **Lightweight models without domain training struggle**
   Domain knowledge cannot be replaced by compression.

---

# **9. Future Work**

###  **1. Stock-wise Sentiment Attribution**

* Identify tickers within news
* Assign sentiment to each entity
* Enables portfolio-level impact analysis

###  **2. Event-Type Classification**

Expand into event categories:

* Earnings beat/miss
* M&A
* Regulatory action
* Layoffs
* Guidance changes

Sentiment × Event-Type = actionable market insights.

---

# **10. References**

Araci, D. T. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. arXiv:1908.10063.

Khaliq, A., Ali, A., Ajaz, S., & Paul, F. (2025). Comparative Analysis of FinBERT and DistilRoBERTa for NLP-Based Financial Insights in Pakistan’s Stock Market.

