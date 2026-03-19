# 🧠 Cognitive Drift & Hallucination Detection in LLMs

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Research-orange?style=for-the-badge)
![IEEE](https://img.shields.io/badge/Published-IEEE-blue?style=for-the-badge)

**A comprehensive, model-agnostic framework for automated hallucination detection in Large Language Models**

</div>

---

## 📌 Overview

Large Language Models (LLMs) are powerful — but they lie. Sometimes confidently, sometimes subtly. This project tackles one of the most critical open problems in AI: **how do we know when a model is hallucinating?**

We built a **6-component evaluation pipeline** that goes far beyond simple text matching. By combining semantic understanding, logical inference, lexical grounding, and structured knowledge verification, our framework provides a nuanced, multi-dimensional picture of LLM reliability — without being tied to any single model architecture.

> Evaluated across **8,742 samples** from **5 major LLMs** — ChatGPT, Claude, Gemini, Perplexity, and Ultrachat — with statistically validated results.

---

## ✨ Key Results at a Glance

| Metric | Value |
|--------|-------|
| 📊 Total Samples Analyzed | 8,742 |
| 🔁 Valid Evaluation Pairs | 1,508 |
| 🤖 Models Evaluated | 5 |
| 🚨 Hallucinations Detected | 3 (0.03%) |
| 🏆 Best Model | ChatGPT (HI: 26.09) |
| 📈 Avg. Hallucination Index | 27.79 ± 17.60 |
| ✅ Good/Excellent Responses | 91.7% |
| 🎯 Citation Accuracy | 95.65% |
| 📉 Statistical Significance | ANOVA F=55.22 (p<0.001) |

---

## 🏗️ Pipeline Architecture

The framework is built around **four phases** and **six specialized detection components**:

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: Input & Data                                          │
│  User Prompts → LLM Generation → Raw Response & Test Samples    │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: Multi-Component Evaluation Pipeline                   │
│                                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                         │
│  │  MPNet   │ │DeBERTa-v3│ │  TF-IDF  │                         │
│  │Semantic  │ │   NLI    │ │Grounding │                         │
│  │Similarity│ │          │ │ Analysis │                         │
│  └──────────┘ └──────────┘ └──────────┘                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                         │
│  │   BM25   │ │  Cross-  │ │Knowledge │                         │
│  │ Document │ │ Encoder  │ │  Graph   │                         │
│  │Retrieval │ │Relevance │ │   (KG)   │                         │
│  └──────────┘ └──────────┘ └──────────┘                         │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: Scoring & Analysis                                    │
│  Metric Aggregation → Hallucination Index → Statistical Tests   │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4: Final Reporting                                       │
│  Model Rankings → Quality Distribution → Deployment Guidance    │
└─────────────────────────────────────────────────────────────────┘
```

### 🔩 Component Breakdown

| Component | Model | Purpose | Avg. Score |
|-----------|-------|---------|------------|
| **Semantic Similarity** | MPNet | Dense vector alignment detection | 0.9431 |
| **Natural Language Inference** | DeBERTa-v3 | Entailment/contradiction detection | 0.7101 |
| **Grounding Analysis** | TF-IDF | Lexical overlap & content grounding | — |
| **Document Retrieval** | BM25 | Context relevance ranking | — |
| **Relevance Scoring** | Cross-Encoder | Fine-grained bi-directional matching | 0.9817 |
| **Fact Verification** | Knowledge Graph | Entity & factual claim validation | 0.9003 |

**Component Contributions to Hallucination Detection:**
- 🟣 NLI (Contradiction): **37.7%**
- 🟠 Semantic Similarity: **34.2%**
- 🔵 Grounding (TF-IDF): **28.1%**

---

## 📊 Model Performance Rankings

```
Rank   Model        HI Score   Std Dev   Deviation%   N Samples
────   ─────────    ────────   ───────   ──────────   ─────────
 1     ChatGPT       26.09      8.13       31.22%        395
 2     Claude        27.67      9.15       33.08%        232
 3     Ultrachat     28.37      9.83       34.66%        812
 4     Gemini        29.89     11.54       38.61%         20
 5     Perplexity    35.97     16.00       44.49%         49
```

> **Lower Hallucination Index = Better performance**

### Effect Sizes (Cohen's d)
- ChatGPT vs Claude: `d = -0.252` (small)
- Claude vs Ultrachat: `d = -0.238` (small)
- Ultrachat vs Gemini: `d = -0.656` (medium)
- Gemini vs Perplexity: `d = -1.208` (**large**)

---

## 📁 Project Structure

```
hallucination-detection/
│
├── 📂 data/
│   ├── raw/                    # Original conversation datasets
│   ├── processed/              # Preprocessed evaluation pairs
│   └── results/                # Evaluation outputs per model
│
├── 📂 pipeline/
│   ├── semantic_similarity.py  # MPNet-based similarity scoring
│   ├── nli_verification.py     # DeBERTa-v3 NLI component
│   ├── tfidf_grounding.py      # TF-IDF lexical analysis
│   ├── bm25_retrieval.py       # BM25 document retrieval
│   ├── cross_encoder.py        # Cross-Encoder relevance scoring
│   └── knowledge_graph.py      # KG fact verification
│
├── 📂 evaluation/
│   ├── metrics.py              # Hallucination Index calculation
│   ├── statistical_tests.py    # ANOVA, Cohen's d, Pearson correlation
│   └── quality_classifier.py  # Excellent/Good/Fair/Poor classification
│
├── 📂 visualization/
│   ├── radar_plots.py          # Multi-dimensional model comparison
│   ├── component_analysis.py   # Contribution pie charts
│   └── violin_plots.py         # Score distribution visualization
│
├── 📂 notebooks/
│   └── full_analysis.ipynb     # End-to-end analysis notebook
│
├── requirements.txt
├── config.yaml
└── README.md
```

---

## 🚀 Getting Started

### Requirements

```txt
transformers>=4.35.0
sentence-transformers>=2.2.0
rank-bm25>=0.2.2
scikit-learn>=1.3.0
networkx>=3.1
scipy>=1.11.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
torch>=2.0.0
```
---

## 📈 Evaluation Metrics

The framework computes five primary metrics for each response:

| Metric | Description | Mean Score |
|--------|-------------|------------|
| **Faithfulness** | Factual consistency with source material | 0.5145 |
| **Context Relevance** | Alignment with the user prompt | 0.7131 |
| **Citation Accuracy** | Accuracy of source attribution | 0.9565 |
| **Answer Relevance** | Appropriateness of the response | 0.8305 |
| **Hallucination Index** | Composite detection score (lower = better) | 27.79 |

### Quality Distribution

```
Excellent  ████████████████████░░░░░░░░░  43.2%  (3,775 samples)
Good       ██████████████████████████░░░  48.5%  (4,243 samples)
Fair       ████░░░░░░░░░░░░░░░░░░░░░░░░░   7.6%  (665 samples)
Poor       ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.7%  (59 samples)
```

---

## 🔍 Key Findings

### What makes ChatGPT perform best?
ChatGPT's edge comes from **consistency** — balanced scores across all six components, especially strong semantic alignment (0.9358) and NLI consistency (0.7523). It has the lowest deviation rate (31.22%), meaning it sticks closest to what users actually asked.

### Why does Perplexity rank last despite perfect citations?
Perplexity's paradox: it achieves **perfect citation accuracy (1.000)** and top Cross-Encoder scores (0.9976), yet has the highest deviation (44.49%). The culprit is extremely low **faithfulness (0.1434)** and weak **NLI performance (0.5396)** — it finds relevant sources but doesn't stay logically consistent with them.

### Why is NLI the hardest component?
NLI scored the lowest across all models (avg: 0.7101) with the highest variance. This suggests **logical consistency** is a fundamental reasoning challenge for LLMs — not just a knowledge retrieval problem.

### Pearson Correlations with Hallucination Score
| Component | r | Significance |
|-----------|---|-------------|
| Semantic Similarity | +0.788 | p < 0.0001 |
| Grounding | +0.639 | p < 0.0001 |
| NLI | -0.214 | p < 0.0001 |

---

## 🎯 Model Selection Guide

| Use Case | Recommended Model | Reason |
|----------|------------------|--------|
| Maximum faithfulness & balance | **ChatGPT** (HI: 26.09) | Best overall; lowest deviation |
| Citation-critical tasks | **Perplexity** | Perfect citation accuracy (1.000) |
| General-purpose deployments | **Claude** (HI: 27.67) | Balanced across all dimensions |
| Cost-sensitive applications | **Ultrachat** | Good performance at scale (812 pairs) |

---

## 🔭 Future Work

- [ ] Extend to **Retrieval-Augmented Generation (RAG)** architectures
- [ ] **Real-time detection** optimized for production latency
- [ ] **Multimodal support** — image, audio, and video generation
- [ ] **Domain-specific calibration** for medical, legal, and scientific contexts
- [ ] **Adversarial testing** against prompt injection and jailbreaking
- [ ] **Human validation** via inter-rater reliability studies
- [ ] **Temporal analysis** — how hallucination rates drift across conversation turns

---

</div>
