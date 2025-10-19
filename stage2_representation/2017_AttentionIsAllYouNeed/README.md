# 📘 Paper Reproduction: Attention Is All You Need

> **Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
> **Published:** 2017
> **Organization:** Google Brain
> **Stage:** Representation

---

## 🎯 Reproduction Objectives

- Implement the original Transformer architecture **from scratch** using PyTorch.
- Reproduce the core **sequence-to-sequence translation** experiments (English→German) on a **toy-scale subset** (e.g., IWSLT14 or Multi30k).
- Verify that **self-attention** can fully replace recurrence or convolution for sequence modeling.
- Analyze **multi-head attention patterns** and compare training efficiency vs. RNN models.
- Document performance gaps and discuss causes (dataset size, training time, initialization).

---

## 🧩 Core Ideas

1. **Self-Attention Mechanism:**
   Each token attends to all others in the sequence, allowing global context capture without recurrence.

2. **Multi-Head Attention:**
   Multiple attention heads learn diverse representations by projecting queries, keys, and values into different subspaces.

3. **Positional Encoding:**
   Since there is no recurrence, positional information is injected via deterministic sine and cosine functions.

4. **Encoder–Decoder Structure:**
   Stacked attention + feedforward layers form the encoder; the decoder adds masked self-attention for autoregressive generation.

5. **Parallelization & Efficiency:**
   The model allows full sequence-level parallel computation, greatly improving training speed compared to RNNs.

---

## ⚙️ Implementation Plan

| Component         | Description                                                                              |
| ----------------- | ---------------------------------------------------------------------------------------- |
| **Model**         | Implement 6-layer encoder–decoder with 8-head attention, hidden size 512, FFN size 2048. |
| **Embedding**     | Token + positional encoding (sine/cosine).                                               |
| **Loss**          | Cross-entropy with label smoothing (ε = 0.1).                                            |
| **Optimizer**     | Adam with learning rate warm-up (4000 steps).                                            |
| **Dataset**       | Small English–German translation subset (IWSLT14 or synthetic “copy task”).              |
| **Evaluation**    | BLEU score on dev/test split; attention visualization.                                   |
| **Visualization** | Plot self-attention maps and encoder–decoder cross-attention patterns.                   |

---

## 🧪 Expected Results

| Metric                  | Target                                 | Notes                                           |
| ----------------------- | -------------------------------------- | ----------------------------------------------- |
| BLEU (EN→DE)            | ≥ 25.0                                 | Small dataset; lower than paper’s 28.4 expected |
| Training Loss           | < 1.0                                  | Indicates correct convergence                   |
| Training Speed          | ≈ 3× faster than RNN baseline          | Validate parallelism benefit                    |
| Attention Visualization | Distinct diagonal / syntactic patterns | Confirms multi-head diversity                   |

---

## 🧭 Notes

- This reproduction targets **conceptual correctness**, not full-scale WMT14 results.
- A small Transformer (2 encoder + 2 decoder layers) is sufficient to demonstrate key properties.
- Visualization of attention weights is crucial — verify that heads attend to syntactic relations (e.g., subject–verb).
- Optional extension: implement **Transformer-XL positional recurrence** or **BERT-style pretraining** on top.
