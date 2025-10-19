# 🧠 ReproduceAI

> **Recreating every milestone in Machine Learning and Artificial Intelligence — from Transformers to Perceptrons.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Project-Active-blue.svg)]()

---

## 🚀 Overview

**ReproduceAI** is an open initiative to **rebuild and verify every major paper in ML/AI history**,  
starting from modern **foundation models (2023–2025)** and tracing backward to the origins of AI.

We believe that **understanding AI means rebuilding it — line by line, layer by layer.**

---

## 🧩 Project Vision

> “Because science means reproducibility.”

- 📜 **Goal**: Faithfully re-implement influential ML/AI papers with open code, datasets, and experiments  
- 🧱 **Scope**: From *Qwen2.5 (2025)* to *Perceptron (1958)*  
- 🧠 **Approach**: Reverse timeline — start with Foundation Models, then trace history backward  
- 🧾 **Output**: Each paper becomes a self-contained, reproducible module with reports and experiments  

---

## 🪐 Stage 1 — Foundation & Multimodal Era (2023–2025)

> *The golden age of open-source foundation models.*

| Year | Paper / Model | Organization | Why It Matters | Reproduce Goal | Status |
|------|----------------|--------------|----------------|----------------|---------|
| **2025** | **Qwen2.5** | Alibaba | Fully open multimodal model (text + image) | Rebuild text/image pipeline | 🧭 Planned |
| **2025** | **DeepSeek-V2** | DeepSeek | MoE + RLHF efficiency breakthrough | Reproduce expert routing and reward pipeline | 🧭 Planned |
| **2025** | **Claude 3 Family** | Anthropic | Leading alignment via Constitutional AI | Explore rule-based alignment principles | 🧭 Planned |
| **2024** | **LLaMA 3** | Meta | Open foundation model standard | Implement scaled transformer + tokenizer | 🧭 Planned |
| **2024** | **Mixtral 8×7B** | Mistral | Sparse Mixture-of-Experts architecture | Implement routing + expert parallelism | 🧭 Planned |
| **2024** | **Phi-2 / Phi-3** | Microsoft | Small but high-quality model; data-centric | Rebuild synthetic data pipeline | 🧭 Planned |
| **2024** | **Gemini 1 / 1.5** | Google DeepMind | Vision + Text + Reasoning | Prototype multimodal reasoning pipeline | 🧭 Planned |
| **2023** | **Qwen-VL** | Alibaba | Vision-language alignment model | Reproduce visual encoder + text fusion | 🧭 Planned |
| **2023** | **BLIP-2 / MiniGPT-4** | Salesforce / HKU | Lightweight multimodal bridging | Implement pretrain connector | 🧭 Planned |
| **2023** | **LLaMA 1 / 2** | Meta | Open LLM baseline | Implement tokenizer + attention stack | 🧭 Planned |

---

## 🔍 Stage 2 — Representation & Sequence Models (2013–2020)

| Year | Paper | Author | Goal | Status |
|------|--------|---------|--------|---------|
| 2018 | BERT | Devlin et al. | Masked Language Modeling | 🧭 Planned |
| 2017 | Transformer | Vaswani et al. | “Attention Is All You Need” | 🧭 Planned |
| 2014 | Seq2Seq | Sutskever et al. | Encoder-decoder translation | 🧭 Planned |
| 2013 | Word2Vec | Mikolov et al. | Learn word embeddings | 🧭 Planned |
| 2015 | Bahdanau Attention | Bahdanau et al. | RNN + Attention | 🧭 Planned |

---

## 🧩 Stage 3 — Deep Learning Renaissance (2006–2014)

| Year | Paper | Author | Goal | Status |
|------|--------|---------|--------|---------|
| 2015 | ResNet | He et al. | Residual learning | 🧭 Planned |
| 2014 | VGG | Simonyan et al. | Deep CNN architectures | 🧭 Planned |
| 2012 | AlexNet | Krizhevsky et al. | GPU-based CNN | 🧭 Planned |
| 2006 | DBN / RBM | Hinton | Layer-wise pretraining | 🧭 Planned |

---

## 📊 Stage 4 — Statistical Learning Era (1990s–2000s)

| Year | Paper | Author | Goal | Status |
|------|--------|---------|--------|---------|
| 2001 | Random Forests | Breiman | Ensemble learning | 🧭 Planned |
| 1997 | AdaBoost | Freund & Schapire | Boosting algorithms | 🧭 Planned |
| 1995 | SVM | Vapnik | Maximum margin classifier | 🧭 Planned |
| 1977 | EM Algorithm | Dempster et al. | Expectation-Maximization | 🧭 Planned |

---

## 🧬 Stage 5 — Early Neural Foundations (1950s–1980s)

| Year | Paper | Author | Goal | Status |
|------|--------|---------|--------|---------|
| 1986 | Backpropagation | Rumelhart et al. | Gradient-based learning | 🧭 Planned |
| 1985 | Boltzmann Machine | Hinton et al. | Generative stochastic model | 🧭 Planned |
| 1982 | Hopfield Network | Hopfield | Associative memory | 🧭 Planned |
| 1958 | Perceptron | Rosenblatt | Linear separability | 🧭 Planned |

---

## 📁 Repository Structure

```

ReproduceAI/
├── stage1_foundation/
│   ├── 2025_Qwen2.5/
│   ├── 2024_LLaMA3/
│   └── 2023_CLIP/
├── stage2_representation/
│   ├── 2018_BERT/
│   ├── 2017_Transformer/
│   └── 2013_Word2Vec/
├── stage3_deep_renaissance/
│   ├── 2015_ResNet/
│   ├── 2012_AlexNet/
│   └── 2006_DBN/
├── stage4_statistical/
│   ├── 2001_RandomForest/
│   └── 1995_SVM/
└── stage5_foundations/
├── 1986_Backprop/
└── 1958_Perceptron/

```

Each paper module includes:
```

📄 README.md   — Paper summary & objective
📘 report.md   — Reproduction results & analysis
📓 notebook/   — Interactive demo
💻 src/        — Core implementation
🔗 references.bib — Original citation

````

---

## 🤝 Contributing

We welcome contributions from researchers, engineers, and students who believe in reproducibility.

1. Fork the repo  
2. Pick a paper or model not yet implemented  
3. Follow the [Paper Template](paper_template/README.md)  
4. Submit a PR with your code and report  

✅ **Please include**:
- clear code (PyTorch / JAX / NumPy)
- short experiment or visualization
- reproducibility notes or deviations

---

## 🧮 Progress Overview

| Stage | Era | Progress |
|--------|-----|-----------|
| 🪐 Foundation (2023–2025) | Modern LLM & Multimodal | ░░░░░░░░░░░░░░ 0% |
| 🔍 Representation (2013–2020) | Transformers & Embeddings | ░░░░░░░░░░░░░░ 0% |
| 🧩 Deep Renaissance (2006–2014) | CNN Era | ░░░░░░░░░░░░░░ 0% |
| 📊 Statistical (1990s–2000s) | Classical ML | ░░░░░░░░░░░░░░ 0% |
| 🧬 Foundations (1950s–1980s) | Neural Origins | ░░░░░░░░░░░░░░ 0% |

---

## 📚 Citation

If you use or reference this project, please cite:

```bibtex
@misc{reproduceai2025,
  author = {ReproduceAI Contributors},
  title = {ReproduceAI: Rebuilding the History of Machine Learning and Artificial Intelligence},
  year = {2025},
  url = {https://github.com/<yourname>/ReproduceAI}
}
```

---

## 💬 Motto

> “Reproduce. Verify. Understand.”

---

⭐️ **Star this repo if you believe reproducibility is the foundation of true intelligence.**


