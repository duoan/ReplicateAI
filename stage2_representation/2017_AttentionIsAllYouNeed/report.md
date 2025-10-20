# 🧪 Experiment Report: Attention Is All You Need

> **Paper:** Vaswani et al., 2017 — *Attention Is All You Need*  
> **Organization:** Google Brain  
> **Stage:** Representation  

---

## ⚙️ Experiment Setup
| Component | Details |
|------------|----------|
| Model | Transformer (Encoder–Decoder), TBD |
| Dataset | TBD |
| Training | TBD |
| Hardware | TBD |
| Framework | TBD |

---

## 📊 Quantitative Results
| Metric | Original (Paper) | Reproduced | Deviation | Notes |
|---------|------------------|-------------|------------|--------|
| BLEU (EN→DE) | 28.4 | TBD | TBD | TBD |
| Perplexity | 4.6 | TBD | TBD | TBD |
| Training Speed | 3.5× RNN baseline | TBD | TBD | TBD |
| Params | 65M | TBD | TBD | TBD |
| Training Time | 12h (8 GPUs) | TBD | TBD | TBD |

---

## 📈 Learning Curves
*(insert plots or text summary after experiments)*  
- TBD

---

## 🔍 Attention Visualization
*(add attention heatmaps or screenshots)*  
- TBD

---

## 💬 Observations & Analysis
1. TBD  
2. TBD  
3. TBD  

---

## 🧠 Lessons Learned
- Scale up is the key to improve the model efficient training on large model. Multi head approach enables this. 
- Designed optimizer, which aligned with the model hidden dimension, why?
- Position encoding is important to understand the sequence char position information. 
  - The paper implementation Sinusoidal Encoding is absolute position, may not accurate like today's [RoPE](https://arxiv.org/pdf/2104.09864) which 
  enables positional encoding in context 
  - [Understanding Positional Encoding in Transformers](https://erdem.pl/2021/05/understanding-positional-encoding-in-transformers)
  - [You could have designed state of the art positional encoding](https://huggingface.co/blog/designing-positional-encoding)
  - https://medium.com/autonomous-agents/math-behind-positional-embeddings-in-transformer-models-921db18b0c28

---

## 📚 References

```bibtex
@article{vaswani2017attention,
  title     = {Attention Is All You Need},
  author    = {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Łukasz Kaiser and Illia Polosukhin},
  journal   = {arXiv preprint arXiv:1706.03762},
  year      = {2017},
  url       = {https://arxiv.org/abs/1706.03762}
}

@article{an2025reproduce_attention,
  title     = {Reproduction Report: Attention Is All You Need},
  author    = {Duo An},
  journal   = {ReproduceAI Project},
  year      = {2025},
  url       = {https://github.com/duoan/ReproduceAI/stage2_representation/2017_AttentionIsAllYouNeed}
}
```
