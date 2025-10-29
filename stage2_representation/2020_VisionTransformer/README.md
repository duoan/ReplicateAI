# 📘 Paper Replication: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

> **Authors:** Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas
> Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
> **Published:** 2020  
> **Organization:** Google  
> **Stage:** Representation

![ViT](./figures/vit.png)

<a target="_blank" href="https://colab.research.google.com/github/duoan/ReplicateAI/blob/master/stage2_representation/2020_VisionTransformer/notebook/Vision%20Transformer_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

---

## 🎯 Replication Objectives

- Implement the model from the paper in a modular, transparent way.
- Replicate key experiments and results.
- Verify main claims and discuss deviations.
- Document issues, reproducibility challenges, and results.

---

## 🧩 Core Ideas

1. Using `Transformer Architecture` to solve computer vision tasks.
2. `PatchEmbedding` split the 224x224 images into patches and flatten into a sequence
3. Pre-train using treat it as classic image classification problem,
    1. add `[CLS]` token, add learnable parameter
4. add a shared learnable parameter for position, shape with [1, 1 + num_patches, hidden_size].
    1. Where the patches near each other in the grid should have correlated positions
    2. Patches far apart should encode distant spatial relations
    3. Not like text, ViT uses learnable position embeddings, because images have bounded size, learned embeddings adapt
       better to 2D spatial structure
4. Only use the encoder + classifier head

---

## ⚙️ Implementation Plan

| Component  | Description |
|------------|-------------|
| Model      | Done        |
| Dataset    | Imagenette  |
| Evaluation | Acc         |
| Notes      | WIP         |

---

## Model Visualization

```text
VisionTransformer(
  (embedding): ViTEmbedding(
    (patch_embeddings): ViTPatchEmbedding(
      (projector): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
      (flatten): Flatten(start_dim=2, end_dim=-1)
    )
    (dropout): Dropout(p=0.0, inplace=False)
  )
  (encoder): ViTEncoder(
    (layers): ModuleList(
      (0-11): 12 x Sequential(
        (0): ViTAttention(
          (attn_drop): Dropout(p=0.0, inplace=False)
          (norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (query): Linear(in_features=768, out_features=768, bias=False)
          (key): Linear(in_features=768, out_features=768, bias=False)
          (value): Linear(in_features=768, out_features=768, bias=False)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_drop): Dropout(p=0.0, inplace=False)
        )
        (1): ViTFeedForward(
          (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (net): Sequential(
            (0): Linear(in_features=768, out_features=3072, bias=True)
            (1): GELU(approximate='none')
            (2): Dropout(p=0.0, inplace=False)
            (3): Linear(in_features=3072, out_features=768, bias=True)
            (4): Dropout(p=0.0, inplace=False)
          )
        )
      )
    )
  )
  (mlp_head): Linear(in_features=768, out_features=10, bias=True)
)
```

![model](./figures/model.png)

## Model Parameters

```text
================================================================
Total params: 85,606,666
Trainable params: 85,606,666
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 18.38
Forward/backward pass size (MB): 11336.94
Params size (MB): 326.56
Estimated Total Size (MB): 11681.88
----------------------------------------------------------------
```

## 🧪 Expected Results

| hidden_size | num_hidden_layers | num_attention_heads | intermediate_size | hidden_dropout_prob | layer_norm_eps | label_smoothing | lr   | batch_size | epochs             | Metric        | Target | Notes                              |
|-------------|-------------------|---------------------|-------------------|---------------------|----------------|-----------------|------|------------|--------------------|---------------|--------|------------------------------------|
| 256         | 4                 | 4                   | 1024              | 0.1                 | 1e-6           | 0               | 3e-5 | 32         | 49(stopped at 30)  | acc: 0.368408 |        |                                    |
| 256         | 4                 | 4                   | 1024              | 0.1                 | 1e-6           | 0.1             | 3e-5 | 32         | 100(stopped at 35) | acc: 0.643567 |        |                                    |
| 256         | 4                 | 4                   | 1024              | 0.1                 | 1e-6           | 0.1             | 1e-5 | 32         | 100(stopped at 35) | acc: 0.643567 |        | add data agumentation as following |

```python
from torchvision import transforms

train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
test_tfm = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

---

## 🧭 Notes

- Label smoothing (ε = 0.1) stabilized gradients early, prevented over-confidence.
    - Stabilizes early optimization
    - Prevents over-confidence
    - Improves generalization
    - Helps ViTs learn better attention patterns

Why `label_smoothing`:

- Softer targets → smaller gradients for the top-1 class
    - ViTs are very prone to over-confidence early in training — the CLS token can quickly drive logits to huge
      magnitudes for random classes. Label smoothing caps that gradient spike, keeping early updates stable.
    - Effect: loss surface becomes smoother → fewer oscillations → faster convergence.
- Forces the model to distribute attention
    - In patch-based ViTs, attention maps often collapse to a few dominant patches if the supervision is too “sharp.”
      When targets are slightly uncertain, the model is encouraged to spread probability mass more evenly → more diverse
      patch attention → better feature learning.
- Regularization = implicit ensemble
    - Label smoothing acts like training on multiple noisy label versions. It prevents the model from memorizing the
      exact training samples and encourages a margin between classes.
- Reduces mismatch between train and test distributions
    - Real-world test images are ambiguous — a “dog” photo might have 5 % visual overlap with “wolf.” Label smoothing
      teaches the model not to collapse into binary thinking (dog vs not-dog), which improves top-1 accuracy on unseen
      data.

---

📅 Generated by **ReplicateAI**
