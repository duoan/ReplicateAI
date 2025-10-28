# 🧠 ReplicateAI Development Notes

> This document records all development operations, workflows, and progress logs for **ReplicateAI**.  
> It serves as both a quick-start guide and an internal research diary.

---

## ⚙️ 1. Environment Setup

Before running any commands, make sure your Python environment is ready.

```bash
uv sync
source .venv/bin/activate
pip install -r requirements.txt   # optional, if you have dependencies later
````

All Makefile commands assume Python is in `.venv/bin/python`.

---

## 🧩 2. Initialize the Project Structure

Run this **once** after cloning the repository:

```bash
make init
```

This will create the following directory layout:

```
ReplicateAI/
├── stage1_foundation/
├── stage2_representation/
├── stage3_deep_renaissance/
├── stage4_statistical/
├── stage5_neural_origins/
├── scripts/
├── paper_template/
└── PAPER_INDEX.json
```

Each stage corresponds to a historical era of AI research:

| Stage                   | Era                  | Description                          |
|-------------------------|----------------------|--------------------------------------|
| stage1_foundation       | 🪐 Modern Foundation | LLMs & Multimodal Models (2023–2025) |
| stage2_representation   | 🔍 Representation    | Transformers, BERT, Embeddings       |
| stage3_deep_renaissance | 🧩 Deep Renaissance  | CNNs, Autoencoders                   |
| stage4_statistical      | 📊 Statistical       | SVMs, Random Forests, EM             |
| stage5_neural_origins   | 🧬 Neural Origins    | Perceptron, Backprop, Hopfield       |

---

## ➕ 3. Add a New Paper Module

Use the `make add` command to create a new paper entry.

Example:

```bash
make add name="Qwen2.5" year=2025 org="Alibaba" stage=foundation
make add name="BERT" year=2018 org="Google" stage=representation
make add name="Perceptron" year=1958 org="Cornell" stage=neural
```

Each command will:

* Create a folder like `stage1_foundation/2025_Qwen2.5/`
* Copy template files from `paper_template/`
* Add metadata to `PAPER_INDEX.json`

After running, check that new directories and index entries were created.

---

## 📊 4. View Project Progress

Check current reproduction progress by stage:

```bash
make status
```

Example output:

```
📊 Current Progress by Stage:
🪐 Modern Foundation       :   1 papers
🔍 Representation           :   1 papers
🧩 Deep Renaissance         :   0 papers
📊 Statistical              :   0 papers
🧬 Neural Origins           :   1 papers
```

---

## 📄 5. List All Papers

To see all registered papers from `PAPER_INDEX.json`:

```bash
make list
```

Example output:

```
2025 | Qwen2.5 | Alibaba | planned
2018 | BERT | Google | planned
1958 | Perceptron | Cornell | planned
```

---

## 🧮 6. Generate a Summary Report

To quickly summarize total paper count and stage distribution:

```bash
make report
```

Example output:

```
📅 Report generated: 2025-10-18
🧩 Total Papers: 3
  Foundation      -> 1 papers
  Representation  -> 1 papers
  Neural Origins  -> 1 papers
```

---

## 🧹 7. Clean Temporary Files

When you want to clean Python caches and temp files:

```bash
make clean
```

This removes:

* `__pycache__/`
* `.pyc` files

---

## 🧱 8. Implement the Paper Code

Each paper module created by `make add` contains a standard structure:

```
2025_Qwen2.5/
├── README.md          ← Paper summary & key ideas
├── report.md          ← Experiment results / analysis
├── notebook/          ← Interactive notebooks
├── src/               ← Implementation code
└── references.bib     ← Original citation
```

### Step-by-step:

1. Open the new folder (e.g. `stage1_foundation/2025_Qwen2.5/`)
2. Edit `README.md` following the [paper_template](../paper_template/README.md)
3. Implement your model under `src/`
4. Add experiment results in `report.md`
5. Update status to `"in progress"` or `"completed"` in `PAPER_INDEX.json`

---

## 🧾 9. Commit and Push

```bash
git add .
git commit -m "Add Qwen2.5 reproduction module"
git push
```

---

## 🧭 10. Quick Reference Cheat Sheet

| Task                    | Command                                                 |
|-------------------------|---------------------------------------------------------|
| Initialize repo         | `make init`                                             |
| Add paper               | `make add name="..." year=YYYY org="..." stage=<stage>` |
| Check status            | `make status`                                           |
| List all papers         | `make list`                                             |
| Generate summary report | `make report`                                           |
| Clean temp files        | `make clean`                                            |

---

## 🧠 11. Development Log (Timeline)

| Date           | Update                              | Notes                                                                                    |
|----------------|-------------------------------------|------------------------------------------------------------------------------------------|
| **2025-10-18** | ✅ Initialized ReplicateAI structure | Added Makefile, scripts, and template                                                    |
| **2025-10-18** | ➕ Added BERT                        | Verified indexing and `make status` output                                               |
| **2025-10-18** | ✅ Attention All You Need            | Implemented Scaled Dot-Product Attention, MultiHeadAttention and PositionwiseFeedforward |
| **2025-10-18** | ✅ Attention All You Need            | Implemented Encoder, Decoder, Transformer,add a toy dataset for train test               |
| **2025-10-21** | ✅ Attention All You Need            | Implemented training on multi30k dataset training                                        |
| **2025-10-22** | ✅ Attention All You Need            | Debug and fixed loss NAN issue, and refacted the code                                    |
| **2025-10-23** | ✅ Attention All You Need            | Implemented multiple tokenizer to abalition test                                         |

> You can continue appending to this table as the project evolves.

---

## 🧭 12. Stage Naming Summary

| Stage Code     | Directory               | Display Name         | Era         |
|----------------|-------------------------|----------------------|-------------|
| foundation     | stage1_foundation       | 🪐 Modern Foundation | 2023–2025   |
| representation | stage2_representation   | 🔍 Representation    | 2013–2020   |
| deep           | stage3_deep_renaissance | 🧩 Deep Renaissance  | 2006–2014   |
| statistical    | stage4_statistical      | 📊 Statistical       | 1990s–2000s |
| neural         | stage5_neural_origins   | 🧬 Neural Origins    | 1950s–1980s |

---

## 💬 Notes

* Always verify that `PAPER_INDEX.json` stays valid JSON.
* Use `make help` to recall all supported commands.
* Each reproduction module should be **independent and documented**.
* When in doubt: check `paper_template/README.md`.

---

🧩 *ReplicateAI — Rebuilding AI, one paper at a time.*
