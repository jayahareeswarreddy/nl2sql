# Design: NL2SQL — Fine-Tuned Text-to-SQL Model

**Date:** 2026-11-16
**Owner:** Jaya Hareeswar Reddy Gudisa
**Status:** Draft — awaiting approval

---

## Intent & why

Fine-tune a small open LLM to translate a plain-English question plus a table
schema into a correct SQL query. The goal is a **portfolio flagship** that
proves a competency currently missing from the resume: *training/adapting a
model*, not just calling LLM APIs. It should produce three linked, public
artifacts — a model on the Hugging Face Hub, a live Space demo, and a GitHub
repo with the training notebook and measured results — so a reviewer can see
the full pipeline: data → training → evaluation → deployment.

Text-to-SQL was chosen because it uses a high-quality public dataset (no
dataset-authoring burden), it is an unambiguous, demonstrable skill, and its
quality is **measurable** with execution/exact-match accuracy — which lets the
project show a credible before/after improvement over the base model.

## Goals

- Fine-tune a small (≤1.5B) base model with **QLoRA** on a free Colab/Kaggle T4.
- Beat the un-tuned base model on a held-out test set by a clear margin, and
  report the numbers honestly (base vs fine-tuned).
- Ship a public HF **model repo** (LoRA adapter + usage snippet), a public HF
  **Space** demo, and a **GitHub** repo (training notebook, eval script, README
  with metrics).
- Keep total cost at **$0** (free GPU for the one training run; free ZeroGPU
  Space for inference).

## Non-goals

- Not state-of-the-art accuracy or beating large proprietary models.
- Not multi-table JOIN-heavy / full Spider-benchmark coverage; single-table and
  simple multi-column queries are enough to demonstrate the skill.
- Not a production SQL agent that connects to a live database.

## Anti-goals (would make this a failure even if it "works")

- A demo that looks impressive but has **no measured eval** — the whole point is
  showing the model improved; unmeasured = not a fine-tuning project.
- Training numbers that can't be reproduced from the committed notebook.
- Over-scoping into a months-long effort; this must be finishable in days.

## Constraints

- **No local GPU.** Training happens on free Colab or Kaggle (T4, ~15GB VRAM).
  Inference demo runs on the free HF ZeroGPU Space (GPU provided at call time).
- Base model must be **ungated** (no access-request wall) and permissively
  licensed, so training and redistribution are frictionless.
- Beginner-friendly, well-trodden tooling so you can run and explain every step.

## Approaches considered

1. **QLoRA fine-tune of a small code model (chosen).** 4-bit base + trainable
   LoRA adapters via PEFT/TRL. Fits a T4 comfortably, trains in ~30–90 min, and
   the adapter is a few MB to host. Standard, explainable, cheap.
2. **Full fine-tune (all weights).** Rejected: won't fit a free T4 for a useful
   model size, slower, heavier to host, no accuracy benefit at this scale.
3. **Prompt-engineering / few-shot only (no training).** Rejected: this is the
   gap we're explicitly filling — it must involve real training.

## The "how" (decisions made — your calls to confirm)

- **Base model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct` — ungated, Apache-2.0,
  strong at code/SQL for its size, fits QLoRA on a T4. Fallback if training is
  slow or memory-tight: `Qwen/Qwen2.5-Coder-0.5B-Instruct`.
- **Dataset:** `gretelai/synthetic_text_to_sql` (large, high-quality, includes
  schema context and gold SQL). Train on a subset (~10–20k rows) for a fast,
  cheap run; hold out a test split for eval. (`b-mc2/sql-create-context` as a
  backup/supplement.)
- **Method:** QLoRA (4-bit NF4) with `peft` + `trl` `SFTTrainer` +
  `bitsandbytes`. LoRA on attention/MLP projections, r=16, 1–2 epochs on the
  subset. Prompt format: instruction + schema + question → SQL.
- **Evaluation:** on a held-out test set, report (a) **exact-match** of
  generated vs gold SQL (normalized), and (b) **execution accuracy** on a sample
  — run both queries against an in-memory SQLite built from the schema and
  compare results. Always report **base vs fine-tuned** side by side.
- **Deployment:** new Gradio **Space** on ZeroGPU (free). UI: a schema box + a
  question box → generated SQL, with examples. Loads base model + LoRA adapter;
  inference uses `@spaces.GPU`.
- **Hub artifacts:** push the LoRA adapter to a HF model repo with a model card
  (base model, dataset, metrics, usage). Reference the dataset; don't re-host it.
- **GitHub repo:** training notebook (Colab-runnable), standalone `eval.py`,
  `README.md` with the metrics table and links to the Space + model.

## What's left to the builder

- Exact train/eval subset sizes and LoRA hyperparameters (tune to fit T4 time
  budget; record final values in the notebook).
- SQL normalization rules for exact-match (lowercasing, whitespace, alias
  handling) and the size of the execution-accuracy sample.
- Space UI polish and example prompts.
- Naming of the repos/Space.

## Deliverables & resume line (draft)

- **HF Model:** fine-tuned NL2SQL LoRA adapter + model card with metrics.
- **HF Space:** live demo.
- **GitHub:** notebook + eval + README.
- **Resume bullet (draft):** "Fine-tuned Qwen2.5-Coder-1.5B with QLoRA for
  natural-language-to-SQL generation on a free T4 GPU; improved execution
  accuracy from X% (base) to Y% on a held-out test set; shipped as a public
  Hugging Face model and live demo."

---

### Open questions for you (before this is final)

1. Approve the base model + dataset choices above, or any preference?
2. Repo/Space naming — any preference, or shall I pick (e.g. `nl2sql-qlora`)?
