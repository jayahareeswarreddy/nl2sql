# NL2SQL — Fine-Tuned Text-to-SQL

Fine-tuning **Qwen2.5-Coder-1.5B** with **QLoRA** to turn an English question
plus a table schema into a SQL query. Trained on a single free T4 GPU.

**Links:** [Live demo (Space)](https://huggingface.co/spaces/JayaHareeswarReddy/nl2sql) · [Model (adapter)](https://huggingface.co/JayaHareeswarReddy/nl2sql-qlora) · Dataset: [gretelai/synthetic_text_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)

## Results

Measured on a held-out test split (base model vs fine-tuned adapter).

| Model | Exact-match | Execution-match |
|-------|-------------|-----------------|
| Qwen2.5-Coder-1.5B (base)     | 23.5% | 66.7% |
| **+ QLoRA fine-tune (ours)**  | **31.0%** | **71.5%** |

Measured on 200 held-out test examples (165 executable). QLoRA fine-tuning
improved exact-match by **+7.5 points (~32% relative)** and execution accuracy
by **+4.8 points** — from a single ~30-minute run on a free T4 GPU.

## Approach

- **Method:** QLoRA (4-bit base weights + trainable LoRA adapters) via
  [Unsloth](https://github.com/unslothai/unsloth) + TRL `SFTTrainer`.
- **Data:** `gretelai/synthetic_text_to_sql` — each row has a schema
  (`sql_context`), a question (`sql_prompt`), and gold SQL (`sql`).
- **Prompt:** system instruction + schema + question → SQL completion.
- **Eval:** exact-match (normalized string) and execution-match (run both
  queries against an in-memory SQLite DB and compare result sets).

## Reproduce

### 1. Train (free GPU)
Open `train.py` in Google Colab (`Runtime > Change runtime type > T4 GPU`) or
Kaggle. Install deps and run:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes datasets
```

Set `HF_TOKEN` (a write token) and `HF_USERNAME` at the top of `train.py`, then
run it. It trains (~30–90 min) and pushes the adapter to your HF Hub repo.

### 2. Evaluate
```bash
pip install transformers peft accelerate datasets torch
python eval.py --adapter JayaHareeswarReddy/nl2sql-qlora --n 200
```
Copy the printed numbers into the results table above.

### 3. Deploy the demo
The `space/` folder is a ready Hugging Face Space (Gradio). Create a new Space,
upload `space/app.py`, `space/requirements.txt`, and `space/README.md`, and set
`ADAPTER_REPO` in `app.py` to your model repo.

## Repo layout
```
train.py            # QLoRA training (Colab/Kaggle)
eval.py             # base vs fine-tuned evaluation
space/app.py        # Gradio demo
space/requirements.txt
space/README.md     # Space config + card
```

## License
Apache-2.0 (matches the base model).
