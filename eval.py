"""
Evaluate the fine-tuned NL2SQL model against the base model on a held-out
test split. Reports exact-match accuracy and execution-match accuracy, so you
get honest BASE vs FINE-TUNED numbers for the README / resume.

Run (on Colab GPU after training, or any machine with the deps):
    pip install transformers peft accelerate datasets torch
    python eval.py --adapter JayaHareeswarReddy/nl2sql-qlora --n 200

Notes
- Execution match builds an in-memory SQLite DB from each row's schema
  (sql_context) and compares the result sets of the gold vs predicted query.
  Rows whose schema can't be executed are skipped for execution-match only.
- Exact match compares normalized SQL strings; it is strict and will under-
  count semantically-correct-but-differently-written queries, so report both.
"""

import argparse
import re
import sqlite3

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
SYSTEM_PROMPT = (
    "You are a text-to-SQL assistant. Given a database schema and a question, "
    "respond with a single correct SQL query and nothing else."
)


def build_prompt(tokenizer, schema, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def clean_sql(text):
    """Strip markdown fences / prose; keep the first SQL statement."""
    text = text.strip()
    text = re.sub(r"```(?:sql)?", "", text, flags=re.IGNORECASE).strip()
    # cut at the first statement terminator if present
    if ";" in text:
        text = text.split(";")[0]
    return text.strip()


def normalize(sql):
    sql = sql.strip().rstrip(";")
    sql = re.sub(r"\s+", " ", sql)
    return sql.lower()


def exec_result(schema, query):
    """Return sorted result rows, or None if it can't run."""
    try:
        con = sqlite3.connect(":memory:")
        con.executescript(schema)
        rows = con.execute(query).fetchall()
        con.close()
        return sorted(str(r) for r in rows)
    except Exception:
        return None


@torch.no_grad()
def generate(model, tokenizer, schema, question):
    prompt = build_prompt(tokenizer, schema, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return clean_sql(text)


def evaluate(model, tokenizer, rows, label):
    exact_hits = 0
    exec_hits = 0
    exec_total = 0
    for ex in rows:
        pred = generate(model, tokenizer, ex["sql_context"], ex["sql_prompt"])
        gold = ex["sql"]

        if normalize(pred) == normalize(gold):
            exact_hits += 1

        gold_res = exec_result(ex["sql_context"], gold)
        if gold_res is not None:  # only count rows whose gold query runs
            exec_total += 1
            pred_res = exec_result(ex["sql_context"], pred)
            if pred_res is not None and pred_res == gold_res:
                exec_hits += 1

    n = len(rows)
    exact = 100 * exact_hits / n if n else 0
    execm = 100 * exec_hits / exec_total if exec_total else 0
    print(f"\n[{label}]  exact-match: {exact:.1f}%  "
          f"execution-match: {execm:.1f}%  (on {n} examples, "
          f"{exec_total} executable)")
    return exact, execm


def load_model(adapter=None):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="HF repo id of your LoRA adapter")
    ap.add_argument("--n", type=int, default=200, help="number of test examples")
    args = ap.parse_args()

    test = load_dataset("gretelai/synthetic_text_to_sql", split="test")
    rows = test.shuffle(seed=3407).select(range(min(args.n, len(test))))

    print("Loading BASE model...")
    base_model, tok = load_model(adapter=None)
    evaluate(base_model, tok, rows, "BASE")
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading FINE-TUNED model...")
    ft_model, tok = load_model(adapter=args.adapter)
    evaluate(ft_model, tok, rows, "FINE-TUNED")


if __name__ == "__main__":
    main()
