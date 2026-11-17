---
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
library_name: peft
license: apache-2.0
datasets:
  - gretelai/synthetic_text_to_sql
language:
  - en
tags:
  - text-to-sql
  - sql
  - qlora
  - peft
  - lora
pipeline_tag: text-generation
---

# NL2SQL — QLoRA fine-tune of Qwen2.5-Coder-1.5B

A LoRA adapter that adapts **Qwen2.5-Coder-1.5B-Instruct** to translate a
natural-language question plus a table schema into a SQL query.

- **Base model:** Qwen/Qwen2.5-Coder-1.5B-Instruct
- **Method:** QLoRA (4-bit) via PEFT + TRL, trained on a free T4 GPU
- **Dataset:** gretelai/synthetic_text_to_sql
- **Demo:** https://huggingface.co/spaces/JayaHareeswarReddy/nl2sql
- **Code:** https://github.com/jayahareeswarreddy/nl2sql

## Results

Measured on 200 held-out test examples (165 executable).

| Model | Exact-match | Execution-match |
|-------|-------------|-----------------|
| Qwen2.5-Coder-1.5B (base) | 23.5% | 66.7% |
| **+ QLoRA fine-tune**     | **31.0%** | **71.5%** |

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, "JayaHareeswarReddy/nl2sql-qlora")

schema = "CREATE TABLE employees (id INT, name TEXT, salary INT, department TEXT);"
question = "List the names of employees in Sales earning more than 50000."
messages = [
    {"role": "system", "content": "You are a text-to-SQL assistant. Given a database schema and a question, respond with a single correct SQL query and nothing else."},
    {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Limitations

Trained on synthetic data; best on single-table and simple multi-column
queries. Not intended for production use against live databases without
validation.

## License

Apache-2.0 (matches the base model).
