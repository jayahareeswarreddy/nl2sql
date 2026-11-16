"""
NL2SQL — live demo Space (ZeroGPU).
Base Qwen2.5-Coder-1.5B + a fine-tuned LoRA adapter turn an English question
plus a table schema into a SQL query.

ZeroGPU note: a GPU is only attached INSIDE an @spaces.GPU function, so the
model is loaded lazily on the first call (not at startup), then cached.
"""

import spaces
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_REPO = "JayaHareeswarReddy/nl2sql-qlora"  # your fine-tuned adapter

SYSTEM_PROMPT = (
    "You are a text-to-SQL assistant. Given a database schema and a question, "
    "respond with a single correct SQL query and nothing else."
)

# Tokenizer load is CPU-only and safe at startup.
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
_model = None  # loaded lazily on the first GPU call


def _get_model():
    global _model
    if _model is None:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16, device_map="cuda"
        )
        m = PeftModel.from_pretrained(base, ADAPTER_REPO)
        m.eval()
        _model = m
    return _model


@spaces.GPU(duration=120)
def generate_sql(schema: str, question: str):
    schema = (schema or "").strip()
    question = (question or "").strip()
    if not schema or not question:
        return "-- Please provide both a schema and a question."

    model = _get_model()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return text.strip()


EXAMPLE_SCHEMA = (
    "CREATE TABLE employees (id INT, name TEXT, salary INT, department TEXT);\n"
    "CREATE TABLE departments (name TEXT, budget INT);"
)
EXAMPLES = [
    [EXAMPLE_SCHEMA, "List the names of employees in Sales earning more than 50000."],
    [EXAMPLE_SCHEMA, "What is the average salary per department?"],
    [EXAMPLE_SCHEMA, "How many employees are in each department, highest first?"],
]

with gr.Blocks(title="NL2SQL", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🧮 NL2SQL — Natural Language to SQL
        A **Qwen2.5-Coder-1.5B** model fine-tuned with **QLoRA** to turn an
        English question and a table schema into a SQL query.
        *(First query loads the model and may take ~30s; later ones are fast.)*
        """
    )
    with gr.Row():
        with gr.Column():
            schema_in = gr.Textbox(
                label="Database schema (CREATE TABLE ...)",
                lines=6,
                value=EXAMPLE_SCHEMA,
            )
            question_in = gr.Textbox(label="Question", lines=2)
            btn = gr.Button("Generate SQL", variant="primary")
        with gr.Column():
            sql_out = gr.Code(label="Generated SQL", language="sql")

    btn.click(generate_sql, [schema_in, question_in], sql_out)
    gr.Examples(EXAMPLES, inputs=[schema_in, question_in])
    gr.Markdown(
        "<sub>Base: Qwen2.5-Coder-1.5B-Instruct · Fine-tuned with QLoRA on "
        "gretelai/synthetic_text_to_sql · Built with Gradio, PEFT & 🤗 Transformers</sub>"
    )

if __name__ == "__main__":
    demo.launch()
