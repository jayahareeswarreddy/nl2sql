# ==========================================================================
# NL2SQL — QLoRA fine-tuning of Qwen2.5-Coder-1.5B on text-to-SQL
# Run this on a FREE GPU: Google Colab (Runtime > Change runtime type > T4 GPU)
# or Kaggle (Settings > Accelerator > GPU T4).
#
# In Colab, paste each "# === CELL" block into its own cell, OR just run the
# whole file. First install the deps (one time):
#
#   !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
#   !pip install -q --no-deps trl peft accelerate bitsandbytes datasets
#
# Then set your Hugging Face token (get one at https://huggingface.co/settings/tokens
# with WRITE access) and your target model repo name below.
# ==========================================================================

# === CELL 1: config =======================================================
import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")      # paste your write token, or set it in Colab "Secrets"
HF_USERNAME = "JayaHareeswarReddy"             # <-- your HF username
MODEL_REPO = f"{HF_USERNAME}/nl2sql-qlora"     # the model repo this will create/push to
    
BASE_MODEL = "unsloth/Qwen2.5-Coder-1.5B-Instruct"  # Unsloth 4-bit-ready base
MAX_SEQ_LEN = 1024
TRAIN_SUBSET = 12000   # rows to train on (raise for better results / longer run)

SYSTEM_PROMPT = (
    "You are a text-to-SQL assistant. Given a database schema and a question, "
    "respond with a single correct SQL query and nothing else."
)

# === CELL 2: load base model + attach LoRA ================================
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,            # auto (bf16 on supported GPUs, else fp16)
    load_in_4bit=True,     # QLoRA: 4-bit base weights
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# === CELL 3: load + format the dataset ====================================
from datasets import load_dataset

# gretelai/synthetic_text_to_sql columns:
#   sql_prompt  -> the natural-language question
#   sql_context -> CREATE TABLE (+ sometimes INSERT) statements = the schema
#   sql         -> the gold SQL answer
raw = load_dataset("gretelai/synthetic_text_to_sql", split="train")
raw = raw.shuffle(seed=3407).select(range(min(TRAIN_SUBSET, len(raw))))


def to_chat_text(ex):
    user = f"Schema:\n{ex['sql_context']}\n\nQuestion: {ex['sql_prompt']}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": ex["sql"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


dataset = raw.map(to_chat_text, remove_columns=raw.column_names)

# === CELL 4: train ========================================================
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,   # if your TRL version errors here, rename to: processing_class=tokenizer
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=20,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

trainer.train()

# === CELL 5: save locally + push the adapter to the Hugging Face Hub ======
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

if HF_TOKEN:
    model.push_to_hub(MODEL_REPO, token=HF_TOKEN)
    tokenizer.push_to_hub(MODEL_REPO, token=HF_TOKEN)
    print(f"Pushed adapter to https://huggingface.co/{MODEL_REPO}")
else:
    print("No HF_TOKEN set — adapter saved locally in ./lora_model only.")

# === CELL 6: quick smoke test =============================================
FastLanguageModel.for_inference(model)
test_schema = "CREATE TABLE employees (id INT, name TEXT, salary INT, dept TEXT);"
test_q = "List the names of employees in the Sales department earning over 50000."
msgs = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"Schema:\n{test_schema}\n\nQuestion: {test_q}"},
]
prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
