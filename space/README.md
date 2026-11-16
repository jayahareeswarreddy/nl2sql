---
title: NL2SQL
emoji: 🧮
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "6.32.0"
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Fine-tuned Qwen2.5-Coder that turns English into SQL
---

# 🧮 NL2SQL

A **Qwen2.5-Coder-1.5B** model fine-tuned with **QLoRA** to translate a
natural-language question plus a table schema into a SQL query.

- **Base model:** Qwen/Qwen2.5-Coder-1.5B-Instruct
- **Adapter:** JayaHareeswarReddy/nl2sql-qlora
- **Dataset:** gretelai/synthetic_text_to_sql

Enter a `CREATE TABLE` schema and a question to generate SQL.
