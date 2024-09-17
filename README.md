---
title: TGI IE AutoBench
emoji: 🤗
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app/ui.py
pinned: false
hf_oauth: true

---

# HF IE AutoBench

## Setup

```
python -m venv .venv
source .venv/bin/activate

make build-k6

poetry install
```
