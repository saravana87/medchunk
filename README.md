# 🧠 Medical Chunker – No LLM Needed

This project is a lightweight, efficient system to **chunk and auto-label medical documents** using classical NLP and unsupervised methods — **without relying on any Large Language Models (LLMs)**.

## 🔧 Features

- 📄 Breaks long medical texts into chunks
- 🔑 Extracts keywords using **KeyBERT**
- 🧬 Embeds chunks with **SentenceTransformer**
- 🔀 Clusters similar chunks using **HDBSCAN**
- 🏷️ Labels and groups them automatically (no manual rules needed)
- 🧠 Optional: Rule-based labeling also included (`labeler_rule_based.py`)

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
