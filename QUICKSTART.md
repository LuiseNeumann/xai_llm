```bash
uv sync --extra ui
lms load google/gemma-4-26b-a4b --gpu 0.6 -c 4096 -y
lms server start
uv run streamlit run interface/app.py
```

```bash
uv run python -m interface.run_benchmark --model google/gemma-4-26b-a4b
```

```bash
uv run python -m interface.download_bbq
uv run python -m interface.run_benchmark --model google/gemma-4-26b-a4b --data data/downloads/bbq_gender_identity.jsonl --limit 20
```

```bash
uv sync --extra ui-hf
uv run python -m interface.run_benchmark --backend hf --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --data data/downloads/bbq_gender_identity.jsonl --limit 10 --samples 1
```

```bash
lms unload google/gemma-4-26b-a4b
lms load qwen3.8-27b --gpu 0.6 -c 4096 -y
uv run python -m interface.run_benchmark --model qwen3.8-27b
```
